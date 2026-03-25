import sys
client_dir = "/home/kamrul/Documents/kamrul_files_Linux/ClusterFed_ICC_IoT/mnist"  # Replace with actual path
dataset_dir = "/home/kamrul/Documents/kamrul_files_Linux/ClusterFed_ICC_IoT/datasets"
sys.path.append(client_dir)
import torch
from torch import nn
import torch.nn.functional as F
import random
import collections

random_seed=42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
 
class CNN_Cifar(nn.Module):
    def __init__(self, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10):
        super(CNN_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Server:
    def __init__(self, clusters, max_clients, quantization_bit, device, learning_rate, threshold, top_n):
        self.max_clients = max_clients
        self.device = device
        self.global_model = CNN_Cifar().to(self.device)
        self.clusters = clusters
        self.selected_clients = []
        self.threshold = threshold
        self.top_n = top_n
        self.learning_rate = learning_rate
        self.quantization_bit = quantization_bit
        
    
    def distribute_model(self):
        
        print("\n\nInside Model Distribution")
        
        global_state_dict = self.global_model.state_dict()
        for cluster in self.clusters:
            for client in cluster.clients:
                client.set_model(global_state_dict)
                    
    def compute_local_loss(self):       
        print("\nComputing local loss...")
        for cluster in self.clusters:
            for client in cluster.clients:
                client.compute_local_loss()   
    
    def train_and_setMinMax(self, LOCAL_EPOCHS):
        #min_val_list = []
        #max_val_list = []
        self.l2_norm_avg = []
        print("\nTraining ongoing...")
        for cluster in self.clusters:
            for client in cluster.clients:
                local_l2_norm = client.train(self.quantization_bit, LOCAL_EPOCHS)
                #total_loss, l2_norm = client.calculate_gradients() #The purpose of total_loss is tracking the loss value during training for each client.
                self.l2_norm_avg.append(local_l2_norm)
        print("Training Done!\n")

    def select_clients(self, K, G, D, client_select_type, FL_round, weight, global_total_samples, num_classes, 
                        selection_scope, num_clients_to_select=None, VRF_scope=False):

            print("\n\nInside Client selection")

            valid_clusters = []
            deadline_avg = []
            original_payload_sizes = []
            quantized_payload_sizes = []
            tranmission_time_avg = []

            self.selected_clients.clear()
            all_valid_clients = []

            for cluster in self.clusters:
                valid_clients = []

                for client in cluster.clients:
                    orig_payload_size_bits = client.calculate_payload_size()
                    original_payload_sizes.append(orig_payload_size_bits / 8)

                    if client_select_type == 'custom':
                        if self.quantization_bit < 32:
                            quantized_gradients = client.quantized_gradient
                            quantized_payload_size_bits = sum(g.numel() * g.element_size() * 8 for g in quantized_gradients)
                            quantized_payload_sizes.append(quantized_payload_size_bits / 8)
                            transmission_time = client.calculate_transmission_time(quantized_payload_size_bits)
                        else:
                            transmission_time = client.calculate_transmission_time(orig_payload_size_bits)

                        deadline_avg.append(transmission_time)
                        client.calculate_energy_consumption(transmission_time)

                        if transmission_time <= D:
                            hybrid_score = (weight * client.local_loss) + ((1 - weight) * client.local_l2_norm * (client.get_num_samples()/global_total_samples))
                            client.metric = hybrid_score
                            valid_clients.append((client, hybrid_score, self.quantization_bit))

                    elif client_select_type == 'random':
                        transmission_time = client.calculate_transmission_time(orig_payload_size_bits)
                        client.calculate_energy_consumption(transmission_time)
                        valid_clients.append((client, 0, self.quantization_bit))  # Metric and quantization bit are dummy here

                if selection_scope == 'local' and len(valid_clients) >= K:
                    valid_clusters.append((cluster, valid_clients))

                if selection_scope == 'global':
                    all_valid_clients.extend(valid_clients)

            if selection_scope == 'local':
                if len(valid_clusters) == 0:
                    return None  # Restart round if no valid clusters

                X = K  # Select K clients per cluster

                for cluster, valid_clients in valid_clusters:
                    if client_select_type == 'custom':
                        valid_clients.sort(key=lambda client_tuple: client_tuple[1], reverse=True)
                        top_clients = valid_clients[:X]
                    elif client_select_type == 'random':
                        top_clients = random.sample(valid_clients, min(X, len(valid_clients)))
                    self.selected_clients.extend(top_clients)

                selected_clients_by_cluster = collections.defaultdict(list)
                for client, hybrid_score, quantization_bit in self.selected_clients:
                    selected_clients_by_cluster[client.cluster_id].append((client, hybrid_score, quantization_bit))

                for cluster_id, selected_clients in selected_clients_by_cluster.items():
                    print(f"Cluster {cluster_id}: Selected {len(selected_clients)} clients")

            elif selection_scope == 'global':
                if not all_valid_clients or num_clients_to_select is None or num_clients_to_select <= 0:
                    return None  # Invalid input or no eligible clients

                if client_select_type == 'custom':
                    all_valid_clients.sort(key=lambda client_tuple: client_tuple[1], reverse=True)
                    top_80_percent = int(0.8 * len(all_valid_clients))
                    primary_pool = all_valid_clients[:top_80_percent]

                    if VRF_scope == False:
                        selected = random.sample(primary_pool, min(num_clients_to_select, len(primary_pool)))
                        self.selected_clients.extend(selected)
                        # Post-filter clusters with < K clients
                        cluster_counts = collections.defaultdict(list)
                        for client, _, _ in self.selected_clients:
                            cluster_counts[client.cluster_id].append((client, _, _))

                        # Remove clients from clusters with fewer than K participants
                        self.selected_clients = [
                            entry for cluster_id, entries in cluster_counts.items()
                            if len(entries) >= K
                            for entry in entries
                        ]
                    else:
                        return primary_pool

                elif client_select_type == 'random':
                    selected = random.sample(all_valid_clients, min(num_clients_to_select, len(all_valid_clients)))
                    self.selected_clients.extend(selected)

            print("\nQuantization using: ", self.quantization_bit, " bits")
            print("Average l2_norm: ", sum(self.l2_norm_avg) / len(self.l2_norm_avg), "Required G: ", G)
            print("Max payload size(original): ", max(original_payload_sizes), " bytes", "Min payload size(original): ", min(original_payload_sizes), " bytes")

            if self.quantization_bit < 32 and client_select_type == 'custom':
                print("Max payload size (quantized): ", max(quantized_payload_sizes), " bytes,", "Min payload size (quantized): ", min(quantized_payload_sizes), " bytes")

            if client_select_type == 'custom':
                print("Average transmission time: ", sum(deadline_avg) / len(deadline_avg), "Required D: ", D)

            return self.selected_clients
   
    
    def aggregate_quantized_grads(self):                
        print("\n\nInside aggregate_quantized_grads")      
        
        total_gradients = None
        total_samples = 0  # Total number of samples across all selected clients
        
        # Iterate over each selected client
        for client, _, _ in self.selected_clients: 

            if(self.quantization_bit <32):                                                                         

                    # Compute scale (same as in quantization)
                    scale = (client.max_val - client.min_val) / (2**self.quantization_bit - 1)

                    # Ensure quantized gradients are on the correct device
                    quantized_gradient = [q_grad.to(self.device) for q_grad in client.quantized_gradient]                    

                    # Dequantize
                    dequantized_gradients = [
                    (q_grad.to(torch.float32) * scale) + client.min_val for q_grad in quantized_gradient
                    ]
                    
                    client_gradients = dequantized_gradients        
                    
            else:                            
                    client_gradients = client.gradients
            
            num_samples = client.get_num_samples()
            total_samples += num_samples
            
            if total_gradients is None:
                total_gradients = [torch.zeros_like(grad) for grad in client_gradients]
            for i, grad in enumerate(client_gradients):
                total_gradients[i] += grad * num_samples               
        
        # Average gradients
        for i in range(len(total_gradients)):
            total_gradients[i] = total_gradients[i] / total_samples
        
        self.global_gradients = total_gradients
        
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), self.global_gradients):            
                param -= self.learning_rate * grad  # Update model parameters with learning rate


   
def download_cifar10():
    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(dataset_dir, train=True, download=True, transform=transform_train)
    
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    test_dataset = datasets.CIFAR10(dataset_dir, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset
    


    
        
        
  