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
                #min_val, max_val = client.find_min_max_grads()
                #min_val_list.append(min_val)
                #max_val_list.append(max_val)
                
        #self.global_min_val = min(min_val_list)
        #self.global_max_val = max(max_val_list)
        print("Training Done!\n")
    
    def select_clients(self, K, G, D, client_select_type, FL_round, weight, global_total_samples, num_classes,
                    selection_scope, num_clients_to_select=None):

        print("\n\nInside Client selection")

        valid_clusters = []
        deadline_avg = []
        original_payload_sizes = []
        quantized_payload_sizes = []
        tranmission_time_avg = []

        self.selected_clients.clear()

        if client_select_type == 'random':
            all_valid_clients = []

            for cluster in self.clusters:
                valid_clients = []
                for client in cluster.clients:
                    orig_payload_size_bits = client.calculate_payload_size()
                    original_payload_sizes.append(orig_payload_size_bits / 8)

                    transmission_time = client.calculate_transmission_time(orig_payload_size_bits)
                    tranmission_time_avg.append(transmission_time)
                    client.calculate_energy_consumption(transmission_time)

                    valid_clients.append((client, None, self.quantization_bit))

                if selection_scope == 'local':
                    valid_clusters.append((cluster, valid_clients))
                elif selection_scope == 'global':
                    all_valid_clients.extend(valid_clients)

            print("\nRandom Client Selection, using Proportional Allocation")
            print("Average transmission time: ", sum(tranmission_time_avg) / len(tranmission_time_avg))
            print("Max payload size(original): ", max(original_payload_sizes), " bytes", 
                "Min payload size(original): ", min(original_payload_sizes), " bytes")

            if selection_scope == 'local':
                X = K
                for cluster, valid_clients in valid_clusters:
                    if len(valid_clients) >= X:
                        rand_clients = random.sample(valid_clients, X)
                        self.selected_clients.extend(rand_clients)

                selected_clients_by_cluster = collections.defaultdict(list)
                for client, hybrid_score, quantization_bit in self.selected_clients:
                    selected_clients_by_cluster[client.cluster_id].append((client, hybrid_score, quantization_bit))

                for cluster_id, selected_clients in selected_clients_by_cluster.items():
                    print(f"Cluster {cluster_id}: Selected {len(selected_clients)} clients")

            elif selection_scope == 'global':
                if not all_valid_clients or num_clients_to_select is None or num_clients_to_select <= 0:
                    return None  # Invalid input or no eligible clients

                selected = random.sample(all_valid_clients, min(num_clients_to_select, len(all_valid_clients)))
                self.selected_clients.extend(selected)
                print(f"\nGlobal selection: Selected {len(selected)} clients randomly from all clusters")

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
    


    
        
        
  