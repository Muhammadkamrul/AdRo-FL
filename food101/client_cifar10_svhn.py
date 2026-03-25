import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

random_seed=42
np.random.seed(random_seed)
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
        
class Client:
    def __init__(self, client_id, cluster_id, data, model, device):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.data = data
        self.model = model.to(device)
        self.device = device
        self.gradients = None  # Store gradients here
        self.bandwidth = 1e6 #1 Mhz
        self.quantized_params = []
        self.power = 0.1 #0.1 joules per second
        self.metric = None

    def set_data(self, data):
        self.data = data
        
    def get_data(self):
        return self.data

    def set_model(self, global_model_state): #update model from global model
        self.model.load_state_dict(global_model_state)     
        
    def calculate_SNR(self):
        snr_db = np.random.uniform(0, 30)
        snr_linear = 10**(snr_db/10)
        return snr_linear      

    def find_min_max_grads(self):
        min_val = min(grad.min().item() for grad in self.gradients)
        max_val = max(grad.max().item() for grad in self.gradients)
        return min_val, max_val
    
    def calculate_payload_size(self): #based on gradient
        payload_size_bits = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_elements = param.grad.numel()  # Number of elements at each param_grad in the gradient tensor
                element_size_bits = param.grad.element_size() * 8  # Size in bits 
                payload_size_bits += num_elements * element_size_bits

        return payload_size_bits 
    
    def calculate_transmission_time(self, payload_size_bits):
        
        snr = self.calculate_SNR()
        channel_capacity_bps = self.bandwidth * np.log2(1 + snr)
        
        self.transmitted_bits = payload_size_bits
        self.transmission_time = payload_size_bits / channel_capacity_bps
        
        return self.transmission_time
    
    def calculate_energy_consumption(self, transmission_time):
        # Implement a realistic model for transmission time calculation
        # This is a placeholder function
        self.energy = self.power * transmission_time  
        
    def get_num_samples(self):
            return len(self.data)
            
            
    def get_num_classes(self):
        """
        Returns the number of unique classes in the client's dataset.
        """
        class_counts = set()
        for _, target in self.data:
            class_counts.add(target)
            
        return len(class_counts)
    

    def quantize_gradient_func(self, gradients, bits):
        """Applies per-client min-max uniform quantization to gradients."""

        gradient = [grad.to(self.device) for grad in gradients]
        
        # Compute per-client min and max
        min_val = min([grad.min().item() for grad in gradient])
        max_val = max([grad.max().item() for grad in gradient])

        # Avoid division by zero in case min == max
        if max_val == min_val:
            max_val += 1e-6

        # Compute scale
        quant_levels = 2 ** bits - 1
        scale = (max_val - min_val) / quant_levels

        # Choose dtype based on bits
        if bits <= 8:
            dtype = torch.uint8
        elif bits <= 16:
            dtype = torch.int16
        elif bits <= 32:
            dtype = torch.int32
        else:
            raise ValueError("bits should be <= 32")

        # Quantize gradients using min-max scaling
        quantized_gradient = [
                torch.clamp(torch.round((grad - min_val) / scale), 0, quant_levels).to(dtype).to(self.device) for grad in gradient
            ]

        self.quantized_gradient =  quantized_gradient
        self.min_val = min_val
        self.max_val = max_val

    def train(self, quantization_bit, epochs=1): #common FedSGD
        self.model.train()
        data_loader = DataLoader(self.data, shuffle=True, batch_size=64)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.model.zero_grad()
        total_loss = 0.0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()  # gradients accumulate here automatically

        # At this point, gradients represent accumulated gradients over entire dataset
        gradients = [param.grad.clone() for param in self.model.parameters()]
        
        # Calculate L2 norm of accumulated gradients
        l2_norm = torch.sqrt(sum(torch.norm(g)**2 for g in gradients)).item()

        self.gradients = gradients
        self.local_l2_norm = l2_norm
        self.local_loss = total_loss

        # Optional quantization step
        if quantization_bit < 32:
            self.quantize_gradient_func(self.gradients, quantization_bit)

        return self.local_l2_norm


    # Function to evaluate the local accuracy of the model on client's local data
    def evaluate_local_accuracy(self):
        self.model.eval()  # Set model to evaluation mode
        data_loader = DataLoader(self.data, batch_size=64, shuffle=False)  # Use the local dataset
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # Get the predicted classes
                correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
                total += target.size(0)  # Keep track of total samples
        
        accuracy = 100. * correct / total  # Calculate accuracy percentage
        return accuracy  # Return accuracy

class Cluster:
    def __init__(self, cluster_id, clients, K):
        self.cluster_id = cluster_id
        self.clients = clients
        self.K = K
        self.participating = False


def create_clusters(num_clusters, clients_per_cluster, K, device):
    clusters = []
    client_id = 0
    for cluster_id in range(num_clusters):
        clients = []
        for _ in range(clients_per_cluster[cluster_id]):
            model = CNN_Cifar().to(device)
            clients.append(Client(client_id, cluster_id, None, model, device))
            client_id += 1
        clusters.append(Cluster(cluster_id, clients, K))
    return clusters
    
    
    
def verify_clusters(clusters, selected_clients):
        
        for cluster in clusters:
            cluster_selected_clients = [client for client,_,_ in selected_clients if client.cluster_id == cluster.cluster_id]
            if len(cluster_selected_clients) >= cluster.K:
                cluster.participating = True
            else:
                cluster.participating = False
                #print(f"Cluster {cluster.cluster_id} does not have enough valid clients.")
                #Need to create a way to prevent a client from participating if the malicious server calls it.

