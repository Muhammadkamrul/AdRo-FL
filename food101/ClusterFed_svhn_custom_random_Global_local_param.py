import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Subset
import client_cifar10_svhn as client_obj
from server_custom_cifar10_svhn import Server, download_cifar10
import csv
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
import time
import random
random.seed(42)

# Define the fixed directory
result_directory = "/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/results/svhn2/"

# Ensure the directory exists, If folder does not exist, it creates it.
os.makedirs(result_directory, exist_ok=True)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
NUM_CLUSTERS = 10
CLIENTS_PER_CLUSTER = [12,8,10,11,9,5,15,10,4,16]
K = 2
SERVER_CAPACITY = NUM_CLUSTERS * K
ROUNDS = 3000
learning_rate = 0.01
samples_per_client = 500
weight = 0.5
D = 0.13 # Example deadline for transmission time (in seconds)

LOCAL_EPOCHS = 1
#local_learning_rate = 0.001
G = 5  # Example threshold for L2 norm
#QUANTIZATION_BITS = [31, 15, 7, 4, 2]  # Possible quantization levels
QUANTIZATION_BIT = 8
threshold = None
top_n = True
client_select_type = 'custom'
selection_scope = 'global'
VRF_scope=True

num_classes=10
global_total_samples = 0

def distribute_dirichlet(clusters, train_dataset, num_classes=10, alpha=0.1):
    num_clients = sum(len(cluster.clients) for cluster in clusters)
    class_indices = [[] for _ in range(num_classes)]
    for idx, (data, target) in enumerate(train_dataset):
        class_indices[target].append(idx)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        p_c = np.random.dirichlet(alpha * np.ones(num_clients))
        num_samples_per_client_c = np.random.multinomial(len(class_indices[c]), p_c)
        shuffled_indices = np.random.permutation(class_indices[c])
        start = 0
        for k in range(num_clients):
            num = num_samples_per_client_c[k]
            client_indices[k].extend(shuffled_indices[start:start + num])
            start += num
    client_id = 0
    for cluster in clusters:
        for client in cluster.clients:
            client.set_data(Subset(train_dataset, client_indices[client_id]))
            client_id += 1

def check_data_distribution(clusters):
    """
    Prints the data distribution (class-wise sample count) for each client.

    Args:
        clusters: A list of cluster objects, each containing a list of clients.

    Returns:
        None. Prints class-wise sample distribution for each client.
    """
    global global_total_samples
    for cluster in clusters:
        for client in cluster.clients:
            client_data = client.get_data()  # Assuming this returns a DataLoader or Dataset
            class_counts = {}

            for _, target in client_data:
                class_counts[target] = class_counts.get(target, 0) + 1

            print(f"Client {client.client_id}:")
            for class_id, count in class_counts.items():
                print(f"  Class {class_id}: {count} samples")
                global_total_samples += count
            print("-" * 20)
            
    print("\nTotal samples across all clusters = ", global_total_samples)
   
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clusters = client_obj.create_clusters(NUM_CLUSTERS, CLIENTS_PER_CLUSTER, K, device)
    server = Server(clusters, SERVER_CAPACITY, QUANTIZATION_BIT, device, learning_rate, threshold = None, top_n = True)
    
    # Download and distribute the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.SVHN('/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/food101/data', split='train', download=False, transform=transform)
    test_dataset = datasets.SVHN('/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/food101/data', split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    distribute_dirichlet(clusters, train_dataset, num_classes=10, alpha=0.1)
    check_data_distribution(clusters)  
    
    # Initialize logging
    #log_file = os.path.join(result_directory, 'output_svhn_'+client_select_type+'_global_dirichlet_'+'sel_client_loss_acc_rnd.csv')  
    #log_file = os.path.join(current_dir, "sel_client_loss_acc_rnd.csv")   
    # with open(log_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Round", "Number of Selected Clients", "Global Loss", "Training Loss", "Accuracy"])
    
    # detail_log_file = os.path.join(result_directory, 'output_svhn_'+client_select_type+'_global_dirichlet_'+'Sel_client_norm_enrgy_bits_time.csv')
    # with open(detail_log_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Round", "Client ID", "Cluster ID", "Gradient Norm",  "local_loss", "hybrid_score", "Transmitted Energy", "Transmitted Bits", "Transmission Time"])
    
    # metric_log_file = os.path.join(result_directory, 'output_svhn_'+client_select_type+'_global_dirichlet_'+'all_client_metric.csv')
    # with open(metric_log_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Round", "Client ID", "Cluster ID", "metric", "Transmitted Energy", "Transmitted Bits", "Transmission Time"])
    
    loss_list = []
    training_loss_list = []
    accuracy_list = []
    total_transmitted_bits_list = []
    total_energy_list = []
    old_list = []
    round_duration_list = []
    
    num_clients = sum(len(cluster.clients) for cluster in clusters) 
    client_selection_tracker = {client_id: 0 for client_id in range(num_clients)}
    
    for round_num in range(ROUNDS):
        start_time = time.time()

        print(f"\n\nFederated Learning Round {round_num + 1} *********************")

        server.distribute_model()
        
        server.train_and_setMinMax(LOCAL_EPOCHS)
        
        #server.compute_local_loss()
        
        #selected_clients = server.select_clients(K, G, D, client_select_type, round_num + 1, weight, global_total_samples, num_classes, 
        #                                         selection_scope, SERVER_CAPACITY,VRF_scope=True)
        
        selected_clients = server.select_clients(
            K, G, D, client_select_type, round_num + 1, weight, global_total_samples,
            num_classes, selection_scope, SERVER_CAPACITY, VRF_scope=VRF_scope
        )
        
        if (VRF_scope == True):
            selected_clients = random.sample(selected_clients, min(SERVER_CAPACITY, len(selected_clients)))
            server.selected_clients = selected_clients

        if not selected_clients:
            print("No valid clusters were selected, restart training round.")         
            total_transmitted_bits_list.append(None)
            total_energy_list.append(None)          
            loss_list.append(None)
            training_loss_list.append(None)
            accuracy_list.append(None)           
            continue
        
        time1 = time.time() - start_time
        
        for client, _, _ in selected_clients:
            # Increment the count for the selected client
            client_selection_tracker[client.client_id] += 1
        
        current_list=[]
        print("\nSelected clients at round: ", round_num + 1)
        for client,_,_ in selected_clients:
            print(f"Client {client.client_id}")
            current_list.append(client.client_id)
        
        if (sorted(current_list) == sorted(old_list)):
            print("\nSelected the same set of clients as last round\n")
        
        old_list = current_list
        
        time2_start = time.time()
        
        #client_obj.verify_clusters(clusters, selected_clients)
        #server.update_learning_rate(round_num)
        server.aggregate_quantized_grads()

        round_duration = (time.time() - time2_start) + time1

        loss, accuracy = evaluate(server.global_model, device, test_loader)
        
        #if (round_num  == 0):
        #    continue
        
        total_training_loss = 0
        
        print("\nAll clients metric for round:", round_num + 1)
        for cluster in clusters:
            for client in cluster.clients:
                total_training_loss += client.local_loss
                #print(f"Client {client.client_id}: Local Loss {client.local_loss}, L2 Norm {client.local_l2_norm}")
                 
        
        total_clients = sum(len(cluster.clients) for cluster in clusters)        
        average_training_loss = total_training_loss / total_clients
        
        # # Log the results
        # with open(log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([round_num + 1, len(selected_clients), loss, average_training_loss, accuracy])

        # # Detailed logging
        # with open(detail_log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     for client, hybrid_score, _ in selected_clients:
        #         writer.writerow([round_num + 1, client.client_id, client.cluster_id, client.local_l2_norm, client.local_loss, hybrid_score, client.energy, client.transmitted_bits, client.transmission_time])
        
        # #Metric logging
        # with open(metric_log_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     for cluster in clusters:
        #         for client in cluster.clients:
        #             writer.writerow([round_num + 1, client.client_id, client.cluster_id, client.metric, client.energy, client.transmitted_bits, client.transmission_time])
        
        
        # Calculate total transmitted bits and energy for the round
        total_transmitted_bits = sum(client.transmitted_bits for client, _, _ in selected_clients)
        total_energy = sum(client.energy for client, _, _ in selected_clients)
        
        total_transmitted_bits_list.append(total_transmitted_bits)
        total_energy_list.append(round(total_energy,5))
        
        loss_list.append(round(loss,4))
        training_loss_list.append(round(average_training_loss,4))
        accuracy_list.append(round(accuracy,2))
        round_duration_list.append(round(round_duration, 2))

    client_ids = list(client_selection_tracker.keys())
    selection_counts = list(client_selection_tracker.values())

    results_as_list = os.path.join(result_directory, 'output_svhn_'+client_select_type+'_'+selection_scope+'_vrf'+str(VRF_scope)+'_dirichlet_Q8_lr0.01_ep1_w0.5.txt') 
    
    with open(results_as_list, 'w') as file:
        # Write each list in the desired format
        file.write(f'loss_{client_select_type} = {loss_list}\n')
        file.write(f'accuracy_{client_select_type} = {accuracy_list}\n')
        file.write(f'bits_{client_select_type} = {total_transmitted_bits_list}\n')
        file.write(f'energy_{client_select_type} = {total_energy_list}\n')
        file.write(f'train_loss_{client_select_type} = {training_loss_list}\n')
        file.write(f'client_id_{client_select_type} = {client_ids}\n') #which client selected how many times (key)
        file.write(f'selection_count_{client_select_type} = {selection_counts}\n') #which client selected how many times (value)
        file.write(f'round_duration = {round_duration_list}\n')
        



if __name__ == "__main__":
    main()


