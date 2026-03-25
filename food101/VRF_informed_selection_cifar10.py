import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import client_VRF_cifar10_svhm as client_obj
from server_vrf_custom_cifar10_svhn import Server
import os
from torchvision import datasets, transforms
import time
import random
import ecvrf_edwards25519_sha512_elligator2
from scipy.special import comb
import nacl.exceptions
import zlib
import struct
 
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

conservative_factor = 1 #it controls the degree of client selection from pool
Phonest = 0.9 #percentage of honest clients
Pdishonest = 1-Phonest #percentage of dishonest clients
PbiasedAttack = 0.001 #Maximum Allowable Probability of biased selection attack


def calculate_hash(secret_key, vrf_token_alpha):
    #vrf_token_alpha = vrf_token_alpha.encode('utf-8')  # Specify encoding if needed
    p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, vrf_token_alpha)
    b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    
    # Convert the bytes object to an integer
    beta_int = int.from_bytes(beta_string, byteorder='big')  # Use 'little' if needed
    
    return beta_int, pi_string

def find_threshold(number_of_clients, conservative_factor, Phonest, Pdishonest, PbiasedAttack, server_capacity):
      
    K_minimum = "No Solution"

    # Iterate over possible values of K
    for K in range(1, int(Pdishonest * number_of_clients) + 1 + 1):
        if ((Phonest * number_of_clients * comb(int(Pdishonest * number_of_clients), K - 1)) / comb(number_of_clients, K)) <= PbiasedAttack:
            K_minimum = K
            break
    
    if(K_minimum == "No Solution"):
        input("No solution found for minimum number of clients. May be because of very few total clients.")
    
    if( K_minimum <= server_capacity):
        K_solution = server_capacity
    else:
        input("Max number of clients server will accept is below the privacy threshold: abort the program")
    
    # Calculate a more conservative threshold T to select more than K clients on average
    T = (conservative_factor * K_solution * 2**512) // number_of_clients
    
    return T, K_solution, K_minimum

def check_winner(Threshold,vrf_token_alpha,client):
    Threshold = Threshold
    vrf_token_alpha = vrf_token_alpha
    vrf_hash, vrf_proof = calculate_hash(client.vrf_privatekey, vrf_token_alpha)
    winner = vrf_hash < Threshold
        
    return winner, vrf_hash, vrf_proof

def make_vrf_token_alpha(all_clients_objects):
    vrf_token_alpha = []
    for client,_,_ in all_clients_objects:
        vrf_token_alpha.append(int(client.metric))
    return vrf_token_alpha

# Utility function to verify signature given chunk position
def verify_signature_from_chunk(client_index, chunk_size, compressed_chunks, value_str, public_key):
    signature_size = 64
    chunk_index = client_index // chunk_size
    offset = (client_index % chunk_size) * signature_size

    decompressed_chunk = zlib.decompress(compressed_chunks[chunk_index])
    signature = decompressed_chunk[offset:offset + signature_size]
    numeric_value = struct.unpack_from('!H', value_str, client_index * 2)[0]

    try:
        public_key.verify(str(numeric_value).encode(), signature)
        print(f"Verification succeeded for client at index {client_index}.")
        return True
    except nacl.exceptions.BadSignatureError:
        print(f"Verification failed for client at index {client_index}.")
        return False

def select_clients_attempt(vrf_token_alpha, compressed_signature_chunks, server_capacity, all_clients_objects):

    #Optional Verification step for clients to see if the server provided alpha string is tempared or not
    #random_client = random.choice(all_clients_objects)
    #client_index = int(random_client.client_id)
    #verify_signature_from_chunk(client_index, Server.chunk_size, compressed_signature_chunks, vrf_token_alpha, random_client.verify_key)
        
    global conservative_factor 
    global Phonest 
    global Pdishonest 
    global PbiasedAttack 

    selected_clients = []
    
    Threshold, K_solution, K_minimum = find_threshold(len(all_clients_objects), conservative_factor, Phonest, Pdishonest, PbiasedAttack, server_capacity)
    
    for client,_,_ in all_clients_objects:
        #s = time.time()
        winner, vrf_hash, vrf_proof = check_winner(Threshold,vrf_token_alpha,client)
        #print("\n\n Time for checking winner: ", round(time.time()-s,2))
        #input("Hello")
        if(winner == True):
            selected_clients.append((client,None,None))
    
    total_selected_clients = len(selected_clients)

    if(total_selected_clients < K_solution):
        return False, None
    
    print("\nMinimum clients needed to prevent BS attack: ",K_minimum, " (attack probability ",PbiasedAttack,")")
    
    return True, selected_clients


def VRF_client_select(SERVER_CAPACITY, all_clients_objects, server): #all_clients_objects is the primary pool, not all real clients

    global conservative_factor
    
    start_time = time.time()

    server.collect_data(all_clients_objects)
    vrf_token_alpha, compressed_signature_chunks = server.broadcast_data()
    #vrf_token_alpha = ''.join([str(number) for number in vrf_token_alpha]) #make a long value string

    counter = 0

    while(1):

        select_clients_success, selected_clients = select_clients_attempt(vrf_token_alpha, compressed_signature_chunks, SERVER_CAPACITY, all_clients_objects)
        
        if (select_clients_success == False) :
            conservative_factor += 0.5
            counter +=0.5
            print(".")
        else:
            conservative_factor -= counter #return the conservative factor to initial value
            print("\n************\n***********\nNumber of Selected Clients using VRF: ",  len(selected_clients))
            break

    duration = time.time() - start_time

    return selected_clients, duration


# Define the fixed directory
result_directory = "/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/results/svhn2/"

# Ensure the directory exists, If folder does not exist, it creates it.
os.makedirs(result_directory, exist_ok=True)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration
NUM_CLUSTERS = 10
CLIENTS_PER_CLUSTER = [12,8,10,11,9,5,15,10,4,16]
K = 2
SERVER_CAPACITY = NUM_CLUSTERS * K
ROUNDS = 3000
learning_rate = 0.01
samples_per_client = 500
weight = 0.4
D = 0.14 # Example deadline for transmission time (in seconds)

LOCAL_EPOCHS = 1
G = 5  # Example threshold for L2 norm
QUANTIZATION_BIT = 8
threshold = None
top_n = True
client_select_type = 'custom'
selection_scope = 'global'
VRF_scope = True

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
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10('/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/food101/data', train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10('/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/food101/data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    distribute_dirichlet(clusters, train_dataset, num_classes=10, alpha=0.1)
    check_data_distribution(clusters)  
    
    loss_list = []
    training_loss_list = []
    accuracy_list = []
    total_transmitted_bits_list = []
    total_energy_list = []
    old_list = []
    round_duration_list = []
    VRF_duration_list = []
    client_selection_duration_list = []
    
    num_clients = sum(len(cluster.clients) for cluster in clusters) 
    client_selection_tracker = {client_id: 0 for client_id in range(num_clients)}
    
    for round_num in range(ROUNDS):
        start_time = time.time()

        print(f"\n\nFederated Learning Round {round_num + 1} *********************")

        server.distribute_model()
        
        server.train_and_setMinMax(LOCAL_EPOCHS)

        client_selection_duration_start = time.time()

        primary_pool = server.select_clients(K, G, D, client_select_type, round_num + 1, weight, global_total_samples, num_classes, 
                                                 selection_scope, SERVER_CAPACITY, VRF_scope)
        
        selected_clients, VRF_duration = VRF_client_select(SERVER_CAPACITY, primary_pool, server)

        if(len(selected_clients)>SERVER_CAPACITY):
            selected_clients = random.sample(selected_clients, SERVER_CAPACITY)

        server.selected_clients = selected_clients

        client_selection_duration = time.time() - client_selection_duration_start

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
        
        server.aggregate_quantized_grads()

        round_duration = (time.time() - time2_start) + time1

        loss, accuracy = evaluate(server.global_model, device, test_loader)
        
        total_training_loss = 0
        
        print("\nAll clients metric for round:", round_num + 1)
        for cluster in clusters:
            for client in cluster.clients:
                total_training_loss += client.local_loss               
        
        total_clients = sum(len(cluster.clients) for cluster in clusters)        
        average_training_loss = total_training_loss / total_clients
        
        # Calculate total transmitted bits and energy for the round
        total_transmitted_bits = sum(client.transmitted_bits for client, _, _ in selected_clients)
        total_energy = sum(client.energy for client, _, _ in selected_clients)
        
        total_transmitted_bits_list.append(total_transmitted_bits)
        total_energy_list.append(round(total_energy,5))
        
        loss_list.append(round(loss,4))
        training_loss_list.append(round(average_training_loss,4))
        accuracy_list.append(round(accuracy,2))
        round_duration_list.append(round(round_duration, 2))
        VRF_duration_list.append(round(VRF_duration,2))
        client_selection_duration_list.append(round(client_selection_duration,2))

    client_ids = list(client_selection_tracker.keys())
    selection_counts = list(client_selection_tracker.values())

    results_as_list = os.path.join(result_directory, 'output_cifar10_'+client_select_type+'_global_dirichlet_Q8_lr0.01_ep1_w0.4_vrf_UtilitySigned.txt') 
    
    with open(results_as_list, 'w') as file:
        # Write each list in the desired format
        file.write(f'loss_{client_select_type} = {loss_list}\n')
        file.write(f'accuracy_{client_select_type} = {accuracy_list}\n')
        file.write(f'bits_{client_select_type} = {total_transmitted_bits_list}\n')
        file.write(f'energy_{client_select_type} = {total_energy_list}\n')
        file.write(f'train_loss_{client_select_type} = {training_loss_list}\n')
        file.write(f'client_id_{client_select_type} = {client_ids}\n') #which client selected how many times (key)
        file.write(f'selection_count_{client_select_type} = {selection_counts}\n') #which client selected how many times (value)
        file.write(f'total_round_duration = {round_duration_list}\n')
        file.write(f'VRF_duration = {VRF_duration_list}\n')
        file.write(f'client_selection_duration = {client_selection_duration_list}\n')
        



if __name__ == "__main__":
    main()