import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from torchvision import datasets, transforms
import argparse
import random
import math
import os
from gurobipy import Model, GRB, quicksum
np.random.seed(42)
 
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Yogi Optimizer (kept for reference, not used)
class Yogi(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Yogi does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_squared = grad.mul(grad)
                v.add_(torch.sign(grad_squared - v).mul(grad_squared), alpha=1 - beta2)
                step_size = group['lr']
                denom = v.sqrt().add_(group['eps'])
                p.data.addcdiv_(m, denom, value=-step_size)
        return loss
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class OortSelector:
    def __init__(self, exploration_factor=0.5, fairness_T=50, deadline=200.0):
        self.clients = {}
        self.round = 0
        self.exploration_factor = exploration_factor
        self.fairness_T = fairness_T
        self.deadline = deadline
        self.cluster_selection_counts = {}

    def register_client(self, cid, duration=0.0):
        self.clients[cid] = {
            'reward': 0.0,
            'duration': duration,
            'count': 0,
            'last_selected': -self.fairness_T,
            'cluster_id': None
        }

    def update_client_feedback(self, cid, reward, duration):
        self.clients[cid]['reward'] = reward
        self.clients[cid]['duration'] = duration
        self.clients[cid]['count'] += 1
        self.clients[cid]['last_selected'] = self.round

    def update_cluster(self, cid, cluster_id):
        self.clients[cid]['cluster_id'] = cluster_id
        if cluster_id not in self.cluster_selection_counts:
            self.cluster_selection_counts[cluster_id] = 0

    def compute_utility(self, cid):
        client = self.clients[cid]
        duration = client['duration'] if client['duration'] > 0 else 1e-4
        exploitation = client['reward'] / duration
        if client['count'] == 0:
            exploration = 1000.0
        else:
            standard = math.sqrt(math.log(self.round + 1) / client['count'])
            cluster_id = client['cluster_id']
            cluster_count = self.cluster_selection_counts.get(cluster_id, 1)
            heterogeneity = 1.0 / (cluster_count + 1e-4)
            exploration = self.exploration_factor * standard * (1.0 + heterogeneity)
        return exploitation + exploration

    def select_participant(self, num_clients, feasible_clients):
        if not feasible_clients:
            return []
        model = Model("client_selection")
        model.setParam('OutputFlag', 0)
        x = model.addVars(feasible_clients, vtype=GRB.BINARY, name="x")
        model.setObjective(quicksum(self.compute_utility(cid) * x[cid] for cid in feasible_clients), GRB.MAXIMIZE)
        model.addConstr(quicksum(x[cid] for cid in feasible_clients) == num_clients)
        model.addConstr(quicksum(self.clients[cid]['duration'] * x[cid] for cid in feasible_clients) <= self.deadline)
        for cid in feasible_clients:
            if self.round - self.clients[cid]['last_selected'] >= self.fairness_T:
                model.addConstr(x[cid] == 1)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            selected = [cid for cid in feasible_clients if x[cid].X > 0.5]
            for cid in selected:
                cluster_id = self.clients[cid]['cluster_id']
                self.cluster_selection_counts[cluster_id] = self.cluster_selection_counts.get(cluster_id, 0) + 1
            return selected
        else:
            logging.warning("No feasible solution found. Falling back to greedy selection.")
            sorted_clients = sorted(feasible_clients, key=lambda cid: self.compute_utility(cid), reverse=True)
            return sorted_clients[:num_clients]

def calculate_reward(loss, gradients, cluster_id, cluster_sizes, w_loss=0.6, w_grad=0.2, w_data=0.2):
    loss_reward = 1.0 / (loss + 1e-4)
    grad_norm = torch.norm(torch.cat([p.flatten() for p in gradients])).item()
    data_quality = 1.0 / (cluster_sizes.get(cluster_id, 1) + 1e-4)
    raw_reward = w_loss * loss_reward + w_grad * grad_norm + w_data * data_quality
    return raw_reward

def quantize_gradient_func(gradient, bits):
    """Applies per-client min-max uniform quantization to gradients."""
    
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
            torch.clamp(torch.round((grad - min_val) / scale), 0, quant_levels).to(dtype) for grad in gradient
        ]

    return quantized_gradient, min_val, max_val

def normalize_rewards(raw_rewards):
    if not raw_rewards:
        return raw_rewards
    min_reward = min(raw_rewards.values())
    max_reward = max(raw_rewards.values())
    if max_reward == min_reward:
        return {cid: 1.0 for cid in raw_rewards}
    return {cid: (r - min_reward) / (max_reward - min_reward) for cid, r in raw_rewards.items()}

def select_clients_with_constraints(selector, num_clients, feasible_clients, cluster_assignments, K, mode):
    selector.round += 1
    if mode == 'global':
        model = Model("global_selection")
        model.setParam('OutputFlag', 0)
        x = model.addVars(feasible_clients, vtype=GRB.BINARY, name="x")
        model.setObjective(quicksum(selector.compute_utility(cid) * x[cid] for cid in feasible_clients), GRB.MAXIMIZE)
        model.addConstr(quicksum(x[cid] for cid in feasible_clients) <= num_clients * 3)
        model.addConstr(quicksum(selector.clients[cid]['duration'] * x[cid] for cid in feasible_clients) <= selector.deadline * 3)
        for cid in feasible_clients:
            if selector.round - selector.clients[cid]['last_selected'] >= selector.fairness_T:
                model.addConstr(x[cid] == 1)
        model.optimize()
        if model.status != GRB.OPTIMAL:
            logging.warning("Global mode: No feasible solution. Falling back to greedy.")
            selected = sorted(feasible_clients, key=lambda cid: selector.compute_utility(cid), reverse=True)[:num_clients * 3]
        else:
            selected = [cid for cid in feasible_clients if x[cid].X > 0.5]
        cluster_counts = {}
        for cid in selected:
            cluster_id = cluster_assignments[cid]
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        valid_clusters = {cid: count for cid, count in cluster_counts.items() if count >= K}
        filtered = [cid for cid in selected if cluster_assignments[cid] in valid_clusters]
        logging.info(f"Cluster counts: {cluster_counts}")
        logging.info(f"Valid clusters (≥{K} clients): {list(valid_clusters.keys())}")
        if len(filtered) < num_clients:
            logging.warning(f"Global mode: Only {len(filtered)} clients from valid clusters. Using available valid clients.")
        filtered = filtered[:num_clients]
        for cid in filtered:
            cluster_id = cluster_assignments[cid]
            selector.cluster_selection_counts[cluster_id] = selector.cluster_selection_counts.get(cluster_id, 0) + 1
        return filtered
    else:
        selected = []
        cluster_groups = {}
        for cid in feasible_clients:
            cluster_id = cluster_assignments[cid]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(cid)
        for cluster_id, clients in cluster_groups.items():
            if len(clients) >= K:
                model = Model(f"cluster_{cluster_id}_selection")
                model.setParam('OutputFlag', 0)
                x = model.addVars(clients, vtype=GRB.BINARY, name="x")
                model.setObjective(quicksum(selector.compute_utility(cid) * x[cid] for cid in clients), GRB.MAXIMIZE)
                model.addConstr(quicksum(x[cid] for cid in clients) == K)
                model.addConstr(quicksum(selector.clients[cid]['duration'] * x[cid] for cid in clients) <= selector.deadline / len(cluster_groups))
                for cid in clients:
                    if selector.round - selector.clients[cid]['last_selected'] >= selector.fairness_T:
                        model.addConstr(x[cid] == 1)
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    cluster_selected = [cid for cid in clients if x[cid].X > 0.5]
                else:
                    logging.warning(f"Cluster {cluster_id}: No feasible solution. Using greedy.")
                    cluster_selected = sorted(clients, key=lambda cid: selector.compute_utility(cid), reverse=True)[:K]
                selected.extend(cluster_selected)
                selector.cluster_selection_counts[cluster_id] = selector.cluster_selection_counts.get(cluster_id, 0) + K
        if len(selected) < num_clients:
            logging.warning(f"Local mode: Selected only {len(selected)} clients due to insufficient clusters with >= {K} clients")
        selected = selected[:num_clients]
        logging.info(f"Selected from clusters: {list(set(cluster_assignments[cid] for cid in selected))}")
        return selected

class Cluster:
    def __init__(self, cluster_id, clients):
        self.cluster_id = cluster_id
        self.clients = clients

class Client:
    def __init__(self, client_id, cluster_id, data, model, device):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.data = data
        self.model = model.to(device)
        self.device = device
        self.local_loss = None
        self.local_l2_norm = None
        self.gradients = None

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_model(self, global_model_state):
        self.model.load_state_dict(global_model_state)

    def train(self, epochs=1):
        self.model.train()
        data_loader = DataLoader(self.data, batch_size=64, shuffle=True)
        total_l2_norm = 0.0
        total_loss = 0.0
        num_total_batches = 0
        accumulated_gradients = None
        for epoch in range(epochs):
            epoch_l2_norm = 0.0
            epoch_loss = 0.0
            num_batches = 0
            for data, target in data_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                self.model.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                l2_norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in self.model.parameters())).item()
                batch_gradients = [param.grad.clone() for param in self.model.parameters()]
                if accumulated_gradients is None:
                    accumulated_gradients = [torch.zeros_like(grad) for grad in batch_gradients]
                for i, grad in enumerate(batch_gradients):
                    accumulated_gradients[i] += grad
                epoch_l2_norm += l2_norm
                epoch_loss += loss.item()
                num_batches += 1
                num_total_batches += 1
            total_l2_norm += epoch_l2_norm / num_batches
            total_loss += epoch_loss / num_batches
        if epochs == 1:
            self.local_l2_norm = total_l2_norm
            self.local_loss = total_loss
        else:
            self.local_l2_norm = total_l2_norm / epochs
            self.local_loss = total_loss / epochs
        self.gradients = [grad / num_total_batches for grad in accumulated_gradients]
        return self.local_l2_norm

class Server:
    def __init__(self, clusters, device, learning_rate=0.01):
        self.clusters = clusters
        self.device = device
        self.global_model = SimpleNN().to(device)
        self.learning_rate = learning_rate

    def distribute_model(self):
        global_state = self.global_model.state_dict()
        for cluster in self.clusters:
            for client in cluster.clients:
                client.set_model(global_state)

    def aggregate_models(self, client_updates, quantization_bit):
        total_gradients = None
        total_samples = sum(update['samples'] for update in client_updates)
        
        for update in client_updates:
            if(quantization_bit<32):
                
                # Compute scale (same as in quantization)
                scale = (update['max_val'] - update['min_val']) / (2**quantization_bit - 1)                  

                # Dequantize
                dequantized_gradients = [
                (q_grad.to(torch.float32) * scale) + update['min_val'] for q_grad in update['gradients']
                ]
                
                client_gradients = dequantized_gradients 
            else:
                client_gradients = update['gradients']
            
            num_samples = update['samples']
            if total_gradients is None:
                total_gradients = [torch.zeros_like(grad) for grad in client_gradients]
            for i, grad in enumerate(client_gradients):
                total_gradients[i] += grad * num_samples
        for i in range(len(total_gradients)):
            total_gradients[i] = total_gradients[i] / total_samples
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), total_gradients):
                param -= self.learning_rate * grad

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return test_loss, accuracy

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
            subset = Subset(train_dataset, client_indices[client_id])
            if len(subset) == 0:
                print(f"[Warning] Client {client_id} has 0 samples. Assigning 1 random sample.")
                subset = Subset(train_dataset, [np.random.randint(0, len(train_dataset)-1)])
            client.set_data(subset)
            client_id += 1

global_total_samples = 0
def check_data_distribution(clusters):
    global global_total_samples
    global_total_samples = 0
    for cluster in clusters:
        for client in cluster.clients:
            client_data = client.get_data()
            class_counts = {}
            for _, target in client_data:
                class_counts[target] = class_counts.get(target, 0) + 1
            logging.info(f"Client {client.client_id}:")
            for class_id, count in class_counts.items():
                logging.info(f"  Class {class_id}: {count} samples")
                global_total_samples += count
            logging.info("-" * 20)
    logging.info(f"Total samples across all clusters = {global_total_samples}")

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--total_worker', type=int, default=100)
    parser.add_argument('--clients_per_round', type=int, default=20)
    parser.add_argument('--num_rounds', type=int, default=3000)
    parser.add_argument('--selection_mode', type=str, default='global', choices=['global', 'local'])
    parser.add_argument('--K', type=int, default=0)
    parser.add_argument('--exploration_factor', type=float, default=0.5)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--fairness_T', type=int, default=50)
    parser.add_argument('--deadline', type=float, default=200.0)
    parser.add_argument('--quantization_bit', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()

    result_directory = "/home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT/results/svhn2/"
    results_as_list = os.path.join(result_directory, 'output_mnist_' + "oort_"+ args.selection_mode + '_dirichlet_NEW_NonCluster.txt')
    configs_as_list = os.path.join(result_directory, 'output_mnist_' + "oort_"+ args.selection_mode + '_dirichlet_args_NEW_NonCluster.txt')
    with open(configs_as_list, 'w') as file:
        file.write(f'Run_configurations = {args}\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    cluster_sizes = [12, 8, 10, 11, 9, 5, 15, 10, 4, 16]
    clusters = []
    client_id = 0
    cluster_assignments = {}
    for cluster_id, size in enumerate(cluster_sizes):
        clients = []
        for _ in range(size):
            model = SimpleNN()
            clients.append(Client(client_id, cluster_id, None, model, device))
            cluster_assignments[client_id] = cluster_id
            client_id += 1
        clusters.append(Cluster(cluster_id, clients))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    distribute_dirichlet(clusters, train_dataset, num_classes=10, alpha=args.alpha)
    check_data_distribution(clusters)

    server = Server(clusters, device, learning_rate=args.learning_rate)
    selector = OortSelector(exploration_factor=args.exploration_factor, fairness_T=args.fairness_T, deadline=args.deadline)

    cluster_sizes_dict = dict(enumerate(cluster_sizes))
    for cluster in clusters:
        for client in cluster.clients:
            duration = random.uniform(1, 10)
            selector.register_client(client.client_id, duration)
            selector.update_cluster(client.client_id, cluster_assignments[client.client_id])

    loss_list = []
    accuracy_list = []
    round_duration_list = []

    num_clients = sum(len(cluster.clients) for cluster in clusters)
    client_selection_tracker = {client_id: 0 for client_id in range(num_clients)}

    for round_idx in range(args.num_rounds):
        start_time = time.time()
        logging.info(f"Round {round_idx + 1}/{args.num_rounds}")
        feasible_clients = list(range(args.total_worker))
        selected_clients = select_clients_with_constraints(
            selector, args.clients_per_round, feasible_clients, cluster_assignments, args.K, args.selection_mode
        )

        time1 = time.time() - start_time

        logging.info(f"Selected clients: {selected_clients}")

        # cluster_counts = {}
        # for cid in selected_clients:
        #     cluster_id = cluster_assignments[cid]
        #     cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # logging.info(f"Post-selection cluster counts: {cluster_counts}")

        # violating_clusters = [cluster_id for cluster_id, count in cluster_counts.items() if 0 < count < args.K]
        # if violating_clusters:
        #     logging.warning(f"Removing clients from violating clusters: {violating_clusters}")
        #     selected_clients = [cid for cid in selected_clients if cluster_assignments[cid] not in violating_clusters]
        #     cluster_counts = {}
        #     for cid in selected_clients:
        #         cluster_id = cluster_assignments[cid]
        #         cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        #     logging.info(f"Updated cluster counts after removal: {cluster_counts}")

        start_time = time.time()
        
        if not selected_clients:
            logging.warning("No valid clients selected after enforcing K constraint. Skipping round.")
            continue

        server.distribute_model()
        client_updates = []
        raw_rewards = {}
        for cid in selected_clients:
            for cluster in clusters:
                for client in cluster.clients:
                    if client.client_id == cid:
                        l2_norm = client.train(epochs=args.local_epochs)
                        #moved client_updates.append from here
                        cluster_id = cluster_assignments[cid]
                        raw_rewards[cid] = calculate_reward(
                            client.local_loss, client.gradients, cluster_id, cluster_sizes_dict
                        )

                        min_val = None
                        max_val = None

                        if (args.quantization_bit<32):
                            client.gradients, min_val, max_val = quantize_gradient_func(client.gradients, args.quantization_bit)

                        client_updates.append({
                            'loss': client.local_loss,
                            'samples': len(client.get_data()),
                            'gradients': client.gradients,
                            'min_val': min_val,
                            'max_val': max_val
                        })
                        break
        normalized_rewards = normalize_rewards(raw_rewards)
        for cid in selected_clients:
            reward = normalized_rewards[cid]
            duration = random.uniform(1, 10)
            selector.update_client_feedback(cid, reward, duration)
        server.aggregate_models(client_updates, args.quantization_bit)

        round_duration = (time.time() - start_time) + time1

        test_loss, accuracy = evaluate(server.global_model, device, test_loader)

        for client in selected_clients:
            client_selection_tracker[client] += 1
        
        loss_list.append(round(test_loss, 4))
        accuracy_list.append(round(accuracy, 2))
        round_duration_list.append(round(round_duration, 2))

    client_ids = list(client_selection_tracker.keys())
    selection_counts = list(client_selection_tracker.values())

    with open(results_as_list, 'w') as file:
        file.write(f'loss_oort = {loss_list}\n')
        file.write(f'accuracy_oort = {accuracy_list}\n')
        file.write(f'client_id_oort = {client_ids}\n')
        file.write(f'selection_count_oort = {selection_counts}\n')
        file.write(f'round_duration = {round_duration_list}\n')

    

if __name__ == "__main__":
    main()