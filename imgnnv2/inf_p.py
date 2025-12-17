import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import time
import os
import random
from tqdm import tqdm
from models import GNN


def read_graph_from_txt(file_path):
    print(f"Reading graph from {file_path}...")
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'): continue
            parts = line.split()
            if len(parts) < 2: continue
            try:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
            except ValueError:
                continue

    unique_nodes = set()
    for u, v in edges:
        unique_nodes.add(u)
        unique_nodes.add(v)
    
    sorted_nodes = sorted(list(unique_nodes))
    node_map = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
    reverse_map = {new_id: old_id for new_id, old_id in enumerate(sorted_nodes)}
    
    num_nodes = len(unique_nodes)
    
    remapped_edges = []
    for u, v in edges:
        remapped_edges.append((node_map[u], node_map[v]))
        
    edges_np = np.array(remapped_edges, dtype=np.int32)
    print(f"Graph loaded: {num_nodes} nodes, {len(edges_np)} edges (Remapped).")
    return num_nodes, edges_np, reverse_map

def build_prob_matrix(num_nodes, edges_np):
    if len(edges_np) > 0:
        counts = np.bincount(edges_np[:, 1])
        if len(counts) < num_nodes:
            counts = np.pad(counts, (0, num_nodes - len(counts)))
        in_degrees = counts.astype(np.float32)
        in_degrees[in_degrees == 0] = 1.0 
        
        row = edges_np[:, 0]
        col = edges_np[:, 1]
        data = 1.0 / in_degrees[col]
        
        prob_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    else:
        prob_matrix = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    return prob_matrix

def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)

class RISSampler_Inference_IC:
    def __init__(self, prob_matrix, num_nodes):
        self.num_nodes = num_nodes
        rev_adj = prob_matrix.transpose().tocoo()
        self.rev_adj_list = [[] for _ in range(num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            self.rev_adj_list[u].append((v, p))

    def get_ris_features(self, theta=5000): 
        print(f"Sampling {theta} RR sets for features...")
        rr_counts = np.zeros(self.num_nodes, dtype=np.float32)
        roots = np.random.randint(0, self.num_nodes, theta)
        
        for root in tqdm(roots, desc="RIS Sampling"):
            q = [root]
            rr_counts[root] += 1
            visited = {root}
            while q:
                u = q.pop(0)
                for v, prob in self.rev_adj_list[u]:
                    if v not in visited:
                        if random.random() <= prob:
                            visited.add(v)
                            rr_counts[v] += 1
                            q.append(v)
        
        return rr_counts / theta


class RISSampler_Inference_LT:
    def __init__(self, prob_matrix, num_nodes):
        self.num_nodes = num_nodes
        rev_adj = prob_matrix.transpose().tocoo()
        self.rev_adj_list = [[] for _ in range(num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            self.rev_adj_list[u].append((v, p))

    def get_ris_features(self, theta=5000): 
        print(f"Sampling {theta} RR sets for features (LT Model)...")
        rr_counts = np.zeros(self.num_nodes, dtype=np.float32)
        roots = np.random.randint(0, self.num_nodes, theta)
        for root in tqdm(roots, desc="RIS Sampling"):
            visited = {root}
            rr_counts[root] += 1
            curr = root
            while True:
                neighbors = self.rev_adj_list[curr]
                if not neighbors:
                    break   
                rand_val = random.random()
                cumulative = 0.0
                selected_v = None
                for v, p in neighbors:
                    cumulative += p
                    if rand_val <= cumulative:
                        selected_v = v
                        break
                if selected_v is not None:
                    if selected_v not in visited:
                        visited.add(selected_v)
                        rr_counts[selected_v] += 1
                        curr = selected_v
                    else:
                        break
                else:
                    break
        return rr_counts / theta


def inference(dataset_name, txtdata_path, seed_ratio, device, infer_model='IC'):

    file_path = f'data/{txtdata_path}.txt'

    model_path = f'{dataset_name}_gcn_{infer_model}.pth'
    if seed_ratio =='1%':
        ratio = 0.01
    elif seed_ratio =='5%':
        ratio = 0.05
    elif seed_ratio =='10%':
        ratio = 0.10
        
    num_nodes, edges_np, reverse_map = read_graph_from_txt(file_path)
    prob_matrix = build_prob_matrix(num_nodes, edges_np)
    
    if infer_model == 'IC':
        sampler = RISSampler_Inference_IC(prob_matrix, num_nodes)
    elif infer_model == 'LT':
        sampler = RISSampler_Inference_LT(prob_matrix, num_nodes)

    ris_scores = sampler.get_ris_features(theta=100) 
    feat_ris = torch.from_numpy(ris_scores).unsqueeze(1).float().to(device)
    
    adj_torch = sparse_mx_to_torch_sparse_tensor(prob_matrix, device)
    
    print("Loading model...")
    model = GNN(in_features=2, hidden_features=64, out_features=1).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        model = torch.load(model_path, map_location=device)
    model.eval()
    
    num_seeds = int(num_nodes * ratio)
    if num_seeds*0.01 < 1:
        batch_size = 1
    else:
        batch_size = int(num_seeds*0.01) 


    print(f"Selecting {num_seeds} seeds...")
    
    seeds = []
    status_np = np.ones(num_nodes, dtype=np.float32)
    
    begin_time = time.time()
    
    with tqdm(total=num_seeds) as pbar:
        while len(seeds) < num_seeds:
            feat_status = torch.from_numpy(status_np).unsqueeze(1).float().to(device)
            x = torch.cat([feat_status, feat_ris], dim=1)
            
            with torch.no_grad():
                logits = model(x, adj_torch)
                scores = torch.sigmoid(logits).cpu().numpy() 
            
            if len(seeds) > 0:
                scores[seeds] = -1.0
            
            needed = num_seeds - len(seeds)
            k = min(batch_size, needed)
            
            current_batch_indices = np.argpartition(scores, -k)[-k:]
            
            top_scores = scores[current_batch_indices]
            sorted_indices = current_batch_indices[np.argsort(top_scores)[::-1]]
            
            seeds.extend(sorted_indices)
            
            status_np[sorted_indices] = 0.0 
            pbar.update(k)
            
    end_time = time.time()
    print(f"Total Inference Time: {end_time - begin_time:.4f}s")
    
    original_id_seeds = [reverse_map[s] for s in seeds]
    save_file = f'seed/{txtdata_path}-{seed_ratio}-{infer_model}.txt'
    
    np.savetxt(save_file, np.array(original_id_seeds), fmt='%d')
    print(f"Seeds saved to {save_file}")
    return original_id_seeds

