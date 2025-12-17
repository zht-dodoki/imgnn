import os
import time
import pickle
import random
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class GraphData:
    def __init__(self, adj_matrix, num_nodes):
        self.adj_matrix = adj_matrix  
        self.num_nodes = num_nodes
        self.rev_adj_list = self._build_reverse_adj_list()

    def _build_reverse_adj_list(self):
        rev_adj = self.adj_matrix.transpose().tocoo()
        adj_list = [[] for _ in range(self.num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            adj_list[u].append((v, p))
        return adj_list


class RISSampler_IC:
    def __init__(self, graph_data):
        self.graph = graph_data
    def generate_rr_sets(self, num_samples):
        rr_sets = []
        roots = np.random.randint(0, self.graph.num_nodes, num_samples)
        for root in roots:
            rr_set = {root}
            q = [root]
            while q:
                u = q.pop(0)
                if u >= len(self.graph.rev_adj_list): continue
                neighbors = self.graph.rev_adj_list[u]
                for v, prob in neighbors:
                    if v not in rr_set:
                        if random.random() <= prob:
                            rr_set.add(v)
                            q.append(v)
            rr_sets.append(rr_set)
        return rr_sets, roots



class RISSampler_LT:
    def __init__(self, graph_data):
        self.graph = graph_data
    def generate_rr_sets(self, num_samples):
        rr_sets = []
        roots = np.random.randint(0, self.graph.num_nodes, num_samples)
        for root in roots:
            rr_set = {root}
            curr = root
            while True:
                if curr >= len(self.graph.rev_adj_list): 
                    break
                neighbors = self.graph.rev_adj_list[curr]
                if not neighbors:
                    break
                candidates = [v for v, w in neighbors]
                weights = [w for v, w in neighbors]
                next_node_list = random.choices(candidates, weights=weights, k=1)
                next_node = next_node_list[0]
                if next_node in rr_set:
                    break
                rr_set.add(next_node)
                curr = next_node
            rr_sets.append(rr_set)
        return rr_sets, roots


def read_graph_topology(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines: raise ValueError("Empty file")        
        first_line = lines[0].split()
        if len(first_line) == 2:
            num_nodes = int(first_line[0])
            raw_edges = [list(map(int, line.split())) for line in lines[1:]]
        else:
            raw_edges = [list(map(int, line.split())) for line in lines]
            nodes = set()
            for u, v in raw_edges:
                nodes.add(u)
                nodes.add(v)
            num_nodes = max(nodes) + 1
            
    edges_np = np.array(raw_edges, dtype=np.int32)
    print(f"Graph Loaded: {num_nodes} nodes, {len(edges_np)} edges.")
    return num_nodes, edges_np


def generate_subgraph_topology(num_nodes_total, edges_np, keep_ratio):

    num_keep = int(num_nodes_total * keep_ratio)
    kept_nodes = np.sort(np.random.choice(num_nodes_total, num_keep, replace=False))

    mask_u = np.isin(edges_np[:, 0], kept_nodes)
    mask_v = np.isin(edges_np[:, 1], kept_nodes)
    valid_edges_mask = mask_u & mask_v
    sub_edges_raw = edges_np[valid_edges_mask]

    new_u = np.searchsorted(kept_nodes, sub_edges_raw[:, 0])
    new_v = np.searchsorted(kept_nodes, sub_edges_raw[:, 1])
    sub_edges_remapped = np.stack([new_u, new_v], axis=1)
    
    return num_keep, sub_edges_remapped

def build_prob_matrix_from_edges(num_nodes, edges_np):
    if len(edges_np) == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    in_degrees = np.zeros(num_nodes, dtype=np.int32)
    counts = np.bincount(edges_np[:, 1])
    if len(counts) < num_nodes:
        counts = np.pad(counts, (0, num_nodes - len(counts)))
    in_degrees = counts
    in_degrees_float = in_degrees.astype(np.float32)
    in_degrees_float[in_degrees_float == 0] = 1.0

    row = edges_np[:, 0]
    col = edges_np[:, 1]
    data = 1.0 / in_degrees_float[col]
    prob_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    return prob_matrix


def process_single_subgraph(sub_adj, seed_ratio, theta_feat=50, theta_label=2000,infer_model='IC'):
    num_nodes = sub_adj.shape[0]
    sub_graph = GraphData(sub_adj, num_nodes)
    if infer_model == 'IC':
        sampler = RISSampler_IC(sub_graph)
    elif infer_model == 'LT':
        sampler = RISSampler_LT(sub_graph)
    num_seeds = int(num_nodes * seed_ratio)
    seeds = set(np.random.choice(num_nodes, num_seeds, replace=False)) if num_seeds > 0 else set()
    rr_feat, _ = sampler.generate_rr_sets(theta_feat)
    rr_label, rr_roots = sampler.generate_rr_sets(theta_label)
    
    covered_nodes = set(seeds)
    covered_rr_indices_label = []
    
    for i, rr in enumerate(rr_label):
        if not rr.isdisjoint(seeds): 
            covered_nodes.add(rr_roots[i]) 
            covered_rr_indices_label.append(i)
            
    status_feat = np.ones(num_nodes, dtype=np.float32)
    for u in covered_nodes:
        status_feat[u] = 0.0

    valid_rr_feat = [rr for rr in rr_feat if rr.isdisjoint(seeds)]
    
    covered_indices_set = set(covered_rr_indices_label)
    valid_rr_label = [rr for i, rr in enumerate(rr_label) if i not in covered_indices_set]
            
    def count_frequency(rr_list, total_rr_count):
        counts = np.zeros(num_nodes, dtype=np.float32)
        if total_rr_count == 0: return counts
        for rr in rr_list:
            for node in rr:
                counts[node] += 1
        return counts / total_rr_count

    ris_feat = count_frequency(valid_rr_feat, theta_feat)
    label = count_frequency(valid_rr_label, theta_label)
    features = np.stack([status_feat, ris_feat], axis=1)
    
    return features, label, sub_adj


def generate_dataset(file_name, num_subgraphs=500, infer='IC'):
    file_txt_path = f'data/{file_name}.txt'
    num_nodes_total, edges_total_np = read_graph_topology(file_txt_path)
    
    dataset = []
    
    print(f"Start generating {num_subgraphs} high-quality (hard) subgraphs...")
    start_time = time.time()  
    generated_count = 0
    pbar = tqdm(total=num_subgraphs)
    
    while generated_count < num_subgraphs:
        keep_ratio = random.uniform(0.7, 1.0)
        num_sub_nodes, sub_edges = generate_subgraph_topology(num_nodes_total, edges_total_np, keep_ratio)
        sub_adj = build_prob_matrix_from_edges(num_sub_nodes, sub_edges)
        rand_val = random.random()

        if rand_val < 0.7:
            seed_ratio = random.uniform(0.0, 0.01) 
        else:
            seed_ratio = random.uniform(0.01, 0.03) 
            
        if random.random() < 0.05:
            seed_ratio = 0.0

        features, label, adj = process_single_subgraph(sub_adj, seed_ratio, theta_feat=50, theta_label=10000, infer_model=infer)
        if np.max(label) < 1e-3:
            continue
            
        dataset.append({'adj': adj, 'features': features, 'label': label})
        
        generated_count += 1
        pbar.update(1)
        
    pbar.close()
    end_time = time.time()
    print(f"Done. Total time: {end_time - start_time:.2f}s")
    
    os.makedirs('inidata', exist_ok=True)
    save_path = f'inidata/{file_name}_data_{infer}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved to {save_path}")

