import numpy as np
import scipy.sparse as sp
import random
import time
import os
from tqdm import tqdm


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
    num_nodes = len(unique_nodes)
    
    remapped_edges = []
    for u, v in edges:
        remapped_edges.append((node_map[u], node_map[v]))
        
    edges_np = np.array(remapped_edges, dtype=np.int32)
    return num_nodes, edges_np, node_map

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


class RISValidator_IC:
    def __init__(self, prob_matrix, num_nodes):
        self.num_nodes = num_nodes
        rev_adj = prob_matrix.transpose().tocoo()
        self.rev_adj_list = [[] for _ in range(num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            self.rev_adj_list[u].append((v, p))
        self.rr_sets = []
        self.node_to_rr_indices = [set() for _ in range(num_nodes)]

    def generate_rr_sets(self, theta=10000):
        print(f"Generating {theta} RR sets for validation...")
        roots = np.random.randint(0, self.num_nodes, theta)
        for rr_id, root in enumerate(tqdm(roots)):
            q = [root]
            visited = {root}
            while q:
                u = q.pop(0)
                self.node_to_rr_indices[u].add(rr_id)
                for v, prob in self.rev_adj_list[u]:
                    if v not in visited:
                        if random.random() <= prob:
                            visited.add(v)
                            q.append(v)
        self.total_rr = theta
        print("RR sets generated.")

    def evaluate(self, seed_set):
        if len(seed_set) == 0:
            return 0.0
        covered_rr_indices = set()
        for seed in seed_set:
            if seed < self.num_nodes:
                covered_rr_indices.update(self.node_to_rr_indices[seed])
        fraction = len(covered_rr_indices) / self.total_rr
        estimated_spread = self.num_nodes * fraction
        return estimated_spread



class RISValidator_LT:
    def __init__(self, prob_matrix, num_nodes):
        self.num_nodes = num_nodes
        rev_adj = prob_matrix.transpose().tocoo()
        self.rev_adj_list = [[] for _ in range(num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            self.rev_adj_list[u].append((v, p))
        self.rr_sets = []
        self.node_to_rr_indices = [set() for _ in range(num_nodes)]

    def generate_rr_sets(self, theta=10000):
        print(f"Generating {theta} RR sets for validation (LT Logic)...")
        roots = np.random.randint(0, self.num_nodes, theta)
        for rr_id, root in enumerate(tqdm(roots)):
            curr = root
            self.node_to_rr_indices[curr].add(rr_id)
            visited = {curr}           
            while True:
                neighbors = self.rev_adj_list[curr]
                if not neighbors:
                    break
                rand_val = random.random()
                cumulative = 0.0
                selected_v = None
                for v, prob in neighbors:
                    cumulative += prob
                    if rand_val <= cumulative:
                        selected_v = v
                        break
                if selected_v is not None:
                    if selected_v in visited:
                        break
                    else:
                        curr = selected_v
                        visited.add(curr)
                        self.node_to_rr_indices[curr].add(rr_id)
                else:
                    break
        self.total_rr = theta
        print("RR sets generated.")

    def evaluate(self, seed_set):
        if len(seed_set) == 0:
            return 0.0
        covered_rr_indices = set()
        for seed in seed_set:
            if seed < self.num_nodes:
                covered_rr_indices.update(self.node_to_rr_indices[seed])
        fraction = len(covered_rr_indices) / self.total_rr
        estimated_spread = self.num_nodes * fraction
        return estimated_spread



def self_inference_validation(dataset_name, seed_ratio, infer_model='IC'):

    graph_path = f'data/{dataset_name}.txt'
    
    seed_files = [f'seed/{dataset_name}-{seed_ratio}-{infer_model}.txt']

    num_nodes, edges_np, node_map = read_graph_from_txt(graph_path)
    prob_matrix = build_prob_matrix(num_nodes, edges_np)

    if infer_model == 'IC':
        validator = RISValidator_IC(prob_matrix, num_nodes)
    elif infer_model == 'LT':
        validator = RISValidator_LT(prob_matrix, num_nodes)

    validator.generate_rr_sets(theta=10000) 

    print("\n" + "="*40)
    print(f"Validation Results ({dataset_name})")
    print("="*40)
    
    for file_path in seed_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}: File not found.")
            continue 
        try:
            with open(file_path, 'r') as f:
                raw_seeds = [int(float(x)) for x in f.read().split()]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        internal_seeds = []
        for s in raw_seeds:
            if s in node_map:
                internal_seeds.append(node_map[s])

        spread = validator.evaluate(internal_seeds)/num_nodes
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"  - Seed Count: {len(internal_seeds)}")
        print(f"  - Est. Spread: {spread:.4f}")
        print("-" * 40)


