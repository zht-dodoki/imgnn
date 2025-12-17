import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import copy

def collate_fn_batch(batch):
    features_list, adj_list, label_list = zip(*batch)
    
    features_list = [f if isinstance(f, torch.Tensor) else torch.from_numpy(f) for f in features_list]
    batch_x = torch.cat(features_list, dim=0)
    label_list = [l if isinstance(l, torch.Tensor) else torch.from_numpy(l) for l in label_list]
    batch_y = torch.cat(label_list, dim=0)
    
    if sp.issparse(adj_list[0]):
        batch_adj_np = sp.block_diag(adj_list, format='csr')
        batch_adj_np = batch_adj_np.tocoo()
        indices = torch.from_numpy(np.vstack((batch_adj_np.row, batch_adj_np.col)).astype(np.int64))
        values = torch.from_numpy(batch_adj_np.data.astype(np.float32))
        shape = torch.Size(batch_adj_np.shape)
        batch_adj = torch.sparse_coo_tensor(indices, values, shape)
        
    else:
        all_indices = []
        all_values = []
        offset = 0
        total_size = 0
        for adj in adj_list:
            if not adj.is_coalesced():
                adj = adj.coalesce()
            indices = adj.indices() # [2, NNZ]
            values = adj.values()   # [NNZ]
            current_size = adj.shape[0]
            
            new_indices = indices.clone()
            new_indices[0, :] += offset
            new_indices[1, :] += offset
            
            all_indices.append(new_indices)
            all_values.append(values)
            offset += current_size

        cat_indices = torch.cat(all_indices, dim=1)
        cat_values = torch.cat(all_values, dim=0)
        batch_adj = torch.sparse_coo_tensor(cat_indices, cat_values, size=(offset, offset))
    batch_num_nodes = [f.shape[0] for f in features_list]
    return batch_x, batch_adj, batch_y, batch_num_nodes



def loss_fun(pred, target):
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)
    pred_diff = pred - pred.t()
    target_diff = target - target.t()
    S = torch.sign(target_diff)
    loss = F.relu(-S * pred_diff)
    mask = (S != 0)
    loss = loss[mask].mean() 
    return loss



def read_graph_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split())
        edges = [tuple(map(int, line.split())) for line in lines[1:]]
    return num_nodes, num_edges, edges


def create_adj_pairs(graph):
    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edges)
    adj_matrix = nx.adjacency_matrix(G)
    return adj_matrix


def add_prob_mat(graph):
    in_degrees = np.array(graph.adj_matrix.sum(axis=0)).flatten()
    in_degrees[in_degrees == 0] = 1
    prob_data = copy.copy(graph.adj_matrix.data)
    prob_indices = copy.copy(graph.adj_matrix.indices)
    prob_indptr = copy.copy(graph.adj_matrix.indptr)
    prob_shape = copy.copy(graph.adj_matrix.shape)
    for i, v in enumerate(prob_data):
        v = prob_indices[i]
        prob_data[i] = 1.0 / in_degrees[v]
    prob_matrix = sp.csr_matrix((prob_data, prob_indices, prob_indptr), shape=prob_shape)
    graph.prob_matrix = prob_matrix.astype(np.float32)
    graph.prob_matrix_copy = graph.prob_matrix.copy()
    return graph



def DeepIM_diffusion_evaluation(adj_matrix, seed, prb, diffusion='IC'):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    total = 0
    for _ in range(1000):
        if diffusion == 'LT':
            model = ep.ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                config.add_node_configuration("threshold", n, 0.5)

        elif diffusion == 'IC':
            model = ep.IndependentCascadesModel(G)
            config = mc.Configuration()
            for e in G.edges():
                config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])

        elif diffusion == 'SIS':
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)

        config.add_model_initial_configuration("Infected", seed)
        model.set_initial_status(config)
        iterations = model.iteration_bunch(50)
        node_status = iterations[0]['status']
        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])
        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1
        total += inf_vec.sum()
    return total/1000

