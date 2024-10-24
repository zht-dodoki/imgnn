import os
import torch

import numpy as np
import copy
import time
import networkx as nx
import random
import pickle

import scipy.sparse as sp
import matplotlib.pyplot as plt
import graphdata
from copy import deepcopy


def read_graph_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split())
        edges = [tuple(map(int, line.split())) for line in lines[1:]]

    return num_nodes, num_edges, edges


def create_adj_pairs(graph,edges):
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(edges)
    adj_matrix = nx.adjacency_matrix(G)
    return adj_matrix


def add_prob_mat(graph):
    """Add a diffusion probability matrix to the graph."""
    prob_data = copy.copy(graph.adj_matrix.data)
    prob_indices = copy.copy(graph.adj_matrix.indices)
    prob_indptr = copy.copy(graph.adj_matrix.indptr)
    prob_shape = copy.copy(graph.adj_matrix.shape)

    for i, v in enumerate(prob_data):
        prob_data[i] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    prob_matrix = sp.csr_matrix((prob_data, prob_indices, prob_indptr), shape=prob_shape)
    graph.prob_matrix = prob_matrix.astype(np.float32)
    graph.prob_matrix_copy = graph.prob_matrix.copy()
    return graph


def gen_seed_vec(N, n):
    seed_vec = np.zeros((N,))
    seeds = np.random.choice(N, size=n, replace=False)
    seed_vec[seeds] = 1

    return seed_vec


def add_data(graph, num_list):
    N = graph.prob_matrix.shape[0]
    seed_vec = np.zeros((N,))

    inverse = torch.zeros((N, 2), dtype=torch.float32)
    inverse_pairs = []

    graph.adj_list = [graph.adj_matrix.copy() for _ in range(50)]
    s = 0
    for i in num_list:
        n = int(graph.num_nodes * i / 100)
        for j in range(5):
            inverse = torch.zeros((N, 2), dtype=torch.float32)

            now_inversed_list = gen_seed_vec(N, n).astype(np.int64)
            mask = now_inversed_list == 1

            graph.prob_matrix[mask, :] = 0.0
            graph.prob_matrix[:, mask] = 0.0

            graph.adj_list[s][mask, :] = 0.0
            graph.adj_list[s][:, mask] = 0.0

            s += 1

            for m in range(graph.num_nodes):
                if now_inversed_list[m] == 1:
                    inverse[m, 0] = 1
                    continue
                b = time.time()
                print('---------------------' + str(i) + '%---------------------' + str(
                    j) + '---------------------node：' + str(m) + '---------------------' + str(
                    s) + '---------------------')
                seed_vec[m] = 1
                res = run_mc_repeats_(graph, seed_vec, 1, 15)
                res = res[:, -1]
                res[res >= 0.1] = 1
                res[res < 0.1] = 0
                inverse[m, 1] = np.sum(res)
                print(inverse[m, 1])
                seed_vec[m] = 0
                e = time.time()
                print(f"Time: {e - b:.4f}s")

            inverse_pairs.append(deepcopy(inverse))
            graph.prob_matrix = graph.prob_matrix_copy.copy()

    graph.prob_matrix = graph.prob_matrix_copy.copy()
    inverse_pairs = torch.stack(inverse_pairs)
    return inverse_pairs


def run_mc_repeats_(graph, seed_vec, repeat=10, diffusion_limit=15, re=True):
    influ_mat = np.zeros((graph.prob_matrix.shape[0], diffusion_limit))

    for i in range(repeat):
        this_mat = run_mc_(graph, seed_vec, diffusion_limit)
        influ_mat += this_mat

    influ_mat /= repeat
    return influ_mat


def run_mc_(graph, seed_vec, diffusion_limit=25) -> np.ndarray:
    activated_vec = seed_vec.copy()
    influ_mat = [seed_vec, ]
    last_activated = np.argwhere(seed_vec == 1).flatten().tolist()
    next_activated = []
    diffusion_count = 0

    while len(last_activated) > 0:
        for u in last_activated:
            u_neighs = graph.adj_matrix[[u]].nonzero()[1]

            for v in u_neighs:
                if activated_vec[v] == 0:
                    active_neighbors = graph.adj_matrix[v:v + 1, :].nonzero()[1]
                    influence_sum = sum([graph.prob_matrix[w, v] for w in active_neighbors if activated_vec[w] == 1])

                    if influence_sum >= 1:
                        activated_vec[v] = 1
                        next_activated.append(v)

        influ_mat.append(activated_vec.copy())
        last_activated = next_activated
        next_activated = []
        diffusion_count += 1
        if len(influ_mat) >= diffusion_limit:
            break

    if len(influ_mat) < diffusion_limit:
        last_state = influ_mat[-1]
        influ_mat.extend([last_state] * (diffusion_limit - len(influ_mat)))

    influ_mat = np.array(influ_mat).T
    return influ_mat


def ltdata(file_name,train=True):
    # file_name = 'CA-GrQc'
    file_txt_path = f'data/{file_name}.txt'

    if os.path.exists(file_txt_path):
        file_pkl_path = f'pkldata/{file_name}_LT.pkl'
        target_pkl_path = f'pkltarget/{file_name}_LT.pkl'

        if os.path.exists(file_pkl_path):
            print(f"The file '{file_pkl_path}' exists.")
        else:
            num_nodes, num_edges, edges = read_graph_from_txt(file_txt_path)
            my_graph = graphdata.myGraph(num_nodes, edges)
            my_graph.adj_matrix = create_adj_pairs(my_graph, edges).astype(np.float32)
            my_graph = add_prob_mat(my_graph)
            b = time.time()
            if train:
                my_graph.inverse_pairs = add_data(my_graph, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
            e = time.time()
            print(f"Time: {e - b:.4f}s")

            if train:
                with open(file_pkl_path, 'wb') as file:
                    pickle.dump(my_graph, file)
            else:
                with open(target_pkl_path, 'wb') as file:
                    pickle.dump(my_graph, file)
    else:
        print(f"The file '{file_txt_path}' does not exist.")
        exit()

# if __name__ == "__main__":
#
#     file_name = 'CA-GrQc'
#     file_txt_path = f'my_data/{file_name}.txt'
#
#     if os.path.exists(file_txt_path):
#         file_pkl_path = f'my_pkl/{file_name}_lt.pkl'
#         if os.path.exists(file_pkl_path):
#             print(f"The file '{file_pkl_path}' exists.")
#         else:
#             num_nodes, num_edges, edges = read_graph_from_txt(file_txt_path)
#             my_graph = graphdata.myGraph(num_nodes, edges)
#             my_graph.adj_matrix = create_adj_pairs(my_graph).astype(np.float32)
#             my_graph = add_prob_mat(my_graph)
#             b = time.time()
#             my_graph.inverse_pairs = add_data(my_graph, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
#             e = time.time()
#             print(f"Time: {e - b:.4f}s")
#
#             with open(file_pkl_path, 'wb') as file:
#                 pickle.dump(my_graph, file)
#     else:
#         print(f"The file '{file_txt_path}' does not exist.")
#         exit()
