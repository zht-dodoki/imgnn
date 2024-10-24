import torch
import numpy as np
import time
import pickle
import scipy.sparse as sp
from predata import run_mc_repeats_
import random

import matplotlib.pyplot as plt
import networkx as nx





print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1


def generate_unique_random_int(low, high, exclude, exclude2):
    exclude_set = set(exclude)
    exclude_set2 = set(exclude2)

    while True:
        random_int = random.randint(low, high)
        if random_int not in exclude_set and random_int not in exclude_set2:
            return random_int


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


file_name = 'youtube'

with open(f'newdata/{file_name}' + '.pkl', 'rb') as f:
    graph = pickle.load(f)

adj_list = []

model = torch.load('cora_ml.pth')
for param in model.parameters():
    param.requires_grad = False
model.eval()

seed_list = []
s = 0
N = graph.prob_matrix.shape[0]
seed_vec = np.zeros((N,))
seed_list_impact = np.zeros((N,))

res = 0
filename = 'my_seed/' + f'{file_name}_seed' + '.txt'

maxseednum = 10
impact = 0



def top_k_positions(dd, k):
    sorted_indices = torch.argsort(dd, descending=True)
    top_k_indices = sorted_indices[:k]
    top_k_positions = torch.zeros_like(dd)
    top_k_positions[top_k_indices] = 1
    return top_k_positions.numpy()


for j in range(1):
    begin = time.time()
    new_impact = 0

    mask = (seed_list_impact == 1).nonzero()[0]
    adj = graph.prob_matrix.copy().tolil()

    adj = adj.tocoo()
    mask_set = set(mask)

    adj.data[np.isin(adj.row, mask_set)] = 0.0
    adj.data[np.isin(adj.col, mask_set)] = 0.0

    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj.tocoo()
    indices = torch.LongTensor([adj.row, adj.col])
    values = torch.FloatTensor(adj.data)
    shape = torch.Size(adj.shape)
    adj = torch.sparse.FloatTensor(indices, values, shape)

    seed_list_impact = torch.from_numpy(seed_list_impact).float()

    # begin = time.time()
    res = model(seed_list_impact.unsqueeze(-1), adj)

    dd = res.squeeze(-1)

    max_value = torch.max(dd)
    max_indices = (dd == max_value).nonzero()
    max_indices_len = len(max_indices)
    dd[max_indices] = 0
    tar = 0
    max_index = 0
    i = 0

    seed_vec = top_k_positions(dd, maxseednum)
    end = time.time()

    print("Time: {:.4f}".format(end - begin))

ff = 'ablation/' + f'{file_name}-10' + '.txt'
np.savetxt(ff, np.nonzero(seed_vec)[0], fmt='%d')

print(True)
