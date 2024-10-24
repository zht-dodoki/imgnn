import torch
import numpy as np
import time
import pickle
import scipy.sparse as sp
from predata import run_mc_repeats_
import random

import matplotlib.pyplot as plt
import networkx as nx


def diffusion(graph, seed_list_impact):
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edges)
    # 为节点生成布局
    pos = nx.spring_layout(G, seed=42)  # 固定种子值，确保布局可重复
    # 设置图的背景颜色
    plt.figure(figsize=(10, 10))
    plt.gca().set_facecolor('whitesmoke')
    # 获取激活和未激活的节点索引
    activated_nodes = np.where(seed_list_impact == 1)[0]
    non_activated_nodes = np.where(seed_list_impact == 0)[0]
    # 绘制未激活的节点（灰色、较小）
    nx.draw_networkx_nodes(G, pos, nodelist=non_activated_nodes,
                           node_color='lightgray', node_size=5,
                           node_shape='o', label='non-activated')
    # 绘制激活的节点（红色、较小）
    nx.draw_networkx_nodes(G, pos, nodelist=activated_nodes,
                           node_color='red', node_size=15,
                           node_shape='s', label='activated')
    # 高亮激活节点间的边
    activated_edges = [(u, v) for u, v in G.edges() if seed_list_impact[u] == 1 and seed_list_impact[v] == 1]
    nx.draw_networkx_edges(G, pos, edgelist=activated_edges, edge_color='red', width=2.5, alpha=0.8)
    # 绘制其他普通的边
    non_activated_edges = [(u, v) for u, v in G.edges() if (u, v) not in activated_edges]
    nx.draw_networkx_edges(G, pos, edgelist=non_activated_edges, edge_color='black', alpha=0.3, width=1.0)
    # 美化图例
    plt.legend(loc='best', fontsize=12)
    # 去除坐标轴
    plt.axis('off')
    # 添加标题，设置字体大小和颜色
    # plt.title('IC 扩散结果可视化', fontsize=18, color='darkblue')
    # 展示图形
    plt.show()


print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1


def generate_unique_random_int(low, high, exclude, exclude2):
    # 排除数组转换为集合以提高查找速度
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
    # 获取dd中的top-k值及其位置
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

    # 将mask中的行和列的值设为0
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

    # seed_list_impact = run_mc_repeats_(graph, seed_vec, 30, 50, re=False)
    # seed_list_impact = seed_list_impact[:, -1]
    # seed_list_impact = np.where(seed_list_impact >= 0.1, 1, 0)
    # # diffusion(graph, seed_list_impact)
    #
    # impact = np.sum(seed_list_impact)
    #
    # end = time.time()
    # print(str(impact))
    # print(np.nonzero(seed_vec))
    # print(j)
    print("Time: {:.4f}".format(end - begin))

ff = 'ablation/' + f'{file_name}-10' + '.txt'
np.savetxt(ff, np.nonzero(seed_vec)[0], fmt='%d')

print(True)
