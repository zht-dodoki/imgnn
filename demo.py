import argparse
import random
import predata
import predata_lt
import predata_sis
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle
import scipy.sparse as sp
import graphdata
from torch.utils.data import DataLoader
from torch.optim import Adam
from gnntest import GNN
from utils import normalize_adj, diffusion_evaluation

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1

parser = argparse.ArgumentParser(description="IMGNN")

train = ['cora_ml', 'CA-GrQc', 'netscience', 'deezer']
parser.add_argument("-trd", "--train_datasets", default="netscience", type=str,
                    help="one of: {}".format(", ".join(sorted(train))))

target = ['cora_ml', 'CA-GrQc', 'netscience', 'deezer', 'writer', 'occupation', 'youtube']
parser.add_argument("-tad", "--target_datasets", default="occupation", type=str,
                    help="one of: {}".format(", ".join(sorted(target))))

diffusion = ['IC', 'LT', 'SIS']
parser.add_argument("-dm", "--diffusion_model", default="IC", type=str,
                    help="one of: {}".format(", ".join(sorted(diffusion))))

K = [0.01, 0.05, 0.1]
parser.add_argument("-K", "--K", default=0.01, type=int,
                    help="one of: {}".format(", ".join(str(sorted(K)))))

b = [1, 5, 10]
parser.add_argument("-b", "--b", default=1, type=int,
                    help="one of: {}".format(", ".join(str(sorted(b)))))

args = parser.parse_args(args=[])

file_name = args.train_datasets

if args.diffusion_model == 'IC':
    predata.icdata(file_name)
elif args.diffusion_model == 'LT':
    predata_lt.ltdata(file_name)
elif args.diffusion_model == 'SIS':
    predata_sis.sisdata(file_name)

with open(f'pkldata/{file_name}' + f'_{args.diffusion_model}' + '.pkl', 'rb') as f:
    graph = pickle.load(f)

adj_list = []
for adjl in graph.adj_list:
    adjl = adjl + adjl.T.multiply(adjl.T > adjl) - adjl.multiply(adjl.T > adjl)
    adjl = normalize_adj(adjl + sp.eye(adjl.shape[0]))
    adjl = torch.Tensor(adjl.toarray()).to_sparse()
    adj_list.append(adjl)
graph.adj_list = adj_list

graph.inverse_pairs[:, :, 1] = (graph.inverse_pairs[:, :, 1] - torch.min(graph.inverse_pairs[:, :, 1])) / (
        torch.max(graph.inverse_pairs[:, :, 1]) - torch.min(graph.inverse_pairs[:, :, 1]))

graph.node_impact = graph.inverse_pairs.clone()

dataset = graphdata.CustomDataset(graph.node_impact, graph.adj_list)
train_size = int(len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])


def collate_fn(batch):
    node_impact, adj_list = zip(*batch)
    return torch.stack(node_impact), adj_list


train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, drop_last=False, collate_fn=collate_fn)
# test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

in_features = 1
hidden_features1 = 16
hidden_features2 = 8
out_features = 4
forward_model = GNN(in_features=in_features, hidden_features1=hidden_features1, hidden_features2=hidden_features2,
                    out_features=out_features, device='cpu')

optimizer = Adam(forward_model.parameters(), lr=1e-4)

forward_model = forward_model.to(device)
forward_model.train()


def loss_all(y, y_hat):
    forward_loss = F.mse_loss(y_hat.relu_(), y, reduction='sum')
    return forward_loss


for epoch in range(500):
    begin = time.time()
    total_overall = 0
    forward_loss = 0

    for batch_idx, data_pair in enumerate(train_loader):
        adj_list = [adj.to(device) for adj in data_pair[1]]
        x = data_pair[0][:, :, 0].float().to(device)
        y = data_pair[0][:, :, 1].float().to(device)
        optimizer.zero_grad()
        loss = 0
        for i, x_i in enumerate(x):
            y_i = y[i]
            y_hat = forward_model(x_i.unsqueeze(-1), adj_list[i])
            total = loss_all(y_i, y_hat.squeeze(-1))
            loss += total

        total_overall += loss.item()
        loss = loss / x.size(0)
        loss.backward()
        optimizer.step()
        for p in forward_model.parameters():
            p.data.clamp_(min=0)
    end = time.time()
    # print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Epoch: {}".format(epoch + 1),
          "\tTotal: {:.4f}".format(total_overall / len(train_set)),
          "\tTime: {:.4f}".format(end - begin)
          )

file_pth_path = f'{file_name}{args.diffusion_model}.pth'

torch.save(forward_model, file_pth_path)
print('forward_model saved')

print("node seed influence prediction")


def generate_unique_random_int(low, high, exclude, exclude2):
    exclude_set = set(exclude)
    exclude_set2 = set(exclude2)

    while True:
        random_int = random.randint(low, high)
        if random_int not in exclude_set and random_int not in exclude_set2:
            return random_int


target_name = args.target_datasets
if args.diffusion_model == 'IC':
    predata.icdata(target_name, False)
elif args.diffusion_model == 'LT':
    predata_lt.ltdata(target_name, False)
elif args.diffusion_model == 'SIS':
    predata_sis.sisdata(target_name, False)

with open(f'pkltarget/{target_name}' + f'_{args.diffusion_model}' + '.pkl', 'rb') as f:
    graph = pickle.load(f)

adj_list = []

for param in forward_model.parameters():
    param.requires_grad = False
forward_model.eval()

seed_list = []
N = graph.prob_matrix.shape[0]
seed_vec = np.zeros((N,))
seed_list_impact = np.zeros((N,))
res = 0
maxseednum = int(N * args.K)
impact = 0


def top_k_positions(seed_vec, dd, hseed, k):
    indices_to_ignore = np.where(hseed == 1)[0]
    dd[indices_to_ignore] = float('-inf')
    sorted_indices = torch.argsort(dd, descending=True)
    top_k_indices = sorted_indices[:k]
    seed_vec[top_k_indices] = 1
    return seed_vec


for j in range(args.b):
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

    res = forward_model(seed_list_impact.unsqueeze(-1), adj)

    dd = res.squeeze(-1)

    max_value = torch.max(dd)
    max_indices = (dd == max_value).nonzero()
    max_indices_len = len(max_indices)
    dd[max_indices] = 0
    tar = 0
    max_index = 0
    i = 0

    seed_vec = top_k_positions(seed_vec, dd, seed_list_impact, int(maxseednum / args.b))
    end = time.time()
    if args.b != 1:
        if args.diffusion_model == 'IC':
            seed_list_impact = predata.run_mc_repeats_(graph, seed_vec, 10, 10)
        elif args.diffusion_model == 'LT':
            seed_list_impact = predata_lt.run_mc_repeats_(graph, seed_vec, 10, 10)
        elif args.diffusion_model == 'SIS':
            seed_list_impact = predata_sis.run_mc_repeats_(graph, seed_vec, 10, 10)

        seed_list_impact = seed_list_impact[:, -1]
        seed_list_impact = np.where(seed_list_impact >= 0.3, 1, 0)
        impact = np.sum(seed_list_impact)
        end = time.time()

    print("Time: {:.4f}".format(end - begin))
print("seed node set for influence maximization finished")

adj = graph.adj_matrix
prb = graph.prob_matrix
test = np.nonzero(seed_vec)[0]

influence = diffusion_evaluation(adj, test, prb, args.diffusion_model)
print('Diffusion count: {}'.format(influence))
