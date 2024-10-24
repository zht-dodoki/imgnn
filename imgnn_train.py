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
import random

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


file_name = 'netscience'
file_txt_path = f'my_data/{file_name}.txt'

with open(f'my_pkl/{file_name}_data' + '.pkl', 'rb') as f:
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
forward_model = GNN(in_features=in_features, hidden_features1=hidden_features1, hidden_features2=hidden_features2, out_features=out_features)

optimizer = Adam(forward_model.parameters(), lr=1e-4)

forward_model = forward_model.to(device)
forward_model.train()


def loss_all(y, y_hat):
    forward_loss = F.mse_loss(y_hat.relu_(), y, reduction='sum')
    return forward_loss


for epoch in range(0):
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
torch.save(forward_model, '0.pth')

print(True)
