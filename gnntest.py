import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features, dropout=0.5, device='cpu'):
        super(GNN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features1, device=device)
        self.gc2 = GraphConvolution(hidden_features1, hidden_features2, device=device)
        self.gc3 = GraphConvolution(hidden_features2, out_features, device=device)
        self.fc = nn.Linear(out_features, 1)
        self.dropout = dropout
        self.device = device

    def forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        x = x.squeeze()
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.to(self.device)
        adj = adj.to(self.device)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output