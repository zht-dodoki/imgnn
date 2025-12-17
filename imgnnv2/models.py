import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, dropout=0.1):
        super(GNN, self).__init__()
        
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features) # LayerNorm
        
        self.gc2 = GraphConvolution(hidden_features, hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        
        self.gc3 = GraphConvolution(hidden_features, hidden_features)
        self.ln3 = nn.LayerNorm(hidden_features)
    
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(hidden_features * 3, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, adj):
        x1 = self.gc1(x, adj)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        
        x2 = self.gc2(x1, adj)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        
        x3 = self.gc3(x2, adj)
        x3 = self.ln3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.dropout, training=self.training)
        
        x_cat = torch.cat([x1, x2, x3], dim=1)
        out = self.fc(x_cat)
        return out.squeeze() # [N]

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
