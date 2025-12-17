import torch
import numpy as np
from torch.utils.data import Dataset


class RISDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List of dicts, keys: 'adj', 'features', 'label'
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        adj_scipy = sample['adj']
        features = torch.from_numpy(sample['features']).float()
        label = torch.from_numpy(sample['label']).float()
        return features, adj_scipy, label


class myGraph:
    def __init__(self, num_nodes, edges):
        self.num_nodes = num_nodes
        self.num_edges = len(edges)
        self.edges = edges
        self.impact_list = np.ones((self.num_nodes,))

        self.adj_matrix = None  # float
        self.prob_matrix = None  # float
        self.prob_matrix_copy = None  # float
        self.inverse_pairs = None 
        self.node_impact = None
        self.first_step_impact = None

        self.first_rhop = None
        self.rhop = None
        self.rhop_exp = None

        self.adj_list = None


def __str__(self):
    return "num_nodes: %d, edges: %s, adj_pairs: %s, prob_mat: %s" % (
        self.num_nodes, self.edges, self.adj_matrix, self.prob_matrix)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, node_impact, adj_list):
        super(CustomDataset, self).__init__()
        self.node_impact = node_impact
        self.adj_list = adj_list

    def __len__(self):
        return len(self.node_impact)

    def __getitem__(self, idx):
        return self.node_impact[idx], self.adj_list[idx]