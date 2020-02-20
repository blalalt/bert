import os
import torch
import numpy as np
import itertools
from utils import *
from torch_geometric.data import Data
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros


def build_label_graph(instance_labels):
    def pmi(p_xy, p_x, p_y):
        return np.log((1.0 * p_xy) / (p_x * p_y))

    def unfold(l_list):
        for l in l_list:
            for item in l:
                yield item

    num_nodes = len(set(unfold(l_list=instance_labels)))
    A = np.zeros(shape=(num_nodes, num_nodes))
    for sample_labels in instance_labels:
        if len(sample_labels) == 1:
            A[sample_labels[0]][sample_labels[0]] += 1
        else:
            for pair in itertools.permutations(sample_labels, 2):
                _node_A, _node_B = pair
                A[_node_A][_node_B] += 1
                A[_node_B][_node_A] += 1
                A[_node_A][_node_A] += 1
                A[_node_B][_node_B] += 1
    # print(A.shape, A)
    # pmi
    source = []
    target = []
    weight = []
    for x in range(num_nodes):
        for y in range(num_nodes):
            if x == y:
                A[x][x] = 1
                continue

            if A[x][y] != 0:
                source.append(x)
                target.append(y)
                freq_x = A[x][x]
                freq_y = A[y][y]
                freq_x_y = A[x][y]
                pmi_score = pmi(freq_x_y, freq_x, freq_y)
                A[x][y] = pmi_score
                weight.append(pmi_score)
            else:
                continue
    assert len(source) == len(target) == len(weight)
    # edge_index = torch.from_numpy(np.array([source, target]))
    # edge_attr = torch.from_numpy(np.array(weight).reshape(len(weight), 1))
    node_features = torch.from_numpy(np.eye(num_nodes, num_nodes)).float()

    # g = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, adj=torch.from_numpy(A))
    g = Data(x=node_features, adj=torch.from_numpy(A).float())
    print(g)
    return g


class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)
        out = out + self.bias
        return out


class LabelEmbedding(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LabelEmbedding, self).__init__()
        _hidden_dim = 100
        self.conv1 = GCNLayer(in_dim, _hidden_dim)
        self.conv2 = GCNLayer(_hidden_dim, out_dim)
        self.out_dim = out_dim

    def get_output_dim(self):
        return self.out_dim

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.conv1(x, adj)
        x = x.squeeze()
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        x = x.squeeze()
        return x
