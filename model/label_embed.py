import torch
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros


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

    def forward(self, data):
        x, adj = data.x, data.adj
        x = self.conv1(x, adj)
        x = x.squeeze()
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        x = x.squeeze()
        return x


class LabelAttention(torch.nn.Module):
    def __init__(self, text_input_dim, label_input_dim):
        super(LabelAttention, self).__init__()
        self.text_proj = torch.nn.Linear(text_input_dim, label_input_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, text_vec, labels_vec):
        # text_vec.size() == (batch_size, num_stamp, input_dim)
        # labels_vec.size() == (batch_size, num_labels, input_dim)
        text_vec = self.text_proj(text_vec)
        attn = torch.matmul(text_vec, labels_vec.t())  # (b, num_stamp, num_labels)
        alpha = self.softmax(attn)  # (b, num_stamp, num_labels)
        weighted_encoding = torch.matmul(alpha, labels_vec)  # (b, num_stamp, input_dim)
        return weighted_encoding
