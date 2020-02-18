import torch

threshold = 0.4


def load_attn(name):
    map_model = {'nor': NormalAttention, 'amp': AmplifierAttention, 'neg': NegativeAttention}
    return map_model[name]


class NormalAttention(torch.nn.Module):
    def __init__(self, text_input_dim, label_input_dim):
        super(NormalAttention, self).__init__()
        self.text_proj = torch.nn.Linear(text_input_dim, label_input_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, text_vec, labels_vec):
        # text_vec.size() == (batch_size, num_stamp, text_input_dim)
        # labels_vec.size() == (batch_size, num_labels, label_input_dim)
        text_vec = self.text_proj(text_vec)
        attn = torch.matmul(text_vec, labels_vec.t())  # (b, num_stamp, num_labels)
        alpha = self.softmax(attn)  # (b, num_stamp, num_labels)
        weighted_encoding = torch.matmul(alpha, labels_vec)  # (b, num_stamp, label_input_dim)
        return weighted_encoding


class AmplifierAttention(torch.nn.Module):
    def __init__(self, text_input_dim, label_input_dim):
        super(AmplifierAttention, self).__init__()
        self.text_proj = torch.nn.Linear(text_input_dim, label_input_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, text_vec, labels_vec):
        # text_vec.size() == (batch_size, num_stamp, text_input_dim)
        # labels_vec.size() == (batch_size, num_labels, label_input_dim)
        text_vec = self.text_proj(text_vec)
        attn = torch.matmul(text_vec, labels_vec.t())  # (b, num_stamp, num_labels)
        alpha = self.sigmoid(attn)  # (b, num_stamp, num_labels)
        alpha = torch.where(alpha < threshold, torch.zeros_like(alpha), alpha)
        alpha = self.softmax(alpha)
        weighted_encoding = torch.matmul(alpha, labels_vec)  # (b, num_stamp, label_input_dim)
        return weighted_encoding


class NegativeAttention(torch.nn.Module):
    def __init__(self, text_input_dim, label_input_dim):
        super(NegativeAttention, self).__init__()
        self.text_proj = torch.nn.Linear(text_input_dim, label_input_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, text_vec, labels_vec):
        # text_vec.size() == (batch_size, num_stamp, text_input_dim)
        # labels_vec.size() == (batch_size, num_labels, label_input_dim)
        text_vec = self.text_proj(text_vec)
        attn = torch.matmul(text_vec, labels_vec.t())  # (b, num_stamp, num_labels)
        # alpha = self.softmax(attn)  # (b, num_stamp, num_labels)
        alpha = torch.nn.functional.softsign(attn)
        # alpha = torch.div()
        weighted_encoding = torch.matmul(alpha, labels_vec)  # (b, num_stamp, label_input_dim)
        return weighted_encoding
