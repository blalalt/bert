import os
import torch
import numpy as np
import itertools
from utils import *
from torch_geometric.data import Data
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import MultiLabelField, TextField, MetadataField
from allennlp.data.token_indexers import PretrainedBertIndexer

DATASETS = {'en': ['aapr', 'aapd'], 'zh': ['high_freq', 'origin']}
DATAPATH = '../data'


def _build_label_graph(instance_labels):
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
    edge_index = torch.from_numpy(np.array([source, target]))
    edge_attr = torch.from_numpy(np.array(weight).reshape(len(weight), 1))
    node_features = torch.from_numpy(np.eye(num_nodes, 100)).float()

    # g = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, adj=torch.from_numpy(A))
    g = Data(x=node_features, adj=torch.from_numpy(A).float())
    return g


def _get_data_labels(data_path):
    names = ['train', 'test', 'val']
    labels = []
    for name in names:
        path = get_abs_path(data_path, name)
        if not os.path.exists(path): continue
        for _, l in _read_json_sample(data_path, name):
            labels.append(l)
    return labels


def pre_trained_name_by_lan(data_name):
    if data_name in DATASETS['en']:
        pre_trained_model_name = 'bert-base-uncased'
    else:
        pre_trained_model_name = 'bert-base-chinese'
    return pre_trained_model_name


def _get_bert_token(data_name):
    model_name = pre_trained_name_by_lan(data_name)
    token_indexer = PretrainedBertIndexer(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return token_indexer, tokenizer


def _read_json_sample(data_path, fname):
    path = get_abs_path(data_path, fname)
    data = read_json(path)
    for sample in data:
        yield sample['text'], sample['label']


def load_dataset_reader(data_name, build_graph=False, device=-1):
    _data_path = os.path.join(DATAPATH, data_name)

    token_indexer, tokenizer = _get_bert_token(data_name)

    if build_graph:
        g = _build_label_graph(_get_data_labels(_data_path))
    else:
        g = None

    return DataLoader(_data_path, token_indexer, tokenizer, g, device)


class DataLoader(DatasetReader):
    def __init__(self, data_path, token_indexer, tokenizer, g=None, device=-1):
        super(DataLoader, self).__init__()
        self.data_path = data_path
        self.token_indexer = token_indexer
        self.tokenizer = tokenizer
        self.device = device
        self.g = g

    def _read(self, fname: str):
        for text, label in _read_json_sample(self.data_path, fname):
            text = [Token(t) for t in self.tokenizer.tokenize(text)]
            yield self.text_to_instance(text, label)

    def text_to_instance(self, text, label):
        fields = {}
        text_field = TextField(
            tokens=text,
            token_indexers={'tokens': self.token_indexer}
        )
        fields['text'] = text_field
        fields['label'] = MultiLabelField(label)
        if self.g:
            fields['graph'] = MetadataField(self.g.to(self.device))

        return Instance(fields=fields)

    def load(self):
        return [self.read(name) for name in ['train.json', 'test.json']]
