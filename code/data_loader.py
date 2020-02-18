import os
import torch
import numpy as np
import itertools
from utils import *
from graph import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField, TextField, MetadataField
from allennlp.data.token_indexers import PretrainedBertIndexer

DATASETS = {'en': ['aapr', 'aapd'], 'zh': ['high_freq', 'origin']}
DATAPATH = '../data'


def _get_data_labels(data_path):
    names = ['train.json', 'test.json', 'val.json']
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


def _get_default_token():
    token_indexer = SingleIdTokenIndexer(namespace='tokens')
    tokenizer = WordTokenizer()
    return token_indexer, tokenizer


def _read_json_sample(data_path, fname):
    path = get_abs_path(data_path, fname)
    data = read_json(path)
    for sample in data:
        yield sample['text'], sample['label']


def load_dataset_reader(data_name, embed_name, device=-1):
    _data_path = os.path.join(DATAPATH, data_name)
    if embed_name.startswith('bert'):
        token_indexer, tokenizer = _get_bert_token(data_name)
    else:
        token_indexer, tokenizer = _get_default_token()
    return DataLoader(_data_path, token_indexer, tokenizer, device)


class DataLoader(DatasetReader):
    def __init__(self, data_path, token_indexer, tokenizer, device=-1):
        super(DataLoader, self).__init__()
        self.data_path = data_path
        self.token_indexer = token_indexer
        self.tokenizer = tokenizer
        self.device = device

        self.label_to_index = self._read_labels()
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.num_labels = len(self.label_to_index)

        self.g = self._build_graph()

    def _read(self, fname: str):
        for text, label in _read_json_sample(self.data_path, fname):
            text = [t if type(t) == Token else Token(t)
                    for t in self.tokenizer.tokenize(text)]
            if len(text) < 6: continue
            yield self.text_to_instance(text, label)

    def _read_labels(self, fname='labels.json'):
        obj = read_json(os.path.join(self.data_path, fname))
        return obj

    def _build_graph(self):
        labels = []
        fname = ['train.json']
        for name in fname:
            for _, label in _read_json_sample(self.data_path, name):
                labels.append(list(map(lambda x: self.label_to_index[x], label)))
        return build_label_graph(labels)

    def label_to_array(self, labels):
        arr = np.zeros(len(self.label_to_index))
        for label in labels:
            arr[self.label_to_index[label]] = 1
        return arr

    def text_to_instance(self, text, label):
        fields = {}
        text_field = TextField(
            tokens=text,
            token_indexers={'tokens': self.token_indexer}
        )
        fields['text'] = text_field
        fields['label'] = ArrayField(self.label_to_array(label))
        if self.g:
            fields['graph'] = MetadataField(self.g.to(self.device))

        return Instance(fields=fields)

    def load(self):
        return [self.read(name) for name in ['train.json', 'test.json']]
