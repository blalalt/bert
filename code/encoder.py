import torch
import torch
from typing import Dict, Union
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary

GLOVE_EMBEDDING_DIM = 300

MAP = {'glove': '../data/glove.6B.300d.txt',
       'bert_en': 'bert-base-uncased',
       'bert_zh': 'bert-base-chinese'}


def load_encoder(name, vocab):
    embedding_file = MAP[name]
    if name == 'glove':
        return GloveEncoder(embedding_file, vocab)
    elif name.startswith('bert'):
        return BertEncoder(embedding_file, vocab)


class GloveEncoder(torch.nn.Module):
    def __init__(self, embedding_file, vocab):
        super(GloveEncoder, self).__init__()
        _out_dim = 100
        self.token_embedding = Embedding.from_params(
            vocab=vocab,
            params=Params({'pretrained_file': embedding_file,
                           'embedding_dim': GLOVE_EMBEDDING_DIM})
        )
        self.embed = BasicTextFieldEmbedder({"tokens": self.token_embedding})
        self.encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(
            batch_first=True,
            bidirectional=True,
            input_size=GLOVE_EMBEDDING_DIM,
            hidden_size=_out_dim
        ))
        self._dropout = torch.nn.Dropout(0.5)

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, tokens):
        e = self.embed(tokens)
        en = self.encoder(e, None)
        en = self._dropout(en)
        return en


class BertEncoder(torch.nn.Module):
    def __init__(self,
                 embedding_file,
                 vocab=None,
                 dropout: float = 0.4,
                 trainable: bool = True):
        super().__init__()

        if isinstance(embedding_file, str):
            self.bert_model = PretrainedBertModel.load(embedding_file)
        else:
            self.bert_model = embedding_file

        self._dropout = torch.nn.Dropout(p=dropout)
        self._index = "tokens"
        self._train_layers = 3
        if trainable:
            self.fine_tune()

    def fine_tune(self):
        _until = -self._train_layers
        _list_models = list(self.bert_model.children())
        assert self._train_layers <= len(_list_models)
        for c in _list_models[_until:]:
            for p in c.parameters():
                p.requires_grad = True

        for c in _list_models[:_until]:
            for p in c.parameters():
                p.requires_grad = False

        # ALL parameters fine-tune
        # for param in self.bert_model.parameters():
        #     param.requires_grad = True

    def get_output_dim(self):
        return self.bert_model.config.hidden_size

    def forward(self, tokens: Dict[str, torch.LongTensor]):
        # print(tokens.keys())
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        encoded, pooled = self.bert_model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=input_mask)
        encoded = self._dropout(encoded[-1])
        return encoded
