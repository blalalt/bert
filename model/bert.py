import torch
from typing import Dict, Union
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel


class Bert(torch.nn.Module):
    def __init__(self,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 trainable: bool = True):
        super().__init__()

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        self._dropout = torch.nn.Dropout(p=dropout)
        self._index = "tokens"
        self._train_layers = 5
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
        #     param.requires_grad = trainable

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

        pooled = self._dropout(pooled)

        return encoded, pooled


class BertForClassifier(torch.nn.Module):
    def __init__(self, bert_model, build_graph=False):
        super(BertForClassifier, self).__init__()
        self.bert_model = Bert(bert_model, 0.5, True)
        self.build_graph = build_graph

    def get_output_dim(self):
        return self.bert_model.get_output_dim()

    def forward(self, tokens, graph):
        _, vec = self.bert_model(tokens)
        return vec
