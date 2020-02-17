import torch
from graph import LabelEmbedding
from metircs import MultiLabelMetric
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.models.model import Model


def load_model(model_name):
    MAP = {'bert': Bert, 'lstm': LSTM, 'lg': LabelEmbedModel}
    return MAP[model_name]


class LabelEmbedModel(Model):
    def __init__(self, vocab, encoder, attention, g, out_dim):
        super(LabelEmbedModel, self).__init__(vocab)

        _label_out_dim = 100
        self.text_encoder = encoder
        self.label_encoder = LabelEmbedding(g.num_features, _label_out_dim)
        self.tl_attn = attention(self.text_encoder.get_output_dim(),
                                 self.label_encoder.get_output_dim())
        self.encoder = CnnEncoder(self.text_encoder.get_output_dim() +
                                  self.label_encoder.get_output_dim(),
                                  num_filters=100)
        # self.encoder = CnnEncoder(self.text_encoder.get_output_dim(),
        #                           num_filters=100)

        self._classification_layer = torch.nn.Linear(self.encoder.get_output_dim(), out_dim)
        self._metric = MultiLabelMetric()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, text, label, graph):
        text_vec = self.text_encoder(text)
        label_vec = self.label_encoder(graph[0])
        att_vec = self.tl_attn(text_vec, label_vec)
        vec = torch.cat([text_vec, att_vec], dim=-1)

        vec = self.encoder(vec, None)

        logits = self._classification_layer(vec)
        probs = torch.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            self._metric(predictions=logits, gold_labels=label)
            output_dict['loss'] = self._loss(input=logits, target=label.float())

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = {'f-score': self._metric.get_metric(reset)}
        return metrics


class LSTM(Model):
    def __init__(self, vocab, encoder, attention, g, out_dim):
        super(LSTM, self).__init__(vocab)
        self.text_encoder = encoder
        self._classification_layer = torch.nn.Linear(
            self.text_encoder.get_output_dim(), out_dim)
        self._metric = MultiLabelMetric()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, text, label, graph):
        text_vec = self.text_encoder(text)
        vec = text_vec[:, -1]  # last hidden state

        logits = self._classification_layer(vec)
        probs = torch.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            self._metric(predictions=logits, gold_labels=label)
            output_dict['loss'] = self._loss(input=logits, target=label.float())

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = {'f-score': self._metric.get_metric(reset)}
        return metrics


class Bert(Model):
    def __init__(self, vocab, encoder, attention, g, out_dim):
        super(Bert, self).__init__(vocab)
        self.encoder = encoder
        self.dense = torch.nn.Linear(self.encoder.get_output_dim(),
                                     self.encoder.get_output_dim())
        self.activation = torch.nn.Tanh()
        self._classification_layer = torch.nn.Linear(self.encoder.get_output_dim(),
                                                     out_dim)
        self._metric = MultiLabelMetric()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, text, label, graph):
        hidden_states = self.encoder(text)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        logits = self._classification_layer(pooled_output)
        probs = torch.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            self._metric(predictions=logits, gold_labels=label)
            output_dict['loss'] = self._loss(input=logits, target=label.float())

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = {'f-score': self._metric.get_metric(reset)}
        return metrics
