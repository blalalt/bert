from bert import Bert
from label_embed import LabelAttention, LabelEmbedding
import torch
from allennlp.modules.seq2vec_encoders import CnnEncoder


class LgBert(torch.nn.Module):
    def __init__(self, bert_model, g):
        super(LgBert, self).__init__()

        _out_dim = 100
        self.text_encoder = Bert(bert_model, 0.5)
        self.label_encoder = LabelEmbedding(g, _out_dim)
        self.tl_attn = LabelAttention(self.text_encoder.get_output_dim(),
                                      self.label_encoder.get_output_dim())
        self.encoder = CnnEncoder(self.tl_attn.get_output_dim(), num_filters=100)

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, tokens, g):
        text_vec = self.text_encoder(tokens)
        label_vec = self.label_encoder(g)
        att_vec = self.tl_attn(text_vec, label_vec)
        vec = self.encoder(att_vec)
        return vec
