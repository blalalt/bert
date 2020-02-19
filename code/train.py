import torch
import warnings
from optparse import OptionParser

from data_loader import load_dataset_reader
from metircs import MultiLabelMetric
from utils import *
from model import load_model
from attention import load_attn
from encoder import load_encoder

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import SlantedTriangular

warnings.filterwarnings('ignore')

log_path = '../log'
checkpoints_path = '../checkpoints'
learning_rate = 2e-5
epoch = 100
batch_size = 8
cuda_device = 0 if torch.cuda.is_available() else -1


class Classifier(Model):
    def __init__(self, vocab, model, out_features):
        super(Classifier, self).__init__(vocab)
        self.model = model
        in_features = model.get_output_dim()

        self._classification_layer = torch.nn.Linear(in_features, out_features)
        self._metric = MultiLabelMetric()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, text, label, graph=None):
        vec = self.model(text, graph)

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


def train(model_name, embed_name, attn_name, data_name):
    name = model_name + '_' + embed_name + '_' + attn_name
    logger = get_train_logger(log_path, name, data_name)
    checkpoints = get_train_checkpoints(checkpoints_path, name, data_name)

    dataset_reader = load_dataset_reader(data_name, embed_name, cuda_device)
    train_set, val_set = dataset_reader.load()

    vocab = Vocabulary.from_instances(train_set + val_set)
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[('text', 'num_tokens')])
    iterator.index_with(vocab=vocab)

    encoder = load_encoder(embed_name, vocab)
    attn = load_attn(attn_name)

    clf = load_model(model_name)(vocab,
                                 encoder=encoder,
                                 attention=attn,
                                 g=dataset_reader.g,
                                 out_dim=dataset_reader.num_labels)
    if cuda_device > -1:
        clf = clf.cuda(cuda_device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=clf,
        optimizer=optimizer,
        iterator=iterator,
        validation_metric='+f-score',
        train_dataset=train_set,
        validation_dataset=val_set,
        patience=10,
        grad_clipping=10,
        num_epochs=epoch,
        cuda_device=cuda_device,
        num_serialized_models_to_keep=1,
        serialization_dir=checkpoints,
    )
    trainer.train()


def main():
    parser = OptionParser()
    parser.add_option('-a', '--attn', dest='attn_name', help="nor, neg, amp", default='nor')
    parser.add_option('-d', '--data', dest='data_name', help='aapd, rcv', default='aapd')
    parser.add_option('-e', '--embed', dest='embed_name', help='glove, bert_en, bert_zh', default='glove')
    # parser.add_option("-g", action="store_true", dest="graph", default=True)
    parser.add_option('-m', '--model', dest='model_name', help="lg, bert, lstm", default='lg')

    (options, args) = parser.parse_args()
    embed_name = options.embed_name
    data_name = options.data_name
    attn_name = options.attn_name
    model_name = options.model_name

    if model_name == 'bert': embed_name = 'bert_en'
    if model_name == 'lstm': embed_name = 'glove'

    train(model_name, embed_name, attn_name, data_name)


if __name__ == '__main__':
    main()
