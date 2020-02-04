import torch
import warnings
from optparse import OptionParser
from data_loader import load_dataset_reader, pre_trained_name_by_lan
from metircs import MultiLabelMetric
from utils import *
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import SlantedTriangular

warnings.filterwarnings('ignore')

log_path = '../log'
learning_rate = 0.004
epoch = 100
batch_size = 8
cuda_device = 0 if torch.cuda.is_available() else -1


def load_model(model_name, data_name, build_graph):
    from bert import BertForClassifier
    from lgbert import LgBert
    _models = {'bert': BertForClassifier, 'lgbert': LgBert}
    bert_model_name = pre_trained_name_by_lan(data_name)
    return _models[model_name](bert_model_name, build_graph)


class Classifier(Model):
    def __init__(self, vocab, model):
        super(Classifier, self).__init__(vocab)
        self.model = model
        in_features = model.get_output_dim()
        out_features = vocab.get_vocab_size(namespace="labels")

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


def train(model_name, data_name, build_graph):
    logger = get_train_logger(log_path, model_name, data_name)
    dataset_reader = load_dataset_reader(data_name, build_graph, cuda_device)
    train_set, val_set = dataset_reader.load()

    vocab = Vocabulary.from_instances(train_set + val_set)
    iterator = BucketIterator(batch_size=batch_size,
                              sorting_keys=[('text', 'num_tokens')])
    iterator.index_with(vocab=vocab)

    model = load_model(model_name, data_name, build_graph)
    clf = Classifier(vocab, model)
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
        num_epochs=epoch,
        cuda_device=cuda_device,
    )
    trainer.train()


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model_name')
    parser.add_option('-d', '--data', dest='data_name')
    parser.add_option("-g", action="store_true", dest="graph", default=False)

    (options, args) = parser.parse_args()
    model_name = options.model_name
    data_name = options.data_name
    build_graph = options.graph
    train(model_name, data_name, build_graph)


if __name__ == '__main__':
    main()
