import torch
import logging
import numpy as np
from typing import *
from allennlp.training.metrics import Metric
from sklearn import metrics
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class MultiLabelMetric(Metric):
    def __init__(self, threshold=0.5):
        super(MultiLabelMetric, self).__init__()
        self.threshold = threshold
        self.metrics_name = ['f1_score', 'recall', 'precision', 'hamming_loss',
                             'zero_one_loss', 'ranking_loss', 'coverage_error',
                             'average_precision']
        self._reset()
        self.average = 'micro'  # for f1, recall, precision and so on...

    def _reset(self):
        self.metrics_score = [0] * len(self.metrics_name)
        self.total_count = 0

    def _calc_f1_score(self, preds: np.ndarray, golds: np.ndarray):
        preds = (preds > self.threshold)
        return metrics.f1_score(y_true=golds, y_pred=preds,
                                average=self.average)

    def _calc_recall(self, preds: np.ndarray, golds: np.ndarray):
        preds = (preds > self.threshold)
        return metrics.recall_score(y_true=golds, y_pred=preds,
                                    average=self.average)

    def _calc_precision(self, preds: np.ndarray, golds: np.ndarray):
        preds = (preds > self.threshold)
        return metrics.precision_score(y_true=golds, y_pred=preds,
                                       average=self.average)

    def _calc_hamming_loss(self, preds: np.ndarray, golds: np.ndarray):
        preds = (preds > self.threshold)
        return metrics.hamming_loss(y_true=golds, y_pred=preds)

    def _calc_zero_one_loss(self, preds: np.ndarray, golds: np.ndarray):
        preds = (preds > self.threshold)
        return metrics.zero_one_loss(y_true=golds, y_pred=preds)

    def _calc_ranking_loss(self, preds: np.ndarray, golds: np.ndarray):
        return metrics.label_ranking_loss(y_true=golds, y_score=preds)

    def _calc_coverage_error(self, preds: np.ndarray, golds: np.ndarray):
        return metrics.coverage_error(y_true=golds, y_score=preds)

    def _calc_average_precision(self, preds: np.ndarray, golds: np.ndarray):
        return metrics.average_precision_score(y_true=golds, y_score=preds,
                                               average=self.average)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                gold_labels,
                                                                mask)
        predictions = predictions.numpy()
        gold_labels = gold_labels.numpy()

        prefix = '_calc_'
        for idx, metric in enumerate(self.metrics_name):
            _func = prefix + metric
            result = getattr(self, _func)(preds=predictions,
                                          golds=gold_labels)
            self.metrics_score[idx] += result
        self.total_count += 1

    def get_metric(self, reset: bool):
        if self.total_count == 0:
            return 0
        logger.debug(
                msg='Metric: {}\n'.format(str(self))
        )
        print('Metric: {}\n'.format(str(self)))
        res = self.metrics_score[0] / self.total_count
        if reset:
            self.reset()
        return res

    def __str__(self):
        scores = []
        for name, score in zip(self.metrics_name, self.metrics_score):
            scores.append(str(name) + ': ' + str(score / self.total_count))
        return '\t'.join(scores)

    def reset(self) -> None:
        self._reset()


class MultiLabelFScore(Metric):
    def __init__(self, threshold=0.5):
        super(MultiLabelFScore, self).__init__()
        self._correct_count = 0.
        self._total_count = 0.
        self.threshold = threshold

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                gold_labels,
                                                                mask)
        num_classes = predictions.size()[-1]
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}")
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(f"mask must have shape == predictions.size() but "
                             f"found tensor of shape: {mask.size()}")

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just
            # checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0].float()
        else:
            keep = torch.ones(batch_size).float()
        predictions = predictions.sigmoid()
        predictions = predictions.view(batch_size, -1).numpy()
        gold_labels = gold_labels.view(batch_size, -1).numpy()
        predictions = (predictions > self.threshold)
        f_score = f1_score(gold_labels, predictions, average='macro')
        self._correct_count += f_score
        self._total_count += 1
        logging.debug(
                msg='multilabel accuracy: corrcet: {}, total: {}'.format(
                    self._correct_count,
                    self._total_count))

    def get_metric(self, reset: bool):
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0


@Metric.register('multilbale-accuracy')
class AccuracyMultiLabel(Metric):
    def __init__(self, threshold=0.5):
        super(AccuracyMultiLabel, self).__init__()
        self._correct_count = 0.
        self._total_count = 0.
        self.threshold = threshold

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                gold_labels,
                                                                mask)

        # Some sanity checks.
        num_classes = predictions.size()[-1]
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}")
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(f"mask must have shape == predictions.size() but "
                             f"found tensor of shape: {mask.size()}")

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just
            # checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0].float()
        else:
            keep = torch.ones(batch_size).float()
        predictions = predictions.sigmoid()
        predictions = predictions.view(batch_size, -1).numpy()
        gold_labels = gold_labels.view(batch_size, -1).numpy()
        predictions = (predictions > self.threshold)
        keep = keep.numpy()
        # keep = np.reshape(keep, (np.shape(keep)[0], 1))
        correct = np.zeros_like(keep)
        for idx, (p, g) in enumerate(zip(predictions, gold_labels)):
            flag = int((p == g).all())
            correct[idx] = flag
            if flag:
                print(np.argwhere(p > 0), np.argwhere(g > 0))
        # correct = np.sum([]) (predictions == gold_labels)
        self._correct_count += np.sum((correct * keep))
        self._total_count += np.sum(keep)
        logging.debug(msg='multilabel accuracy: corrcet: {}, total: {}'.format(
            self._correct_count,
            self._total_count))

    def get_metric(self, reset: bool):
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
