from ..registry import LOSSES
from .base import BaseWeightedLoss
import paddle.nn as nn


@LOSSES.register()
class LabelSmoothingCrossEntropy(BaseWeightedLoss):
    def __init__(self, reduction='mean', epsilon: float = 0.1, ignore_index=-1):
        super().__init__()
        self.class_aixs = -1  # NCWH
        self.epsilon = epsilon
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(axis=self.class_aixs)
        self.nll_loss = nn.NLLLoss(reduction="none", ignore_index=ignore_index)

    def _forward(self, score, labels, **kwargs):
        n_class = score.shape[self.class_aixs]
        log_preds = self.log_softmax(score)

        target_loss = self.nll_loss(log_preds, labels)
        target_loss = target_loss * (1 - self.epsilon - self.epsilon / (n_class - 1))

        untarget_loss = -log_preds.sum(axis=self.class_aixs) * (self.epsilon / (n_class - 1))
        loss = target_loss + untarget_loss
        return self.reduce_loss(loss, self.reduction)

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
