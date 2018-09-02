import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
        reduction: how to combine the elmentwise losses
            'none': return an unreduced list of elementwise losses
            'elementwise_mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses

    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(
        self, weight=None, reduction="elementwise_mean", use_cuda=False
    ):
        super().__init__()
        assert weight is None or isinstance(weight, torch.FloatTensor)
        self.weight = weight
        self.reduction = reduction
        self.use_cuda = use_cuda

    def forward(self, input, target):
        n, k = input.shape
        cum_losses = torch.zeros(n)
        if self.use_cuda:
            cum_losses = cum_losses.cuda()
        for y in range(k):
            cls_idx = torch.full((n,), y, dtype=torch.long)
            if self.use_cuda:
                cls_idx = cls_idx.cuda()
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y] * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "elementwise_mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")
