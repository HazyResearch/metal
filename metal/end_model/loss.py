import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
        size_average:
        reduce:

    Accepts:
        input: An [n, K_t] float tensor of prediction logits (not probabilities)
        target: An [n, K_t] float tensor of target probabilities
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super().__init__()
        assert(weight is None or isinstance(weight, torch.FloatTensor))
        self.weight = weight
        self.reduce = reduce
        self.size_average = size_average and reduce

    def forward(self, input, target):
        N, K_t = input.shape
        total_loss = torch.tensor(0.0)
        cum_losses = torch.zeros(N)
        for y in range(K_t):
            cls_idx = torch.full((N,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduce=False)
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y] * y_loss
        if not self.reduce:
            return cum_losses
        elif self.size_average:
            return cum_losses.mean()
        else:
            return cum_losses.sum()