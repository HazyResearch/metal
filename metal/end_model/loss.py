import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
        size_average: if True and reduce==True, return the average loss per 
            element
        reduce: if True, reduces the losses (sum or mean, depending on 
            size_average)

    Accepts:
        input: An [n, k+1] float tensor of prediction logits (not probabilities)
        target: An [n, k+1] float tensor of target probabilities
    """
    def __init__(self, weight=None, size_average=True, reduce=True):
        super().__init__()
        assert(weight is None or isinstance(weight, torch.FloatTensor))
        self.weight = weight
        self.reduce = reduce
        self.size_average = size_average and reduce

    def forward(self, input, target):
        n, k = input.shape
        cum_losses = torch.zeros(n)
        for y in range(k):
            cls_idx = torch.full((n,), y, dtype=torch.long)
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