import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
        reduction:

    Accepts:
        input: An [n, K_t] float tensor of prediction logits (not probabilities)
        target: An [n, K_t] float tensor of target probabilities
    """
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super().__init__()
        assert(weight is None or isinstance(weight, torch.FloatTensor))
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        N, K_t = input.shape
        total_loss = torch.tensor(0.0)
        cum_losses = torch.zeros(N)
        for y in range(K_t):
            cls_idx = torch.full((N,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction='none')
            if self.weight is not None:
                y_loss = y_loss * self.weight[y]
            cum_losses += target[:, y] * y_loss
        if self.reduction == 'none':
            return cum_losses
        elif self.reduction == 'elementwise_mean':
            return cum_losses.mean()
        elif self.reduction == 'sum':
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")