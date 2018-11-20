import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets

    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elmentwise losses
            'none': return an unreduced list of elementwise losses
            'elementwise_mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses

    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="elementwise_mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        self.register_buffer("loss_weights", weight)
        if self.loss_weights is not None:  # pylint: disable=E0203
            self.loss_weights = torch.FloatTensor(weight)
        self.reduction = reduction

    def forward(self, input, target):
        n, k = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(n)
        for y in range(k):
            cls_idx = input.new_full((n,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.loss_weights is not None:
                y_loss = y_loss * self.loss_weights[y]
            cum_losses += target[:, y].float() * y_loss
        if self.reduction == "none":
            return cum_losses
        elif self.reduction == "elementwise_mean":
            return cum_losses.mean()
        elif self.reduction == "sum":
            return cum_losses.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")
