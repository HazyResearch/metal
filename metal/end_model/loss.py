import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets"""
    def __init__(self, class_weights=None, size_average=True, reduce=True):
        super().__init__()
        self.class_weights = class_weights
        self.reduce = reduce
        self.size_average = size_average and reduce

    def forward(self, input, target):
        N, K_t = input.shape
        total_loss = torch.tensor(0.0)
        losses = torch.zeros(N)
        for i in range(K_t):
            cls_idx = torch.full((N,), i, dtype=torch.long)
            loss = F.cross_entropy(input, cls_idx, weight=self.class_weights, 
                reduce=False)
            losses += target[:, i] * loss
        total_loss = losses.sum() if self.reduce else losses
        total_loss = total_loss / N if self.size_average else total_loss
        return total_loss