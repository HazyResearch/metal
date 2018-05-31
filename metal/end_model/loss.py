import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting soft (float) targets"""
    def __init__(self, class_weights=None, size_average=True, reduce=True):
        super().__init__()
        self.class_weights = class_weights
        self.size_average = size_average * reduce

    def forward(self, input, target):
        N, K_t = input.shape
        total_loss = torch.tensor(0.0)
        for i in range(K_t):
            cls_idx = torch.full((N,), i, dtype=torch.long)
            loss = F.cross_entropy(input, cls_idx, weight=self.class_weights, 
                reduce=False)
            total_loss += target[:, i].dot(loss)
        if self.size_average:
            return total_loss / N
        else:
            return total_loss
