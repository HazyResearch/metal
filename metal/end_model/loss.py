import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, Y_tp, Y_t):
        N, K_t = Y_tp.shape
        loss = torch.tensor(0.0)
        for y in range(K_t):
            Y_fixed = y * torch.ones(N).long()
            _losses = Y_t[:,y] * F.cross_entropy(Y_tp, Y_fixed, 
                weight=self.class_weights, reduce=False)
            loss += torch.sum(_losses)
        return loss