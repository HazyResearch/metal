import torch.nn as nn


class MetalModule(nn.Module):
    """An abstract class of a module that accepts and returns a dict"""

    def __init__(self):
        super().__init__()


class MetalModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        X["data"] = self.module(X["data"])
        return X
