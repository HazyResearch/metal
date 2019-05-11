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
        # The object that is passed out must be different from the object that gets
        # passed in so that cached outputs from intermediate modules aren't mutated
        X_out = {k: v for k, v in X.items()}
        X_out["data"] = self.module(X["data"])
        return X_out
