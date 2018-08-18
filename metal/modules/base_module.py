import torch.nn as nn


class InputModule(nn.Module):
    """A generic nn.Module class with a method for getting the output dim."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        raise NotImplementedError
