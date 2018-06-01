import torch
import torch.nn as nn

from metal.modules.base_module import InputModule

class IdentityModule(InputModule):
    """A default identity input layer that simply passes the input through."""
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
   
    def reset_parameters(self):
        pass

    def get_output_dim(self):
        return self.output_dim

    def forward(self, x):
        return x