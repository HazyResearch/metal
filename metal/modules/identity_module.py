import torch
import torch.nn as nn

from metal.modules.base_module import InputModule

class IdentityModule(InputModule):
    """A generic nn.Module class with a method for getting the output dim."""
    def __init__(self):
        super().__init__() 
   
    def get_output_dim(self):
        raise NotImplementedError()