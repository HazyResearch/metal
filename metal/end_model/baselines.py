import torch.nn as nn

from metal.end_model import EndModel
from metal.utils import recursive_merge_dicts
from metal.end_model.em_defaults import em_model_defaults, em_train_defaults

class SoftmaxRegression(EndModel):
    """A softmax regression classifier for a multi-class single-task problem"""
    def __init__(self, input_dim, output_dim, **kwargs):
        overrides = {
            'batchnorm': False,
            'layer_output_dims': [input_dim],
            'head_output_dims': [output_dim],
        }
        kwargs = recursive_merge_dicts(kwargs, overrides, misses='insert', 
            verbose=False)
        label_map = [range(output_dim)]
        super().__init__(label_map, **kwargs)

class LogisticRegression(SoftmaxRegression):
    """A logistic regression classifier for a binary single-task problem"""
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, 1, **kwargs)
