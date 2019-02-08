from functools import partial
from typing import Callable, List

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Task(object):
    """A task for use in an MMTL MetalModel

    Args:
        name: The name of the task
            TODO: replace this with a more fully-featured path through the network
        input_module: The input module
        head_module: The task head module
        data: A list of DataLoaders (instances and labels) to feed through the network.
            The list contains [train, dev, test].
        scorers: A list of Scorers that return metrics_dict objects.
        loss_hat_func
    """

    def __init__(
        self,
        name: str,
        data_loaders: List[DataLoader],
        input_module: nn.Module,
        head_module: nn.Module,
        scorers: List[Callable] = None,
        loss_hat_func: Callable = F.cross_entropy,
        probs_hat_func: Callable = partial(F.softmax, dim=1),
    ) -> None:
        if len(data_loaders) != 3:
            msg = "Arg data_loaders must be a list of length 3 [train, valid, test]"
            raise Exception(msg)
        self.name = name
        self.data_loaders = data_loaders
        self.input_module = input_module
        self.head_module = head_module
        self.scorers = scorers
        self.loss_hat_func = loss_hat_func
        self.probs_hat_func = probs_hat_func
