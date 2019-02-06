from typing import Callable, List

from torch.utils.data import DataLoader


class Task(object):
    """A task for use in an MMTL MetalModel

    Args:
        name: The name of the task
        head_name: The name of the task head to use
            TODO: replace this with a more fully-featured path through the network
        data: A list of DataLoaders (instances and labels) to feed through the network.
            The list contains [train, dev, test].
        scorers: A list of Scorers that return metrics_dict objects.
    """

    def __init__(
        self,
        name: str,
        head_name: str,
        data_loaders: List[DataLoader],
        scorers: List[Callable] = None,
    ) -> None:
        self.name = name
        self.data_loaders = data_loaders
        self.head_name = head_name
        self.scorers = scorers
