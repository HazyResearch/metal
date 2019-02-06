from typing import Callable, List, Optional

from torch.utils.data import DataLoader


class Task(object):
    """A task for use in an MMTL MetalModel

    Args:
        name: (str) The name of the task
        head_name: (str) The name of the task head to use
            TODO: replace this with a more fully-featured path through the network
        data: (DataLoader) The data (instances and labels) to feed through the network
        scorers: (list) A list of Scorers that return metrics_dict objects.
    """

    def __init__(
        self,
        name: str,
        head_name: str,
        data: DataLoader,
        scorers: List[Callable] = None,
    ) -> None:
        self.name = name
        self.data = data
        self.head_name = head_name
        self.scorers = scorers
