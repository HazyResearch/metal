import random
from abc import ABC, abstractmethod


class TaskScheduler(ABC):
    """Determines in what order to use batches from multiple tasks for MTL training"""

    def __init__(self, model, tasks, **kwargs):
        pass

    @abstractmethod
    def get_batches(self, tasks, split, **kwargs):
        """Returns batches from all tasks in some order until one 'epoch' is reached

        For now, an epoch is defined as one full pass through all datasets.
        This is required because of assumptions currently made in the logger and
        training loop about the number of batches that will be seen per epoch.
        """
        pass


class ProportionalScheduler(TaskScheduler):
    """Returns batches proportional to the total number of batches"""

    def get_batches(self, tasks, split, **kwargs):
        data_loaders = [iter(t.data_loaders[split]) for t in tasks]
        approx_batch_counts = [len(t.data_loaders[split]) for t in tasks]
        batch_assignments = []
        for task_idx, task in enumerate(tasks):
            batch_assignments.extend([task_idx] * approx_batch_counts[task_idx])
        random.shuffle(batch_assignments)

        for task_idx in batch_assignments:
            yield ([tasks[task_idx].name], next(data_loaders[task_idx]))
