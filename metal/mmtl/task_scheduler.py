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
    """Returns batches proportional to the fraction of the total number of batches"""

    def get_batches(self, tasks, split, **kwargs):
        data_loaders = [iter(t.data_loaders[split]) for t in tasks]
        batch_counts = [len(t.data_loaders[split]) for t in tasks]
        batch_assignments = []
        for task_idx, task in enumerate(tasks):
            batch_assignments.extend([task_idx] * batch_counts[task_idx])
        random.shuffle(batch_assignments)

        for task_idx in batch_assignments:
            yield ([tasks[task_idx].name], next(data_loaders[task_idx]))


class StagedScheduler(TaskScheduler):
    """Returns batches from an increasing number of tasks over the epoch

    Start by training only on the task with the largest number of batches.
    Gradually increase the tasks being trained on so that the final batches are
    sampling equally from all tasks.

    For example, if X is the largest task, start by training only on batches of X, then
    on X and Y, then on X, Y, and Z:

    XXXXXXXXXXXXXXX
         YYYYYYYYYY
                ZZZ
    """

    def get_batches(self, tasks, split, **kwargs):
        data_loaders = [iter(t.data_loaders[split]) for t in tasks]
        batch_counts = [len(t.data_loaders[split]) for t in tasks]
        max_count = max(batch_counts)
        for i in reversed(range(max_count)):
            for task, count, loader in zip(tasks, batch_counts, data_loaders):
                if count > i:
                    yield ([task.name], next(loader))
