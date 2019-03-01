import random
from abc import ABC, abstractmethod


class TaskScheduler(ABC):
    """Determines in what order to use batches from multiple tasks for MTL training"""

    def __init__(self, model, tasks, split, **kwargs):
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
            yield (next(data_loaders[task_idx]), [tasks[task_idx].name])


class StagedScheduler(TaskScheduler):
    """Returns batches from an increasing number of tasks over the epoch

    Start by training only on the task with the largest number of batches.
    Gradually increase the tasks being trained on so that the final batches are
    sampling equally from all tasks.

    For example, if X is the largest task, start by training only on batches of X, then
    on X and Y, then on X, Y, and Z. This resets every epoch:

    XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX
         YYYYYYYYYY|     YYYYYYYYYY
                ZZZ|            ZZZ
    """

    def get_batches(self, tasks, split, **kwargs):
        data_loaders = [iter(t.data_loaders[split]) for t in tasks]
        batch_counts = [len(t.data_loaders[split]) for t in tasks]
        max_count = max(batch_counts)
        for i in reversed(range(max_count)):
            for task, count, loader in zip(tasks, batch_counts, data_loaders):
                if count > i:
                    yield (next(loader), [task.name])


class SuperStagedScheduler(TaskScheduler):
    """Returns batches from an increasing number of tasks over _all_ epochs

    For example, if X is the largest task, start by training only on batches of X, then
    on X and Y, then on X, Y, and Z. So with two epochs worth of data (where Z has
    three batches total):

    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
              YYYYYYYYYYYYYYYYYYYY
                            ZZZZZZ

    Thus, the first "epoch" may only consist of task 1 if it has many more examples than
    the other tasks.
    """

    def __init__(self, model, tasks, n_epochs, split="train"):
        batch_counts = [len(t.data_loaders[split]) for t in tasks]
        self.batches_per_epoch = sum(batch_counts)

        total_batch_counts = [count * n_epochs for count in batch_counts]
        max_count = max(total_batch_counts)
        batch_assignments = []
        for i in reversed(range(max_count)):
            for idx, count in zip(range(len(tasks)), total_batch_counts):
                if count > i:
                    batch_assignments.append(idx)
        self.batch_assignments = batch_assignments
        self.epoch = 0

    def get_batches(self, tasks, split, **kwargs):
        data_loaders = [iter(t.data_loaders[split]) for t in tasks]
        start_idx = self.epoch * self.batches_per_epoch
        end_idx = (self.epoch + 1) * self.batches_per_epoch
        for idx in self.batch_assignments[start_idx:end_idx]:
            task_name = tasks[idx].name
            batch = next(data_loaders[idx], None)
            # If a data_loader runs out, reset it
            if batch is None:
                data_loaders[idx] = iter(tasks[idx].data_loaders[split])
                batch = next(data_loaders[idx], None)
            yield (batch, [task_name])
        self.epoch += 1
