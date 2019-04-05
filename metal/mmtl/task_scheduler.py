import random
from abc import ABC, abstractmethod


class PayloadScheduler(ABC):
    """Returns batches from multiple payloads in some order for MTL training"""

    def __init__(self, model, payloads, split, **kwargs):
        pass

    @abstractmethod
    def get_batches(self, payloads, split, **kwargs):
        """Returns batches from all payloads in some order until one 'epoch' is reached

        Args:
            payloads: a list of Payloads
            split: only Payloads belonging to this split will be returned

        For now, an epoch is defined as one full pass through all datasets.
        This is required because of assumptions currently made in the logger and
        training loop about the number of batches that will be seen per epoch.
        """
        pass


class ProportionalScheduler(PayloadScheduler):
    """Returns batches proportional to the fraction of the total number of batches"""

    def get_batches(self, payloads, split, **kwargs):
        # First filter to only those payloads belonging to the given split
        payloads = [p for p in payloads if p.split == split]
        data_loaders = [iter(p.data_loader) for p in payloads]
        batch_counts = [len(p.data_loader) for p in payloads]
        batch_assignments = []
        for payload_idx in range(len(payloads)):
            batch_assignments.extend([payload_idx] * batch_counts[payload_idx])
        random.shuffle(batch_assignments)

        for payload_idx in batch_assignments:
            yield (next(data_loaders[payload_idx]), payloads[payload_idx].task_names)


class StagedScheduler(PayloadScheduler):
    """Returns batches from an increasing number of payloads over the epoch

    Start by training only on the payload with the largest number of batches.
    Gradually increase the payloads being trained on so that the final batches are
    sampling equally from all tasks.

    For example, if X is the largest task, start by training only on batches of X, then
    on X and Y, then on X, Y, and Z. This resets every epoch:

    XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX
         YYYYYYYYYY|     YYYYYYYYYY
                ZZZ|            ZZZ
    """

    def get_batches(self, payloads, split, **kwargs):
        # First filter to only those payloads belonging to the given split
        payloads = [p for p in payloads if p.split == split]
        data_loaders = [iter(p.data_loader) for p in payloads]
        batch_counts = [len(p.data_loader) for p in payloads]
        max_count = max(batch_counts)
        for i in reversed(range(max_count)):
            for payload, count, loader in zip(payloads, batch_counts, data_loaders):
                if count > i:
                    yield (next(loader), payload.task_names)


class SuperStagedScheduler(PayloadScheduler):
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

    def __init__(self, model, payloads, n_epochs, split="train"):
        payloads = [p for p in payloads if p.split == split]
        batch_counts = [len(p.data_loader) for p in payloads]
        self.batches_per_epoch = sum(batch_counts)

        total_batch_counts = [count * n_epochs for count in batch_counts]
        max_count = max(total_batch_counts)
        batch_assignments = []
        for i in reversed(range(max_count)):
            for idx, count in zip(range(len(payloads)), total_batch_counts):
                if count > i:
                    batch_assignments.append(idx)
        self.batch_assignments = batch_assignments
        self.epoch = 0

    def get_batches(self, payloads, split, **kwargs):
        payloads = [p for p in payloads if p.split == split]
        data_loaders = [iter(p.data_loader) for p in payloads]
        start_idx = self.epoch * self.batches_per_epoch
        end_idx = (self.epoch + 1) * self.batches_per_epoch
        for idx in self.batch_assignments[start_idx:end_idx]:
            task_names = payloads[idx].task_names
            batch = next(data_loaders[idx], None)
            # If a data_loader runs out, reset it
            if batch is None:
                data_loaders[idx] = iter(payloads[idx].data_loader)
                batch = next(data_loaders[idx], None)
            yield (batch, task_names)
        self.epoch += 1
