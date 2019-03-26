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
        Yields:
            batch: a tuple of (X_batch_dict, Y_batch_dict)
            payload_name: the name of the payload returned
            labels_to_tasks: a dict indicating which task each label set belongs to

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
            batch = next(data_loaders[payload_idx])
            payload = payloads[payload_idx]
            yield (batch, payload.name, payload.labels_to_tasks)
