from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from metal.utils import padded_tensor


class MmtlDataset(Dataset):
    """A pairing of data with one or more fields to one or more label sets

    Args:
        X_dict: a dict of the form {field_name: values} where field_name is a string and
            values is an [n]-length iterable.
        Y_dict: a dict of the form {label_name: values} where label_name is a string and
            values is an [n]-length iterable.
    """

    def __init__(self, X_dict, Y_dict):
        self.X_dict = X_dict
        self.Y_dict = Y_dict

    def __getitem__(self, index):
        x_dict = {key: field[index] for key, field in self.X_dict.items()}
        y_dict = {key: label[index] for key, label in self.Y_dict.items()}
        return x_dict, y_dict

    def __len__(self):
        return len(next(iter(self.X_dict.values())))


def mmtl_collate_fn(batch_list):
    """Collates a batch of (x_dict, y_dict) tuples into padded (X_dict, Y_dict)

    Assumes that all values are torch Tensors

    Args:
        batch_list: a list of tuples containing (x_dict, y_dict), where x_dict
            and y_dict each contain a fields or labels for a single instance.
    Returns:
        X_batch: a dict of the form {field_name: values} where field_name is a string
            and values is a [batch_size]-length iterable
        Y_batch: a dict of the form {label_name: values} where label_name is a string
            and values is a Tensor where values.shape[0] == batch_size

    Resulting values may be [n, 1] (e.g., instance labels) or [n, seq_len] (e.g.,
        token labels)
    """

    def list_to_tensor(list_):
        if all(value.dim() == 0 for value in list_):
            tensor_ = torch.stack(list_, dim=0).view(batch_size, -1)
        elif all(len(list_[i]) == len(list_[0]) for i in range(len(list_))):
            tensor_ = torch.stack(list_, dim=0).view(batch_size, -1)
        else:
            tensor_ = padded_tensor(list_).view(batch_size, -1)
        return tensor_

    batch_size = len(batch_list)

    X_batch = defaultdict(list)
    Y_batch = defaultdict(list)
    for x_dict, y_dict in batch_list:
        for field_name, value in x_dict.items():
            X_batch[field_name].append(value)
        for label_name, value in y_dict.items():
            Y_batch[label_name].append(value)

    for field_name, values in X_batch.items():
        # Merge lists of tensors, leave other data types alone
        if isinstance(values[0], torch.Tensor):
            X_batch[field_name] = list_to_tensor(values)

    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(values)

    # Remove 'defaultdict' property
    return dict(X_batch), dict(Y_batch)


class MmtlDataLoader(DataLoader):
    def __init__(self, dataset, collate_fn=mmtl_collate_fn, **kwargs):
        assert isinstance(dataset, MmtlDataset)
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
