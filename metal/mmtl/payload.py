import torch


class Payload(object):
    """A bundle of data_loaders...

    Args:
        name: the name of the payload (i.e., the name of the instance set)
        data_loaders: A dict of DataLoaders to feed through the network
            Each key in data.keys() should be in ["train", "valid", "test"]
            The DataLoaders should return batches of (X, Ys) pairs, where X[0] returns
                a complete input for feeding into the input_module, and Ys is a dict of
                the form {task_name: Y} and Y is an [n, ?] tensor
        labels_to_tasks
    """

    def __init__(self, name, data_loader, labels_to_tasks, split):
        self.name = name
        self.data_loader = data_loader
        self.labels_to_tasks = labels_to_tasks
        self.split = split

    def __repr__(self):
        return (
            f"Payload({self.name}: labels_to_tasks=[{self.labels_to_tasks}], "
            f"split={self.split})"
        )

    def add_label_set(
        self, task_name, label_name, label_list=None, label_fn=None, verbose=True
    ):
        """Adds a new label_set to an existing payload

        Args:
            task_name: the name of the Task to which the label_set belongs
            label_name: the name of the label_set being added
            label_fn: a function which maps a dataset item to a label
                labels will be combined using torch.stack(labels, dim=0)
            label_list: a list of labels in the correct order

        Note that either label_fn or label_list should be provided, but not both.
        """

        if label_fn is not None:
            assert label_list is None
            assert callable(label_fn)
            new_labels = torch.stack(
                [label_fn(x) for x in self.data_loader.dataset], dim=0
            )
        elif label_list is not None:
            assert label_fn is None
            assert isinstance(label_list, torch.Tensor)
            new_labels = label_list
        else:
            raise ValueError("Incorrect label object type -- supply list or function")

        if new_labels.dim() < 2:
            raise Exception("New label_set must have at least two dimensions: [n, ?]")

        self.data_loader.dataset.labels[task_name] = new_labels
        self.labels_to_tasks[label_name] = task_name

        if verbose:
            active = torch.any(new_labels != 0, dim=1)
            msg = (
                f"Added label_set with {sum(active.long())}/{len(active)} labels for "
                f"task {task_name} to payload {self.name}."
            )
            print(msg)

    def remove_label_set(self, label_name, verbose=True):
        self.data_loader.dataset.labels.pop(label_name)
        task_name = self.labels_to_tasks[label_name]
        del self.labels_to_tasks[label_name]

        if verbose:
            print(
                f"Removed label_set {label_name} for task {task_name} from payload {self.name}."
            )
