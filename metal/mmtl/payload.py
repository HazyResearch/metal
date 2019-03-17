class Payload(object):
    """A bundle of data_loaders...

    Args:
        name: the name of the payload (i.e., the name of the instance set)
        data_loaders: A dict of DataLoaders to feed through the network
            Each key in data.keys() should be in ["train", "valid", "test"]
            The DataLoaders should return batches of (X, Ys) pairs, where X[0] returns
                a complete input for feeding into the input_module, and Ys is a list
                containing S label sets, such that Y[0][0] is the appropriate label(s)
                to pass into the loss head for the first example and first label set.
        task_names: a list of the ids of the tasks that the label_sets in Y correspond
            to (1 instanceset + 1 or more label_sets)
    """

    def __init__(self, name, data_loader, task_names, split):
        self.name = name
        self.data_loader = data_loader
        self.task_names = task_names
        self.split = split

    def __repr__(self):
        return (
            f"Payload({self.name}: tasks=[{','.join(self.task_names)}], "
            f"split={self.split})"
        )

    def add_label_set(self, task_name, label_set=None, label_fn=None, verbose=True):
        """Adds a new label_set to an existing payload

        Args:
            task_name: the name of the Task to which the label_set belongs.
            label_fn: a function which maps a dataset item to a label
            label_set: a list of labels in the correct order

        Note that either label_fn or label_set should be provided, but not both.
        """

        if label_fn is not None:
            assert callable(label_fn)
            assert label_set is None
            labels_new = [label_fn(x) for x in self.data_loader.dataset]
        elif label_set is not None:
            assert isinstance(label_set, list)
            assert label_fn is None
            labels_new = label_set
        else:
            raise ValueError("Incorrect label object type -- supply list or function")

        self.data_loader.dataset.labels[task_name] = labels_new
        self.task_names.append(task_name)

        if verbose:
            msg = (
                f"Added label_set with {len(labels_new)} labels for task "
                f"{task_name} to payload {self.name}."
            )
            print(msg)
