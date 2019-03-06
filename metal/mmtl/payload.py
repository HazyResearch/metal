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
        task_names: a list of the ids of the tasks that the labelsets in Y correspond to
    (1 instanceset + 1 or more labelsets)
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
