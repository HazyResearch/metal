import codecs
import os
import pathlib

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    # Only use tqdm notebook if not in travis testing
    if "CI" not in os.environ:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm


def tsv_path_for_dataset(dataset_name, dataset_split):
    """ Returns dataset location on disk given name and split. """
    return os.path.join(
        os.environ["CXRDATA"], "{}/{}.tsv".format(dataset_name, dataset_split)
    )

def get_label_fn(input_dict):
    """ Given mapping (specified as dict), return two-way functions for mapping."""
    reverse_dict = {y: x for x, y in input_dict.items()}
    return input_dict.get, reverse_dict.get


def get_task_config(dataset_name, split, subsample, finding):
    """ Returns the tsv_config to be used in params of CXRDataset.from_tsv for
    specific task and split. """

    if dataset_name == "CXR8":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "path_to_labels": tsv_path_for_dataset("CXR8", split),
            "path_to_images": "/lfs/1/jdunnmon/data/nih/images/images", 
            "transform": None,
            "subsample": subsample,
            "finding": finding,
            "label_type": int,
            "get_uid": False,
        }

