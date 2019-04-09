import codecs
import os
import pathlib

from torchvision import transforms

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

def transform_for_dataset(dataset_name, dataset_split):

    if dataset_split in ["val", "valid", "test", "dev"]:
        dataset_split = "val"

    if dataset_name == "CXR8":
        # use imagenet mean,std for normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
        # define torchvision transforms
        data_transforms = {
            "train": transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.Scale(1024),
                    # because scale doesn't always give 224 x 224, this ensures 224 x
                    # 224
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [   
                    transforms.Scale(1024),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }
    return data_transforms[dataset_split]


def get_task_config(dataset_name, split, subsample, finding):
    """ Returns the tsv_config to be used in params of CXRDataset.from_tsv for
    specific task and split. """

    if dataset_name == "CXR8":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "path_to_labels": tsv_path_for_dataset("CXR8", split),
            "path_to_images": "/data/datasets/nih/images/images", 
            "transform": transform_for_dataset("CXR8", split),
            "subsample": subsample,
            "finding": finding,
            "label_type": int,
            "get_uid": False,
        }

