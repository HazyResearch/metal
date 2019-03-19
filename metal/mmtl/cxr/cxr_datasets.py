import os
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data as Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from tqdm import tqdm

from metal.mmtl.glue.glue_preprocess import get_task_tsv_config, load_tsv
from metal.utils import padded_tensor, set_seed


DATASET_CLASS_DICT = {
                "CXR8": CXR8Dataset,
                }
get_cxr_dataset(
            dataset_name,
            split=split,
            max_datapoints=max_datapoints,
            generate_uids=generate_uids,
        )


def get_cxr_dataset(dataset_name, split, **kwargs):
    """ Create and returns specified cxr dataset based on image path."""

    # MODIFY THIS TO GET THE RIGHT LOCATIONS FOR EACH!!
    config = get_task_tsv_config(dataset_name, split)
    dataset_class = DATASET_CLASS_DICT[dataset_name]

    return dataset_class(
        config["path_to_images"],
        config["path_to_labels"],
        split,
        transform=config["transform"],
        sample=config["sample"],
        finding=config["finding"],
        get_filename=False, 
        **kwargs,
    )

class CXR8Dataset(Dataset):
    """
    Dataset to load NIH Chest X-ray 14 dataset

    Modified from reproduce-chexnet repo
    https://github.com/jrzech/reproduce-chexnet
    """

    def __init__(
        self,
        path_to_images,
        path_to_labels,
        split,
        transform=None,
        sample=0,
        finding="any",
        get_filename=False,
    ):

        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labe;s
        label_file = os.path.join(self.path_to_labels, "nih_labels.csv")
        self.df = pd.read_csv(label_file)
        self.df = self.df[self.df["fold"] == split]
        self.get_filename = get_filename

        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if sample > 0 and sample < len(self.df):
            self.df = self.df.sample(sample)

        if (
            not finding == "any"
        ):  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print(
                        "No positive cases exist for "
                        + LABEL
                        + ", returning all unfiltered cases"
                    )
            else:
                print(
                    "cannot filter on finding "
                    + finding
                    + " as not in data - please check spelling"
                )

        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]
    
    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.path_to_images, self.df.index[idx]))
        image = image.convert("RGB")

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            # can leave zero if zero, else make one
            if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype("int") > 0:
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype("int")

        label = torch.Tensor(label)

        if self.transform:
            image = self.transform(image)

        if self.get_filename:
            # This mode exists for exploring predictions as in reproduce-chexnet
            return [image, label, self.df.index[idx]]

        else:
            return [image, label]

    def __len__(self):
        return len(self.df)
