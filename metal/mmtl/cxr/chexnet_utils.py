import cxr_dataset as CXR
import pandas as pd
import torch
from metal.end_model import EndModel
from torch import nn
from torchvision import transforms


def fetch_dataloader(
    image_path, num_workers=8, batch_size=16, get_filename=False, subsample=None
):
    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # load labels
    df = pd.read_csv("nih_labels.csv", index_col=0)

    # define torchvision transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Scale(224),
                # because scale doesn't always give 224 x 224, this ensures 224 x
                # 224
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    # create train/val/test dataloaders
    transformed_datasets = {}

    transformed_datasets["train"] = CXR.CXRDataset(
        path_to_images=image_path,
        fold="train",
        transform=data_transforms["train"],
        get_filename=get_filename,
        subsample=subsample,
    )

    transformed_datasets["val"] = CXR.CXRDataset(
        path_to_images=image_path,
        fold="val",
        transform=data_transforms["val"],
        get_filename=get_filename,
        subsample=subsample,
    )

    transformed_datasets["test"] = CXR.CXRDataset(
        path_to_images=image_path,
        fold="test",
        transform=data_transforms["val"],
        get_filename=get_filename,
        subsample=subsample,
    )

    dataloaders = {}

    dataloaders["train"] = torch.utils.data.DataLoader(
        transformed_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    dataloaders["val"] = torch.utils.data.DataLoader(
        transformed_datasets["val"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    dataloaders["test"] = torch.utils.data.DataLoader(
        transformed_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloaders


class CheXNetEndModel(EndModel):
    def _get_loss_fn(self):
        # Overwriting what was done in _build to have normal BCELoss
        self.criteria = nn.BCELoss()
        if self.config["use_cuda"]:
            criteria = self.criteria.cuda()
        else:
            criteria = self.criteria
        # This self.preprocess_Y allows us to not handle preprocessing
        # in a custom dataloader, but decreases speed a bit
        loss_fn = lambda X, Y: criteria(self.forward(X), Y)
        return loss_fn
