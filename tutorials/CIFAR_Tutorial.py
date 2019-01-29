"""
This tutorial script runs a simple ResNet to classify the CIFAR-10 dataset using MeTaL.
The purpose of this particular tutorial is to demonstrate how a standard machine learning
task can be run using MeTaL utilities.

Running this script with settings in run_CIFAR_Tutorial.sh should give performance on the
order of 92-93% dev set accuracy, which is comparable to the performance numbers presented in
the pytorch-cifar repo (https://github.com/kuangliu/pytorch-cifar).  Note that as in the
pytorch-cifar repo, the dev and test set are the same.  This is not generally the case!

Running this script with default parameters should recover ~88% accuracy after 10 epochs.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from torchvision.models import resnet
import metal.contrib.modules.resnet_cifar10 as resnet
from metal import EndModel
from metal.utils import convert_labels

# Checking to see if cuda is available for GPU use
cuda = torch.cuda.is_available()

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Training CIFAR 10")

parser.add_argument(
    "--epochs", default=10, type=int, help="number of total epochs to run"
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    help="mini-batch size (default: 10)",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    help="initial learning rate",
)

parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    help="weight decay (default: 1e-4)",
)

# Setting up dataset to adjust CIFAR indices to one-index,
# per MeTaL convention


class MetalCIFARDataset(Dataset):
    """A dataset that group each item in X with it label from Y

    Args:
        X: an n-dim iterable of items
        Y: a torch.Tensor of labels
            This may be hard labels [n] or soft labels [n, k]
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        # convert to metal form
        y += 1
        return tuple([x, y])

    def __len__(self):
        return len(self.dataset)


def train_model():

    global args
    args = parser.parse_args()

    # Set up transformations for incoming data
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    # Create datasets and data loaders
    trainset = CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = dataloader.DataLoader(
        MetalCIFARDataset(trainset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    testset = CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = dataloader.DataLoader(
        MetalCIFARDataset(testset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Defining classes in CIFAR-10 in order
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Define input encoder
    model = resnet.ResNet18()
    encode_dim = 512

    # Define end model. Note that in MeTaL we supply the encoding dimension
    # to enable auto-compilation of a multi-task end model.  This abstraction
    # is required here, even in the single task case.  See the default end
    # model config file for all possible options.
    end_model = EndModel(
        [encode_dim, len(classes)],
        input_module=model,
        seed=123,
        use_cuda=cuda,
        skip_head=True,
        input_relu=False,
        input_batchnorm=False,
        middle_relu=False,
        middle_batchnorm=False,
    )

    # Train end model
    end_model.train_model(
        train_data=train_loader,
        valid_data=test_loader,
        l2=args.weight_decay,
        lr=args.lr,
        n_epochs=args.epochs,
        print_every=1,
        validation_metric="accuracy",
    )

    # Test end model
    end_model.score(
        test_loader, metric=["accuracy", "precision", "recall", "f1"]
    )


if __name__ == "__main__":
    train_model()
