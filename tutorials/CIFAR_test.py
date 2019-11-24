import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import CIFAR10

from resnet import ResNet18
from metal import EndModel
from metal.utils import convert_labels
from torch.utils.data import Dataset

import argparse


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size (default: 1)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
    

# The following identity module is to essentially replace the last FC layer
# in the resnet model by the FC in MeTal

class IdentityModule(nn.Module):
    """A default identity input module that simply passes the input through."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x

# Here we create a dataloader that transforms CIFAR labels from 0-9, to 1-10,
# We do this because MeTal treats a 0 label as abstain
class MetalDataset(Dataset):
    """A dataset that group each item in X with it label from Y

    Args:
        X: an n-dim iterable of items
        Y: a torch.Tensor of labels
            This may be hard labels [n] or soft labels [n, k]
    """

    def __init__(self, dataset):
        self.dataset = dataset
        #Y = convert_labels(Y,'onezero','categorical')
        #self.Y = Y
        #assert len(X) == len(Y)

    def __getitem__(self, index):
        x,y = self.dataset[index]
        # convert to metal form
        y += 1
        return tuple([x,y])

    def __len__(self):
        return len(self.dataset)
    


def train_model():
    
    global args
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = dataloader.DataLoader(MetalDataset(trainset), batch_size=128, shuffle=True, num_workers=2)


    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = dataloader.DataLoader(MetalDataset(testset), batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    model = ResNet18()
    model.linear = IdentityModule()

    end_model = EndModel([512,10], input_module=model, seed=123, use_cuda=True, relu=False)

    end_model.train_model(train_data=train_loader, dev_data=test_loader, l2=args.weight_decay, lr=args.lr,  n_epochs=args.epochs, print_every=1, validation_metric='accuracy')

    end_model.score(test_loader, metric=['accuracy', 'precision', 'recall', 'f1'])



if __name__ == "__main__":
    train_model()
    