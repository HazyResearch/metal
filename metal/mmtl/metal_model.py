import torch
import torch.nn as nn
import torch.nn.functional as F

from metal.classifier import Classifier
from metal.end_model.em_defaults import em_default_config
from metal.end_model.identity_module import IdentityModule
from metal.end_model.loss import SoftCrossEntropyLoss
from metal.utils import (
    MetalDataset,
    move_to_device,
    pred_to_prob,
    recursive_merge_dicts,
)

model_config = {
    "seed": None,
    "device": 0,  # gpu id (int) or -1 for cpu
    "verbose": True,
}


class MetalModel(nn.Module):
    """A dynamically constructed discriminative classifier

    Args:
        tasks: a list of Task objects which bring their own (named) modules

    We currently support up to N input modules -> middle layers -> up to N heads
    TODO: Accept specifications for more exotic structure (e.g., via user-defined graph)
    """

    def __init__(self, tasks, **kwargs):
        super().__init__()
        self.config = recursive_merge_dicts(model_config, kwargs, misses="insert")
        self._build(tasks)

        # Move model to device now, then move data to device in forward() or calcluate_loss()
        if self.config["verbose"] and self.config["device"] >= 0:
            print("Using GPU...")
            self.to(f"cuda:{self.config['device']}")

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            print(self)
            print()

    def _build(self, tasks):
        """Iterates over tasks, adding their input_modules and head_modules

        Do this naively for now with a double for-loop
        # TODO: Can do better than O(n^2), though not a big deal
        """
        self.input_modules = nn.ModuleDict(
            {task.name: task.input_module for task in tasks}
        )
        self.head_modules = nn.ModuleDict(
            {task.name: task.head_module for task in tasks}
        )
        self.loss_hat_funcs = {task.name: task.loss_hat_func for task in tasks}
        self.output_hat_funcs = {task.name: task.output_hat_func for task in tasks}

        # TODO: allow some number of middle modules (of arbitrary sizes) to be specified
        self.middle_modules = None

        # HACK: this does not allow reuse of intermediate computation or middle modules
        # Not hard to change this, but not necessary for GLUE
        task_paths = {}
        for task in tasks:
            input_module = self.input_modules[task.name]
            head_module = self.head_modules[task.name]
            task_paths[task.name] = nn.Sequential(input_module, head_module)
        self.task_paths = nn.ModuleDict(task_paths)

    def forward(self, X, task_names):
        """Returns the outputs of the task heads in a dictionary

        Args:
            X: a [batch_size, ...] batch from a DataLoader
        Returns:
            output_dict: {task_name (str): output (Tensor)}
        """
        return {
            t: self.task_paths[t](move_to_device(X, self.config["device"]))
            for t in task_names
        }

    def calculate_loss(self, X, Y, task_names):
        """Returns a dict of {task_name: loss (an FloatTensor scalar)}."""
        return {
            t: self.loss_hat_funcs[t](out, move_to_device(Y, self.config["device"]))
            for t, out in self.forward(X, task_names).items()
        }

    @torch.no_grad()
    def calculate_output(self, X, task_names):
        """Returns a dict of {task_name: probs (an [n, k] Tensor of probabilities)}."""
        # return F.softmax(self.forward(X), dim=1).data.cpu().numpy()
        return {
            t: self.output_hat_funcs[t](out)
            for t, out in self.forward(X, task_names).items()
        }

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)
