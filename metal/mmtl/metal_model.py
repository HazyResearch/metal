import numpy as np
import torch
import torch.nn as nn

from metal.mmtl.utils.utils import stack_batches
from metal.utils import move_to_device, recursive_merge_dicts

model_config = {"seed": 123, "device": 0, "verbose": True}  # gpu id (int) or -1 for cpu


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

        # Move model to device now, then move data to device in forward() or calculate_loss()
        if self.config["device"] >= 0:
            if self.config["verbose"]:
                print("Using GPU...")
            self.to(torch.device(f"cuda:{self.config['device']}"))

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            print(self)
            print()

    def _build(self, tasks):
        """Iterates over tasks, adding their input_modules and head_modules"""
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
            task_paths[task.name] = nn.DataParallel(
                nn.Sequential(input_module, head_module)
            )
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

    def load_weights(self, model_path):
        """Load model weights from checkpoint"""
        if self.config["device"] >= 0:
            map_location = f"cuda:{self.config['device']}"
        else:
            map_location = "cpu"
        self.load_state_dict(torch.load(model_path, map_location=map_location)["model"])

    def save_weights(self, model_path):
        """Saves weight in checkpoint directory"""
        raise NotImplementedError

    # Single task prediction helpers (for convenience)

    @torch.no_grad()
    def predict_probs(self, task, split):
        """Return probabilistic labels for a single task and split

        Returns:
            probs: an [n, k] np.ndarray of probabilities
        """
        _, Y_probs = self._predict_probs(task, split)
        return Y_probs

    @torch.no_grad()
    def predict(self, task, split, return_probs=False, **kwargs):
        """Return predictions for a single task and split (and optionally return probs)

        Returns:
            preds: an [n, 1] np.ndarray of probabilities
            probs: (optional) an [n, k] np.ndarray of probabilities if return_probs=True
        """
        _, Y_probs, Y_preds = self._predict_probs(
            task, split, return_preds=True, **kwargs
        )
        if return_probs:
            return Y_preds, Y_probs
        else:
            return Y_preds

    @torch.no_grad()
    def score(self, task, split, metrics, verbose=True, **kwargs):
        """Calculate one or more metrics for a single task and split

        Args:
            task: a Task
            split: a target split to score on
            metrics: a list of simple metric names supported by the task's Scorer
                (optionally a string may be passed instead and a single score returned)

        Returns:
            scores: a list of scores corresponding to the requested metrics
                (optionally a single score if metrics is a string instead of a list)
        """
        return_list = isinstance(metrics, list)
        metrics_list = metrics if isinstance(metrics, list) else [metrics]
        target_metrics = [f"{task.name}/{split}/{metric}" for metric in metrics_list]
        metrics_dict = task.scorer.score(self, task, target_metrics=target_metrics)

        scores = []
        for metric in target_metrics:
            score = metrics_dict[metric]
            scores.append(score)
            if self.config["verbose"]:
                print(f"{metric.capitalize()}: {score:.3f}")

        # If a single metric was given as a string (not list), return a float
        if len(scores) == 1 and not return_list:
            return scores[0]
        else:
            return scores

    @torch.no_grad()
    def _predict_probs(self, task, split, return_preds=False, max_examples=0, **kwargs):
        """Unzips the dataloader of a task's split and returns Y, Y_prods, Y_preds

        Note: it is generally preferable to use predict() or predict_probs() unless
            the gold labels Y are requires as well.

        Args:
            task: a Task to predict on
            split: the split to predict on
            return_preds: if True, also include preds in return values
            max_examples: if > 0, predict for a maximum of this many examples

        Returns:
            Y: [n] np.ndarray of ints
            Y_probs: [n, k] np.ndarray of floats
            Y_preds: [n, 1] np.ndarray of ints
        """
        Y = []
        Y_probs = []
        total = 0
        for batch_num, batch in enumerate(task.data_loaders[split]):
            Xb, Yb = batch
            Y.append(Yb)
            Y_probs.append(self.calculate_output(Xb, [task.name])[task.name])
            total += Yb.shape[0]
            if max_examples and total >= max_examples:
                break

        # Stack batches
        # TODO: (VC) replace this with the regression head abstraction
        if task.name != "STSB":
            Y = stack_batches(Y).astype(np.int)
        else:
            Y = stack_batches(Y).astype(np.float)
        Y_probs = stack_batches(Y_probs).astype(np.float)

        if max_examples:
            Y = Y[:max_examples]
            Y_probs = Y_probs[:max_examples, :]

        if return_preds:
            Y_preds = self._break_ties(Y_probs, **kwargs).astype(np.int)
            return Y, Y_probs, Y_preds
        else:
            return Y, Y_probs

    def _break_ties(self, Y_probs, break_ties="random"):
        """Break ties in each row of a tensor according to the specified policy

        Args:
            Y_probs: An [n, k] np.ndarray of probabilities
            break_ties: A tie-breaking policy:
                "abstain": return an abstain vote (0)
                "random": randomly choose among the tied options
                    NOTE: if break_ties="random", repeated runs may have
                    slightly different results due to difference in broken ties
                [int]: ties will be broken by using this label
        """
        n, k = Y_probs.shape
        Y_preds = np.zeros(n)
        diffs = np.abs(Y_probs - Y_probs.max(axis=1).reshape(-1, 1))

        TOL = 1e-5
        for i in range(n):
            max_idxs = np.where(diffs[i, :] < TOL)[0]
            if len(max_idxs) == 1:
                Y_preds[i] = max_idxs[0] + 1
            # Deal with "tie votes" according to the specified policy
            elif break_ties == "random":
                Y_preds[i] = np.random.choice(max_idxs) + 1
            elif break_ties == "abstain":
                Y_preds[i] = 0
            elif isinstance(break_ties, int):
                Y_preds[i] = break_ties
            else:
                ValueError(f"break_ties={break_ties} policy not recognized.")
        return Y_preds
