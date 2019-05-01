from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from metal.utils import move_to_device, recursive_merge_dicts, set_seed

model_defaults = {
    "seed": None,
    "device": 0,  # gpu id (int) or -1 for cpu
    "verbose": True,
    "fp16": False,
    "model_weights": None,  # the path to a saved checkpoint to initialize with
}


class MetalModel(nn.Module):
    """A dynamically constructed discriminative classifier

    Args:
        tasks: a list of Task objects which bring their own (named) modules

    We currently support up to N input modules -> middle layers -> up to N heads
    TODO: Accept specifications for more exotic structure (e.g., via user-defined graph)
    """

    def __init__(self, tasks, **kwargs):
        self.config = recursive_merge_dicts(model_defaults, kwargs, misses="insert")

        # Set random seed before initializing module weights
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        set_seed(self.config["seed"])

        super().__init__()

        # Build network
        self._build(tasks)
        self.task_map = {task.name: task for task in tasks}

        # Load weights
        if self.config["model_weights"]:
            self.load_weights(self.config["model_weights"])

        # Half precision
        if self.config["fp16"]:
            print("metal_model.py: Using fp16")
            self.half()

        # Move model to device now, then move data to device in forward() or calculate_loss()
        if self.config["device"] >= 0:
            if torch.cuda.is_available():
                if self.config["verbose"]:
                    print("Using GPU...")
                self.to(torch.device(f"cuda:{self.config['device']}"))
            else:
                if self.config["verbose"]:
                    print("No cuda device available. Using cpu instead.")

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            print(self)
            print()
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total number of parameters: {num_params}")

    def _build(self, tasks):
        """Iterates over tasks, adding their input_modules and head_modules"""
        # TODO: Allow more flexible specification of network structure
        self.input_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.input_module) for task in tasks}
        )
        self.middle_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.middle_module) for task in tasks}
        )
        self.head_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.head_module) for task in tasks}
        )

        self.loss_hat_funcs = {task.name: task.loss_hat_func for task in tasks}
        self.output_hat_funcs = {task.name: task.output_hat_func for task in tasks}

    def forward(self, X, task_names):
        """Returns the outputs of the requested task heads in a dictionary

        The output of each task is the result of passing the input through the
        input_module, middle_module, and head_module for that task, in that order.
        Before calculating any intermediate values, we first check whether a previously
        evaluated task has produced that intermediate result. If so, we use that.

        Args:
            X: a [batch_size, ...] batch from a DataLoader
        Returns:
            output_dict: {task_name (str): output (Tensor)}
        """
        input = move_to_device(X, self.config["device"])
        outputs = {}
        # TODO: Replace this naive caching scheme with a more intelligent and feature-
        # complete approach where arbitrary DAGs of modules are specified and we only
        # cache things that will be reused by another task
        for task_name in task_names:
            # Extra .module call is to get past DataParallel wrapper
            input_module = self.input_modules[task_name].module
            if input_module not in outputs:
                output = input_module(input)
                outputs[input_module] = output

            middle_module = self.middle_modules[task_name].module
            if middle_module not in outputs:
                output = middle_module(outputs[input_module])
                outputs[middle_module] = output

            head_module = self.head_modules[task_name].module
            if head_module not in outputs:
                output = head_module(outputs[middle_module])
                outputs[head_module] = output
        return {t: outputs[self.head_modules[t].module] for t in task_names}

    def calculate_loss(self, X, Ys, payload_name, labels_to_tasks):
        """Returns a dict of {task_name: loss (a FloatTensor scalar)}.

        Args:
            X: an appropriate input for forward(), either a Tensor or tuple
            Ys: a dict of {task_name: labels} where labels is [n, ?]
            labels_to_tasks: a dict of {label_name: task_name} indicating which task
                head to use to calculate the loss for each labelset.
        """
        task_names = set(labels_to_tasks.values())
        outputs = self.forward(X, task_names)
        loss_dict = {}  # Stores the loss by task
        count_dict = {}  # Stores the number of active examples by task

        for label_name, task_name in labels_to_tasks.items():
            loss_name = f"{task_name}/{payload_name}/{label_name}/loss"

            Y = Ys[label_name]
            assert isinstance(Y, torch.Tensor)

            out = outputs[task_name]
            # Identify which instances have at least one non-zero target labels
            active = torch.any(Y.detach() != 0, dim=1)
            count_dict[loss_name] = active.sum().item()

            # If there are inactive instances, slice them out to save computation
            # and ignore their contribution to the loss
            if 0 in active:
                Y = Y[active]
                if isinstance(out, torch.Tensor):
                    out = out[active]
                # If the output of the head has multiple fields, slice them all
                elif isinstance(out, dict):
                    out = move_to_device({k: v[active] for k, v in out.items()})

            # Convert to half precision last thing if applicable
            if self.config["fp16"] and Y.dtype == torch.float32:
                out["data"] = out["data"].half()
                Y = Y.half()

            # If no examples in this batch have labels for this task, skip loss calc
            # Active has type torch.uint8; avoid overflow with long()
            if active.long().sum():
                label_loss = self.loss_hat_funcs[task_name](
                    out, move_to_device(Y, self.config["device"])
                )
                assert isinstance(label_loss.item(), float)
                loss_dict[loss_name] = (
                    label_loss * self.task_map[task_name].loss_multiplier
                )

        return loss_dict, count_dict

    @torch.no_grad()
    def calculate_probs(self, X, task_names):
        """Returns a dict of {task_name: probs}

        Args:
            X: instances to feed through the network
            task_names: the names of the tasks for which to calculate outputs
        Returns:
            {task_name: probs}: probs is the output of the output_hat for the given
                task_head

        The type of each entry in probs depends on the task type:
            instance-based tasks: each entry in probs is a [k]-len array
            token-based tasks: each entry is a  [seq_len, k] array
        """
        assert self.eval()
        return {
            t: [probs.cpu().numpy() for probs in self.output_hat_funcs[t](out)]
            for t, out in self.forward(X, task_names).items()
        }

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary."""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def load_weights(self, model_path):
        """Load model weights from checkpoint."""
        if self.config["device"] >= 0:
            device = torch.device(f"cuda:{self.config['device']}")
        else:
            device = torch.device("cpu")
        try:
            self.load_state_dict(torch.load(model_path, map_location=device)["model"])
        except RuntimeError:
            print("Your destination state dict has different keys for the update key.")
            self.load_state_dict(
                torch.load(model_path, map_location=device)["model"], strict=False
            )

    def save_weights(self, model_path):
        """Saves weight in checkpoint directory"""
        raise NotImplementedError

    @torch.no_grad()
    def score(self, payload, metrics=[], verbose=True, **kwargs):
        """Calculate the requested metrics for the given payload

        Args:
            payload: a Payload to score
            metrics: a list of full metric names, a single full metric name, or []:
                list: a list of full metric names supported by the tasks' Scorers.
                    (full metric names are of the form task/payload/labelset/metric)
                    Only these metrics will be calculated and returned.
                []: defaults to all supported metrics for the given payload's Tasks
                str: a single full metric name
                    A single score will be returned instead of a dictionary

        Returns:
            scores: a dict of the form {metric_name: score} corresponding to the
                requested metrics (optionally a single score if metrics is a string
                instead of a list)
        """
        self.eval()
        return_unwrapped = isinstance(metrics, str)

        # If no specific metrics were requested, calculate all available metrics
        if metrics:
            metrics_list = metrics if isinstance(metrics, list) else [metrics]
            assert all(len(metric.split("/")) == 4 for metric in metrics_list)
            target_metrics = defaultdict(list)
            target_tasks = []
            target_labels = []
            for full_metric_name in metrics:
                task_name, payload_name, label_name, metric_name = full_metric_name.split(
                    "/"
                )
                target_tasks.append(task_name)
                target_labels.append(label_name)
                target_metrics[label_name].append(metric_name)
        else:
            target_tasks = set(payload.labels_to_tasks.values())
            target_labels = set(payload.labels_to_tasks.keys())
            target_metrics = {
                label_name: None for label_name in payload.labels_to_tasks
            }

        Ys, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, target_tasks, target_labels, return_preds=True, **kwargs
        )
        metrics_dict = {}
        for label_name, task_name in payload.labels_to_tasks.items():
            scorer = self.task_map[task_name].scorer
            task_metrics_dict = scorer.score(
                Ys[label_name],
                Ys_probs[task_name],
                Ys_preds[task_name],
                target_metrics=target_metrics[label_name],
            )
            # Expand short metric names into full metric names
            for metric_name, score in task_metrics_dict.items():
                full_metric_name = (
                    f"{task_name}/{payload.name}/{label_name}/{metric_name}"
                )
                metrics_dict[full_metric_name] = score
        # If a single metric was given as a string (not list), return a float
        if return_unwrapped:
            metric, score = metrics_dict.popitem()
            return score
        else:
            return metrics_dict

    @torch.no_grad()
    def predict_with_gold(
        self,
        payload,
        target_tasks=None,
        target_labels=None,
        return_preds=False,
        max_examples=0,
        **kwargs,
    ):
        """Extracts Y and calculates Y_prods, Y_preds for the given payload and tasks

        To get just the probabilities or predictions for a single task, consider using
        predict() or predict_probs().

        Args:
            payload: the Payload to make predictions for
            target_tasks: if not None, predict probs only for the specified tasks;
                otherwise, predict probs for all tasks with corresponding labelsets
                in the payload
            target_labels: if not None, return labels for only the specified labelsets;
                otherwise, return all labelsets
            return_preds: if True, also include preds in return values
            max_examples: if > 0, predict for a maximum of this many examples

        # TODO: consider returning Ys as tensors instead of lists (padded if necessary)
        Returns:
            Ys: a {label_name: Y} dict where Y is an [n] list of labels (often ints)
            Ys_probs: a {task_name: Y_probs} dict where Y_probs is a [n] list of
                probabilities
            Ys_preds: a {task_name: Y_preds} dict where Y_preds is a [n] list of
                predictions
        """
        validate_targets(payload, target_tasks, target_labels)
        if target_tasks is None:
            target_tasks = set(payload.labels_to_tasks.values())
        elif isinstance(target_tasks, str):
            target_tasks = [target_tasks]

        Ys = defaultdict(list)
        Ys_probs = defaultdict(list)

        total = 0
        for batch_num, (Xb, Yb) in enumerate(payload.data_loader):
            Yb_probs = self.calculate_probs(Xb, target_tasks)
            for task_name, yb_probs in Yb_probs.items():
                Ys_probs[task_name].extend(yb_probs)
            for label_name, yb in Yb.items():
                if target_labels is None or label_name in target_labels:
                    Ys[label_name].extend(yb.cpu().numpy())
            total += len(Xb)
            if max_examples > 0 and total >= max_examples:
                break

        if max_examples:
            Ys = {label_name: Y[:max_examples] for label_name, Y in Ys.items()}
            Ys_probs = {
                task_name: Y_probs[:max_examples]
                for task_name, Y_probs in Ys_probs.items()
            }

        if return_preds:
            Ys_preds = {
                task_name: [probs_to_preds(y_probs) for y_probs in Y_probs]
                for task_name, Y_probs in Ys_probs.items()
            }
            return Ys, Ys_probs, Ys_preds
        else:
            return Ys, Ys_probs

    # Single-task prediction helpers (for convenience)
    @torch.no_grad()
    def predict_probs(self, payload, task_name=None, **kwargs):
        """Return probabilistic labels for a single task of a payload

        Args:
            payload: a Payload
            task_name: the task to calculate probabilities for
                If task_name is None and the payload includes labels for only one task,
                return predictions for that task. If task_name is None and the payload
                includes labels for more than one task, raise an exception.
        Returns:
            Y_probs: an [n] list of probabilities
        """
        self.eval()

        if task_name is None:
            if len(payload.labels_to_tasks) > 1:
                msg = (
                    "The payload you provided contains labels for more than one "
                    "task, so task_name cannot be None."
                )
                raise Exception(msg)
            else:
                task_name = next(iter(payload.labels_to_tasks.values()))

        target_tasks = [task_name]
        _, Ys_probs = self.predict_with_gold(payload, target_tasks, **kwargs)
        return Ys_probs[task_name]

    @torch.no_grad()
    def predict(self, payload, task_name=None, return_probs=False, **kwargs):
        """Return predicted labels for a single task of a payload

        Args:
            payload: a Payload
            task_name: the task to calculate predictions for
                If task_name is None and the payload includes labels for only one task,
                return predictions for that task. If task_name is None and the payload
                includes labels for more than one task, raise an exception.

        Returns:
            Y_probs: an [n] list of probabilities
            Y_preds: an [n] list of predictions
        """
        self.eval()

        if task_name is None:
            if len(payload.labels_to_tasks) > 1:
                msg = (
                    "The payload you provided contains labels for more than one "
                    "task, so task_name cannot be None."
                )
                raise Exception(msg)
            else:
                task_name = next(iter(payload.labels_to_tasks.values()))

        target_tasks = [task_name]
        _, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, target_tasks, return_preds=True, **kwargs
        )
        Y_probs = Ys_probs[task_name]
        Y_preds = Ys_preds[task_name]
        if return_probs:
            return Y_preds, Y_probs
        else:
            return Y_preds


def validate_targets(payload, target_tasks, target_labels):
    if target_tasks:
        for task_name in target_tasks:
            if task_name not in set(payload.labels_to_tasks.values()):
                msg = (
                    f"Could not find the specified task_name {task_name} in "
                    f"payload {payload}."
                )
                raise Exception(msg)

    if target_labels:
        for label_name in target_labels:
            if label_name not in payload.labels_to_tasks:
                msg = (
                    f"Could not find the specified labelset {label_name} in "
                    f"payload {payload}."
                )
                raise Exception(msg)


def probs_to_preds(probs):
    """Identifies the largest probability in each column on the last axis

    We add 1 to the argmax to account for the fact that all labels in MeTaL are
    categorical and the 0 label is reserved for abstaining weak labels.
    """
    # TODO: Consider replacing argmax with a version of the rargmax utility to randomly
    # break ties instead of accepting the first one, or allowing other tie-breaking
    # strategies
    return np.argmax(probs, axis=-1) + 1
