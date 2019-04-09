import copy

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
            if self.config["verbose"]:
                print("Using GPU...")
            self.to(torch.device(f"cuda:{self.config['device']}"))

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
        self.attention_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.attention_module) for task in tasks}
        )
        self.head_modules = nn.ModuleDict(
            {task.name: nn.DataParallel(task.head_module) for task in tasks}
        )

        self.loss_hat_funcs = {task.name: task.loss_hat_func for task in tasks}
        self.output_hat_funcs = {task.name: task.output_hat_func for task in tasks}

    def _copy_task_modules(self, from_task, to_task, deepcopy=False):
        """ Maps all modules from_task --> to_task."""
        if self.config["verbose"]:
            print(f"Shallow copying modules from {from_task} to {to_task}")

        if deepcopy:
            self.task_map[to_task] = copy.deepcopy(self.task_map[from_task])
            self.input_modules[to_task] = copy.deepcopy(self.input_modules[from_task])
            self.middle_modules[to_task] = copy.deepcopy(self.middle_modules[from_task])
            self.attention_modules[to_task] = copy.deepcopy(
                self.attention_modules[from_task]
            )
            self.head_modules[to_task] = copy.deepcopy(self.head_modules[from_task])
            self.loss_hat_funcs[to_task] = copy.deepcopy(self.loss_hat_funcs[from_task])
            self.output_hat_funcs[to_task] = copy.deepcopy(
                self.output_hat_funcs[from_task]
            )
        else:
            self.task_map[to_task] = self.task_map[from_task]
            self.input_modules[to_task] = self.input_modules[from_task]
            self.middle_modules[to_task] = self.middle_modules[from_task]
            self.attention_modules[to_task] = self.attention_modules[from_task]
            self.head_modules[to_task] = self.head_modules[from_task]
            self.loss_hat_funcs[to_task] = self.loss_hat_funcs[from_task]
            self.output_hat_funcs[to_task] = self.output_hat_funcs[from_task]

    def add_missing_slice_heads(self, all_tasks, deepcopy):
        """Given a set of all tasks (likely defined by model payloads), add additional
        task heads that are copies (deep or shallow) of the original tasks.

        For example, "COLA:question" might be a labelset, but the model might be missing
        the corresponding task head. Call this function to make a copy of the (pretrained)
        "COLA" task head with the key "COLA:question"
        """

        for task_name in all_tasks:
            if task_name not in self.task_map:
                # NOTE: assume that all slice tasks are structured "{foo_task}:{bar_slice}"
                orig_task_name = task_name.split(":")[0]
                if orig_task_name not in all_tasks:
                    raise ValueError(
                        f"'{orig_task_name}' task was not found to evaluate '{task_name}' slice"
                    )
                self._copy_task_modules(orig_task_name, task_name)

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
        for task_name in task_names:
            # Extra .module because of DataParallel wrapper!
            input_module = self.input_modules[task_name]
            if input_module.module not in outputs:
                outputs[input_module.module] = input_module(input)
            middle_module = self.middle_modules[task_name]
            if middle_module.module not in outputs:
                outputs[middle_module.module] = middle_module(outputs[input_module.module])
            attention_module = self.attention_modules[task_name]
            if attention_module.module not in outputs:
                outputs[attention_module.module] = attention_module(outputs[middle_module.module])
            head_module = self.head_modules[task_name]
            if head_module.module not in outputs:
                outputs[head_module.module] = head_module(outputs[attention_module.module])
        return {t: outputs[self.head_modules[t].module] for t in task_names}

    def calculate_loss(self, X, Ys, task_names):
        """Returns a dict of {task_name: loss (a FloatTensor scalar)}.

        Args:
            X: an appropriate input for forward(), either a Tensor or tuple
            Ys: a dict of {task_name: labels} where labels is [n, ?]
            task_names: a list of the names of the tasks to compute loss for
        """
        outputs = self.forward(X, task_names)
        loss_dict = {}  # Stores the loss by task
        count_dict = {}  # Stores the number of active examples by task
        for task_name in task_names:
            Y = Ys[task_name]
            out = outputs[task_name]
            # Identify which instances have at least one non-zero target labels
            active = torch.any(Y.detach() != 0, dim=1)
            count_dict[task_name] = active.sum().item()
            # If there are inactive instances, slice them out to save computation
            if 0 in active:
                Y = Y[active]
                # NOTE: This makes an assumption we should list elsewhere (and confirm
                # with helpful error messages) that if X has multiple fields, they'll
                # be arranged in a tuple where each component has batch_size first dim.
                if isinstance(out, torch.Tensor):
                    out = out[active]
                elif isinstance(out, tuple):
                    out = move_to_device(tuple(x[active] for x in out))
            # If no examples in this batch have labels for this task, skip loss calc
            if self.config["fp16"] and Y.dtype == torch.float32:
                out = out.half()
                Y = Y.half()
            if active.sum():
                task_loss = self.loss_hat_funcs[task_name](
                    out, move_to_device(Y, self.config["device"])
                )
                assert isinstance(task_loss.item(), float)
                loss_dict[task_name] = (
                    task_loss * self.task_map[task_name].loss_multiplier
                )

        return loss_dict, count_dict

    @torch.no_grad()
    def calculate_probs(self, X, task_names):
        """Returns a dict of {task_name: probs} where probs is [n]-length list}.

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
                    (full metric names are of the form task/payload/metric)
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
            assert all(len(metric.split("/")) == 2 for metric in metrics_list)
            task_names = set([metric.split("/")[0] for metric in metrics_list])
            target_metrics = {}
            for task_name in task_names:
                target_metrics[task_name] = [m for m in metrics if task_name in m]
        else:
            task_names = payload.task_names
            target_metrics = {task_name: None for task_name in task_names}

        # NOTE: If evaluating on slice payloads that have no corresponding task head
        # create that slice head and re-map this payload.
        if set(task_names) != set(self.task_map.keys()):
            self.add_missing_slice_heads(task_names, deepcopy=False)

        Ys, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, task_names, return_preds=True, **kwargs
        )

        metrics_dict = {}
        for task_name in task_names:
            scorer = self.task_map[task_name].scorer
            task_metrics_dict = scorer.score(
                Ys[task_name],
                Ys_probs[task_name],
                Ys_preds[task_name],
                target_metrics=target_metrics[task_name],
            )
            # Expand short metric names into full metric names
            for metric, score in task_metrics_dict.items():
                full_metric_name = f"{task_name}/{payload.name}/{metric}"
                metrics_dict[full_metric_name] = score

        # If a single metric was given as a string (not list), return a float
        if return_unwrapped:
            metric, score = metrics_dict.popitem()
            return score
        else:
            return metrics_dict

    @torch.no_grad()
    def predict_with_gold(
        self, payload, task_names=None, return_preds=False, max_examples=0, **kwargs
    ):
        """Extracts (Y) or calculates (Y_prods, Y_preds) for the given payload and tasks

        To get just the probabilities or predictions for a single task, consider using
        predict() or predict_probs().

        Args:
            payload: the Payload to make predictions for
            task_names: if not None, predict probs only for the specified tasks;
                otherwise, predict probs for all tasks present in the payload
            return_preds: if True, also include preds in return values
            max_examples: if > 0, predict for a maximum of this many examples

        Returns:
            Ys: a {task_name: Y} dict where Y is an [n] list of labels (often ints)
            Ys_probs: a {task_name: Y_probs} dict where Y_probs is a [n] list of
                probabilities
            Ys_preds: a {task_name: Y_preds} dict where Y_preds is a [n] list of
                predictions
        """
        if task_names:
            for task_name in task_names:
                if task_name not in payload.task_names:
                    msg = (
                        f"Could not find the specified task_name {task_name} in "
                        f"payload {payload}"
                    )
                    raise Exception(msg)
        else:
            task_names = payload.task_names

        Ys = {task_name: [] for task_name in task_names}
        Ys_probs = {task_name: [] for task_name in task_names}

        total = 0
        for batch_num, batch in enumerate(payload.data_loader):
            Xb = batch[0]
            Yb = batch[1]
            Yb_probs = self.calculate_probs(Xb, task_names)
            for task_name in task_names:
                Ys[task_name].extend(Yb[task_name].cpu().numpy())
                Ys_probs[task_name].extend(Yb_probs[task_name])
            total += len(Xb)
            if max_examples > 0 and total >= max_examples:
                break

        if max_examples:
            Ys = {task_name: Y[:max_examples] for task_name, Y in Ys.items()}
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
            task_name: the task to calculate probabilities for; defaults to the name of
                the payload if none
        Returns:
            Y_probs: an [n] list of probabilities
        """
        self.eval()
        if task_name is None and payload.name in payload.task_names:
            task_name = payload.name
        else:
            msg = (
                f"Argument task_name must be in payload.task_names "
                f"({payload.task_names})"
            )
            raise Exception(msg)

        _, Ys_probs = self.predict_with_gold(payload, task_name, **kwargs)
        return Ys_probs[task_name]

    @torch.no_grad()
    def predict(self, payload, task_name=None, return_probs=False, **kwargs):
        """Return predicted labels for a single task of a payload

        Args:
            payload: a Payload
            task_name: the task to calculate probabilities for; defaults to the name of
                the payload if none
        Returns:
            Y_probs: an [n] list of probabilities
            Y_preds: an [n] list of predictions
        """
        self.eval()
        if task_name is None and payload.name in payload.task_names:
            task_name = payload.name
        else:
            msg = (
                f"Argument task_name must be in payload.task_names "
                f"({payload.task_names})"
            )
            raise Exception(msg)

        _, Ys_probs, Ys_preds = self.predict_with_gold(
            payload, task_name, return_preds=True, **kwargs
        )
        Y_probs = Ys_probs[task_name]
        Y_preds = Ys_preds[task_name]
        if return_probs:
            return Y_preds, Y_probs
        else:
            return Y_preds


def probs_to_preds(probs):
    """Identifies the largest probability in each column on the last axis

    We add 1 to the argmax to account for the fact that all labels in MeTaL are
    categorical and the 0 label is reserved for abstaining weak labels.
    """
    # TODO: Consider replacing argmax with a version of the rargmax utility to randomly
    # break ties instead of accepting the first one, or allowing other tie-breaking
    # strategies
    return np.argmax(probs, axis=-1) + 1
