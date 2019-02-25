import copy
import os
import random
import warnings
from collections import defaultdict
from pprint import pprint
from shutil import copy2

import numpy as np
import torch
import torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.utils import clip_grad_norm_

from metal.logging import Checkpointer, LogWriter, TensorBoardWriter
from metal.logging.utils import split_full_metric
from metal.mmtl.mmtl_logger import Logger  # NOTE: we load special MTL logger
from metal.mmtl.task_scheduler import (
    ProportionalScheduler,
    StagedScheduler,
    SuperStagedScheduler,
)
from metal.mmtl.utils.metrics import GLUE_METRICS, glue_score
from metal.utils import recursive_merge_dicts

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

trainer_config = {
    "verbose": True,
    "seed": None,
    # Commit hash
    "commit_hash": None,
    # Display
    "progress_bar": False,
    # Dataloader
    # TODO: Restore the option for them to pass in raw simple data which we wrap up
    # "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True},
    # TODO: Restore ability to weight losses by class and/or task
    # "loss_weights": None,
    # Train Loop
    "n_epochs": 1,
    "l2": 0.0,
    "grad_clip": 1.0,
    # Evaluate dev for during training every this many epochs
    # Optimizer
    "optimizer_config": {
        "optimizer": "adam",
        "optimizer_common": {"lr": 0.01},
        # Optimizer - SGD
        "sgd_config": {"momentum": 0.9},
        # Optimizer - Adam
        "adam_config": {"betas": (0.9, 0.999)},
        # Optimizer - RMSProp
        "rmsprop_config": {},  # Use defaults
    },
    # LR Scheduler (for learning rate)
    "lr_scheduler": None,
    # ['linear', 'exponential', 'reduce_on_plateau']
    # 'reduce_on_plateau' uses checkpoint_metric to assess plateaus
    "lr_scheduler_config": {
        # Linearly increase lr up to "lr" over this many warmup_units
        "warmup_steps": 0.0,
        "warmup_unit": "batches",  # ["epochs", "batches"]
        # The minimum lr that will ever be used after warmup.
        "min_lr": 0.0,
        # Scheduler - exponential
        "exponential_config": {"gamma": 0.999},  # decay rate
        # Scheduler - reduce_on_plateau
        "plateau_config": {"factor": 0.5, "patience": 10, "threshold": 0.0001},
    },
    # Metrics
    "metrics_config": {
        # The list of task metrics (task/split/metric) to calculate (and log);
        # if empty, calculate all metrics supported by all tasks' Scorers.
        "task_metrics": [],
        # The list of trainer standard metrics to calculate (and log)
        "trainer_metrics": [],  # e.g., "glue"
        # Run scorers over a maximum of this many examples if > 0.
        "max_valid_examples": 0,
        # The name of the split to run scoring on during training
        # To score over multiple splits, set valid_split=None and use task_metrics
        "valid_split": "valid",
        # The name of the split to run final evaluation on after training
        "test_split": "test",
        # If non-None, only calculate and report these metrics every `score_every`
        # units (this can include the names of built-in and user-defined metrics);
        # otherwise, include all metrics returned by task Scorers.
        # TODO: "metrics_filter": None,
        # TODO: "score_limit": None,  # Evaluate scorer on only this many examples
    },
    # Task Scheduler
    "task_scheduler": "proportional",  # ["proportional", "staged"]
    # Logger (see metal/logging/logger.py for descriptions)
    "logger": True,
    "logger_config": {
        "log_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
        # Report loss every this many log_units
        "log_every": 1.0,
        # Calculate and report metrics every this many log_units:
        #   -1: default to log_every
        #   0: do not calculate or log metrics
        #   otherwise: must be a multiple of log_every
        "score_every": -1.0,
        "log_lr": True,  # If True, also log learning rate whenever loss is logged
    },
    # LogWriter/Tensorboard (see metal/logging/writer.py for descriptions)
    "writer": None,  # [None, "json", "tensorboard"]
    "writer_config": {  # Log (or event) file stored at log_dir/run_dir/run_name
        "log_dir": f"{os.environ['METALHOME']}/logs",
        "run_dir": None,
        "run_name": None,
        # May specify a subset of metrics in metrics_dict to be written.
        # If [], write all available metrics to the logs
        "writer_metrics": [],
    },
    # Checkpointer (see metal/logging/checkpointer.py for descriptions)
    "checkpoint": True,  # If True, checkpoint models when certain conditions are met
    # EXPERIMENTAL: If True, save a separate set of checkpoints (assuming strategy of
    # checkpoint_best) for each task
    "checkpoint_tasks": False,
    # If true, checkpoint directory will be cleaned after training (if checkpoint_best
    # is True, the best model will first be copied to the log_dir/run_dir/run_name/)
    "checkpoint_cleanup": True,
    "checkpoint_config": {
        # TODO: unify checkpoint=['every', 'best', 'final']; specify one strategy
        "checkpoint_every": 0,  # Save a model checkpoint every this many log_units
        # If checkpoint_best, also save the "best" model according to some metric
        # The "best" model will have the ['max', 'min'] value of checkpoint_metric
        # This metric must be produced by one of the task Scorer objects so it will be
        # available for lookup; assumes valid split unless appended with "train/"
        "checkpoint_best": False,
        # "checkpoint_final": False,  # Save a model checkpoint at the end of training
        "checkpoint_metric": "model/train/loss",
        "checkpoint_metric_mode": "min",
        # If None, checkpoint_dir defaults to the log_dir/run_dir/run_name/checkpoints
        # Note that using this default path is strongly recommended.
        # If you hardcode checkpoint_dir, checkpoints from concurrent runs may overwrite
        # each other.
        "checkpoint_dir": None,
        "checkpoint_runway": 0,
    },
}


class MultitaskTrainer(object):
    """Driver for the MTL training process"""

    def __init__(self, **kwargs):
        self.config = recursive_merge_dicts(trainer_config, kwargs, misses="insert")

        # Set random seeds
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        self._set_seed(self.config["seed"])

    def train_model(self, model, tasks, **kwargs):
        # NOTE: misses="insert" so we can log extra metadata (e.g. num_parameters)
        # and eventually write to disk.
        self.config = recursive_merge_dicts(self.config, kwargs, misses="insert")

        # Calculate epoch statistics
        # NOTE: Because we use SubsetSampler, one dataset may actually include two
        # splits, so we calculate approximate count size using batch_size * num_batches
        # examples_per_epoch = sum([len(t.data_loaders["train"].dataset) for t in tasks])
        self.task_names = [task.name for task in tasks]
        self.batches_per_epoch = sum([len(t.data_loaders["train"]) for t in tasks])
        self.examples_per_epoch = sum(
            [
                len(t.data_loaders["train"]) * t.data_loaders["train"].batch_size
                for t in tasks
            ]
        )
        if self.config["verbose"]:
            print(f"Beginning train loop.")
            print(
                f"Expecting a total of approximately {self.examples_per_epoch} examples "
                f"and {self.batches_per_epoch} batches per epoch from {len(tasks)} tasks."
            )

        # Set training components
        self._set_writer()
        self._set_logger()
        self._set_checkpointer(tasks)
        self._set_optimizer(model)
        self._set_lr_scheduler(model)  # TODO: Support more detailed training schedules
        self._set_task_scheduler(model, tasks)

        # Record config
        if self.writer:
            self.writer.write_config(self.config)

        # Train the model
        # TODO: Allow other ways to train besides 1 epoch of all datasets
        model.train()
        # Dict metrics_hist contains the most recently recorded value of all metrics
        self.metrics_hist = {}
        self._reset_losses()
        for epoch in range(self.config["n_epochs"]):
            progress_bar = self.config["progress_bar"] and self.config["verbose"]
            t = tqdm(
                enumerate(self.task_scheduler.get_batches(tasks, "train")),
                total=self.batches_per_epoch,
                disable=(not progress_bar),
            )
            for batch_num, (task_names, batch) in t:
                # NOTE: actual batch_size may not equal config's target batch_size,
                # for example due to orphan batches
                # TODO (BH): Determine very explicitly what we require of X or Y in
                # order to get the batchsize; we only need one dimension, but would
                # like to allow for maximum flexibilty of data/label formatting.
                # Will Y always be a list or 1D np.ndarray? No!
                _, Y = batch
                batch_size = len(Y)
                batch_id = epoch * self.batches_per_epoch + batch_num

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to calculate the average loss per example
                losses = model.calculate_loss(*batch, task_names)
                loss = sum(losses.values())
                if torch.isnan(loss):
                    msg = "Loss is NaN. Consider reducing learning rate."
                    raise Exception(msg)

                # Backward pass to calculate gradients
                # Loss is an average loss per example
                if model.config["fp16"]:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()

                # Clip gradient norm (not individual gradient magnitudes)
                # max_grad_value = max([p.grad.abs().max().item() for p in model.parameters()])
                if self.config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config["grad_clip"]
                    )

                # Perform optimizer step
                self.optimizer.step()

                # Update loss
                for task_name in task_names:
                    self.running_losses[task_name] += (
                        losses[task_name].item() * batch_size
                    )
                    self.running_examples[task_name] += batch_size

                # Calculate metrics, log, and checkpoint as necessary
                metrics_dict = self._execute_logging(model, tasks, batch_size)

                # Apply learning rate scheduler
                self._update_lr_scheduler(model, batch_id)

                # tqdm output
                if len(tasks) == 1:
                    t.set_postfix(loss=metrics_dict["model/train/loss"])
                else:
                    losses = {}
                    for key in metrics_dict:
                        if "loss" in key:
                            losses[key] = metrics_dict[key]
                    t.set_postfix(losses)

        model.eval()
        # Restore best model if applicable
        if self.checkpointer and self.checkpointer.checkpoint_best:
            # First do a final checkpoint at the end of training
            metrics_dict = self._execute_logging(
                model, tasks, batch_size, force_log=True
            )

            self.checkpointer.load_best_model(model=model)
            # Copy best model to log directory
            if self.writer:
                path_to_best = os.path.join(
                    self.checkpointer.checkpoint_dir, "best_model.pth"
                )
                path_to_logs = self.writer.log_subdir
                if os.path.isfile(path_to_best):
                    copy2(path_to_best, path_to_logs)

        # Print final performance values
        if self.config["verbose"]:
            print("Finished training")
        test_split = self.config["metrics_config"]["test_split"]
        if test_split is not None:
            metrics_dict = self.calculate_metrics(model, tasks, split=test_split)
        if self.config["verbose"]:
            pprint(metrics_dict)

        # EXPERIMENTAL:
        # Recalculate scores using task-specific checkpoints
        if self.config["checkpoint_tasks"]:
            metrics_dict = copy.deepcopy(metrics_dict)
            valid_split = self.config["metrics_config"]["valid_split"]
            test_split = self.config["metrics_config"]["test_split"]
            for task, checkpointer in zip(tasks, self.task_checkpointers):
                checkpointer.load_best_model(model=model)
                task_metrics = [
                    checkpointer.checkpoint_metric.replace(valid_split, test_split)
                ]
                metrics_dict_task = task.scorer.score(
                    model, task, task_metrics, test_split
                )
                metrics_dict.update(metrics_dict_task)
                if self.writer:
                    path_to_best = os.path.join(
                        checkpointer.checkpoint_dir, "best_model.pth"
                    )
                    path_to_logs = os.path.join(
                        self.writer.log_subdir, f"{task.name}_best_model.pth"
                    )
                    copy2(path_to_best, path_to_logs)
            print("Final scores using task-specific checkpoints:")
            pprint(metrics_dict)

        # Clean up checkpoints
        if self.checkpointer and self.config["checkpoint_cleanup"]:
            print("Cleaning checkpoints")
            self.checkpointer.clean_up()

        # Write log if applicable
        if self.writer:
            self.writer.write_metrics(metrics_dict)
            self.writer.write_log()
            self.writer.close()

    def _execute_logging(self, model, tasks, batch_size, force_log=False):
        model.eval()
        metrics_dict = {}
        metrics_dict.update(self.aggregate_losses(tasks))
        self.logger.increment(batch_size)

        do_log = False
        if self.logger.loss_time():
            self._reset_losses()
            do_log = True
        if self.logger.metrics_time() or force_log:
            # Unless valid_split is None, Scorers will only score on one split
            valid_split = self.config["metrics_config"]["valid_split"]
            metrics_dict.update(self.calculate_metrics(model, tasks, split=valid_split))
            do_log = True
        if do_log or force_log:
            # Log to screen/file/TensorBoard
            self.logger.log(metrics_dict)
            # Save best model if applicable
            self._checkpoint(model, metrics_dict)

        self.metrics_hist.update(metrics_dict)
        model.train()
        return metrics_dict

    def aggregate_losses(self, tasks):
        """Calculate the average loss for each task since the last calculation

        If no examples of a certain task have been seen since the losses were reset,
        use the most recently reported value again (stored in metrics_hist).
        If the loss for a certain task has never been reported, report it as None.
        """
        metrics_dict = {}
        for task in tasks:
            if self.running_examples[task.name]:
                loss = self.running_losses[task.name] / self.running_examples[task.name]
            elif self.metrics_hist.get(f"{task.name}/train/loss"):
                loss = self.metrics_hist[f"{task.name}/train/loss"]
            else:
                loss = None
            metrics_dict[f"{task.name}/train/loss"] = loss
        # Report micro average of losses
        total_loss = sum(self.running_losses.values())
        total_examples = sum(self.running_examples.values())
        if total_examples > 0:
            metrics_dict["model/train/loss"] = total_loss / total_examples
        # Log learning rate
        if self.config["logger_config"]["log_lr"]:
            # For now just report one global lr; eventually support lr groups
            metrics_dict[f"model/train/lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics_dict

    def calculate_metrics(self, model, tasks, split=None):
        metrics_dict = {}
        # Update metrics_hist after task_metrics so trainer_metrics have access to most
        # recently calculated numbers (e.g., glue score aggregates task scores)
        metrics_dict.update(self.calculate_task_metrics(model, tasks, split))
        self.metrics_hist.update(metrics_dict)
        metrics_dict.update(self.calculate_trainer_metrics(model, tasks, split))
        self.metrics_hist.update(metrics_dict)
        return metrics_dict

    def calculate_task_metrics(self, model, tasks, split=None):
        metrics_dict = {}
        task_metrics = self.config["metrics_config"]["task_metrics"]
        trainer_metrics = self.config["metrics_config"]["trainer_metrics"]
        # Pull out any requested loss metrics
        # NOTE: We currently break our own rule and calculate model-wide overall loss
        # in calculate_task_metrics. We do this because we need access to the total
        # loss and examples counts; we can't just average the task-specific losses
        # equally after the fact.
        loss_metrics = [
            metric
            for metric in (task_metrics + trainer_metrics)
            if "/loss" in metric and "/train/" not in metric
        ]
        task_metrics = [metric for metric in task_metrics if "/loss" not in metric]
        max_examples = self.config["metrics_config"]["max_valid_examples"]

        # Calculate loss for non-train splits
        if loss_metrics:
            # TODO: (BH) handle max_examples
            loss_dict = self._calculate_valid_losses(
                model, tasks, split, max_examples=max_examples
            )
            # TODO: improve efficiency by only calculating the losses the user requested
            # rather than computing all and filtering at the end.
            for loss_name, loss_value in loss_dict.items():
                if loss_name in loss_metrics:
                    metrics_dict[loss_name] = loss_value

        # Calculate metrics from Scorers
        for task in tasks:
            metrics_dict_task = task.scorer.score(
                model, task, task_metrics, split, max_examples=max_examples
            )
            metrics_dict.update(metrics_dict_task)
        return metrics_dict

    def calculate_trainer_metrics(self, model, tasks, split):
        trainer_metrics = self.config["metrics_config"]["trainer_metrics"]
        metrics_dict = {}
        # HACK: glue should not be hardcoded
        if "glue" in trainer_metrics:
            if len(tasks) != 9:
                msg = "You requested glue score but submitted fewer than 9 tasks. Use 'glue_partial' instead."
                raise Exception(msg)
            metric = "glue"
            metrics_dict[f"model/{split}/{metric}"] = glue_score(
                self.metrics_hist, split
            )
        elif "glue_partial" in trainer_metrics:
            if len(tasks) == 9:
                msg = "You requested glue_partial score but submitted all 9 tasks. Use 'glue' instead."
                raise Exception(msg)
            metric = "glue_partial"
            metrics_dict[f"model/{split}/{metric}"] = glue_score(
                self.metrics_hist, split
            )
        return metrics_dict

    @torch.no_grad()
    def _calculate_valid_losses(self, model, tasks, split, max_examples=0):
        """Calculate the loss for the valid split"""
        # Error checing
        assert split != "train"
        if split is None:
            msg = "MeTaL does not currently support calculating the loss for multiple non-train splits"
            raise NotImplementedError(msg)
        elif split == "test":
            msg = "MeTaL does not support calculating loss on the test set during training."
        # Calculate task-specific losses
        task_losses = defaultdict(float)
        task_examples = defaultdict(float)
        total_examples = 0
        # WARNING: For calculating valid loss, we simply use a proportional scheduler.
        # Note that if max_examples > 0, some tasks may be underrepresented in the first
        # max_examples examples.
        task_scheduler = ProportionalScheduler(model, tasks, split)
        for task_names, batch in task_scheduler.get_batches(tasks, split):
            _, Y = batch
            batch_size = len(Y)
            losses = model.calculate_loss(*batch, task_names)
            for task_name, loss in losses.items():
                task_losses[task_name] += loss.item() * batch_size
                task_examples[task_name] += batch_size
            total_examples += batch_size
            if max_examples > 0 and total_examples >= max_examples:
                break
        # Aggregate losses and store in dictionary
        metrics_dict = {}
        for task in tasks:
            full_name = f"{task.name}/{split}/loss"
            if task_examples[task.name] > 0:
                metrics_dict[full_name] = (
                    task_losses[task.name] / task_examples[task.name]
                )
            else:
                metrics_dict[full_name] = np.nan
        metrics_dict[f"model/{split}/loss"] = sum(task_losses.values()) / sum(
            task_examples.values()
        )
        return metrics_dict

    def _checkpoint(self, model, metrics_dict):
        if self.checkpointer is None:
            return
        iteration = self.logger.unit_total
        self.checkpointer.checkpoint(
            metrics_dict, iteration, model, self.optimizer, self.lr_scheduler
        )
        # EXPERIMENTAL:
        if self.config["checkpoint_tasks"]:
            for checkpointer in self.task_checkpointers:
                checkpointer.checkpoint(
                    metrics_dict, iteration, model, self.optimizer, self.lr_scheduler
                )

    def _reset_losses(self):
        self.running_losses = defaultdict(float)
        self.running_examples = defaultdict(int)

    def _set_writer(self):
        writer_config = self.config["writer_config"]
        writer_config["verbose"] = self.config["verbose"]
        if self.config["writer"] is None:
            self.writer = None
        elif self.config["writer"] == "json":
            self.writer = LogWriter(**writer_config)
        elif self.config["writer"] == "tensorboard":
            self.writer = TensorBoardWriter(**writer_config)
        else:
            raise Exception(f"Unrecognized writer: {self.config['writer']}")

    def _set_logger(self):
        # If not provided, set score_every to log_every
        logger_config = self.config["logger_config"]
        if logger_config["score_every"] < 0:
            logger_config["score_every"] = logger_config["log_every"]
        self.logger = Logger(
            logger_config,
            self.batches_per_epoch,
            self.writer,
            verbose=self.config["verbose"],
        )

    def _set_checkpointer(self, tasks):
        if (
            self.config["checkpoint"]
            or self.config["lr_scheduler"] == "reduce_on_plateau"
        ):
            self._validate_checkpoint_metric(tasks)
            # Set checkpoint_dir to log_dir/checkpoints/
            if self.writer:
                if not self.config["checkpoint_config"]["checkpoint_dir"]:
                    self.config["checkpoint_config"]["checkpoint_dir"] = os.path.join(
                        self.writer.log_subdir, "checkpoints"
                    )
                else:
                    # If you hardcode checkpoint_dir, checkpoints from concurrent runs
                    # may overwrite each other.
                    msg = (
                        "You have provided checkpoint_dir, overriding the default "
                        "of using log_dir/run_dir/run_name/checkpoints. Be careful: "
                        "multiple concurrent runs may override each other."
                    )
                    warnings.warn(msg)
            else:
                self.config["checkpoint_config"]["checkpoint_dir"] = "checkpoints"
            # Create Checkpointer
            self.checkpointer = Checkpointer(
                self.config["checkpoint_config"], verbose=self.config["verbose"]
            )
        else:
            self.checkpointer = None

        # EXPERIMENTAL: Optionally add task-specific checkpointers
        # HACK: This is hard-coded in a way specific to Glue!
        self.task_checkpointers = []
        if self.config["checkpoint_tasks"]:
            msg = (
                "checkpoint_tasks setting does not have the same thorough error "
                "checking that the normal checkpoint operation has, so you may "
                "accidentally be trying to checkpoint metrics that aren't going to be "
                "found in the metrics_dict if you're not careful."
            )
            warnings.warn(msg)
            for task in tasks:
                checkpoint_config = copy.deepcopy(self.config["checkpoint_config"])
                checkpoint_config["checkpoint_dir"] += f"/{task.name}"
                checkpoint_config["checkpoint_best"] = True
                checkpoint_metric = f"{task.name}/valid/{GLUE_METRICS[task.name]}"
                checkpoint_config["checkpoint_metric"] = checkpoint_metric
                print(checkpoint_config["checkpoint_metric"])
                checkpoint_config["checkpoint_metric_mode"] = "max"
                task_checkpointer = Checkpointer(
                    checkpoint_config, verbose=self.config["verbose"]
                )
                self.task_checkpointers.append(task_checkpointer)

    def _set_optimizer(self, model):
        optimizer_config = self.config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # Special optimizer for fp16
        if model.config["fp16"]:

            # TODO(maxlam): Figure out a cleaner way to do this
            from apex.optimizers import FP16_Optimizer, FusedAdam

            class FP16_OptimizerMMTLModified(FP16_Optimizer):
                def step(self, closure=None):
                    """
                    Not supporting closure.
                    """
                    # First compute norm for all group so we know if there is overflow
                    grads_groups_flat = []
                    norm_groups = []
                    skip = False
                    for i, group in enumerate(self.fp16_groups):

                        # Only part that's changed -- zero out grads that are None
                        grads_to_use = []
                        for p in group:
                            if p.grad is None:
                                size = list(p.size())
                                grads_to_use.append(p.new_zeros(size))
                            else:
                                grads_to_use.append(p.grad)
                        grads_groups_flat.append(_flatten_dense_tensors(grads_to_use))

                        norm_groups.append(
                            self._compute_grad_norm(grads_groups_flat[i])
                        )
                        if norm_groups[i] == -1:  # TODO: early break
                            skip = True

                    if skip:
                        self._update_scale(skip)
                        return

                    # norm is in fact norm*cur_scale
                    self.optimizer.step(
                        grads=[[g] for g in grads_groups_flat],
                        output_params=[[p] for p in self.fp16_groups_flat],
                        scale=self.cur_scale,
                        grad_norms=norm_groups,
                    )

                    # TODO: we probably don't need this? just to be safe
                    for i in range(len(norm_groups)):
                        updated_params = _unflatten_dense_tensors(
                            self.fp16_groups_flat[i], self.fp16_groups[i]
                        )
                        for p, q in zip(self.fp16_groups[i], updated_params):
                            p.data = q.data

                    self._update_scale(False)
                    return

            optimizer = FusedAdam(
                parameters,
                **optimizer_config["optimizer_common"],
                bias_correction=False,
                max_grad_norm=1.0,
            )
            optimizer = FP16_OptimizerMMTLModified(optimizer, dynamic_loss_scale=True)

        elif opt == "sgd":
            optimizer = optim.SGD(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["sgd_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "rmsprop":
            optimizer = optim.RMSprop(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["rmsprop_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "adam":
            optimizer = optim.Adam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "adamax":
            optimizer = optim.Adamax(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
                weight_decay=self.config["l2"],
            )
        elif opt == "sparseadam":
            optimizer = optim.SparseAdam(
                parameters,
                **optimizer_config["optimizer_common"],
                **optimizer_config["adam_config"],
            )
            if self.config["l2"]:
                raise Exception(
                    "SparseAdam optimizer does not support weight_decay (l2 penalty)."
                )
        else:
            raise ValueError(f"Did not recognize optimizer option '{opt}'")
        self.optimizer = optimizer

    def _set_lr_scheduler(self, model):
        lr_scheduler = self.config["lr_scheduler"]
        lr_scheduler_config = self.config["lr_scheduler_config"]

        # Create warmup scheduler for first warmup_steps warmup_units if applicable
        self._set_warmup_scheduler(model)

        optimizer_to_config = self.optimizer

        # If using half precision, configure the underlying
        # optimizer of FP16_Optimizer
        if model.config["fp16"]:
            optimizer_to_config = self.optimizer.optimizer

        # Create regular lr scheduler for use after warmup
        if lr_scheduler is None:
            lr_scheduler = None
        else:
            lr_scheduler_config = self.config["lr_scheduler_config"]
            if lr_scheduler == "linear":
                total_steps = self.batches_per_epoch * self.config["n_epochs"]
                cooldown_steps = total_steps - self.warmup_steps
                linear_cooldown_func = lambda x: (cooldown_steps - x) / cooldown_steps
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer_to_config, linear_cooldown_func
                )
            elif lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer_to_config, **lr_scheduler_config["exponential_config"]
                )
            elif lr_scheduler == "reduce_on_plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_to_config,
                    min_lr=lr_scheduler_config["min_lr"],
                    **lr_scheduler_config["plateau_config"],
                )
            else:
                raise ValueError(
                    f"Did not recognize lr_scheduler option '{lr_scheduler}'"
                )
        self.lr_scheduler = lr_scheduler

    def _set_warmup_scheduler(self, model):
        optimizer_to_use = self.optimizer
        if model.config["fp16"]:
            optimizer_to_use = self.optimizer.optimizer

        if self.config["lr_scheduler_config"]["warmup_steps"]:
            warmup_unit = self.config["lr_scheduler_config"]["warmup_unit"]
            warmup_steps = self.config["lr_scheduler_config"]["warmup_steps"]
            # Convert warmup unit to batches
            if warmup_unit == "epochs":
                self.warmup_steps = max(1, int(warmup_steps * self.batches_per_epoch))
            elif warmup_unit == "batches":
                self.warmup_steps = max(1, int(warmup_steps))
            else:
                msg = f"warmup_unit must be 'epochs' or 'batches', not {warmup_unit}"
                raise Exception(msg)
            # This function returns a multiplicative factor based on iteration number
            linear_warmup_func = lambda x: x / self.warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer_to_use, linear_warmup_func
            )
        else:
            warmup_scheduler = None
            self.warmup_steps = 0
        self.warmup_scheduler = warmup_scheduler

    def _update_lr_scheduler(self, model, step):
        """Optionally update the learning rate scheduler with each batch"""

        optimizer_to_use = self.optimizer
        if model.config["fp16"]:
            optimizer_to_use = self.optimizer.optimizer

        lr_scheduler_config = self.config["lr_scheduler_config"]

        if self.warmup_scheduler and (step < self.warmup_steps):
            self.warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            # Metrics-based scheduler(s)
            if self.config["lr_scheduler"] == "reduce_on_plateau":
                checkpoint_config = self.config["checkpoint_config"]
                metric_name = checkpoint_config["checkpoint_metric"]
                score = self.metrics_hist.get(metric_name, None)
                if score is not None:
                    self.lr_scheduler.step(score)
            # Iteration-based scheduler(s)
            else:
                self.lr_scheduler.step()
                # HACK: We enforce min_lr right now by just overwriting
                min_lr = lr_scheduler_config["min_lr"]
                if min_lr and optimizer_to_use.param_groups[0]["lr"] < min_lr:
                    optimizer_to_use.param_groups[0]["lr"] = min_lr

    def _set_task_scheduler(self, model, tasks):
        if self.config["task_scheduler"] == "proportional":
            self.task_scheduler = ProportionalScheduler(model, tasks, "train")
        elif self.config["task_scheduler"] == "staged":
            self.task_scheduler = StagedScheduler(model, tasks, "train")
        elif self.config["task_scheduler"] == "superstaged":
            if self.config["lr_scheduler"] is not None:
                msg = (
                    "When using task_scheduler=='superstaged', lr_scheduler should be "
                    "None"
                )
                warnings.warn(msg)
            self.task_scheduler = SuperStagedScheduler(
                model, tasks, self.config["n_epochs"], "train"
            )
        else:
            raise NotImplementedError

    def _validate_checkpoint_metric(self, tasks):
        # Confirm that checkpoint_metric is a metric that will be available
        checkpoint_metric = self.config["checkpoint_config"]["checkpoint_metric"]
        if checkpoint_metric.startswith("model"):
            metric_name = checkpoint_metric.split("/")[-1]
            if (
                metric_name != "loss"
                and metric_name not in self.config["metrics_config"]["trainer_metrics"]
            ):
                msg = (
                    f"The checkpoint_metric you specified ('{checkpoint_metric}') is "
                    f"not currently supported."
                )
                raise Exception(msg)
        else:
            if checkpoint_metric.count("/") != 2:
                msg = (
                    f"checkpoint_metric must have a full metric name "
                    f"(task/split/metric); you submitted: {checkpoint_metric}"
                )
                raise Exception(msg)

            task_name, split, metric = split_full_metric(checkpoint_metric)
            try:
                task = [t for t in tasks if t.name == task_name][0]
            except IndexError:
                msg = (
                    f"The task for your specified checkpoint_metric "
                    f"({checkpoint_metric}) was not found in the list of "
                    f"submitted tasks: {[t.name for t in tasks]}."
                )
                raise Exception(msg)

            if metric != "loss" and metric not in task.scorer.metrics:
                msg = (
                    f"The checkpoint_metric you specified "
                    f"({checkpoint_metric}) is not in the list of supported "
                    f"metrics ({task.scorer.metrics}) for the Scorer of that task. "
                    f"Either change your checkpoint_metric, use a different Scorer, "
                    f"or add a custom_metric_func that outputs your desired metric."
                )
                raise Exception(msg)

        task_metrics = self.config["metrics_config"]["task_metrics"]
        if task_metrics and checkpoint_metric not in task_metrics:
            msg = (
                "checkpoint_metric must be a metric in task_metrics if "
                "task_metrics is not empty"
            )
            raise Exception(msg)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
