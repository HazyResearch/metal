import math
import os
import random
from collections import defaultdict
from pprint import pprint
from shutil import copy2

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from metal.logging import Checkpointer, LogWriter, TensorBoardWriter
from metal.mmtl.mmtl_logger import Logger  # NOTE: we load special MTL logger
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
    # Display
    "progress_bar": False,
    # Dataloader
    # TODO: Restore the option for them to pass in raw simple data which we wrap up
    # "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True},
    # Loss weights
    # TODO: Restore ability to weight losses by class and/or task
    # "loss_weights": None,
    # Train Loop
    "n_epochs": 10,
    "l2": 0.0,
    "grad_clip": 0.0,
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
    "lr_scheduler": "reduce_on_plateau",
    # [None, 'exponential', 'reduce_on_plateau']
    # 'reduce_on_plateau' uses checkpoint_metric to assess plateaus
    "lr_scheduler_config": {
        # Freeze learning rate initially this many epochs
        # "lr_freeze": 0,
        # Linearly increase lr from "lr_cold" to "lr" over this many steps (batches)
        # Scheduler - exponential
        "exponential_config": {"gamma": 0.9},  # decay rate
        # Scheduler - reduce_on_plateau
        "plateau_config": {
            "factor": 0.5,
            "patience": 10,
            "threshold": 0.0001,
            "min_lr": 1e-4,
        },
    },
    # Logger (see metal/logging/logger.py for descriptions)
    "logger": True,
    "logger_config": {
        "log_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
        # Report loss every this many log_units
        "log_every": 1,
        # Calculate and report metrics every this many log_units:
        #   None: default to log_every
        #   0: do not calculate or log metrics
        #   otherwise: must be a multiple of log_every
        "score_every": None,
        # The list of metrics (task/split/metric) to calculate; if empty, calculate
        # all metrics supported by all Scorers.
        "score_metrics": [],
        # The name of the split to run scoring on during training
        # To score over multiple splits, set valid_split=None and use score_metrics
        "valid_split": "valid",
        # The name of the split to run final evaluation on after training
        "test_split": "test",
        # If non-None, only calculate and report these metrics every `score_every`
        # units (this can include the names of built-in and user-defined metrics);
        # otherwise, include all metrics returned by task Scorers.
        # TODO: "metrics_filter": None,
        # TODO: "score_limit": None,  # Evaluate scorer on only this many examples
    },
    # LogWriter/Tensorboard (see metal/logging/writer.py for descriptions)
    "writer": None,  # [None, "json", "tensorboard"]
    "writer_config": {  # Log (or event) file stored at log_dir/run_dir/run_name
        "log_dir": f"{os.environ['METALHOME']}/logs",
        "run_dir": None,
        "run_name": None,
        "writer_metrics": [],  # May specify a subset of metrics in metrics_dict to be written
        "include_config": True,  # If True, include model config in log
    },
    # Checkpointer (see metal/logging/checkpointer.py for descriptions)
    "checkpoint": True,  # If True, checkpoint models when certain conditions are met
    "checkpoint_config": {
        # TODO: unify checkpoint=['every', 'best', 'final']; specify one strategy
        "checkpoint_every": 0,  # Save a model checkpoint every this many log_units
        # If checkpoint_best, also save the "best" model according to some metric
        # The "best" model will have the ['max', 'min'] value of checkpoint_metric
        # This metric must be produced by one of the task Scorer objects so it will be
        # available for lookup; assumes valid split unless appended with "train/"
        "checkpoint_best": False,
        # "checkpoint_final": False,  # Save a model checkpoint at the end of training
        "checkpoint_metric": "train/loss",
        "checkpoint_metric_mode": "min",
        "checkpoint_dir": f"{os.environ['METALHOME']}/checkpoints",
        "checkpoint_runway": 0,
    },
}


class MultitaskTrainer(object):
    """Driver for the MTL training process"""

    def __init__(self, config={}):
        self.config = recursive_merge_dicts(trainer_config, config)

    def train_model(self, model, tasks, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)
        self.task_names = [task.name for task in tasks]

        # Calculate epoch statistics
        # NOTE: Because we use SubsetSampler, one dataset may actually include two
        # splits, so we calculate approximate count size using batch_size * num_batches
        # examples_per_epoch = sum([len(t.data_loaders["train"].dataset) for t in tasks])
        batches_per_epoch = sum([len(t.data_loaders["train"]) for t in tasks])
        examples_per_epoch = sum(
            [
                len(t.data_loaders["train"]) * t.data_loaders["train"].batch_size
                for t in tasks
            ]
        )
        if self.config["verbose"]:
            print(f"Beginning train loop.")
            print(
                f"Expecting a total of _approximately_ {examples_per_epoch} examples "
                f"and {batches_per_epoch} batches per epoch from {len(tasks)} tasks."
            )

        # Set training components
        self._set_writer()
        self._set_logger(batches_per_epoch)
        self._set_checkpointer()
        self._set_optimizer(model)
        # TODO: Accept training matrix to give more fine-tuned training commands
        self._set_scheduler()

        # TODO: Restore the ability to resume training from a given epoch
        # That code goes here

        # Train the model
        # TODO: Allow other ways to train besides 1 epoch of all datasets
        model.train()
        # Dict metrics_hist contains the most recently recorded value of all metrics
        self.metrics_hist = {}
        self._reset_losses()
        for epoch in range(self.config["n_epochs"]):
            progress_bar = self.config["progress_bar"] and self.config["verbose"]
            t = tqdm(
                enumerate(self._get_train_batches(tasks)),
                total=batches_per_epoch,
                disable=(not progress_bar),
            )
            for batch_num, (task_name, batch) in t:
                # NOTE: actual batch_size may not equal config's target batch_size
                _, Y = batch
                batch_size = len(Y)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to calculate the average loss per example
                loss = model.calculate_loss(*batch, [task_name])[task_name]
                if torch.isnan(loss):
                    msg = "Loss is NaN. Consider reducing learning rate."
                    raise Exception(msg)

                # Backward pass to calculate gradients
                # Loss is an average loss per example
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
                self.running_losses[task_name] += loss.item() * batch_size
                self.running_examples[task_name] += batch_size

                # Calculate metrics, log, and checkpoint as necessary
                metrics_dict = self._execute_logging(model, tasks, batch_size)
                # Update most recently seen value of each metric
                self.metrics_hist.update(metrics_dict)

                # Apply learning rate scheduler
                # self._update_scheduler(batch_num, metrics_hist)

                # tqdm output
                if len(tasks) == 1:
                    t.set_postfix(loss=metrics_dict["train/loss"])
                else:
                    losses = {}
                    for key in metrics_dict:
                        if "loss" in key:
                            losses[key] = metrics_dict[key]
                    t.set_postfix(losses)

        model.eval()

        # Restore best model if applicable
        if self.checkpointer and self.checkpointer.checkpoint_best:
            self.checkpointer.load_best_model(model=model)
            path_to_best = os.path.join(
                self.checkpointer.checkpoint_dir, "best_model.pth"
            )
            path_to_logs = self.writer.log_subdir
            if os.path.isfile(path_to_best):
                copy2(path_to_best, path_to_logs)

        # Write log if applicable
        if self.writer:
            if self.writer.include_config:
                self.writer.add_config(self.config)
            self.writer.close()

        # Print final performance values
        if self.config["verbose"]:
            print("Finished Training")
            test_split = self.config["logger_config"]["test_split"]
            metrics_dict = self.calculate_metrics(model, tasks, split=test_split)
            pprint(metrics_dict)

    def _execute_logging(self, model, tasks, batch_size):
        model.eval()
        metrics_dict = {}
        metrics_dict.update(self.calculate_losses(tasks))
        # HACK: This exposes more of the logger than it should; abstract away
        self.logger.increment(batch_size)
        if self.logger.loss_time():
            self._reset_losses()
            self.logger.loss_ticks += 1
        if self.logger.metrics_time():
            # Unless valid_split is None, Scorers will only score on one split
            valid_split = self.config["logger_config"]["valid_split"]
            metrics_dict.update(self.calculate_metrics(model, tasks, split=valid_split))
            self.logger.loss_ticks = 0
        if self.logger.loss_time() or self.logger.metrics_time():
            # Log to screen/file/TensorBoard
            self.logger.log(metrics_dict)
            # Save best model if applicable
            self._checkpoint(model, metrics_dict)
        model.train()
        return metrics_dict

    def calculate_losses(self, tasks):
        """Calculate the average loss for each task since the last calculation

        If no examples of a certain task have been seen since the losses were reset,
        use the most recently reported value again (stored in metrics_hist).
        If the loss for a certain task has never been reported, report it as None.
        """
        metrics_dict = {}
        for task in tasks:
            if self.running_examples[task.name]:
                loss = self.running_losses[task.name] / self.running_examples[task.name]
            elif self.metrics_hist.get(f"{task.name}/loss"):
                loss = self.metrics_hist[f"{task.name}/loss"]
            else:
                loss = None
            metrics_dict[f"{task.name}/loss"] = loss
        # Report micro average of losses
        total_loss = sum(self.running_losses.values())
        total_examples = sum(self.running_examples.values())
        # TODO: Don't report task loss and "overall" loss if there is only one task?
        # But they may be planning on their named task loss being in the metrics_dict...
        metrics_dict["train/loss"] = total_loss / total_examples
        return metrics_dict

    def calculate_metrics(self, model, tasks, split=None):
        metrics_dict = {}
        score_metrics = self.config["logger_config"]["score_metrics"]
        for task in tasks:
            task_metrics = task.scorer.score(model, task, score_metrics, split)
            metrics_dict.update(task_metrics)
        return metrics_dict

    def _get_train_batches(self, tasks):
        """Yields batches one at a time sampled from tasks with some strategy"""
        # TODO: Allow more involved strategies for sampling from tasks
        # For now, just use proportional sampling
        # Length of a dataloader is the number of batches it contains
        approx_batch_counts = [len(t.data_loaders["train"]) for t in tasks]
        batch_assignments = []
        for task_idx, task in enumerate(tasks):
            batch_assignments.extend([task_idx] * approx_batch_counts[task_idx])
        random.shuffle(batch_assignments)
        train_loaders = [iter(t.data_loaders["train"]) for t in tasks]

        for task_idx in batch_assignments:
            yield (tasks[task_idx].name, next(train_loaders[task_idx]))

    def _checkpoint(self, model, metrics_dict):
        if self.checkpointer is None:
            return
        iteration = self.logger.unit_total
        self.checkpointer.checkpoint(
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

    def _set_logger(self, batches_per_epoch):
        # If not provided, set score_every to log_every
        logger_config = self.config["logger_config"]
        if logger_config["score_every"] is None:
            logger_config["score_every"] = logger_config["log_every"]
        self.logger = Logger(
            logger_config,
            batches_per_epoch,
            self.writer,
            verbose=self.config["verbose"],
        )

    def _set_checkpointer(self):
        if self.config["checkpoint"]:
            checkpoint_metric = self.config["checkpoint_config"]["checkpoint_metric"]
            if checkpoint_metric != "train/loss" and checkpoint_metric.count("/") != 2:
                msg = (
                    f"checkpoint_metric must be train/loss or have a full metric name "
                    f"(task/split/metric); you submitted: {checkpoint_metric}"
                )
                raise Exception(msg)
            score_metrics = self.config["logger_config"]["score_metrics"]
            if score_metrics and checkpoint_metric not in score_metrics:
                msg = (
                    "checkpoint_metric must be a metric in score_metrics if "
                    "score_metrics is not empty"
                )
                raise Exception(msg)
            self.checkpointer = Checkpointer(
                self.config["checkpoint_config"], verbose=self.config["verbose"]
            )
        else:
            self.checkpointer = None

    def _set_optimizer(self, model):
        optimizer_config = self.config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if opt == "sgd":
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

    def _set_scheduler(self):
        lr_scheduler = self.config["lr_scheduler"]
        if lr_scheduler is None:
            lr_scheduler = None
        else:
            lr_scheduler_config = self.config["lr_scheduler_config"]
            if lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, **lr_scheduler_config["exponential_config"]
                )
            elif lr_scheduler == "reduce_on_plateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **lr_scheduler_config["plateau_config"]
                )
            else:
                raise ValueError(
                    f"Did not recognize lr_scheduler option '{lr_scheduler}'"
                )
        self.lr_scheduler = lr_scheduler

    # def _update_scheduler(self, epoch, metrics_dict):
    #     """Optionally update the learning rate scheduler with each batch"""
    #     if self.lr_scheduler is not None:
    #         lr_scheduler_config = train_config["lr_scheduler_config"]
    #         if epoch + 1 >= lr_scheduler_config["lr_freeze"]:
    #             if train_config["lr_scheduler"] == "reduce_on_plateau":
    #                 checkpoint_config = train_config["checkpoint_config"]
    #                 metric_name = checkpoint_config["checkpoint_metric"]
    #                 score = metrics_dict.get(metric_name, None)
    #                 if score is not None:
    #                     self.lr_scheduler.step(score)
    #             else:
    #                 self.lr_scheduler.step()
