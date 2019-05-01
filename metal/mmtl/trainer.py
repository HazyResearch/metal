import copy
import os
import warnings
from collections import defaultdict
from pprint import pprint
from shutil import copy2

import dill
import numpy as np
import torch
import torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from metal.logging import Checkpointer, LogWriter, TensorBoardWriter
from metal.mmtl.mmtl_logger import Logger  # NOTE: we use special MMTL logger
from metal.mmtl.task_scheduler import ProportionalScheduler
from metal.utils import recursive_merge_dicts, recursive_transform, set_seed

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


trainer_defaults = {
    "verbose": True,
    "seed": None,
    # Commit hash
    "commit_hash": None,
    "ami": None,  # ami id for aws
    # Display
    "progress_bar": False,
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
        # A list of functions that operate on a metrics_dict and return a dict with
        # additional metrics (e.g., aggregated metrics)
        "aggregate_metric_fns": [],
        # Run scorers over a maximum of this many examples if > 0.
        "max_valid_examples": 0,
        # The name of the split to run scoring on during training
        # To score over multiple splits, set valid_split=None and use task_metrics
        "valid_split": "valid",
        # The name of the split to run final evaluation on after training
        "test_split": None,  # If None, calculate final metrics over all splits
        # If non-None, only calculate and report these metrics every `score_every`
        # units (this can include the names of built-in and user-defined metrics);
        # otherwise, include all metrics returned by task Scorers.
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
        "checkpoint_metric": "model/train/all/loss",
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
        self.config = recursive_merge_dicts(trainer_defaults, kwargs, misses="insert")

        # Set random seeds
        if self.config["seed"] is None:
            self.config["seed"] = np.random.randint(1e6)
        set_seed(self.config["seed"])

    def train_model(self, model, payloads, **kwargs):
        # NOTE: misses="insert" so we can log extra metadata (e.g. num_parameters)
        # and eventually write to disk.
        self.config = recursive_merge_dicts(self.config, kwargs, misses="insert")

        self.task_names = [task_name for task_name in model.task_map]
        self.payload_names = [payload.name for payload in payloads]
        train_payloads = [p for p in payloads if p.split == "train"]
        if not train_payloads:
            msg = "At least one payload must have property payload.split=='train'"
            raise Exception(msg)

        # Calculate epoch statistics
        # NOTE: We calculate approximate count size using batch_size * num_batches
        self.batches_per_epoch = sum([len(p.data_loader) for p in train_payloads])
        self.examples_per_epoch = sum(
            [len(p.data_loader) * p.data_loader.batch_size for p in train_payloads]
        )
        if self.config["verbose"]:
            print(f"Beginning train loop.")
            print(
                f"Expecting approximately {self.examples_per_epoch} examples total "
                f"and {self.batches_per_epoch} batches per epoch from "
                f"{len(train_payloads)} payload(s) in the train split."
            )

        # Check inputs
        self._check_metrics()

        # Set training components
        self._set_writer()
        self._set_logger()
        self._set_checkpointer(model)
        self._set_optimizer(model)
        self._set_lr_scheduler(model)  # TODO: Support more detailed training schedules
        self._set_task_scheduler(model, payloads)

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
                enumerate(self.task_scheduler.get_batches(payloads, "train")),
                total=self.batches_per_epoch,
                disable=(not progress_bar),
            )
            for batch_num, (batch, payload_name, labels_to_tasks) in t:
                # NOTE: actual batch_size may not equal config's target batch_size,
                # for example due to orphan batches. We base batch size off of Y instead
                # of X because we know Y will contain tensors, whereas X can be of any
                # format the input_module accepts, including tuples of tensors, etc.
                _, Ys = batch
                batch_size = len(next(iter(Ys.values())))
                batch_id = epoch * self.batches_per_epoch + batch_num

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to calculate the average loss per example by task
                # Counts stores the number of examples in each batch with labels by task
                loss_dict, count_dict = model.calculate_loss(
                    *batch, payload_name, labels_to_tasks
                )

                # NOTE: If there were no "active" examples, loss_dict is empty
                # Skip additional loss-based computation at this point
                if not loss_dict:
                    continue

                loss = sum(loss_dict.values())
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
                for loss_name in loss_dict:
                    if count_dict[loss_name]:
                        self.running_losses[loss_name] += (
                            loss_dict[loss_name].item() * count_dict[loss_name]
                        )
                        self.running_examples[loss_name] += count_dict[loss_name]

                # Calculate metrics, log, and checkpoint as necessary
                metrics_dict = self._execute_logging(model, payloads, batch_size)
                # Confirm metrics being produced are in proper format
                if epoch == 0 and batch_num == 0:
                    self._validate_metrics_dict(metrics_dict)

                # Apply learning rate scheduler
                self._update_lr_scheduler(model, batch_id)

                # tqdm output
                if len(model.task_map) == 1:
                    t.set_postfix(loss=metrics_dict["model/train/all/loss"])
                else:
                    losses = {}
                    for key, val in metrics_dict.items():
                        if "loss" in key:
                            losses[key] = val
                    t.set_postfix(losses)

        model.eval()
        # Restore best model if applicable
        if self.checkpointer and self.checkpointer.checkpoint_best:
            # First do a final checkpoint at the end of training
            metrics_dict = self._execute_logging(
                model, payloads, batch_size, force_log=True
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
        # Calculate metrics for all splits if test_split=None
        test_split = self.config["metrics_config"]["test_split"]
        metrics_dict = self.calculate_metrics(model, payloads, split=test_split)
        if self.config["verbose"]:
            pprint(metrics_dict)

        # Clean up checkpoints
        if self.checkpointer and self.config["checkpoint_cleanup"]:
            print("Cleaning checkpoints")
            self.checkpointer.clean_up()

        # Write log if applicable
        if self.writer:
            # convert from numpy to python float
            metrics_dict = recursive_transform(
                metrics_dict, lambda x: type(x).__module__ == np.__name__, float
            )

            self.writer.write_metrics(metrics_dict)
            self.writer.write_log()
            self.writer.close()

            # pickle and save the full model
            full_model_path = os.path.join(self.writer.log_subdir, "model.pkl")
            torch.save(model, full_model_path, pickle_module=dill)
            print(f"Full model saved at {full_model_path}")

        return metrics_dict

    def _execute_logging(self, model, payloads, batch_size, force_log=False):
        model.eval()
        metrics_dict = {}
        metrics_dict.update(self.aggregate_losses())
        self.logger.increment(batch_size)

        do_log = False
        if self.logger.loss_time():
            self._reset_losses()
            do_log = True
        if self.logger.metrics_time() or force_log:
            # Unless valid_split is None, Scorers will only score on one split
            valid_split = self.config["metrics_config"]["valid_split"]
            metrics_dict.update(
                self.calculate_metrics(model, payloads, split=valid_split)
            )
            do_log = True
        if do_log or force_log:
            # Log to screen/file/TensorBoard
            self.logger.log(metrics_dict)
            # Save best model if applicable
            self._checkpoint(model, metrics_dict)

        self.metrics_hist.update(metrics_dict)
        model.train()

        return metrics_dict

    def aggregate_losses(self):
        """Calculate the average loss for each task since the last calculation

        If no examples of a certain task have been seen since the losses were reset,
        use the most recently reported value again (stored in metrics_hist).
        If the loss for a certain task has never been reported, report it as None.
        """
        metrics_dict = {}
        for loss_name in self.running_losses:
            if self.running_examples[loss_name]:
                loss = self.running_losses[loss_name] / self.running_examples[loss_name]
            elif self.metrics_hist.get(loss_name):
                loss = self.metrics_hist[loss_name]
            else:
                loss = None
            metrics_dict[loss_name] = loss
        # Report micro average of losses
        total_loss = sum(self.running_losses.values())
        total_examples = sum(self.running_examples.values())
        if total_examples > 0:
            metrics_dict["model/train/all/loss"] = total_loss / total_examples
        # Log learning rate
        if self.config["logger_config"]["log_lr"]:
            # For now just report one global lr; eventually support lr groups
            metrics_dict[f"model/train/all/lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics_dict

    def calculate_metrics(self, model, payloads, split=None):
        metrics_dict = {}
        # Update metrics_hist after task_metrics so aggregates metrics have access to
        # most recently calculated numbers
        metrics_dict.update(self.calculate_task_metrics(model, payloads, split))
        self.metrics_hist.update(metrics_dict)
        metrics_dict.update(self.calculate_aggregate_metrics())
        self.metrics_hist.update(metrics_dict)
        return metrics_dict

    def calculate_task_metrics(self, model, payloads, split=None):
        metrics_dict = {}
        max_examples = self.config["metrics_config"]["max_valid_examples"]
        task_metrics = self.config["metrics_config"]["task_metrics"]

        # Losses are handled specially; we drop them from task_metrics
        target_metrics = [metric for metric in task_metrics if "/loss" not in metric]

        # Calculate metrics from Scorers
        for payload in payloads:
            if split and payload.split != split:
                continue
            payload_metrics_dict = model.score(
                payload, target_metrics, max_examples=max_examples
            )
            metrics_dict.update(payload_metrics_dict)
        return metrics_dict

    def calculate_aggregate_metrics(self):
        aggregate_metric_fns = self.config["metrics_config"]["aggregate_metric_fns"]
        aggregate_metrics = {}
        for metric_fn in aggregate_metric_fns:
            aggregate_metrics.update(metric_fn(self.metrics_hist))
        return aggregate_metrics

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

    def _set_checkpointer(self, model):
        if (
            self.config["checkpoint"]
            or self.config["lr_scheduler"] == "reduce_on_plateau"
        ):
            self._validate_checkpoint_metric(model)
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

    def _set_optimizer(self, model):
        optimizer_config = self.config["optimizer_config"]
        opt = optimizer_config["optimizer"]

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # Special optimizer for fp16
        if model.config["fp16"]:
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

                    # TODO: This may not be necessary; confirm if it is
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

    def _set_task_scheduler(self, model, payloads):
        if self.config["task_scheduler"] == "proportional":
            self.task_scheduler = ProportionalScheduler(model, payloads, "train")
        else:
            raise NotImplementedError

    def _validate_checkpoint_metric(self, model):
        # Confirm that checkpoint_metric is a metric that will be available
        checkpoint_config = self.config["checkpoint_config"]
        checkpoint_metric = checkpoint_config["checkpoint_metric"]
        if checkpoint_metric.startswith("model"):
            metric_name = checkpoint_metric.split("/")[-1]
            aggregate_metric_fns = self.config["metrics_config"]["aggregate_metric_fns"]
            aggregate_metric_names = [
                getattr(metric_fn, "__name__") for metric_fn in aggregate_metric_fns
            ]
            if metric_name != "loss" and metric_name not in aggregate_metric_names:
                msg = (
                    f"The checkpoint_metric you specified ('{checkpoint_metric}') is "
                    f"not currently supported."
                )
                raise Exception(msg)
        else:
            if checkpoint_metric.count("/") != 3:
                msg = (
                    f"checkpoint_metric must have a full metric name "
                    f"(task/payload/split/metric); you submitted: {checkpoint_metric}"
                )
                raise Exception(msg)

            task_name, payload_name, label_name, metric = checkpoint_metric.split("/")
            try:
                task = model.task_map[task_name]
            except KeyError:
                msg = (
                    f"The task for your specified checkpoint_metric "
                    f"({checkpoint_metric}) was not found in the list of "
                    f"submitted tasks: {[t for t in self.task_names]}."
                )
                raise Exception(msg)

            if payload_name not in self.payload_names:
                msg = (
                    f"The payload for your specified checkpoint_metric "
                    f"({checkpoint_metric}) was not found in the list of "
                    f"submitted payloads: {self.payload_names}."
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

    def _validate_metrics_dict(self, metrics_dict):
        for full_name in metrics_dict:
            if len(full_name.split("/")) != 4:
                msg = (
                    f"Metric should have form task/payload/label_name/metric, not: "
                    f"{full_name}"
                )
                raise Exception(msg)

    def _check_metrics(self):
        assert isinstance(self.config["metrics_config"]["task_metrics"], list)
        assert isinstance(self.config["metrics_config"]["aggregate_metric_fns"], list)
