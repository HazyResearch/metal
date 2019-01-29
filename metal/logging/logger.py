import re
import time

from metal.logging.tensorboard import TensorBoardWriter
from metal.metrics import METRICS as standard_metrics, metric_score


class Logger(object):
    """
    TODO
    """

    def __init__(self, config, tb_config=None):
        # Strip split name from config keys
        self.config = config
        self.log_unit = self.config["log_unit"]
        self.unit_count = 0
        self.unit_total = 0
        self.timer = Timer() if self.log_unit == "seconds" else None
        self.log_count = 0  # Count how many times logging has occurred
        self.valid_every_X = int(
            self.config["log_valid_every"] / self.config["log_train_every"]
        )

        if tb_config is not None:
            self.tb_writer = TensorBoardWriter
        else:
            self.tb_writer = None

    def check(self, epoch, batch_size):
        """Returns True if the logging frequency has been met."""
        self.increment(epoch, batch_size)
        return self.unit_count >= self.config["log_train_every"]

    def increment(self, epoch, batch_size):
        if self.log_unit == "seconds":
            self.unit_count = self.timer.elapsed()
            self.unit_total = self.timer.total()
        elif self.log_unit == "examples":
            self.unit_count += batch_size
            self.unit_total += batch_size
        elif self.log_unit == "batches":
            self.unit_count += 1
            self.unit_total += 1
        elif self.log_unit == "epochs":
            # TODO: consider calculating epoch by example count instead?
            if epoch != self.unit_total:
                self.unit_count += 1
                self.unit_total += 1
        else:
            raise Exception(f"Unrecognized log_unit: {self.log_unit}")

    def calculate_metrics(
        self, model, train_loader, valid_loader, metrics_dict
    ):
        """TODO"""
        # Check whether or not it's time for validation as well
        self.log_count += 1
        log_valid = not (self.log_count % self.valid_every_X)

        # Calculate custom metrics
        if self.config["log_train_metrics_func"] is not None:
            custom_train_metrics = self.config["log_train_metrics_func"](
                model, train_loader
            )
            metrics_dict.update(custom_train_metrics)
        if self.config["log_valid_metrics_func"] is not None and log_valid:
            custom_valid_metrics = self.config["log_train_metrics_func"](
                model, valid_loader
            )
            metrics_dict.update(custom_valid_metrics)

        # Calculate standard metrics
        # Only calculate if there are other metrics besides default train/loss
        if len(self.config["log_train_metrics"]) > 1:
            target_metrics = self.config["log_train_metrics"]
            standard_train_metrics = self._calculate_standard_metrics(
                model, train_loader, target_metrics, metrics_dict, "train"
            )
            metrics_dict.update(standard_train_metrics)

        if log_valid and len(self.config["log_valid_metrics"]) > 0:
            target_metrics = self.config["log_valid_metrics"]
            standard_valid_metrics = self._calculate_standard_metrics(
                model, valid_loader, target_metrics, metrics_dict, "valid"
            )
            metrics_dict.update(standard_valid_metrics)

        return metrics_dict

    def _calculate_standard_metrics(
        self, model, data_loader, target_metrics, metrics_dict, split
    ):
        Y_preds, Y, Y_probs = model._get_predictions(
            data_loader, return_probs=True
        )
        standard_metrics_dict = {}
        for full_name in target_metrics:
            metric_name = full_name[full_name.find("/") + 1 :]
            if full_name in metrics_dict:
                # Don't overwrite train/loss or a clashing custom metric
                continue
            elif metric_name in standard_metrics:
                standard_metrics_dict[full_name] = metric_score(
                    Y, Y_preds, metric_name, probs=Y_probs
                )
            else:
                msg = (
                    f"Metric name '{metric_name}' could not be found in "
                    f"standard metrics or custom metrics for {split} split."
                )
                raise Exception(msg)
        return standard_metrics_dict

    def log(self, metrics_dict):
        """TODO"""
        if self.tb_writer:
            for metric, value in metrics_dict.items():
                if (
                    self.tb_writer.tb_metrics is None
                    or metric in self.tb_writer.tb_metrics
                ):
                    self.tb_writer.add_scalar(
                        self, metric, value, self.log_unit
                    )
        # TODO: add print_freq control (can log more frequently than we print?)
        self.print(metrics_dict)
        self.reset()

    def print(self, metrics_dict):
        score_strings = []
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                score_strings.append(f"{metric}={value:0.3f}")
            else:
                score_strings.append(f"{metric}={value}")
        print(
            f"[{self.unit_total} {self.log_unit}]: {', '.join(score_strings)}"
        )

    def reset(self):
        self.unit_count = 0


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Initialize timer."""
        self.reset()

    def reset(self):
        """Reset timer to zero."""
        self.start = time.time()
        self.click = self.start

    def elapsed(self):
        """Get time elapsed since last click (elapsed() or reset())."""
        elapsed = time.time() - self.click
        self.click = time.time()
        return elapsed

    def total(self):
        return time.time() - self.start
