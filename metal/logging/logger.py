import time
from collections import defaultdict

from metal.metrics import METRICS as standard_metric_names, metric_score


class Logger(object):
    """Tracks when it is time to calculate train/valid metrics and logs them"""

    def __init__(self, config, writer={}, epoch_size=None, verbose=True):
        # Strip split name from config keys
        self.config = config
        self.writer = writer
        self.verbose = verbose
        self.log_unit = self.config["log_unit"]
        self.epoch_size = epoch_size
        self.example_count = 0
        self.example_total = 0
        self.unit_count = 0
        self.unit_total = 0
        self.log_count = 0  # Count how many times logging has occurred
        self.valid_every_X = int(
            self.config["log_valid_every"] / self.config["log_train_every"]
        )

        # Specific to log_unit == "seconds"
        self.timer = Timer() if self.log_unit == "seconds" else None

        # Normalize all target metric names to include split prefix
        self.log_train_metrics = [
            self.add_split_prefix(m, "train")
            for m in self.config["log_train_metrics"]
        ]
        self.log_valid_metrics = [
            self.add_split_prefix(m, "valid")
            for m in self.config["log_valid_metrics"]
        ]

        assert isinstance(self.config["log_train_every"], int)
        assert isinstance(self.config["log_valid_every"], int)
        if (
            self.config["log_valid_every"] < self.config["log_train_every"]
            or self.config["log_valid_every"] % self.config["log_train_every"]
        ):
            raise Exception(
                f"Setting log_valid_every ({self.config['log_valid_every']}) "
                f"must be a multiple of log_train_every "
                f"({self.config['log_train_every']})."
            )

    def check(self, batch_size):
        """Returns True if the logging frequency has been met."""
        self.increment(batch_size)
        return self.unit_count >= self.config["log_train_every"]

    def increment(self, batch_size):
        """Update the total and relative unit counts"""
        self.example_count += batch_size
        self.example_total += batch_size
        if self.log_unit == "seconds":
            self.unit_count = int(self.timer.elapsed())
            self.unit_total = int(self.timer.total_elapsed())
        elif self.log_unit == "examples":
            self.unit_count = self.example_count
            self.unit_total = self.example_total
        elif self.log_unit == "batches":
            self.unit_count += 1
            self.unit_total += 1
        elif self.log_unit == "epochs":
            # Track epoch by example count because otherwise we only know when
            # a new epoch starts, not when an epoch ends
            if self.example_count >= self.epoch_size:
                self.unit_count += 1
                self.unit_total += 1
        else:
            raise Exception(f"Unrecognized log_unit: {self.log_unit}")

    def calculate_metrics(
        self, model, train_loader, valid_loader, metrics_dict
    ):
        """Add standard and custom metrics to metrics_dict"""
        # Check whether or not it's time for validation as well
        self.log_count += 1
        log_valid = valid_loader is not None and not (
            self.log_count % self.valid_every_X
        )

        metrics_dict = {}

        # Calculate custom metrics
        if self.config["log_train_metrics_func"] is not None:
            func = self.config["log_train_metrics_func"]
            metrics_dict = self._calculate_custom_metrics(
                model, train_loader, func, metrics_dict, split="train"
            )
        if self.config["log_valid_metrics_func"] is not None and log_valid:
            func = self.config["log_valid_metrics_func"]
            metrics_dict = self._calculate_custom_metrics(
                model, train_loader, func, metrics_dict, split="valid"
            )

        # Calculate standard metrics
        metrics_dict = self._calculate_standard_metrics(
            model, train_loader, self.log_train_metrics, metrics_dict, "train"
        )

        if log_valid:
            metrics_dict = self._calculate_standard_metrics(
                model,
                valid_loader,
                self.log_valid_metrics,
                metrics_dict,
                "valid",
            )

        return metrics_dict

    def _calculate_custom_metrics(
        self, model, data_loader, func, metrics_dict, split
    ):
        custom_metrics = func(model, data_loader)
        # Normalize all custom metrics to include split prefix
        for metric, value in custom_metrics.items():
            metric = self.add_split_prefix(metric, split)
            metrics_dict[metric] = value
        return metrics_dict

    def _calculate_standard_metrics(
        self, model, data_loader, target_metrics, metrics_dict, split
    ):
        standard_metrics_list = []
        for split_metric in target_metrics:
            metric = self.remove_split_prefix(split_metric)
            if metric in standard_metric_names:
                standard_metrics_list.append(metric)

        # Only calculate predictions if at least one standard metric requires it
        if standard_metrics_list:
            Y_preds, Y, Y_probs = model._get_predictions(
                data_loader, return_probs=True
            )
            for metric in standard_metrics_list:
                score = metric_score(Y, Y_preds, metric, probs=Y_probs)
                metrics_dict[self.add_split_prefix(metric, split)] = score
        return metrics_dict

    @staticmethod
    def add_split_prefix(metric, split):
        """Prepend "{split}/" to the metric name if it is not already present"""
        if not metric.startswith(f"{split}/"):
            metric = f"{split}/{metric}"
        return metric

    @staticmethod
    def remove_split_prefix(metric):
        """Remove "{split}/" from begininng of metric name if it is present"""
        split, metric = metric.split("/", 1)
        return metric

    def log(self, metrics_dict):
        """Print calculated metrics and optionally write to file (json/tb)"""
        if self.writer:
            self.write_to_file(metrics_dict)

        if self.verbose:
            self.print_to_screen(metrics_dict)
        self.reset()

    def print_to_screen(self, metrics_dict):
        """Print all metrics in metrics_dict to screen"""
        score_strings = defaultdict(list)
        for split_metric, value in metrics_dict.items():
            split, metric = split_metric.split("/", 1)

            if isinstance(value, float):
                score_strings[split].append(f"{metric}={value:0.3f}")
            else:
                score_strings.append(f"{metric}={value}")

        header = f"{self.unit_total} {self.log_unit[:3]}"
        if self.log_unit != "epochs":
            epochs = self.example_total / self.epoch_size
            header += f" ({epochs:0.2f} epo)"
        string = f"[{header}]:"

        if score_strings["train"]:
            train_scores = f"{', '.join(score_strings['train'])}"
            string += f" TRAIN: [{train_scores}]"
        if score_strings["valid"]:
            valid_scores = f"{', '.join(score_strings['valid'])}"
            string += f" VALID: [{valid_scores}]"
        print(string)

    def write_to_file(self, metrics_dict):
        for metric, value in metrics_dict.items():
            self.writer.add_scalar(metric, value, self.unit_total)

    def reset(self):
        self.unit_count = 0
        self.example_count = 0
        if self.timer is not None:
            self.timer.update()


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Initialize timer"""
        self.reset()

    def reset(self):
        """Reset timer, completely obliterating history"""
        self.start = time.time()
        self.update()

    def update(self):
        """Update timer with most recent click point"""
        self.click = time.time()

    def elapsed(self):
        """Get time elapsed since last recorded click"""
        elapsed = time.time() - self.click
        return elapsed

    def total_elapsed(self):
        return time.time() - self.start
