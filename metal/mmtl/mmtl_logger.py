import time
from collections import defaultdict


class Logger(object):
    """Tracks when it is time to calculate train/valid metrics and logs them"""

    def __init__(self, config, batches_per_epoch, writer={}, verbose=True):
        # Strip split name from config keys
        self.config = config
        self.writer = writer
        self.verbose = verbose
        self.log_unit = self.config["log_unit"]
        self.batches_per_epoch = batches_per_epoch
        self.example_count = 0
        self.example_total = 0
        self.batch_count = 0
        self.batch_total = 0
        self.unit_count = 0
        self.unit_total = 0
        self.loss_ticks = 0  # Count how many times loss logging has occurred

        # Specific to log_unit == "seconds"
        self.timer = Timer() if self.log_unit == "seconds" else None

        # Calculate how many log_train steps to take per log_valid steps
        self.valid_every_X = self._calculate_valid_frequency()

    def increment(self, batch_size):
        """Update the total and relative unit counts"""
        self.example_count += batch_size
        self.example_total += batch_size
        self.batch_count += 1
        self.batch_total += 1
        if self.log_unit == "seconds":
            self.unit_count = int(self.timer.elapsed())
            self.unit_total = int(self.timer.total_elapsed())
        elif self.log_unit == "examples":
            self.unit_count = self.example_count
            self.unit_total = self.example_total
        elif self.log_unit == "batches":
            self.unit_count = self.batch_count
            self.unit_total = self.batch_total
        elif self.log_unit == "epochs":
            # Track epoch by example count rather than epoch number because otherwise
            # we only know when a new epoch starts, not when an epoch ends
            self.unit_count = self.batch_count / self.batches_per_epoch
            self.unit_total = self.batch_total / self.batches_per_epoch
        else:
            raise Exception(f"Unrecognized log_unit: {self.log_unit}")

    def loss_time(self):
        """Returns True if it is time to calculate and report loss"""
        is_time = self.unit_count >= self.config["log_every"]
        if is_time:
            self.loss_ticks += 1
        return is_time

    def metrics_time(self):
        """Returns True if it is time to calculate and report loss

        TODO: Currently, score_every is a multiple of log_every so there is
        only one set of counters to reset. These two could be made independent by
        creating a separate counter set for loss_time and metrics_time.
        """
        is_time = self.loss_ticks == self.valid_every_X
        if is_time:
            self.loss_ticks = 0
        return is_time

    def _calculate_valid_frequency(self):
        if self.config["score_every"]:
            # Do integer check on ratio instead of using mod due to float issues:
            # e.g., 1.0 % 0.1 == 0.0999999995 for some reason
            ratio = self.config["score_every"] / self.config["log_every"]
            if self.config["score_every"] < self.config["log_every"] or ratio != int(
                ratio
            ):
                msg = (
                    f"Parameter `score_every` "
                    f"({self.config['score_every']}) must be a multiple of "
                    f"`log_every` ({self.config['log_every']})."
                )
                raise Exception(msg)
            return int(ratio)
        else:
            return 0

    def log(self, metrics_dict):
        """Print calculated metrics and optionally write to file (json/tb)"""
        if self.writer:
            self.write_to_file(metrics_dict)

        if self.verbose:
            self.print_to_screen(metrics_dict)
        self.reset()

    def print_to_screen(self, metrics_dict):
        """Print all metrics in metrics_dict to screen"""
        score_strings_by_task = defaultdict(list)
        for full_metric_name, value in metrics_dict.items():
            task_name, metric_name = full_metric_name.split("/", maxsplit=1)
            if isinstance(value, float):
                score_strings_by_task[task_name].append(f"{metric_name}={value:0.2e}")
            else:
                score_strings_by_task[task_name].append(f"{metric_name}={value}")

        if self.log_unit == "epochs":
            if int(self.unit_total) == self.unit_total:
                header = f"{self.unit_total} {self.log_unit[:3]}"
            else:
                header = f"{self.unit_total:0.2f} {self.log_unit[:3]}"
        else:
            epochs = self.batch_total / self.batches_per_epoch
            header = f" ({epochs:0.2f} epo)"
        string = f"[{header}]:"

        for task, score_strings in score_strings_by_task.items():
            concatenated_scores = f"{', '.join(score_strings)}"
            string += f" {task}:[{concatenated_scores}]"
        print(string)

    def write_to_file(self, metrics_dict):
        for metric, value in metrics_dict.items():
            if self.log_unit == "epochs":
                # Use batches b/c Tensorboard cannot handle non-integer iteration #s
                self.writer.add_scalar(metric, value, self.batch_total)
            else:
                self.writer.add_scalar(metric, value, self.unit_total)

    def reset(self):
        self.unit_count = 0
        self.example_count = 0
        self.batch_count = 0
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
