import time

from metal.logging.tensorboard import TensorBoardWriter


class Logger(object):
    """

    """

    def __init__(self, config):
        self.config = config
        self.log_unit = config["log_unit"]
        self.unit_count = 0
        self.unit_total = 0
        self.timer = Timer()
        self.epoch = 0

        if config["tensorboard"]:
            self.tb_writer = TensorBoardWriter
        else:
            self.tb_writer = None

    def check(self, epoch, batch_size):
        """Returns True if the logging/checkpoint frequency has been met."""
        self.increment(epoch, batch_size)
        return self.log_unit >= self.log_freq

    def increment(self, epoch, batch_size):
        if self.log_unit == "seconds":
            self.unit_total = self.timer.total()
            self.unit_count = self.timer.elapsed()
        elif self.log_unit == "examples":
            self.unit_count += batch_size
            self.unit_total += batch_size
        elif self.log_unit == "batches":
            self.unit_count += 1
            self.unit_total += 1
        elif self.log_unit == "epochs":
            if epoch != self.epoch:
                self.epoch = epoch
                self.unit_count += 1
                self.unit_total += 1
        else:
            raise Exception(f"Unrecognized log_unit: {self.log_unit}")

    def record(self, metrics):
        for metric, value in metrics.items():
            if self.tb_writer:
                self.tb_writer.add_scalar(self, metric, value, self.log_unit)
        self.print_update(metrics)
        self.reset()

    def print_update(self, metrics):
        scores = ""
        for metric, value in metrics.items():
            if isinstance(value, float):
                scores += f"{metric}={value:0.3f}"
            else:
                scores += f"{metric}={value}"
        print(f"[{self.unit_total} {self.log_units}]: {scores}")

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
