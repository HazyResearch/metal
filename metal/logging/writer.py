import copy
import json
import os
from collections import defaultdict
from subprocess import check_output
from time import strftime

from metal.utils import recursive_transform


class LogWriter(object):
    """Class for writing simple JSON logs at end of runs, with interface for
    storing per-iter data as well.

    Config contains:
        log_dir: (str) The path to the base log directory, or defaults to
            current working directory.
        run_dir: (str) The name of the sub-directory, or defaults to the date,
            strftime("%Y_%m_%d").
        run_name: (str) The name of the run + the time, or defaults to the time,
            strftime("%H_%M_%S).
        writer_metrics: (list) An optional whitelist of metrics to write,
            ignoring all others. (If None, write all available metrics).

        Log is saved to 'log_dir/run_dir/{run_name}_H_M_S.json'
    """

    def __init__(
        self,
        log_dir="logs",
        run_dir=None,
        run_name=None,
        writer_metrics=None,
        include_config=True,
    ):
        start_date = strftime("%Y_%m_%d")
        start_time = strftime("%H_%M_%S")

        # Set logging subdirectory + make sure exists
        log_dir = log_dir or os.getcwd()
        run_dir = run_dir or start_date
        if run_name is not None:
            run_name = f"{run_name}_{start_time}"
        else:
            run_name = start_time
        self.log_subdir = os.path.join(log_dir, run_dir, run_name)
        if not os.path.exists(self.log_subdir):
            os.makedirs(self.log_subdir)

        # Set JSON log path
        self.log_path = os.path.join(self.log_subdir, f"{run_name}.json")

        # Save other settings
        self.writer_metrics = writer_metrics
        self.include_config = include_config

        # Initialize log
        # Note we have a separate section for during-run metrics
        commit = check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
        self.log_dict = {
            "start_date": start_date,
            "start_time": start_time,
            "commit": str(commit),
            "config": None,
            "run_log": defaultdict(list),
        }

    def add_config(self, config):
        config = copy.deepcopy(config)
        is_func = lambda x: callable(x)
        replace_with_name = lambda f: f.__name__
        config = recursive_transform(config, is_func, replace_with_name)
        self.log_dict["config"] = config

    def add_scalar(self, name, val, i):
        # Note: Does not handle deduplication of (name, val) entries w same i
        if not self.writer_metrics or name in self.write_metrics:
            self.log_dict["run_log"][name].append((i, val))
            return True
        else:
            return False

    def write(self):
        """Dump JSON to file"""
        with open(self.log_path, "w") as f:
            json.dump(self.log_dict, f, indent=1)

    def close(self):
        self.write()
