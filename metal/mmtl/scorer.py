from collections import defaultdict

import numpy as np

from metal.metrics import METRICS as STANDARD_METRICS, metric_score
from metal.mmtl.utils import utils


"""
Scorer class which evaluates metrics given a task and model.

#TODO: Update this after redesign
Example usage 1: standalone
scorer = Scorer(standard_metrics=["accuracy"])
model = MetalModel(tasks)
trainer = Trainer()
trainer.train_model(model, tasks)
scorer.score(model, tasks[0])

Example usage 2: integrated into task
scorer = Scorer(standard_metrics=["accuracy"])
task1 = Task(...,scorer=scorer)
...
taskn = ...
tasks = [task1, ..., taskn]
model = MetalModel(tasks)

----------------------------------------------------------------------------------------
NOTE: There are some issues with this design.
Currently, if a user includes custom test metrics in their Scorer, they'll be executed
during training with score_every frequency. And if they want to test just at the end,
they need to make a new scorer (so that now it _does_ have test metrics).

Instead, a scorer should define metrics that are defined and can be calculated over an
arbitrary DataLoader belonging to this task. We (maybe?) don't care what the name of
the task is (so we can support people having differently named splits than us)?
Or maybe we only require that "train" be used for the train split?

Which metrics to print out of all the ones supported by the scorers?
Currently we default to all on unless the user specified some; then we only show those
We could also default to all off (and the user specifies what they want to see)
"""


class Scorer(object):
    """
    # TODO: clarify this even further
    DESIGN:
    - A Scorer is a bundle of metrics; it defines what metrics _can_ be calculated on a
    given task (may be able to use smart defaults based on the Task subclass; e.g.,
    classification comes with many nicely defined).
        - custom functions come with a list of names of the metrics they produce (with
        error checking to confirm they don't produce more than that)
    - A Scorer is applied to a task with a list of metrics to return
    - All metrics in a scorer produce simple metric name only
        - score() has task and split so it can create the full metric name
        - The metrics dict only ever contains full metric names
    - The user submits a list of score_metrics and test_metrics to the trainer
        - test_metrics defaults to score_metrics
        - when score() is called, only score_metrics are calculated and returned
        - score_metrics and test_metrics contain only full metric names
            - these can use arbitrary split names
        - [later] we can optionally allow regexes instead of explicit names

    Args:
        data_loader: DataLoader on which to calculate metrics.
        standard_metrics: List of strings of standard metrics for which to evaluate.
            By default, calculate on valid split. Optionally, prepend metric with
            "train/" to calculate on train split instead.
        custom_metric_funcs: Dict of the form:

            {metric_fn1: ["metric1a", ..., "metric1z"],
             metric_fn2: ["metric2a", ..., "metric2z]}

            where metric_fn is a function of the form:

            metric_fn1(Y, Y_preds, probs=Y_probs) -> {metric1a: value1, ..., metric1z: valueN}

            metric_names will automatically have task and split prefixes added by the
            Scorer.
    """

    def __init__(self, standard_metrics=["accuracy"], custom_metric_funcs=[]):
        self.standard_metrics = standard_metrics
        for metric_name in standard_metrics:
            if "/" in metric_name:
                msg = f"Standard metrics at Scorer initialization time must not include task or split name, but you submitted: {metric_name}"
                raise Exception(msg)
            if metric_name not in STANDARD_METRICS:
                msg = f"Requested standard metric {metric_name} could not be found in metrics.py."
                raise Exception(msg)

        # Create a map from custom metric names to the function that creates them
        self.custom_metric_funcs = custom_metric_funcs
        self.custom_metric_map = {}
        for metric_fn, metric_names in custom_metric_funcs:
            assert isinstance(metric_names, list)
            for metric_name in metric_names:
                if "/" in metric_name:
                    msg = f"Metric produced by custom_metric_funcs must not include task or split name, but you submitted: {metric_name}."
                    raise Exception(msg)
                self.custom_metric_map[metric_name] = metric_fn

    def score(self, model, task, target_metrics):
        """
        Calculates and returns a metrics_dict for a given task

        Args:
            model: MetalModel to score (data will be moved to the same device as model)
            task: Task to calculate metrics on
            target_metrics: List of full metric names (task/split/metric) or
                (split/metric) to be calculated.
        Returns:
            a metrics_dict object of the form:
                {task/split/metric1 : score1, ...., task/split/metricN: score N}
        """
        metrics_dict = {}

        # Identify splits and functions required to collect the target_metrics
        target_standard_metrics = defaultdict(set)
        target_custom_metrics = defaultdict(set)
        for full_metric_name in target_metrics:
            target_task, split, metric = self._split_full_metric(full_metric_name)
            if target_task != task.name:
                continue
            if metric in self.standard_metrics:
                target_standard_metrics[split].add(metric)
            elif metric in self.custom_metric_map:
                target_custom_metrics[split].add(metric)
            else:
                msg = f"Target metric {full_metric_name} is not supported by the Scorer for task {task.name}"
                raise Exception(msg)

        for split in task.data_loaders:
            if not (target_standard_metrics[split] or not target_custom_metrics[split]):
                continue

            # Calculate probs and preds in batches from data_loader
            Y_preds, Y, Y_probs = [], [], []
            for batch_num, batch in enumerate(task.data_loaders[split]):
                Xb, Yb = batch
                Y.append(Yb)

                Yb_probs = model.calculate_output(Xb, [task.name])[task.name]
                Y_probs.append(Yb_probs)

            # Stack batches
            Y = utils.stack_batches(Y)
            Y_probs = utils.stack_batches(Y_probs)
            Y_preds = utils.break_ties(Y_probs, "random").astype(np.int)

            # From the labels and predictions calculate metrics
            for metric in target_standard_metrics[split]:
                score = metric_score(Y, Y_preds, metric, probs=Y_probs)
                full_metric_name = self._join_full_metric(task.name, split, metric)
                metrics_dict[full_metric_name] = score

            # Calculate custom fns
            for custom_metric_func in target_custom_metrics[split]:
                custom_metric_dict = custom_metric_func(Y, Y_preds, probs=Y_probs)
                for metric, score in custom_metric_dict:
                    if metric not in self.custom_metric_map:
                        expected_metrics = [
                            metrics
                            for func, metrics in self.custom_metric_funcs
                            if func == custom_metric_func
                        ][0]
                        msg = f"Custom metric func {custom_metric_func} yielded a metric `{metric}` that was not included in the list of corresponding metrics: {expected_metrics}"
                        raise Exception(msg)
                    full_metric_name = self._join_full_metric(task.name, split, metric)
                    metrics_dict[full_metric_name] = score

        return metrics_dict

    def _split_full_metric(self, full_metric):
        """Splits a full metric name (task/split/name or split/name) into its pieces"""
        pieces = full_metric.split("/")
        if len(pieces) == 2:
            split, name = pieces
            task = None
        elif len(pieces) == 3:
            task, split, name = pieces
        else:
            msg = f"Required a full metric name (task/split/name or split/name) but instead received: {full_metric}"
            raise Exception(msg)
        return task, split, name

    def _join_full_metric(self, task, split, metric):
        """Creates a full  metric name from its component pieces"""
        return f"{task}/{split}/{metric}"
