import numpy as np

from metal.metrics import metric_score
from metal.mmtl.utils import utils


"""
Scorer class which evaluates metrics given a task and model.

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
    def __init__(
        self,
        standard_metrics=["accuracy"],
        custom_valid_funcs=[],
        custom_train_funcs=[],
    ):
        """
        Creates a scorer object.

        data_loader: DataLoader on which to calculate metrics.
        standard_metrics: List of strings of standard metrics for which to evaluate.
            By default, calculate on valid split. Optionally, prepend metric with
            "train/" to calculate on train split instead.
        custom_valid_funcs: List of functions to calculate on valid split of the form:
            metric_fn(Y, Y_preds, probs=Y_probs) -> {metric_name: metric, ...}
            metric_names will automatically have task and split prefixes added
        custom_train_funcs: List of functions to calculate on train split.
        """
        self.standard_metrics = self._normalize_metric_names(standard_metrics)
        self.custom_metric_funcs = {
            "train": custom_train_funcs,
            "valid": custom_valid_funcs,
        }

    def score(self, model, task, split=None):
        """
        Calculates and returns a metrics_dict for a given task

        model: MetalModel to score (data will be moved to the same device as model)
        task: Task to calculate metrics on
        split: If non-None, only calculate metrics for this split

        return: a metrics_dict object which contains:
        {
            metric : score
        }
        """
        metrics_dict = {}
        splits = [split] if split else ["train", "valid"]

        for split in splits:
            standard_metrics = [
                metric
                for metric in self.standard_metrics
                if metric.startswith(f"{split}/")
            ]
            custom_metric_funcs = self.custom_metric_funcs[split]
            if not (standard_metrics or custom_metric_funcs):
                continue

            # TODO(maxlam) Perhaps refactor (gather Y_preds, Y, Y_probs)
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
            for standard_metric_name in self.standard_metrics:
                metric = standard_metric_name.split("/")[-1]
                standard_metric_score = metric_score(Y, Y_preds, metric, probs=Y_probs)
                metrics_dict[standard_metric_name] = standard_metric_score

            # Calculate custom fns
            for custom_metric_func in custom_metric_funcs:
                custom_metric_dict = custom_metric_func(Y, Y_preds, probs=Y_probs)
                metrics_dict.update(custom_metric_dict)

            # Construct full metric names: task/split/metric
            metrics_dict = {
                f"{task.name}/{split}/{metric.split('/')[-1]}": value
                for metric, value in metrics_dict.items()
            }
        return metrics_dict

    def _normalize_metric_names(self, metric_names):
        """Adds 'valid/' as a prefix to all metrics unless split is already specified"""
        full_names = []
        for metric in metric_names:
            if "train/" in metric or "valid/" in metric:
                full_names.append(metric_names)
            else:
                full_names.append(f"valid/{metric}")
        return full_names
