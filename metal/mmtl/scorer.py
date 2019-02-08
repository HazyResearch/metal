import numpy as np

from metal.metrics import metric_score
from metal.mmtl.utils import utils
from metal.utils import place_on_gpu, recursive_merge_dicts


"""
Scorer class which evaluates metrics given a task and model.

Example usage:
scorer = Scorer()
task1 = Task(..., scorer=scorer)
...
taskn = Task(..., scorer=scorer)
model = MetalModel(tasks=[task1, ..., taskn])

"""


class Scorer(object):
    def __init__(self, standard_metrics=["accuracy"], custom_metric_fns=[]):
        """
        Creates a scorer object.

        data_loader: DataLoader on which to calculate metrics.
        standard_metrics: List of strings of standard metrics for which to evaluate.
        custom_metric_fns: List of functions of the form:

           metric_fn(Y, Y_preds, probs=Y_probs)
           - Return a dict with name of metric to metric

        scorer_prefix: String prefix to tag metrics calculated by the current scorer.
        """
        self.standard_metrics = standard_metrics
        self.custom_metric_fns = custom_metric_fns

    def score(self, model, data_loader, task_name, split="valid"):
        """
        Calculates and returns a metrics_dict for a given task

        model: MetalModel to score
        data_loader: DataLoader on which to evaluate metrics
        task_name: the name of the Task being scored

        return: a metrics_dict object which contains:
        {
            metric : score
        }
        """
        metrics_dict = {}

        # TODO(maxlam) Perhaps refactor
        # Gather Y_preds, Y, Y_probs
        Y_preds, Y, Y_probs = [], [], []

        for batch_num, data in enumerate(data_loader):
            # print("Batch %d of %d" % (batch_num, len(data_loader)))

            Xb, Yb = data
            Y.append(utils.to_numpy(Yb))

            # Place data on gpu if necessary
            if model.config["device"] != "cpu":
                Xb = place_on_gpu(Xb)

            Yb_probs = model.calculate_output(Xb, [task_name])[task_name]
            Y_probs.append(Yb_probs)
            Yb_preds = utils.break_ties(Yb_probs.numpy(), "random").astype(np.int)
            Y_preds.append(Yb_preds)

        # Stack batches
        Y_preds, Y, Y_probs = map(utils.stack_batches, [Y_preds, Y, Y_probs])

        # From the labels and predictions calculate metrics
        for standard_metric_name in self.standard_metrics:
            standard_metric_score = metric_score(
                Y, Y_preds, standard_metric_name, probs=Y_probs
            )
            metrics_dict[standard_metric_name] = standard_metric_score

        # Calculate custom fns
        for custom_metric_fn in self.custom_metric_fns:
            custom_metric_dict = custom_metric_fn(Y, Y_preds, probs=Y_probs)
            metrics_dict.update(custom_metric_dict)

        # Construct full metric names: task/split/metric
        metrics_dict = {
            f"{task_name}/{split}/{metric.split('/')[-1]}": value
            for metric, value in metrics_dict.items()
        }
        return metrics_dict
