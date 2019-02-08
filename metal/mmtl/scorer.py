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

    def score(self, model, data_loader, task_name):
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
            print("Batch %d of %d" % (batch_num, len(data_loader)))

            Xb, Yb = data
            Y.append(utils.to_numpy(Yb))

            # Place data on gpu if necessary
            if model.config["device"] != "cpu":
                Xb = place_on_gpu(Xb)

            # Optimized out if head_output is passed
            # if head_output is None:
            #     # Only works for end_model
            #     # Y_p, Y_s = model.predict(Xb, return_probs=True)
            #     Y_s = model.calculate_output(Xb, [task_name])
            #     Y_s_to_npy = Y_s[task_name].numpy()
            #     Y_p = utils.break_ties(Y_s_to_npy, "random").astype(np.int)
            #     Y_preds.append(utils.to_numpy(Y_p))
            #     Y_probs.append(utils.to_numpy(Y_s_to_npy))

        # Pass through head_output to task
        # if head_output is not None:
        # Y_probs = model.output_hat_funcs[task_name](head_output)

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
            self.update_metrics_dict(metrics_dict, custom_metric_dict)

        return metrics_dict

    def update_metrics_dict(self, metrics_dict, metric):
        for k, v in metric.items():
            metrics_dict[k] = v
