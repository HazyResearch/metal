from metal.metrics import METRICS as STANDARD_METRICS, metric_score


class Scorer(object):
    """
    DESIGN:
    - A Scorer is a bundle of metrics; it defines what metrics _can_ be calculated on a
    given task (may be able to use smart defaults based on the Task subclass; e.g.,
    classification comes with many nicely defined).
        - custom functions come with a list of names of the metrics they produce (with
        error checking to confirm they don't produce more than that)
    - A Scorer operates over gold labels, probabilities, and predictions
        NOTE: we use
    - All metrics in a scorer produce simple metric name only
        - a simple metric name looks like "accuracy"
        - a full metric name looks like "foo_task/bar_payload/accuracy"

    Args:
        standard_metrics: List of strings of standard metrics for which to evaluate.
            By default, calculate on valid split. Optionally, prepend metric with
            "train/" to calculate on train split instead.
        custom_metric_funcs: Dict of the form:

            {metric_fn1: ["metric1a", ..., "metric1z"],
             metric_fn2: ["metric2a", ..., "metric2z]}

            where metric_fn is a function of the form:

            metric_fn1(Y, Y_preds, probs=Y_probs) ->
                {metric1a: value1, ..., metric1z: valueN}
    """

    def __init__(self, standard_metrics=[], custom_metric_funcs={}):
        self.standard_metrics = standard_metrics
        for metric_name in standard_metrics:
            if "/" in metric_name:
                msg = (
                    f"Standard metrics at Scorer initialization time must not "
                    "include task or split name, but you submitted: {metric_name}"
                )
                raise Exception(msg)
            if metric_name not in STANDARD_METRICS:
                msg = (
                    f"Requested standard metric {metric_name} could not be found in "
                    "metrics.py."
                )
                raise Exception(msg)

        # Create a map from custom metric names to the function that creates them
        self.custom_metric_funcs = custom_metric_funcs
        self.custom_metric_map = {}
        for metric_fn, metric_names in custom_metric_funcs.items():
            assert isinstance(metric_names, list)
            for metric_name in metric_names:
                if "/" in metric_name:
                    msg = (
                        f"Metrics produced by custom_metric_funcs must not include "
                        f"task or split name, but you submitted: {metric_name}."
                    )
                    raise Exception(msg)
                self.custom_metric_map[metric_name] = metric_fn

    def score(self, Y, Y_probs, Y_preds, target_metrics=None):
        """
        Calculates and returns a metrics_dict for a given set of predictions and labels

        Args:
            Y: an [n] list of gold labels
            Y_probs: an [n] list of probabilities
            Y_preds: an [n] list of predictions
            target_metrics: a list of simple metrics to calculate
        Returns:
            a metrics_dict object of the form:
                {metric1 : score1, ...., metricN: score N}

        Note that the returned metrics dict will be transformed to have full metric
        names (e.g., "accuracy" -> "foo_task/bar_payload/accuracy") in the trainer.
        """
        self.validate_target_metrics(target_metrics)

        # TODO: Tighen this up; it can be much more efficient
        # The main issue is that we currently require Y/Y_probs/Y_preds to be lists
        # so that they can support sequence-based tasks that have arbitrary length
        # labels. But there is certainly a way we can be more strict/certain about
        # what our data types will be and do some much more efficient slice operation
        # instead of list comprehension.

        # Identify all examples with at least one non-zero (i.e., non-abstain) label
        active = [bool(y != 0) for y in Y]
        if sum(active) != len(active):
            Y = [y for a, y in zip(active, Y) if a]
            if Y_probs:
                Y_probs = [y for a, y in zip(active, Y_probs) if a]
            if Y_preds:
                Y_preds = [y for a, y in zip(active, Y_preds) if a]

        simple_metrics_dict = {}
        for metric in self.standard_metrics:
            # If target metrics were specified and this is not one of them, skip it
            if target_metrics and metric not in target_metrics:
                continue
            score = metric_score(Y, Y_preds, metric, probs=Y_probs)
            simple_metrics_dict[metric] = score

        for metric, custom_metric_func in self.custom_metric_map.items():
            # If target metrics were specified and this is not one of them, skip it
            if target_metrics and metric not in target_metrics:
                continue
            # If the current metric is already in the simple_metrics_dict, skip it
            # This is possible because a custom_metric_func can return multiple metrics
            if metric in simple_metrics_dict:
                continue
            custom_metric_dict = custom_metric_func(Y, Y_preds, probs=Y_probs)
            for metric, score in custom_metric_dict.items():
                if not target_metrics or metric in target_metrics:
                    simple_metrics_dict[metric] = score

        return simple_metrics_dict

    def validate_target_metrics(self, target_metrics):
        if not target_metrics:
            return
        for metric in target_metrics:
            if "/" in metric:
                msg = (
                    "Target metrics must be in simple form (e.g., accuracy), "
                    "not full form (e.g., foo_task/bar_payload/accuracy) and "
                    "should not include the character '/'."
                )
                raise Exception(msg)
            elif metric not in self.metrics:
                msg = (
                    f"Target metric {metric} is not supported by the given Scorer. "
                    f"Supported tasks are: {self.metrics}."
                )
                raise Exception(msg)

    @property
    def metrics(self):
        """Returns a list of short metric names supported by this Scorer"""
        return self.standard_metrics + list(self.custom_metric_map.keys())
