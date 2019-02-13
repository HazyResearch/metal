import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from metal.metrics import metric_score


def spearman_corr(gold, _, probs):
    # NOTE: computes using "probs", which is the non-rounded output of the model.
    # TODO: fix this poor naming convention to better support regression tasks.
    probs = probs.squeeze()
    corr, p_value = spearmanr(gold, probs)
    return {"spearman_corr": corr}


def pearson_corr(gold, _, probs):
    # NOTE: computes using "probs", which is the non-rounded output of the model.
    # TODO: fix this poor naming convention to better support regression tasks.
    probs = probs.squeeze()
    corr, p_value = pearsonr(gold, probs)
    return {"pearson_corr": corr}


def matthews_corr(gold, outputs, **kwargs):
    return {"matthews_corr": matthews_corrcoef(gold, outputs)}


def acc_f1(gold, outputs, **kwargs):
    """A convenience custom function that returns accuracy, f1, and their mean"""
    accuracy = metric_score(gold, outputs, metric="accuracy")
    f1 = metric_score(gold, outputs, metric="f1")
    return {"accuracy": accuracy, "f1": f1, "acc_f1": np.mean([accuracy, f1])}


def pearson_spearman(gold, outputs, probs):
    """A convenience custom function that return pearson, spearman, and their mean"""
    metrics_dict = {}
    metrics_dict.update(spearman_corr(gold, outputs, outputs))
    metrics_dict.update(pearson_corr(gold, outputs, outputs))
    metrics_dict["pearson_spearman"] = np.mean(
        [metrics_dict["pearson_corr"], metrics_dict["spearman_corr"]]
    )
    return metrics_dict
