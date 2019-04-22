import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, mean_squared_error

from metal.metrics import metric_score


def mse(gold, _, probs):
    return {"mse": float(mean_squared_error(gold, probs))}


def spearman_corr(gold, _, probs):
    # NOTE: computes using "probs", which is the non-rounded output of the model.
    # TODO: fix this poor naming convention to better support regression tasks.
    probs_array = np.vstack(probs).squeeze()
    gold_array = np.vstack(gold).squeeze()
    corr, p_value = spearmanr(gold_array, probs_array)
    if np.isnan(corr):
        print(f"Warning: converting nan -> 0.0 for spearman_corr")
        corr = 0.0
    return {"spearman_corr": corr}


def pearson_corr(gold, _, probs):
    # NOTE: computes using "probs", which is the non-rounded output of the model.
    # TODO: fix this poor naming convention to better support regression tasks.
    probs_array = np.vstack(probs).squeeze()
    gold_array = np.vstack(gold).squeeze()
    corr, p_value = pearsonr(gold_array, probs_array)
    if np.isnan(corr):
        print(f"Warning: converting nan -> 0.0 for spearman_corr")
        corr = 0.0
    return {"pearson_corr": corr}


def matthews_corr(gold, outputs, **kwargs):
    return {"matthews_corr": matthews_corrcoef(gold, outputs)}


def acc_f1(gold, outputs, **kwargs):
    """A convenience custom function that returns accuracy, f1, and their mean"""
    accuracy = metric_score(gold, outputs, metric="accuracy")
    f1 = metric_score(gold, outputs, metric="f1")
    return {"accuracy": accuracy, "f1": f1, "acc_f1": np.mean([accuracy, f1])}


def ranking_acc_f1(gold, outputs, probs):
    """A convenience custom function that returns accuracy, f1, and their mean for ranking task heads."""
    gold = (1 - gold) + 1
    outputs = 1 * (probs.reshape((-1,)) > 0.5)
    accuracy = metric_score(gold, outputs, metric="accuracy")
    f1 = metric_score(gold, outputs, metric="f1")
    return {"accuracy": accuracy, "f1": f1, "acc_f1": np.mean([accuracy, f1])}


def pearson_spearman(gold, outputs, probs):
    """A convenience custom function that return pearson, spearman, and their mean"""
    metrics_dict = {}
    metrics_dict.update(spearman_corr(gold, outputs, probs))
    metrics_dict.update(pearson_corr(gold, outputs, probs))
    metrics_dict["pearson_spearman"] = np.mean(
        [metrics_dict["pearson_corr"], metrics_dict["spearman_corr"]]
    )
    return metrics_dict


GLUE_METRICS = {
    "COLA": "matthews_corr",
    "SST2": "accuracy",
    "MRPC": "acc_f1",
    "STSB": "pearson_spearman",
    "QQP": "acc_f1",
    "MNLI": "accuracy",
    "QNLI": "accuracy",
    "RTE": "accuracy",
    "WNLI": "accuracy",
}


def glue_score(metrics_dict={}, split="valid"):
    """Computes the glue_score (mean of individual task metrics) from a metrics_dict"""
    target_metrics = [
        f"{task}/{task}_{split}/{task}_gold/{metric}"
        for task, metric in GLUE_METRICS.items()
    ]
    scores = []
    for metric in target_metrics:
        if metric in metrics_dict:
            scores.append(metrics_dict[metric])
    return {f"model/{split}/gold/glue_score": np.mean(scores)}
