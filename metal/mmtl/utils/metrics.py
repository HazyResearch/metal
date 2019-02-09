from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef


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
