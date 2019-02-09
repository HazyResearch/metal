from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef


def spearman_corr(gold, outputs, **kwargs):
    corr, p_value = spearmanr(gold, outputs)
    return {"spearman_corr": corr}


def pearson_corr(gold, outputs, **kwargs):
    corr, p_value = pearsonr(gold, outputs)
    return {"pearson_corr": corr}


def matthews_corr(gold, outputs, **kwargs):
    return {"matthews_corr": matthews_corrcoef(gold, outputs)}
