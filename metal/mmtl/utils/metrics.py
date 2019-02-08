from scipy.stats import pearsonr, spearmanr


def spearman_corr(gold, outputs):
    corr, p_value = spearmanr(gold, outputs)
    return {"spearman_corr": corr}


def pearson_corr(gold, outputs):
    corr, p_value = pearsonr(gold, outputs)
    return {"pearson_corr": corr}
