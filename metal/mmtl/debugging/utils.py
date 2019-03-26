import os
import random

import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads
from metal.mmtl.metal_model import MetalModel
from metal.utils import convert_labels


def load_data_and_model(model_path, task_names, split, bert_model="bert-base-uncased"):
    """
    Loads the model specified by model_path and dataset specified by task_name, split.
    """

    # Create DataLoader
    max_len = 200
    dl_kwargs = {"batch_size": 1, "shuffle": False}
    if not isinstance(task_names, list):
        task_names = [task_name for task_name in task_names.split(",")]

    # Load best model for specified task
    tasks, payloads = create_glue_tasks_payloads(
        task_names=task_names,
        bert_model=bert_model,
        max_len=max_len,
        dl_kwargs=dl_kwargs,
        splits=[split],
        max_datapoints=-1,
        generate_uids=True,
    )

    #  Load and EVAL model
    model_path = os.path.join(model_path, "best_model.pth")
    model = MetalModel(tasks, verbose=False, device=0)
    model.load_weights(model_path)
    model.eval()

    for payload in payloads:
        if payload.split == split:
            break

    return model, payload.data_loader


# Debugging Related Functions
def create_dataframe(
    task_name,
    model,
    dl,
    target_uids=None,
    max_batches=None,
    bert_model="bert-base-uncased",
):
    """Create dataframe with datapoint, predicted score, and true label.

    Args:
        task_name: task to create evaluation information for
        model: MetalModel object of model to evaluate
        dl: DataLoader object for task_name task
        target_uids: uids to evaluate on
        max_batches: number of batches to eval before stopping (useful for large datasets)

    Returns:
        DataFrame object: info. about datapoints, labels, score
    """
    if task_name == "MNLI":
        raise NotImplementedError("We currently assume binary tasks")

    # Use BERT model to convert tokenization to sentence
    data = {"sentence1": [], "sentence2": [], "label": [], "score": [], "uid": []}
    do_lower_case = "uncased" in bert_model
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    # Create a list of examples and associated predicted score and true label
    count = 0
    all_uids = dl.dataset.uids

    # assuming data_loader batch_size=1
    for (x, y), uid in tqdm(zip(list(dl), all_uids)):
        if target_uids and uid not in target_uids:
            continue

        tokens_idx = x[0][0]
        tokens = tokenizer.convert_ids_to_tokens(tokens_idx.numpy())
        phrases = (
            " ".join(tokens).replace("[PAD]", "").replace("[CLS]", "").split("[SEP]")
        )
        data["sentence1"] += [phrases[0]]
        if len(phrases) > 1:
            data["sentence2"] += [phrases[1]]
        else:
            data["sentence2"] += ["NA"]

        scores = np.array(model.calculate_probs(x, [task_name])[task_name])[:, 0]

        # Score is the predicted probabilistic label, label is the ground truth
        data["score"] += list(scores)
        data["label"] += list(y[task_name].numpy())
        data["uid"].append(uid)
        count += 1
        if max_batches and count > max_batches:
            break

    # Create DataFrame with datapoint, score, label, pred, uid
    df_error = pd.DataFrame(
        data, columns=["sentence1", "sentence2", "score", "label", "uid"]
    )
    df_error["label"] = convert_labels(
        df_error["label"].values, "categorical", "onezero"
    )
    df_error["pred"] = 1 * (df_error["score"] > 0.5)

    df_error["is_wrong"] = df_error["pred"] != df_error["label"]
    return df_error


# Helper Functions for Debugging
def save_dataframe(df, filepath):
    """ Save DataFrame as filepath (TSV)"""
    df.to_csv(filepath, sep="\t")
    print("Saved dataframe to: ", filepath)


def load_dataframe(filepath):
    """ Load DataFrame from filepath (TSV)"""
    df = pd.read_csv(filepath, sep="\t")
    return df


def print_row(row):
    """ Pretty prints a row of error dataframe """
    print(f"sentence1: \t{row.sentence1}")
    print(f"sentence2: \t{row.sentence2}")
    print("score: \t{:.4f}".format(row.score))
    print(f"label: \t{row.label}")
    print()


def print_random_pred(df):
    """Print random row of error dataframe"""
    idx = np.range(np.shape(df)[0])
    id = np.random.choice(list(idx), replace=False)
    print("ID: ", id)
    row = df.iloc[id]
    print(row)


def print_examples(df, idxs, n=1):
    n = min(n, len(idxs))
    for idx in np.random.choice(idxs, size=n, replace=False):
        # Select random example and print
        row = df.iloc[idx]
        # print("UID: ", row["uid"])
        print_row(row)


def print_barely_pred(df, is_incorrect=True, thresh=0.05, n=1):
    """Print prediction that's close to 0.5 and correct/incorrect.
    Args:
        df:  info. about datapoints, labels, score
        is_incorrect (bool): which datapoints to print
        thresh (float): distance predicted score is from 0.5
    """
    # Find examples that are is_incorrect and thresh near 0.5
    thresh_idx = np.where(np.abs(df.score - 0.5) <= thresh)[0]
    idx_true = np.where(df.is_wrong == is_incorrect)[0]
    matches = list(set(thresh_idx).intersection(set(idx_true)))
    if matches:
        print(f"{len(matches)} matches were found with the given criteria.\n")
        print_examples(df, matches, n)
        return True
    else:
        print("No matches were found for the given criteria.")
        return False


def print_barely_right(df, **kwargs):
    print_barely_pred(df, is_incorrect=False, **kwargs)


def print_barely_wrong(df, **kwargs):
    print_barely_pred(df, is_incorrect=True, **kwargs)


def print_very_pred(df, is_incorrect=True, thresh=0.95, n=1):
    """Print prediction that's very confident and correct/incorrect.
    Args:
        df:  info. about datapoints, labels, score
        thresh (float): confidence of the prediction (closer to 1 = more confident)
    """
    # Find examples that are incorrect and thresh away from true label
    thresh_idx = np.where(np.abs(df.score - 0.5) >= (thresh - 0.5))[0]
    idx_true = np.where(df.is_wrong == is_incorrect)[0]
    matches = list(set(thresh_idx).intersection(set(idx_true)))
    if matches:
        print(f"{len(matches)} matches were found with the given criteria.\n")
        print_examples(df, matches, n)
        return True
    else:
        print("No matches were found for the given criteria.")
        return False


def print_very_right(df, **kwargs):
    print_very_pred(df, is_incorrect=False, **kwargs)


def print_very_wrong(df, **kwargs):
    print_very_pred(df, is_incorrect=True, **kwargs)


def print_systematic_wrong(df, num_features=5, n=1):
    """Print prediction that's close to 0.5 and correct/incorrect.

    Args:
        df:  info. about datapoints, labels, score
        num_features (int): number of correlated features to print examples for
    """

    # Create a vector of correct/incorrect predictions
    # TODO: use MeTaL function for label conversion
    y = 2 * (np.array(df.is_wrong.astype(float)) - 0.5)

    # Create corpus by combining sentences
    combined = []
    for a, b in zip(np.array(df.sentence1), np.array(df.sentence2)):
        combined.append(str(a) + str(b))

    # Create BoW featurization
    corpus = np.array(list(df.sentence1))
    vectorizer = CountVectorizer(ngram_range=(2, 5), stop_words="english")
    X = vectorizer.fit_transform(corpus)

    # Run LR to find correlations
    lr_model = LogisticRegression(penalty="l1")
    lr_model.fit(X, y)

    # Find datapoints with incorrect prediction and top feature present
    top_idx = np.argsort(lr_model.coef_[0])[::-1][0:50]
    names = vectorizer.get_feature_names()
    feat_idx = []
    for i in range(num_features):
        print(names[top_idx[i]])
        feat_idx += list(np.where(X.todense()[:, top_idx[i]] == 1)[0])

    incorrect_idx = np.where(df.is_wrong)[0]
    matches = list(set(feat_idx).intersection(incorrect_idx))
    print()

    if matches:
        print(f"{len(matches)} matches were found with the given criteria.\n")
        print_examples(df, matches, n)
        return True
    else:
        print("No matches were found for the given criteria.")
        return False


def apply_lfs_to_df(df, lfs):
    """Applies a list of lfs that operate over rows to each row in a dataframe

    Returns: L, an [m,n] matrix of weak labels
    """
    n = len(df)
    m = len(lfs)
    L = np.zeros((n, m))
    for i in range(n):
        row = df.iloc[i]
        for j, lf in enumerate(lfs):
            L[i, j] = lf(row)
    Y = df["label"].values
    return L, Y


def view_matches(df, lf, n=0, shuffle=True):
    """Returns up to n rows that lf does not abstain on"""
    L, Y = apply_lfs_to_df(df, [lf])
    idxs = np.where(L[:, 0] != 0)[0]
    if shuffle:
        random.shuffle(idxs)

    if n == 0:
        print(f"Displaying all {len(idxs)} matches")
    else:
        print(f"Displaying {n}/{len(idxs)} matches")
    print()

    for i, idx in enumerate(idxs):
        if n > 0 and i >= n:
            break
        print_row(df.iloc[idx])
