import os

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import metal.mmtl.dataset as dataset
from metal.mmtl.glue_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.utils import convert_labels


def load_data_and_model(model_path, task_name, split):
    """
    Loads the model specified by model_path and dataset specified by task_name, split.
    """

    # Create DataLoader
    bert_model = "bert-base-uncased"
    max_len = 256
    dl_kwargs = {"batch_size": 1, "shuffle": False}

    # Load best model for specified task
    task = create_tasks(
        task_names=[task_name],
        bert_model=bert_model,
        max_len=max_len,
        dl_kwargs=dl_kwargs,
        splits=[split],
        max_datapoints=-1,
        generate_uids=True,
    )[0]

    #  Load and EVAL model
    model_path = os.path.join(model_path, "best_model.pth")
    model = MetalModel([task], verbose=False, device=0)
    model.load_weights(model_path)
    model.eval()

    return model, task.data_loaders[split]


# Debugging Related Functions
def create_dataframe(task_name, model, dl, target_uids=None, max_batches=None):
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

    # Use BERT model to convert tokenization to sentence
    bert_model = "bert-base-uncased"
    data = {"sentence1": [], "sentence2": [], "label": [], "score": [], "uid": []}
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

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
        scores = (
            model.calculate_output(x, [task_name])[task_name]
            .detach()
            .cpu()
            .numpy()[:, 0]
        )  # .flatten()

        # Score is the predicted probabilistic label, label is the ground truth
        data["score"] += list(scores)
        data["label"] += list(convert_labels(y, "categorical", "onezero").numpy())
        data["uid"].append(uid)
        count += 1
        if max_batches and count > max_batches:
            break

    # Create DataFrame with datapoint and scores, labels
    df_error = pd.DataFrame(
        data, columns=["sentence1", "sentence2", "score", "label", "uid"]
    )
    df_error["is_wrong"] = 1 * (df_error.score > 0.5) != df_error["label"]
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
    id = np.random.choice(list(idx))
    print("ID: ", id)
    row = df.iloc[id]
    print(row)


def print_barely_pred(df, is_incorrect=True, thresh=0.05):
    """Print prediction that's close to 0.5 and correct/incorrect.
    Args:
        df:  info. about datapoints, labels, score
        is_incorrect (bool): which datapoints to print
        thresh (float): distance predicted score is from 0.5
    """
    # Find examples that are is_incorrect and thresh near 0.5
    thresh_idx = np.where(np.abs(df.score - df.label) >= thresh)[0]
    idx_true = np.where(df.is_wrong == is_incorrect)[0]
    idx = list(set(thresh_idx).intersection(set(idx_true)))

    # Select random example and print
    id = np.random.choice(list(idx))
    print("ID: ", id)
    row = df.iloc[id]
    print_row(row)


def print_very_wrong_pred(df, thresh=0.95):
    """Print prediction that's close to 0.5 and correct/incorrect.
    Args:
        df:  info. about datapoints, labels, score
        thresh (float): distance predicted score is from true label
    """
    try:
        # Find examples that are incorrect and thresh away from true label
        thresh_idx = np.where(np.abs(df.score - df.label) >= thresh)[0]
        idx_true = np.where(df.is_wrong)[0]
        idx = list(set(thresh_idx).intersection(set(idx_true)))

        # Select random example and print
        id = np.random.choice(list(idx))
        print("ID: ", id)
        row = df.iloc[id]
    # TODO: can remove try/except by checking if list is empty
    except ValueError:
        print("Threshold too high, reducing by 0.05")
        thresh_idx = np.where(np.abs(df.score - df.label) >= thresh)[0]
        idx_true = np.where(df.is_wrong)[0]
        idx = list(set(thresh_idx).intersection(set(idx_true)))
        id = np.random.choice(list(idx))
        print("ID: ", id)
        row = df.iloc[id]

    print_row(row)


def print_systematic_wrong(df_error, num_features=5):
    """Print prediction that's close to 0.5 and correct/incorrect.

    Args:
        df:  info. about datapoints, labels, score
        num_features (int): number of correlated features to print examples for
    """

    # Create a vector of correct/incorrect predictions
    # TODO: use MeTaL function for label conversion
    y = 2 * (np.array(df_error.is_wrong.astype(float)) - 0.5)

    # Create corpus by combining sentences
    combined = []
    for a, b in zip(np.array(df_error.sentence1), np.array(df_error.sentence2)):
        combined.append(str(a) + str(b))

    # Create BoW featurization
    corpus = np.array(list(df_error.sentence1))
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

    incorrect_idx = np.where(df_error.is_wrong)[0]
    idx = list(set(feat_idx).intersection(incorrect_idx))
    print()

    # Print random example
    row = df_error.iloc[np.random.choice(list(idx))]
    print_row(row)
