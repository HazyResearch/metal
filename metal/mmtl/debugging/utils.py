import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import metal.mmtl.dataset as dataset
from metal.mmtl.bert_tasks import create_tasks
from metal.mmtl.metal_model import MetalModel
from metal.utils import convert_labels


def load_data_and_model(model_path, task_name, split):
    """
    Loads the model specified by model_path and dataset specified by task_name, split.
    """

    # Create DataLoader
    bert_model = "bert-base-uncased"
    max_len = 256
    bert_output_dim = 768
    dl_kwargs = {"batch_size": 32, "shuffle": False}

    dataset_cls = getattr(dataset, task_name.upper() + "Dataset")
    ds = dataset_cls(
        split=split, bert_model=bert_model, max_len=max_len, max_datapoints=-1
    )
    dl = ds.get_dataloader(**dl_kwargs)

    # Load best model for specified task
    tasks = create_tasks(
        task_names=[task_name],
        bert_model=bert_model,
        max_len=max_len,
        dl_kwargs={"batch_size": 1},
        bert_output_dim=bert_output_dim,
        max_datapoints=10,
    )

    # Load and EVAL model
    # TODO: this is broken. model loading needs to be fixed
    model = MetalModel(tasks, verbose=False, device=-1)
    try:
        model.load_state_dict(torch.load(model_path)["model"])
    except KeyError:
        model.load_weights(model_path)

    model.eval()
    return model, dl


def pred_to_word(proba, task_name):
    """Given probabilistic labels, convert to labels for GLUE submission.

    Args:
        proba: list of probabilistic labels
        task_name: task associated with proba labels

    Returns:
        labels: list of strings of labels for GLUE submission
    """

    # [a,b] corresponds to pred=[0,1] #NOTE THAT 1 ALWAYS MEANS THE SAME
    # from dataloader, [a,b] corresponds to label=[1,2]
    pred_to_word_dict = {
        "QNLI": ["not_entailment", "entailment"],
        "STSB": None,
        "SST2": ["0", "1"],
        "COLA": ["0", "1"],
        "MNLI-m": ["entailment", "contradiction", "neutral"],
        "MNLI": ["entailment", "contradiction", "neutral"],
        "MNLI-mm": ["entailment", "contradiction", "neutral"],
        "RTE": ["not_entailment", "entailment"],
        "WNLI": ["0", "1"],
        "QQP": ["0", "1"],
        "MRPC": ["0", "1"],
    }

    # Regression mapping [0.0,1.0] to [0.0,5.0]
    if task_name == "STSB":
        return list(5.0 * np.array(proba))

    # TODO: this should be fixed, be the same for all classification tasks
    if task_name in ["MNLI", "MNLI-m", "MNLI-mm"]:
        pred = proba
    else:
        pred = 1 * (np.array(proba) > 0.5)

    # Convert integer prediction to string label
    label_map = pred_to_word_dict[task_name]
    labels = []
    for p in pred:
        labels.append(label_map[p])
    return labels


def create_submit_dataframe(task_name, model, dl):
    """Create dataframe for GLUE submission for given task and model.

    Args:
        task_name: task to create submission for
        model: MetalModel object of model to evaluate
        dl: DataLoader object for task_name task

    Returns:
        DataFrame object: predictions in DataFrame object
    """
    proba = []
    # For each example, save probabilistic predictions
    # TODO: use model.predict() and simplify pred_to_word() functions
    for x, y in tqdm(list(dl)):
        if task_name in ["MNLI", "MNLI-m", "MNLI-mm"]:
            proba_batch = np.argmax(
                (
                    model.calculate_output(x, [task_name])[task_name]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                axis=1,
            )
        else:
            proba_batch = (
                model.calculate_output(x, [task_name])[task_name]
                .detach()
                .cpu()
                .numpy()[:, 0]
            )
        proba += list(proba_batch)

    # Convert probabilistic predictions to string label for GLUE
    predictions = pred_to_word(proba, task_name)
    return pd.DataFrame(predictions, columns=["prediction"])


def save_tsv(df, task_name, filepath="./"):
    """Saves TSV as task_name.tsv for given DataFrame in filepath.

    Args:
        df: predictions in DataFrame object
        task_name: task to create submission for
        filepath: path to save TSV
    """

    task_to_name_dict = {
        "QNLI": "QNLI",
        "STSB": "STS-B",
        "SST2": "SST-2",
        "COLA": "CoLA",
        "MNLI-m": "MNLI-m",
        "MNLI-mm": "MNLI-mm",
        "RTE": "RTE",
        "WNLI": "WNLI",
        "QQP": "QQP",
        "MRPC": "MRPC",
    }

    # Creates filename based on task_name and saves TSV
    tsv_name = task_to_name_dict[task_name]
    filename = f"{filepath}{tsv_name}.tsv"
    df.to_csv(filename, sep="\t", index_label="index")
    print("Saved TSV to: ", filename)


# Debugging Related Functions
def create_dataframe(task_name, model, dl):
    """Create dataframe with datapoint, predicted score, and true label.

    Args:
        task_name: task to create evaluation information for
        model: MetalModel object of model to evaluate
        dl: DataLoader object for task_name task

    Returns:
        DataFrame object: info. about datapoints, labels, score
    """

    # Use BERT model to convert tokenization to sentence
    bert_model = "bert-base-uncased"
    data = {"sentence1": [], "sentence2": [], "label": [], "score": []}
    max_batches = 100
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    # Create a list of examples and associated predicted score and true label
    count = 0
    for x, y in tqdm(list(dl)):
        for tokens_idx in x[0]:
            tokens = tokenizer.convert_ids_to_tokens(tokens_idx.numpy())
            phrases = (
                " ".join(tokens)
                .replace("[PAD]", "")
                .replace("[CLS]", "")
                .split("[SEP]")
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
        count += 1
        if count > max_batches:
            break

    # Create DataFrame with datapoint and scores, labels
    df_error = pd.DataFrame(data, columns=["sentence1", "sentence2", "score", "label"])
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
