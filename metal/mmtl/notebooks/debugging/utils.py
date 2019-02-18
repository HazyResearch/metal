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

    model = MetalModel(tasks, verbose=False, device=-1)
    try:
        model.load_state_dict(torch.load(model_path)["model"])
    except KeyError:
        model.load_weights(model_path)

    return model, dl


def pred_to_word(proba, task_name):

    # [a,b] corresponds to pred=[0,1] #NOTE THAT ! ALWAYS MEANS THE SAME
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

    # regression
    if task_name == "STSB":
        return list(5.0 * np.array(proba))

    # three-class
    if task_name in ["MNLI", "MNLI-m", "MNLI-mm"]:
        pred = proba
    else:
        pred = 1 * (np.array(proba) > 0.5)

    label_map = pred_to_word_dict[task_name]
    labels = []
    for p in pred:
        labels.append(label_map[p])
    return labels


def create_submit_dataframe(model_path, task_name, model, dl):
    proba = []
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

    predictions = pred_to_word(proba, task_name)
    return pd.DataFrame(predictions, columns=["prediction"])


def save_tsv(df, task_name, filepath="./"):
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

    tsv_name = task_to_name_dict[task_name]
    filename = f"{filepath}{tsv_name}.tsv"
    df.to_csv(filename, sep="\t", index_label="index")
    print("Saved TSV to: ", filename)


def create_dataframe(model_path, task_name, model, dl):
    bert_model = "bert-base-uncased"

    data = {"sentence1": [], "sentence2": [], "label": [], "score": []}
    max_batches = 100
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
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
        data["score"] += list(scores)
        data["label"] += list(convert_labels(y, "categorical", "onezero").numpy())
        count += 1
        if count > max_batches:
            break

    df_error = pd.DataFrame(data, columns=["sentence1", "sentence2", "score", "label"])
    df_error["is_wrong"] = 1 * (df_error.score > 0.5) != df_error["label"]
    return df_error


# DEBUGGING RELATED HELPER FUNCTIONS
def save_dataframe(df, filepath):
    df.to_csv(filepath, sep="\t")
    print("Saved dataframe to: ", filepath)


def load_dataframe(filepath):
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
    """Print prediction that's close to 0.5 and correct/incorrect"""
    thresh_idx = np.where(np.abs(df.score - df.label) >= thresh)[0]
    idx_true = np.where(df.is_wrong == is_incorrect)[0]
    idx = list(set(thresh_idx).intersection(set(idx_true)))
    id = np.random.choice(list(idx))
    print("ID: ", id)
    row = df.iloc[id]
    print_row(row)


def print_very_wrong_pred(df, thresh=0.95):
    """Print predictions that are thresh away from true label"""
    try:
        thresh_idx = np.where(np.abs(df.score - df.label) >= thresh)[0]
        idx_true = np.where(df.is_wrong)[0]
        idx = list(set(thresh_idx).intersection(set(idx_true)))
        id = np.random.choice(list(idx))
        print("ID: ", id)
        row = df.iloc[id]
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
    """Prints predictions that are incorrect and share a common
    feature correlated with incorrect predictions"""

    # Create a vector of correct/incorrect predictions
    # TODO: use metal function
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

    row = df_error.iloc[np.random.choice(list(idx))]
    print_row(row)
