import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
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


def save_dataframe(df, task_name, filepath="./"):
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
    print("Saved dataframe to: ", filename)


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
