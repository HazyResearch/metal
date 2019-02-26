import copy

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

import metal.mmtl.dataset as dataset


def get_all_dataloaders(
    dataset_name,
    bert_model,
    max_len,
    dl_kwargs,
    split_prop,
    max_datapoints,
    splits,
    verbose=True,
):
    """ Initializes train/dev/test dataloaders given dataset_class"""

    if verbose:
        print(f"Loading {dataset_name} Dataset")

    dataset_cls = getattr(dataset, dataset_name.upper() + "Dataset")

    datasets = {}
    for split_name in splits:
        # Codebase uses valid but files are saved as dev.tsv
        if split_name == "valid":
            split = "dev"
        else:
            split = split_name
        datasets[split_name] = dataset_cls(
            split=split,
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )

    dataloaders = {}

    # When split_prop is not None, we use create an artificial dev set from the train set and

    if split_prop and "train" in splits:

        dataloaders["train"], dataloaders["valid"] = datasets["train"].get_dataloader(
            split_prop=split_prop, **dl_kwargs
        )

        # Use the dev set as test set if available.

        if "valid" in datasets:

            dataloaders["test"] = datasets["valid"].get_dataloader(**dl_kwargs)

    # When split_prop is None, we use standard train/dev/test splits.

    else:

        for split_name in datasets:

            dataloaders[split_name] = datasets[split_name].get_dataloader(**dl_kwargs)

    return dataloaders


def get_dataloader_with_label(dataloader, label_obj):
    """
    dataloader: dataloader wrapping Dataset
    label_obj: function operating on a dataset item or list of labels in correct order
    """

    dataloader_new = copy.deepcopy(dataloader)

    if isinstance(label_obj, list):
        labels_new = label_obj
    elif callable(label_obj):
        labels_new = [label_obj(i) for i in dataloader_new.dataset]
    else:
        raise ValueError("Incorrect label object type -- supply list or function")

    dataloader_new.dataset.labels = labels_new

    return dataloader_new


def get_bleu_dataloader(dataloader):
    def get_bleu_label(it):
        toks, segs = it[0]
        toks = dataloader.dataset.tokenizer.convert_ids_to_tokens(toks)
        toks, segs = np.array(toks), np.array(segs)
        sent1 = list(toks[segs == 0])
        sent2 = list(toks[segs == 1])
        bleu_score = sentence_bleu(sent1, sent2, weights=(1, 0, 0, 0))
        return bleu_score

    return get_dataloader_with_label(dataloader, get_bleu_label)
