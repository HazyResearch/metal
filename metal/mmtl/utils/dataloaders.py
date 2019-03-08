import copy

from metal.mmtl.dataset import get_glue_dataset


def get_all_dataloaders(
    dataset_name,
    bert_vocab,
    max_len,
    dl_kwargs,
    split_prop,
    max_datapoints,
    splits,
    generate_uids=False,
    seed=123,
    verbose=True,
):
    """ Initializes train/dev/test dataloaders given dataset_class"""

    if verbose:
        print(f"Loading {dataset_name} Dataset")

    datasets = {}
    for split_name in splits:
        # Codebase uses valid but files are saved as dev.tsv
        if split_name == "valid":
            split = "dev"
        else:
            split = split_name
        datasets[split_name] = get_glue_dataset(
            dataset_name,
            split=split,
            bert_vocab=bert_vocab,
            max_len=max_len,
            max_datapoints=max_datapoints,
            generate_uids=generate_uids,
        )

    dataloaders = {}

    # When split_prop is not None, we use create an artificial dev set from the train set
    if split_prop and "train" in splits:
        dataloaders["train"], dataloaders["valid"] = datasets["train"].get_dataloader(
            split_prop=split_prop, split_seed=seed, **dl_kwargs
        )

        # Use the dev set as test set if available.
        if "valid" in datasets:
            dataloaders["test"] = datasets["valid"].get_dataloader(**dl_kwargs)

    # When split_prop is None, we use standard train/dev/test splits.
    else:
        for split_name in datasets:
            dataloaders[split_name] = datasets[split_name].get_dataloader(**dl_kwargs)
    return dataloaders


def add_labels_to_payload(payload, task_name, label_obj):
    """
    payload: dataloader wrapping Payload
    label_obj: function operating on a dataset item or list of labels in correct order
    """

    if isinstance(label_obj, list):
        labels_new = label_obj
    elif callable(label_obj):
        labels_new = [label_obj(i) for i in range(len(payload.data_loader.dataset))]
    else:
        raise ValueError("Incorrect label object type -- supply list or function")

    payload.data_loader.dataset.labels[task_name] = labels_new
