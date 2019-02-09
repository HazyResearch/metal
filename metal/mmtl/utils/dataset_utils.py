import os

import metal.mmtl.dataset as dataset


def get_all_dataloaders(
    dataset_name,
    bert_model,
    train_dev_split_prop=0.8,
    max_len=512,
    dl_kwargs={},
    verbose=True,
):
    """ Initializes train/dev/test dataloaders given dataset_class"""

    if verbose:
        print(f"Loading {dataset_name} Dataset")

    dataset_cls = getattr(dataset, dataset_name.upper() + "Datasets")

    # split train -> artificial train/dev
    train_ds = dataset_cls(split="train", bert_model=bert_model, max_len=max_len)
    train_dl, dev_dl = train_ds.get_dataloader(
        split_prop=train_dev_split_prop, **dl_kwargs
    )

    # treat dev -> test
    test_ds = dataset_cls(split="dev", bert_model=bert_model, max_len=max_len)
    test_dl = test_ds.get_dataloader(**dl_kwargs)

    return {"train": train_dl, "valid": dev_dl, "test": test_dl}
