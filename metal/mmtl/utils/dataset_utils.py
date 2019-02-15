import metal.mmtl.dataset as dataset


def get_all_dataloaders(
    dataset_name,
    bert_model,
    max_len,
    dl_kwargs,
    split_prop,
    max_datapoints,
    verbose=True,
):
    """ Initializes train/dev/test dataloaders given dataset_class"""

    if verbose:
        print(f"Loading {dataset_name} Dataset")

    dataset_cls = getattr(dataset, dataset_name.upper() + "Dataset")

    if not split_prop:
        # When split_prop is None, we use standard train/dev/test splits.

        # train
        train_ds = dataset_cls(
            split="train",
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )
        train_dl = train_ds.get_dataloader(**dl_kwargs)

        # dev
        dev_ds = dataset_cls(
            split="dev",
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )
        dev_dl = dev_ds.get_dataloader(**dl_kwargs)

        # test for leader-board submission
        test_ds = dataset_cls(
            split="test",
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )
        test_dl = test_ds.get_dataloader(**dl_kwargs)
    else:
        # When split_prop is not None, we use create an artificial dev set from the train set.

        # split train -> artificial train/dev
        train_ds = dataset_cls(
            split="train",
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )
        train_dl, dev_dl = train_ds.get_dataloader(split_prop=split_prop, **dl_kwargs)

        # treat dev -> test
        test_ds = dataset_cls(
            split="dev",
            bert_model=bert_model,
            max_len=max_len,
            max_datapoints=max_datapoints,
        )
        test_dl = test_ds.get_dataloader(**dl_kwargs)

    return {"train": train_dl, "valid": dev_dl, "test": test_dl}
