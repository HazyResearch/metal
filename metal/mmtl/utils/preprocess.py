import codecs
import os
import pathlib

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    # Only use tqdm notebook if not in travis testing
    if "CI" not in os.environ:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm


def tsv_path_for_dataset(dataset_name, dataset_split):
    """ Returns dataset location on disk given name and split. """
    return os.path.join(
        os.environ["GLUEDATA"], "{}/{}.tsv".format(dataset_name, dataset_split)
    )


def get_uid(path, line):
    """ Returns unique ID for example in this path/line number"""
    # remove the GLUEDATA directory from path
    p = pathlib.Path(path)
    glue_dir = pathlib.Path(os.environ["GLUEDATA"])
    path_suffix = p.relative_to(glue_dir)

    return f"{path_suffix}:{line}"


def get_label_fn(input_dict):
    """ Given mapping (specified as dict), return two-way functions for mapping."""
    reverse_dict = {y: x for x, y in input_dict.items()}
    return input_dict.get, reverse_dict.get


def get_task_tsv_config(task_name, split):
    """ Returns the tsv_config to be used in params of GLUEDataset.form_tsv for
    specific task and split. """

    if task_name == "QNLI":
        label_fn, inv_label_fn = get_label_fn({"entailment": 1, "not_entailment": 2})
        return {
            "tsv_path": tsv_path_for_dataset("QNLI", split),
            "sent1_idx": 1,
            "sent2_idx": 2,
            "label_idx": 3 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "STSB":
        label_fn, inv_label_fn = (
            lambda x: float(x) / 5,
            lambda x: float(x) * 5,
        )  # labels are continuous [0, 5]
        return {
            "tsv_path": tsv_path_for_dataset("STS-B", split),
            "sent1_idx": 7,
            "sent2_idx": 8,
            "label_idx": 9 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": float,
        }
    elif task_name == "SST2":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})  # reserve 0 for abstain
        return {
            "tsv_path": tsv_path_for_dataset("SST-2", split),
            "sent1_idx": 0 if split in ["train", "dev"] else 1,
            "sent2_idx": -1,
            "label_idx": 1 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "COLA":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "tsv_path": tsv_path_for_dataset("CoLA", split),
            "sent1_idx": 3 if split in ["train", "dev"] else 1,
            "sent2_idx": -1,
            "label_idx": 1 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "MNLI":
        gold_cols = {
            "train": 11,
            "dev": 15,
            "dev_mismatched": 15,
            "dev_matched": 15,
            "test": -1,
            "test_mismatched": -1,
            "test_matched": -1,
            "diagnostic": -1,
        }

        label_fn, inv_label_fn = get_label_fn(
            {"entailment": 1, "contradiction": 2, "neutral": 3}
        )

        return {
            "tsv_path": tsv_path_for_dataset("MNLI", split),
            "sent1_idx": 8 if split != "diagnostic" else 1,
            "sent2_idx": 9 if split != "diagnostic" else 2,
            "label_idx": gold_cols[split],
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "RTE":
        label_fn, inv_label_fn = get_label_fn({"entailment": 1, "not_entailment": 2})
        return {
            "tsv_path": tsv_path_for_dataset("RTE", split),
            "sent1_idx": 1,
            "sent2_idx": 2,
            "label_idx": 3 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "WNLI":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "tsv_path": tsv_path_for_dataset("WNLI", split),
            "sent1_idx": 1,
            "sent2_idx": 2,
            "label_idx": 3 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "QQP":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "tsv_path": tsv_path_for_dataset("QQP", split),
            "sent1_idx": 3 if split in ["train", "dev"] else 1,
            "sent2_idx": 4 if split in ["train", "dev"] else 2,
            "label_idx": 5 if split in ["train", "dev"] else -1,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    elif task_name == "MRPC":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "tsv_path": tsv_path_for_dataset("MRPC", split),
            "sent1_idx": 3,
            "sent2_idx": 4,
            "label_idx": 0,
            "skip_rows": 1,
            "label_fn": label_fn,
            "inv_label_fn": inv_label_fn,
            "label_type": int,
        }
    else:
        raise ValueError(f"{task_name} not found!")


def load_tsv(
    tsv_path,
    sent1_idx,
    sent2_idx,
    label_idx,
    skip_rows,
    tokenizer=None,
    delimiter="\t",
    label_fn=lambda x: x,
    max_len=-1,
    max_datapoints=-1,
    generate_uids=False,
):
    """ Loads and tokenizes .tsv dataset into BERT-friendly sentences / segments.
    Then, sets instance variables self.tokens, self.segments, self.labels.

    Args:
        tsv_path: location of .tsv on disk
        sent1_idx: tsv index for sentence1 (or question)
        sent2_idx: tsv index for sentence2
        label_idx: tsv index for label field
        skip_rows: number of rows to skip (i.e. header rows) in .tsv
        tokenizer: tokenizer to map sentences to tokens using `.tokenize(sent)` method
            if None, returns raw examples.
        delimiter: delimiter between columns (likely '\t') for tab-separated-values
        label_fn: maps labels to desired format (usually for training)
        max_len: maximum sequence length of each sentence
        max_datapoints: maximum len of the dataset.
            used for debugging without loading all data.
        generate_uids: whether to return uids in addition to payload
    """
    # if generating UIDs, must pass in ALL datapoints
    if generate_uids:
        assert max_datapoints == -1
        uids = []

    return_raw = tokenizer is None
    labels = []
    if return_raw:
        raw_examples = []
    else:
        tokens, segments = [], []

    # TODO: Replace a lot of this boilerplate with:
    #  pd.read_csv(filepath, sep='\t', error_bad_lines=False)
    with codecs.open(tsv_path, "r", "utf-8") as data_fh:
        # skip "header" rows
        for _ in range(skip_rows):
            data_fh.readline()

        # process data rows
        rows = list(enumerate(data_fh))
        if max_datapoints > 0:
            rows = rows[:max_datapoints]
        for row_idx, row in tqdm(rows):
            # only look at top max_datapoints examples for debugging
            if max_datapoints > 0:
                if row_idx > max_datapoints:
                    break
            row = row.strip().split(delimiter)
            if len(row) <= sent1_idx or len(row) <= sent2_idx or len(row) <= label_idx:
                continue

            if generate_uids:
                uids.append(get_uid(tsv_path, skip_rows + row_idx + 1))

            # process labels
            if label_idx >= 0:
                label = row[label_idx]
                label = label_fn(label)
            else:
                label = -1
            labels.append(label)

            sent1 = row[sent1_idx]
            sent2 = row[sent2_idx]

            if return_raw:
                raw_examples.append({"sent1": sent1, "sent2": sent2})
                # no need to proceed with remainder of processing
                continue

            # tokenize sentences
            sent1_tokenized = tokenizer.tokenize(sent1)
            if sent2_idx >= 0:
                sent2_tokenized = tokenizer.tokenize(sent2)
            else:
                sent2_tokenized = []

            # truncate sequences if number of tokens is greater than max_len
            if max_len > 0:
                if sent2_idx >= 0:
                    # remove tokens from the longer sequence
                    while len(sent1_tokenized) + len(sent2_tokenized) > max_len - 3:
                        if len(sent1_tokenized) > len(sent2_tokenized):
                            sent1_tokenized.pop()
                        else:
                            sent2_tokenized.pop()
                else:
                    # remove tokens from sentence 2 to match max_len
                    if len(sent1_tokenized) > max_len - 2:
                        sent1_tokenized = sent1_tokenized[: max_len - 2]

            # convert to token ids
            sent1_ids = tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + sent1_tokenized + ["[SEP]"]
            )
            if sent2_idx >= 0:
                sent2_ids = tokenizer.convert_tokens_to_ids(sent2_tokenized + ["[SEP]"])
            else:
                sent2_ids = []

            # combine sentence pair
            sent = sent1_ids + sent2_ids

            # sentence-pair segments
            seg = [0] * len(sent1_ids) + [1] * len(sent2_ids)

            tokens.append(sent)
            segments.append(seg)

    if return_raw:
        payload = (raw_examples, labels)
    else:
        payload = (tokens, segments, labels)

    if generate_uids:
        return payload, uids
    else:
        return payload
