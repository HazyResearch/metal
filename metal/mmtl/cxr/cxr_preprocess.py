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
        os.environ["CXRDATA"], "{}/{}.tsv".format(dataset_name, dataset_split)
    )

def get_label_fn(input_dict):
    """ Given mapping (specified as dict), return two-way functions for mapping."""
    reverse_dict = {y: x for x, y in input_dict.items()}
    return input_dict.get, reverse_dict.get


def get_task_tsv_config(dataset_name, split):
    """ Returns the tsv_config to be used in params of CXRDataset.from_tsv for
    specific task and split. """

    if dataset_name == "CXR8":
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        return {
            "path_to_labels": tsv_path_for_dataset("CXR8", split),
            "path_to_images": "/lfs/1/jdunnmon/data/nih/images/images" 
            "transform": 2,
            "sample": ,
            "finding": 1,
            "label_type": int,
        }

def load_tsv(
    tsv_path,
    label_idx,
    skip_rows,
    delimiter="\t",
    label_fn=lambda x: x,
    max_datapoints=-1,
    generate_uids=False,
):
    """ Loads and tokenizes .tsv dataset into BERT-friendly sentences / segments.
    Then, sets instance variables self.tokens, self.segments, self.labels.

    Args:
        tsv_path: location of .tsv on disk
        label_idx: tsv index for label field
        skip_rows: number of rows to skip (i.e. header rows) in .tsv
        delimiter: delimiter between columns (likely '\t') for tab-separated-values
        label_fn: maps labels to desired format (usually for training)
        max_datapoints: maximum len of the dataset.
            used for debugging without loading all data.
        generate_uids: whether to return uids in addition to payload
    """
    # if generating UIDs, must pass in ALL datapoints
    if generate_uids:
        assert max_datapoints == -1
        uids = []

    sentences = []
    labels = []

    # TODO: Replace a lot of this boilerplate with:
    #  pd.read_csv(filepath, sep='\t', error_bad_lines=False)
    with codecs.open(tsv_path, "r", "utf-8") as data_fh:
        # skip "header" rows
        num_cols = None
        for _ in range(skip_rows):
            headers = data_fh.readline()
            headers = headers.strip().split(delimiter)
            num_cols = len(headers)

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
                print(
                    "WARNING: skipping example due to sentence or label index greater than number of columns"
                )
                continue
            if num_cols is not None and num_cols != len(row):
                print("WARNING: skipping example with mismatched number of columns")
                continue
            if generate_uids:
                # Creates UID to match line number in specificed file, accounting for header rows/1-idx
                uids.append(get_uid(tsv_path, skip_rows + row_idx + 1))

            # process labels
            if label_idx >= 0:
                label = row[label_idx]
                label = label_fn(label)
            else:
                label = -1
            labels.append(label)

            sent1 = row[sent1_idx] if sent1_idx >= 0 else None
            sent2 = row[sent2_idx] if sent2_idx >= 0 else None

            assert sent1 is not None
            sentence_list = [sent1] if sent2 is None else [sent1, sent2]
            sentences.append(sentence_list)

    if generate_uids:
        return (sentences, labels), uids
    else:
        return (sentences, labels)
