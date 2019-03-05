import codecs
import os
import pathlib

import numpy as np
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from metal.utils import padded_tensor, set_seed

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
    reverse_dict = {y: x for x, y in input_dict.items()}
    return input_dict.get, reverse_dict.get


class BERTDataset(data.Dataset):
    """
    Torch dataset object for Glue task, to work with BERT architecture.
    """

    def __init__(
        self,
        tsv_path,
        sent1_idx,
        sent2_idx,
        label_idx,
        skip_rows,
        bert_model,
        delimiter="\t",
        label_fn=lambda x: x,
        inv_label_fn=lambda x: x,
        max_len=-1,
        label_type=int,
        max_datapoints=-1,
        generate_uids=False,
    ):
        """
        Args:
            tsv_path: location of .tsv on disk
            sent1_idx: tsv index for sentence1 (or question)
            sent2_idx: tsv index for sentence2
            label_idx: tsv index for label field
            skip_rows: number of rows to skip (i.e. header rows) in .tsv
            tokenizer: tokenizer to map sentences to tokens using `.tokenize(sent)` method
            delimiter: delimiter between columns (likely '\t') for tab-separated-values
            label_map: dictionary or function mapping from raw labels to desired format
            label_type: data type (int, float) of labels. used to cast values downstream.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        payload = self.load_tsv(
            tsv_path,
            sent1_idx,
            sent2_idx,
            label_idx,
            skip_rows,
            self.tokenizer,
            delimiter,
            label_fn,
            max_len,
            max_datapoints,
            generate_uids,
        )

        if generate_uids:
            (tokens, segments, labels), uids = payload
            self.uids = uids
        else:
            tokens, segments, labels = payload

        self.label_type = label_type
        self.label_fn = label_fn
        self.inv_label_fn = inv_label_fn
        self.tokens = tokens
        self.segments = segments
        self.labels = [labels]

    @staticmethod
    def load_tsv(
        data_file,
        sent1_idx,
        sent2_idx,
        label_idx,
        skip_rows,
        tokenizer,
        delimiter,
        label_fn,
        max_len,
        max_datapoints,
        generate_uids=False,
    ):
        """ Loads and tokenizes .tsv dataset into BERT-friendly sentences / segments.
        Then, sets instance variables self.tokens, self.segments, self.labels.
        """
        # if generating UIDs, must pass in ALL datapoints
        if generate_uids:
            assert max_datapoints == -1
            uids = []

        tokens, segments, labels = [], [], []

        # TODO: Replace a lot of this boilerplate with:
        #  pd.read_csv(filepath, sep='\t', error_bad_lines=False)
        with codecs.open(data_file, "r", "utf-8") as data_fh:
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
                if (
                    len(row) <= sent1_idx
                    or len(row) <= sent2_idx
                    or len(row) <= label_idx
                ):
                    continue
                # tokenize sentences
                sent1_tokenized = tokenizer.tokenize(row[sent1_idx])
                if sent2_idx >= 0:
                    sent2_tokenized = tokenizer.tokenize(row[sent2_idx])
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
                    sent2_ids = tokenizer.convert_tokens_to_ids(
                        sent2_tokenized + ["[SEP]"]
                    )
                else:
                    sent2_ids = []

                # combine sentence pair
                sent = sent1_ids + sent2_ids

                # sentence-pair segments
                seg = [0] * len(sent1_ids) + [1] * len(sent2_ids)

                # process labels
                if label_idx >= 0:
                    label = row[label_idx]
                    label = label_fn(label)
                else:
                    label = -1
                tokens.append(sent)
                segments.append(seg)
                labels.append(label)

                if generate_uids:
                    uids.append(get_uid(data_file, skip_rows + row_idx + 1))

        if generate_uids:
            return (tokens, segments, labels), uids
        else:
            return tokens, segments, labels

    def __getitem__(self, index):
        """Retrieves a single instance with its labels

        Note that the attention mask for each batch is added in the collate function,
        once the max_seq_len for that batch is known.
        """
        x = (self.tokens[index], self.segments[index])
        ys = [labelset[index] for labelset in self.labels]
        return x, ys

    def __len__(self):
        return len(self.tokens)

    def get_dataloader(self, split_prop=None, split_seed=123, **kwargs):
        """Returns a dataloader based on self (dataset). If split_prop is specified,
        returns a split dataset assuming train -> split_prop and dev -> 1 - split_prop."""

        if split_prop:
            assert split_prop > 0 and split_prop < 1

            # choose random indexes for train/dev
            N = len(self)
            full_idx = np.arange(N)
            set_seed(split_seed)
            np.random.shuffle(full_idx)

            # split into train/dev
            split_div = int(split_prop * N)
            train_idx = full_idx[:split_div]
            dev_idx = full_idx[split_div:]

            # create data loaders
            train_dataloader = data.DataLoader(
                self,
                collate_fn=self._collate_fn,
                sampler=SubsetRandomSampler(train_idx),
                **kwargs,
            )

            dev_dataloader = data.DataLoader(
                self,
                collate_fn=self._collate_fn,
                sampler=SubsetRandomSampler(dev_idx),
                **kwargs,
            )

            return train_dataloader, dev_dataloader

        else:
            return data.DataLoader(self, collate_fn=self._collate_fn, **kwargs)

    def _collate_fn(self, batch_list):
        """Collates batch of ((tokens, segments), labels) into padded (X, Ys) tensors

        Args:
            batch_list: a list of tuples containing ((tokens, segments), labels)
        Returns:
            X: instances for BERT: (tok_matrix, seg_matrix, mask_matrix)
            Y: a list of label sets is any [n, :] tensor.
        """
        # batch_size = len(batch_list)
        num_labelsets = len(batch_list[0][1])
        tokens_list = []
        segments_list = []
        Ys = [[] for _ in range(num_labelsets)]

        for instance in batch_list:
            x, ys = instance
            (tokens, segments) = x
            tokens_list.append(tokens)
            segments_list.append(segments)
            for i, y in enumerate(ys):
                Ys[i].append(y)
        tokens_tensor, _ = padded_tensor(tokens_list)
        segments_tensor, _ = padded_tensor(segments_list)
        masks_tensor = torch.gt(tokens_tensor.data, 0)
        assert tokens_tensor.shape == segments_tensor.shape

        X = (tokens_tensor.long(), segments_tensor.long(), masks_tensor.long())
        Ys = self._collate_labels(Ys)
        return X, Ys

    def _collate_labels(self, Ys):
        """Collate potentially multiple labelsets

        Args:
            Ys: a [num_labelsets] list, with each entry containing a list of labels
                (ints, floats, numpy, or torch) belonging to the same labelset
        Returns:
            Ys: a [num_labelsets] list, with each entry containing a torch Tensor
                (padded if necessary) of labels belonging to the same labelset


        Convert each Y in Ys from:
            list of scalars (instance labels) -> [n,] tensor
            list of tensors/lists/arrays (token labels) -> [n, seq_len] padded tensor
        """
        for i, Y in enumerate(Ys):
            if isinstance(Y[0], int):
                Y = torch.tensor(Y, dtype=torch.long)
            elif isinstance(Y[0], np.integer):
                Y = torch.from_numpy(Y)
            elif isinstance(Y[0], float):
                Y = torch.tensor(Y, dtype=torch.float)
            elif isinstance(Y[0], np.float):
                Y = torch.from_numpy(Y)
            elif (
                isinstance(Y[0], list)
                or isinstance(Y[0], np.ndarray)
                or isinstance(Y[0], torch.Tensor)
            ):
                if isinstance(Y[0][0], (int, np.integer)):
                    dtype = torch.long
                elif isinstance(Y[0][0], (float, np.float)):
                    # TODO: WARNING: this may not handle half-precision correctly!
                    dtype = torch.float
                else:
                    msg = f"Unrecognized dtype of elements in label set {i}: {type(Y[0][0])}"
                    raise Exception(msg)
                Y, _ = padded_tensor(Y, dtype=dtype)
            else:
                msg = f"Unrecognized dtype of label set {i}: {type(Y[0])}"
                raise Exception(msg)
            Ys[i] = Y
        return Ys


class QNLIDataset(BERTDataset):
    """
    Torch dataset object for QNLI binary classification task, to work with BERT architecture.
    """

    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"entailment": 1, "not_entailment": 2})
        super(QNLIDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("QNLI", split),
            sent1_idx=1,
            sent2_idx=2,
            label_idx=3 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class STSBDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = (
            lambda x: float(x) / 5,
            lambda x: float(x) * 5,
        )  # labels are scores [1, 2, 3, 4, 5]
        super(STSBDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("STS-B", split),
            sent1_idx=7,
            sent2_idx=8,
            label_idx=9 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            label_type=float,
            **kwargs,
        )


class SST2Dataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        # TODO: why do we want 1 to stay 1?
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})  # reserve 0 for abstain
        super(SST2Dataset, self).__init__(
            tsv_path=tsv_path_for_dataset("SST-2", split),
            sent1_idx=0 if split in ["train", "dev"] else 1,
            sent2_idx=-1,
            label_idx=1 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class COLADataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        super(COLADataset, self).__init__(
            tsv_path=tsv_path_for_dataset("CoLA", split),
            sent1_idx=3 if split in ["train", "dev"] else 1,
            sent2_idx=-1,
            label_idx=1 if split in ["train", "dev"] else -1,
            skip_rows=0 if split in ["train", "dev"] else 1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class MNLIDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        # split = "dev_matched" if split == "dev" else "train"
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
        super(MNLIDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("MNLI", split),
            sent1_idx=8 if split != "diagnostic" else 1,
            sent2_idx=9 if split != "diagnostic" else 2,
            label_idx=gold_cols[split],
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class RTEDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"entailment": 1, "not_entailment": 2})
        super(RTEDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("RTE", split),
            sent1_idx=1,
            sent2_idx=2,
            label_idx=3 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class WNLIDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        super(WNLIDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("WNLI", split),
            sent1_idx=1,
            sent2_idx=2,
            label_idx=3 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class QQPDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        super(QQPDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("QQP", split),
            sent1_idx=3 if split in ["train", "dev"] else 1,
            sent2_idx=4 if split in ["train", "dev"] else 2,
            label_idx=5 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


class MRPCDataset(BERTDataset):
    def __init__(self, split, bert_model, **kwargs):
        label_fn, inv_label_fn = get_label_fn({"1": 1, "0": 2})
        super(MRPCDataset, self).__init__(
            tsv_path=tsv_path_for_dataset("MRPC", split),
            sent1_idx=3,
            sent2_idx=4,
            label_idx=0,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            **kwargs,
        )


# ----------Exotic Datasets---------


class PairwiseRankingSampler(Sampler):
    r"""Samples consecutive pairs randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (
            int(2 * self.indices[i] + j)
            for i in torch.randperm(len(self.indices))
            for j in range(2)
        )

    def __len__(self):
        return 2 * len(self.indices)


class QNLIRDataset(BERTDataset):
    """
    Torch dataset object for QNLI pairwise ranking task, to work with BERT architecture.
    The input tsv should be sorted such that every two consecutive lines
    are pairs of positive and negative examples.
    """

    def __init__(self, split, bert_model, max_datapoints=-1, **kwargs):
        self.split = split
        max_datapoints *= 2  # make sure we take pairs
        if self.split == "train":
            dataset_folder = "QNLIR"
        else:
            dataset_folder = "QNLI"
        label_fn, inv_label_fn = get_label_fn({"entailment": 1, "not_entailment": 2})
        super(QNLIRDataset, self).__init__(
            tsv_path=tsv_path_for_dataset(dataset_folder, split),
            sent1_idx=1,
            sent2_idx=2,
            label_idx=3 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            max_datapoints=max_datapoints,
            label_type=float,
            **kwargs,
        )
        if self.split == "train":
            assert len(self.tokens) % 2 == 0

    def get_dataloader(self, split_prop=None, split_seed=123, **kwargs):
        """Returns a dataloader based on self (dataset). If split_prop is specified,
        returns a split dataset assuming train -> split_prop and dev -> 1 - split_prop."""
        if self.split == "train":
            # choose random indices
            num_pairs = len(self) / 2
            full_idx = np.arange(num_pairs)
            np.random.seed(split_seed)
            np.random.shuffle(full_idx)
            assert kwargs["batch_size"] % 2 == 0
            if split_prop:
                assert split_prop >= 0 and split_prop <= 1

                # split into train/dev
                split_div = int(split_prop * num_pairs)
                train_idx = full_idx[:split_div]
                dev_idx = full_idx[split_div:]

                # create data loaders
                train_dataloader = data.DataLoader(
                    self,
                    collate_fn=self._collate_fn,
                    sampler=PairwiseRankingSampler(train_idx),
                    **kwargs,
                )

                dev_dataloader = data.DataLoader(
                    self,
                    collate_fn=self._collate_fn,
                    sampler=PairwiseRankingSampler(dev_idx),
                    **kwargs,
                )

                return train_dataloader, dev_dataloader
            else:
                # create data loaders
                train_dataloader = data.DataLoader(
                    self,
                    collate_fn=self._collate_fn,
                    sampler=PairwiseRankingSampler(full_idx),
                    **kwargs,
                )
                return train_dataloader
        else:
            return data.DataLoader(self, collate_fn=self._collate_fn, **kwargs)
