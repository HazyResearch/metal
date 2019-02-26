import codecs
import os

import numpy as np
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

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
        include_segments=True,
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
            include_segments: __getitem__() will return:
                True: (tokens, segment), labels
                False: tokens, labels
        """
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        tokens, segments, labels = self.load_tsv(
            tsv_path,
            sent1_idx,
            sent2_idx,
            label_idx,
            skip_rows,
            tokenizer,
            delimiter,
            label_fn,
            max_len,
            max_datapoints,
        )
        self.label_type = label_type
        self.label_fn = label_fn
        self.inv_label_fn = inv_label_fn
        self.tokens = tokens
        self.segments = segments
        self.labels = labels
        self.include_segments = include_segments

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
    ):
        """ Loads and tokenizes .tsv dataset into BERT-friendly sentences / segments.
        Then, sets instance variables self.tokens, self.segments, self.labels.
        """

        tokens, segments, labels = [], [], []
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
        return tokens, segments, labels

    def __getitem__(self, index):
        return (self.tokens[index], self.segments[index]), self.labels[index]

    def __len__(self):
        return len(self.tokens)

    def get_dataloader(self, split_prop=None, split_seed=123, **kwargs):
        """Returns a dataloader based on self (dataset). If split_prop is specified,
        returns a split dataset assuming train -> split_prop and dev -> 1 - split_prop."""

        if split_prop:
            assert split_prop >= 0 and split_prop <= 1

            # choose random indexes for train/dev
            N = len(self)
            full_idx = np.arange(N)
            np.random.seed(split_seed)
            np.random.shuffle(full_idx)

            # split into train/dev
            split_div = int(split_prop * N)
            train_idx = full_idx[:split_div]
            dev_idx = full_idx[split_div:]

            # create data loaders
            train_dataloader = data.DataLoader(
                self,
                collate_fn=lambda batch: self._collate_fn(batch),
                sampler=SubsetRandomSampler(train_idx),
                **kwargs,
            )

            dev_dataloader = data.DataLoader(
                self,
                collate_fn=lambda batch: self._collate_fn(batch),
                sampler=SubsetRandomSampler(dev_idx),
                **kwargs,
            )

            return train_dataloader, dev_dataloader

        else:
            return data.DataLoader(
                self, collate_fn=lambda batch: self._collate_fn(batch), **kwargs
            )

    def _collate_fn(self, batch):
        """ Collates batch of (tokens, segments, labels), collates into tensors of
        ((token_idx_matrix, seg_matrix, mask_matrix), label_matrix). Handles padding
        based on specific max_len.
        """
        batch_size = len(batch)
        # max_len == -1 defaults to using max_sent_len
        max_sent_len = int(np.max([len(tok) for ((tok, seg), _) in batch]))
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)
        seg_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)
        label_dtype = np.float if self.label_type is float else np.int
        label_matrix = np.zeros((batch_size), dtype=label_dtype)

        for idx1 in np.arange(len(batch)):
            (tokens, segments), labels = batch[idx1]
            label_matrix[idx1] = labels
            for idx2 in np.arange(len(tokens)):
                if idx2 >= max_sent_len:
                    break
                idx_matrix[idx1, idx2] = tokens[idx2]
                seg_matrix[idx1, idx2] = segments[idx2]

        idx_matrix = torch.LongTensor(idx_matrix)
        seg_matrix = torch.LongTensor(seg_matrix)
        mask_matrix = torch.gt(idx_matrix.data, 0).long()

        # cast torch labels
        if self.label_type is float:
            label_matrix = torch.FloatTensor(label_matrix)
        else:
            label_matrix = torch.LongTensor(label_matrix)

        if self.include_segments:
            return (idx_matrix, seg_matrix, mask_matrix), label_matrix
        else:
            return idx_matrix, label_matrix


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
                    collate_fn=lambda batch: self._collate_fn(batch),
                    sampler=PairwiseRankingSampler(train_idx),
                    **kwargs,
                )

                dev_dataloader = data.DataLoader(
                    self,
                    collate_fn=lambda batch: self._collate_fn(batch),
                    sampler=PairwiseRankingSampler(dev_idx),
                    **kwargs,
                )

                return train_dataloader, dev_dataloader
            else:
                # create data loaders
                train_dataloader = data.DataLoader(
                    self,
                    collate_fn=lambda batch: self._collate_fn(batch),
                    sampler=PairwiseRankingSampler(full_idx),
                    **kwargs,
                )
                return train_dataloader
        else:
            return data.DataLoader(
                self, collate_fn=lambda batch: self._collate_fn(batch), **kwargs
            )
