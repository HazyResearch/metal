import os

import numpy as np
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from metal.mmtl.utils.preprocess import get_task_tsv_config, load_tsv
from metal.utils import set_seed


def get_glue_dataset(task_name, split, bert_vocab, **kwargs):
    """ Create and returns specified glue dataset from data in tsv file."""
    config = get_task_tsv_config(task_name, split)

    return GLUEDataset.from_tsv(
        tsv_path=config["tsv_path"],
        sent1_idx=config["sent1_idx"],
        sent2_idx=config["sent2_idx"],
        label_idx=config["label_idx"],
        skip_rows=config["skip_rows"],
        bert_vocab=bert_vocab,
        delimiter="\t",
        label_fn=config["label_fn"],
        inv_label_fn=config["inv_label_fn"],
        label_type=config["label_type"],
        **kwargs,
    )


class GLUEDataset(data.Dataset):
    """
    Torch dataset object for Glue task, to work with BERT architecture.
    """

    def __init__(
        self,
        tokens,
        segments,
        labels,
        label_type,
        label_fn,
        inv_label_fn,
        include_segments=True,
    ):
        """
        Args:
            tokens: list of sentences (lists) containing token indexes
            segments: list of segment masks indicating whether each token in sent1/sent2
                e.g. [[0, 0, 0, 0, 1, 1, 1], ...]
            labels: list of labels (int or float)
            label_type: data type (int, float) of labels. used to cast values downstream.
            label_fn: dictionary or function mapping from raw labels to desired format
            inv_label_fn: inverse function mapping format back to raw labels
            include_segments: __getitem__() will return:
                True: (tokens, segment), labels
                False: tokens, labels
        """
        self.tokens = tokens
        self.segments = segments
        self.labels = labels
        self.label_type = label_type
        self.label_fn = label_fn
        self.inv_label_fn = inv_label_fn
        self.include_segments = include_segments

    def __getitem__(self, index):
        return (self.tokens[index], self.segments[index]), self.labels[index]

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

    @classmethod
    def from_tsv(
        cls,
        tsv_path,
        sent1_idx,
        sent2_idx,
        label_idx,
        skip_rows,
        bert_vocab,
        delimiter="\t",
        label_fn=lambda x: x,
        inv_label_fn=lambda x: x,
        max_len=-1,
        label_type=int,
        max_datapoints=-1,
        generate_uids=False,
        include_segments=True,
    ):

        # initialize BERT tokenizer
        do_lower_case = "uncased" in bert_vocab
        tokenizer = BertTokenizer.from_pretrained(
            bert_vocab, do_lower_case=do_lower_case
        )

        # load and preprocess data from tsv
        tokens, segments, labels = load_tsv(
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
            generate_uids,
        )

        # initialize class with data
        return cls(
            tokens,
            segments,
            labels,
            label_type,
            label_fn,
            inv_label_fn,
            include_segments,
        )
