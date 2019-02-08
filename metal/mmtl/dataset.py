import codecs
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm


class BERTDataset(data.Dataset):
    """
    Torch dataset object for Glue task, to work with BERT architecture.
    """

    def __init__(
        self,
        dataset_name,
        dataset_split,
        sent1_idx=0,
        sent2_idx=-1,
        label_idx=1,
        skip_rows=0,
        bert_model="bert-base-uncased",
        delimiter="\t",
        label_fn=None,
        max_len=-1,
        label_type=int,
    ):
        """
        Args:
            src_path: path name of .tsv file for this dataset split.
            sent1_idx: tsv index for sentence1
            sent2_idx: tsv index for sentence2
            label_idx: tsv index for label field
            skip_rows: number of rows to skip (i.e. header rows) in .tsv
            tokenizer: tokenizer to map sentences to tokens using `.tokenize(sent)` method
            delimiter: delimiter between columns (likely '\t') for tab-separated-values
            label_fn: function mapping from raw labels to desired format
            label_type: data type (int, float) of labels. used to cast values downstream.
        """
        src_path = os.path.join(
            os.environ["GLUEDATA"], "{}/{}.tsv".format(dataset_name, dataset_split)
        )
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        tokens, segments, labels = self.load_tsv(
            src_path,
            sent1_idx,
            sent2_idx,
            label_idx,
            skip_rows,
            tokenizer,
            delimiter,
            label_fn,
            max_len,
        )
        self.label_type = label_type

        self.tokens = tokens
        self.segments = segments
        self.labels = labels

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
            for row_idx, row in tqdm(list(enumerate(data_fh))):
                row = row.strip().split(delimiter)

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
                    if label_fn:
                        label = label_fn(label)
                else:
                    label = 0

                tokens.append(sent)
                segments.append(seg)
                labels.append(label)
        return tokens, segments, labels

    def __getitem__(self, index):
        return (self.tokens[index], self.segments[index]), self.labels[index]

    def __len__(self):
        return len(self.tokens)

    def get_dataloader(self, **kwargs):
        """Initializes a dataloader based on self (dataset)."""

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

        return (idx_matrix, seg_matrix, mask_matrix), label_matrix


class QNLI(BERTDataset):
    """
    Torch dataset object for QNLI ranking task, to work with BERT architecture.
    """

    def __init__(self, split, bert_model, max_len=-1):
        super(QNLI, self).__init__(
            dataset_name="QNLI",
            dataset_split=split,
            sent1_idx=1,
            sent2_idx=2,
            label_idx=3 if split in ["train", "dev"] else -1,
            skip_rows=1,
            bert_model=bert_model,
            delimiter="\t",
            label_fn=lambda label: 1 if label == "entailment" else 2,
            max_len=max_len,
        )


class SSTB(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class CoLA(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class STS(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class MNLI(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class RTE(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class WNLI(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class QQP(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError


class MRPC(BERTDataset):
    def __init__(self, split, bert_model, max_len=-1):
        raise NotImplementedError
