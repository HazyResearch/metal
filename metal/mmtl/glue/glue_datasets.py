import os
import pathlib
from collections import defaultdict

import numpy as np
import spacy
import torch
import torch.utils.data as data
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from tqdm import tqdm

from metal.mmtl.glue.glue_preprocess import get_task_tsv_config, load_tsv
from metal.utils import padded_tensor, set_seed

nlp = spacy.load("en_core_web_sm")


def get_glue_dataset(dataset_name, split, bert_vocab, **kwargs):
    """ Create and returns specified glue dataset from data in tsv file."""
    config = get_task_tsv_config(dataset_name, split)

    return GLUEDataset.from_tsv(
        # class kwargs
        dataset_name,
        bert_vocab=bert_vocab,
        # load_tsv kwargs
        tsv_path=config["tsv_path"],
        sent1_idx=config["sent1_idx"],
        sent2_idx=config["sent2_idx"],
        label_idx=config["label_idx"],
        skip_rows=config["skip_rows"],
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
        dataset_name,
        sentences,
        labels,
        label_type,
        label_fn,
        inv_label_fn,
        max_len=0,
        bert_vocab=None,
        tokenize_bert=True,
        run_spacy=False,
        uids=None,
    ):
        """
        Args:
            sentences: [n] list of lists containing a string for each sentence
            labels: [n] list of labels (int or float for glue tasks, but potentially
                anything)
            label_type: data type (int, float) of labels. used to cast values downstream.
            label_fn: dictionary or function mapping from raw labels to desired format
            inv_label_fn: inverse function mapping format back to raw labels
            max_len: if non-zero, truncate inputs to a maximum of this many tokens

        After tokenization:
            bert_tokens: an [n] list of longs corresponding to the indices of each
                bert-tokenized wordpiece token in the vocabulary for that bert model
            bert_segments: an [n] list of segment masks indicating whether each token
                is in sent1/sent2 (e.g. [[0, 0, 0, 0, 1, 1, 1], ...])
        """
        self.sentences = sentences
        label_name = f"{dataset_name}_gold"
        self.labels = {label_name: torch.tensor(labels, dtype=label_type).view(-1, 1)}
        self.label_fn = label_fn
        self.inv_label_fn = inv_label_fn
        self.uids = uids

        if tokenize_bert:
            assert bert_vocab is not None
            bert_tokenizer, bert_tokens, bert_segments = self.tokenize_bert(
                bert_vocab, max_len
            )
            self.bert_tokenizer = bert_tokenizer
            self.bert_tokens = bert_tokens
            self.bert_segments = bert_segments
        if run_spacy:
            self.spacy_tokens = self.run_spacy()

    def __getitem__(self, index):
        """Retrieves a single instance with its labels

        Note that the attention mask for each batch is added in the collate function,
        once the max_seq_len for that batch is known.
        """
        x = {"data": self.bert_tokens[index], "segments": self.bert_segments[index]}
        ys = {
            label_name: label_set[index]
            for label_name, label_set in self.labels.items()
        }
        return x, ys

    def __len__(self):
        return len(self.bert_tokens)

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
            Y: a dict of {label_name: labels} where labels[idx] are the appropriate
                labels for that task
        """
        tokens_list = []
        segments_list = []
        Y_lists = {label_name: [] for label_name in self.labels}

        for instance in batch_list:
            x, ys = instance
            tokens_list.append(x["data"])
            segments_list.append(x["segments"])
            for label_name, y in ys.items():
                Y_lists[label_name].append(y)
        tokens_tensor = padded_tensor(tokens_list)
        segments_tensor = padded_tensor(segments_list)
        masks_tensor = torch.gt(tokens_tensor.data, 0)
        assert tokens_tensor.shape == segments_tensor.shape

        X = {
            "data": tokens_tensor.long(),
            "segments": segments_tensor.long(),
            "masks": masks_tensor.long(),
        }
        Ys = self._collate_labels(Y_lists)
        return X, Ys

    def _collate_labels(self, Ys):
        """Collate potentially multiple label_sets

        Args:
            Ys: a dict of the form {task_name: label_list}, where label_list is a
                list of individual labels (ints, floats, numpy, or torch) belonging to
                the same label_set; labels may be a scalar or a sequence.
        Returns:
            Ys: a dict of the form {task_name: labels}, with labels containing a torch
                Tensor (padded if necessary) of labels belonging to the same label_set

        Convert each Y in Ys from:
            list of scalars (instance labels) -> [n,] tensor
            list of tensors/lists/arrays (token labels) -> [n, seq_len] padded tensor
        """
        for label_name, Y in Ys.items():
            if isinstance(Y[0], int):
                Y = torch.tensor(Y, dtype=torch.long)
            elif isinstance(Y[0], float):
                Y = torch.tensor(Y, dtype=torch.float)
            elif isinstance(Y[0], np.integer):
                Y = torch.from_numpy(Y)
            elif isinstance(Y[0], np.float):
                Y = torch.from_numpy(Y)
            elif (
                isinstance(Y[0], list)
                or isinstance(Y[0], np.ndarray)
                or isinstance(Y[0], torch.Tensor)
            ):
                Y = padded_tensor(Y)
            else:
                msg = f"Unrecognized dtype of label_set {label_name}: " f"{type(Y[0])}"
                raise Exception(msg)
            # Ensure that first dimension of Y is n
            if Y.dim() == 1:
                Y = Y.view(-1, 1)
            Ys[label_name] = Y
        return Ys

    def tokenize_bert(self, bert_vocab, max_len):
        # Initialize BERT tokenizer
        do_lower_case = "uncased" in bert_vocab
        tokenizer = BertTokenizer.from_pretrained(
            bert_vocab, do_lower_case=do_lower_case
        )
        bert_tokens = []
        bert_segments = []

        for sentence_pair in self.sentences:
            assert len(sentence_pair) in [1, 2]

            # Tokenize sentences
            tokenized_sents = [tokenizer.tokenize(sent) for sent in sentence_pair]
            sent1_tokens = tokenized_sents[0]
            sent2_tokens = tokenized_sents[1] if len(tokenized_sents) > 1 else None

            # Truncate if necessary
            if max_len > 0:
                if sent2_tokens:
                    # Remove tokens from the longer sentence
                    # Save room for [CLS] and [SEP] x 2
                    while len(sent1_tokens) + len(sent2_tokens) > max_len - 3:
                        if len(sent1_tokens) > len(sent2_tokens):
                            sent1_tokens.pop()
                        else:
                            sent2_tokens.pop()
                else:
                    # Remove tokens from the only sentence
                    if len(sent1_tokens) > max_len - 2:  # Save room for [CLS], [SEP]
                        sent1_tokens = sent1_tokens[: max_len - 2]

            # Add markers
            sent1_tokens = ["[CLS]"] + sent1_tokens + ["[SEP]"]
            if sent2_tokens:
                sent2_tokens += ["[SEP]"]
            else:
                sent2_tokens = []
            token_ids = tokenizer.convert_tokens_to_ids(sent1_tokens + sent2_tokens)
            segments = [0] * len(sent1_tokens) + [1] * len(sent2_tokens)

            bert_tokens.append(torch.LongTensor(token_ids))
            bert_segments.append(torch.LongTensor(segments))

        return tokenizer, bert_tokens, bert_segments

    def run_spacy(self):
        nlp_out = []
        print("Applying spacy to all sentence pairs in dataset")
        for sentence_pair in tqdm(self.sentences):
            nlp_out.append([nlp(sent) for sent in sentence_pair])
        return nlp_out

    @classmethod
    def from_tsv(
        # class kwargs
        cls,
        dataset_name,
        bert_vocab,
        # load_tsv kwargs
        tsv_path,
        sent1_idx,
        sent2_idx,
        label_idx,
        skip_rows,
        delimiter="\t",
        label_fn=lambda x: x,
        inv_label_fn=lambda x: x,
        label_type=torch.long,
        # class kwargs
        max_len=-1,
        max_datapoints=-1,
        generate_uids=False,
        tokenize_bert=True,
        run_spacy=False,
    ):

        # load and preprocess data from tsv
        out = load_tsv(
            tsv_path=tsv_path,
            sent1_idx=sent1_idx,
            sent2_idx=sent2_idx,
            label_idx=label_idx,
            skip_rows=skip_rows,
            delimiter=delimiter,
            label_fn=label_fn,
            max_datapoints=max_datapoints,
            generate_uids=generate_uids,
        )

        if generate_uids:
            (sentences, labels), uids = out
        else:
            sentences, labels = out
            uids = None

        # initialize class with data
        return cls(
            dataset_name,
            sentences=sentences,
            labels=labels,
            label_type=label_type,
            label_fn=label_fn,
            inv_label_fn=inv_label_fn,
            max_len=max_len,
            bert_vocab=bert_vocab,
            tokenize_bert=tokenize_bert,
            run_spacy=run_spacy,
            uids=uids,
        )
