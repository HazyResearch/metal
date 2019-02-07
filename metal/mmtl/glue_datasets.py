import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


class GlueDataset(data.Dataset):
    """
    Torch dataset object for Glue task. Each specific task should override
    the preprocess_data and load_data methods.
    """

    def __init__(self, src_path, tokenizer):
        super(GlueDataset, self).__init__()
        self.src_path = src_path
        self.tokenizer = tokenizer
        self.raw_data = None
        self.tokens = []
        self.segments = []
        self.labels = []

    def __getitem__(self, index):
        return (self.tokens[index], self.segments[index]), self.labels[index]

    def __len__(self):
        return self.raw_data.shape[0]

    def load_data(self):
        """
        Loads data from source file (e.g. tsv of csv file)
        """
        raise NotImplementedError

    def preprocess_data(self):
        """
        Processes raw data and tokenizes it.
        """
        raise NotImplementedError

    def get_dataloader(self, max_len=-1, batch_size=1):
        return data.DataLoader(
            self,
            collate_fn=lambda batch: self.collate_fn(batch, max_len),
            batch_size=batch_size,
            shuffle=True,
        )

    def collate_fn(self, batch, max_len):
        batch_size = len(batch)
        max_sent_len = int(np.max([len(tok) for ((tok, seg), _) in batch]))
        if (max_len > 0) and (max_len < max_sent_len):
            max_sent_len = max_len
        idx_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)
        seg_matrix = np.zeros((batch_size, max_sent_len), dtype=np.int)
        label_matrix = np.zeros((batch_size, 1), dtype=np.int)

        for idx1 in np.arange(len(batch)):
            (tokens, segments), labels = batch[idx1]
            label_matrix[idx1, :] = labels
            for idx2 in np.arange(len(tokens)):
                if idx2 >= max_sent_len:
                    break
                idx_matrix[idx1, idx2] = tokens[idx2]
                seg_matrix[idx1, idx2] = segments[idx2]

        idx_matrix = torch.LongTensor(idx_matrix)
        seg_matrix = torch.LongTensor(seg_matrix)
        mask_matrix = torch.eq(idx_matrix.data, -1).long()
        label_matrix = torch.LongTensor(label_matrix)
        return (idx_matrix, seg_matrix, mask_matrix), label_matrix


class QNLIDataset(GlueDataset):
    def __init__(self, src_path, tokenizer):
        super(QNLIDataset, self).__init__(src_path, tokenizer)

    def load_data(self):
        self.raw_data = pd.read_csv(
            self.src_path,
            sep="\t",
            header=0,
            index_col=0,
            error_bad_lines=False,
            warn_bad_lines=False,
        )
        if "label" not in self.raw_data.columns:
            # add dummy column to match data input format
            self.raw_data["label"] = ["entailment"] * self.__len__()

    def preprocess_data(self):
        for i, row in tqdm(list(self.raw_data.iterrows())):
            tokenized_question = self.tokenizer.tokenize(row.question)
            tokenized_sentence = self.tokenizer.tokenize(row.sentence)
            tokenized_text = tokenized_question + tokenized_sentence
            self.tokens.append(self.tokenizer.convert_tokens_to_ids(tokenized_text))
            self.segments.append(
                ([0] * len(tokenized_question)) + ([1] * len(tokenized_sentence))
            )
            self.labels.append([(1 * (self.raw_data.label[i] == "entailment")) + 1])


class RTEDataset(GlueDataset):
    def __init__(self, src_path, tokenizer):
        super(RTEDataset, self).__init__(src_path, tokenizer)
        self.sentence1_tokens = []
        self.sentence2_tokens = []

    def load_data(self):
        self.raw_data = pd.read_csv(
            self.src_path,
            sep="\t",
            header=0,
            index_col=0,
            error_bad_lines=False,
            warn_bad_lines=False,
        )
        if "label" not in self.raw_data.columns:
            # add dummy column to match data input format
            self.raw_data["label"] = ["entailment"] * self.__len__()

    def __getitem__(self, index):
        return (
            (self.sentence1_tokens[index], self.sentence2_tokens[index]),
            self.labels[index],
        )

    def preprocess_data(self):
        for i, row in tqdm(list(self.raw_data.iterrows())):
            tokenized_sentence1 = self.tokenizer.tokenize(row.sentence1)
            tokenized_sentence2 = self.tokenizer.tokenize(row.sentence2)
            self.sentence1_tokens.append(
                self.tokenizer.convert_tokens_to_ids(tokenized_sentence1)
            )
            self.sentence2_tokens.append(
                self.tokenizer.convert_tokens_to_ids(tokenized_sentence2)
            )
            #             tokenized_text = tokenized_sentence1 + tokenized_sentence2
            #             self.tokens.append(self.tokenizer.convert_tokens_to_ids(tokenized_text))
            #             self.segments.append(
            #                 ([0] * len(tokenized_sentence1)) + ([1] * len(tokenized_sentence2))
            #             )
            self.labels.append([(1 * (self.raw_data.label[i] == "entailment")) + 1])

    def collate_fn(self, batch, max_len):
        batch_size = len(batch)

        # Find the max length of sentences
        max_sent1_len = int(np.max([len(sent1) for ((sent1, sent2), _) in batch]))
        max_sent2_len = int(np.max([len(sent1) for ((sent1, sent2), _) in batch]))
        if (max_len > 0) and (max_len < max_sent1_len):
            max_sent1_len = max_len
        if (max_len > 0) and (max_len < max_sent2_len):
            max_sent2_len = max_len

        sent1_matrix = np.zeros((batch_size, max_sent1_len), dtype=np.int)
        sent2_matrix = np.zeros((batch_size, max_sent2_len), dtype=np.int)
        label_matrix = np.zeros((batch_size, 1), dtype=np.int)

        for idx1 in np.arange(len(batch)):
            (sent1, sent2), labels = batch[idx1]
            label_matrix[idx1, :] = labels
            for idx2 in np.arange(len(sent1)):
                if idx2 >= max_sent1_len:
                    break
                sent1_matrix[idx1, idx2] = sent1[idx2]
            for idx2 in np.arange(len(sent2)):
                if idx2 >= max_sent2_len:
                    break
                sent2_matrix[idx1, idx2] = sent2[idx2]

        sent1_matrix = torch.LongTensor(sent1_matrix)
        sent2_matrix = torch.LongTensor(sent2_matrix)
        sent1_mask_matrix = torch.eq(sent1_matrix.data, 0).long()
        sent2_mask_matrix = torch.eq(sent2_matrix.data, 0).long()
        label_matrix = torch.LongTensor(label_matrix)
        return (
            (sent1_matrix, sent1_mask_matrix, sent2_matrix, sent2_mask_matrix),
            label_matrix,
        )
