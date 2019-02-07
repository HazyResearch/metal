import pandas as pd
import torch.utils.data as data
from tqdm import tqdm


class GlueDataset(data.Dataset):
    """
    Torch dataset object for Glue task. Each specific task should override the preprocess_data and load_data methods.
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


class QNLIDataset(GlueDataset):
    def __init__(self, src_path, tokenizer):
        super(QNLIDataset, self).__init__(src_path, tokenizer)

    def load_data(self):
        self.raw_data = pd.read_csv(
                self.src_path, sep='\t', header=0,
                index_col=0, error_bad_lines=False, warn_bad_lines=False
        )
        if 'label' not in self.raw_data.columns:
            # add dummy column to match data input format
            self.raw_data['label'] = ['entailment'] * self.__len__()

    def preprocess_data(self):
        for i, row in tqdm(list(self.raw_data.iterrows())):
            tokenized_question = self.tokenizer.tokenize(row.question)
            tokenized_sentence = self.tokenizer.tokenize(row.sentence)
            tokenized_text = tokenized_question + tokenized_sentence
            self.tokens.append(self.tokenizer.convert_tokens_to_ids(tokenized_text))
            self.segments.append(([0] * len(tokenized_question)) + ([1] * len(tokenized_sentence)))
            self.labels.append(
                    [1 * (self.raw_data.label[i] == 'entailment')] + [1 * (self.raw_data.label[i] == 'not_entailment')])
