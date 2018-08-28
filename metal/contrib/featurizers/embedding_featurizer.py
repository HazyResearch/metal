import itertools
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab

from metal.contrib.featurizers.featurizer import Featurizer


class EmbeddingFeaturizer(Featurizer):
    """Converts lists of tokens into a padded Tensor of embedding indices."""

    def __init__(self, markers=[]):
        self.specials = markers + ["<pad>"]
        self.vocab = None

    def build_vocab(self, counter):
        raise NotImplementedError

    def fit(self, sents, **kwargs):
        """Builds a vocabulary object based on the tokens in the input.

        Args:
            sents: A list of lists of tokens (representing sentences)

        Vocab kwargs include:
            max_size
            min_freq
            specials
            unk_init
        """
        tokens = list(itertools.chain.from_iterable(sents))
        counter = Counter(tokens)
        self.vocab = self.build_vocab(counter, **kwargs)

    def transform(self, sents):
        """Converts lists of tokens into a Tensor of embedding indices.

        Args:
            sents: A list of lists of tokens (representing sentences)
                NOTE: These sentences should already be marked using the
                mark_entities() helper.
        Returns:
            X: A Tensor of shape (num_items, max_seq_len)
        """

        def convert(tokens):
            return torch.tensor(
                [self.vocab.stoi[t] for t in tokens], dtype=torch.long
            )

        if self.vocab is None:
            raise Exception(
                "Must run .fit() for .fit_transform() before "
                "calling .transform()."
            )

        seqs = sorted([convert(s) for s in sents], key=lambda x: -len(x))
        X = torch.LongTensor(pad_sequence(seqs, batch_first=True))
        return X


class TrainableEmbeddingFeaturizer(EmbeddingFeaturizer):
    def build_vocab(self, counter, **kwargs):
        return Vocab(counter, specials=self.specials, **kwargs)
