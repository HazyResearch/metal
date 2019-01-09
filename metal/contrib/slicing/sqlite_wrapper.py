import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from metal.contrib.info_extraction.utils import mark_entities


class SnorkelDataset(Dataset):
    """
    Self-contained wrapper class for Snorkel 0.7 database instance.
    Suitable for datasets that fit entirely in memory.
    """

    session = None

    def __init__(
        self,
        conn_str,
        candidate_def,
        split=0,
        use_lfs=False,
        word_dict=None,
        pretrained_word_dict=None,
        max_seq_len=125,
        L_train=None,
        train_marginals=None,
    ):
        """
        Assumes a Snorkel database that is fully instantiated with:
        - Candidates generated and assigned to train/dev/test splits
        - Labeling functions are applied and probabilistic labels are generated for train split(s)
        - Gold labels are stored in the database under 'annotator_name = gold'

        :param conn_str:
        :param candidate_def:
        :param split:
        :param use_lfs:
        :param word_dict:
        :param pretrained_word_dict:
        :param max_seq_len:

        """
        if os.path.exists(conn_str):
            os.environ["SNORKELDB"] = "sqlite:///{}".format(conn_str)
        else:
            os.environ["SNORKELDB"] = "postgresql:///{}".format(conn_str)
        print("Connected to {}".format(os.environ["SNORKELDB"]))

        # defer imports until SNORKELDB is defined to prevent initalizing an empty sqlite instance
        from snorkel import SnorkelSession
        from snorkel.models import candidate_subclass, Candidate
        from snorkel.annotations import load_gold_labels, load_marginals

        # sqlite3 doesn't support multiple connections, so use a singleton-style connection object
        if not SnorkelDataset.session:
            SnorkelDataset.session = SnorkelSession()
        self.session = SnorkelDataset.session

        self.class_type = candidate_subclass(*candidate_def)
        self.cardinality = len(candidate_def[-1])
        self.split = split
        self.max_seq_len = max_seq_len
        self.use_lfs = use_lfs

        # create markup sequences and labels
        markers = [
            m.format(i)
            for i in range(self.cardinality)
            for m in ["~~[[{}", "{}]]~~"]
        ]
        self.X = (
            self.session.query(Candidate)
            .filter(Candidate.split == split)
            .order_by(Candidate.id)
            .all()
        )
        self.X = [self._mark_entities(x, markers) for x in self.X]

        # initalize vocabulary
        self.word_dict = (
            self._build_vocab(self.X, markers) if not word_dict else word_dict
        )
        if pretrained_word_dict:
            # include pretrained embedding terms
            self._include_pretrained_vocab(
                pretrained_word_dict, self.session.query(Candidate).all()
            )

        # initalize labels (from either LFs or gold labels)
        if use_lfs:
            marginals = load_marginals(self.session, split=split)
            multitask_marginals = np.vstack((marginals, 1 - marginals)).T
            self.Y = torch.tensor(multitask_marginals.astype(np.float32))
        elif train_marginals is not None:
            self.Y = torch.tensor(train_marginals.astype(np.float32))
        else:
            self.Y = load_gold_labels(
                self.session, annotator_name="gold", split=split
            )
            self.Y = [int(y) for y in np.nditer(self.Y.todense())]
            # remap class labels to not include 0 (reserved by MeTaL)
            labels = {
                y: i + 1
                for i, y in enumerate(sorted(np.unique(self.Y), reverse=1))
            }
            self.Y = torch.tensor([labels[y] for y in self.Y])

        assert self.Y.shape[0] == len(self.X)

        # initialize LFs for slice reweighting
        if L_train is not None:
            self.L = torch.from_numpy(L_train.astype(np.float32))
        else:
            self.L = None

    @classmethod
    def splits(
        cls,
        conn_str,
        candidate_def,
        word_dict=None,
        train=0,
        dev=1,
        test=2,
        use_lfs=(0, 0, 0),
        pretrained_word_dict=None,
        max_seq_len=125,
    ):
        """
        Create train/dev/test splits (mapped to split numbers)

        :param conn_str:
        :param candidate_def:
        :param word_dict:
        :param train:
        :param dev:
        :param test:
        :param use_lfs:
        :param pretrained_word_dict:
        :param max_seq_len:
        :return:

        """
        # initialize word_dict if needed
        train_set = cls(
            conn_str,
            candidate_def,
            word_dict=word_dict,
            split=train,
            use_lfs=use_lfs[train],
            pretrained_word_dict=pretrained_word_dict,
            max_seq_len=max_seq_len,
        )
        return (
            train_set,
            cls(
                conn_str,
                candidate_def,
                word_dict=train_set.word_dict,
                split=dev,
                use_lfs=use_lfs[dev],
                max_seq_len=max_seq_len,
            ),
            cls(
                conn_str,
                candidate_def,
                word_dict=train_set.word_dict,
                split=test,
                use_lfs=use_lfs[test],
                max_seq_len=max_seq_len,
            ),
        )

    def _mark_entities(self, c, markers):
        """
        Convert Snorkel candidates to marked up sequences

        :param c:
        :param markers:
        :return:

        """
        sent = c.get_parent().words
        positions = [
            [c[i].get_word_start(), c[i].get_word_end()]
            for i in range(self.cardinality)
        ]
        seq = mark_entities(sent, positions, markers=markers, style="insert")
        return [w for w in seq if w.strip()]

    def _include_pretrained_vocab(self, pretrained_word_dict, candidates):
        """
        Include terms available via pretrained embeddings

        :param pretrained_word_dict:
        :param candidates:
        :return:

        """
        terms = Counter()
        for c in candidates:
            for w in c.get_parent().words:
                if w in pretrained_word_dict:
                    terms[w] += 1
        list(map(self.word_dict.get, terms))

    def _build_vocab(self, sentences, markers=[]):
        """
        Initalize symbol table dictionary

        :param sentences:
        :param markers:
        :return:

        """
        from snorkel.learning.pytorch.rnn.utils import SymbolTable

        vocab = Counter()
        for sent in sentences:
            for w in sent:
                vocab[w] += 1
        word_dict = SymbolTable()
        list(map(word_dict.get, vocab))
        list(map(word_dict.get, markers))
        return word_dict

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Assume fixed length sequences. Pad or clip all sequences to be max_seq_len.

        :param idx:
        :return:
        """
        x = torch.tensor([self.word_dict.lookup(w) for w in self.X[idx]])
        if x.size(0) > self.max_seq_len:
            x = x[..., 0 : min(x.size(0), self.max_seq_len)]
        else:
            k = self.max_seq_len - x.size(0)
            x = torch.cat((x, torch.zeros(k, dtype=torch.long)))

        if self.L is not None:
            if self.use_lfs:
                return x, self.L[idx], self.Y[idx]
            else:
                return x, self.L[idx]
        else:
            return x, self.Y[idx]


if __name__ == "__main__":

    db_conn_str = "cdr.db"
    candidate_def = ["ChemicalDisease", ["chemical", "disease"]]
    train, dev, test = SnorkelDataset.splits(db_conn_str, candidate_def)

    print("[TRAIN] {}".format(len(train)))
    print("[DEV]   {}".format(len(dev)))
    print("[TEST]  {}".format(len(test)))
