import os
import string

import numpy as np
import torch
import torch.nn.init as init


class EmbeddingLoader(object):
    """
    Simple text file embedding loader. Words with GloVe and FastText.
    """

    def __init__(self, fpath, fmt="text", dim=None, normalize=True):
        assert os.path.exists(fpath)
        self.fpath = fpath
        self.dim = dim
        self.fmt = fmt
        # infer dimension
        if not self.dim:
            header = open(self.fpath, "rU").readline().strip().split(" ")
            self.dim = len(header) - 1 if len(header) != 2 else int(header[-1])

        self.vocab, self.vectors = zip(*[(w, vec) for w, vec in self._read()])
        self.vocab = {w: i for i, w in enumerate(self.vocab)}
        self.vectors = np.vstack(self.vectors)
        if normalize:
            self.vectors = (
                self.vectors.T / np.linalg.norm(self.vectors, axis=1, ord=2)
            ).T

    def _read(self):
        start = 0 if self.fmt == "text" else 1
        for i, line in enumerate(open(self.fpath, "rU")):
            if i < start:
                continue
            line = line.rstrip().split(" ")
            vec = np.array([float(x) for x in line[1:]])
            if len(vec) != self.dim:
                # errors += [line[0]]
                continue
            yield (line[0], vec)


def load_embeddings(vocab, embeddings):
    """
    Load pretrained embeddings
    """

    def get_word_match(w, word_dict):
        if w in word_dict:
            return word_dict[w]
        elif w.lower() in word_dict:
            return word_dict[w.lower()]
        elif w.strip(string.punctuation) in word_dict:
            return word_dict[w.strip(string.punctuation)]
        elif w.strip(string.punctuation).lower() in word_dict:
            return word_dict[w.strip(string.punctuation).lower()]
        else:
            return -1

    num_words = vocab.len()
    emb_dim = embeddings.vectors.shape[1]
    vecs = init.xavier_normal_(torch.empty(num_words, emb_dim))
    vecs[0] = torch.zeros(emb_dim)

    n = 0
    for w in vocab.d:
        idx = get_word_match(w, embeddings.vocab)
        if idx == -1:
            continue
        i = vocab.lookup(w)
        vecs[i] = torch.FloatTensor(embeddings.vectors[idx])
        n += 1

    print(
        "Loaded {:2.1f}% ({}/{}) pretrained embeddings".format(
            float(n) / vocab.len() * 100.0, n, vocab.len()
        )
    )
    return vecs
