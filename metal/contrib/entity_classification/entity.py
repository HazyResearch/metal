import numpy as np


class Entity(object):
    """A particular entity (span of text) in a document

    If tokens are not provided, tokenize on whitespace by default
    """

    def __init__(self, doc_id, doc, char_start, char_end, tokens=None):
        self.doc_id = doc_id
        self.doc = doc
        self.char_start = int(char_start)
        self.char_end = int(char_end)

        self.entity = doc[char_start:char_end]
        self.tokens = doc.split() if tokens is None else tokens

        self._char_offsets = np.zeros(len(self.tokens) + 1)
        for i, tok in enumerate(self.tokens):
            # Add 1 to account for the spaces between tokens
            self._char_offsets[i + 1] = self._char_offsets[i] + len(tok) + 1
        self._char_offsets = np.array(self._char_offsets)

        # Convert exclusive character offsets to inclusive token indices
        self.word_start = self._char_to_idx(self.char_start)
        self.word_end = self._char_to_idx(self.char_end - 1)

        ent = " ".join(self.tokens[self.word_start : self.word_end + 1])
        assert ent == self.entity

    def _char_to_idx(self, char):
        """Converts a character offset to a token index

        Finds the first index of a True (i.e., the index of the first token that
        is past the desired token) and subtracts one.
        """
        return np.argmax(self._char_offsets > char) - 1

    def __repr__(self):
        return (
            f"""Entity("{self.doc_id}-{self.char_start}:"""
            f"""{self.char_end}, '{self.entity}'")"""
        )

    def __hash__(self):
        return hash((self.doc_id, self.char_start, self.char_end))
