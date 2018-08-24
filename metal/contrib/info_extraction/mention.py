import numpy as np


class EntityMention(object):
    """A mention of an entity (span of text) in a document

    If tokens are not provided, tokenize on whitespace by default
    """

    def __init__(self, doc_id, text, char_start, char_end, tokens=None):
        self.doc_id = doc_id
        self.text = text
        self.char_start = int(char_start)
        self.char_end = int(char_end)
        self.id = hash(self)

        self.entity = text[self.char_start : self.char_end]
        self.tokens = text.split() if tokens is None else tokens

        self._char_offsets = np.zeros(len(self.tokens) + 1)
        for i, tok in enumerate(self.tokens):
            # Add 1 to account for the spaces between tokens
            self._char_offsets[i + 1] = self._char_offsets[i] + len(tok) + 1
        self._char_offsets = np.array(self._char_offsets)

        # Convert exclusive character offsets to inclusive token indices
        self.word_start = self._char_to_idx(self.char_start)
        self.word_end = self._char_to_idx(self.char_end - 1)

        ent = " ".join(self.tokens[self.word_start : self.word_end + 1])
        if ent != self.entity:
            msg = (
                f"Warning: Misalignment between character-based entity "
                f"({self.entity}) and word-based entity ({self.entity})."
            )
            print(msg)

    def _char_to_idx(self, char):
        """Converts a character offset to a token index

        Finds the first index of a True (i.e., the index of the first token that
        is past the desired token) and subtracts one.
        """
        return np.argmax(self._char_offsets > char) - 1

    def __repr__(self):
        return (
            f"""EntityMention("{self.doc_id}-{self.char_start}:"""
            f"""{self.char_end}, '{self.entity}'")"""
        )

    def __hash__(self):
        return hash((self.doc_id, self.char_start, self.char_end))


class RelationMention(object):
    """A mention of an relation between two spans of text (entities) in a doc

    If tokens are not provided, tokenize on whitespace by default
    """

    def __init__(self, doc_id, text, char_positions, tokens=None):
        self.doc_id = doc_id
        self.text = text
        self.char_positions = char_positions
        self.entities = [
            EntityMention(doc_id, text, *cp, tokens) for cp in char_positions
        ]
        self.id = hash(self)

    @property
    def tokens(self):
        return self.entities[0].tokens

    @property
    def word_starts(self):
        return [e.word_start for e in self.entities]

    @property
    def word_ends(self):
        return [e.word_end for e in self.entities]

    @property
    def word_positions(self):
        return [(e.word_start, e.word_end) for e in self.entities]

    def __repr__(self):
        entities = ", ".join(
            [
                f'"{e.entity}"({e.char_start}:{e.char_end})'
                for e in self.entities
            ]
        )
        return (
            f"""RelationMention(doc_id={self.doc_id}: entities=({entities})"""
        )

    def __hash__(self):
        return hash((self.doc_id, tuple(self.char_positions)))
