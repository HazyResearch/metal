import numpy as np


class EntityMention(object):
    """A mention of an entity (span of text) in a document

    Args:
        doc_id: a unique identifier for the document
        text: a single string of text corresponding to the document
        char_start: the integer offset of the first character in the entity
            mention
        char_end: the integer offset of the last character in the entity
            mention, plus one (so that text[char_start:char_end] returns the
            full entity).
        tokens: (optional) a list of tokens corresponding to the text.
            If None, tokenization on whitespace is the default.
        char_offsets: (optional) a list of character offsets corresponding
            to the tokens in tokens.
            If None, we assume all tokens are separated by a single space.
        attributes: (optional) additional lists of attributes corresponding to
            the provided tokens (e.g., pos tags, ner tags, types, etc.)
    """

    def __init__(
        self,
        doc_id,
        text,
        char_start,
        char_end,
        tokens=None,
        char_offsets=None,
        mention_id=None,
        **attributes,
    ):
        self.doc_id = doc_id
        self.text = text
        self.char_start = int(char_start)
        self.char_end = int(char_end)
        self.mention_id = mention_id if mention_id else hash(self)

        self.entity = text[self.char_start : self.char_end]

        self.tokens = tokens if tokens is not None else text.split()
        self.char_offsets = self._get_char_offsets(char_offsets)

        # Convert exclusive character offsets to inclusive token indices
        self.word_start = self.char_to_word_idx(self.char_start)
        self.word_end = self.char_to_word_idx(self.char_end - 1)

        # Store extra attributes
        for attr, values in attributes.items():
            setattr(self, attr, values)

    def _get_char_offsets(self, char_offsets):
        """Store or calculate char_offsets, adding the offset of the doc end"""
        if char_offsets:
            char_offsets = char_offsets
            char_offsets.append(len(self.text))
        else:
            char_offsets = np.zeros(len(self.tokens) + 1)
            for i, tok in enumerate(self.tokens):
                # Add 1 to account for the spaces between tokens
                char_offsets[i + 1] = char_offsets[i] + len(tok) + 1
            char_offsets[-1] = len(self.text)
        return np.array(char_offsets)

    def word_to_char_idx(self, word_idx):
        """Converts a word index to a character offset

        Returns the offset of the first character of the token with the given
        index.
        """
        return self.char_offsets[word_idx]

    def char_to_word_idx(self, char_offset):
        """Converts a character offset to a token index

        Finds the first index of a True (i.e., the index of the first token that
        is past the desired token) and subtracts one.
        """
        return np.argmax(self.char_offsets > char_offset) - 1

    def get_entity_attrib(self, attrib):
        attrib_tokens = self.get(attrib, None)
        return attrib_tokens[self.word_start : self.word_end + 1]

    @property
    def words(self):
        return self.tokens

    def __repr__(self):
        return (
            f"EntityMention(doc_id={self.doc_id}: '{self.entity}'"
            f"({self.char_start}:{self.char_end})"
        )

    def __hash__(self):
        return hash((self.doc_id, self.char_start, self.char_end))


class RelationMention(object):
    """A mention of a relation between two spans of text (entities) in a doc

    Args:
        doc_id: a unique identifier for the document
        text: a single string of text corresponding to the document
        entity_positions: a list with two elements, each a tuple of the integer
            offsets (in characters) of the corresponding entity in the text so
            that text[char_start:char_end] returns the full entity
        tokens: (optional) a list of tokens corresponding to the text.
            If None, tokenization on whitespace is the default.
        char_offsets: (optional) a list of character offsets corresponding
            to the tokens in tokens.
            If None, we assume all tokens are separated by a single space.
        attributes: (optional) additional lists of attributes corresponding to
            the provided tokens (e.g., pos tags, ner tags, types, etc.)

    TODO: There is currently inefficiency in the way each EntityMention in a
    RelationMention stores all properties of a sentence. Instead, create a
    Sentence object that each EntityMention points to and store the properties
    with the sentence.
    """

    def __init__(
        self,
        doc_id,
        text,
        entity_positions,
        tokens=None,
        char_offsets=None,
        mention_id=None,
        **attributes,
    ):
        self.doc_id = doc_id
        self.entity_positions = entity_positions
        self.entities = [
            EntityMention(doc_id, text, *cp, tokens, char_offsets, **attributes)
            for cp in entity_positions
        ]
        self.mention_id = mention_id if mention_id else hash(self)

    @property
    def text(self):
        return self.entities[0].text

    @property
    def tokens(self):
        return self.entities[0].tokens

    @property
    def words(self):
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

    def get_attr(self, attr):
        return self.entities[0].get(attr, None)

    def __getitem__(self, key):
        return self.entities[key]

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
        return hash((self.doc_id, tuple(self.entity_positions)))
