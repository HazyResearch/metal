import nltk
from sklearn.feature_extraction.text import CountVectorizer

from metal.contrib.featurizers.featurizer import Featurizer


class RelationNgramFeaturizer(Featurizer):
    """A featurizer for relations that preprocesses and extracts ngrams

    This featurizer operates on RelationMention objects

    Args:
        anonymize: if True, replace each entity with a single token: "EntityX"
            where X is the index of the entity in the relation (0 or 1)
        trim_window: if non-zero, the sentence will be trimmed to this many
            words before the first entity in the sentence to this many words
            after the last entity in the sentence.
        lowercase: if True, convert all tokens to lowercase
        drop_stopwords: if True, drop all tokens that are stopwords
        stem: if True, stem all tokens
        ngram_range: a tuple corresponding to the smallest sized ngrams and
            largest sized ngrams to be included in the feature set
        kwargs: keyword arguments to pass on to the CountVectorizer.
            (See http://scikit-learn.org/stable/modules/generated/sklearn.
            feature_extraction.text.CountVectorizer.html for full details)
            Options include max_features, min_df, max_df, etc.
    """

    def __init__(
        self,
        anonymize=True,
        trim_window=5,
        lowercase=True,
        drop_stopwords=True,
        stem=True,
        ngram_range=(1, 3),
        **vectorizer_kwargs,
    ):
        self.anonymize = anonymize
        self.lowercase = lowercase
        self.drop_stopwords = drop_stopwords
        if drop_stopwords:
            nltk.download("stopwords")
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.trim_window = trim_window
        self.stem = stem
        if stem:
            self.porter = nltk.PorterStemmer()

        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range, binary=True, **vectorizer_kwargs
        )

    def preprocess(self, mentions):
        return [" ".join(self._preprocess(mention)) for mention in mentions]

    def _preprocess(self, mention):
        tokens = mention.tokens
        word_positions = mention.word_positions
        if self.anonymize:
            tokens, word_positions = self._anonymize(tokens, word_positions)
        if self.trim_window:
            tokens, word_positions = self._trim(tokens, word_positions)
        if self.lowercase:
            tokens = self._lowercase(tokens)
        if self.drop_stopwords:
            # TODO: update word_positions after stopword removal
            tokens = self._drop_stopwords(tokens)
        if self.stem:
            tokens = self._stem(tokens)
        return tokens

    def _anonymize(self, tokens, word_positions):
        offset = 0
        for i, (word_start, word_end) in enumerate(word_positions):
            word_start -= offset
            word_end -= offset
            tokens = (
                tokens[:word_start] + [f"ENTITY_{i}"] + tokens[(word_end + 1) :]
            )
            word_positions[i] = (word_start, word_start)
            offset += word_end - word_start
        return tokens, word_positions

    def _trim(self, tokens, word_positions):
        word_starts, word_ends = list(zip(*word_positions))
        lb = max(0, min(word_starts) - self.trim_window)
        ub = min(len(tokens), max(word_ends) + self.trim_window + 1)
        word_positions = [(wp[0] - lb, wp[1] - lb) for wp in word_positions]
        return tokens[lb:ub], word_positions

    def _lowercase(self, tokens):
        return [t.lower() for t in tokens]

    def _drop_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stopwords]

    def _stem(self, tokens):
        return [self.porter.stem(t) for t in tokens]

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

    def fit(self, input):
        preprocessed = self.preprocess(input)
        self.vectorizer.fit(preprocessed)

    def transform(self, input):
        preprocessed = self.preprocess(input)
        return self.vectorizer.transform(preprocessed)

    def fit_transform(self, input):
        preprocessed = self.preprocess(input)
        return self.vectorizer.fit_transform(preprocessed)
