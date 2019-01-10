import csv
import re
from collections import defaultdict


def tokens_to_ngrams(tokens, n_max=3, delim=" "):
    N = len(tokens)
    for root in range(N):
        for n in range(min(n_max, N - root)):
            yield delim.join(tokens[root : root + n + 1])


def get_left_tokens(c, window=3, attrib="words", n_max=1, case_sensitive=False):
    """
    Return the tokens within a window to the _left_ of the Candidate.
    For higher-arity Candidates, defaults to the _first_ argument.
    :param window: The number of tokens to the left of the first argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    # HACK we don't want to create Span objects, so just construct
    # parent sentence manually form tuple of (TemporarySpan, Sentence)
    if type(c) is tuple:
        span = c[0]
        sentence = c[-1]
        i = span.get_word_start()
    else:
        try:
            span = c
            sentence = span.get_parent()
            i = span.get_word_start()
        except Exception:
            span = c[0]
            sentence = span.get_parent()
            i = span.get_word_start()

    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(
        list(map(f, sentence._asdict()[attrib][max(0, i - window) : i])),
        n_max=n_max,
    )


def get_right_tokens(
    c, window=3, attrib="words", n_max=1, case_sensitive=False
):
    """
    Return the tokens within a window to the _right_ of the Candidate.
    For higher-arity Candidates, defaults to the _last_ argument.
    :param window: The number of tokens to the right of the last argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    """
    #     try:
    #         span = c
    #         i = span.get_word_end()
    #     except:
    #         span = c[-1]
    #         i = span.get_word_end()

    if type(c) is tuple:
        span = c[0]
        sentence = c[-1]
        i = span.get_word_start()
    else:
        try:
            span = c
            sentence = span.get_parent()
            i = span.get_word_start()
        except Exception:
            span = c[0]
            sentence = span.get_parent()
            i = span.get_word_start()

    f = (lambda w: w) if case_sensitive else (lambda w: w.lower())
    return tokens_to_ngrams(
        list(map(f, sentence._asdict()[attrib][i + 1 : i + 1 + window])),
        n_max=n_max,
    )


class NegEx(object):
    """
    Negex

    Chapman, Wendy W., et al. "A simple algorithm for identifying negated findings and
    diseases in discharge summaries." Journal of biomedical informatics 34.5 (2001): 301-310.
    """

    def __init__(self, data_root="."):
        self.data_root = data_root
        self.filename = "negex_multilingual_lexicon-en-de-fr-sv.csv"
        self.dictionary = NegEx.load(
            "{}/{}".format(self.data_root, self.filename)
        )
        self.rgxs = NegEx.build_regexs(self.dictionary)

    def negation(self, span, category, direction, window=3):
        """
        Return any matched negex phrases

        :param span:
        :param category:
        :param direction:
        :param window:
        :return:
        """
        rgx = self.rgxs[category][direction]
        if not rgx.pattern:
            return None

        context = (
            get_left_tokens(span, window)
            if direction == "left"
            else get_right_tokens(span, window)
        )
        context = " ".join(context).strip()
        # m = rgx.search(context)
        m = rgx.findall(context)

        return m if m else None

    def is_negated(self, span, category, direction, window=3):
        """
        Boolean test for negated spans

        :param span:
        :param category:
        :param direction:
        :param window:
        :return:
        """
        rgx = self.rgxs[category][direction]
        if not rgx.pattern:
            return False

        negation_match = self.negation(span, category, direction, window)
        return True if negation_match else False

    def all_negations(self, span, window=3):

        ntypes = []
        for category in self.rgxs:
            for direction in self.rgxs[category]:
                m = self.negation(span, category, direction, window)
                if m:
                    ntypes.append((category, direction, m))

        return ntypes

    @staticmethod
    def build_regexs(dictionary):
        """

        :param dictionary:
        :return:
        """
        rgxs = defaultdict(dict)
        for category in dictionary:
            fwd = [
                t["term"]
                for t in dictionary[category]
                if t["direction"] in ["forward", "bidirectional"]
            ]
            bwd = [
                t["term"]
                for t in dictionary[category]
                if t["direction"] in ["backward", "bidirectional"]
            ]
            rgxs[category]["left"] = "|".join(sorted(fwd, key=len, reverse=1))
            rgxs[category]["right"] = "|".join(sorted(bwd, key=len, reverse=1))

            if not rgxs[category]["left"]:
                del rgxs[category]["left"]
            if not rgxs[category]["right"]:
                del rgxs[category]["right"]
            for direction in rgxs[category]:
                p = rgxs[category][direction]
                rgxs[category][direction] = re.compile(
                    r"({})(?:\s|$)".format(p), flags=re.I
                )

        return rgxs

    @staticmethod
    def load(filename):
        """
        Load negex definitions
        :param filename:
        :return:
        """
        negex = defaultdict(list)
        with open(filename, "rU") as of:
            reader = csv.reader(of, delimiter=",")
            for row in reader:
                term = row[0]
                category = row[30]
                direction = row[32]
                if category == "definiteNegatedExistence":
                    negex["definite"].append(
                        {"term": term, "direction": direction}
                    )
                elif category == "probableNegatedExistence":
                    negex["probable"].append(
                        {"term": term, "direction": direction}
                    )
                elif category == "pseudoNegation":
                    negex["pseudo"].append(
                        {"term": term, "direction": direction}
                    )
        return negex
