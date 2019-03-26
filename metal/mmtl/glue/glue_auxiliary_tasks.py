import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from metal.utils import padded_tensor

SPACY_TAGS = {
    "SPACY_NER": [
        "NULL",
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "SPACY_POS": [  # Coarse-grained POS tags
        "NULL",
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CONJ",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
        "SPACE",
    ],
}


# Function to add BLEU labels
def add_bleu_labels(payload):
    """
    Adds 1-gram bleu score label_set for sentence similarity tasks
    """
    raise NotImplementedError("Update the signature of label_fn")
    # def get_bleu_label(it):
    #     # toks, segs = payload.data_loader.dataset[it][0]
    #     toks = payload.data_loader.dataset.bert_tokens[it]
    #     toks = payload.data_loader.dataset.bert_tokenizer.convert_ids_to_tokens(toks)
    #     segs = payload.data_loader.dataset.bert_segments[it]
    #     toks, segs = np.array(toks), np.array(segs)
    #     sent1 = list(toks[segs == 0])
    #     sent2 = list(toks[segs == 1])
    #     bleu_score = sentence_bleu(sent1, sent2, weights=(1, 0, 0, 0))
    #     return float(bleu_score)

    # return payload.add_label_set("BLEU", label_fn=get_bleu_label)


# Function add THIRD labels
def add_third_labels(payload):
    """Marks whether each unmasked token is in the 1st/2nd/3rd third of the sentence"""

    def mark_thirds(x):
        tokens = x
        count = len(tokens)
        Y_thirds = np.ceil(
            np.array([idx * 3 / count for idx in range(1, count + 1)])
        ).astype(np.int64)
        return Y_thirds

    X = payload.data_loader.dataset.bert_tokens
    Y_list = []
    for x in X:
        Y_list.append(mark_thirds(x))
    Y = padded_tensor(Y_list)

    payload.add_label_set("THIRD", label_list=Y)
    return payload


def add_spacy_pos_labels(payload):
    return add_spacy_labels(payload, label_name="SPACY_POS", spacy_attr="pos_")


def add_spacy_ner_labels(payload):
    return add_spacy_labels(payload, label_name="SPACY_NER", spacy_attr="ent_type_")


# Add token-based tags from Spacy
def add_spacy_labels(payload, label_name, spacy_attr, null_label="NULL"):
    """
    Adds a spacy POS label_set, mapping through the different tokenizations

    Args:
        payload
        attr: the name of the spacy attribute to extract
        tag_name:
    """
    Y_list = []
    bert_tokenizer = payload.data_loader.dataset.bert_tokenizer

    for i, data in enumerate(payload.data_loader.dataset):
        X, _ = data
        (tokens, segments) = X
        bert_ints = tokens
        bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_ints)
        spacy_tokens = []
        for sentence_tokens in payload.data_loader.dataset.spacy_tokens[i]:
            spacy_tokens.extend(sentence_tokens)
        # bert_to_spacy maps a bert token index to the corresponding spacy token index
        assignments = map_bert_to_spacy_tokens(bert_tokens, spacy_tokens)
        if assignments is None:
            # If a mapping could not be found, abstain from labelling this example
            token_labels = [0 for _ in bert_tokens]
        else:
            token_tags = []
            for idx in assignments:
                if idx is None:
                    token_tags.append(null_label)
                else:
                    tag = getattr(spacy_tokens[idx], spacy_attr)
                    if not tag:  # 'not' covers default None and "" (ner)
                        tag = null_label
                    token_tags.append(tag)
            token_labels = [SPACY_TAGS[label_name].index(tag) + 1 for tag in token_tags]
            # print([(token, tag) for token, tag in zip(bert_tokens, token_tags)])
        Y_list.append(token_labels)

    Y = padded_tensor(Y_list)
    payload.add_label_set(label_name, label_list=Y)
    return payload


def map_bert_to_spacy_tokens(bert_tokens, spacy_tokens):
    # If the token sets do not have the same number of characters, abstain
    spacy_chars = "".join([token.text.strip() for token in spacy_tokens])
    bert_chars = "".join(
        [
            token.replace("##", "")
            for token in bert_tokens
            if token not in ["[CLS]", "[SEP]"]
        ]
    )
    if len(spacy_chars) != len(bert_chars):
        print("SKIPPING: len(spacy_chars) != len(bert_chars)")
        return None

    # Otherwise, find a mapping
    spacy_token_index_by_char = []
    for i, token in enumerate(spacy_tokens):
        spacy_token_index_by_char.extend([i for _ in range(len(token.text))])
    char_count = 0
    assignments = []
    for token in bert_tokens:
        if token in ["[CLS]", "[SEP]"]:
            assignments.append(None)
        else:
            assignments.append(spacy_token_index_by_char[char_count])
            char_count += len(token.replace("##", ""))
    return assignments


auxiliary_task_functions = {
    "BLEU": add_bleu_labels,
    "THIRD": add_third_labels,
    "SPACY_NER": add_spacy_ner_labels,
    "SPACY_POS": add_spacy_pos_labels,
}
