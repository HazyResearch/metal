import warnings

import spacy
import torch

question_words = set(["who", "what", "where", "when", "why", "how"])
nlp = spacy.load("en_core_web_sm")


def more_people(dataset, idx):
    people = 0
    sentence = dataset.sentences[idx][0].split()
    for pronoun in ["she", "her", "hers"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["he", "him", "his"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["you", "your", "yours"]:
        if pronoun in sentence:
            people += 1
            break
    for pronoun in ["I", "my", "me", "mine"]:
        if pronoun in sentence:
            people += 1
            break
    return people > 1


def entity_secondonly(dataset, idx):
    sent1 = dataset.sentences[idx][0]
    sent2 = nlp(dataset.sentences[idx][1])
    for ent in sent2.ents:
        if ent.text not in sent1:
            return True
    return False


def ends_with_question_word(dataset, idx):
    """Returns True if a question word is in the last three tokens of any sentence"""
    # HACK: For now (speedy POC), just use the BERT tokens
    # Eventually, we'd like to have access to the raw text (e.g., via the Spacy
    # tokenization) and have the two sentences separated for pair tasks

    # spacy_sentences = dataset.spacy_tokens[idx]
    # for spacy_sentence in spacy_sentences:
    #     if any(token.text in question_words for token in spacy_sentence):
    #         return True

    bert_ints = dataset.bert_tokens[idx]
    bert_tokens = dataset.bert_tokenizer.convert_ids_to_tokens(bert_ints)
    return any(token in question_words for token in bert_tokens[-3:])
    # match = any(token in question_words for token in bert_tokens[-3:])
    # if match:
    #     print(bert_tokens)
    # return match


def is_statement_has_question(dataset, idx):
    """Returns True if question word exists in statement that doesn't end with ?"""
    bert_ints = dataset.bert_tokens[idx]
    bert_tokens = dataset.bert_tokenizer.convert_ids_to_tokens(bert_ints)

    return (
        any(t.lower() in question_words for t in bert_tokens) and bert_tokens[-2] != "?"
    )


def ends_with_question_mark(dataset, idx):
    """Returns True if last token is '?" symbol"""
    bert_ints = dataset.bert_tokens[idx]
    bert_tokens = dataset.bert_tokenizer.convert_ids_to_tokens(bert_ints)

    # last token is '[SEP]'
    # check the second to last token for end of sentence
    return bert_tokens[-2] == "?"


def dash_semicolon(dataset, idx):
    """Returns True if there is a dash or semicolon in sentence1"""
    bert_ints = dataset.bert_tokens[idx]
    bert_tokens = dataset.bert_tokenizer.convert_ids_to_tokens(bert_ints)
    return "-" in bert_tokens or ";" in bert_tokens


# Functions which map a payload and index with an indicator if that example is in slice
# NOTE: the indexing is left to the functions so that extra fields helpful for slicing
# but not required by the model (e.g., spacy-based features) can be stored with the
# dataset but not necessarily passed back by __getitem__() which is called by the
# DataLoaders.
# No longer needed: can do same thing as below with globals()[slice_name]
# slice_functions = {
#     "ends_with_question_word": ends_with_question_word,
#     "ends_with_question_mark": ends_with_question_mark,
#     "is_statement_has_question": is_statement_has_question,
# }


def create_slice_labels(dataset, base_task_name, slice_name, verbose=False):
    """Returns a label set masked to include only those labels in the specified slice"""
    # TODO: break this out into more modular pieces oncee we have multiple slices
    slice_fn = globals()[slice_name]
    slice_indicators = torch.tensor(
        [slice_fn(dataset, idx) for idx in range(len(dataset))], dtype=torch.uint8
    ).view(-1, 1)
    Y_base = dataset.labels[base_task_name]
    Y_slice = Y_base.clone().masked_fill_(slice_indicators == 0, 0)

    if verbose:
        if not any(Y_slice):
            warnings.warn(f"No examples were found to belong to slice {slice_name}")
        else:
            print(f"Found {sum(slice_indicators)} examples in slice {slice_name}.")

    # NOTE: we assume here that all slice labels are for sentence-level tasks only
    return Y_slice
