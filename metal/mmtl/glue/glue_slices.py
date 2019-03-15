import warnings

question_words = set(["who", "what", "where", "when", "why", "how"])


def ends_with_question(dataset, idx):
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


def create_slice_labels(dataset, base_task_name, slice_name, verbose=False):
    """Returns a label set masked to include only those labels in the specified slice"""
    # TODO: break this out into more modular pieces one we have multiple slices
    slice_fn = slice_functions[slice_name]
    slice_indicators = [slice_fn(dataset, idx) for idx in range(len(dataset))]
    base_labels = dataset.labels[base_task_name]
    slice_labels = [
        label * indicator for label, indicator in zip(base_labels, slice_indicators)
    ]
    if verbose:
        print(f"Found {sum(slice_indicators)} examples in slice {slice_name}.")
        if not any(slice_labels):
            warnings.warn("No examples were found to belong to ")
    return slice_labels


# Functions which map a payload and index with an indicator if that example is in slice
# NOTE: the indexing is left to the functions so that extra fields helpful for slicing
# but not required by the model (e.g., spacy-based features) can be stored with the
# dataset but not necessarily passed back by __getitem__() which is called by the
# DataLoaders.
slice_functions = {"ends_with_question": ends_with_question}
