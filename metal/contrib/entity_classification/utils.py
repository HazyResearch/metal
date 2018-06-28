import numpy as np

def mark_entities(tokens, positions, markers=[]):
    """Adds special markers around tokens at specific positions (e.g., entities)

    Args:
        tokens: A list of tokens (the sentence)
        positions: 
            1) A list of inclusive ranges (tuples) corresponding to the 
            token ranges of the entities in order. (Assumes each entity
            has only one corresponding mention.)
            OR
            2) A dict of lists with keys corresponding to mention indices and 
            values corresponding to one or more inclusive ranges corresponding
            to that mention. (Allows entities to potentially have multiple 
            mentions)
    Returns:
        toks: An extended list of tokens with markers around the mentions

    Example:
        Input:  (['The', 'cat', 'sat'], [(1,1)])
        Output: ['The', '[[BEGIN0]]', 'cat', '[[END0]]', 'sat']
    """
    if markers and len(markers) != 2 * len(positions): 
        msg = (f"Expected len(markers) == 2 * len(positions), "
            f"but {len(markers)} != {2 * len(positions)}.")
        raise ValueError(msg)
    if not markers:
        print("Warning: if using pretrained embeddings, provide markers with "
            "non-UNK embeddings")

    toks = list(tokens)
    positions_out = []

    # markings will be of the form: 
    # [(position, entity_idx), (position, entity_idx), ...]
    if isinstance(positions, list):
        markings = [(position, idx) for idx, position in enumerate(positions)]
    elif isinstance(positions, dict):
        markings = []
        for idx, v in positions.items():
            for position in v:
                markings.append((position, idx))
    else:
        msg = (f"Argument _positions_ must be a list or dict. "
            f"Instead, got {type(positions)}")
        raise ValueError(msg)

    markings = sorted(markings)
    for i, ((si, ei), idx) in enumerate(markings):
        if markers:
            start_marker = markers[2*idx]
            end_marker = markers[2*idx + 1]
        else:
            start_marker = f'[[BEGIN{idx}]]'
            end_marker = f'[[END{idx}]]'
        toks.insert(si + 2*i, start_marker)
        toks.insert(ei + 2*(i+1), end_marker)
    return toks


def make_entities(Y, vocab, word_dists, sentence_length, entity_length,
    mask_entities):
    print("Warning: this function has not yet been tested!")
    num_items = len(Y)
    sentence_lengths = np.random.randint(
        min(sentence_length), max(sentence_length), num_items)
    
    entities = []
    for i, (y, sent_len) in enumerate(zip(Y, sentence_lengths)):
        doc_id = i
        selected_words = np.random.choice(
            len(vocab), sent_len, p=word_dists[y])
        tokens = [vocab[j] for j in selected_words]

        ent_word_len = np.random.choice(entity_length)
        word_start = np.random.choice(sent_len - (ent_word_len - 1))
        word_end = word_start + ent_word_len

        if mask_entities:
            for k in range(word_start, word_end):
                tokens[k] = 'entity'

        doc = ' '.join(tokens)
        char_start = sum(len(t) + 1 for t in tokens[:word_start])
        ent_char_len = sum(len(t) for t in tokens[word_start:word_end])
        char_end = char_start + ent_char_len

        ent = Entity(doc_id, doc, char_start, char_end)
        entities.append(doc)
    return entities