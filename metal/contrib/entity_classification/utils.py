def mark_entities(tokens, positions, markers=[], style="insert"):
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
        markers: A list of strings (length of 2 * the number of entities) to
            use as markers of the entities.
        style: Where to apply the markers:
            'insert': Insert the markers as new tokens before/after each entity
            'concatenate': Prepend/append the markers to the first/last token
                of each entity
            If the tokens are going to be input to an LSTM, then it is usually
            best to use the 'insert' option; 'concatenate' may be better for
            viewing.

    Returns:
        toks: An extended list of tokens with markers around the mentions

    WARNING: if the marked token set will be used with pretrained embeddings,
        provide markers that will not result in UNK embeddings!

    Example:
        Input:  (['The', 'cat', 'sat'], [(1,1)])
        Output: ['The', '[[BEGIN0]]', 'cat', '[[END0]]', 'sat']
    """
    if markers and len(markers) != 2 * len(positions):
        msg = (
            f"Expected len(markers) == 2 * len(positions), "
            f"but {len(markers)} != {2 * len(positions)}."
        )
        raise ValueError(msg)

    toks = list(tokens)

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
        msg = (
            f"Argument _positions_ must be a list or dict. "
            f"Instead, got {type(positions)}"
        )
        raise ValueError(msg)

    markings = sorted(markings)
    for i, ((si, ei), idx) in enumerate(markings):
        if markers:
            start_marker = markers[2 * idx]
            end_marker = markers[2 * idx + 1]
        else:
            start_marker = f"[[BEGIN{idx}]]"
            end_marker = f"[[END{idx}]]"
        if style == "insert":
            toks.insert(si + 2 * i, start_marker)
            toks.insert(ei + 2 * (i + 1), end_marker)
        elif style == "concatenate":
            toks[si] = start_marker + toks[si]
            toks[ei] = toks[ei] + end_marker
        else:
            raise NotImplementedError
    return toks
