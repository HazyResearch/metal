def split_full_metric(full_metric):
    """Splits a full metric name (split/name or task/split/label/name) into pieces"""
    pieces = full_metric.split("/")
    if len(pieces) == 2:  # Single-task metric
        split, name = pieces
        return split, name
    elif len(pieces) == 4:  # Mmtl metric
        task, payload, label, name = pieces
        return task, payload, label, name
    else:
        msg = (
            f"Required a full metric name (split/name or task/payload/label/name) but "
            f"instead received: {full_metric}"
        )
        raise Exception(msg)
