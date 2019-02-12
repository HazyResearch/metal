def split_full_metric(full_metric):
    """Splits a full metric name (task/split/name or split/name) into its pieces"""
    pieces = full_metric.split("/")
    if len(pieces) == 2:
        split, name = pieces
        task = None
    elif len(pieces) == 3:
        task, split, name = pieces
    else:
        msg = (
            f"Required a full metric name (task/split/name or split/name) but "
            f"instead received: {full_metric}"
        )
        raise Exception(msg)
    return task, split, name


def join_full_metric(task, split, metric):
    """Creates a full  metric name from its component pieces"""
    return f"{task}/{split}/{metric}"
