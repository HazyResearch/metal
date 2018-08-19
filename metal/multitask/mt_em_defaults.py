mt_em_default_config = {
    "task_head_layers": "top",
    # Optionally specify the layers that each head should attach to
    # For single-task settings, this is always 'top'
    #   'top': connect all heads to the final (top) layer
    #   [list]: specify explicitly the layer for each head
    "pass_predictions": False,
    # If True, pass output of parent tasks as additional input to children tasks
}
