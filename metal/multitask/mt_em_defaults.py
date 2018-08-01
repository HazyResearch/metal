mt_em_default_config = {
    'task_head_output_dims': None,
        # A list of ints (defaults to K_t if None)
    'task_head_layers': 'top',
        # Optionally specify the layers that each head should attach to
        # For single-task settings, this is always 'top'
        #   'top': connect all heads to the final (top) layer
        #   'auto': connect heads at layers corresponding to placement in the 
        #       task graph; the deepest leaf attaches to the top layer, then 
        #       work backward
        #   [list]: specify explicitly the layer for each head
    'pass_predictions': False,
        # If True, pass output of parent tasks as additional input to children tasks
}