em_model_defaults = {
    # General
    'seed': None,
    'verbose': False,
    # Network
    'batchnorm': True,
    'dropout': 0.0,
        # The first value is the output dim of the input module
    'layer_output_dims': [100, 50],
        # If head_output_dims is None, defaults to K_t for each task t
    'head_output_dims': None,  # Optionally a list
        # Optionally specify the layers that each head should attach to
        # 'top': connect all heads to the final (top) layer
        # 'auto': connect heads at layers corresponding to placement in the task
        #    graph; the deepest leaf attaches to the top layer, then work backward
        # [list]: specify explicitly the layer for each head
    'head_layers': 'top',
        # If True, pass output of parent tasks as additional input to children tasks
    'pass_predictions': False,
}

em_train_defaults = {
    # Display
    'verbose': True,
    'print_every': 1, # Print after this many epochs

    # GPU
    'use_cuda': False,

    # Dataloader
    'data_loader_params': {
        'batch_size': 128, 
        'num_workers': 1,
    },

    # Train Loop
    'n_epochs': 10,
    # 'grad_clip': 0.0,
    # 'converged': 20,  # if not 0, stop early if this many epochs pass without improvement in training metric
    # 'early_stopping': False, # if true, save any model with best score so far
    # 'checkpoint_runway': 0, # if early stopping, don't save checkpoints until after at least this many epochs
    'l2': 0.0,

    # Optimizer
    'optimizer': 'sgd',
    'optimizer_params': {
        'lr': 0.01,
        'weight_decay': 0.0,
    },
    # Optimizer - SGD
    'sgd_params': {
        'momentum': 0.9,
    },
    # Optimizer - Adam
    'adam_params': {
        'beta1': 0.9,
        'beta2': 0.999,
    },

    # Scheduler
    'scheduler': 'reduce_on_plateau', # ['constant', 'exponential', 'reduce_on_plateu']
    'scheduler_params': {
        'lr_freeze': 0, # Freeze learning rate initially this many epochs
        # Scheduler - exponential
        'exponential_params': {
            'gamma': 0.9, # decay rate
        },
        # Scheduler - reduce_on_plateau
        'plateau_params': {
            'factor': 0.5,
            'patience': 10,
            'threshold': 0.0001,
            'min_lr': 1e-4,
        },
    },
}