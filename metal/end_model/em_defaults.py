em_model_defaults = {
    'batchnorm': True,
    'dropout': 0.5,
}

em_train_defaults = {
    'print_every': 1, # Print after this many epochs
    'verbose': True,
    'n_epochs': 10,
    # Optimizer
    'optimizer': 'sgd',
    'lr': 0.01,
    'momentum': 0.9,
    'l2': 0.0,
    'converged': 20,  # if not 0, stop early if this many epochs pass without improvement in training metric
    'early_stopping': False, # if true, save any model with best score so far
    'checkpoint_runway': 0, # if early stopping, don't save checkpoints until after at least this many epochs
    'grad_clip': 0.0,
    # Optimizer - Adam
    'beta1': 0.9,
    'beta2': 0.999,
    'data_loader_params': {
        'batch_size': 128, 
        'num_workers': 1,
    },
    # Scheduler
    'scheduler':'reduce_on_plateau', # ['constant', 'exponential', 'reduce_on_plateu']
    'scheduler_params': {
        'lr_freeze': 0, # Freeze learning rate initially this many epochs
        # Scheduler - exponential
        'gamma': 0.9, # decay rate
        # Scheduler - reduce_on_plateau
        'factor': 0.1,
        'patience': 10,
        'threshold': 0.0001,
        'min_lr': 1e-4
    }
}