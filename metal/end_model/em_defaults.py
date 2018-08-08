em_default_config = {
    ### GENERAL
    'seed': None,
    'verbose': True,
    'show_plots': True,

    # Network
    'batchnorm': False,
    'dropout': 0.0,
    'layer_out_dims': [100, 50],
        # The first value is the output dim of the input module (or the sum of
        # the output dims of all the input modules if multitask=True and 
        # multiple input modules are provided). The input module is layer 0.
        # The remaining values are the output dims of middle layers
        # The task head is attached to the final middle layer and has an
        # output dim equal to the cardinality of the classifier.

    ### TRAINING
    'train_config': {
        # Display
        'print_every': 1, # Print after this many epochs

        # GPU
        'use_cuda': False,

        # Dataloader
        'data_loader_config': {
            'batch_size': 32, 
            'num_workers': 1,
        },

        # Train Loop
        'n_epochs': 10,
        # 'grad_clip': 0.0,
        # 'converged': 20,  # if not 0, stop early if this many epochs pass without improvement in training metric
        # 'early_stopping': False, # if true, save any model with best score so far
        # 'checkpoint_runway': 0, # if early stopping, don't save checkpoints until after at least this many epochs
        'l2': 0.0,
        'validation_metric': 'accuracy',

        # Optimizer
        'optimizer_config': {
            'optimizer': 'adam',
            'optimizer_common': {
                'lr': 0.01,
                'weight_decay': 0.0,
            },
            # Optimizer - SGD
            'sgd_config': {
                'momentum': 0.9,
            },
            # Optimizer - Adam
            'adam_config': {
                'betas': (0.9, 0.999)
            },
        },

        # Scheduler
        'scheduler_config': {
            'scheduler': 'reduce_on_plateau', 
                # ['constant', 'exponential', 'reduce_on_plateu']
            'lr_freeze': 0, 
                # Freeze learning rate initially this many epochs
            # Scheduler - exponential
            'exponential_config': {
                'gamma': 0.9, # decay rate
            },
            # Scheduler - reduce_on_plateau
            'plateau_config': {
                'factor': 0.5,
                'patience': 10,
                'threshold': 0.0001,
                'min_lr': 1e-4,
            },
        },

        # Checkpointer
        'checkpoint': True,
        'checkpoint_config': {
            'checkpoint_min': 0,
            'checkpoint_runway': 0,
        }
    },
}
