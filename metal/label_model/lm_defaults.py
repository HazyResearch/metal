lm_model_defaults = {
    ### GENERAL
    'seed': None,
    'verbose': True,
    
    ### TRAIN
    'train_config': {
        # Modeling
        'gamma_init': 0.5, 
        'l2': 0.0, 
        # Optimizer
        'optimizer_config': {
            'optimizer_common': {
                'lr': 0.1,
            },
            # Optimizer - SGD
            'sgd_config': {
                'momentum': 0.9, 
            },
        },
        # Train loop
        'n_epochs': 100, 
        'print_at': 10, 
    },
}
