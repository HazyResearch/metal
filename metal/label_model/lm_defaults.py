lm_model_defaults = {
    ### GENERAL
    'seed': None,
    'verbose': True,
    
    ### TRAIN
    'train_config': {
        # Classifier
        'cardinality': 2,
        # Class balance- fixed value
        'y_pos': 0.5,
        # Class balance initialization / prior (only if y_pos=None)
        'y_pos_init': 0.5,
        # Model params initialization / priors
        'mu_init': 0.4, 
        # L@ regularization (around prior values)
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
