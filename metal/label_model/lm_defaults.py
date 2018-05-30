lm_model_defaults = {

}

lm_train_defaults = {
    # Modeling
    'gamma_init': 0.5, 
    'l2': 0.0, 
    # Optimizer
    'optimizer_params': {
        'lr': 0.1,
    },
    # Optimizer - SGD
    'sgd_params': {
        'momentum': 0.9, 
    },
    # Train loop
    'n_epochs': 100, 
    'print_at': 10, 
    'verbose': True
}