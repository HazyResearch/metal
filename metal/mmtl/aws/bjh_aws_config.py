# Arguments for launch script

search_space = {
    # hyperparams
    "tasks": ["COLA"],  # "COLA,SPACY_NER", "COLA,SPACY_POS"],
    "l2": [1e-5],
    "batch_size": 32,
}

launch_args = {
    "tasks": "RTE",
    "max_len": 200,
    "n_epochs": 1,
    "score_every": 0.2,
    "log_every": 0.02,
}
