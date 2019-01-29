from metal.tuners.tuner import ModelTuner


class RandomSearchTuner(ModelTuner):
    """A tuner for models

    Args:
        model: (nn.Module) The model class to train (uninitiated)
        log_dir: The directory in which to save intermediate results
            If no log_dir is given, the model tuner will attempt to keep
            best trained model in memory.
    """

    def search(
        self,
        search_space,
        valid_data,
        init_args=[],
        train_args=[],
        init_kwargs={},
        train_kwargs={},
        module_args={},
        module_kwargs={},
        max_search=None,
        shuffle=True,
        verbose=True,
        clean_up=True,
        seed=None,
        **score_kwargs,
    ):
        """
        Args:
            search_space: see config_generator() documentation
            valid_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the dev split
            init_args: (list) positional args for initializing the model
            train_args: (list) positional args for training the model
            init_kwargs: (dict) keyword args for initializing the model
            train_kwargs: (dict) keyword args for training the model
            module_args: (dict) Dictionary of lists of module args
            module_kwargs: (dict) Dictionary of dictionaries of module kwargs
            max_search: see config_generator() documentation
            shuffle: see config_generator() documentation

        Returns:
            best_model: the highest performing trained model

        Note: Initialization is performed by ModelTuner instead of passing a
        pre-initialized model so that tuning may be performed over all model
        parameters, including the network architecture (which is defined before
        the train loop).
        """
        self._clear_state(seed)
        self.search_space = search_space

        # Generate configs
        configs = self.config_generator(
            search_space, max_search, self.rng, shuffle
        )

        # Commence search
        for i, config in enumerate(configs):
            score, model = self._test_model_config(
                i,
                config,
                valid_data,
                init_args=init_args,
                train_args=train_args,
                init_kwargs=init_kwargs,
                train_kwargs=train_kwargs,
                module_args=module_args,
                module_kwargs=module_kwargs,
                verbose=verbose,
                **score_kwargs,
            )

        if verbose:
            print("=" * 60)
            print(f"[SUMMARY]")
            print(f"Best model: [{self.best_index}]")
            print(f"Best config: {self.best_config}")
            print(f"Best score: {self.best_score}")
            print("=" * 60)

        self._save_report()

        # Return best model
        return self._load_best_model(clean_up=clean_up)
