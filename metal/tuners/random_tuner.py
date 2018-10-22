import time

from metal.tuners.tuner import ModelTuner
from metal.utils import recursive_merge_dicts


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
        dev_data,
        init_args=[],
        train_args=[],
        init_kwargs={},
        train_kwargs={},
        max_search=None,
        shuffle=True,
        verbose=True,
        **score_kwargs,
    ):
        """
        Args:
            search_space: see config_generator() documentation
            dev_data: a tuple of Tensors (X,Y), a Dataset, or a DataLoader of
                X (data) and Y (labels) for the dev split
            init_args: (list) positional args for initializing the model
            train_args: (list) positional args for training the model
            init_kwargs: (dict) keyword args for initializing the model
            train_kwargs: (dict) keyword args for training the model
            max_search: see config_generator() documentation
            shuffle: see config_generator() documentation

        Returns:
            best_model: the highest performing trained model

        Note: Initialization is performed by ModelTuner instead of passing a
        pre-initialized model so that tuning may be performed over all model
        parameters, including the network architecture (which is defined before
        the train loop).
        """
        # Clear run stats
        self.run_stats = []

        configs = self.config_generator(search_space, max_search, shuffle)
        print_worthy = [
            k
            for k, v in search_space.items()
            if isinstance(v, list) or isinstance(v, dict)
        ]

        best_index = 0
        best_score = -1
        best_model = None
        start_time = time.time()
        for i, config in enumerate(configs):
            # Unless seeds are given explicitly, give each config a unique one
            if config.get("seed", None) is None:
                config["seed"] = self.seed + i

            # Integrating generated config into init kwargs and train kwargs
            init_kwargs = recursive_merge_dicts(init_kwargs, config)
            train_kwargs = recursive_merge_dicts(train_kwargs, config)

            # Removing potential duplicate keys -- error occurs without this!
            # for ky in config.keys():
            #     if init_kwargs.get("train_config", None) is not None:
            #         init_kwargs["train_config"] = self.remove_key(
            #             init_kwargs["train_config"],ky)
            #     train_kwargs = self.remove_key(
            #         init_kwargs,ky)

            # Initializing model
            # import ipdb; ipdb.set_trace()
            model = self.model_class(*init_args, **init_kwargs)

            if verbose:
                print_config = {
                    k: v for k, v in config.items() if k in print_worthy
                }
                print("=" * 60)
                print(f"[{i + 1}] Testing {print_config}")
                print("=" * 60)

            model.train(
                *train_args, **train_kwargs, dev_data=dev_data, verbose=verbose
            )
            score = model.score(dev_data, verbose=verbose, **score_kwargs)

            if score > best_score or best_model is None:
                best_index = i + 1
                best_model = model
                best_score = score
                best_config = config

                # Keep track of running statistics
                time_elapsed = time.time() - start_time
                self.run_stats.append(
                    {
                        "time_elapsed": time_elapsed,
                        "best_score": best_score,
                        "best_config": best_config,
                    }
                )

        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"Best model: [{best_index}]")
        print(f"Best config: {best_config}")
        print(f"Best score: {best_score}")
        print("=" * 60)

        return best_model
