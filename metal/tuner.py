from itertools import cycle, product
import random

from pprint import pprint
import numpy as np


class ModelTuner(object):
    """A tuner for models

    Args:
        model: (nn.Module) The model class to train (uninitiated)
        log_dir: The directory in which to save intermediate results
            If no log_dir is given, the model tuner will attempt to keep
            all trained models in memory.
    """
    def __init__(self, model_class, log_dir=None, seed=None):
        self.model_class = model_class

        if log_dir is not None:
            raise NotImplementedError
        self.log_dir = log_dir

        if seed is not None:
            np.random.seed(seed)

    def search(self, init_args, train_args, X_dev, Y_dev, search_space, 
        max_search=None, shuffle=True, **score_kwargs):
        """
        Args:
            init_args: (list) positional args for initializing the model
            train_args: (list) positional args for training the model
            X_dev: The appropriate input for evaluating the given model
            Y_dev: An [N] or [N, 1] tensor of gold labels in {1,...,K_t} or a
                T-length list of such tensors if model.multitask=True.
            search_space: see config_generator() documentation
            max_search: see config_generator() documentation
            shuffle: see config_generator() documentation

        Returns:
            best_model: the highest performing trained model
            best_config: (dict) the config corresponding to the best model

        Note: Initialization is performed by ModelTuner instead of passing a
        pre-initialized model so that tuning may be performed over all model
        parameters, including the network architecture (which is defined before
        the train loop).
        """
        configs = self.config_generator(search_space, max_search, shuffle)
        print_worthy = [k for k, v in search_space.items() 
            if isinstance(v, list) or isinstance(v, dict)]
        
        best_score = 0
        best_model = None
        for i, config in enumerate(configs):
            # Set verbose=False to avoid reporting config overwrites
            config['verbose'] = False
            # Unless seeds are given explicitly, give each config a unique one
            if config.get('seed', None) is None:
                config['seed'] = i
            model = self.model_class(*init_args, **config)

            print_config = {k: v for k, v in config.items() if k in print_worthy}
            print("=" * 60)
            print(f"[{i + 1}] Testing {print_config}")
            print("=" * 60)
            model.train(*train_args, X_dev=X_dev, Y_dev=Y_dev, verbose=True, 
                show_plots=False)
            score = model.score(X_dev, Y_dev, **score_kwargs)

            if score > best_score:
                best_index = i + 1
                best_model = model
                best_score = score
                best_config = config

        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"Best model: [{best_index}]")
        print(f"Best config: {config}")
        print(f"Best score: {best_score}")
        print("=" * 60)
        
        return best_model, best_config
        
    @staticmethod
    def config_generator(search_space, max_search, shuffle=True):
        """Generates config dicts from the given search space

        Args:
            search_space: (dict) A dictionary of parameters to search over.
                See note below for more details.
            max_search: (int) The maximum number of configurations to search.
                If max_search is None, do a full grid search of all discrete
                    parameters, filling in range parameters as needed. 
                Otherwise, do a full grid search of all discrete
                    parameters and then cycle through again filling in new
                    range parameters values; if there are no range parameters,
                    stop after yielding the full cross product of parameters
                    once.
            shuffle: (bool) If True, shuffle the order of generated configs

        Yields:
            configs: each config is a dict of parameter values based on the
                provided search space

        The search_space dictionary may consist of two types of parameters:
            --discrete: a discrete parameter is either a single value or a
                list of values. Use single values, for example, to override
                a default model parameter or set a flag such as 'verbose'=True.
            --range: a range parameter is a dict of the form:
                {'range': [<min>, <max>], 'scale': <scale>}
                where <min> and <max> are the min/max values to search between
                and scale is one of ['linear', 'log'] (defaulting to 'linear')
                representing the scale to use when searching the given range
        
        Example:
            search_space = {
                'verbose': True,                              # discrete
                'n_epochs': 100,                              # discrete
                'l1': [0.0, 0.01, 0.1],                       # discrete
                'l2': {'range': [0.0001, 10]}                 # linear range
                'lr': {'range': [0.001, 1], 'scale': 'log'},  # log range
            }
            If max_search is None, this will return 3 configurations (enough to
                just cover the full cross-product of discrete values, filled
                in with sampled range values)
            Otherewise, this will return max_search configurations
                (cycling through the discrete value combinations multiple times
                if necessary)
        """
        def dict_product(d):
            keys = d.keys()
            for element in product(*d.values()):
                yield dict(zip(keys, element))
        
        def range_param_func(v):
            scale = v.get('scale', 'linear')
            mini = min(v['range'])
            maxi = max(v['range'])
            if scale == 'linear':
                func = lambda rand: mini + (maxi - mini) * rand
            elif scale == 'log':
                mini = np.log(mini)
                maxi = np.log(maxi)
                func = lambda rand: np.exp(mini + (maxi - mini) * rand)
            else:
                raise ValueError(f"Unrecognized scale '{scale}' for "
                    "parameter {k}")
            return func

        discretes = {}
        ranges = {}
        for k, v in search_space.items():
            if isinstance(v, dict):
                ranges[k] = range_param_func(v)
            elif isinstance(v, list):
                discretes[k] = v
            else:
                discretes[k] = [v]

        discrete_configs = list(dict_product(discretes))

        if shuffle:
            random.shuffle(discrete_configs)

        # If there are range parameters and a non-None max_search, cycle 
        # through the discrete_configs (with new range values) until max_search 
        # is met
        if ranges and max_search:
            discrete_configs = cycle(discrete_configs)

        for i, config in enumerate(discrete_configs):
            # We may see the same config twice due to cycle
            config = config.copy()  
            if max_search and i == max_search:
                break
            for k, v in ranges.items():
                config[k] = float(v(random.random()))
            yield config