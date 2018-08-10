from itertools import cycle, product, islice
import math
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

        if seed is None:
            self.seed = 0
        else:
            random.seed(seed)
            self.seed = seed

    def search(self, init_args, train_args, X_dev, Y_dev, search_space, 
        max_search=None, shuffle=True, verbose=True, **score_kwargs):
        """
        Args:
            init_args: (list) positional args for initializing the model
            train_args: (list) positional args for training the model
            X_dev: The appropriate input for evaluating the given model
            Y_dev: An [n] or [n, 1] tensor of gold labels in {0,...,K_t} or a
                t-length list of such tensors if model.multitask=True.
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
        
        best_index = 0
        best_score = -1
        best_model = None
        for i, config in enumerate(configs):
            # Unless seeds are given explicitly, give each config a unique one
            if config.get('seed', None) is None:
                config['seed'] = self.seed + i
            model = self.model_class(*init_args, **config)

            if verbose:
                print_config = {k: v for k, v in config.items() if k in 
                    print_worthy}
                print("=" * 60)
                print(f"[{i + 1}] Testing {print_config}")
                print("=" * 60)

            try:
                model.train(*train_args, X_dev=X_dev, Y_dev=Y_dev, **config)
                score = model.score(X_dev, Y_dev, verbose=verbose, **score_kwargs)
            except:
                score = float("nan")

            if score > best_score:
                best_index = i + 1
                best_model = model
                best_score = score
                best_config = config

        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"Best model: [{best_index}]")
        print(f"Best config: {best_config}")
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

class HyperbandTuner(ModelTuner):
    """Performs hyperparameter search according to the Hyperband algorithm (https://arxiv.org/pdf/1603.06560.pdf).

    Args:
        model: (nn.Module) The model class to train (uninitiated)
        hyperband_epochs_budget: Number of total epochs of training to perform in search. 
        hyperband_proportion_discard: proportion of configurations to discard in 
            each round of Hyperband's SuccessiveHalving. An integer.
        log_dir: The directory in which to save intermediate results
            If no log_dir is given, the model tuner will attempt to keep
            all trained models in memory.
        seed: Random seed        
    """
    def __init__(self, model_class, 
                 hyperband_epochs_budget=200,
                 hyperband_proportion_discard=3,
                 log_dir=None,
                 seed=None):
        super().__init__(model_class, log_dir, seed)
        
        # Set random seed (Note this only makes sense in single threaded mode)
        self.rand_state = np.random.RandomState()
        self.rand_state.seed(self.seed)
        
        # Hyperband parameters
        self.hyperband_epochs_budget = hyperband_epochs_budget
        self.hyperband_proportion_discard = hyperband_proportion_discard

        # Given the budget, generate the largest hyperband schedule within budget
        self.hyperband_schedule = (
            self.get_largest_hyperband_schedule_within_budget(self.hyperband_epochs_budget,
                                                              self.hyperband_proportion_discard))
        
        # Print the search schedule
        self.pretty_print_schedule(self.hyperband_schedule)

    def pretty_print_schedule(self, hyperband_schedule, describe_hyperband=True):
        """
        Prints scheduler for user to read.
        """
        print("=========================================")
        print("|           Hyperband Schedule          |")
        print("=========================================")
        if describe_hyperband:
            # Print a message indicating what the below schedule means
            print("Table consists of tuples of (num configs, num_resources_per_config) which specify "
                  "how many configs to run and for how many epochs. ")
            print("Each bracket starts with a list of random configurations which is successively halved "
                  "according the schedule.")
            print("See the Hyperband paper (https://arxiv.org/pdf/1603.06560.pdf) for more details.")
            print("-----------------------------------------")
        for bracket_index, bracket in enumerate(hyperband_schedule):
            bracket_string = "Bracket %d:" % bracket_index
            for n_i,r_i in bracket:
                bracket_string += " (%d, %d)" % (n_i, r_i)
            print(bracket_string)
        print("-----------------------------------------")
    
    def get_largest_hyperband_schedule_within_budget(self, budget, proportion_discard):
        """
        Gets the largest hyperband schedule within target_budget.
        This is required since the original hyperband algorithm uses R, 
        the maximum number of resources per configuration.
        TODO(maxlam): Possibly binary search it if this becomes a bottleneck.
        
        Args:
            budget: total budget of the schedule.
            proportion_discard: hyperband parameter that specifies 
                the proportion of configurations to discard per iteration.
        """
        
        # Exhaustively generate schedules and check if they're within budget, adding to a list.
        valid_schedules_and_costs = []
        for R in range(1, budget):
            schedule = self.generate_hyperband_schedule(R, proportion_discard)
            cost = self.compute_schedule_cost(schedule)
            if cost <= budget:
                valid_schedules_and_costs.append((schedule, cost))

        # Choose a valid schedule that maximizes usage of the budget.
        valid_schedules_and_costs.sort(key=lambda x:x[1], reverse=True)
        return valid_schedules_and_costs[0][0]

    def compute_schedule_cost(self, schedule):
        # Sum up all n_i * r_i for each band.
        flattened = [item for sublist in schedule for item in sublist]
        return sum([x[0]*x[1] for x in flattened])

    def generate_hyperband_schedule(self, R, eta):
        """
        Generate hyperband schedule according to the paper.

        Args:
            R: maximum resources per config.
            eta: proportion of configruations to discard per 
                iteration of successive halving.        
            
        Returns: hyperband schedule, which is represented as a list of brackets, 
            where each bracket contains a list of (num configurations, 
            num resources to use per configuration). See the paper for more details.
        """
        schedule = []
        s_max = int(math.floor(math.log(R, eta)))
        B = (s_max + 1) * R        
        for s in range(0, s_max+1):
            n = math.ceil(int((s_max+1)/(s+1)) * eta**s)
            r = R*eta**(-s)
            num_hyperparameters = n
            bracket = []
            for i in range(0, s+1):
                n_i = int(math.floor(n*eta**(-i)))
                r_i = int(r*eta**i)
                num_hyperparameters = math.floor(n_i / eta)
                bracket.append((n_i, r_i))
            schedule = [bracket] + schedule
        return schedule

    def search(self, init_args, train_args, X_dev, Y_dev, search_space, 
               verbose=True, **score_kwargs):
        """
        Performs hyperband search according to the generated schedule.
        
        At the beginning of each bracket, we generate a list of random configurations
        and perform successive halving on it; we repeat this process for the number
        of brackets in the schedule.        

        Args:
            init_args: (list) positional args for initializing the model
            train_args: (list) positional args for training the model
            X_dev: The appropriate input for evaluating the given model
            Y_dev: An [n] or [n, 1] tensor of gold labels in {0,...,K_t} or a
                t-length list of such tensors if model.multitask=True.
            search_space: see ModelTuner's config_generator() documentation
            max_search: see ModelTuner's config_generator() documentation
            shuffle: see ModelTuner's config_generator() documentation

        Returns:
            best_model: the highest performing trained model found by Hyperband
            best_config: (dict) the config corresponding to the best model

        Note: Initialization is performed by ModelTuner instead of passing a
        pre-initialized model so that tuning may be performed over all model
        parameters, including the network architecture (which is defined before
        the train loop).
        """

        print_worthy = [k for k, v in search_space.items() 
                        if isinstance(v, list) or isinstance(v, dict)]
        
        # Loop over each bracket
        best_model, best_score, best_configuration, best_model_index = None, float("-inf"), None, -1
        n_models_scored = 0
        for bracket_index, bracket in enumerate(self.hyperband_schedule):
            
            # Sample random configurations to seed SuccessiveHalving
            n_starting_configurations, _ = bracket[0]
            configurations = list(self.config_generator(search_space, 
                                                        max_search=n_starting_configurations, 
                                                        shuffle=True))

            # Successive Halving
            for band_index, (n_i, r_i) in enumerate(bracket):

                assert(len(configurations) == n_i)
                
                # Evaluate each configuration for r_i epochs
                scored_configurations = []
                for configuration in configurations:

                    cur_model_index = n_models_scored
                    
                    # Set seed
                    if configuration.get('seed', None) is None:
                        configuration['seed'] = self.seed + cur_model_index
                                        
                    # Set epochs of the configuration
                    configuration['n_epochs'] = r_i

                    # Train model and get the score
                    model = self.model_class(*init_args, **configuration)                    
                    if verbose:
                        print_config = {k: v for k, v in configuration.items() if k in 
                                        print_worthy}
                        print("=" * 60)
                        print(f"[{cur_model_index} Testing {print_config}")
                        print("=" * 60)

                    try:
                        model.train(*train_args, X_dev=X_dev, Y_dev=Y_dev, **configuration)
                        score = model.score(X_dev, Y_dev, verbose=verbose, **score_kwargs)
                    except:
                        score = float("nan")

                    # Add score and model to list
                    scored_configurations.append((model, score, cur_model_index, configuration))

                    n_models_scored += 1

                # Sort scored configurations by score
                scored_configurations.sort(key=lambda x: x[1], reverse=True)

                # Update the best configuration and score
                for model, score, model_index, configuration in scored_configurations:
                    if score > best_score:
                        best_model, best_score, best_model_index, best_configuration = (
                            model, score, model_index, configuration
                        )                        

                # Successively halve the configurations
                if band_index+1 < len(bracket):
                    n_to_keep, _ = bracket[band_index+1]
                    configurations = [x[3] for x in scored_configurations][:n_to_keep]        

        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"Best model: [{best_model_index}]")
        print(f"Best config: {best_configuration}")
        print(f"Best score: {best_score}")
        print("=" * 60)
        return best_model, best_configuration
