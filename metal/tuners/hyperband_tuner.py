import math
import time

import numpy as np

from metal.tuners.tuner import ModelTuner


class HyperbandTuner(ModelTuner):
    """Performs hyperparameter search according to the Hyperband algorithm

    Reference: (https://arxiv.org/pdf/1603.06560.pdf)

    Args:
        model: (nn.Module) The model class to train (uninitiated)
        hyperband_epochs_budget: Number of total epochs of training to perform
            in search.
        hyperband_proportion_discard: proportion of configurations to discard
            in each round of Hyperband's SuccessiveHalving. An integer.
        log_dir: The directory in which to save intermediate results
            If no log_dir is given, the model tuner will attempt to keep
            all trained models in memory.
        seed: Random seed
    """

    def __init__(
        self,
        model_class,
        hyperband_epochs_budget=200,
        hyperband_proportion_discard=3,
        log_dir=None,
        seed=None,
    ):
        super().__init__(model_class, log_dir, seed)

        # Set random seed (Note this only makes sense in single threaded mode)
        self.rand_state = np.random.RandomState()
        self.rand_state.seed(self.seed)

        # Hyperband parameters
        self.hyperband_epochs_budget = hyperband_epochs_budget
        self.hyperband_proportion_discard = hyperband_proportion_discard

        # Given the budget, generate the largest hyperband schedule
        # within budget
        self.hyperband_schedule = self.get_largest_schedule_within_budget(
            self.hyperband_epochs_budget, self.hyperband_proportion_discard
        )

        # Print the search schedule
        self.pretty_print_schedule(self.hyperband_schedule)

    def pretty_print_schedule(
        self, hyperband_schedule, describe_hyperband=True
    ):
        """
        Prints scheduler for user to read.
        """
        print("=========================================")
        print("|           Hyperband Schedule          |")
        print("=========================================")
        if describe_hyperband:
            # Print a message indicating what the below schedule means
            print(
                "Table consists of tuples of "
                "(num configs, num_resources_per_config)"
                "which specify how many configs to run and"
                "for how many epochs. "
            )
            print(
                "Each bracket starts with a list of random "
                "configurations which is successively halved "
                "according the schedule."
            )
            print(
                "See the Hyperband paper "
                "(https://arxiv.org/pdf/1603.06560.pdf) for more details."
            )
            print("-----------------------------------------")
        for bracket_index, bracket in enumerate(hyperband_schedule):
            bracket_string = "Bracket %d:" % bracket_index
            for n_i, r_i in bracket:
                bracket_string += " (%d, %d)" % (n_i, r_i)
            print(bracket_string)
        print("-----------------------------------------")

    def get_largest_schedule_within_budget(self, budget, proportion_discard):
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

        # Exhaustively generate schedules and check if
        # they're within budget, adding to a list.
        valid_schedules_and_costs = []
        for R in range(1, budget):
            schedule = self.generate_hyperband_schedule(R, proportion_discard)
            cost = self.compute_schedule_cost(schedule)
            if cost <= budget:
                valid_schedules_and_costs.append((schedule, cost))

        # Choose a valid schedule that maximizes usage of the budget.
        valid_schedules_and_costs.sort(key=lambda x: x[1], reverse=True)
        return valid_schedules_and_costs[0][0]

    def compute_schedule_cost(self, schedule):
        # Sum up all n_i * r_i for each band.
        flattened = [item for sublist in schedule for item in sublist]
        return sum([x[0] * x[1] for x in flattened])

    def generate_hyperband_schedule(self, R, eta):
        """
        Generate hyperband schedule according to the paper.

        Args:
            R: maximum resources per config.
            eta: proportion of configruations to discard per
                iteration of successive halving.

        Returns: hyperband schedule, which is represented
            as a list of brackets, where each bracket
            contains a list of (num configurations,
            num resources to use per configuration).
            See the paper for more details.
        """
        schedule = []
        s_max = int(math.floor(math.log(R, eta)))
        # B = (s_max + 1) * R
        for s in range(0, s_max + 1):
            n = math.ceil(int((s_max + 1) / (s + 1)) * eta ** s)
            r = R * eta ** (-s)
            bracket = []
            for i in range(0, s + 1):
                n_i = int(math.floor(n * eta ** (-i)))
                r_i = int(r * eta ** i)
                bracket.append((n_i, r_i))
            schedule = [bracket] + schedule
        return schedule

    def search(
        self,
        search_space,
        X_dev,
        Y_dev,
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
        Performs hyperband search according to the generated schedule.

        At the beginning of each bracket, we generate a
        list of random configurations and perform
        successive halving on it; we repeat this process
        for the number of brackets in the schedule.

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

        # Clear run stats
        self.run_stats = []

        print_worthy = [
            k
            for k, v in search_space.items()
            if isinstance(v, list) or isinstance(v, dict)
        ]

        # Loop over each bracket
        best_model, best_score, best_config, best_model_index = (
            None,
            float("-inf"),
            None,
            -1,
        )
        n_models_scored = 0
        start_time = time.time()
        for bracket_index, bracket in enumerate(self.hyperband_schedule):

            # Sample random configurations to seed SuccessiveHalving
            n_starting_configurations, _ = bracket[0]
            configurations = list(
                self.config_generator(
                    search_space,
                    max_search=n_starting_configurations,
                    shuffle=True,
                )
            )

            # Successive Halving
            for band_index, (n_i, r_i) in enumerate(bracket):

                assert len(configurations) <= n_i

                # Evaluate each configuration for r_i epochs
                scored_configurations = []
                for configuration in configurations:

                    cur_model_index = n_models_scored

                    # Set seed
                    if configuration.get("seed", None) is None:
                        configuration["seed"] = self.seed + cur_model_index

                    # Set epochs of the configuration
                    configuration["n_epochs"] = r_i

                    # Train model and get the score
                    model = self.model_class(*init_args, **configuration)
                    if verbose:
                        print_config = {
                            k: v
                            for k, v in configuration.items()
                            if k in print_worthy
                        }
                        print("=" * 60)
                        print(f"[{cur_model_index} Testing {print_config}")
                        print("=" * 60)

                    model.train(
                        *train_args,
                        X_dev=X_dev,
                        Y_dev=Y_dev,
                        **configuration,
                        verbose=verbose,
                    )
                    score = model.score(
                        X_dev, Y_dev, verbose=verbose, **score_kwargs
                    )

                    # Add score and model to list
                    scored_configurations.append(
                        (model, score, cur_model_index, configuration)
                    )

                    # Update best model and score
                    if score > best_score or best_model is None:
                        best_model = model
                        best_score = score
                        best_model_index = cur_model_index
                        best_config = configuration

                        # Keep track of running statistics
                        time_elapsed = time.time() - start_time
                        self.run_stats.append(
                            {
                                "time_elapsed": time_elapsed,
                                "best_score": best_score,
                                "best_config": best_config,
                            }
                        )

                    n_models_scored += 1

                # Sort scored configurations by score
                scored_configurations.sort(key=lambda x: x[1], reverse=True)

                # Successively halve the configurations
                if band_index + 1 < len(bracket):
                    n_to_keep, _ = bracket[band_index + 1]
                    configurations = [x[3] for x in scored_configurations][
                        :n_to_keep
                    ]

        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"Best model: [{best_model_index}]")
        print(f"Best config: {best_config}")
        print(f"Best score: {best_score}")
        print("=" * 60)
        return best_model
