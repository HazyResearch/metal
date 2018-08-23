import unittest

import numpy as np

from metal.tuners.random_tuner import RandomSearchTuner


class RandomSearchModelTunerTest(unittest.TestCase):
    def test_config_constant(self):
        search_space = {"a": 1}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 1)

    def test_config_list(self):
        search_space = {"a": [1, 2]}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 2)

    def test_config_two_values(self):
        search_space = {"a": [1], "b": [1, 2, 3]}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 3)

    def test_config_range(self):
        search_space = {"a": [1], "b": [1, 2, 3], "c": {"range": [1, 10]}}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 10)

    def test_config_unbounded_max_search(self):
        search_space = {"a": [1], "b": [1, 2, 3], "c": {"range": [1, 10]}}
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=0))
        self.assertEqual(len(configs), 3)

    def test_config_log_range(self):
        search_space = {
            "a": [1],
            "b": [1, 2, 3],
            "c": {"range": [1, 10]},
            "d": {"range": [1, 10], "scale": "log"},
        }
        tuner = RandomSearchTuner(None, None, seed=123)
        configs = list(tuner.config_generator(search_space, max_search=20))
        self.assertEqual(len(configs), 20)
        self.assertGreater(
            np.mean([c["c"] for c in configs]),
            np.mean([c["d"] for c in configs]),
        )


if __name__ == "__main__":
    unittest.main()
