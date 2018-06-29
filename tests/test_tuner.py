import sys
import unittest

import numpy as np

sys.path.append("../metal")
from metal.tuner import ModelTuner

class TunerTest(unittest.TestCase):
    
    # @classmethod
    # def setUpClass(cls):
    #     cls.tuner = ModelTuner(None, None, 123)

    def test_config_generator(self):
        search_space = {'a': 1}
        tuner = ModelTuner(None, None, 123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 1)

        search_space = {'a': [1, 2]}
        tuner = ModelTuner(None, None, 123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 2)

        search_space = {'a': [1],
                        'b': [1, 2, 3]}
        tuner = ModelTuner(None, None, 123)
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 3)

        search_space = {'a': [1],
                        'b': [1, 2, 3],
                        'c': {'range': [1, 10]}}
        tuner = ModelTuner(None, None, 123)                        
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 10)

        search_space = {'a': [1],
                        'b': [1, 2, 3],
                        'c': {'range': [1, 10]}}
        tuner = ModelTuner(None, None, 123)                        
        configs = list(tuner.config_generator(search_space, max_search=0))
        self.assertEqual(len(configs), 3)

        search_space = {'a': [1],
                        'b': [1, 2, 3],
                        'c': {'range': [1, 10]},
                        'd': {'range': [1, 10], 'scale': 'log'}}
        tuner = ModelTuner(None, None, 123)                        
        configs = list(tuner.config_generator(search_space, max_search=10))
        self.assertEqual(len(configs), 10)
        self.assertGreater(
            np.mean([c['c'] for c in configs]), 
            np.mean([c['d'] for c in configs]))


if __name__ == '__main__':
    unittest.main()