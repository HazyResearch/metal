import sys
sys.path.append("../metal")

import unittest

import numpy as np
import torch

from synthetics.generate import generate_single_task_unipolar
from metal.label_model.label_model import LabelModel
from metal.label_model.baselines import MajorityLabelVoter


class LabelModelTest(unittest.TestCase):

    def test_single_task_basic(self):
        # Set seed
        np.random.seed(1)

        # Generate unipolar L for single task
        N, M = 10000, 20
        L, Y, metadata = generate_single_task_unipolar(N, M, acc=[0.4, 0.8], 
            rec=[0.5])

        # Initialize label model
        model = LabelModel()

        # Train label model
        model.train(L, accs=metadata['accs'])

        # Test label model
        model.score([L], [Y])

        # Compare with MV
        print("\nMajority vote baseline:")
        mv = MajorityLabelVoter()
        mv.train([L])
        mv.score([L], [Y])


if __name__ == '__main__':
    unittest.main()