import unittest

from metal import LabelModel, EndModel

class ClassifierTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load synthetic dataset?
        pass

    # def test_label_model_load_save(self):
    #     # Create first model
    #     lm1 = LabelModel()
    #     lm1.train()
    #     score1 = lm1.score()
    #     saved = lm1.save()

    #     lm2 = LabelModel.load(saved)
    #     score2 = lm2.score()

    #     self.assertEqual(score1, score2)

    # def test_end_model_load_save(self):
    #     # Create first model
    #     lm1 = EndModel()
    #     lm1.train()
    #     score1 = lm1.score()
    #     saved = lm1.save()

    #     lm2 = EndModel.load(saved)
    #     score2 = lm2.score()

    #     self.assertEqual(score1, score2)