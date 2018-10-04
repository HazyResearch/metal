import unittest

from metal.multitask.task_graph import TaskGraph


class TaskGraphTest(unittest.TestCase):
    def test_binary_tree(self):
        cardinalities = [2, 2, 2]
        edges = [(0, 1), (0, 2)]
        tg = TaskGraph(cardinalities, edges)
        self.assertTrue(tg.parents[0] == [])
        self.assertTrue(tg.parents[1] == [0])
        self.assertTrue(tg.parents[2] == [0])
        self.assertTrue(tg.children[0] == [1, 2])
        self.assertTrue(tg.children[1] == [])
        self.assertTrue(tg.children[2] == [])

    def test_nonbinary_tree(self):
        cardinalities = [3, 2, 2, 2]
        edges = [(0, 1), (0, 2), (0, 3)]
        tg = TaskGraph(cardinalities, edges)
        self.assertTrue(tg.parents[1] == [0])
        self.assertTrue(tg.children[0] == [1, 2, 3])

    def test_binary_tree_depth3(self):
        cardinalities = [2, 2, 2, 2, 2, 2, 2]
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        tg = TaskGraph(cardinalities, edges)
        self.assertTrue(tg.parents[1] == [0])
        self.assertTrue(tg.children[1] == [3, 4])

    def test_unbalanced_tree(self):
        cardinalities = [2, 2, 2]
        edges = [(0, 1), (1, 2)]
        tg = TaskGraph(cardinalities, edges)
        self.assertTrue(tg.parents[1] == [0])
        self.assertTrue(tg.children[1] == [2])


if __name__ == "__main__":
    unittest.main()
