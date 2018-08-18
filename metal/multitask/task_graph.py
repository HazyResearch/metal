import itertools
from collections import defaultdict

import networkx as nx
import numpy as np


class TaskGraph(object):
    """A directed graph defining dependencies between tasks

    In the MTLabelModel, the TaskGraph is used to define a feasible subset of 
    all t-dimensional label vectors Y = [Y_1,...,Y_t]; for example, in a 
    mutually exclusive hierarchy, an example cannot have multiple non-zero leaf
    labels.

    In the MTEndModel, the TaskGraph is optionally used for auto-compilation of
    an MTL network that attaches task heads at appropriate levels and passes
    relevant information between tasks.

    Args:
        edges: A list of (a,b) tuples meaning a is a parent of b in a tree.
        cardinalities: A t-length list of integers corresponding to the
            cardinalities of each task.

    Defaults to a single binary task.
    """

    def __init__(self, cardinalities=[2], edges=[]):
        self.K = cardinalities  # Max cardinality
        self.t = len(cardinalities)  # Total number of tasks
        self.edges = edges

        # Create the graph of tasks
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.t))
        self.G.add_edges_from(edges)

        # Pre-compute parents, children, and leaf nodes
        self.leaf_nodes = [
            i for i in self.G.nodes() if self.G.out_degree(i) == 0
        ]
        self.parents = {t: self.get_parent(t) for t in range(self.t)}
        self.children = {t: self.get_children(t) for t in range(self.t)}

        # Save the cardinality of the feasible set
        self.k = len(list(self.feasible_set()))

    def __eq__(self, other):
        return self.edges == other.edges and self.K == other.K

    def get_parent(self, node):
        return sorted(list(self.G.predecessors(node)))

    def get_children(self, node):
        return sorted(list(self.G.successors(node)))

    def is_feasible(self, y):
        """Boolean indicator if the given y vector is valid (default: True)"""
        return True

    def feasible_set(self):
        """Iterator over values in feasible set"""
        for y in itertools.product(*[range(k) for k in self.K]):
            yield y


class TaskHierarchy(TaskGraph):
    """A mutually-exclusive task hierarchy (a tree)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check that G is a tree
        if not nx.is_tree(self.G):
            raise ValueError(
                f"G is not a tree with edges {self.edges}. "
                "If a tree is not required, use the generic TaskGraph class."
            )

    def is_feasible(self, y):
        return y in list(self.feasible_set())

    def feasible_set(self):
        for i in self.leaf_nodes:
            for yi in range(1, max(self.K) + 1):
                # Set all values to default of -1 = not applicable, except leaf
                y = -1 * np.ones(self.t)
                y[i] = yi

                # Traverse up the tree
                pi = i
                while pi > 0:
                    ci = pi
                    pi = list(self.G.predecessors(pi))[0]
                    y[pi] = list(self.G.successors(pi)).index(ci) + 1
                yield y
