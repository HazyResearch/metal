import itertools

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
        self.K = cardinalities  # Cardinalities for each task
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
        for y in itertools.product(*[range(1, k + 1) for k in self.K]):
            yield np.array(y)


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
        # Every feasible vector corresponds to a leaf node value in
        # {1, ..., K[t]-1}, with the K[t] value reserved for special "N/A" val
        if self.t > 1:
            for t in self.leaf_nodes:
                for yt in range(1, self.K[t]):
                    # By default set all task labels to "N/A" value to start
                    # The default "N/A" value for each task is the last value
                    y = np.array(self.K)

                    # Traverse up the tree
                    y[t] = yt
                    pt = t
                    while pt > 0:
                        ct = pt
                        pt = list(self.G.predecessors(pt))[0]
                        y[pt] = list(self.G.successors(pt)).index(ct) + 1
                    yield y

        # Handle the trivial single-node setting, since technically this is a
        # hierarchy still...
        else:
            for yt in range(self.K[0]):
                yield np.array([yt + 1])
