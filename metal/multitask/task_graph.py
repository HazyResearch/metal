from collections import defaultdict

import numpy as np
import networkx as nx


class TaskGraph(object):
    """A TaskGraph for t tasks is an object that defines a feasible subset of
    all t-dimensional label vectors Y = [Y_1,...,Y_t].
    A TaskGraph may also implement additional functionality which connects this
    feasible set function to more interpretable semantics for the user.
    """
    def __init__(self):
        pass
    
    def is_feasible(self, y):
        return False
    
    def feasible_set(self):
        """Iterator over values in feasible set"""
        pass


class TaskHierarchy(TaskGraph):
    """A mutually-exclusive task hierarchy
    
    Args:
        edges: A list of (a,b) tuples meaning a is a parent of b in a tree.
        cardinalities: A t-length list of integers corresponding to the
            cardinalities of each task.
        
    Defaults to a single binary task.
    """
    def __init__(self, edges=[], cardinalities=[2]):
        # Number of tasks
        self.K = cardinalities  # Max cardinality
        self.t = len(cardinalities)  # Total number of tasks
        self.edges = edges
        
        # Create a task hierarchy
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.t))
        self.G.add_edges_from(edges)
        
        # Check that G is a tree
        # if not nx.is_tree(self.G):
        #     raise ValueError(f"G is not a tree with edges {edges}.")
        
        # Check that the cardinality matches the number of children
        # Note: Here we assume that the task indices are topologically ordered!
        # TODO: remove this assumption of topological ordering
        # for s in range(self.t):
        #     if len([r for r in self.G[s].keys() if r > s]) not in [0, self.k]:
        #         raise ValueError("Mismatch between cardinality and tree width.")
        
        # Get the leaf nodes
        # Note: by this definition, not all leaf nodes need be at the same level
        self.leaf_nodes = [i for i in self.G.nodes() if self.G.out_degree(i)==0]
        self.parents = {t: self.get_parent(t) for t in range(self.t)}
        self.children = {t: self.get_children(t) for t in range(self.t)}

    def __len__(self):
        return len(self.leaf_nodes) * max(self.K)

    def __eq__(self, other):
        return self.edges == other.edges and self.K == other.K

    def get_parent(self, node):
        return sorted(list(self.G.predecessors(node)))

    def get_children(self, node):
        return sorted(list(self.G.successors(node)))

    def is_feasible(self, y):
        return (y in list(self.feasible_set()))
    
    def feasible_set(self):
        for i in self.leaf_nodes:
            for yi in range(1, max(self.K)+1):
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


class SingleTaskGraph(TaskHierarchy):
    """
    A TaskGraph with a single task.
    
    For example, a flat binary classification task can be represented as a
    SingleTaskTree with one node whose cardinality is 2.
    """    
    def __init__(self, k=2):
        cardinalities = [k]
        edges = []
        super().__init__(edges, cardinalities)