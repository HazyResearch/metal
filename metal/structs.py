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
    """A mutually-exclusive task hierarchy"""
    def __init__(self, t, edges, k=2):
        # Number of tasks
        self.t = t

        # Cardinality of each task
        # Note: For now, we assume all tasks in the tree have the same 
        # cardinality, which implies a balanced tree- but is easily extended.
        self.k = k
        
        # Create a task hierarchy
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(t))
        self.G.add_edges_from(edges)
        
        # Check that G is a tree
        if not nx.is_tree(self.G):
            raise ValueError(f"G is not a tree with edges {edges}.")
        
        # Check that the cardinality matches the number of children
        # Note: Here we assume that the task indices are topologically ordered!
        for s in range(self.t):
            if len([r for r in self.G[s].keys() if r > s]) not in [0,k]:
                raise ValueError("Mismatch between cardinality and tree width.")
        
        # Get the leaf nodes
        self.leaf_nodes = [i for i in self.G.nodes() if self.G.out_degree(i)==0]
    
    def __len__(self):
        return len(self.leaf_nodes) * self.k

    def is_feasible(self, y):
        return (y in list(self.feasible_set()))
    
    def feasible_set(self):
        for i in self.leaf_nodes:
            for yi in range(1, self.k+1):
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