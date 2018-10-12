import numpy as np

class DependencyLearner(object):
    """Simple abstract base class for an LF dependency learner.

    The main contribution of the children classes will be an implementation of the 
    find_edges() method. The find_edges() method returns a list of tuples, where 
    each tuple represents a pair of LFs that are dependent.

    Args:
        k: (int) The cardinality of the dataset
    """

    def __init__(self, k):
        super().__init__()
        if self.k != 2:
            raise NotImplementedError("Dependency learning only works for k=2")

    def _force_singleton(self,deps):
        """Ensure that learned dependencies for singleton separators.

        TODO: build checker for no complete graphs with these additions

        Args:
            deps: (list) List of tuples of LF pairs that are dependent

        Example:
            if (0,1) and (1,2) exist, add (0,2)
            if (0,1) and (0,2) exist, add (1,2)
            if (0,3) and (2,3) exist, add (0,2)
            if (1,3) and (0,1) exist, add (0,3)
        """
        deps_singleton = []
        for i,j in deps:
            if i < j:
                deps_singleton.append((i,j))

        for i,j in deps:
            for k,l in deps:
                if (i == k) and (j < l):
                    deps_singleton.append((j,l))
                if (j == l) and (i < k):
                    deps_singleton.append((i,k))
                if (j == k) and (i < l):
                    deps_singleton.append((i,l))
                if (i == l) and (j < k):
                    deps_singleton.append((j,k))
        return deps_singleton