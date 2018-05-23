from collections import defaultdict

class TaskGraph(object):
    """A directed acyclic graph of tasks with utility methods.
    TODO: Update this docstring

    Each task t=0,...,T-1 has categorical task labels Y_t=0,...
    For example, we might have three binary prediction tasks:

                                    t_0
                                   /   \
                                 (1)   (2)
                                 t_1   t_2
                                 / \   / \
                                1   2 1   2

    We reserve the value 0 at each node to mean abstain (for labeling) or the
    null label (for gold labels).

    The input to initialize this class is:
        - edges: A set of tuples (p, c) representing a directed edge from task
            p -> c.
        - cardinalities: A list of cardinalities of tasks 0,1,...,T-1 (not 
            counting the "abstain" value in non-root nodes).
    """
    def __init__(self, edges, cardinalities):
        self.edges = edges
        self.K_t = cardinalities     # Cardinalities by task
        self.K = max(cardinalities)  # Max cardinality
        self.T = len(cardinalities)  # Total number of tasks

        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        self.neighbors = defaultdict(list)
        for p, c in self.edges:
            self.children[p].append(c)
            self.parents[c].append(p)
            self.neighbors[c].append(p)
            self.neighbors[p].append(c)

        # Sort parents, children, and neighbors by index
        # Note that this means that task label k corresponds to child k-1
        for p, children in self.children.items():
            self.children[p] = sorted(list(set(children)))
        for c, parents in self.parents.items():
            self.parents[c] = sorted(list(set(parents)))
        for x, neighbors in self.neighbors.items():
            self.neighbors[x] = sorted(list(set(neighbors)))

        # Get depth (max-length path) of DAG
        # Note that a single node has depth 0)
        # TODO: implement _compute_depth()
        # self.depth = self._compute_depth()


    def __eq__(self, other):
        return self.edges == other.edges and self.K_t == other.K_t

    def _compute_depth(self):
        raise NotImplementedError

class SingleTaskGraph(TaskGraph):
    """
    A TaskTree with a single task.
    
    For example, a flat binary classification task can be represented as a
    SingleTaskTree with one node whose cardinality is 2.
    """    
    def __init__(self, k=2):
        cardinalities = [k]
        edges = []
        super().__init__(edges, cardinalities)