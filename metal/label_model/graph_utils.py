from itertools import combinations, product

import networkx as nx
import numpy as np

def generate_edges(M, clique_size, num_cliques):
    """Given a type of dependency structure to create, returns a list of edges.

    Args:
        M: (int) Number of labeling functions 
        clique_size: (int) Maximum size of clique/block in Sigma matrix
        num_cliques: (int) Number of cliques/blocks 
    Returns:
        edges: (list) List of edge tuples
    """
    
    #Valid Input Structure
    assert M/float(clique_size) >= num_cliques
    
    edges = []
    i = 0
    while (True):
        cluster_lfs = range(i,i+clique_size)
        #edge tuples are size 2
        for comb in combinations(cluster_lfs, 2):
            edges.append(comb)
        
        #created a clique
        num_cliques-=1
        if num_cliques <= 0:   
            return edges

        #move to next LF for new clique
        i+=clique_size

def add_y(M, edges):
    """Given a list of edges, add an edge from each node to Y (m+1).

    Args:
        M: (int) Number of labeling functions 
        clique_size: (int) Maximum size of clique/block in Sigma matrix
        num_cliques: (int) Number of cliques/blocks 
    Returns:
        edges: (list) List of edge tuples
    """
    
    #Valid Input Structure
    assert M/float(clique_size) >= num_cliques
    
    edges = []
    i = 0
    while (True):
        cluster_lfs = range(i,i+clique_size)
        #edge tuples are size 2
        for comb in combinations(cluster_lfs, 2):
            edges.append(comb)
        
        #created a clique
        num_cliques-=1
        if num_cliques <= 0:   
            return edges

        #move to next LF for new clique
        i+=clique_size

class DependenciesGraph(object):
    """Helper data structures for source dependencies graph.

    Also adds an (m+1)th node (index = m) representing Y, which is connected to
    all the other nodes."""

    def __init__(self, m, edges=[]):
        self.m = m
        self.G = nx.Graph()
        for i in range(self.m):
            self.G.add_node(i)
            self.G.add_edge(i, self.m)

        # Optionally add manually-specified set of edges here
        self.G.add_edges_from(edges)

    def draw(self):
        labels = dict([(i, f"$\lambda_{{{i}}}$") for i in range(self.m)])
        labels[self.m] = "$Y$"
        nx.draw(self.G, with_labels=True, node_size=500, labels=labels)


class JunctionTree(object):
    """A data structure for representing a set of m labeling functions with
    cardinality k, and their dependencies, as a junction tree.

    Args:
        m: (int) Number of labeling functions
        k: (int) Cardinality of the classification problem
        t: (int) Number of tasks
        abstains: (bool) Whether to include abstains (0 label)
        deps_graph [optional]: A DependenciesGraph object; must specify either
            this or a set of edges
        edges [optional]: A list of tuples (i,j) representing edges; this will
            be used to create a DependenciesGraph object
        higher_order_cliques: (bool) Set here whether to track the higher-order
            cliques or else just the unary cliques throughout. We set this once
            globally for simplicity and to avoid confusions / mismatches.
    """

    def __init__(
        self,
        m,
        k,
        t=1,
        abstains=True,
        deps_graph=None,
        edges=None,
        higher_order_cliques=True,
    ):
        self.m = m
        self.k = k
        self.t = t
        self.abstains = abstains
        self.k0 = 0 if abstains else 1
        if deps_graph is not None:
            self.deps_graph = deps_graph
        elif edges is not None:
            self.deps_graph = DependenciesGraph(self.m, edges=edges)
        else:
            raise ValueError("Must provide either deps_graph or edges.")
        self.G = self._get_junction_tree()
        self.higher_order_cliques = higher_order_cliques

        # Materialize the mappings from index --> val set once here
        # Observed vars
        self.O_map = [vals for idx, vals in self.iter_observed()]
        self.O_d = len(self.O_map)
        self.O_map_full = [vals for idx, vals in self.iter_observed(offset=0)]
        self.O_d_full = len(self.O_map_full)
        # Hidden vars
        self.H_map = [vals for idx, vals in self.iter_hidden()]
        self.H_d = len(self.H_map)

    def _get_junction_tree(self):
        """Given a set of int nodes i and edges (i,j), returns an nx.Graph
        object G which is a clique tree, where:
            - G.node[i]['members'] contains the set of original nodes in the ith
                maximal clique
            - G[i][j]['members'] contains the set of original nodes in the
                seperator set between maximal cliques i and j
        Note: The clique tree of a graph is a junction tree iff the graph is
        triangulated (chordal); thus this function returns only junction trees!

        Note: This method is currently only implemented for chordal graphs;
        TODO: add a step to triangulate non-chordal graphs.
        """
        # Check if graph is chordal
        # TODO: Add step to triangulate graph if not
        if not nx.is_chordal(self.deps_graph.G):
            raise NotImplementedError("Graph triangulation not implemented.")

        # Create maximal clique graph G2
        # Each node is a maximal clique C_i
        # Let w = |C_i \cap C_j|; C_i, C_j have an edge with weight w if w > 0
        G2 = nx.Graph()
        for i, c in enumerate(nx.chordal_graph_cliques(self.deps_graph.G)):
            G2.add_node(i, members=c)
        for i in G2.nodes:
            for j in G2.nodes:
                S = G2.node[i]["members"].intersection(G2.node[j]["members"])
                w = len(S)
                if w > 0:
                    G2.add_edge(i, j, weight=w, members=S)

        # Return a *maximum* spanning tree of G2
        return nx.algorithms.tree.maximum_spanning_tree(G2)

    def draw(self):
        labels = dict(
            [
                (i, f"{i}: {set(self.G.node[i]['members']) - set([self.m])}")
                for i in self.G.nodes()
            ]
        )
        nx.draw(self.G, with_labels=True, node_size=500, labels=labels)

    def get_members(self, clique_id):
        """Returns the member variable ids of a clique or separator set"""
        if isinstance(clique_id, int):
            return self.G.node[clique_id]["members"]
        else:
            return self.G.edges()[clique_id[0], clique_id[1]]["members"]

    def get_maximal_cliques(self, idxs):
        """Returns the indices of the maximal cliques that the variables with
        indices in idxs are in, or else the empty set"""
        try:
            idxs = set(idxs)
        except Exception as e:
            idxs = set([idxs])
        cids = set()
        for ci in self.G.nodes():
            if idxs.issubset(self.get_members(ci)):
                cids.add(ci)
        return cids

    def iter_vals(self, var_ids, fixed={}, offset=0):
        """Iterator over the possible values of a set of variables, yielding a
        dictionary mapping from variable id to value

        Args:
            - var_ids: A list of variable ids; note that self.m is assumed to
                correspond to Y, which does not take on abstain values
            - fixed: A dictionary mapping from variable id to fixed value
            - offset: Offset to use when iterating over value ranges; set to 1
                if forming minimal set of statistics for generalized cov. matrix
        """
        val_ranges = {}
        for i in var_ids:
            if i in fixed:
                val_ranges[i] = [fixed[i]]

            # If variable is Y, do not iterate over abstain values
            elif i == self.m:
                val_ranges[i] = range(1 + offset, self.k + 1)

            # Else, self.k0 controls whether abstains are included or not
            else:
                val_ranges[i] = range(self.k0 + offset, self.k + 1)

        # Make sure variables and values returned are sorted!
        var_ids = sorted(list(var_ids))
        val_ranges_s = [val_ranges[i] for i in var_ids]
        for vals in product(*val_ranges_s):
            yield dict(zip(var_ids, vals))

    def iter_observed(self, offset=1):
        """Iterates over (i, vals) where i is an index and vals is a dictionary
        mapping from LF indices to values, iterating over all observed values:
            - The m LFs
            - If self.higher_order_cliques=True: also iterates over all maximal
                cliques in the junction tree, *but with Y removed*
        """
        # Unary observed cliques, {\lf_i}
        cliques = [{i} for i in range(self.m)]

        # Optionally, add higher-order observed cliques, which are the maximal
        # cliques (nodes) of the junction tree, minus Y
        if self.higher_order_cliques:
            for ci in self.G.nodes():
                cids = self.get_members(ci) - {self.m}

                # Iterate through the full powerset of observed cliques
                for r in range(2, len(cids) + 1):
                    for cids_ss in combinations(cids, r):
                        cids_ss = set(cids_ss)
                        if cids_ss not in cliques:
                            cliques.append(set(cids_ss))

        # Iterate over the index, minimal set of values as dict
        idx = -1
        for cids in cliques:
            for vals in self.iter_vals(cids, offset=offset):
                idx += 1
                yield idx, vals

    def iter_hidden(self, offset=1):
        """Iterates over (i, vals) where i is an index and vals is a dictionary
        mapping from LF indices to values, iterating over all hidden values:
            - Y
            - All the non-singleton separator sets (which all include Y)

        Note: Currently we *don't* ever add the higher-order cliques.
        """
        # Must include at least Y- now observed \union hidden contains all nodes
        cliques = [{self.m}]

        # Next, add all separator sets (i.e. add all new non-singleton sep sets)
        for i, j in self.G.edges():
            cids = self.get_members((i, j))
            if cids not in cliques:
                cliques.append(set(cids))

        # Iterate over the index, minimal set of values as dict
        idx = -1
        for cids in cliques:
            for vals in self.iter_vals(cids, offset=offset):
                idx += 1
                yield idx, vals

    def observed_maximal_mask(self, offset=1):
        """Returns a binary mask selecting just those observed indices
        corresponding to maximal cliques (nodes) in the junction tree."""
        d = self.O_d if offset == 1 else self.O_d_full
        mask = np.zeros(d)
        for idx, vals in self.iter_observed(offset=offset):
            ci_max = list(self.get_maximal_cliques(vals.keys()))
            assert len(ci_max) == 1
            ci_max = ci_max[0]
            ci_max_members = self.get_members(ci_max)
            if len(vals) + 1 == len(ci_max_members):
                mask[idx] = 1
        return mask

    @property
    def ind_model(self):
        """Returns True if there are only edges between LFs and Y (no LF-LF)
        dependencies"""
        return len(self.deps_graph.G.edges()) - self.m == 0

    @property
    def singleton_sep_sets(self):
        """Returns True if the only separator set is {Y}."""
        for i, j in self.G.edges():
            cids = self.get_members((i, j))
            if len(cids) > 1:
                return False
        return True
