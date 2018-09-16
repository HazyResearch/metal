from collections import OrderedDict
from itertools import chain, product

import networkx as nx


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
        abstains: (bool) Whether to include abstains (0 label)
        deps_graph [optional]: A DependenciesGraph object; must specify either
            this or a set of edges
        edges [optional]: A list of tuples (i,j) representing edges; this will
            be used to create a DependenciesGraph object
    """

    def __init__(
        self,
        m,
        k,
        abstains=True,
        deps_graph=None,
        edges=None,
        higher_order_cliques=False,
    ):
        self.m = m
        self.k = k
        self.abstains = abstains
        self.k0 = 0 if abstains else 1
        if deps_graph is not None:
            self.deps_graph = deps_graph
        elif edges is not None:
            self.deps_graph = DependenciesGraph(self.m, edges=edges)
        else:
            raise ValueError("Must provide either deps_graph or edges.")
        self.G = self._get_junction_tree()

        # Materialize the mappings from index --> val set once here
        self.O_map = [
            vals
            for idx, vals in self.iter_observed(
                higher_order_cliques=higher_order_cliques
            )
        ]
        self.O_d = len(self.O_map)

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

    def _get_members(self, clique_id):
        """Returns the member variable ids of a clique or separator set"""
        if isinstance(clique_id, int):
            return self.G.node[clique_id]["members"]
        else:
            return self.G.edges()[clique_id[0], clique_id[1]]["members"]

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

        for vals in product(*val_ranges.values()):
            yield dict(zip(val_ranges.keys(), vals))

    def iter_observed(self, higher_order_cliques=True):
        """Iterates over (i, vals) where i is an index and vals is a dictionary
        mapping from LF indices to values

        If higher_order_cliques = False, just iterates over the self.m
        individual LF indicator blocks (each of which has self.k entries
        if abstains=False, else self.k-1).

        If higher_order_cliques = True, includes all maximal cliques in the
        junction tree as well, minus Y.
        """

        # Unary observed cliques, {\lf_i}
        cliques = [{i} for i in range(self.m)]

        # Optionally, add higher-order observed cliques, which are the maximal
        # cliques (nodes) of the junction tree, minus Y
        if higher_order_cliques:
            for ci in self.G.nodes():
                cids = self._get_members(ci) - {self.m}
                if cids not in cliques:
                    cliques.append(cids)

        # Iterate over the index, minimal set of values as dict
        idx = -1
        for cids in cliques:
            for vals in self.iter_vals(cids, offset=1):
                idx += 1
                yield idx, vals

    # TODO: REMOVE THIS?
    def _get_clique_data(self, higher_order_cliques=True):
        """Create a helper data structure which maps cliques (as tuples of
        member sources) --> {start_index, end_index, maximal_cliques}, where
        the last value is a set of indices in this data structure.
        """
        c_data = OrderedDict()

        # Add all the unary cliques
        for i in range(self.m):
            c_data[(i,)] = {
                "start_index": i * self.k,
                "end_index": (i + 1) * self.k,
                "max_cliques": set(
                    [
                        j
                        for j in self.G.nodes()
                        if i in self.G.node[j]["members"]
                    ]
                ),
                "size": 1,
                "members": {i},
            }

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        start_index = self.m * self.k
        if higher_order_cliques:
            for item in chain(self.G.nodes(), self.G.edges()):
                if isinstance(item, int):
                    C = self.G.node[item]
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = self.G[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)

                # Important to sort here!!
                members = sorted(list(C["members"]))
                nc = len(members)
                id = tuple(members)

                # Check if already added
                if id in c_data:
                    continue

                if nc > 1:
                    w = self.k ** nc
                    c_data[id] = {
                        "start_index": start_index,
                        "end_index": start_index + w,
                        "max_cliques": set([item])
                        if C_type == "node"
                        else set(item),
                        "size": nc,
                        "members": set(members),
                    }
                    start_index += w
        d = start_index
        return c_data, d

    def iter_index(self):
        """Iterates over the (clique_members, values) indices, returning them
        as a dictionary {lf_idx : val, ...}."""
        for c, c_data in self.c_data.items():
            for vals in product(range(1, self.k + 1), repeat=c_data["size"]):
                yield {li: v for li, v in zip(c, vals)}
