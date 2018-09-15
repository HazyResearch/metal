from collections import OrderedDict
from itertools import chain, product

import networkx as nx


class JunctionTree(object):
    """A data structure for representing a set of m labeling functions with
    cardinality k, and their dependencies, as a junction tree.

    Args:
        m: (int) Number of labeling functions
        k: (int) Cardinality of the classification problem
        edges: (list of tuples of ints) Edges (i,j) between labeling functions
            indicating a conditional dependency between LF i and LF j
    """

    def __init__(self, m, k, edges, higher_order_cliques=True):
        self.m = m
        self.k = k
        self.edges = edges
        self.G = self._get_junction_tree(range(self.m), self.edges)
        self.c_data, self.d = self._get_clique_data(
            higher_order_cliques=higher_order_cliques
        )

    def _get_junction_tree(self, nodes, edges):
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
        # Form the original graph G1
        G1 = nx.Graph()
        G1.add_nodes_from(nodes)
        G1.add_edges_from(edges)

        # Check if graph is chordal
        # TODO: Add step to triangulate graph if not
        if not nx.is_chordal(G1):
            raise NotImplementedError("Graph triangulation not implemented.")

        # Create maximal clique graph G2
        # Each node is a maximal clique C_i
        # Let w = |C_i \cap C_j|; C_i, C_j have an edge with weight w if w > 0
        G2 = nx.Graph()
        for i, c in enumerate(nx.chordal_graph_cliques(G1)):
            G2.add_node(i, members=c)
        for i in G2.nodes:
            for j in G2.nodes:
                S = G2.node[i]["members"].intersection(G2.node[j]["members"])
                w = len(S)
                if w > 0:
                    G2.add_edge(i, j, weight=w, members=S)

        # Return a minimum spanning tree of G2
        return nx.minimum_spanning_tree(G2)

    def draw(self):
        labels = dict(
            [
                (i, f"{set(self.G.node[i]['members']) - set([self.m+1])}")
                for i in self.G.nodes()
            ]
        )
        nx.draw(self.G, with_labels=True, node_size=500, labels=labels)

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
