from collections import defaultdict
from itertools import chain, combinations, product

import networkx as nx
import numpy as np
from numpy.random import choice, random

from metal.label_model.graph_utils import DependenciesGraph, JunctionTree


# Dependencies Graphs
class TreeDependencies(DependenciesGraph):
    """Generate a random tree-structured dependency graph based on a
    specified edge probability.
    """

    def __init__(self, m, edge_prob=1.0):
        super().__init__(m)
        for i in range(1, m):
            if random() < edge_prob:
                self.G.add_edge(i, choice(i))


class ChainDependencies(DependenciesGraph):
    """Generate a chain-structured dependency graph."""

    def __init__(self, m, edge_prob=1.0):
        super().__init__(m)
        for i in range(1, m):
            if random() < edge_prob:
                p_i = i - 1
                self.G.add_edge(i, p_i)


class ClusterDependencies(DependenciesGraph):
    """Generate a cluster-structured dependency graph."""

    def __init__(self, m, n_clusters, edge_prob=1.0):
        super().__init__(m)
        self.clusters = defaultdict(set)
        for i in range(m):
            if random() < edge_prob:
                c = choice(n_clusters)
                for j in self.clusters[c]:
                    self.G.add_edge(i, j)
                self.clusters[c].add(i)


# DATA GENERATORS
class DataGenerator(object):
    """Generates a synthetic dataset

    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        abstains: (bool) Whether to include 0 labels as abstains
        class_balance: (np.array) each class's percentage of the population
        deps_graph: (DependenciesGraph) A DependenciesGraph object
            specifying the dependencies structure of the sources
        param_ranges: (dict) A dictionary of ranges to draw the model parameters
            from:
            - theta_range_acc: (tuple) The min and max possible values for the
                class conditional accuracy for each labeling source
            - theta_range_edge: The min and max possible values for the strength
                of correlation between correlated sources
        higher_order_cliques: (bool) Set here whether to track the higher-order
            cliques or else just the unary cliques throughout. We set this once
            globally for simplicity and to avoid confusions / mismatches.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains, of {1,...,k} if abstains=False
    """

    def __init__(
        self,
        n,
        m,
        k=2,
        abstains=True,
        class_balance="random",
        deps_graph=None,
        param_ranges={
            "theta_acc_range": (-0.25, 1),
            "theta_edge_range": (-1, 1),
        },
        higher_order_cliques=True,
        **kwargs,
    ):
        self.n = n
        self.m = m
        self.k = k
        self.abstains = abstains
        self.k0 = 0 if abstains else 1

        # Form DependenciesGraph and JunctionTree
        self.deps_graph = (
            DependenciesGraph(m) if deps_graph is None else deps_graph
        )
        self.jt = JunctionTree(
            self.m,
            self.k,
            abstains=self.abstains,
            deps_graph=self.deps_graph,
            higher_order_cliques=higher_order_cliques,
        )

        # Generate class-conditional LF & edge parameters, stored in self.theta
        self.theta = self._generate_params(param_ranges)

        # Generate class balance self.p
        if class_balance is None:
            self.p = np.full(k, 1 / k)
        elif class_balance == "random":
            self.p = np.random.random(k)
            self.p /= self.p.sum()
        else:
            self.p = class_balance

        # Cache for sum-product algorithm
        self.msg_cache = {}
        self.p_marginal_cache = {}

    def _generate_params(self, param_ranges, rank_one_model=True):
        """This function generates the parameters of the data generating model

        Note that along with the potential functions of the SPA algorithm, this
        essentially defines our model. This model is the most general form,
        where each marginal conditional probability for each clique C,
        P(\lf_C | Y), is generated randomly.
        """
        theta = defaultdict(float)
        acc_range = param_ranges["theta_acc_range"]
        acc_min, acc_max = min(acc_range), max(acc_range)
        edge_range = param_ranges["theta_edge_range"]
        edge_min, edge_max = min(edge_range), max(edge_range)

        # Unary clique factors
        # TODO: Set class balance here!

        # Pairwise (edge) factors
        for edge in self.jt.deps_graph.G.edges():
            (i, j) = sorted(edge)

            # Pairwise accuracy factors (\lf_i, Y); note Y is index j = self.m
            if j == self.m:
                for vi, vj in product(
                    range(self.k0, self.k + 1), range(1, self.k + 1)
                ):
                    # Use a random positive theta value for correct, negative
                    # for incorrect...
                    acc = (acc_max - acc_min) * random() + acc_min
                    theta[((i, j), (vi, vj))] = acc if vi == vj else -acc

            # Pairwise correlation factors (\lf_i, \lf_j)
            else:
                for vi, vj in product(range(self.k0, self.k + 1), repeat=2):
                    theta[((i, j), vi, vj)] = (
                        edge_max - edge_min
                    ) * random() + edge_min

            # Populate the other ordering too
            theta[((j, i), (vj, vi))] = theta[((i, j), (vi, vj))]
        return theta

    def _exp_model(self, vals):
        """Compute the unnormalized exponential model for a set of variables and
        values assuming an Ising model (i.e. only edge or node factors)

        Args:
            - vals: (dict) A dictionary of (LF index, value) entries.
        """
        x = 0.0

        # Node factors
        for i, val in vals.items():
            x += self.theta[(i, val)]

        # Edge factors
        for (i, val_i), (j, val_j) in combinations(vals.items(), 2):
            x += self.theta[((i, j), (val_i, val_j))]
        return np.exp(x)

    def P_marginal(self, query, condition_on={}, clique_id=None):
        """Compute P(query|condition_on) using the sum-product algorithm over
        the junction tree `self.jt`"""

        # Make sure query and condition_on are disjoint!
        for k in condition_on.keys():
            if k in query:
                del query[k]

        # Check the cache first, keyed by query (projected onto members of
        # clique i), i, j
        cache_key = (tuple(query.items()), tuple(condition_on.items()))
        if cache_key in self.p_marginal_cache:
            return self.p_marginal_cache[cache_key]

        # Check inputs to make sure refers to valid variable ids and values
        for i, vi in {**query, **condition_on}.items():
            if (
                i < 0
                or i > self.m
                or vi < self.k0
                or vi > self.k
                or (i == self.m and vi == 0)
            ):
                raise ValueError(f"Error with input {{{i}:{vi}}}")

        if clique_id is not None:
            cids = [clique_id]

        # Route P(Y) queries correctly here
        elif len(query) == 1 and self.m in query:
            cids = [0]

        # Else get the set of containing maximal cliques
        else:

            # Get the set of cliques containing the (non-Y) query variables
            _cids = [
                ci
                for ci in self.jt.G.nodes()
                if len(
                    self.jt.get_members(ci).intersection(query.keys())
                    - {self.m}
                )
                > 0
            ]

            # Extend so that the set of nodes constitutes a connected subgraph
            # Note: This is just a quick heuristic way of doing this
            cids = set(_cids)
            for (ci, cj) in combinations(_cids, 2):
                cids = cids.union(nx.shortest_path(self.jt.G, ci, cj))

        if len(cids) > 1:
            p_marginal = 0.0

            # Compute recursively by multiplying clique marginals, dividing by
            # separator set marginals, and marginalizing over non-query vars
            nq_ids = set()
            for ci in cids:
                nq_ids = nq_ids.union(
                    self.jt.get_members(ci)
                    - set(query.keys())
                    - set(condition_on.keys())
                )
            for nq in self.jt.iter_vals(nq_ids):
                q = {**query, **nq}
                p = 1.0

                # Cliques in subgraph of junction tree
                for ci in cids:
                    qi = dict(
                        [
                            (i, q[i])
                            for i in range(self.m + 1)
                            if i
                            in (
                                self.jt.get_members(ci)
                                - set(condition_on.keys())
                            )
                        ]
                    )
                    p *= self.P_marginal(
                        qi, condition_on=condition_on, clique_id=ci
                    )

                # Separator sets in subgraph of junction tree
                for (ci, cj) in combinations(cids, 2):
                    if (ci, cj) in self.jt.G.edges():
                        qi = dict(
                            [
                                (i, q[i])
                                for i in range(self.m + 1)
                                if i
                                in (
                                    self.jt.get_members((ci, cj))
                                    - set(condition_on.keys())
                                )
                            ]
                        )
                        p /= self.P_marginal(
                            qi, condition_on=condition_on, clique_id=ci
                        )
                p_marginal += p
            self.p_marginal_cache[cache_key] = p_marginal
            return p_marginal

        else:
            ci = list(cids)[0]

            # Run the sum-product algorithm recursively
            p = self._message({**query, **condition_on}, ci)

            # Return normalized probability
            Z = sum(
                self._message({**q, **condition_on}, ci)
                for q in self.jt.iter_vals(query.keys())
            )
            p_marginal = p / Z
            self.p_marginal_cache[cache_key] = p_marginal
            return p_marginal

    def _message(self, query, i, j=None):
        """Computes the sum-product algorithm message from junction tree clique
        i --> j"""
        clique_members = self.jt.G.node[i]["members"]

        # Check the cache first, keyed by query (projected onto members of
        # clique i), i, j
        cache_key = (
            i,
            j,
            tuple([(ti, v) for ti, v in query.items() if ti in clique_members]),
        )
        if cache_key in self.msg_cache:
            return self.msg_cache[cache_key]

        # Sum over the values of clique i not in the target set
        # Note that the target set will include the separator set values, so
        # these will not be summed over (as desired)
        msg = 0
        for vals_dict in self.jt.iter_vals(clique_members, fixed=query):

            # Compute the local message for current node i
            msg_v = self._exp_model(vals_dict)

            # Recursively compute the messages from children
            children = set(self.jt.G.neighbors(i))
            if j is not None:
                children -= {j}
            for c in children:
                msg_v *= self._message({**vals_dict, **query}, c, i)
            msg += msg_v

        # Cache message and return
        self.msg_cache[cache_key] = msg
        return msg

    def P_marginal_brute_force(self, query, condition_on={}):
        """Compute P(query|condition_on)"""

        # Check inputs to make sure refers to valid variable ids and values
        for i, vi in {**query, **condition_on}.items():
            if (
                i < 0
                or i > self.m
                or vi < self.k0
                or vi > self.k
                or (i == self.m and vi == 0)
            ):
                raise ValueError(f"Error with input {{{i}:{vi}}}")

        # The numerator has the target and condition_on variables fixed, and
        # sums over all the remaining vars
        p = sum(
            self._exp_model(q)
            for q in self.jt.iter_vals(
                range(self.m + 1), fixed={**query, **condition_on}
            )
        )

        # The demoninator has only condition_on variables fixed, and sums over
        # all the remaining vars
        Z = sum(
            self._exp_model(q)
            for q in self.jt.iter_vals(
                range(self.m + 1), fixed={**condition_on}
            )
        )
        return p / Z

    def get_class_balance(self):
        """Generate the vector of elements P(Y=i) for i in {1,...,k}"""
        return np.array(
            [self.P_marginal({self.m: i}) for i in range(1, self.k + 1)]
        )

    def _get_joint_prob(self, vals_i, vals_j):
        # Note: Need to check that they don't conflict!
        conflict = False
        for k in set(vals_i.keys()).intersection(vals_j.keys()):
            if vals_i[k] != vals_j[k]:
                conflict = True
                break
        return self.P_marginal({**vals_i, **vals_j}) if not conflict else 0.0

    def _get_covariance(self, vals_i, vals_j):
        """Get the covariance of two clique, value pairs as dictionaries vals_i,
        vals_j."""
        sigma = self._get_joint_prob(vals_i, vals_j)
        sigma -= self.P_marginal(vals_i) * self.P_marginal(vals_j)
        return sigma

    def get_sigma_O(self):
        sigma_O = np.zeros((self.jt.O_d, self.jt.O_d))
        for ((i, vi), (j, vj)) in product(self.jt.iter_observed(), repeat=2):
            sigma_O[i, j] = self._get_covariance(vi, vj)
        return sigma_O

    def get_sigma_H(self):
        sigma_H = np.zeros((self.jt.H_d, self.jt.H_d))
        for ((i, vi), (j, vj)) in product(self.jt.iter_hidden(), repeat=2):
            sigma_H[i, j] = self._get_covariance(vi, vj)
        return sigma_H

    def get_sigma_OH(self):
        sigma_OH = np.zeros((self.jt.O_d, self.jt.H_d))
        for ((i, vi), (j, vj)) in product(
            self.jt.iter_observed(), self.jt.iter_hidden()
        ):
            sigma_OH[i, j] = self._get_covariance(vi, vj)
        return sigma_OH

    def get_mu(self):
        mu = np.zeros((self.jt.O_d, self.jt.H_d))
        for i, vi in self.jt.iter_observed():
            for j, vj in self.jt.iter_hidden():
                mu[i, j] = self._get_joint_prob(vi, vj)
        return mu

    def generate_label_matrix(self, n):
        """Generate an m x n label matrix."""
        L = np.zeros((n, self.m))
        Y = np.zeros(n)
        Y_range = range(1, self.k + 1)
        p_Y = np.array([self.P_marginal({self.m: y}) for y in Y_range])
        for i in range(n):
            y = choice(Y_range, p=p_Y)
            Y[i] = y

            # Traverse the junction tree DFS starting at node 0
            for p, c in chain([(-1, 0)], nx.dfs_edges(self.jt.G, source=0)):

                # The fixed values are the separator set members, which will
                # always contain y; else y if at node 0
                S = self.jt.G.edges[p, c]["members"] if p > 0 else set([self.m])
                fixed = dict(
                    [(j, L[i, j]) if j < self.m else (self.m, y) for j in S]
                )

                # Start with sources in node 0 of the junction tree
                vals = list(
                    self.jt.iter_vals(self.jt.get_members(c), fixed=fixed)
                )
                p_L = [self.P_marginal(v, condition_on=fixed) for v in vals]

                # Pick one of the value sets randomly and put into L
                for j, val in choice(vals, p=p_L).items():
                    L[i, j] = val
        return L, Y


class SimpleDataGenerator(DataGenerator):
    """Generates a synthetic dataset

    In this simple model (which corresponds to a rank-one problem if singleton
    separator sets), we tie weights such that
        P(\lf_C | Y) = P(\lf_C' | Y)
    if \lf_C, \lf_C' only differ on which incorrect values are chosen.

    Args:
        n: (int) The number of data points
        m: (int) The number of labeling sources
        k: (int) The cardinality of the classification task
        abstains: (bool) Whether to include 0 labels as abstains
        class_balance: (np.array) each class's percentage of the population
        deps_graph: (DependenciesGraph) A DependenciesGraph object
            specifying the dependencies structure of the sources
        param_ranges: (dict) A dictionary of ranges to draw the model parameters
            from:
            - theta_range_acc: (tuple) The min and max possible values for the
                class conditional accuracy for each labeling source
            - theta_range_edge: The min and max possible values for the strength
                of correlation between correlated sources
        higher_order_cliques: (bool) Set here whether to track the higher-order
            cliques or else just the unary cliques throughout. We set this once
            globally for simplicity and to avoid confusions / mismatches.

    Note that k = the # of true classes; thus source labels are in {0,1,...,k}
    because they include abstains, of {1,...,k} if abstains=False
    """

    def _generate_params(self, param_ranges):
        """This function generates the parameters of the data generating model

        Note that along with the potential functions of the SPA algorithm, this
        essentially defines our model. This model is the most general form,
        where each marginal conditional probability for each clique C,
        P(\lf_C | Y), is generated randomly.
        """
        theta = defaultdict(float)
        theta_r1 = {}

        # Binary clique factors
        for idxs in self.jt.deps_graph.G.edges():
            (i, j) = sorted(idxs)

            # Binary cliques (i,j) = (\lf_i, Y)
            if j == self.m:
                for vals in product(
                    range(self.k0, self.k + 1), range(1, self.k + 1)
                ):
                    vi, vj = vals
                    key = (i, j, vj)

                    # For each (\lf_i, Y) pair, store the single parameter
                    if key not in theta_r1:
                        theta_range = param_ranges["theta_range_acc"]
                        t_min, t_max = min(theta_range), max(theta_range)
                        theta_r1[key] = (t_max - t_min) * random() + t_min

                    # Store for the specific value set, either correct or not
                    if vi == vj:
                        theta[((i, j), vals)] = theta_r1[key]
                    else:
                        theta[((i, j), vals)] = (1 - theta_r1[key]) / (
                            self.k - 1
                        )
                    theta[((j, i), vals[::-1])] = theta[((i, j), vals)]

            # Binary cliques (i,j) = (\lf_i, \lf_j)
            else:
                for vals in product(range(self.k0, self.k + 1), repeat=2):
                    theta_range = param_ranges["theta_range_edge"]
                    t_min, t_max = min(theta_range), max(theta_range)
                    theta[((i, j), vals)] = (t_max - t_min) * random() + t_min
                    theta[((j, i), vals[::-1])] = theta[((i, j), vals)]
        return theta
