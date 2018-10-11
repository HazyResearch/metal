from collections import defaultdict
from itertools import combinations, product

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
            "theta_range_acc": (0.1, 1),
            "theta_range_edge": (0.1, 1),
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

        # Generate O, mu, Sigma, Sigma_inv
        # Y = (
        #     1
        # )  # Note: we pick an arbitrary Y here, since assuming doesn't matter
        # self.O = self._generate_O_Y(Y=Y)
        # self.mu = self._generate_mu_Y(Y=Y)
        # self.sig, self.sig_inv = self._generate_sigma(self.O, self.mu)

        # # Generate the true labels self.Y and label matrix self.L
        # self._generate_label_matrix()

    #HACK: new data generator
    # def _generate_params(self, param_ranges, rank_one_model=True):
    #     """This function generates the parameters of the data generating model
    #     Note that along with the potential functions of the SPA algorithm, this
    #     essentially defines our model. This model is the most general form,
    #     where each marginal conditional probability for each clique C,
    #     P(\lf_C | Y), is generated randomly.
    #     """
    #     theta = defaultdict(float)
    #     acc_range = param_ranges["theta_range_acc"]
    #     acc_min, acc_max = min(acc_range), max(acc_range)
    #     edge_range = param_ranges["theta_range_edge"]
    #     edge_min, edge_max = min(edge_range), max(edge_range)

    #     # Unary clique factors
    #     # TODO: Set class balance here!

    #     # Pairwise (edge) factors
    #     for edge in self.jt.deps_graph.G.edges():
    #         (i, j) = sorted(edge)

    #         # Pairwise accuracy factors (\lf_i, Y); note Y is index j = self.m
    #         if j == self.m:
    #             for vi, vj in product(
    #                 range(self.k0, self.k + 1), range(1, self.k + 1)
    #             ):
    #                 # Use a random positive theta value for correct, negative
    #                 # for incorrect...
    #                 acc = (acc_max - acc_min) * random() + acc_min
    #                 theta[((i, j), (vi, vj))] = acc if vi == vj else -acc

    #         # Pairwise correlation factors (\lf_i, \lf_j)
    #         else:
    #             for vi, vj in product(range(self.k0, self.k + 1), repeat=2):
    #                 theta[((i, j), vi, vj)] = (
    #                     edge_max - edge_min
    #                 ) * random() + edge_min

    #         # Populate the other ordering too
    #         theta[((j, i), (vj, vi))] = theta[((i, j), (vi, vj))]
    #     return theta
    
    def _generate_params(self, param_ranges):
        """This function generates the parameters of the data generating model

        Note that along with the potential functions of the SPA algorithm, this
        essentially defines our model. This model is the most general form,
        where each marginal conditional probability for each clique C,
        P(\lf_C | Y), is generated randomly.
        """
        theta = defaultdict(float)

        # Unary clique factors
        # TODO: Set class balance here!

        # Binary clique factors
        for (i, j) in self.jt.deps_graph.G.edges():

            # Separate parameters for (\lf_i, Y) factors vs. (\lf_i, \lf_j)
            if i == self.m or j == self.m:
                theta_range = param_ranges["theta_range_acc"]
            else:
                theta_range = param_ranges["theta_range_edge"]
            t_min, t_max = min(theta_range), max(theta_range)

            for vals in product(range(self.k0, self.k + 1), repeat=2):
                # Y does not have abstain votes
                if (i == self.m and vals[0] == 0) or (
                    j == self.m and vals[1] == 0
                ):
                    continue
                theta[((i, j), vals)] = (t_max - t_min) * random() + t_min
                theta[((j, i), vals[::-1])] = theta[((i, j), vals)]
        return theta

    def _exp_model(self, vals):
        """Compute the exponential model for a set of variables and values
        assuming an Ising model (i.e. only edge or node factors)

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
                            if i in self.jt.get_members(ci)
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
                                if i in self.jt.get_members((ci, cj))
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

    #HACK: from new data generator
    def _get_joint_prob(self, vals_i, vals_j):
        # Note: Need to check that they don't conflict!
        conflict = False
        for k in set(vals_i.keys()).intersection(vals_j.keys()):
            if vals_i[k] != vals_j[k]:
                conflict = True
                break
        return self.P_marginal({**vals_i, **vals_j}) if not conflict else 0.0

    def get_sigma_O(self):
        d = self.jt.O_d
        sigma_O = np.zeros((d, d))
        for ((i, vi), (j, vj)) in product(self.jt.iter_observed(), repeat=2):
            sigma_O[i, j] = self.P_marginal({**vi, **vj}) - self.P_marginal(
                vi
            ) * self.P_marginal(vj)
        return sigma_O

    #HACK: from new data generator
    def get_mu(self): #new get_mu function
        mu = np.zeros((self.jt.O_d, self.jt.H_d))
        for i, vi in self.jt.iter_observed():
            for j, vj in self.jt.iter_hidden():
                mu[i, j] = self._get_joint_prob(vi, vj)
        return mu
    
    # def get_mu(self):
    #     d = self.jt.O_d
    #     mu = np.zeros(d)
    #     for i, vi in self.jt.iter_observed(): #HACK: removed add_Y since using the new junction tree methods 
    #         mu[i] = self.P_marginal(vi)
    #     return mu
