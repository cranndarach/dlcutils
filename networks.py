#!/usr/bin/env python3

"""
Network science utilities.
"""

import random as rd
import itertools as it


# There were originally classes for Node and Edge, but they became
# difficult to handle, given that their existence was so closely tied
# to their parent network. So, I decided to handle them more abstractly.
# Most of the network representation is handled in Network.edges, which
# stores a dict of dicts. Top-level keys refer to nodes, and
# Network.edges[node] is a dict of (neighbor, weight) pairs. There are
# also two convenience attributes: Network.node_list, which is really
# just a list of ints from 1 to N, representing each node, and
# Network.edge_list, which is a list of (from, to) pairs, representing
# the links in the network.
class Network:
    """
    A collection of nodes and the links between them. In its current
    state, this is a weighted, directed, fully-connected graph.

    Attributes:
        n (int): number of nodes
        node_list (list of ints): list of all of the nodes in the network.
        edge_list (list of (from, to) pairs): list of all edges in the
            network.
        edges (dict of dicts): dict of nodes and their edges, where
            edges[node] is a dict of (neighbor, weight) entries.
        focus_node (int): the current "focus node."
        next_focus (int): the next focus node, randomly selected from
            focus_node's neighbors, weighted by edge weights.
        focus_edge ((from, to) pair): the edge linking (focus_node,
            next_focus)
    """
    def __init__(self, n):
        self.n = n
        # Will be overwritten, not appended to, in self.make_nodes().
        self.node_list = []
        # Will be overwritten in self.make_edges().
        self.edge_list = []
        self.edges = {}
        self._make_nodes()
        self._make_edges()
        self.focus_node = None
        self.next_focus = None
        self.focus_edge = None

    def _make_nodes(self):
        """
        Creates n nodes for the network. Mostly intended to be called
        by __init__().
        """
        indices = range(1, self.n+1)
        self.node_list = list(indices)

    def _make_edges(self):
        """
        Fills out self.edge_list with pairs of (from, to) links, and
        fills self.edges with dicts for each node, with (neighbor,
        weight) pairs.
        """
        # This feels really hacky, but having individual classes and
        # objects for each level was not working.
        self.edge_list = list(it.product(self.node_list, self.node_list))
        for node in self.node_list:
            self.edges[node] = dict([(nbr, 1/self.n) for nbr in
                                     self.node_list])

    def set_focus(self, node=None):
        """
        Sets the "focus node." If `node` is not specified, one will be
        selected randomly from the nodes in the network. If `node` is
        "next," focus_node will be set to the current next_focus.
        """
        if not node:
            self.focus_node = rd.choice(self.node_list)
        elif node == "next":
            self.focus_node = self.next_focus
        elif node not in self.node_list:
            raise ValueError("The specified node is not in the network.")
        else:
            self.focus_node = node
        # And clear out related variables.
        self.next_focus = None
        self.focus_edge = None

    def select_focus_edge(self):
        """
        Randomly selects an edge from focus_node to an out-neighbor,
        weighted by the weights of focus_node's out-connections.
        """
        ego = self.edges[self.focus_node]
        nbrs = list(ego.keys())
        weights = list(ego.values())
        self.next_focus = rd.choices(nbrs, weights)[0]
        self.focus_edge = (self.focus_node, self.next_focus)

    def update_focus_weights(self):
        """
        Updates the weights of all out-connections from focus_node.
        Adds 1/N to the weight of (focus_node, next_focus). Then
        divides the weights of all of focus_node's out-connections by
        the sum of all of focus_node's out-connections' weights.
        """
        ego = self.edges[self.focus_node]
        # Increase weight by 1/N.
        ego[self.next_focus] += 1 / self.n
        # Calculate the sum of the focus nodes' weights.
        total_weights = sum(ego.values())
        # Then renormalize by dividing the weight of each edge by the
        # sum of all weights emanating from focus_node.
        for neighbor in ego.keys():
            ego[neighbor] /= total_weights

    def step(self):
        """
        Selects a focus edge, sets focus to the pointed node, and
        returns that node.
        """
        self.select_focus_edge()
        self.set_focus("next")
        return self.focus_node


class BANetwork:
    """
    Barabasi & Albert network.
    """
    def __init__(self, n, k):
        """
        n is the desired number of nodes in the final network.
        k is the number of edges to allot to each added node.
        """
        self.n = n
        self.k = k
        self.nodes = list(range(1, k+1))
        self.size = k
        self.degrees = dict(zip(self.nodes, [0]*k))
        self.edges = []
        self.add_first_node()
        self.make_network()

    def make_edges(self, node, targets):
        for neighbor in targets:
            self.edges.append((node, neighbor))
            self.degrees[node] += 1
            self.degrees[neighbor] += 1

    def add_first_node(self):
        if self.size != self.k:
            raise ValueError("The network is already initialized.")
        self.size += 1
        targets = self.nodes.copy()
        # Nodes will just be an integer representing its index.
        node = self.size
        self.nodes.append(node)
        self.degrees[node] = 0
        self.make_edges(node, targets)

    def add_node(self):
        if self.size == self.k:
            raise ValueError("Network not yet initialized.")
        self.size += 1
        node = self.size
        self.nodes.append(node)
        self.degrees[node] = 0
        # Wanted to use rd.choices, but that samples with replacement.
        # I guess instead it should sample from a list with each node
        # repeated as many times as its degree.
        # candidates, weights = [*zip(self.degrees.items())]
        # targets = rd.choices(candidates, weights, k=self.k)
        candidates = list(it.chain.from_iterable(self.edges))
        targets = rd.sample(candidates, self.k)
        self.make_edges(node, targets)

    def make_network(self):
        while self.size < self.n:
            self.add_node()


class WSNetwork:
    """
    Watts-Strogatz Small-World network.
    """
    def __init__(self, n, k, b):
        if k % 2 != 0:
            raise ValueError("k must be an even integer.")
        if b < 0 or b > 1:
            raise ValueError("b must be between 0 and 1 (inclusive).")
        if n <= k:
            raise ValueError("n must be greater than k.")
        self.n = n
        self.k = k
        self.b = b
        self.nodes = list(range(1, n+1))
        self.edges = []
        self.neighbors = {}
        self.make_ring()
        self.rewire()

    def make_ring(self):
        per_side = self.k/2
        for node in self.nodes:
            neighbors = []
            if node <= per_side:
                padding = per_side - node + 1
                neighbors.append(self.nodes[-padding:])
                neighbors.append(self.nodes[:node-1])
            elif node >= len(self.nodes) - per_side:
                # Haven't thought through whether that's the right condition.
                pass
            else:
                pass
