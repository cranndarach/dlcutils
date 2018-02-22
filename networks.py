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


class BANetwork(Network):
    """
    Extension of the basic "Network" class from assignment 5. Trying to
    make a Barabasi & Albern network. Might actually change it to not
    be an extension.
    """
    def __init__(self, n, k):
        super().__init__(n)
        self.k = k
        self.nodes = list(range(1, k+1))
        self.degrees = dict(zip(self.nodes, [0]*k))
