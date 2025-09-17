"""Graph utility functions for computation graph operations."""

import functools

from loman.util import apply_n


def contract_node_one(g, n):
    """Remove a node from graph and connect its predecessors to its successors."""
    for p in g.predecessors(n):
        for s in g.successors(n):
            g.add_edge(p, s)
    g.remove_node(n)


def contract_node(g, ns):
    """Remove multiple nodes from graph and connect their predecessors to successors."""
    apply_n(functools.partial(contract_node_one, g), ns)
