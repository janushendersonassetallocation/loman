"""Graph utility functions for computation graph operations."""

import functools

from loman.util import apply_n


def contract_node_one(g, n):
    for p in g.predecessors(n):
        for s in g.successors(n):
            g.add_edge(p, s)
    g.remove_node(n)


def contract_node(g, ns):
    apply_n(functools.partial(contract_node_one, g), ns)
