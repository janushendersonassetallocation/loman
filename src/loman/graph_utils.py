"""Graph utility functions for computation graph operations."""

import functools
from typing import Any

import networkx as nx

from loman.exception import LoopDetectedError
from loman.util import apply_n


def contract_node_one(g: nx.DiGraph, n: Any) -> None:
    """Remove a node from graph and connect its predecessors to its successors."""
    for p in g.predecessors(n):
        for s in g.successors(n):
            g.add_edge(p, s)
    g.remove_node(n)


def contract_node(g: nx.DiGraph, ns: Any) -> None:
    """Remove multiple nodes from graph and connect their predecessors to successors."""
    apply_n(functools.partial(contract_node_one, g), ns)


def topological_sort(g: nx.DiGraph) -> list[Any]:
    """Performs a topological sort on a directed acyclic graph (DAG).

    This function attempts to compute the topological order of the nodes in
    the given graph `g`. If the graph contains a cycle, it raises a
    `LoopDetectedError` with details about the detected cycle, making it
    informative for debugging purposes.

    Parameters:
    g : networkx.DiGraph
        A directed graph to be sorted. Must be provided as an instance of
        `networkx.DiGraph`. The function assumes the graph is acyclic unless
        a cycle is detected.

    Returns:
    list
        A list of nodes in topologically sorted order, if the graph has no
        cycles.

    Raises:
    LoopDetectedError
        If the graph contains a cycle, a `LoopDetectedError` is raised with
        information about the detected cycle if available. The detected cycle
        is presented as a list of directed edges forming the cycle.

    NetworkXUnfeasible
        If topological sorting fails due to reasons other than cyclic
        dependencies in the graph.
    """
    try:
        return list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible as e:
        cycle_lst = None
        if g is not None:
            try:
                cycle_lst = nx.find_cycle(g)
            except nx.NetworkXNoCycle:  # pragma: no cover
                # there must non-cycle reason NetworkXUnfeasible, leave as is
                raise e from None
        args: list[str] = []
        if cycle_lst:
            lst = [f"{n_src}->{n_tgt}" for n_src, n_tgt in cycle_lst]
            args = [f"DAG cycle: {', '.join(lst)}"]
        raise LoopDetectedError(*args) from e
