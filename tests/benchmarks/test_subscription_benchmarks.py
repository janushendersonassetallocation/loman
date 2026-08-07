"""Benchmarks for the notification path added by the UI extra.

Run with ``make benchmark``. These measure rather than assert: what they give is
the shape of the cost, so a change that is merely slower rather than
asymptotically worse is still visible in the report.

The thresholds that *fail* live in ``test_subscription_cost`` alongside this,
and in ``test_subscription_scaling`` in the stress tier. The asymptotic guard is
``test_computation_graph_construction_stays_linear`` in ``test_computeengine``,
which runs in the ordinary test run.

The regression in question: an early version of ``_notifies_subscribers``
snapshotted the whole node set on every structural mutation whether or not
anything was subscribed, which made building a graph quadratic.
"""

from loman import Computation
from loman.ui.viewmodel import node_states

GRAPH_SIZE = 400


def build_graph(size: int = GRAPH_SIZE) -> Computation:
    """Build a fan-out graph of ``size`` calculated nodes over one input."""
    comp = Computation()
    comp.add_node("x", value=1)
    for i in range(size):
        comp.add_node(f"n{i}", lambda x: x + 1, kwds={"x": "x"})
    return comp


def test_build_graph_without_subscribers(benchmark):
    """Graph construction on the path every Loman user takes."""
    benchmark(build_graph)


def test_build_graph_with_subscriber(benchmark):
    """The same construction with a widget-like subscriber attached.

    Should track the unsubscribed case closely. A growing gap means the
    per-mutation bookkeeping has started doing real work again.
    """

    def build_with_subscriber():
        """Attach a subscriber before building, as the widget would."""
        comp = Computation()
        comp.subscribe(lambda _event: None)
        comp.add_node("x", value=1)
        for i in range(GRAPH_SIZE):
            comp.add_node(f"n{i}", lambda x: x + 1, kwds={"x": "x"})
        return comp

    benchmark(build_with_subscriber)


def test_compute_all_with_subscriber(benchmark):
    """One batched event for a whole computation, not one per node."""

    def setup():
        """Provide a fresh stale graph for each round."""
        comp = build_graph()
        comp.subscribe(lambda _event: None)
        comp.insert("x", 2)
        return (comp,), {}

    benchmark.pedantic(lambda comp: comp.compute_all(), setup=setup, rounds=20)


def test_node_states_payload(benchmark):
    """The repaint payload, rebuilt on every state-only change.

    This runs far more often than a relayout does, so it is the one piece of the
    widget's update path that has to stay cheap.
    """
    view = build_graph().draw(collapse_all=False)

    benchmark(node_states, view)


def test_graph_view_layout(benchmark):
    """A full relayout, including the ``dot`` subprocess.

    The expensive path, and the reason state changes repaint in place instead.
    """
    comp = build_graph(size=100)

    benchmark(lambda: comp.draw(collapse_all=False).svg())
