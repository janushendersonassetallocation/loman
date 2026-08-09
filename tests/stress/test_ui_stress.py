"""Stress tests for the subscription layer and the notebook widget.

Run with ``make stress``. These are about behaviour that only misbehaves at
volume or over many repetitions: unbounded growth, relayout storms, and state
that drifts as operations are repeated. They are excluded from the ordinary test
run because they are slow, not because they are optional.
"""

import gc

import pytest

from loman import Computation, States
from loman.nodekey import to_nodekey
from loman.ui.widget import _REQUEST_HISTORY
from tests.conftest import BasicFourNodeComputation

#: Large enough to make a relayout obviously expensive, small enough that the
#: one layout these tests do take stays a couple of seconds.
LARGE_GRAPH = 300


def build_large_graph(size: int = LARGE_GRAPH) -> Computation:
    """Build a wide graph of calculated nodes over a single input."""
    comp = Computation()
    comp.add_node("x", value=1)
    for i in range(size):
        comp.add_node(f"n{i}", lambda x: x + 1, kwds={"x": "x"})
    return comp


@pytest.mark.stress
def test_repeated_state_changes_never_relayout():
    """A long editing session must not re-run Graphviz on every change.

    This is the core scaling claim of the widget: state changes repaint the
    existing picture, and only structural changes cost a layout.
    """
    comp = build_large_graph()
    widget = comp.widget(collapse_all=False)
    try:
        original_svg = widget.graph_svg
        assert original_svg

        for i in range(200):
            comp.insert("x", i)
            comp.compute_all()

        assert widget.graph_svg == original_svg
        assert widget.revision == comp.revision
        assert len(widget.node_states) == LARGE_GRAPH + 1
        assert set(widget.node_states.values()) == {States.UPTODATE.name}
    finally:
        widget.close()


@pytest.mark.stress
def test_subscription_churn_does_not_leak():
    """Subscribing and unsubscribing repeatedly must not accumulate anything."""
    comp = Computation()
    comp.add_node("a", value=1)

    for i in range(5_000):
        unsubscribe = comp.subscribe(lambda _event: None)
        comp.insert("a", i)
        unsubscribe()
        unsubscribe()

    assert comp._subscriptions == []
    assert comp._pending_changed_nodes == set()


@pytest.mark.stress
def test_collected_subscribers_are_pruned_over_time():
    """Weakly held subscribers must not pile up as dead entries."""

    class Watcher:
        """A short-lived subscriber, like a widget from a re-run cell."""

        def on_event(self, _event):
            """Ignore the event."""

    comp = Computation()
    comp.add_node("a", value=1)
    keepalive = Watcher()
    comp.subscribe(keepalive.on_event)

    for i in range(500):
        comp.subscribe(Watcher().on_event)
        comp.insert("a", i)

    gc.collect()
    comp.insert("a", -1)

    assert [s.resolve() for s in comp._subscriptions] == [keepalive.on_event]


@pytest.mark.stress
def test_rapid_expand_and_collapse_stays_consistent():
    """Drilling in and out repeatedly must not drift or accumulate state."""
    comp = Computation()
    comp.add_block("foo", BasicFourNodeComputation(), keep_values=False, links={"a": "input_foo"})
    comp.add_node("input_foo", value=7)
    widget = comp.widget()
    try:
        foo = to_nodekey("foo")
        collapsed_svg = widget.graph_svg

        for cycle in range(25):
            widget.toggle_request = {"id": widget._view.node_index_map[foo], "request_id": f"open-{cycle}"}
            assert widget.status == "Opened foo"
            assert foo not in widget._view.composite_nodes

            widget.toggle_request = {"collapse_all": True, "request_id": f"close-{cycle}"}
            assert widget.status == "Collapsed all blocks"
            assert foo in widget._view.composite_nodes

        assert widget.graph_svg == collapsed_svg
        assert widget._expanded == set()
        assert len(widget._seen_requests) <= _REQUEST_HISTORY
    finally:
        widget.close()


@pytest.mark.stress
def test_request_history_stays_bounded():
    """A long session must not grow the replay guard without limit."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("b", lambda a: a + 1)
    widget = comp.widget(collapse_all=False)
    try:
        for i in range(_REQUEST_HISTORY * 10):
            widget.compute_request = {"all": True, "request_id": f"r{i}"}

        assert len(widget._seen_requests) == _REQUEST_HISTORY
        assert widget.status == "Computed all available nodes"
    finally:
        widget.close()


@pytest.mark.stress
def test_many_widgets_on_one_computation():
    """Several views of the same graph all stay in step."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("b", lambda a: a + 1)
    widgets = [comp.widget(collapse_all=False) for _ in range(10)]
    try:
        comp.compute_all()

        assert all(w.revision == comp.revision for w in widgets)
        assert all(w.node_states == widgets[0].node_states for w in widgets)

        for widget in widgets:
            widget.close()

        assert comp._subscriptions == []
    finally:
        for widget in widgets:
            widget.close()


@pytest.mark.stress
def test_many_subscribers_all_receive_every_event():
    """Fan-out to many observers stays complete and ordered."""
    comp = Computation()
    comp.add_node("a", value=1)
    seen: list[tuple[int, int]] = []
    for index in range(200):
        comp.subscribe(lambda event, index=index: seen.append((index, event.revision)))

    for i in range(20):
        comp.insert("a", i)

    assert len(seen) == 200 * 20
    assert [index for index, _revision in seen[:200]] == list(range(200))
