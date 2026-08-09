"""Stress tests for how the subscription layer scales and what it retains.

Run with ``make stress``. The benchmark tier asserts the per-node cost at one
or two sizes; this tier pushes the sizes far enough that a cost which is
*almost* flat separates from one that is, and checks the bookkeeping does not
accumulate over many mutations.

Thresholds are on counted operations and on retained state, both of which are
deterministic. The one timing assertion compares an operation against itself
later in the same run, so it measures drift rather than machine speed.
"""

import gc

import pytest

from tests.perf import (
    MAX_HASHES_PER_STALE_NODE,
    MAX_HASHES_PER_STALE_NODE_SUBSCRIBED,
    best_of,
    chain,
    counting_hashes,
    hashes_per_stale_node,
    hashes_per_stale_node_subscribed,
)

#: Sizes far enough apart that a per-node cost which secretly depends on graph
#: size shows up as a difference in slope rather than as noise.
SCALING_SIZES = (400, 800, 1600)

#: How much the slope may vary across those sizes. Measured at 7.07, 7.04 and
#: 7.02 --- converging, not growing --- so anything beyond a few percent means
#: the per-node work is not actually per-node.
MAX_SLOPE_DRIFT = 0.15

#: How much slower the last batch of inserts may be than the first. Catches
#: state that accumulates across mutations, which a single-shot measurement
#: cannot see.
MAX_DRIFT_RATIO = 1.5


@pytest.mark.stress
@pytest.mark.parametrize("size", SCALING_SIZES)
def test_unsubscribed_hashing_per_node_holds_at_scale(size):
    """The ceiling has to hold at 1600 nodes, not only at 400."""
    per_node = hashes_per_stale_node(size)

    assert per_node <= MAX_HASHES_PER_STALE_NODE, (
        f"{per_node:.2f} hashes per stale node at {size} nodes, over the {MAX_HASHES_PER_STALE_NODE} ceiling"
    )


@pytest.mark.stress
def test_the_per_node_cost_does_not_itself_grow_with_the_graph():
    """A flat ceiling can hide a slope that is still climbing.

    Measuring at three sizes and comparing the slopes distinguishes a genuine
    per-node cost from one that is creeping towards quadratic.
    """
    slopes = {size: hashes_per_stale_node(size) for size in SCALING_SIZES}
    smallest, largest = min(slopes.values()), max(slopes.values())
    drift = largest - smallest

    assert drift <= MAX_SLOPE_DRIFT, (
        f"hashes per stale node vary by {drift:.2f} across {SCALING_SIZES} "
        f"({slopes}), so the per-node cost depends on the size of the graph"
    )


@pytest.mark.stress
def test_subscribed_hashing_per_node_holds_at_scale():
    """The widget's own cost must stay proportional too."""
    per_node = hashes_per_stale_node_subscribed(max(SCALING_SIZES))

    assert per_node <= MAX_HASHES_PER_STALE_NODE_SUBSCRIBED, (
        f"{per_node:.2f} hashes per stale node with a subscriber at "
        f"{max(SCALING_SIZES)} nodes, over the "
        f"{MAX_HASHES_PER_STALE_NODE_SUBSCRIBED} ceiling"
    )


@pytest.mark.stress
def test_unsubscribed_mutations_leave_no_pending_state_behind():
    """Nothing is listening, so nothing should be recorded for delivery.

    If the pending set filled up regardless it would grow without bound for the
    entire life of an unsubscribed computation, which is every computation that
    never opens a widget.
    """
    comp = chain(200)

    for i in range(500):
        comp.insert("x0", float(i))

    assert comp._pending_changed_nodes == set(), (
        f"{len(comp._pending_changed_nodes)} nodes recorded for delivery with nothing subscribed"
    )
    assert comp._change_depth == 0
    assert comp._pending_graph_changed is False


@pytest.mark.stress
def test_a_subscribers_pending_state_is_cleared_after_every_event():
    """With a subscriber it fills, but it must empty again each time."""
    comp = chain(200)
    sizes = []
    comp.subscribe(lambda event: sizes.append(len(event.changed_nodes)))

    # From 1: the head already holds 0.0, and inserting a value a node already
    # has is a no-op that correctly publishes nothing.
    for i in range(1, 201):
        comp.insert("x0", float(i))
        assert comp._pending_changed_nodes == set(), "pending nodes survived publication and would be re-delivered"

    assert len(sizes) == 200
    # Every insert makes the whole chain stale, so the payload is the graph and
    # must not exceed it however many times the loop runs.
    assert max(sizes) <= 200, f"an event carried {max(sizes)} nodes for a 200-node graph"


@pytest.mark.stress
def test_repeated_inserts_do_not_get_slower():
    """Cost per insert must not drift upwards over a long session.

    Compared against itself rather than against a fixed time, so this says
    nothing about how fast the machine is --- only whether the thousandth
    insert costs materially more than the first.
    """
    comp = chain(300)
    comp.insert("x0", 1.0)

    early = best_of(lambda: comp.insert("x0", 2.0), rounds=5)
    for i in range(2000):
        comp.insert("x0", float(i))
    late = best_of(lambda: comp.insert("x0", 3.0), rounds=5)
    ratio = late / early

    assert ratio <= MAX_DRIFT_RATIO, (
        f"inserts became {ratio:.2f}x slower after 2000 of them, over the {MAX_DRIFT_RATIO}x ceiling"
    )


@pytest.mark.stress
def test_dead_subscribers_are_dropped_rather_than_accumulating():
    """Widgets are opened and discarded; their subscriptions must not pile up."""

    class Watcher:
        """Stands in for a widget: subscribes with a bound method."""

        def on_event(self, event):
            """Ignore the event; only the reference matters here."""

    comp = chain(50)
    for _ in range(500):
        watcher = Watcher()
        comp.subscribe(watcher.on_event)
        del watcher
    gc.collect()

    # Each dead subscription is dropped when next walked, so one mutation is
    # enough to sweep them.
    comp.insert("x0", 1.0)

    assert len(comp._subscriptions) == 0, (
        f"{len(comp._subscriptions)} dead subscriptions retained after 500 short-lived subscribers"
    )


@pytest.mark.stress
def test_hashing_does_not_grow_when_a_subscriber_comes_and_goes():
    """Unsubscribing has to restore the cheap path, not merely stop delivery."""
    comp = chain(400)
    comp.insert("x0", 1.0)

    unsubscribe = comp.subscribe(lambda _event: None)
    with counting_hashes() as watched:
        comp.insert("x0", 2.0)

    unsubscribe()
    with counting_hashes() as unwatched:
        comp.insert("x0", 3.0)

    assert unwatched[0] < watched[0], (
        f"{unwatched[0]} hashes after unsubscribing versus {watched[0]} while "
        f"subscribed; the cheap path was not restored"
    )
