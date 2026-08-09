"""Benchmarks that fail rather than merely report.

``test_subscription_benchmarks`` measures the shape of the cost. This module
asserts the part of it that must not move: that a computation nobody subscribes
to pays nothing for the subscription machinery being there.

Every test here takes the ``benchmark`` fixture, because ``make benchmark`` runs
with ``--benchmark-only`` and *skips* tests that do not --- a threshold test
without the fixture would sit here looking like a guard while never running.
The fixture's own timings go to the report; the assertions are made against
counted operations, which do not vary with what else the machine is doing.

The regression this exists for: ``_set_states`` materialised its keys into a
tuple so an event payload could read them twice, and ``set.update(tuple)``
rehashes every element where ``set.update(set)`` reuses the hashes the source
set already holds. That is one extra hash per stale node on every insert, for
every caller, subscribed or not --- 8.4% on a 400-node chain, and far too small
per operation for a timing run to catch.
"""

import pytest

from tests.perf import (
    MAX_HASHES_PER_STALE_NODE,
    MAX_HASHES_PER_STALE_NODE_SUBSCRIBED,
    MAX_SUBSCRIBER_TIME_RATIO,
    best_of,
    chain,
    fan_out,
    hashes_per_stale_node,
    hashes_per_stale_node_subscribed,
)

#: Long enough that per-node costs dominate the fixed cost of one insert.
CHAIN = 400

#: Width for construction benchmarks. Construction is a one-time cost, so this
#: only has to be big enough to measure.
WIDTH = 400


def test_unsubscribed_insert_does_not_rehash_the_stale_nodes(benchmark):
    """The propagation path, on the branch every Loman user takes.

    Fails at 8.0 hashes per stale node, passes at the 7.0 that merging the keys
    set-to-set gives.
    """
    comp = chain(CHAIN)
    comp.insert("x0", 1.0)
    benchmark(lambda: comp.insert("x0", 2.0))

    per_node = hashes_per_stale_node(CHAIN)

    assert per_node <= MAX_HASHES_PER_STALE_NODE, (
        f"{per_node:.2f} hashes per stale node, over the {MAX_HASHES_PER_STALE_NODE} "
        f"ceiling. About 7 is expected; 8 means the keys are rehashed on merge "
        f"rather than reusing the hashes the source set already holds."
    )


def test_subscribed_insert_pays_for_the_payload_and_no_more(benchmark):
    """A subscriber's own cost also has a ceiling.

    Building the event is work the subscriber asked for, so this is looser than
    the unsubscribed guard --- but it still has to be bounded, or the widget
    makes every mutation progressively more expensive.
    """
    comp = chain(CHAIN, subscribed=True)
    comp.insert("x0", 1.0)
    benchmark(lambda: comp.insert("x0", 2.0))

    per_node = hashes_per_stale_node_subscribed(CHAIN)

    assert per_node <= MAX_HASHES_PER_STALE_NODE_SUBSCRIBED, (
        f"{per_node:.2f} hashes per stale node with a subscriber attached, over "
        f"the {MAX_HASHES_PER_STALE_NODE_SUBSCRIBED} ceiling (about 12 expected)."
    )


def test_a_subscriber_does_not_multiply_the_cost_of_an_insert(benchmark):
    """Wall clock, so the assertion is a ratio measured back to back.

    An absolute threshold here would encode the speed of whatever machine last
    ran it. The ratio between the two paths is the thing that should be stable.
    """
    watched = chain(CHAIN, subscribed=True)
    unwatched = chain(CHAIN)
    benchmark(lambda: watched.insert("x0", 2.0))

    without = best_of(lambda: unwatched.insert("x0", 3.0))
    with_sub = best_of(lambda: watched.insert("x0", 3.0))
    ratio = with_sub / without

    assert ratio <= MAX_SUBSCRIBER_TIME_RATIO, (
        f"an insert costs {ratio:.2f}x more with a subscriber attached, over the "
        f"{MAX_SUBSCRIBER_TIME_RATIO}x ceiling (about 1.07x expected)"
    )


def test_a_subscriber_does_not_multiply_the_cost_of_construction(benchmark):
    """Every ``add_node`` is a structural event, so this is the worst case.

    Construction publishes one event per node rather than one per batch, which
    makes it the most exposed of the mutating paths.
    """
    benchmark(lambda: fan_out(WIDTH, subscribed=True))

    without = best_of(lambda: fan_out(WIDTH), rounds=5)
    with_sub = best_of(lambda: fan_out(WIDTH, subscribed=True), rounds=5)
    ratio = with_sub / without

    assert ratio <= MAX_SUBSCRIBER_TIME_RATIO, (
        f"building a {WIDTH}-node graph costs {ratio:.2f}x more with a subscriber "
        f"attached, over the {MAX_SUBSCRIBER_TIME_RATIO}x ceiling (about 1.14x expected)"
    )


@pytest.mark.parametrize("size", [200, 400])
def test_hashing_stays_flat_as_the_graph_grows(benchmark, size):
    """Per-node cost must not itself depend on the number of nodes.

    A ceiling alone would not catch a cost that is linear per node today and
    quadratic tomorrow; measuring the same slope at two sizes does.
    """
    comp = chain(size)
    comp.insert("x0", 1.0)
    benchmark(lambda: comp.insert("x0", 2.0))

    per_node = hashes_per_stale_node(size)

    assert per_node <= MAX_HASHES_PER_STALE_NODE, (
        f"{per_node:.2f} hashes per stale node at {size} nodes, over the {MAX_HASHES_PER_STALE_NODE} ceiling"
    )
