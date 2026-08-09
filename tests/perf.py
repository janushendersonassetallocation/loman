"""Shared measurement helpers for the benchmark and stress tiers.

Both tiers guard the same property --- that the subscription machinery costs
nothing to callers who never subscribe --- so they measure it the same way.

**Thresholds are on counted operations wherever possible, not on wall clock.**
Counting is deterministic: it gives the same answer on a laptop and on a shared
CI runner under load, so a threshold can sit close to the true value and still
never flake. The regression these guard against was found exactly this way ---
it was one extra ``hash`` per stale node, far too small to see in a timing run
that varies by more than that between invocations.

Where wall clock is unavoidable the assertion is a *ratio* between two things
measured back to back in the same process, and the threshold is set well clear
of the observed value.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from loman import Computation
from loman.nodekey import NodeKey

#: Hashes per stale node during propagation, with nothing subscribed. Measured
#: at 7.0 and flat from 200 to 800 nodes. It reads 8.0 if the keys are
#: materialised before the state map merges them, because ``set.update(set)``
#: reuses the source set's hashes and ``set.update(tuple)`` recomputes them.
MAX_HASHES_PER_STALE_NODE = 7.5

#: The same, with a subscriber attached. Measured at 12.0: the extra is the
#: event payload, which is work a subscriber asked for. This is a ceiling on
#: how much the widget's presence may cost, not on the unsubscribed path.
MAX_HASHES_PER_STALE_NODE_SUBSCRIBED = 13.0

#: How much slower an operation may be with a subscriber attached than without.
#: Measured at 1.07x for insert and 1.14x for graph construction; the headroom
#: is for CI noise, since this one is wall clock.
MAX_SUBSCRIBER_TIME_RATIO = 1.6


def chain(length: int, *, subscribed: bool = False) -> Computation:
    """Build a linear chain, the shape that makes staleness propagate widest.

    One insert at the head marks every other node stale, so per-node costs in
    the propagation path show up multiplied by ``length``.

    :param length: Number of nodes in the chain.
    :param subscribed: Attach a do-nothing subscriber, as a widget would.
    :return: The computation, with its head node holding ``0.0``.
    """
    comp = Computation()
    comp.add_node("x0", value=0.0)
    for i in range(1, length):
        comp.add_node(f"x{i}", (lambda p: p + 1), kwds={"p": f"x{i - 1}"})
    if subscribed:
        comp.subscribe(lambda _event: None)
    return comp


def fan_out(width: int, *, subscribed: bool = False) -> Computation:
    """Build a wide graph of calculated nodes over one input."""
    comp = Computation()
    if subscribed:
        comp.subscribe(lambda _event: None)
    comp.add_node("x", value=1)
    for i in range(width):
        comp.add_node(f"n{i}", lambda x: x + 1, kwds={"x": "x"})
    return comp


@contextmanager
def counting_hashes() -> Iterator[list[int]]:
    """Count :class:`NodeKey` hashes, restoring the real one on the way out.

    Yields a one-element list so the count is readable after the block.
    """
    original = NodeKey.__hash__
    calls = [0]

    def counting(self: NodeKey) -> int:
        """Tally the call, then hash as usual."""
        calls[0] += 1
        return original(self)

    NodeKey.__hash__ = counting  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        NodeKey.__hash__ = original  # type: ignore[method-assign]


def hashes_per_stale_node(length: int) -> float:
    """Measure hashing during propagation, per node made stale.

    Counted against a second, shorter chain and divided by the difference, so
    the fixed cost of the insert itself cancels and only the per-node slope
    remains. That is the quantity a regression in the propagation path moves.

    :param length: Length of the chain to measure.
    :return: Hashes attributable to each additional stale node.
    """
    baseline_length = length // 4
    long_chain, short_chain = chain(length), chain(baseline_length)
    # Warm both: the first insert on a fresh graph does one-off work.
    long_chain.insert("x0", 1.0)
    short_chain.insert("x0", 1.0)

    with counting_hashes() as long_count:
        long_chain.insert("x0", 2.0)
    with counting_hashes() as short_count:
        short_chain.insert("x0", 2.0)

    return (long_count[0] - short_count[0]) / (length - baseline_length)


def hashes_per_stale_node_subscribed(length: int) -> float:
    """As :func:`hashes_per_stale_node`, with a subscriber attached."""
    baseline_length = length // 4
    long_chain = chain(length, subscribed=True)
    short_chain = chain(baseline_length, subscribed=True)
    long_chain.insert("x0", 1.0)
    short_chain.insert("x0", 1.0)

    with counting_hashes() as long_count:
        long_chain.insert("x0", 2.0)
    with counting_hashes() as short_count:
        short_chain.insert("x0", 2.0)

    return (long_count[0] - short_count[0]) / (length - baseline_length)


def best_of(action: Callable[[], Any], rounds: int = 9) -> float:
    """Return the fastest of ``rounds`` runs, in seconds.

    The minimum rather than the mean: it is the run least disturbed by whatever
    else the machine was doing, which is what makes a ratio between two of these
    stable enough to assert on.
    """
    action()  # warm
    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        action()
        timings.append(time.perf_counter() - start)
    return min(timings)
