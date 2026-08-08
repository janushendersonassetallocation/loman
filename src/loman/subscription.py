"""Batched change notification, one layer above the shared state.

The subscription machinery sits directly on :class:`~loman.base.ComputationBase`
because the lowest layer that has to report a change is
:mod:`loman.state`: every state transition marks a node, and the public
mutations further up the chain decide where a batch begins and ends. Hosting it
here keeps that direction intact --- state and everything above it can notify,
and nothing below it needs to know notification exists.
"""

import contextlib
import functools
import inspect
import logging
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast

from .base import ComputationBase
from .consts import NodeAttributes, States
from .nodekey import NodeKey

if TYPE_CHECKING:
    from .computeengine import Computation

LOG = logging.getLogger("loman.computeengine")

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ComputationEvent:
    """A batched notification describing a mutation to a computation.

    Subscribers receive one event after each outermost public mutation, even
    when that operation performs many internal state transitions. Values are
    deliberately excluded: consumers can fetch a changed value lazily from
    :attr:`computation` without copying large objects into every event.

    :ivar computation: The live computation that produced the event. It is not a
        snapshot, and continues to change after the event is delivered.
    :ivar revision: Monotonic counter, matching :attr:`Computation.revision` at
        the moment the event was published.
    :ivar changed_nodes: Nodes whose state changed during the operation. When
        :attr:`graph_changed` is true this is *not* a complete description of the
        change, because adding, deleting or renaming nodes and altering tags or
        styles need not change any node's state. Consumers reacting to a
        structural event should re-read the graph rather than trusting this set.
    :ivar states: The state of each entry in :attr:`changed_nodes` that still
        exists, as of publication. Deleted nodes are absent.
    :ivar graph_changed: True when the structure or presentation of the graph
        changed, so any cached rendering of it is stale.
    """

    computation: "Computation"
    revision: int
    changed_nodes: frozenset[NodeKey]
    states: Mapping[NodeKey, States]
    graph_changed: bool = False


ComputationSubscriber = Callable[[ComputationEvent], None]

#: Cap on how many times subscriber-initiated mutations may cascade within one
#: dispatch before Loman gives up. A well-behaved subscriber settles in one or
#: two rounds; anything beyond this is a feedback loop rather than useful work.
_MAX_NOTIFICATION_CASCADES = 16


class _Subscription:
    """One registered subscriber, held weakly when that is safe to do.

    A callback bound to an object is held weakly, so subscribing a widget's
    handler does not keep the widget alive for the lifetime of the computation.
    Everything with no object behind it --- plain functions, lambdas, callable
    objects, :func:`functools.partial` --- is held strongly, because callers
    routinely pass a closure they retain no other reference to and holding
    those weakly would collect them immediately.

    "Bound to an object" means carrying a ``__self__``, which covers methods
    written in Python and those written in C alike. The two need different
    holders: :class:`weakref.WeakMethod` needs a ``__func__`` to rebind
    against, and C methods have none, so those are held as a weak reference to
    the owner plus the attribute name.

    A few owners cannot be weakly referenced at all --- :class:`list`,
    :class:`dict` and :class:`bytearray` among them --- so ``some_list.append``
    falls back to a strong reference. That is a limitation of the type rather
    than a decision here, and it errs towards a subscription that keeps
    delivering rather than one that silently stops.
    """

    __slots__ = ("_name", "_owner", "_strong", "_weak")

    def __init__(self, callback: ComputationSubscriber) -> None:
        """Wrap ``callback``, choosing weak or strong ownership."""
        self._weak: weakref.WeakMethod | None = None
        self._owner: weakref.ref[Any] | None = None
        self._name: str = ""
        self._strong: ComputationSubscriber | None = None
        if inspect.ismethod(callback):
            self._weak = weakref.WeakMethod(callback)
            return
        owner = getattr(callback, "__self__", None)
        name = getattr(callback, "__name__", None)
        if owner is not None and name:
            try:
                self._owner = weakref.ref(owner)
            except TypeError:
                # list, dict, bytearray and friends support no weak references.
                self._strong = callback
            else:
                self._name = name
            return
        self._strong = callback

    def resolve(self) -> ComputationSubscriber | None:
        """Return the callback, or ``None`` once a weakly held owner is gone."""
        if self._strong is not None:
            return self._strong
        if self._weak is not None:
            return self._weak()
        if self._owner is None:  # pragma: no cover - constructor covers all paths
            return None
        owner = self._owner()
        return None if owner is None else getattr(owner, self._name, None)


def _notifies_subscribers(*, graph_changed: bool = False) -> Callable[[F], F]:
    """Batch changes made by a public mutation and notify on completion.

    With no subscribers attached the wrapper is a straight pass-through, so
    ordinary use of Loman pays nothing for the notification machinery.
    """

    def decorate(method: F) -> F:
        """Wrap ``method`` so its changes are batched into a single event."""

        @functools.wraps(method)
        def wrapped(self: "SubscriptionMixin", *args: Any, **kwargs: Any) -> Any:
            """Run the method inside a change batch, publishing on the way out."""
            if self._change_depth == 0 and not self._subscriptions:
                return method(self, *args, **kwargs)
            self._change_depth += 1
            try:
                result = method(self, *args, **kwargs)
                if graph_changed:
                    self._pending_graph_changed = True
                return result
            finally:
                self._change_depth -= 1
                if self._change_depth == 0:
                    self._publish_pending_events()

        return cast("F", wrapped)

    return decorate


class SubscriptionMixin(ComputationBase):
    """Registration, batching and dispatch of computation change events.

    The notification state is declared here and populated by
    :meth:`Computation.__init__`, following the same convention as the shared
    state on :class:`~loman.base.ComputationBase`. None of it is copied or
    serialized: :meth:`~loman.persistence.PersistenceMixin.copy` builds a fresh
    :class:`~loman.computeengine.Computation`, so a copy starts with no
    subscribers.
    """

    _subscriptions: list[_Subscription]
    _revision: int
    _change_depth: int
    _publishing: bool
    _pending_changed_nodes: set[NodeKey]
    _pending_graph_changed: bool

    @property
    def revision(self) -> int:
        """Return the revision number of the most recently published change."""
        return self._revision

    def subscribe(self, callback: ComputationSubscriber) -> Callable[[], None]:
        """Subscribe to batched computation changes.

        Subscribers are notified in registration order, synchronously, on the
        thread that completes the outermost public mutation. A subscriber that
        raises is logged and skipped; it never interrupts the mutation or the
        other subscribers. A subscriber that itself mutates the computation
        causes a further event to be published once the current round finishes,
        up to a bounded number of cascades.

        A callback with an object behind it --- anything carrying a
        ``__self__``, whether written in Python or in C --- is held weakly, so
        subscribing ``obj.handler`` or ``events.append`` does not keep the
        owner alive; callers must retain it themselves. Everything else ---
        plain functions, lambdas, callable objects, :func:`functools.partial`
        --- is held strongly until unsubscribed, because callers commonly pass
        a throwaway closure that nothing else references.

        The exception is an owner that supports no weak references at all, such
        as :class:`list`, :class:`dict` and :class:`bytearray`. There
        ``some_list.append`` falls back to a strong reference, which is a
        limitation of the type rather than a choice, and errs towards a
        subscription that keeps delivering over one that silently stops.

        Subscriptions are not copied by :meth:`copy` and are not serialized.

        :param callback: Function accepting a :class:`ComputationEvent`.
        :return: An idempotent, no-argument unsubscribe function.
        """
        if not callable(callback):
            msg = "callback must be callable"
            raise TypeError(msg)
        subscription = _Subscription(callback)
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            """Remove this subscription, ignoring repeat calls."""
            with contextlib.suppress(ValueError):
                self._subscriptions.remove(subscription)

        return unsubscribe

    def _mark_changed(self, *node_keys: NodeKey) -> None:
        """Record nodes changed by the current public mutation."""
        if self._subscriptions:
            self._pending_changed_nodes.update(node_keys)

    def _take_pending_event(self) -> ComputationEvent:
        """Consume the batched changes and turn them into one event."""
        changed_nodes = frozenset(self._pending_changed_nodes)
        graph_changed = self._pending_graph_changed
        self._pending_changed_nodes.clear()
        self._pending_graph_changed = False
        self._revision += 1
        states = {
            node_key: self.dag.nodes[node_key][NodeAttributes.STATE]
            for node_key in changed_nodes
            if node_key in self.dag
        }
        return ComputationEvent(
            cast("Computation", self), self._revision, changed_nodes, MappingProxyType(states), graph_changed
        )

    def _publish_pending_events(self) -> None:
        """Publish batched events, including any a subscriber triggers in turn.

        Re-entrant calls return immediately: the dispatch loop already running
        picks up whatever the subscriber changed, so a subscriber that mutates
        the computation cannot recurse into the stack.
        """
        if self._publishing:
            return
        self._publishing = True
        try:
            for _ in range(_MAX_NOTIFICATION_CASCADES):
                if not self._pending_changed_nodes and not self._pending_graph_changed:
                    return
                self._dispatch(self._take_pending_event())
            if self._pending_changed_nodes or self._pending_graph_changed:
                LOG.error(
                    "Computation subscribers kept mutating the computation after %s rounds; "
                    "discarding further notifications to break the loop",
                    _MAX_NOTIFICATION_CASCADES,
                )
                self._pending_changed_nodes.clear()
                self._pending_graph_changed = False
        finally:
            self._publishing = False

    def _dispatch(self, event: ComputationEvent) -> None:
        """Deliver one event to every live subscriber, isolating failures."""
        dead = False
        for subscription in tuple(self._subscriptions):
            callback = subscription.resolve()
            if callback is None:
                dead = True
                continue
            try:
                callback(event)
            except Exception:
                LOG.exception("Computation subscriber failed at revision %s", event.revision)
        if dead:
            self._subscriptions = [s for s in self._subscriptions if s.resolve() is not None]
