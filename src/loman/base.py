"""Shared state declaration at the bottom of the Computation stack.

:class:`~loman.computeengine.Computation` is built as a chain of layers, each
in its own module and each permitted to call only the layers below it::

    ComputationBase        the shared state declared here
      StateMixin           node state transitions        (loman.state)
        QueryMixin         read-only accessors           (loman.query)
          AttributeMixin   tags, styles, metadata        (loman.attributes)
            ExecutionMixin scheduling and running        (loman.execution)
              PersistenceMixin  copying and file I/O     (loman.persistence)
                MutationMixin   structural changes       (loman.mutation)
                  Computation                            (loman.computeengine)

The direction is the point: a lower layer never reaches upward, so the import
graph is linear and a change to, say, mutation cannot ripple into state. This
class declares the state they all operate on, which keeps that contract
explicit and gives the type checker something to resolve against.
"""

from collections import defaultdict
from concurrent.futures import Executor
from typing import Any

import networkx as nx

from .consts import States
from .nodekey import NodeKey
from .util import AttributeView


class ComputationBase:
    """The state every layer of a Computation operates on."""

    dag: nx.DiGraph
    default_executor: Executor
    executor_map: dict[str, Executor]
    _metadata: dict[NodeKey, Any]
    _tag_map: defaultdict[str, set[NodeKey]]
    _state_map: dict[States, set[NodeKey]]

    #: Attribute-style accessors, built in :meth:`Computation.__init__`.
    v: AttributeView
    s: AttributeView
    i: AttributeView
    o: AttributeView
    t: AttributeView
    style: AttributeView
    tim: AttributeView
    x: AttributeView
    src: AttributeView
