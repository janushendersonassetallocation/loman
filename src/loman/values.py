"""Value types shared by the computation engine and its serialization layer.

This module sits near the bottom of the import graph: it depends only on
:mod:`loman.consts`. That is deliberate. Both :mod:`loman.computeengine` and
:mod:`loman.serialization.computation` need these types, and hosting them here
lets each import them at module scope instead of reaching into the other at
call time.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .consts import States


@dataclass
class Error:
    """Container for error information during computation."""

    exception: Exception
    traceback: str


@dataclass
class NodeData:
    """Data associated with a computation node."""

    state: States
    value: object


@dataclass
class TimingData:
    """Timing information for computation execution."""

    start: datetime
    end: datetime
    duration: float


@dataclass()
class ConstantValue:
    """Container for constant values in computations."""

    value: object


C = ConstantValue


class _ParameterType(Enum):
    """Internal enum for distinguishing positional and keyword parameters."""

    ARG = 1
    KWD = 2


@dataclass
class _ParameterItem:
    """Internal container for parameter information during computation."""

    type: _ParameterType
    name: int | str
    value: object
