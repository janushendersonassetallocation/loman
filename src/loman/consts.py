"""Constants and enumerations for the loman computation engine."""

from enum import Enum


class States(Enum):
    """Possible states for a computation node."""

    PLACEHOLDER = 0
    UNINITIALIZED = 1
    STALE = 2
    COMPUTABLE = 3
    UPTODATE = 4
    ERROR = 5
    PINNED = 6


class NodeAttributes:
    """Constants for node attribute names in the computation graph."""

    VALUE = "value"
    STATE = "state"
    FUNC = "func"
    GROUP = "group"
    TAG = "tag"
    STYLE = "style"
    ARGS = "args"
    KWDS = "kwds"
    TIMING = "timing"
    EXECUTOR = "executor"
    CONVERTER = "converter"


class EdgeAttributes:
    """Constants for edge attribute names in the computation graph."""

    PARAM = "param"


class SystemTags:
    """System-level tags used internally by loman."""

    SERIALIZE = "__serialize__"
    EXPANSION = "__expansion__"


class NodeTransformations:
    """Node transformation types for visualization."""

    CONTRACT = "contract"
    COLLAPSE = "collapse"
    EXPAND = "expand"
