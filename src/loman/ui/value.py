"""Small, deliberately limited value wire format for the notebook UI."""

from __future__ import annotations

import math
from typing import Any


class ValueWireError(ValueError):
    """Raised when an edited UI value has an invalid wire representation."""


_FLOAT_SENTINELS = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}

#: Longest ``repr`` the detail panel will carry. Anything larger is truncated:
#: the panel is for orientation, and the real object stays in Python.
MAX_REPR_LENGTH = 2_000

_ELLIPSIS = "..."


def _float_to_wire(value: float) -> float | str:
    """Convert non-finite floats to JSON-safe sentinel strings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def to_wire(value: Any) -> dict[str, Any]:
    """Describe a value without serializing arbitrary Python objects."""
    if value is None:
        return {"kind": "scalar", "type": "none", "value": None}
    if isinstance(value, bool):
        return {"kind": "scalar", "type": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "scalar", "type": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "scalar", "type": "float", "value": _float_to_wire(value)}
    if isinstance(value, str):
        return {"kind": "scalar", "type": "str", "value": value}
    try:
        value_repr = repr(value)
    except Exception:  # a broken __repr__ must not break the detail panel
        value_repr = f"<{type(value).__name__}: repr unavailable>"
    if len(value_repr) > MAX_REPR_LENGTH:
        value_repr = value_repr[: MAX_REPR_LENGTH - len(_ELLIPSIS)] + _ELLIPSIS
    return {"kind": "repr", "type": type(value).__name__, "repr": value_repr}


def from_wire(data: Any) -> Any:
    """Decode a scalar value produced by :func:`to_wire`."""
    if not isinstance(data, dict) or data.get("kind") != "scalar":
        msg = "Only scalar UI values can be edited"
        raise ValueWireError(msg)
    value_type = data.get("type")
    value = data.get("value")
    if value_type == "none":
        if value is not None:
            msg = "A none value must contain JSON null"
            raise ValueWireError(msg)
        return None
    if value_type == "bool":
        if not isinstance(value, bool):
            msg = "A bool value must contain a boolean"
            raise ValueWireError(msg)
        return value
    if value_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            msg = "An int value must contain an integer"
            raise ValueWireError(msg)
        return value
    if value_type == "float":
        if isinstance(value, str) and value in _FLOAT_SENTINELS:
            return _FLOAT_SENTINELS[value]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = "A float value must contain a number"
            raise ValueWireError(msg)
        return float(value)
    if value_type == "str":
        if not isinstance(value, str):
            msg = "A str value must contain text"
            raise ValueWireError(msg)
        return value
    msg = f"Unsupported scalar type: {value_type!r}"
    raise ValueWireError(msg)
