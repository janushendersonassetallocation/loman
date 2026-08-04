"""Small, deliberately limited value wire format for the notebook UI."""

from __future__ import annotations

import math
from collections.abc import Callable
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


def _decode_none(value: Any) -> None:
    """Decode a JSON null."""
    if value is not None:
        msg = "A none value must contain JSON null"
        raise ValueWireError(msg)
    return None


def _decode_bool(value: Any) -> bool:
    """Decode a JSON boolean."""
    if not isinstance(value, bool):
        msg = "A bool value must contain a boolean"
        raise ValueWireError(msg)
    return value


def _decode_int(value: Any) -> int:
    """Decode a JSON integer, rejecting the bool that would pass isinstance."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "An int value must contain an integer"
        raise ValueWireError(msg)
    return value


def _decode_float(value: Any) -> float:
    """Decode a JSON number, or one of the non-finite sentinel strings."""
    if isinstance(value, str) and value in _FLOAT_SENTINELS:
        return _FLOAT_SENTINELS[value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "A float value must contain a number"
        raise ValueWireError(msg)
    return float(value)


def _decode_str(value: Any) -> str:
    """Decode a JSON string."""
    if not isinstance(value, str):
        msg = "A str value must contain text"
        raise ValueWireError(msg)
    return value


#: One decoder per supported scalar type. Each validates strictly rather than
#: coercing, so a browser cannot silently change a node's Python type.
_DECODERS: dict[str, Callable[[Any], Any]] = {
    "none": _decode_none,
    "bool": _decode_bool,
    "int": _decode_int,
    "float": _decode_float,
    "str": _decode_str,
}


def from_wire(data: Any) -> Any:
    """Decode a scalar value produced by :func:`to_wire`.

    :param data: A payload from the browser, which is untrusted.
    :return: The decoded Python value.
    :raises ValueWireError: If the payload is not a scalar this format supports,
        or does not match the type it declares.
    """
    if not isinstance(data, dict) or data.get("kind") != "scalar":
        msg = "Only scalar UI values can be edited"
        raise ValueWireError(msg)
    value_type = data.get("type")
    decoder = _DECODERS.get(value_type) if isinstance(value_type, str) else None
    if decoder is None:
        msg = f"Unsupported scalar type: {value_type!r}"
        raise ValueWireError(msg)
    return decoder(data.get("value"))
