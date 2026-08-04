"""Small, deliberately limited value wire format for the notebook UI.

Four kinds go over the wire:

``scalar``
    ``int``, ``float``, ``str``, ``bool`` and ``None``, carried losslessly and
    editable in both directions.
``table``
    A window onto a DataFrame, Series or 2-D array. Frames and Series are
    editable cell by cell; arrays are shown read-only.
``tree``
    A bounded view of nested dicts and lists.
``repr``
    Everything else, as read-only text.

The windows are the point. The widget's scaling rule is never to serialize node
values in bulk, so a table sends its first :data:`MAX_TABLE_ROWS` rows and
:data:`MAX_TABLE_COLS` columns plus the true shape, and a tree is bounded by
depth and breadth. Anything larger stays in Python, where it belongs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


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

#: Window sent for tabular values. Sized so the payload stays in the tens of
#: kilobytes even for wide frames, which is the same order as the state map.
MAX_TABLE_ROWS = 50
MAX_TABLE_COLS = 20

#: Bounds on a nested dict or list view.
MAX_TREE_DEPTH = 4
MAX_TREE_CHILDREN = 50

#: Longest repr used for a single table cell or tree leaf.
MAX_CELL_LENGTH = 120

_ELLIPSIS = "..."


def _float_to_wire(value: float) -> float | str:
    """Convert non-finite floats to JSON-safe sentinel strings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _truncate(text: str, limit: int) -> str:
    """Shorten text to ``limit`` characters, marking where it was cut."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_ELLIPSIS)] + _ELLIPSIS


def _safe_repr(value: Any, limit: int = MAX_REPR_LENGTH) -> str:
    """Return a bounded repr, tolerating a broken ``__repr__``."""
    try:
        text = repr(value)
    except Exception:  # a broken __repr__ must not break the detail panel
        text = f"<{type(value).__name__}: repr unavailable>"
    return _truncate(text, limit)


def _unwrap(value: Any) -> Any:
    """Convert a NumPy or pandas scalar to its plain Python equivalent.

    ``np.int64`` is not a Python ``int`` on every platform, so cells have to be
    unwrapped before the scalar checks can recognise them.
    """
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", ()) == ():
        try:
            return item()
        except (ValueError, TypeError):  # pragma: no cover - exotic dtypes only
            return value
    return value


def _cell_to_wire(value: Any) -> Any:
    """Render one table cell or tree leaf as a JSON-safe plain value."""
    value = _unwrap(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _float_to_wire(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _safe_repr(value, MAX_CELL_LENGTH)


def _column_kind(dtype: Any) -> str:
    """Classify a column so the browser knows how to render and edit it."""
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "str"
    return "other"


def _frame_to_wire(frame: pd.DataFrame, *, type_name: str, editable: bool) -> dict[str, Any]:
    """Describe a window onto a DataFrame."""
    rows, cols = frame.shape
    window = frame.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLS]
    return {
        "kind": "table",
        "type": type_name,
        "columns": [str(column) for column in window.columns],
        "index": [_cell_to_wire(label) for label in window.index],
        "rows": [[_cell_to_wire(cell) for cell in record] for record in window.itertuples(index=False, name=None)],
        "column_kinds": [_column_kind(window.dtypes.iloc[i]) for i in range(window.shape[1])],
        "shape": [int(rows), int(cols)],
        "shown": [int(window.shape[0]), int(window.shape[1])],
        "editable": editable,
    }


def _table_to_wire(value: Any) -> dict[str, Any] | None:
    """Describe a tabular value, or return None if this is not one.

    Frames and Series are editable cell by cell. Arrays are not: NumPy coerces
    silently on assignment, so an edit could change a value without saying so.
    """
    if isinstance(value, pd.DataFrame):
        return _frame_to_wire(value, type_name="DataFrame", editable=True)
    if isinstance(value, pd.Series):
        name = str(value.name) if value.name is not None else "value"
        return _frame_to_wire(value.to_frame(name=name), type_name="Series", editable=True)
    if isinstance(value, np.ndarray) and value.ndim in (1, 2):
        frame = pd.DataFrame(value if value.ndim == 2 else value.reshape(-1, 1))
        wire = _frame_to_wire(frame, type_name="ndarray", editable=False)
        wire["shape"] = [int(size) for size in value.shape]
        return wire
    return None


def _tree_node(value: Any, depth: int) -> dict[str, Any]:
    """Describe one node of a nested dict or list, bounded by depth and breadth."""
    if isinstance(value, Mapping):
        items = list(value.items())
        node: dict[str, Any] = {"type": "dict", "size": len(items)}
        if depth >= MAX_TREE_DEPTH:
            node["truncated"] = True
            return node
        node["children"] = [
            {"key": _truncate(str(key), MAX_CELL_LENGTH), **_tree_node(child, depth + 1)}
            for key, child in items[:MAX_TREE_CHILDREN]
        ]
        node["truncated"] = len(items) > MAX_TREE_CHILDREN
        return node
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        items = list(value)
        node = {"type": "list", "size": len(items)}
        if depth >= MAX_TREE_DEPTH:
            node["truncated"] = True
            return node
        node["children"] = [
            {"key": str(position), **_tree_node(child, depth + 1)}
            for position, child in enumerate(items[:MAX_TREE_CHILDREN])
        ]
        node["truncated"] = len(items) > MAX_TREE_CHILDREN
        return node
    return {"type": "leaf", "value": _cell_to_wire(value), "repr": type(_unwrap(value)).__name__}


def _tree_to_wire(value: Any) -> dict[str, Any] | None:
    """Describe a nested dict or list, or return None if this is neither."""
    if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        return {"kind": "tree", "type": type(value).__name__, "root": _tree_node(value, 0)}
    return None


def to_wire(value: Any) -> dict[str, Any]:
    """Describe a value without serializing arbitrary Python objects.

    :param value: Any node value.
    :return: A JSON-safe description; see the module docstring for the kinds.
    """
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
    table = _table_to_wire(value)
    if table is not None:
        return table
    tree = _tree_to_wire(value)
    if tree is not None:
        return tree
    return {"kind": "repr", "type": type(value).__name__, "repr": _safe_repr(value)}


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


def apply_cell_edit(value: Any, row: int, column: int, cell: Any) -> Any:
    """Return a copy of a tabular value with one cell replaced.

    The original is never mutated. Loman's :meth:`Computation.copy` is shallow,
    so a node's value can be shared with another computation; editing in place
    would change both.

    :param value: The current node value, a DataFrame or Series.
    :param row: Absolute row position, as counted in the full value.
    :param column: Absolute column position. Ignored for a Series.
    :param cell: Decoded replacement, from :func:`from_wire`.
    :return: A new DataFrame or Series with the cell replaced.
    :raises ValueWireError: If the value is not editable, the address is out of
        range, or the replacement does not fit the column's type.
    """
    if isinstance(value, pd.Series):
        frame, was_series, name = value.to_frame(), True, value.name
    elif isinstance(value, pd.DataFrame):
        frame, was_series, name = value, False, None
    else:
        msg = f"Cells of a {type(value).__name__} cannot be edited"
        raise ValueWireError(msg)

    rows, cols = frame.shape
    if not (0 <= row < rows) or not (0 <= column < cols):
        msg = f"Cell ({row}, {column}) is outside a {rows} by {cols} table"
        raise ValueWireError(msg)

    kind = _column_kind(frame.dtypes.iloc[column])
    if kind == "other":
        msg = f"Column {frame.columns[column]!r} holds values this editor cannot set"
        raise ValueWireError(msg)
    if cell is not None and kind != "str":
        expected = {"int": int, "float": (int, float), "bool": bool}[kind]
        if isinstance(cell, bool) is not (kind == "bool") or not isinstance(cell, expected):
            msg = f"Column {frame.columns[column]!r} holds {kind} values"
            raise ValueWireError(msg)

    updated = frame.copy()
    try:
        updated.isetitem(column, updated.iloc[:, column].astype(object))
        updated.iloc[row, column] = cell
        updated.isetitem(column, updated.iloc[:, column].astype(frame.dtypes.iloc[column]))
    except (TypeError, ValueError) as exc:
        # A None into an int column, or a value the dtype cannot hold.
        msg = f"Cannot put that value in column {frame.columns[column]!r}: {exc}"
        raise ValueWireError(msg) from exc
    return updated.iloc[:, 0].rename(name) if was_series else updated
