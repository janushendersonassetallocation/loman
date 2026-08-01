"""Tests for the small UI value wire format.

This module tests:
- Round-tripping the scalar types the detail panel can edit
- JSON-hostile floats, which need explicit sentinels
- Degrading to a read-only repr for everything else
- Rejecting malformed input from the browser
"""

import math

import pytest

from loman.ui.value import MAX_REPR_LENGTH, ValueWireError, from_wire, to_wire


@pytest.mark.parametrize("value", [None, True, False, 0, 42, -3, 1.25, "hello"])
def test_scalar_roundtrip(value):
    """Supported scalar values round-trip without type loss."""
    result = from_wire(to_wire(value))
    assert result == value
    assert type(result) is type(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_roundtrip(value):
    """Non-JSON floats use explicit sentinel strings."""
    result = from_wire(to_wire(value))
    if math.isnan(value):
        assert math.isnan(result)
    else:
        assert result == value


def test_arbitrary_values_are_repr_only():
    """The browser never receives an arbitrary Python object."""
    value = [1, 2, 3]
    wire = to_wire(value)
    assert wire == {"kind": "repr", "type": "list", "repr": "[1, 2, 3]"}
    with pytest.raises(ValueWireError, match="Only scalar"):
        from_wire(wire)


def test_integer_wire_value_widens_to_float():
    """A browser number input reports 5, not 5.0, for a float node."""
    assert from_wire({"kind": "scalar", "type": "float", "value": 5}) == 5.0


def test_broken_repr_does_not_break_the_panel():
    """An object whose __repr__ raises still yields a usable description."""

    class Hostile:
        """Raises whenever anyone tries to describe it."""

        def __repr__(self):
            """Refuse to be represented."""
            raise RuntimeError

    wire = to_wire(Hostile())

    assert wire == {"kind": "repr", "type": "Hostile", "repr": "<Hostile: repr unavailable>"}


def test_oversized_repr_is_truncated():
    """The panel is for orientation; the real object stays in Python."""
    wire = to_wire(list(range(MAX_REPR_LENGTH)))

    assert len(wire["repr"]) == MAX_REPR_LENGTH
    assert wire["repr"].endswith("...")


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "scalar", "type": "bool", "value": 1},
        {"kind": "scalar", "type": "int", "value": True},
        {"kind": "scalar", "type": "float", "value": "not-a-number"},
        {"kind": "scalar", "type": "float", "value": True},
        {"kind": "scalar", "type": "str", "value": 1},
        {"kind": "scalar", "type": "none", "value": "surprise"},
        {"kind": "scalar", "type": "mystery", "value": 1},
    ],
)
def test_malformed_scalar_is_rejected(wire):
    """Malformed browser input cannot silently change Python types."""
    with pytest.raises(ValueWireError):
        from_wire(wire)


@pytest.mark.parametrize("wire", ["not-a-dict", None, 42, {"kind": "repr", "type": "list", "repr": "[]"}])
def test_non_scalar_payload_is_rejected(wire):
    """Only the scalar wire format can be turned back into a Python value."""
    with pytest.raises(ValueWireError, match="Only scalar"):
        from_wire(wire)
