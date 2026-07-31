"""Tests for the small UI value wire format."""

import math

import pytest

from loman.ui.value import ValueWireError, from_wire, to_wire


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


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "scalar", "type": "bool", "value": 1},
        {"kind": "scalar", "type": "int", "value": True},
        {"kind": "scalar", "type": "float", "value": "not-a-number"},
        {"kind": "scalar", "type": "str", "value": 1},
        {"kind": "scalar", "type": "mystery", "value": 1},
    ],
)
def test_malformed_scalar_is_rejected(wire):
    """Malformed browser input cannot silently change Python types."""
    with pytest.raises(ValueWireError):
        from_wire(wire)
