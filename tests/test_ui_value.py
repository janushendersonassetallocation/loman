"""Tests for the small UI value wire format.

This module tests:
- Round-tripping the scalar types the detail panel can edit
- JSON-hostile floats, which need explicit sentinels
- Degrading to a read-only repr for everything else
- Rejecting malformed input from the browser
- Properties that must hold for every value, not just the chosen examples
"""

import contextlib
import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from loman.ui.value import MAX_REPR_LENGTH, ValueWireError, from_wire, to_wire

#: Every Python value the wire format claims to carry losslessly.
SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)

#: Values the wire format must degrade to a read-only repr instead.
NON_SCALARS = st.one_of(
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
    st.tuples(st.integers(), st.text()),
    st.sets(st.integers()),
    st.binary(),
    st.complex_numbers(allow_nan=False, allow_infinity=False),
)

#: Arbitrary decoded JSON, which is exactly what an untrusted browser can send.
JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=8,
)


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


class TestWireFormatProperties:
    """Properties that must hold for every value, not just chosen examples.

    The wire format sits between untrusted browser input and ``comp.insert``,
    and its output is synced as JSON. Both of those are total claims about all
    values, which is what property-based testing is for.
    """

    @pytest.mark.property
    @given(SCALARS)
    def test_scalars_round_trip_without_loss(self, value):
        """Any supported scalar survives the trip to the browser and back."""
        result = from_wire(to_wire(value))

        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(result)
        else:
            assert result == value
        assert type(result) is type(value)

    @pytest.mark.property
    @given(SCALARS)
    def test_wire_form_is_always_valid_json(self, value):
        """Traits sync as JSON, which has no NaN or Infinity.

        This is the property the float sentinels exist for: ``allow_nan=False``
        is what a strict JSON encoder does, and anything that trips it would
        break the sync rather than merely look wrong.
        """
        json.dumps(to_wire(value), allow_nan=False)

    @pytest.mark.property
    @given(NON_SCALARS)
    def test_non_scalars_degrade_to_bounded_json_safe_repr(self, value):
        """Arbitrary objects never reach the browser, and never blow the payload."""
        wire = to_wire(value)

        assert wire["kind"] == "repr"
        assert len(wire["repr"]) <= MAX_REPR_LENGTH
        json.dumps(wire, allow_nan=False)
        with pytest.raises(ValueWireError):
            from_wire(wire)

    @pytest.mark.property
    @given(JSON_VALUES)
    def test_untrusted_input_only_ever_raises_value_wire_error(self, payload):
        """Decoding browser input must fail predictably or not at all.

        Any other exception type escaping here would be reported to the user as
        an unexplained failure, and would mean the widget was guessing at what
        the browser sent.
        """
        with contextlib.suppress(ValueWireError):
            from_wire(payload)

    @pytest.mark.property
    @given(SCALARS)
    def test_decoding_an_encoded_value_never_rejects_it(self, value):
        """to_wire and from_wire agree on what counts as a scalar."""
        wire = to_wire(value)

        if wire["kind"] == "scalar":
            from_wire(wire)
        else:
            with pytest.raises(ValueWireError):
                from_wire(wire)
