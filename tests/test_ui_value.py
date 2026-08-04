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

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from loman.ui.value import (
    MAX_CELL_LENGTH,
    MAX_REPR_LENGTH,
    MAX_TABLE_COLS,
    MAX_TABLE_ROWS,
    MAX_TREE_CHILDREN,
    MAX_TREE_DEPTH,
    ValueWireError,
    apply_cell_edit,
    from_wire,
    to_wire,
)

#: Every Python value the wire format claims to carry losslessly.
SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)

#: Values that render as a bounded tree rather than a scalar.
TREE_VALUES = st.one_of(
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
    st.tuples(st.integers(), st.text()),
)

#: Values the wire format must degrade to a read-only repr instead.
OPAQUE_VALUES = st.one_of(
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
    wire = to_wire(object())
    assert wire["kind"] == "repr"
    assert wire["type"] == "object"
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
    wire = to_wire({object() for _ in range(MAX_REPR_LENGTH)})

    assert wire["kind"] == "repr"
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
    @given(OPAQUE_VALUES)
    def test_opaque_values_degrade_to_bounded_json_safe_repr(self, value):
        """Arbitrary objects never reach the browser, and never blow the payload."""
        wire = to_wire(value)

        assert wire["kind"] == "repr"
        assert len(wire["repr"]) <= MAX_REPR_LENGTH
        json.dumps(wire, allow_nan=False)
        with pytest.raises(ValueWireError):
            from_wire(wire)

    @pytest.mark.property
    @given(TREE_VALUES)
    def test_containers_render_as_json_safe_trees(self, value):
        """Nested data is shown structurally, and still cannot be sent back."""
        wire = to_wire(value)

        assert wire["kind"] == "tree"
        assert wire["root"]["size"] == len(value)
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


class TestTableWireFormat:
    """DataFrames, Series and arrays, sent as a bounded window."""

    def test_dataframe_sends_a_window_and_the_true_shape(self):
        """The scaling rule is never to serialize a value in bulk."""
        frame = pd.DataFrame({"a": range(500), "b": [float(i) for i in range(500)]})

        wire = to_wire(frame)

        assert wire["kind"] == "table"
        assert wire["type"] == "DataFrame"
        assert wire["shape"] == [500, 2]
        assert wire["shown"] == [MAX_TABLE_ROWS, 2]
        assert len(wire["rows"]) == MAX_TABLE_ROWS
        assert wire["column_kinds"] == ["int", "float"]
        assert wire["editable"] is True

    def test_wide_frames_are_windowed_by_column_too(self):
        """A thousand-column frame must not arrive column by column."""
        wire = to_wire(pd.DataFrame({f"c{i}": [i] for i in range(200)}))

        assert wire["shape"] == [1, 200]
        assert wire["shown"] == [1, MAX_TABLE_COLS]
        assert len(wire["columns"]) == MAX_TABLE_COLS

    def test_table_payload_is_valid_json(self):
        """Missing values are the usual reason a frame will not serialize."""
        frame = pd.DataFrame({"x": [1.0, float("nan"), float("inf")], "s": ["a", None, "c"]})

        json.dumps(to_wire(frame), allow_nan=False)

    def test_series_is_a_single_column_table(self):
        """A Series is tabular, and keeps its name as the column heading."""
        wire = to_wire(pd.Series([1.5, 2.5], name="dv01"))

        assert wire["type"] == "Series"
        assert wire["columns"] == ["dv01"]
        assert wire["editable"] is True

    def test_unnamed_series_still_has_a_heading(self):
        """A blank column heading would be unreadable."""
        assert to_wire(pd.Series([1, 2]))["columns"] == ["value"]

    def test_arrays_render_but_are_not_editable(self):
        """NumPy coerces silently on assignment, so an edit could lie."""
        wire = to_wire(np.arange(6).reshape(2, 3))

        assert wire["type"] == "ndarray"
        assert wire["shape"] == [2, 3]
        assert wire["editable"] is False

    def test_one_dimensional_arrays_render_as_a_column(self):
        """A 1-D array still has a shape worth reporting honestly."""
        wire = to_wire(np.array([1.0, 2.0, 3.0]))

        assert wire["shape"] == [3]
        assert len(wire["rows"]) == 3

    def test_high_dimensional_arrays_fall_back_to_repr(self):
        """There is no honest table view of a 3-D array."""
        assert to_wire(np.zeros((2, 2, 2)))["kind"] == "repr"


class TestTreeWireFormat:
    """Nested dicts and lists, bounded by depth and breadth."""

    def test_nested_structure_is_described(self):
        """The shape of the data is what the panel is for."""
        wire = to_wire({"a": 1, "b": [1, 2], "c": {"d": "x"}})

        assert wire["kind"] == "tree"
        assert wire["root"]["type"] == "dict"
        assert [child["key"] for child in wire["root"]["children"]] == ["a", "b", "c"]

    def test_breadth_is_capped_and_reported(self):
        """A large dict must not arrive whole."""
        wire = to_wire({str(i): i for i in range(MAX_TREE_CHILDREN * 3)})

        assert len(wire["root"]["children"]) == MAX_TREE_CHILDREN
        assert wire["root"]["truncated"] is True
        assert wire["root"]["size"] == MAX_TREE_CHILDREN * 3

    def test_depth_is_capped(self):
        """Deep nesting must terminate rather than recurse without bound."""
        value: dict = {"leaf": 1}
        for _ in range(MAX_TREE_DEPTH * 2):
            value = {"down": value}

        def deepest(node, depth=0):
            """Walk to the bottom of the returned tree."""
            children = node.get("children")
            return depth if not children else deepest(children[0], depth + 1)

        assert deepest(to_wire(value)["root"]) <= MAX_TREE_DEPTH

    def test_leaves_carry_json_safe_values(self):
        """A non-finite float inside a list still has to serialize."""
        json.dumps(to_wire([1.0, float("nan"), object()]), allow_nan=False)


class TestCellEditing:
    """Replacing one cell of a tabular value."""

    def test_edit_returns_a_copy_and_preserves_dtype(self):
        """Loman's copy is shallow, so a value may be shared; never mutate it."""
        frame = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})

        updated = apply_cell_edit(frame, 1, 1, 9.5)

        assert updated["b"].tolist() == [1.5, 9.5]
        assert frame["b"].tolist() == [1.5, 2.5]
        assert updated.dtypes.tolist() == frame.dtypes.tolist()

    def test_series_edit_keeps_its_name(self):
        """A Series round-trips through a frame internally; that must not show."""
        updated = apply_cell_edit(pd.Series([1.0, 2.0], name="dv01"), 0, 0, 7.5)

        assert isinstance(updated, pd.Series)
        assert updated.name == "dv01"
        assert updated.tolist() == [7.5, 2.0]

    @pytest.mark.parametrize(
        ("row", "column"),
        [(99, 0), (0, 99), (-1, 0), (0, -1)],
    )
    def test_out_of_range_cells_are_rejected(self, row, column):
        """A stale window from the browser cannot write outside the frame."""
        with pytest.raises(ValueWireError, match="outside"):
            apply_cell_edit(pd.DataFrame({"a": [1]}), row, column, 1)

    def test_wrong_type_for_the_column_is_rejected(self):
        """Editing must not silently change a column's dtype."""
        with pytest.raises(ValueWireError, match="float"):
            apply_cell_edit(pd.DataFrame({"a": [1.5]}), 0, 0, "not a number")

    def test_bool_and_int_columns_are_kept_apart(self):
        """A bool is an int subclass, which would otherwise slip through."""
        with pytest.raises(ValueWireError, match="int"):
            apply_cell_edit(pd.DataFrame({"a": [1]}), 0, 0, True)
        with pytest.raises(ValueWireError, match="bool"):
            apply_cell_edit(pd.DataFrame({"a": [True]}), 0, 0, 1)

    def test_text_columns_accept_text(self):
        """The common case still has to work."""
        updated = apply_cell_edit(pd.DataFrame({"ccy": ["USD"]}), 0, 0, "GBP")

        assert updated["ccy"].tolist() == ["GBP"]

    def test_non_tabular_values_are_rejected(self):
        """A cell address is meaningless for a scalar."""
        with pytest.raises(ValueWireError, match="cannot be edited"):
            apply_cell_edit(42, 0, 0, 1)

    def test_unsupported_column_types_are_rejected(self):
        """Better to refuse than to coerce a datetime from a text box."""
        frame = pd.DataFrame({"t": pd.to_datetime(["2026-01-01"])})

        with pytest.raises(ValueWireError, match="cannot set"):
            apply_cell_edit(frame, 0, 0, "tomorrow")


class TestWireFormatEdgeCases:
    """Corners that only appear with real pandas and NumPy data."""

    def test_numpy_scalars_are_unwrapped_to_python_types(self):
        """np.int64 is not a Python int on every platform."""
        wire = to_wire(pd.DataFrame({"a": np.array([7], dtype=np.int64)}))

        assert wire["rows"] == [[7]]
        json.dumps(wire, allow_nan=False)

    def test_missing_values_become_null(self):
        """NaT and NA have no JSON form, and read better as empty than as text."""
        frame = pd.DataFrame({"t": pd.to_datetime(["2026-01-01", None])})

        assert to_wire(frame)["rows"][1] == [None]

    def test_unorderable_cell_values_fall_back_to_repr(self):
        """pd.isna raises for some containers; that must not escape."""
        wire = to_wire(pd.DataFrame({"a": [object()]}))

        assert isinstance(wire["rows"][0][0], str)
        assert len(wire["rows"][0][0]) <= MAX_CELL_LENGTH

    def test_deeply_nested_branch_reports_truncation_without_children(self):
        """At the depth cap a branch says so rather than descending further."""

        def deepest(node):
            """Walk to the bottom of the returned tree."""
            children = node.get("children")
            return node if not children else deepest(children[0])

        value: dict = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}

        bottom = deepest(to_wire(value)["root"])

        assert bottom["truncated"] is True

    def test_edit_rejected_when_the_dtype_cannot_hold_the_value(self):
        """A null into an integer column has no integer representation."""
        with pytest.raises(ValueWireError, match="Cannot put that value"):
            apply_cell_edit(pd.DataFrame({"a": [1, 2]}), 0, 0, None)

    def test_zero_dimensional_numpy_scalars_are_unwrapped(self):
        """A 0-d array is a scalar, and must not render as array(3)."""
        assert to_wire({"a": np.float64(3.5)})["root"]["children"][0]["value"] == 3.5

    def test_cells_that_defeat_isna_fall_back_to_repr(self):
        """pd.isna raises on an array, which must not escape the cell renderer."""
        wire = to_wire(pd.DataFrame({"a": [np.array([1, 2])]}))

        assert isinstance(wire["rows"][0][0], str)

    def test_deeply_nested_lists_are_capped_too(self):
        """The depth cap applies to lists as well as dicts."""

        def deepest(node):
            """Walk to the bottom of the returned tree."""
            children = node.get("children")
            return node if not children else deepest(children[0])

        value: list = [1]
        for _ in range(MAX_TREE_DEPTH * 2):
            value = [value]

        assert deepest(to_wire(value)["root"])["truncated"] is True
