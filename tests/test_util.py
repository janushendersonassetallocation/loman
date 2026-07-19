"""Tests for the util module.

This module tests:
- apply1 function for applying functions to various input types
- as_iterable function for converting inputs to iterables
- apply_n function for cartesian product application
- repeated block, fan-out, and fan-in graph builders
- AttributeView class for dynamic attribute access
- value_eq function for safe value comparison
"""

import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from loman import Computation, NodeKey, States
from loman.util import (
    AttributeView,
    add_fan_in,
    add_fan_out,
    add_repeated_blocks,
    add_repeated_pipeline,
    apply1,
    apply_n,
    as_iterable,
    value_eq,
)


def _double_block() -> Computation:
    """Create a reusable block that doubles its input."""
    block = Computation()
    block.add_node("data")
    block.add_node("result", lambda data: data * 2)
    return block


class TestComputationUtilities:
    """Test graph-building computation utilities."""

    def test_add_repeated_blocks_uses_keys_as_path_parts(self):
        """Create one namespaced block for every supplied key."""
        comp = Computation()

        blocks = add_repeated_blocks(comp, _double_block(), [101, 202], base_path="instruments")

        assert blocks == {
            101: NodeKey(("instruments", 101)),
            202: NodeKey(("instruments", 202)),
        }
        assert comp.nodes() == [
            NodeKey(("instruments", 101, "data")),
            NodeKey(("instruments", 101, "result")),
            NodeKey(("instruments", 202, "data")),
            NodeKey(("instruments", 202, "result")),
        ]

    def test_add_repeated_blocks_rejects_duplicates_before_mutating(self):
        """Reject duplicate keys without adding a partial set of blocks."""
        comp = Computation()

        with pytest.raises(ValueError, match="Duplicate repeated block key"):
            add_repeated_blocks(comp, _double_block(), ["a", "a"], base_path="blocks")

        assert comp.nodes() == []

    def test_add_repeated_blocks_rejects_existing_nodes_before_mutating(self):
        """Reject repeated blocks that would replace an existing graph node."""
        comp = Computation()
        comp.add_node("blocks/a/data")

        with pytest.raises(ValueError, match="would replace existing nodes"):
            add_repeated_blocks(comp, _double_block(), ["a", "b"], base_path="blocks")

        assert comp.nodes() == ["blocks/a/data"]

    def test_add_repeated_blocks_does_not_copy_values_by_default(self):
        """Treat a populated block as a calculation template by default."""
        block = _double_block()
        block.insert("data", 3)
        block.compute_all()
        comp = Computation()

        blocks = add_repeated_blocks(comp, block, ["a"], base_path="blocks")

        assert comp.state(blocks["a"] / "data") == States.UNINITIALIZED
        assert comp.state(blocks["a"] / "result") == States.UNINITIALIZED

    def test_add_fan_out_broadcasts_and_invalidates_normally(self):
        """Broadcast values through normal dependency and invalidation behavior."""
        comp = Computation()
        comp.add_node("source")
        targets = {"a": "blocks/a/data", "b": "blocks/b/data"}

        result = add_fan_out(comp, "source", targets)

        assert result == {"a": NodeKey(("blocks", "a", "data")), "b": NodeKey(("blocks", "b", "data"))}
        assert comp.state("blocks/a/data") == States.UNINITIALIZED
        comp.insert("source", 3)
        comp.compute_all()
        assert comp.v[["blocks/a/data", "blocks/b/data"]] == [3, 3]

        comp.insert("source", 5)
        assert comp.s[["blocks/a/data", "blocks/b/data"]] == [States.COMPUTABLE, States.COMPUTABLE]
        comp.compute_all()
        assert comp.v[["blocks/a/data", "blocks/b/data"]] == [5, 5]

    def test_add_fan_out_transforms_dataframes_at_computation_time(self):
        """Slice one source dataframe independently for each target key."""
        comp = Computation()
        comp.add_node("all_data")

        def select_instrument(frame, instrument_id):
            return frame.loc[[instrument_id]]

        add_fan_out(
            comp,
            "all_data",
            {"AAPL": "blocks/AAPL/data", "MSFT": "blocks/MSFT/data"},
            transform=select_instrument,
        )

        assert comp.state("blocks/AAPL/data") == States.UNINITIALIZED
        frame = pd.DataFrame({"price": [190.0, 430.0]}, index=pd.Index(["AAPL", "MSFT"], name="instrument_id"))
        comp.insert("all_data", frame)
        comp.compute_all()

        pd.testing.assert_frame_equal(comp.v["blocks/AAPL/data"], frame.loc[["AAPL"]])
        pd.testing.assert_frame_equal(comp.v["blocks/MSFT/data"], frame.loc[["MSFT"]])

    def test_add_fan_out_validates_target_nodes(self):
        """Reject ambiguous and self-replacing transformed fan-out targets."""
        comp = Computation()
        with pytest.raises(ValueError, match="targets must be unique"):
            add_fan_out(comp, "source", {"a": "target", "b": "target"})
        with pytest.raises(ValueError, match="cannot also be the source"):
            add_fan_out(comp, "source", {"a": "source"}, transform=lambda value, key: (value, key))

        comp.add_node("calculation", lambda: 1)
        with pytest.raises(ValueError, match="must be an input or placeholder"):
            add_fan_out(comp, "source", {"a": "calculation"})

    def test_add_fan_out_rejects_cycles_before_mutating(self):
        """Reject a fan-out edge that would reverse an existing dependency."""
        comp = Computation()
        comp.add_node("target")
        comp.add_node("source", lambda target: target)

        with pytest.raises(ValueError, match="would create a cycle"):
            add_fan_out(comp, "source", {"a": "target"})

        assert comp.i.source == ["target"]
        assert comp.i.target == []

    def test_add_fan_in_collects_values_when_computed(self):
        """Collect keyed values without reading uninitialized sources eagerly."""
        comp = Computation()
        comp.add_node("blocks/a/result")
        comp.add_node("blocks/b/result")

        result = add_fan_in(
            comp,
            "combined",
            {"a": "blocks/a/result", "b": "blocks/b/result"},
        )

        assert result == NodeKey(("combined",))
        assert comp.state(result) == States.UNINITIALIZED
        comp.insert("blocks/a/result", 10)
        comp.insert("blocks/b/result", 20)
        comp.compute(result)
        assert comp.value(result) == {"a": 10, "b": 20}

    def test_add_fan_in_combines_keyed_dataframes(self):
        """Pass an ordered instrument mapping to a custom dataframe combiner."""
        comp = Computation()
        comp.add_node("AAPL", value=pd.DataFrame({"price": [190.0]}))
        comp.add_node("MSFT", value=pd.DataFrame({"price": [430.0]}))

        def concat_instruments(frames):
            return pd.concat(frames, names=["instrument_id"])

        add_fan_in(comp, "prices", {"AAPL": "AAPL", "MSFT": "MSFT"}, combine=concat_instruments)
        comp.compute("prices")

        expected = pd.DataFrame(
            {"price": [190.0, 430.0]},
            index=pd.MultiIndex.from_tuples([("AAPL", 0), ("MSFT", 0)], names=["instrument_id", None]),
        )
        pd.testing.assert_frame_equal(comp.v.prices, expected)

    def test_add_fan_in_supports_reductions(self):
        """Use the same combine contract for scalar reductions."""
        comp = Computation()
        comp.add_node("a", value=10)
        comp.add_node("b", value=20)

        add_fan_in(comp, "total", {"a": "a", "b": "b"}, combine=lambda values: sum(values.values()))
        comp.compute("total")

        assert comp.v.total == 30

    def test_add_fan_in_validates_source_nodes(self):
        """Reject ambiguous fan-in dependencies and result cycles."""
        comp = Computation()
        with pytest.raises(ValueError, match="source nodes must be unique"):
            add_fan_in(comp, "result", {"a": "source", "b": "source"})
        with pytest.raises(ValueError, match="cannot also be a source"):
            add_fan_in(comp, "result", {"a": "result"})
        comp.add_node("result")
        with pytest.raises(ValueError, match="already exists"):
            add_fan_in(comp, "result", {"a": "source"})

    @pytest.mark.parametrize(
        ("block_input", "block_output", "message"),
        [
            ("missing", "result", "block input does not exist"),
            ("data", "missing", "block output does not exist"),
            ("result", "data", "block input must be an input node"),
        ],
    )
    def test_add_repeated_pipeline_validates_ports_before_mutating(self, block_input, block_output, message):
        """Validate block ports before adding any pipeline nodes."""
        comp = Computation()

        with pytest.raises(ValueError, match=message):
            add_repeated_pipeline(
                comp,
                _double_block(),
                ["a"],
                base_path="blocks",
                source="source",
                block_input=block_input,
                block_output=block_output,
                result="combined",
            )

        assert comp.nodes() == []

    def test_add_repeated_pipeline_rejects_existing_result_before_mutating(self):
        """Preserve an existing result node when pipeline validation fails."""
        comp = Computation()
        comp.add_node("combined", value=10)

        with pytest.raises(ValueError, match="result node already exists"):
            add_repeated_pipeline(
                comp,
                _double_block(),
                ["a"],
                base_path="blocks",
                source="source",
                block_input="data",
                block_output="result",
                result="combined",
            )

        assert comp.nodes() == ["combined"]
        assert comp.v.combined == 10

    def test_add_repeated_pipeline_rejects_transformed_self_link_before_mutating(self):
        """Reject a transformed source that is also a generated block input."""
        comp = Computation()

        with pytest.raises(ValueError, match="cannot also be the source"):
            add_repeated_pipeline(
                comp,
                _double_block(),
                ["a"],
                base_path="blocks",
                source="blocks/a/data",
                block_input="data",
                block_output="result",
                result="combined",
                transform=lambda value, key: value,
            )

        assert comp.nodes() == []

    def test_add_repeated_pipeline_rejects_source_result_overlap_before_mutating(self):
        """Reject a source/result overlap without adding a partial pipeline."""
        comp = Computation()
        block = Computation()
        block.add_node("data")
        block.add_node("result", lambda: 1)

        with pytest.raises(ValueError, match="source cannot also be the result"):
            add_repeated_pipeline(
                comp,
                block,
                ["a"],
                base_path="blocks",
                source="combined",
                block_input="data",
                block_output="result",
                result="combined",
            )

        assert comp.nodes() == []

    def test_public_utility_type_hints_are_runtime_resolvable(self):
        """Support runtime introspection of public utility annotations."""
        from typing import get_type_hints

        assert get_type_hints(add_repeated_blocks)["return"] is not None
        assert get_type_hints(add_fan_out)["return"] is not None
        assert get_type_hints(add_fan_in)["return"] is not None
        assert get_type_hints(add_repeated_pipeline)["return"] is not None

    def test_add_repeated_pipeline_rejects_block_output_source_before_mutating(self):
        """Reject using a generated output as the source for its own block."""
        comp = Computation()

        with pytest.raises(ValueError, match="would create a cycle"):
            add_repeated_pipeline(
                comp,
                _double_block(),
                ["a"],
                base_path="blocks",
                source="blocks/a/result",
                block_input="data",
                block_output="result",
                result="combined",
            )

        assert comp.nodes() == []

    def test_add_repeated_pipeline_composes_fan_out_blocks_and_fan_in(self):
        """Build and recalculate a complete keyed repeated-block pipeline."""
        comp = Computation()
        comp.add_node("all_values")

        def select_value(values, key):
            return values[key]

        pipeline = add_repeated_pipeline(
            comp,
            _double_block(),
            ["a", "b"],
            base_path="blocks",
            source="all_values",
            block_input="data",
            block_output="result",
            result="total",
            transform=select_value,
            combine=lambda values: sum(values.values()),
        )

        assert pipeline.blocks == {"a": NodeKey(("blocks", "a")), "b": NodeKey(("blocks", "b"))}
        assert pipeline.result == NodeKey(("total",))
        assert comp.state("total") == States.UNINITIALIZED

        comp.insert("all_values", {"a": 2, "b": 3})
        comp.compute("total")
        assert comp.v[["blocks/a/result", "blocks/b/result", "total"]] == [4, 6, 10]

        comp.insert("all_values", {"a": 5, "b": 7})
        comp.compute("total")
        assert comp.v[["blocks/a/result", "blocks/b/result", "total"]] == [10, 14, 24]


class TestApply1:
    """Test apply1 function."""

    def test_apply1_to_single_value(self):
        """Test apply1 applies function to single value."""
        result = apply1(lambda x: x * 2, 5)
        assert result == 10

    def test_apply1_to_list(self):
        """Test apply1 applies function to each element in list."""
        result = apply1(lambda x: x * 2, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_apply1_to_generator(self):
        """Test apply1 returns generator for generator input."""

        def gen():
            yield 1
            yield 2
            yield 3

        result = apply1(lambda x: x * 2, gen())

        assert isinstance(result, types.GeneratorType)
        assert list(result) == [2, 4, 6]

    def test_apply1_with_extra_args(self):
        """Test apply1 passes extra args to function."""
        result = apply1(lambda x, y: x + y, [1, 2], 10)
        assert result == [11, 12]

    def test_apply1_with_extra_kwds(self):
        """Test apply1 passes extra kwargs to function."""
        result = apply1(lambda x, y=0: x + y, [1, 2], y=10)
        assert result == [11, 12]

    def test_apply1_with_generator(self):
        """Test apply1 with a generator input."""

        def double(x):
            return x * 2

        gen = (x for x in [1, 2, 3])
        result = apply1(double, gen)
        # Result should be a generator
        assert list(result) == [2, 4, 6]


class TestAsIterable:
    """Test as_iterable function."""

    def test_as_iterable_single_value(self):
        """Test as_iterable wraps single value in tuple."""
        result = as_iterable(5)
        assert result == (5,)

    def test_as_iterable_list(self):
        """Test as_iterable returns list unchanged."""
        lst = [1, 2, 3]
        result = as_iterable(lst)
        assert result is lst

    def test_as_iterable_set(self):
        """Test as_iterable returns set unchanged."""
        s = {1, 2, 3}
        result = as_iterable(s)
        assert result is s

    def test_as_iterable_generator(self):
        """Test as_iterable returns generator unchanged."""

        def gen():
            yield 1

        g = gen()
        result = as_iterable(g)
        assert result is g


class TestApplyN:
    """Test apply_n function."""

    def test_apply_n_single_iterable(self):
        """Test apply_n with single iterable."""
        results = []
        apply_n(lambda x: results.append(x), [1, 2, 3])
        assert results == [1, 2, 3]

    def test_apply_n_cartesian_product(self):
        """Test apply_n applies to cartesian product."""
        results = []
        apply_n(lambda x, y: results.append((x, y)), [1, 2], ["a", "b"])
        assert results == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

    def test_apply_n_with_single_values(self):
        """Test apply_n converts single values to iterables."""
        results = []
        apply_n(lambda x, y: results.append((x, y)), 1, 2)
        assert results == [(1, 2)]

    def test_apply_n_with_kwargs(self):
        """Test apply_n passes kwargs to function."""
        results = []
        apply_n(lambda x, y=0: results.append(x + y), [1, 2], y=10)
        assert results == [11, 12]


class TestAttributeView:
    """Test AttributeView class."""

    def test_attribute_view_getattr(self):
        """Test AttributeView attribute access."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView(d.keys, d.get)

        assert view.foo == 1
        assert view.bar == 2

    def test_attribute_view_getattr_raises_for_missing(self):
        """Test AttributeView raises AttributeError for missing attribute."""
        d = {"foo": 1}
        # Use __getitem__ instead of get, so KeyError is raised for missing keys
        view = AttributeView(d.keys, d.__getitem__)

        with pytest.raises(AttributeError, match="missing"):
            _ = view.missing

    def test_attribute_view_getitem(self):
        """Test AttributeView item access."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView(d.keys, d.get)

        assert view["foo"] == 1
        assert view["bar"] == 2

    def test_attribute_view_dir(self):
        """Test AttributeView dir() returns attribute list."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView(d.keys, d.get)

        assert set(dir(view)) == {"foo", "bar"}

    def test_attribute_view_custom_get_item(self):
        """Test AttributeView with custom get_item function."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView(d.keys, d.get, get_item=lambda k: d.get(k, "default"))

        assert view["missing"] == "default"

    def test_attribute_view_from_dict(self):
        """Test AttributeView.from_dict factory method."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView.from_dict(d)

        assert view.foo == 1
        assert view.bar == 2

    def test_attribute_view_from_dict_with_use_apply1_false(self):
        """Test AttributeView.from_dict with use_apply1=False."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView.from_dict(d, use_apply1=False)

        assert view.foo == 1
        assert view.bar == 2

    def test_attribute_view_getstate_setstate(self):
        """Test AttributeView serialization via __getstate__ and __setstate__."""
        d = {"foo": 1, "bar": 2}
        view = AttributeView(d.keys, d.get)

        state = view.__getstate__()
        new_view = AttributeView.__new__(AttributeView)
        new_view.__setstate__(state)

        assert new_view.foo == 1
        assert new_view.bar == 2

    def test_attribute_view_setstate_with_none_get_item(self):
        """Test AttributeView __setstate__ handles None get_item."""
        d = {"foo": 1}
        view = AttributeView(d.keys, d.get, get_item=None)

        state = view.__getstate__()
        new_view = AttributeView.__new__(AttributeView)
        new_view.__setstate__(state)

        # get_item should default to get_attribute
        assert new_view["foo"] == 1

    def test_attribute_view_getstate_setstate_from_dict(self):
        """Test AttributeView serialization methods."""
        d = {"a": 1, "b": 2}
        av = AttributeView.from_dict(d)

        state = av.__getstate__()
        assert "get_attribute_list" in state
        assert "get_attribute" in state
        assert "get_item" in state

        # Create new AttributeView and restore state
        new_av = AttributeView(lambda: [], lambda x: None)
        new_av.__setstate__(state)
        assert new_av.a == 1

    def test_attribute_view_setstate_with_none_get_item_manual(self):
        """Test AttributeView setstate when get_item is None."""
        d = {"a": 1}
        # Create AttributeView without using from_dict to have None get_item
        av = AttributeView.__new__(AttributeView)
        av.get_attribute_list = d.keys
        av.get_attribute = d.get
        av.get_item = None  # Explicitly set to None

        state = av.__getstate__()
        assert state["get_item"] is None

        new_av = AttributeView(lambda: [], lambda x: None)
        new_av.__setstate__(state)
        # After setstate with get_item=None, get_item should default to get_attribute
        assert new_av["a"] == 1

    def test_attribute_view_from_dict_no_apply1(self):
        """Test AttributeView.from_dict with use_apply1=False."""
        d = {"a": 1, "b": 2}
        av = AttributeView.from_dict(d, use_apply1=False)
        assert av.a == 1
        assert av.b == 2

    def test_attribute_view_dir_from_dict(self):
        """Test AttributeView __dir__ returns attribute list."""
        d = {"a": 1, "b": 2}
        av = AttributeView.from_dict(d)

        attrs = dir(av)
        assert "a" in attrs
        assert "b" in attrs


class TestValueEq:
    """Test value_eq function."""

    def test_value_eq_same_object(self):
        """Test value_eq returns True for same object."""
        obj = [1, 2, 3]
        assert value_eq(obj, obj) is True

    def test_value_eq_equal_primitives(self):
        """Test value_eq for equal primitive values."""
        assert value_eq(1, 1) is True
        assert value_eq("hello", "hello") is True
        assert value_eq(3.14, 3.14) is True

    def test_value_eq_unequal_primitives(self):
        """Test value_eq for unequal primitive values."""
        assert value_eq(1, 2) is False
        assert value_eq("hello", "world") is False

    def test_value_eq_pandas_series(self):
        """Test value_eq for pandas Series."""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 3])
        s3 = pd.Series([1, 2, 4])

        assert value_eq(s1, s2) is True
        assert value_eq(s1, s3) is False

    def test_value_eq_pandas_dataframe(self):
        """Test value_eq for pandas DataFrame."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df3 = pd.DataFrame({"a": [1, 2], "b": [3, 5]})

        assert value_eq(df1, df2) is True
        assert value_eq(df1, df3) is False

    def test_value_eq_numpy_arrays(self):
        """Test value_eq for numpy arrays."""
        a1 = np.array([1, 2, 3])
        a2 = np.array([1, 2, 3])
        a3 = np.array([1, 2, 4])

        assert value_eq(a1, a2) is True
        assert value_eq(a1, a3) is False

    def test_value_eq_numpy_arrays_with_nan(self):
        """Test value_eq handles NaN in numpy arrays."""
        a1 = np.array([1, np.nan, 3])
        a2 = np.array([1, np.nan, 3])

        assert value_eq(a1, a2) is True

    def test_value_eq_mixed_types(self):
        """Test value_eq with mixed types."""
        assert value_eq(1, "1") is False
        assert value_eq([1, 2], (1, 2)) is False

    def test_value_eq_lists_equal(self):
        """Test value_eq for equal lists."""
        a = [1, 2, 3]
        b = [1, 2, 3]
        assert value_eq(a, b)

    def test_value_eq_lists_not_equal(self):
        """Test value_eq for unequal lists."""
        a = [1, 2, 3]
        b = ["a", "b", "c"]
        assert not value_eq(a, b)

    def test_value_eq_dicts_equal(self):
        """Test value_eq for equal dicts."""
        a = {"x": 1, "y": 2}
        b = {"x": 1, "y": 2}
        assert value_eq(a, b)

    def test_value_eq_dicts_not_equal(self):
        """Test value_eq for unequal dicts."""
        a = {"x": 1, "y": 2}
        b = {"x": 1, "z": 2}
        assert not value_eq(a, b)
        b = {"x": 1, "y": 3}
        assert not value_eq(a, b)

    def test_value_eq_series_with_nan_equal(self):
        """Test value_eq for pandas Series with NaN."""
        a = pd.Series([1.0, np.nan])
        b = pd.Series([1.0, np.nan])
        assert value_eq(a, b)

    def test_value_eq_series_with_nan_not_equal(self):
        """Test value_eq for pandas Series with NaN mismatch."""
        a = pd.Series([1.0, 1.0])
        b = pd.Series([1.0, np.nan])
        assert not value_eq(a, b)
        a = pd.Series([1.0, np.nan])
        b = pd.Series([1.0, 1.0])
        assert not value_eq(a, b)

    def test_value_eq_dataframe_with_nan_equal(self):
        """Test value_eq for pandas DataFrame with NaN."""
        a = pd.DataFrame({"a": [1.0, np.nan]})
        b = pd.DataFrame({"a": [1.0, np.nan]})
        assert value_eq(a, b)

    def test_value_eq_dataframe_with_nan_not_equal(self):
        """Test value_eq for pandas DataFrame with NaN mismatch."""
        a = pd.DataFrame({"a": [1.0, 1.0]})
        b = pd.DataFrame({"a": [1.0, np.nan]})
        assert not value_eq(a, b)
        a = pd.DataFrame({"a": [1.0, np.nan]})
        b = pd.DataFrame({"a": [1.0, 1.0]})
        assert not value_eq(a, b)

    def test_value_eq_numpy_arrays_without_nan_not_equal(self):
        """Test value_eq for numpy arrays without NaN not equal."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert not value_eq(a, b)

    def test_value_eq_ndarray_exception(self):
        """Test value_eq when numpy comparison raises exception."""
        a = np.array([1, 2, 3])
        # Create something that causes array_equal to fail
        b = "not an array that can be compared"
        result = value_eq(a, b)
        assert result is False

    def test_value_eq_fallback_exception(self):
        """Test value_eq when comparison raises exception."""

        class BadComparison:
            def __eq__(self, other):
                msg = "Cannot compare"
                raise ValueError(msg)

        a = BadComparison()
        b = BadComparison()
        result = value_eq(a, b)
        # Should return False when comparison fails
        assert result is False

    def test_value_eq_result_is_ndarray(self):
        """Test value_eq when result is ndarray-like."""
        # Create objects where == returns an array
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        result = value_eq(a, b)
        assert result is True

    def test_value_eq_pandas_different(self):
        """Test value_eq with different pandas Series."""
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 4])

        assert value_eq(s1, s2) is False

    def test_value_eq_dataframe(self):
        """Test value_eq with DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 2]})

        assert value_eq(df1, df2) is True

    def test_value_eq_raises_exception(self):
        """Test value_eq handles objects that raise on comparison."""

        class RaisingObj:
            def __eq__(self, other):
                msg = "Cannot compare"
                raise ValueError(msg)

        obj1 = RaisingObj()
        obj2 = RaisingObj()  # Different object, so a is b is False

        # Should return False when comparison raises
        result = value_eq(obj1, obj2)
        assert result is False

    def test_value_eq_ndarray_exception_mock(self):
        """Test value_eq handles numpy arrays that raise on comparison - covers line 114."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])

        # Force np.array_equal to raise an exception
        with patch("numpy.array_equal", side_effect=Exception("Mock failure")):
            result = value_eq(a, b)
            # Should return False when np.array_equal raises
            assert result is False

    def test_value_eq_default_comparison_exception(self):
        """Test value_eq handles exception in default comparison path - covers line 121."""

        class NonArrayRaisingObj:
            """Object that raises on comparison but is not ndarray-like."""

            def __eq__(self, other):
                msg = "Cannot compare"
                raise ValueError(msg)

        obj1 = NonArrayRaisingObj()
        obj2 = NonArrayRaisingObj()

        # Should return False when comparison raises
        result = value_eq(obj1, obj2)
        assert result is False

    def test_value_eq_comparison_returns_ndarray(self):
        """Test value_eq when comparison returns an ndarray - covers line 121."""

        class ArrayReturningObj:
            """Object whose __eq__ returns an ndarray instead of bool."""

            def __eq__(self, other):
                return np.array([True, True, True])

        obj1 = ArrayReturningObj()
        obj2 = ArrayReturningObj()

        # Should handle ndarray result via np.all
        result = value_eq(obj1, obj2)
        assert result is True

    def test_value_eq_comparison_returns_ndarray_false(self):
        """Test value_eq when comparison returns ndarray with False values."""

        class ArrayReturningObjFalse:
            """Object whose __eq__ returns an ndarray with False."""

            def __eq__(self, other):
                return np.array([True, False, True])

        obj1 = ArrayReturningObjFalse()
        obj2 = ArrayReturningObjFalse()

        # Should return False when not all elements are True
        result = value_eq(obj1, obj2)
        assert result is False
