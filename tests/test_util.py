"""Tests for the util module.

This module tests:
- apply1 function for applying functions to various input types
- as_iterable function for converting inputs to iterables
- apply_n function for cartesian product application
- AttributeView class for dynamic attribute access
- value_eq function for safe value comparison
"""

import types

import numpy as np
import pandas as pd
import pytest

from loman.util import AttributeView, apply1, apply_n, as_iterable, value_eq


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
