import numpy as np
import pandas as pd

from loman.util import value_eq


def test_ints_equal():
    a = 1
    b = 1
    assert value_eq(a, b)


def test_ints_not_equal():
    a = 1
    b = 2
    assert not value_eq(a, b)


def test_floats_equal():
    a = 1.0
    b = 1.0
    assert value_eq(a, b)


def test_floats_not_equal():
    a = 1.0
    b = 2.0
    assert not value_eq(a, b)


def test_lists_equal():
    a = [1, 2, 3]
    b = [1, 2, 3]
    assert value_eq(a, b)


def test_lists_not_equal():
    a = [1, 2, 3]
    b = ["a", "b", "c"]
    assert not value_eq(a, b)


def test_dicts_equal():
    a = {"x": 1, "y": 2}
    b = {"x": 1, "y": 2}
    assert value_eq(a, b)


def test_dicts_not_equal():
    a = {"x": 1, "y": 2}
    b = {"x": 1, "z": 2}
    assert not value_eq(a, b)
    b = {"x": 1, "y": 3}
    assert not value_eq(a, b)


def test_series_equal():
    a = pd.Series([1.0, np.nan])
    b = pd.Series([1.0, np.nan])
    assert value_eq(a, b)


def test_series_not_equal():
    a = pd.Series([1.0, 1.0])
    b = pd.Series([1.0, np.nan])
    assert not value_eq(a, b)
    a = pd.Series([1.0, np.nan])
    b = pd.Series([1.0, 1.0])
    assert not value_eq(a, b)


def test_df_equal():
    a = pd.DataFrame({"a": [1.0, np.nan]})
    b = pd.DataFrame({"a": [1.0, np.nan]})
    assert value_eq(a, b)


def test_df_not_equal():
    a = pd.DataFrame({"a": [1.0, 1.0]})
    b = pd.DataFrame({"a": [1.0, np.nan]})
    assert not value_eq(a, b)
    a = pd.DataFrame({"a": [1.0, np.nan]})
    b = pd.DataFrame({"a": [1.0, 1.0]})
    assert not value_eq(a, b)
