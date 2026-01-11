"""Utility functions and classes for loman computation graphs."""

import itertools
import types

import numpy as np
import pandas as pd


def apply1(f, xs, *args, **kwds):
    """Apply function f to xs, handling generators, lists, and single values."""
    if isinstance(xs, types.GeneratorType):
        return (f(x, *args, **kwds) for x in xs)
    if isinstance(xs, list):
        return [f(x, *args, **kwds) for x in xs]
    return f(xs, *args, **kwds)


def as_iterable(xs):
    """Convert input to iterable form if not already iterable."""
    if isinstance(xs, (types.GeneratorType, list, set)):
        return xs
    return (xs,)


def apply_n(f, *xs, **kwds):
    """Apply function f to the cartesian product of iterables xs."""
    for p in itertools.product(*[as_iterable(x) for x in xs]):
        f(*p, **kwds)


class AttributeView:
    """Provides attribute-style access to dynamic collections."""

    def __init__(self, get_attribute_list, get_attribute, get_item=None):
        """Initialize with functions to get attribute list and individual attributes.

        Args:
            get_attribute_list: Function that returns list of available attributes
            get_attribute: Function that takes an attribute name and returns its value
            get_item: Optional function for item access, defaults to get_attribute
        """
        self.get_attribute_list = get_attribute_list
        self.get_attribute = get_attribute
        self.get_item = get_item
        if self.get_item is None:
            self.get_item = get_attribute

    def __dir__(self):
        """Return list of available attributes."""
        return self.get_attribute_list()

    def __getattr__(self, attr):
        """Get attribute by name, raising AttributeError if not found."""
        try:
            return self.get_attribute(attr)
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key):
        """Get item by key."""
        return self.get_item(key)

    def __getstate__(self):
        """Prepare object for serialization."""
        return {
            "get_attribute_list": self.get_attribute_list,
            "get_attribute": self.get_attribute,
            "get_item": self.get_item,
        }

    def __setstate__(self, state):
        """Restore object from serialized state."""
        self.get_attribute_list = state["get_attribute_list"]
        self.get_attribute = state["get_attribute"]
        self.get_item = state["get_item"]
        if self.get_item is None:
            self.get_item = self.get_attribute

    @staticmethod
    def from_dict(d, use_apply1=True):
        """Create an AttributeView from a dictionary."""
        if use_apply1:

            def get_attribute(xs):
                return apply1(d.get, xs)
        else:
            get_attribute = d.get
        return AttributeView(d.keys, get_attribute)


pandas_types = (pd.Series, pd.DataFrame)


def value_eq(a, b):
    """Compare two values for equality, handling pandas and numpy objects safely.

    - Uses .equals for pandas Series/DataFrame
    - For numpy arrays, returns a single boolean using np.array_equal (treats NaNs as equal)
    - Falls back to == and coerces to bool when possible
    """
    if a is b:
        return True

    # pandas objects: use robust equality
    if isinstance(a, pandas_types):
        return a.equals(b)
    if isinstance(b, pandas_types):  # pragma: no cover
        return b.equals(a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b, equal_nan=True)
        except Exception:
            return False

    # Default comparison; ensure a single boolean
    try:
        result = a == b
        # If result is an array-like truth value, reduce safely
        if isinstance(result, (np.ndarray,)):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False
