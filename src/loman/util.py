"""Utility functions and classes for loman computation graphs."""

import itertools
import types
from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

T = TypeVar("T")
R = TypeVar("R")


def apply1(
    f: Callable[..., R], xs: T | list[T] | Generator[T, None, None], *args: Any, **kwds: Any
) -> R | list[R] | Generator[R, None, None]:
    """Apply function f to xs, handling generators, lists, and single values."""
    if isinstance(xs, types.GeneratorType):
        return (f(x, *args, **kwds) for x in xs)
    if isinstance(xs, list):
        return [f(x, *args, **kwds) for x in xs]
    return f(xs, *args, **kwds)


def as_iterable(xs: T | Iterable[T]) -> Iterable[T]:
    """Convert input to iterable form if not already iterable."""
    if isinstance(xs, (types.GeneratorType, list, set)):
        return xs
    return (xs,)  # type: ignore[return-value]


def apply_n(f: Callable[..., Any], *xs: Any, **kwds: Any) -> None:
    """Apply function f to the cartesian product of iterables xs."""
    for p in itertools.product(*[as_iterable(x) for x in xs]):
        f(*p, **kwds)


class AttributeView:
    """Provides attribute-style access to dynamic collections."""

    def __init__(
        self,
        get_attribute_list: Callable[[], Iterable[str]],
        get_attribute: Callable[[str], Any],
        get_item: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize with functions to get attribute list and individual attributes.

        Args:
            get_attribute_list: Function that returns list of available attributes
            get_attribute: Function that takes an attribute name and returns its value
            get_item: Optional function for item access, defaults to get_attribute
        """
        self.get_attribute_list = get_attribute_list
        self.get_attribute = get_attribute
        self.get_item: Callable[[Any], Any] = get_item if get_item is not None else get_attribute

    def __dir__(self) -> list[str]:
        """Return list of available attributes."""
        return list(self.get_attribute_list())

    def __getattr__(self, attr: str) -> Any:
        """Get attribute by name, raising AttributeError if not found."""
        try:
            return self.get_attribute(attr)
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, key: Any) -> Any:
        """Get item by key."""
        return self.get_item(key)

    def __getstate__(self) -> dict[str, Any]:
        """Prepare object for serialization."""
        return {
            "get_attribute_list": self.get_attribute_list,
            "get_attribute": self.get_attribute,
            "get_item": self.get_item,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore object from serialized state."""
        self.get_attribute_list = state["get_attribute_list"]
        self.get_attribute = state["get_attribute"]
        self.get_item = state["get_item"]
        if self.get_item is None:
            self.get_item = self.get_attribute

    @staticmethod
    def from_dict(d: dict[Any, Any], use_apply1: bool = True) -> "AttributeView":
        """Create an AttributeView from a dictionary."""
        if use_apply1:

            def get_attribute(xs: Any) -> Any:
                """Get attribute value from dictionary with apply1 support."""
                return apply1(d.get, xs)
        else:
            get_attribute = d.get  # type: ignore[assignment]
        return AttributeView(d.keys, get_attribute)


pandas_types = (pd.Series, pd.DataFrame)


def value_eq(a: Any, b: Any) -> bool:
    """Compare two values for equality, handling pandas and numpy objects safely.

    - Uses .equals for pandas Series/DataFrame
    - For numpy arrays, returns a single boolean using np.array_equal (treats NaNs as equal)
    - Falls back to == and coerces to bool when possible
    """
    if a is b:
        return True

    # pandas objects: use robust equality
    if isinstance(a, pandas_types):
        return bool(a.equals(b))
    if isinstance(b, pandas_types):  # pragma: no cover
        return bool(b.equals(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
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
