"""Applying a function over one value or many.

A leaf module: it imports nothing from ``loman`` at all. These three helpers
lived in :mod:`loman.util`, which made every module needing them --- including
:mod:`loman.nodekey` and :mod:`loman.graph_utils` --- depend on the whole
utility layer, and left ``util`` unable to import ``nodekey`` at module level
without a cycle.
"""

from __future__ import annotations

import itertools
import types
from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeVar, overload

T = TypeVar("T")
R = TypeVar("R")


@overload
def apply1(f: Callable[..., R], xs: list[T], *args: Any, **kwds: Any) -> list[R]: ...


@overload
def apply1(f: Callable[..., R], xs: T, *args: Any, **kwds: Any) -> R: ...


@overload
def apply1(f: Callable[..., R], xs: Generator[T, None, None], *args: Any, **kwds: Any) -> Generator[R, None, None]: ...


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
        return xs  # type: ignore[return-value]
    return (xs,)  # type: ignore[return-value]


def apply_n(f: Callable[..., Any], *xs: Any, **kwds: Any) -> None:
    """Apply function f to the cartesian product of iterables xs."""
    for p in itertools.product(*[as_iterable(x) for x in xs]):
        f(*p, **kwds)
