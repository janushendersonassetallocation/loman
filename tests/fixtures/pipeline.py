"""A small pipeline used by the cross-process serialization tests.

It lives in an importable module rather than in the test body because a saved
computation stores each node's function as a module path and a qualified name.
A function defined inside a test, or in a ``__main__`` script, has no importable
path, so the reader process would load the values but not the functions --- and
the test would pass while proving less than it appears to.
"""

import numpy as np
import pandas as pd

from loman import Computation

ROWS = 500


def make_prices(rows: int = ROWS) -> pd.DataFrame:
    """Return a deterministic price frame with a timezone-aware index."""
    rng = np.random.default_rng(0)
    walk = 100 + np.cumsum(rng.standard_normal(rows) * 0.01)
    return pd.DataFrame(
        {"px": np.round(walk, 4), "size": np.arange(rows, dtype="int64")},
        index=pd.date_range("2020-01-01", periods=rows, freq="min", tz="UTC"),
    )


def weighted(prices: pd.DataFrame, multiplier: float) -> pd.Series:
    """Scale prices by a multiplier."""
    return prices["px"] * multiplier


def total(weighted: pd.Series) -> float:
    """Sum a series to a single float."""
    return float(weighted.sum())


def build_pipeline() -> Computation:
    """Return an uncomputed pipeline covering values, functions and edges."""
    comp = Computation()
    comp.add_node("prices", value=make_prices(), group="inputs", tags=["market"])
    comp.add_node("multiplier", value=2.0, group="inputs")
    comp.add_node("weighted", weighted)
    comp.add_node("total", total)
    return comp
