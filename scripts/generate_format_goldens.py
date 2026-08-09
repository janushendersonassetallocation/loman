"""Regenerate the serialized-format golden corpus for the *current* format version.

The corpus under ``tests/data/formats/vN/`` pins what each released format
version actually looked like on disk.  ``tests/test_format_compat.py`` reads
every file in every directory, which is what turns loman's
backward-compatibility promise into something CI can fail on.

Run this **once per format version bump**, immediately after ``FORMAT_VERSION``
is raised and before the new encoders land::

    uv run python scripts/generate_format_goldens.py

Files for versions that already exist are left alone — regenerating an old
version's corpus with new code would defeat its entire purpose.  Pass
``--force`` only if you are certain a golden file was captured in error.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loman import Computation  # noqa: E402
from loman.serialization.computation import FORMAT_VERSION  # noqa: E402
from tests import format_fixtures as fx  # noqa: E402

GOLDEN_ROOT = REPO_ROOT / "tests" / "data" / "formats"


def _scalars_and_funcs() -> Computation:
    """Scalar values, importable functions, and computed results."""
    comp = Computation()
    comp.add_node("a", value=3)
    comp.add_node("b", fx.add_one, kwds={"x": "a"})
    comp.add_node("c", fx.double, kwds={"x": "a"})
    comp.add_node("d", fx.add, kwds={"x": "b", "y": "c"})
    comp.compute_all()
    return comp


def _all_states() -> Computation:
    """One node in each of the states the format can record.

    STALE needs two levels: inserting at the root leaves the direct child
    COMPUTABLE (its own parents are all up to date) and only the grandchild
    STALE, so a single-level chain never captures the state at all.
    """
    comp = Computation()
    comp.add_node("uptodate", value=1)
    comp.add_node("pinned", value=2)
    comp.pin("pinned")
    comp.add_node("uninitialized")
    comp.add_node("errored", fx.raise_value_error)

    comp.add_node("root", value=10)
    comp.add_node("computable", fx.add_one, kwds={"x": "root"})
    comp.add_node("stale", fx.double, kwds={"x": "computable"})
    comp.compute_all()
    comp.insert("root", 11)
    return comp


def _tags_and_exclusions() -> Computation:
    """User tags, and a node explicitly excluded from serialization."""
    comp = Computation()
    comp.add_node("kept", value=1, tags=["alpha", "beta"])
    comp.add_node("excluded", value=object(), serialize=False)
    comp.add_node("plain", value="hello")
    return comp


def _numpy_and_pandas() -> Computation:
    """Arrays and frames, in the encoding current at capture time."""
    comp = Computation()
    comp.add_node("arr_f8", value=np.array([1.0, 2.0, 3.0]))
    comp.add_node("arr_2d", value=np.arange(6, dtype=np.int64).reshape(2, 3))
    comp.add_node("arr_f4", value=np.array([1.5, 2.5], dtype=np.float32))
    comp.add_node("df", value=pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]}))
    comp.add_node("series", value=pd.Series([1, 2, 3], name="s"))
    return comp


def _containers() -> Computation:
    """Nested lists, tuples and dicts."""
    comp = Computation()
    comp.add_node("lst", value=[1, "two", 3.0, None, True])
    comp.add_node("tup", value=(1, (2, 3), "four"))
    comp.add_node("dct", value={"a": 1, "b": [2, 3], "c": {"d": 4}})
    return comp


def _hierarchical_keys() -> Computation:
    """Block-structured computation, exercising ``/``-separated node keys."""
    inner = Computation()
    inner.add_node("a", value=7)
    inner.add_node("b", fx.add_one, kwds={"x": "a"})
    inner.compute_all()

    comp = Computation()
    comp.add_block("blk", inner, keep_values=True)
    comp.add_node("top", value=1)
    return comp


def _temporal() -> Computation:
    """Datetimes in every shape the codec handles.

    Format version 1 could not express any of this — a frame with a datetime
    column raised outright — so this case first appears in the v2 corpus.
    """
    comp = Computation()
    comp.add_node("dt_col", value=pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3, freq="D")}))
    comp.add_node("dt_nat", value=pd.DataFrame({"t": [pd.Timestamp("2024-01-01"), pd.NaT]}))
    comp.add_node("dt_tz", value=pd.DataFrame({"t": pd.date_range("2024-01-01", periods=2, tz="Europe/London")}))
    comp.add_node("timedelta_col", value=pd.DataFrame({"d": pd.to_timedelta([1, 2, 3], unit="D")}))
    comp.add_node("dt_index", value=pd.DataFrame({"v": [1, 2]}, index=pd.date_range("2024-01-01", periods=2)))
    comp.add_node("dt_array", value=np.array(["2024-01-01", "2024-06-01"], dtype="datetime64[ns]"))
    comp.add_node("scalar_timestamp", value=pd.Timestamp("2024-05-05 13:00"))
    comp.add_node("scalar_date", value=datetime.date(2024, 5, 5))
    comp.add_node("scalar_time", value=datetime.time(13, 30))
    comp.add_node("scalar_timedelta", value=datetime.timedelta(days=2, hours=3))
    return comp


def _extension_dtypes() -> Computation:
    """Categoricals, nullable integers, and structured indexes."""
    comp = Computation()
    comp.add_node(
        "categorical",
        value=pd.DataFrame({"c": pd.Categorical(["a", "b", "a"], categories=["b", "a"], ordered=True)}),
    )
    comp.add_node("nullable", value=pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")}))
    comp.add_node(
        "multiindex",
        value=pd.DataFrame(
            {"v": [1, 2]},
            index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["k", "n"]),
        ),
    )
    comp.add_node("mixed", value=pd.DataFrame({"i": [1, 2], "f": [1.5, 2.5], "s": ["a", "b"], "b": [True, False]}))
    comp.add_node("str_array", value=np.array(["alpha", "beta"]))
    return comp


CASES = {
    "scalars_and_funcs": _scalars_and_funcs,
    "temporal": _temporal,
    "extension_dtypes": _extension_dtypes,
    "all_states": _all_states,
    "tags_and_exclusions": _tags_and_exclusions,
    "numpy_and_pandas": _numpy_and_pandas,
    "containers": _containers,
    "hierarchical_keys": _hierarchical_keys,
}


def main() -> int:
    """Write one golden file per case for the current FORMAT_VERSION."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing version directory (almost never correct)",
    )
    args = parser.parse_args()

    out_dir = GOLDEN_ROOT / f"v{FORMAT_VERSION}"
    if out_dir.exists() and not args.force:
        print(f"{out_dir} already exists — nothing to do.")
        print("Golden files for a released version must not be regenerated with newer code.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, build in CASES.items():
        comp = build()
        path = out_dir / f"{name}.json"
        comp.write_json(str(path))
        # Re-emit indented, so diffs on the corpus are reviewable.
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {path.relative_to(GOLDEN_ROOT.parent.parent)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
