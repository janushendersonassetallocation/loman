"""Property-based tests for the serialization value model.

Run by ``make hypothesis-test``, which selects ``-m "hypothesis or property"``;
hypothesis itself comes from Rhiza's ``tests`` requirement bundle.

The rest of the suite checks a curated list of awkward values. That list was
written by someone who already knew where the bugs were, which is exactly its
weakness: it cannot find a case nobody thought of. These tests state the
properties instead and let hypothesis look for counterexamples --- nested
containers, unusual keys, extreme floats, and the boundaries around the
inline/out-of-line threshold, which is where two encodings meet and so the most
likely place for them to disagree.

The central property is that encoding is *lossless*: whatever goes in comes back
equal, of the same type, through every container.
"""

import datetime
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loman import Computation
from loman.serialization import ComputationSerializer

pytestmark = pytest.mark.property

# Saving touches the filesystem, so the default deadline is both too tight and
# not measuring anything useful. Examples are kept modest for the same reason.
SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Floats are drawn including NaN and both infinities: they have no JSON literal
# and are the reason the encoding tags them rather than emitting bare tokens.
floats = st.floats(allow_nan=True, allow_infinity=True)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    floats,
    st.text(),
    st.binary(max_size=64),
    st.datetimes(),
    st.dates(),
    st.times(),
    st.timedeltas(),
    st.decimals(allow_nan=False, allow_infinity=False),
)

# Anything hashable enough to be a dict key or a set member. Non-string keys are
# the interesting half: JSON objects cannot express them.
hashable_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.binary(max_size=16),
)


def containers(inner):
    """Return a strategy for containers holding *inner* values."""
    return st.one_of(
        st.lists(inner, max_size=6),
        st.tuples(inner, inner),
        st.dictionaries(hashable_scalars, inner, max_size=6),
        st.sets(hashable_scalars, max_size=6),
        st.frozensets(hashable_scalars, max_size=6),
    )


nested = st.recursive(scalars, containers, max_leaves=15)

numpy_dtypes = st.sampled_from(["int8", "int32", "int64", "float32", "float64", "bool"])


@st.composite
def numpy_arrays(draw):
    """Return arrays spanning both sides of the inline/blob threshold."""
    dtype = draw(numpy_dtypes)
    # 1024 float64 elements is the default threshold, so these straddle it.
    size = draw(st.sampled_from([0, 1, 2, 500, 1023, 1024, 1025, 4000]))

    if dtype == "bool":
        elements = st.booleans()
    elif dtype.startswith("float"):
        # Finite only: NaN equality is covered by the scalar strategies, and an
        # array of them says nothing extra about the encoding.
        elements = st.floats(allow_nan=False, allow_infinity=False, width=32 if dtype == "float32" else 64)
    else:
        info = np.iinfo(dtype)
        elements = st.integers(min_value=int(info.min), max_value=int(info.max))

    values = draw(st.lists(elements, min_size=size, max_size=size))
    return np.array(values, dtype=dtype)


@st.composite
def frames(draw):
    """Return DataFrames with varied index types, dtypes and sizes."""
    rows = draw(st.sampled_from([0, 1, 3, 200, 1200]))
    index_kind = draw(st.sampled_from(["range", "int", "str", "datetime", "datetime_tz", "multi"]))

    if index_kind == "range":
        index = pd.RangeIndex(rows)
    elif index_kind == "int":
        index = pd.Index(range(rows, 0, -1), dtype="int64")
    elif index_kind == "str":
        index = pd.Index([f"r{i}" for i in range(rows)])
    elif index_kind == "datetime":
        index = pd.date_range("2020-01-01", periods=rows, freq="min")
    elif index_kind == "datetime_tz":
        index = pd.date_range("2020-01-01", periods=rows, freq="min", tz="Europe/London")
    else:
        index = pd.MultiIndex.from_arrays([np.arange(rows) // 10, np.arange(rows) % 10])

    data = {
        "f": np.arange(rows, dtype="float64"),
        "i": np.arange(rows, dtype="int64"),
        "s": [f"v{i}" for i in range(rows)],
    }
    keep = draw(st.lists(st.sampled_from(sorted(data)), min_size=1, max_size=3, unique=True))
    return pd.DataFrame({k: data[k] for k in keep}, index=index)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _equivalent(original, restored):
    """Return whether *restored* is the same value as *original*."""
    if isinstance(original, (pd.DataFrame, pd.Series, pd.Index)):
        return type(restored) is type(original) and restored.equals(original)
    if isinstance(original, np.ndarray):
        equal_nan = original.dtype.kind == "f"
        return restored.dtype == original.dtype and np.array_equal(restored, original, equal_nan=equal_nan)
    if isinstance(original, float) and math.isnan(original):
        return isinstance(restored, float) and math.isnan(restored)
    if isinstance(original, (list, tuple, set, frozenset, dict)):
        return type(restored) is type(original) and _containers_match(original, restored)
    return type(restored) is type(original) and restored == original


def _containers_match(original, restored):
    """Compare containers elementwise, tolerating NaN, which is never equal to itself."""
    if isinstance(original, dict):
        if set(map(_key, original)) != set(map(_key, restored)):
            return False
        by_key = {_key(k): v for k, v in restored.items()}
        return all(_equivalent(v, by_key[_key(k)]) for k, v in original.items())
    if isinstance(original, (set, frozenset)):
        return {_key(x) for x in original} == {_key(x) for x in restored}
    return len(original) == len(restored) and all(_equivalent(a, b) for a, b in zip(original, restored, strict=True))


def _key(value):
    """Return a hashable stand-in that treats NaN as equal to itself."""
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    return (type(value).__name__, value)


def _roundtrip(value, tmp_path, container="zip", profile=None):
    """Save a single-node computation holding *value* and load it back."""
    comp = Computation()
    comp.add_node("x", value=value)

    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    name = {"zip": "c.loman", "dir": "c_dir", "json": "c.json"}[container]
    path = str(tmp_path / name)
    comp.save(path, container=container, profile=profile)
    return Computation.load(path).v.x


class TestValuesRoundTrip:
    """Whatever goes in comes back equal, and of the same type."""

    @SETTINGS
    @given(value=nested)
    def test_nested_values(self, value, tmp_path):
        """Arbitrarily nested scalars and containers survive a round-trip."""
        assert _equivalent(value, _roundtrip(value, tmp_path))

    @SETTINGS
    @given(value=numpy_arrays())
    def test_numpy_arrays(self, value, tmp_path):
        """Arrays survive, on either side of the out-of-line threshold."""
        assert _equivalent(value, _roundtrip(value, tmp_path))

    @SETTINGS
    @given(value=frames())
    def test_dataframes(self, value, tmp_path):
        """Frames survive with their index type, dtypes and values."""
        assert _equivalent(value, _roundtrip(value, tmp_path))

    @SETTINGS
    @given(value=st.dictionaries(hashable_scalars, scalars, max_size=8))
    def test_dict_keys_of_any_type(self, value, tmp_path):
        """Keys keep their type, which a JSON object cannot express directly."""
        restored = _roundtrip(value, tmp_path)
        assert {_key(k) for k in restored} == {_key(k) for k in value}


class TestEncodingIsContainerIndependent:
    """The container changes where bytes go, never what the value means."""

    @SETTINGS
    @given(value=st.one_of(nested, numpy_arrays(), frames()))
    @pytest.mark.parametrize("container", ["zip", "dir", "json"])
    def test_same_value_from_every_container(self, value, container, tmp_path):
        """A value read back is the same whichever container held it."""
        profile = "readable" if container == "json" else None
        assert _equivalent(value, _roundtrip(value, tmp_path, container=container, profile=profile))

    @SETTINGS
    @given(value=st.one_of(numpy_arrays(), frames()))
    def test_inline_and_blob_agree(self, value, tmp_path):
        """The inline and out-of-line encodings produce the same value.

        These are two independent code paths for the same data, chosen by size.
        Nothing else checks that they agree on values near the threshold.
        """
        inline = _roundtrip(value, tmp_path / "a", profile="readable")
        blob = _roundtrip(value, tmp_path / "b", profile="efficient")
        assert _equivalent(inline, blob)


class TestDocumentIsAlwaysValidJson:
    """No value can produce a document a strict JSON parser rejects."""

    @SETTINGS
    @given(value=st.one_of(nested, numpy_arrays(), frames()))
    def test_no_non_standard_tokens(self, value):
        """NaN and the infinities never reach the file as bare tokens."""

        def reject(token):
            msg = f"non-standard JSON token: {token}"
            raise AssertionError(msg)

        comp = Computation()
        comp.add_node("x", value=value)

        json.loads(ComputationSerializer().dumps(comp), parse_constant=reject)


class TestSavingIsDeterministic:
    """The same value saved twice produces the same bytes."""

    @SETTINGS
    @given(value=st.one_of(nested, numpy_arrays(), frames()))
    def test_two_saves_are_identical(self, value, tmp_path):
        """Byte-level reproducibility holds for any value, not just simple ones.

        Sets are the interesting case: their iteration order varies between
        processes, so this only holds because members are sorted before writing.
        """
        comp = Computation()
        comp.add_node("x", value=value)

        first, second = tmp_path / "one.loman", tmp_path / "two.loman"
        comp.save(str(first))
        comp.save(str(second))

        assert first.read_bytes() == second.read_bytes()


class TestSaveDoesNotMutateValues:
    """Encoding reads a value; it never alters it."""

    @SETTINGS
    @given(value=st.one_of(numpy_arrays(), frames()))
    def test_value_is_unchanged_by_saving(self, value, tmp_path):
        """The in-memory value is the same after a save as before it."""
        before = value.copy()

        comp = Computation()
        comp.add_node("x", value=value)
        comp.save(str(tmp_path / "c.loman"))

        assert _equivalent(before, value)


class TestDatetimeResolution:
    """Temporal resolution is recorded, not assumed."""

    @SETTINGS
    @given(
        unit=st.sampled_from(["s", "ms", "us", "ns"]),
        rows=st.integers(min_value=0, max_value=200),
        tz=st.sampled_from([None, "UTC", "Europe/London", "America/New_York"]),
    )
    def test_any_resolution_and_zone(self, unit, rows, tz, tmp_path):
        """Every resolution and timezone combination round-trips exactly."""
        index = pd.date_range("2020-01-01", periods=rows, freq="min", tz=tz).as_unit(unit)
        frame = pd.DataFrame({"a": np.arange(rows, dtype="float64")}, index=index)

        restored = _roundtrip(frame, tmp_path)

        assert restored.index.unit == unit
        assert str(restored.index.tz) == str(tz)
        assert restored.equals(frame)

    @SETTINGS
    @given(
        value=st.datetimes(min_value=datetime.datetime(1900, 1, 1), max_value=datetime.datetime(2200, 1, 1)),
        unit=st.sampled_from(["s", "ms", "us", "ns"]),
    )
    def test_timestamp_resolution(self, value, unit, tmp_path):
        """A scalar Timestamp keeps its resolution and its instant."""
        timestamp = pd.Timestamp(value).as_unit(unit)

        restored = _roundtrip(timestamp, tmp_path)

        assert restored.unit == unit
        assert restored == timestamp
