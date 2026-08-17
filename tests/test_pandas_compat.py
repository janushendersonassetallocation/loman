"""Compatibility across pandas 2 and pandas 3.

Loman supports both, and the awkward part is temporal resolution: pandas 2
defaults datetimes to nanoseconds and pandas 3 to microseconds. A file must
therefore record the unit its values were held at, or a frame saved under one
version comes back at the wrong scale under the other --- wrong timestamps, not
an error.

`tests/fixtures/pandas2.loman` was written by pandas 2.3.3 and is committed, so
the "old file, newer pandas" direction is guarded in a single environment. The
other direction is covered by the pandas-2 CI job, which runs the whole
serialization suite against the minimum supported version.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

from loman import Computation

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "pandas2.loman"

# The APIs the value model depends on, with the version that introduced them.
# Listed so a future floor change is a deliberate decision rather than a
# discovery in someone's traceback.
REQUIRED_PANDAS_APIS = [
    ("Timestamp.unit", lambda: pd.Timestamp("2020").unit),
    ("Timestamp.as_unit", lambda: pd.Timestamp("2020").as_unit("us")),
    ("Timedelta.unit", lambda: pd.Timedelta("1s").unit),
    ("Timedelta.as_unit", lambda: pd.Timedelta("1s").as_unit("us")),
    ("DatetimeIndex.unit", lambda: pd.date_range("2020", periods=1).unit),
    ("TimedeltaIndex.unit", lambda: pd.timedelta_range("1D", periods=1).unit),
]


class TestRequiredApis:
    """The pandas surface the serializer relies on, and the declared floor."""

    @pytest.mark.parametrize(("name", "call"), REQUIRED_PANDAS_APIS, ids=[n for n, _ in REQUIRED_PANDAS_APIS])
    def test_api_is_available(self, name, call):
        """Each API the value model needs exists in the installed pandas."""
        assert call() is not None

    def test_declared_floor_is_at_least_2(self):
        """Pyproject must not claim support below where those APIs exist."""
        import re

        pyproject = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
        match = re.search(r'"pandas\s*>=\s*([\d.]+)"', pyproject)
        assert match, "no pandas requirement found in pyproject.toml"
        major = int(match.group(1).split(".")[0])
        assert major >= 2, (
            f"pyproject declares pandas>={match.group(1)}, but Timestamp.as_unit and "
            "DatetimeIndex.unit need pandas 2.0. Below that, resolution is silently wrong."
        )


@pytest.fixture(scope="module")
def loaded():
    """Load the committed pandas 2 fixture."""
    return Computation.load(str(FIXTURE))


class TestPandas2FixtureLoads:
    """A container written by pandas 2 loads correctly under any supported pandas."""

    def test_naive_datetime_index(self, loaded):
        """A naive DatetimeIndex frame keeps its values and its resolution."""
        frame = loaded.v.naive

        assert isinstance(frame.index, pd.DatetimeIndex)
        assert list(frame["a"]) == [1.0, 2.0, 3.0]
        assert frame.index[0] == pd.Timestamp("2020-01-01")
        # Written under pandas 2, so the recorded unit is nanoseconds and must
        # survive even where the running pandas would default to microseconds.
        assert frame.index.unit == "ns"

    def test_timezone_aware_index(self, loaded):
        """A tz-aware index keeps its zone and its instants."""
        frame = loaded.v.tz

        assert str(frame.index.tz) == "Europe/London"
        assert frame.index[0] == pd.Timestamp("2020-01-01", tz="Europe/London")

    def test_multiindex(self, loaded):
        """A MultiIndex comes back as one, with its level names."""
        index = loaded.v.multi.index

        assert isinstance(index, pd.MultiIndex)
        assert list(index.names) == ["n", "l"]
        assert list(index) == [(1, "x"), (2, "y")]

    def test_scalar_timestamp(self, loaded):
        """A Timestamp keeps sub-second precision."""
        assert loaded.v.ts == pd.Timestamp("2020-01-01 12:34:56.789012")

    def test_scalar_timedelta(self, loaded):
        """A Timedelta keeps sub-second precision."""
        assert loaded.v.td == pd.Timedelta("1 days 2:03:04.5")

    def test_series(self, loaded):
        """A Series keeps its name and index."""
        series = loaded.v.series

        assert series.name == "s"
        assert list(series) == [1.0, 2.0]
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_blob_backed_array(self, loaded):
        """An out-of-line array reads back byte for byte."""
        assert np.array_equal(loaded.v.array, np.arange(3000, dtype="float64"))

    def test_fixture_stays_small(self):
        """The fixture is a guard, not a data set; keep it out of LFS territory."""
        assert FIXTURE.stat().st_size < 100_000


class TestResolutionIsRecorded:
    """Whatever the running pandas defaults to, the file states the unit."""

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_each_resolution_roundtrips(self, unit, tmp_path):
        """A frame at any supported resolution comes back at that resolution."""
        index = pd.date_range("2020-01-01", periods=3, freq="min").as_unit(unit)
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=index)

        comp = Computation()
        comp.add_node("frame", value=frame)
        comp.save(str(tmp_path / "c.loman"))

        restored = Computation.load(str(tmp_path / "c.loman")).v.frame
        assert restored.index.unit == unit
        assert restored.equals(frame)

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_timestamp_resolution_roundtrips(self, unit, tmp_path):
        """A scalar Timestamp keeps its resolution too."""
        value = pd.Timestamp("2020-01-01 00:00:01").as_unit(unit)

        comp = Computation()
        comp.add_node("ts", value=value)
        comp.save(str(tmp_path / "c.loman"))

        restored = Computation.load(str(tmp_path / "c.loman")).v.ts
        assert restored.unit == unit
        assert restored == value

    def test_manifest_states_the_unit(self, tmp_path):
        """The unit is visible in the manifest, not implied by the reader."""
        import json
        import zipfile

        comp = Computation()
        comp.add_node("frame", value=pd.DataFrame({"a": [1.0]}, index=pd.date_range("2020", periods=1)))
        comp.save(str(tmp_path / "c.loman"))

        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        index = manifest["nodes"][0]["value"]["index"]

        assert index["unit"] in {"s", "ms", "us", "ns"}


class TestStandardLibraryDatetimes:
    """Plain datetimes are unaffected by pandas resolution defaults."""

    def test_datetime_roundtrips(self, tmp_path):
        """A stdlib datetime is stored as ISO 8601 and returns unchanged."""
        value = datetime.datetime(2020, 1, 1, 12, 30, 15, 123456)

        comp = Computation()
        comp.add_node("dt", value=value)
        comp.save(str(tmp_path / "c.loman"))

        assert Computation.load(str(tmp_path / "c.loman")).v.dt == value
