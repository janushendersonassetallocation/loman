"""Tests for the ``.loman`` / ``.lm`` archive container.

Two things need covering that the JSON tests do not: that payloads round-trip
through a real zip, and that everything still works when pyarrow is absent.
CI installs all extras, so the no-parquet path is exercised by forcing
``use_parquet=False`` rather than by hoping for an environment without it.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from loman import ArchiveSerializer, Computation, ComputationSerializer, SerializationError, States
from loman.serialization import CustomTransformer
from loman.serialization.archive import (
    ARCHIVE_EXTENSIONS,
    MANIFEST_NAME,
    PAYLOAD_MARKER,
    has_parquet_support,
    is_archive_path,
)
from tests.format_fixtures import add_one, double

requires_parquet = pytest.mark.skipif(not has_parquet_support(), reason="pyarrow is not installed")

# Comfortably above the default inline threshold, so values become payloads.
BIG = 3000


def _big_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "px": rng.standard_normal(BIG).round(4),
            "qty": rng.integers(1, 1000, BIG),
            "t": pd.date_range("2024-01-01", periods=BIG, freq="min"),
        }
    )


def _roundtrip(comp: Computation, **kwargs) -> Computation:
    buf = io.BytesIO()
    comp.write_archive(buf, **kwargs)
    buf.seek(0)
    return Computation.read_archive(buf)


# ---------------------------------------------------------------------------
# Container shape
# ---------------------------------------------------------------------------


def test_archive_is_a_zip_with_a_manifest():
    """The container is an ordinary zip, inspectable without loman."""
    comp = Computation()
    comp.add_node("frame", value=_big_frame())
    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)

    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        assert MANIFEST_NAME in names
        assert any(n.startswith("payloads/") for n in names)
        manifest = json.loads(zf.read(MANIFEST_NAME))

    assert manifest["version"] >= 2
    assert [n["key"] for n in manifest["nodes"]] == ["frame"]


def test_manifest_keeps_structure_readable_without_the_payloads():
    """Graph shape stays greppable in the manifest, values do not bloat it."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("b", add_one, kwds={"x": "a"})
    comp.add_node("frame", value=_big_frame())
    comp.compute_all()

    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        manifest_bytes = zf.read(MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)

    # The manifest is small — the frame is not in it.
    assert len(manifest_bytes) < 2000
    assert {e["src"] for e in manifest["edges"]} == {"a"}
    frame_node = next(n for n in manifest["nodes"] if n["key"] == "frame")
    assert frame_node["value"][PAYLOAD_MARKER] is True


def test_small_values_stay_inline():
    """A graph of scalars produces no payload entries at all."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("b", value="two")
    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert zf.namelist() == [MANIFEST_NAME]


def test_inline_threshold_is_respected():
    """Lowering the threshold pushes even small values out of the manifest."""
    comp = Computation()
    comp.add_node("arr", value=np.arange(100, dtype="float64"))

    buf = io.BytesIO()
    comp.write_archive(buf, inline_threshold=10**9)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert zf.namelist() == [MANIFEST_NAME]

    buf = io.BytesIO()
    comp.write_archive(buf, inline_threshold=1)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert any(n.startswith("payloads/") for n in zf.namelist())


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_frame_roundtrips_exactly():
    """A large frame survives the payload path unchanged."""
    frame = _big_frame()
    comp = Computation()
    comp.add_node("frame", value=frame)
    pd.testing.assert_frame_equal(_roundtrip(comp).v.frame, frame)


def test_array_roundtrips_exactly():
    """A large array survives with its dtype and shape."""
    arr = np.random.default_rng(0).standard_normal((BIG, 4))
    comp = Computation()
    comp.add_node("arr", value=arr)
    got = _roundtrip(comp).v.arr
    assert got.dtype == arr.dtype
    np.testing.assert_array_equal(got, arr)


def test_series_roundtrips_with_name_and_index():
    """Series keep their name and index through the parquet detour."""
    series = pd.Series(
        np.random.default_rng(0).standard_normal(BIG),
        index=pd.date_range("2024-01-01", periods=BIG, freq="min"),
        name="prices",
    )
    comp = Computation()
    comp.add_node("s", value=series)
    pd.testing.assert_series_equal(_roundtrip(comp).v.s, series)


def test_unnamed_series_roundtrips():
    """A Series with no name comes back with no name, not a placeholder."""
    series = pd.Series(np.random.default_rng(0).standard_normal(BIG))
    comp = Computation()
    comp.add_node("s", value=series)
    assert _roundtrip(comp).v.s.name is None


def test_states_and_errors_survive():
    """The archive carries the same state information the JSON format does."""
    comp = Computation()
    comp.add_node("uptodate", value=1)
    comp.add_node("pinned", value=2)
    comp.pin("pinned")
    comp.add_node("frame", value=_big_frame())
    got = _roundtrip(comp)
    assert got.state("uptodate") == States.UPTODATE
    assert got.state("pinned") == States.PINNED
    assert got.state("frame") == States.UPTODATE


def test_functions_survive_and_the_graph_recomputes():
    """An archive round trip leaves a working, recomputable graph."""
    comp = Computation()
    comp.add_node("a", value=3)
    comp.add_node("b", add_one, kwds={"x": "a"})
    comp.add_node("c", double, kwds={"x": "b"})
    comp.compute_all()

    got = _roundtrip(comp)
    assert got.v.c == 8
    got.insert("a", 10)
    got.compute_all()
    assert got.v.c == 22


def test_duplicate_columns_fall_back_from_parquet():
    """Parquet rejects duplicate column names; the write still succeeds."""
    frame = pd.DataFrame(np.random.default_rng(0).standard_normal((BIG, 2)), columns=["a", "a"])
    comp = Computation()
    comp.add_node("frame", value=frame)
    pd.testing.assert_frame_equal(_roundtrip(comp).v.frame, frame)


def test_custom_transformers_work_inside_archives():
    """A registered custom type round-trips through a JSON payload."""

    class Point:
        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            return isinstance(other, Point) and other.x == self.x

    class PointTransformer(CustomTransformer):
        @property
        def name(self):
            return "point"

        def to_dict(self, transformer, o):
            return {"x": o.x}

        def from_dict(self, transformer, d):
            return Point(d["x"])

        @property
        def supported_direct_types(self):
            return [Point]

    serializer = ComputationSerializer()
    serializer._t.register(PointTransformer())

    comp = Computation()
    comp.add_node("points", value=np.array([Point(i) for i in range(BIG)], dtype=object))

    buf = io.BytesIO()
    comp.write_archive(buf, serializer=serializer)
    buf.seek(0)
    got = Computation.read_archive(buf, serializer=serializer)
    assert got.v.points[5] == Point(5)


def test_excluded_nodes_are_not_written():
    """serialize=False still means the value never reaches the file."""
    comp = Computation()
    comp.add_node("secret", value=_big_frame(), serialize=False)
    comp.add_node("kept", value=1)
    got = _roundtrip(comp)
    assert got.state("secret") == States.UNINITIALIZED
    assert got.v.kept == 1


# ---------------------------------------------------------------------------
# Partial reads
# ---------------------------------------------------------------------------


def test_partial_read_materialises_only_requested_nodes():
    """Unrequested payloads are not decoded."""
    comp = Computation()
    comp.add_node("small", value=42)
    comp.add_node("frame", value=_big_frame())
    comp.add_node("arr", value=np.random.default_rng(0).standard_normal((BIG, 3)))

    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)
    got = Computation.read_archive(buf, nodes=["small"])

    assert got.v.small == 42
    assert got.state("frame") == States.UNINITIALIZED
    assert got.state("arr") == States.UNINITIALIZED
    # The nodes themselves remain, so the graph is intact.
    assert {str(k) for k in got.dag.nodes} == {"small", "frame", "arr"}


def test_partial_read_can_select_a_payload_node():
    """Requesting a payload-backed node reads exactly that payload."""
    frame = _big_frame()
    comp = Computation()
    comp.add_node("frame", value=frame)
    comp.add_node("other", value=_big_frame())

    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)
    got = Computation.read_archive(buf, nodes=["frame"])

    pd.testing.assert_frame_equal(got.v.frame, frame)
    assert got.state("other") == States.UNINITIALIZED


def test_partial_read_of_a_missing_node_name_is_harmless():
    """Naming a node that is not in the file loads nothing extra, silently."""
    comp = Computation()
    comp.add_node("a", value=1)
    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)
    got = Computation.read_archive(buf, nodes=["nonexistent"])
    assert got.state("a") == States.UNINITIALIZED


# ---------------------------------------------------------------------------
# Working without pyarrow
# ---------------------------------------------------------------------------


def test_archive_works_without_parquet():
    """Frames fall back to JSON payloads; the archive still round-trips."""
    frame = _big_frame()
    comp = Computation()
    comp.add_node("frame", value=frame)

    buf = io.BytesIO()
    ArchiveSerializer(use_parquet=False).dump(comp, buf)
    buf.seek(0)
    got = ArchiveSerializer(use_parquet=False).load(buf)
    pd.testing.assert_frame_equal(got.v.frame, frame)

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert any(n.endswith(".json") for n in zf.namelist() if n.startswith("payloads/"))


def test_arrays_still_use_npy_without_parquet():
    """Arrays do not need pyarrow, so they keep their compact encoding."""
    comp = Computation()
    comp.add_node("arr", value=np.random.default_rng(0).standard_normal((BIG, 3)))
    buf = io.BytesIO()
    ArchiveSerializer(use_parquet=False).dump(comp, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert any(n.endswith(".npy") for n in zf.namelist())


def test_requesting_parquet_without_pyarrow_is_refused(monkeypatch):
    """Asking for parquet when it cannot work fails at construction."""
    monkeypatch.setattr("loman.serialization.archive.has_parquet_support", lambda: False)
    with pytest.raises(SerializationError, match=r"loman\[archive\]"):
        ArchiveSerializer(use_parquet=True)


@requires_parquet
def test_reading_a_parquet_payload_without_pyarrow_explains_itself(monkeypatch):
    """A file we cannot decode produces an actionable message, not a traceback."""
    comp = Computation()
    comp.add_node("frame", value=_big_frame())
    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)

    monkeypatch.setattr("loman.serialization.archive.has_parquet_support", lambda: False)
    with pytest.raises(SerializationError, match=r"needs pyarrow"):
        Computation.read_archive(buf)


# ---------------------------------------------------------------------------
# Extensions and dispatch
# ---------------------------------------------------------------------------


def test_both_extensions_are_recognised():
    """'.loman' and '.lm' name the same format, case-insensitively."""
    assert ARCHIVE_EXTENSIONS == (".loman", ".lm")
    assert is_archive_path("run.loman")
    assert is_archive_path("run.lm")
    assert is_archive_path("RUN.LOMAN")
    assert not is_archive_path("run.json")
    assert not is_archive_path("run.lmx")


@pytest.mark.parametrize("suffix", [".loman", ".lm"])
def test_write_dispatches_on_extension_to_an_archive(tmp_path, suffix):
    """A path with an archive extension writes a zip."""
    comp = Computation()
    comp.add_node("frame", value=_big_frame())
    path = tmp_path / f"run{suffix}"
    comp.write(str(path))

    assert zipfile.is_zipfile(path)
    got = Computation.read(str(path))
    pd.testing.assert_frame_equal(got.v.frame, comp.v.frame)


def test_write_dispatches_to_json_for_other_extensions(tmp_path):
    """Anything else stays a plain JSON document."""
    comp = Computation()
    comp.add_node("a", value=1)
    path = tmp_path / "run.json"
    comp.write(str(path))

    assert not zipfile.is_zipfile(path)
    with path.open(encoding="utf-8") as f:
        assert json.load(f)["version"] >= 2
    assert Computation.read(str(path)).v.a == 1


def test_read_dispatch_supports_partial_reads(tmp_path):
    """The convenience reader passes 'nodes' through to either format."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("frame", value=_big_frame())
    path = tmp_path / "run.lm"
    comp.write(str(path))

    got = Computation.read(str(path), nodes=["a"])
    assert got.v.a == 1
    assert got.state("frame") == States.UNINITIALIZED


def test_write_archive_accepts_a_path(tmp_path):
    """Paths and file objects behave the same."""
    comp = Computation()
    comp.add_node("frame", value=_big_frame())
    path = tmp_path / "run.loman"
    comp.write_archive(str(path))
    pd.testing.assert_frame_equal(Computation.read_archive(str(path)).v.frame, comp.v.frame)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_reading_a_non_zip_says_so():
    """A JSON document handed to read_archive gets a useful message."""
    comp = Computation()
    comp.add_node("a", value=1)
    text = io.StringIO()
    comp.write_json(text)
    with pytest.raises(SerializationError, match="not a zip container"):
        Computation.read_archive(io.BytesIO(text.getvalue().encode()))


def test_reading_a_zip_without_a_manifest_says_so():
    """A zip that is not a loman archive is rejected clearly."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("something.txt", "hello")
    buf.seek(0)
    with pytest.raises(SerializationError, match=r"no manifest\.json"):
        Computation.read_archive(buf)


def test_missing_payload_is_reported_against_its_manifest():
    """A truncated archive names the payload it could not find."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(
                {
                    "version": 2,
                    "edges": [],
                    "nodes": [
                        {
                            "key": "x",
                            "state": "UPTODATE",
                            "has_value": True,
                            "serialize": True,
                            "tags": [],
                            "func": None,
                            "value": {PAYLOAD_MARKER: True, "id": "p0", "encoding": "npy"},
                        }
                    ],
                }
            ),
        )
    buf.seek(0)
    with pytest.raises(SerializationError, match="missing payload"):
        Computation.read_archive(buf)


def test_unknown_payload_encoding_is_reported():
    """An encoding this build does not know is named rather than guessed at."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(
                {
                    "version": 2,
                    "edges": [],
                    "nodes": [
                        {
                            "key": "x",
                            "state": "UPTODATE",
                            "has_value": True,
                            "serialize": True,
                            "tags": [],
                            "func": None,
                            "value": {PAYLOAD_MARKER: True, "id": "p0", "encoding": "featherx"},
                        }
                    ],
                }
            ),
        )
        zf.writestr("payloads/p0", b"nonsense")
    buf.seek(0)
    with pytest.raises(SerializationError, match="unknown encoding"):
        Computation.read_archive(buf)


def test_archive_manifest_version_is_negotiated():
    """Archives get the same version checking JSON documents do."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps({"version": 999, "nodes": [], "edges": []}))
    buf.seek(0)
    with pytest.raises(SerializationError, match="Upgrade loman"):
        Computation.read_archive(buf)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_one_serializer_can_be_reused_and_shared():
    """No per-operation state lives on the serializer.

    Writer and reader are bound when each operation starts, so a single
    ArchiveSerializer can be reused — including from several threads — without
    two operations treading on each other.
    """
    import concurrent.futures

    serializer = ArchiveSerializer()
    frames = {f"f{i}": _big_frame().assign(tag=i) for i in range(8)}

    def roundtrip(name: str) -> pd.DataFrame:
        comp = Computation()
        comp.add_node(name, value=frames[name])
        buf = io.BytesIO()
        serializer.dump(comp, buf)
        buf.seek(0)
        return serializer.load(buf).v[name]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip(frames, pool.map(roundtrip, frames), strict=True))

    for name, frame in frames.items():
        pd.testing.assert_frame_equal(results[name], frame)


def test_payload_summary_describes_entries_without_decoding():
    """The summary reports sizes so a large file can be attributed to a node."""
    comp = Computation()
    comp.add_node("frame", value=_big_frame())
    buf = io.BytesIO()
    comp.write_archive(buf)
    buf.seek(0)

    summary = ArchiveSerializer().payload_summary(buf)
    assert list(summary.columns) == ["name", "size", "compressed"]
    assert MANIFEST_NAME in set(summary["name"])
    assert (summary["size"] > 0).all()


@requires_parquet
def test_archive_is_substantially_smaller_than_json():
    """The whole point: an archive of realistic data beats the JSON document.

    Realistic means repeated categorical strings and bounded-precision floats —
    the shape of data a scheduled job actually captures, and where a columnar
    format earns its keep.  Purely random float64 is the worst case for any
    encoding and still comes out ahead, just less dramatically.
    """
    rng = np.random.default_rng(0)
    n = 20000
    frame = pd.DataFrame(
        {
            "ticker": rng.choice(["AAPL", "MSFT", "GOOG", "AMZN"], n),
            "px": rng.standard_normal(n).round(2),
            "qty": rng.integers(1, 1000, n),
        }
    )
    comp = Computation()
    comp.add_node("frame", value=frame)

    archive = io.BytesIO()
    comp.write_archive(archive)
    text = io.StringIO()
    comp.write_json(text)

    ratio = len(text.getvalue().encode()) / len(archive.getvalue())
    assert ratio > 3, f"expected the archive to be >3x smaller, got {ratio:.1f}x"
