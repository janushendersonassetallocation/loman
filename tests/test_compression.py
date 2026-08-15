"""Compression, deduplication, checksums and the optional codecs.

The sampling heuristic is the interesting part. It exists because blanket
compression measured wrong in both directions on this codebase --- roughly 8x on
realistic data and 4% for three seconds on random floats --- so the tests here
assert that behaviour on both kinds of data, not just that compression runs.
"""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from loman import Computation
from loman.exception import SerializationError
from loman.serialization import SerializationProfile
from loman.serialization.blobs import MANIFEST_NAME
from loman.serialization.compression import (
    MIN_SAVING,
    compress_blob,
    decompress_blob,
    describe_available,
    get_codec,
    parse_spec,
    register_codec,
)

# A blob-sized threshold so test values do not have to be huge to exercise the
# out-of-line path.
SMALL_THRESHOLD = 1024


def _profile(**kwargs):
    """Return an efficient-style profile with the given settings."""
    kwargs.setdefault("inline_max_bytes", SMALL_THRESHOLD)
    return SerializationProfile(name="test", **kwargs)


def _manifest(path):
    """Return the manifest from a .loman archive."""
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(MANIFEST_NAME))


def _compressible_bytes():
    """Return bytes that compress well: a rounded random walk, as prices do."""
    values = np.round(100 + np.cumsum(np.random.default_rng(0).standard_normal(200_000) * 0.01), 2)
    return values.tobytes()


def _incompressible_bytes():
    """Return bytes that do not compress: raw random float64."""
    return np.random.default_rng(0).standard_normal(200_000).tobytes()


class TestAutoHeuristic:
    """``auto`` decides from the data, which is the whole point of sampling."""

    def test_compresses_compressible_data(self):
        """Realistic numeric data is compressed."""
        data = _compressible_bytes()

        stored, spec = compress_blob(data, "auto")

        assert spec != "none"
        assert len(stored) < len(data) * (1 - MIN_SAVING)

    def test_leaves_incompressible_data_alone(self):
        """Random floats are stored raw rather than burning time for nothing."""
        data = _incompressible_bytes()

        stored, spec = compress_blob(data, "auto")

        assert spec == "none"
        assert stored == data

    def test_roundtrips_either_way(self):
        """Whatever auto decides, the bytes come back identical."""
        for data in (_compressible_bytes(), _incompressible_bytes()):
            stored, spec = compress_blob(data, "auto")
            assert decompress_blob(stored, spec) == data

    def test_compressible_false_short_circuits(self):
        """A self-compressing payload is not compressed again."""
        data = _compressible_bytes()

        stored, spec = compress_blob(data, "auto", compressible=False)

        assert spec == "none"
        assert stored == data

    def test_empty_payload(self):
        """An empty payload is left alone rather than dividing by zero."""
        assert compress_blob(b"", "auto") == (b"", "none")

    def test_grown_payload_is_stored_raw(self):
        """If compression makes it bigger, the original is kept."""
        data = b"\x00"  # too short to shrink once headers are added
        stored, spec = compress_blob(data, "zlib:9")

        assert spec == "none"
        assert stored == data


class TestCodecs:
    """Named codecs, their levels, and the errors for unknown ones."""

    def test_zlib_roundtrip(self):
        """Zlib is always available, being in the standard library."""
        data = _compressible_bytes()
        stored, spec = compress_blob(data, "zlib:6")

        assert spec == "zlib:6"
        assert decompress_blob(stored, spec) == data

    def test_none_is_a_passthrough(self):
        """'none' stores and returns the bytes unchanged."""
        assert compress_blob(b"abc", "none") == (b"abc", "none")
        assert decompress_blob(b"abc", "none") == b"abc"

    def test_default_level_when_unspecified(self):
        """A family with no level uses its default."""
        assert parse_spec("zlib") == ("zlib", None)
        assert parse_spec("zlib:1") == ("zlib", 1)
        get_codec("zlib")  # must not raise

    def test_unknown_family_raises(self):
        """An unrecognised compression family raises rather than guessing."""
        with pytest.raises(ValueError, match="Unknown compression"):
            get_codec("brotli")

    def test_bad_level_raises(self):
        """A non-integer level raises."""
        with pytest.raises(ValueError, match="Invalid compression level"):
            parse_spec("zlib:high")

    def test_reading_an_unknown_compression_raises_clearly(self):
        """A file compressed with something unavailable says so."""
        with pytest.raises(SerializationError, match="Cannot read a blob compressed"):
            decompress_blob(b"data", "brotli:5")

    def test_custom_codec_can_be_registered(self):
        """A user can bring their own compression family.

        The codec here reverses the bytes before deflating them, purely so the
        output is distinguishable from plain zlib and the hook is demonstrably
        being used. It must actually shrink the payload, or the "stored raw
        because it grew" guard would discard it.
        """
        import bz2

        def factory(level):
            return (lambda d: bz2.compress(d, level), bz2.decompress)

        register_codec("bzip2", factory, default_level=9)
        try:
            data = _compressible_bytes()

            stored, spec = compress_blob(data, "bzip2:9")

            assert spec == "bzip2:9"
            assert len(stored) < len(data)
            assert decompress_blob(stored, spec) == data
        finally:
            from loman.serialization import compression

            compression._FAMILIES.pop("bzip2", None)
            compression._DEFAULT_LEVELS.pop("bzip2", None)

    def test_describe_available_reports_families(self):
        """Availability can be queried without importing optional packages."""
        available = describe_available()

        assert available["zlib"] is True
        assert "zstd" in available


class TestCompressionThroughSave:
    """Compression as observed through an actual saved container."""

    def test_default_profile_compresses_realistic_data(self, tmp_path):
        """Saving a realistic series records that it was compressed."""
        values = np.round(100 + np.cumsum(np.random.default_rng(0).standard_normal(200_000) * 0.01), 2)
        comp = Computation()
        comp.add_node("px", value=values)
        path = tmp_path / "c.loman"

        comp.save(str(path))

        entry = _manifest(path)["blobs"][0]
        assert entry["compression"] != "none"
        assert entry["stored_size"] < entry["size"] / 2
        assert np.array_equal(Computation.load(str(path)).v.px, values)

    def test_default_profile_skips_incompressible_data(self, tmp_path):
        """Random floats are stored uncompressed."""
        values = np.random.default_rng(0).standard_normal(200_000)
        comp = Computation()
        comp.add_node("r", value=values)
        path = tmp_path / "c.loman"

        comp.save(str(path))

        entry = _manifest(path)["blobs"][0]
        assert entry["compression"] == "none"
        assert "stored_size" not in entry
        assert np.array_equal(Computation.load(str(path)).v.r, values)

    def test_explicit_compression_is_honoured(self, tmp_path):
        """A named compression is used as given, without probing."""
        comp = Computation()
        comp.add_node("r", value=np.random.default_rng(0).standard_normal(200_000))
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(compression="zlib:9"))

        assert _manifest(path)["blobs"][0]["compression"] == "zlib:9"

    def test_compression_none_is_honoured(self, tmp_path):
        """Compression can be turned off entirely."""
        values = np.round(np.arange(200_000, dtype="float64"), 2)
        comp = Computation()
        comp.add_node("px", value=values)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(compression="none"))

        assert _manifest(path)["blobs"][0]["compression"] == "none"

    def test_compressed_blobs_are_still_zip_stored(self, tmp_path):
        """The container never deflates on top of the blob layer's work."""
        values = np.round(np.arange(200_000, dtype="float64"), 2)
        comp = Computation()
        comp.add_node("px", value=values)
        path = tmp_path / "c.loman"

        comp.save(str(path))

        with zipfile.ZipFile(path) as zf:
            blobs = [i for i in zf.infolist() if i.filename != MANIFEST_NAME]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in blobs)

    def test_dir_container_compresses_too(self, tmp_path):
        """Compression is a blob-layer decision, so it applies to both containers."""
        values = np.round(np.arange(200_000, dtype="float64"), 2)
        comp = Computation()
        comp.add_node("px", value=values)
        path = tmp_path / "c_dir"

        comp.save(str(path), container="dir")

        manifest = json.loads((path / MANIFEST_NAME).read_text())
        assert manifest["blobs"][0]["compression"] != "none"
        assert np.array_equal(Computation.load(str(path)).v.px, values)


class TestDeduplication:
    """Two nodes holding the same object share one blob."""

    def test_identical_objects_share_a_blob(self, tmp_path):
        """The same array on two nodes is stored once."""
        shared = np.arange(50_000, dtype="float64")
        comp = Computation()
        comp.add_node("a", value=shared)
        comp.add_node("b", value=shared)
        path = tmp_path / "c.loman"

        comp.save(str(path))

        assert len(_manifest(path)["blobs"]) == 1
        restored = Computation.load(str(path))
        assert np.array_equal(restored.v.a, shared)
        assert np.array_equal(restored.v.b, shared)

    def test_equal_but_distinct_objects_are_not_deduped_by_identity(self, tmp_path):
        """Identity dedup does not hash, so two equal arrays are stored twice."""
        comp = Computation()
        comp.add_node("a", value=np.arange(50_000, dtype="float64"))
        comp.add_node("b", value=np.arange(50_000, dtype="float64"))
        path = tmp_path / "c.loman"

        comp.save(str(path))

        assert len(_manifest(path)["blobs"]) == 2

    def test_content_dedup_catches_equal_objects(self, tmp_path):
        """Content dedup hashes, so equal-but-distinct arrays share a blob."""
        comp = Computation()
        comp.add_node("a", value=np.arange(50_000, dtype="float64"))
        comp.add_node("b", value=np.arange(50_000, dtype="float64"))
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(dedupe="content"))

        assert len(_manifest(path)["blobs"]) == 1
        restored = Computation.load(str(path))
        assert np.array_equal(restored.v.a, restored.v.b)

    def test_dedup_can_be_disabled(self, tmp_path):
        """Turning dedup off stores every blob separately."""
        shared = np.arange(50_000, dtype="float64")
        comp = Computation()
        comp.add_node("a", value=shared)
        comp.add_node("b", value=shared)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(dedupe="none"))

        assert len(_manifest(path)["blobs"]) == 2


class TestChecksums:
    """Optional per-blob digests."""

    def test_off_by_default(self, tmp_path):
        """No digest is recorded unless asked for."""
        comp = Computation()
        comp.add_node("a", value=np.arange(50_000, dtype="float64"))
        path = tmp_path / "c.loman"

        comp.save(str(path))

        assert "sha256" not in _manifest(path)["blobs"][0]

    def test_recorded_when_enabled(self, tmp_path):
        """A digest of the stored bytes is recorded."""
        import hashlib

        comp = Computation()
        comp.add_node("a", value=np.arange(50_000, dtype="float64"))
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(checksums=True))

        entry = _manifest(path)["blobs"][0]
        with zipfile.ZipFile(path) as zf:
            stored = zf.read(entry["path"])
        assert entry["sha256"] == hashlib.sha256(stored).hexdigest()


zstandard = pytest.importorskip("zstandard", reason="zstd needs the 'efficient' extra")


class TestZstd:
    """The zstd codec, when the efficient extra is installed."""

    def test_roundtrip(self):
        """Bytes survive a zstd round-trip."""
        data = _compressible_bytes()
        stored, spec = compress_blob(data, "zstd:9")

        assert spec == "zstd:9"
        assert decompress_blob(stored, spec) == data

    def test_through_a_save(self, tmp_path):
        """A container can be written with zstd blobs."""
        values = np.round(np.arange(200_000, dtype="float64"), 2)
        comp = Computation()
        comp.add_node("px", value=values)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_profile(compression="zstd:9"))

        assert _manifest(path)["blobs"][0]["compression"] == "zstd:9"
        assert np.array_equal(Computation.load(str(path)).v.px, values)


pyarrow = pytest.importorskip("pyarrow", reason="parquet needs the 'efficient' extra")


def _parquet_profile():
    """Return a profile that stores frames as parquet."""
    return _profile(compression="auto", frame_encoding="parquet")


PARQUET_FRAMES = {
    "tz_index": pd.DataFrame(
        {"a": np.arange(20_000, dtype="float64"), "s": ["x"] * 20_000},
        index=pd.date_range("2020", periods=20_000, freq="min", tz="UTC"),
    ),
    "multiindex": pd.DataFrame(
        {"a": np.arange(20_000, dtype="float64")},
        index=pd.MultiIndex.from_arrays([np.arange(20_000) // 100, np.arange(20_000) % 100]),
    ),
    "categorical": pd.DataFrame(
        {"c": pd.Categorical(["a", "b", "c", "d"] * 5_000), "v": np.arange(20_000, dtype="float64")}
    ),
    "mixed_dtypes": pd.DataFrame(
        {"i": np.arange(20_000), "f": np.arange(20_000, dtype="float64"), "s": ["a"] * 20_000}
    ),
    "non_finite": pd.DataFrame({"a": [np.nan, np.inf, -np.inf, *([1.0] * 19_997)]}),
}


class TestParquet:
    """Frames stored as parquet, when the efficient extra is installed."""

    @pytest.mark.parametrize("label", sorted(PARQUET_FRAMES))
    def test_roundtrip_is_exact(self, tmp_path, label):
        """Every frame comes back equal, with its index and dtypes intact."""
        frame = PARQUET_FRAMES[label]
        comp = Computation()
        comp.add_node("frame", value=frame)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        assert Computation.load(str(path)).v.frame.equals(frame)

    def test_parquet_is_actually_used(self, tmp_path):
        """The manifest records the parquet encoding, not the column-wise one."""
        comp = Computation()
        comp.add_node("frame", value=PARQUET_FRAMES["mixed_dtypes"])
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        manifest = _manifest(path)
        assert manifest["nodes"][0]["value"]["encoding"] == "parquet"
        assert manifest["blobs"][0]["codec"] == "parquet"

    def test_parquet_blobs_are_not_compressed_again(self, tmp_path):
        """Parquet compresses itself, so the blob layer leaves it alone."""
        comp = Computation()
        comp.add_node("frame", value=PARQUET_FRAMES["mixed_dtypes"])
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        assert _manifest(path)["blobs"][0]["compression"] == "none"

    def test_unrepresentable_frame_falls_back(self, tmp_path):
        """A frame pyarrow cannot take is saved the other way, not refused.

        Duplicate column names are the easy example. Falling back matters more
        than the encoding does: a save must not fail because an optional codec
        has a limitation.
        """
        frame = pd.DataFrame(np.arange(40_000, dtype="float64").reshape(20_000, 2), columns=["a", "a"])
        comp = Computation()
        comp.add_node("frame", value=frame)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        assert _manifest(path)["nodes"][0]["value"].get("encoding") != "parquet"
        assert Computation.load(str(path)).v.frame.equals(frame)

    def test_small_frames_stay_inline(self, tmp_path):
        """Below the threshold, parquet is not used even when selected."""
        comp = Computation()
        comp.add_node("frame", value=pd.DataFrame({"a": [1.0, 2.0]}))
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        assert _manifest(path)["blobs"] == []

    def test_npy_is_the_default(self, tmp_path):
        """Without asking for parquet, frames use the npy column path."""
        comp = Computation()
        comp.add_node("frame", value=PARQUET_FRAMES["mixed_dtypes"])
        path = tmp_path / "c.loman"

        comp.save(str(path))

        assert _manifest(path)["nodes"][0]["value"].get("encoding") != "parquet"
        assert all(entry["codec"] == "npy" for entry in _manifest(path)["blobs"])


class TestOptionalDependencyFallback:
    """The no-pyarrow path must be exercised even where pyarrow installs.

    Otherwise the fallback has coverage only on machines that cannot install the
    extra --- which is exactly where nobody is watching the test output.
    """

    def test_frame_falls_back_without_pyarrow(self, tmp_path, monkeypatch):
        """With pyarrow unavailable, a parquet profile still saves, as npy."""
        import loman._extras

        def _no_extras(module, extra):
            msg = f"'{module}' is required for loman's '{extra}' extra."
            raise ImportError(msg)

        monkeypatch.setattr(loman._extras, "require", _no_extras)

        frame = PARQUET_FRAMES["mixed_dtypes"]
        comp = Computation()
        comp.add_node("frame", value=frame)
        path = tmp_path / "c.loman"

        comp.save(str(path), profile=_parquet_profile())

        assert _manifest(path)["nodes"][0]["value"].get("encoding") != "parquet"
        assert Computation.load(str(path)).v.frame.equals(frame)

    def test_zstd_unavailable_raises_a_useful_error(self, monkeypatch):
        """Asking for zstd without the extra explains how to install it."""
        import loman._extras

        def _no_extras(module, extra):
            msg = f"'{module}' is required for loman's '{extra}' extra."
            raise ImportError(msg)

        monkeypatch.setattr(loman._extras, "require", _no_extras)

        with pytest.raises(ImportError, match="efficient"):
            compress_blob(b"x" * 10_000, "zstd:3")


class TestDedupeIdentityReuse:
    """Regression: identity dedup must not confuse two short-lived temporaries.

    A frame's columns are encoded via ``column.to_numpy()``, which builds a new
    array each time. If the store keys dedup on ``id()`` without holding a
    reference, each temporary is collected as soon as it is written and CPython
    hands the same id to the next one --- so column 2 is deduplicated onto
    column 0's blob and the frame reloads with repeated columns. Silently.
    """

    def test_wide_frame_columns_stay_distinct(self, tmp_path):
        """Every column of a wide frame survives as itself."""
        frame = pd.DataFrame(
            np.random.default_rng(0).standard_normal((5_000, 12)),
            columns=[f"c{i}" for i in range(12)],
        )
        comp = Computation()
        comp.add_node("frame", value=frame)
        path = tmp_path / "wide.loman"

        comp.save(str(path))

        restored = Computation.load(str(path)).v.frame
        assert restored.equals(frame)
        # Distinct data must not collapse onto a shared blob.
        assert len(_manifest(path)["blobs"]) == 12

    def test_many_separate_arrays_stay_distinct(self, tmp_path):
        """The same hazard across nodes rather than columns."""
        rng = np.random.default_rng(0)
        arrays = {f"n{i}": rng.standard_normal(5_000) for i in range(10)}
        comp = Computation()
        for name, array in arrays.items():
            comp.add_node(name, value=array)
        path = tmp_path / "many.loman"

        comp.save(str(path))

        restored = Computation.load(str(path))
        for name, array in arrays.items():
            assert np.array_equal(getattr(restored.v, name), array), name
