"""Container tests: the .loman archive, the directory form, and save/load.

The profile/container matrix is the point of this module. A value that survives
the readable single-document path can still be broken by the blob path, and vice
versa, so the interesting values are asserted across every valid combination
rather than once.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loman import Computation
from loman.exception import SerializationError
from loman.serialization.blobs import MANIFEST_NAME, _validate_member_path
from loman.serialization.computation import (
    FORMAT_VERSION,
    ComputationSerializer,
    infer_container_from_path,
    sniff_container,
)
from loman.serialization.profile import EFFICIENT, READABLE, SerializationProfile, resolve_profile

# (profile, container) combinations that are valid. efficient+json is not, since
# a single JSON document has nowhere to put out-of-line bytes.
VALID_COMBINATIONS = [
    ("readable", "json"),
    ("readable", "zip"),
    ("efficient", "zip"),
    ("efficient", "dir"),
    ("readable", "dir"),
]


def _add_one(a):
    """Module-level function so it round-trips as an importable reference."""
    return a + 1


def _sample_computation():
    """Return a computation covering the awkward corners of the value model."""
    comp = Computation()
    comp.add_node("scalar", value=1)
    comp.add_node("a", value=2)
    comp.add_node("b", _add_one)
    comp.add_node(
        "frame",
        value=pd.DataFrame(
            {"x": np.arange(5000, dtype="float64"), "s": ["a"] * 5000},
            index=pd.date_range("2020-01-01", periods=5000, freq="min", tz="UTC"),
        ),
    )
    comp.add_node("array", value=np.arange(5000, dtype="float64"))
    comp.add_node("small_array", value=np.array([1.0, 2.0]))
    comp.add_node("nan", value=float("nan"))
    comp.add_node("odd_dict", value={1: "a", "type": "not-metadata"})
    comp.add_node("multi", value=pd.DataFrame({"a": [1, 2]}, index=pd.MultiIndex.from_tuples([(1, "x"), (2, "y")])))
    comp.compute_all()
    return comp


def _target(tmp_path: Path, container: str) -> Path:
    """Return a destination path appropriate to *container*."""
    return {"json": tmp_path / "c.json", "zip": tmp_path / "c.loman", "dir": tmp_path / "c_dir"}[container]


@pytest.mark.parametrize(("profile", "container"), VALID_COMBINATIONS)
class TestProfileContainerMatrix:
    """Every valid profile/container combination round-trips identically."""

    def test_values_survive(self, tmp_path, profile, container):
        """Values come back equal regardless of how they were stored."""
        comp = _sample_computation()
        path = _target(tmp_path, container)

        comp.save(str(path), profile=profile, container=container)
        restored = Computation.load(str(path))

        assert restored.v.scalar == 1
        assert restored.v.b == 3
        assert restored.v.frame.equals(comp.v.frame)
        assert np.array_equal(restored.v.array, comp.v.array)
        assert np.array_equal(restored.v.small_array, comp.v.small_array)
        assert np.isnan(restored.v.nan)
        assert restored.v.odd_dict == {1: "a", "type": "not-metadata"}
        assert restored.v.multi.equals(comp.v.multi)

    def test_structure_survives(self, tmp_path, profile, container):
        """Edges, states and functions come back intact."""
        comp = _sample_computation()
        path = _target(tmp_path, container)

        comp.save(str(path), profile=profile, container=container)
        restored = Computation.load(str(path))

        assert set(restored.dag.edges()) == set(comp.dag.edges())
        restored.insert("a", 10)
        restored.compute_all()
        assert restored.v.b == 11

    def test_manifest_is_strict_json(self, tmp_path, profile, container):
        """The manifest never contains bare NaN or Infinity tokens."""

        def _reject(token):
            msg = f"non-standard JSON token: {token}"
            raise AssertionError(msg)

        comp = _sample_computation()
        path = _target(tmp_path, container)
        comp.save(str(path), profile=profile, container=container)

        json.loads(_read_manifest_text(path, container), parse_constant=_reject)


def _read_manifest_text(path: Path, container: str) -> str:
    """Return the manifest text from any container."""
    if container == "json":
        return path.read_text(encoding="utf-8")
    if container == "dir":
        return (path / MANIFEST_NAME).read_text(encoding="utf-8")
    with zipfile.ZipFile(path) as zf:
        return zf.read(MANIFEST_NAME).decode("utf-8")


class TestOneSpecTwoSerializations:
    """The zip and directory containers are the same layout, written twice."""

    def test_manifests_are_byte_identical(self, tmp_path):
        """A zip and a directory save of one computation produce the same manifest.

        This is the "one spec, not two" property. Asserting it here means it
        cannot quietly stop being true.
        """
        comp = _sample_computation()

        comp.save(str(tmp_path / "a.loman"), container="zip")
        comp.save(str(tmp_path / "a_dir"), container="dir")

        zip_manifest = json.loads(_read_manifest_text(tmp_path / "a.loman", "zip"))
        dir_manifest = json.loads(_read_manifest_text(tmp_path / "a_dir", "dir"))

        # The container field is the one legitimate difference.
        assert zip_manifest.pop("container") == "zip"
        assert dir_manifest.pop("container") == "dir"
        assert json.dumps(zip_manifest) == json.dumps(dir_manifest)

    def test_blob_bytes_are_identical(self, tmp_path):
        """The same blobs, byte for byte, land in both containers."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "a.loman"), container="zip")
        comp.save(str(tmp_path / "a_dir"), container="dir")

        with zipfile.ZipFile(tmp_path / "a.loman") as zf:
            zip_blobs = {n: zf.read(n) for n in zf.namelist() if n != MANIFEST_NAME}
        # as_posix, not str: a blob key is "blobs/0000.npy" in the manifest on
        # every platform, but relative_to gives it back with a backslash on
        # Windows. The separator is the test's, not the format's.
        dir_blobs = {
            p.relative_to(tmp_path / "a_dir").as_posix(): p.read_bytes()
            for p in (tmp_path / "a_dir" / "blobs").iterdir()
        }

        assert zip_blobs == dir_blobs


class TestDeterminism:
    """Two saves of the same computation produce the same bytes."""

    def test_zip_save_is_reproducible(self, tmp_path):
        """Byte-identical archives, which is what makes saves comparable.

        Zip records a modification time per member and defaults it to "now", so
        this only holds because the writer stamps a fixed timestamp.
        """
        comp = _sample_computation()

        comp.save(str(tmp_path / "one.loman"))
        comp.save(str(tmp_path / "two.loman"))

        assert (tmp_path / "one.loman").read_bytes() == (tmp_path / "two.loman").read_bytes()

    def test_json_save_is_reproducible(self, tmp_path):
        """The single document is reproducible too."""
        comp = _sample_computation()

        comp.save(str(tmp_path / "one.json"))
        comp.save(str(tmp_path / "two.json"))

        assert (tmp_path / "one.json").read_text() == (tmp_path / "two.json").read_text()


class TestBlobTable:
    """Invariants the blob table must hold."""

    def test_every_reference_resolves(self, tmp_path):
        """No encoded blob reference points at a blob that is not listed."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))
        listed = {entry["id"] for entry in manifest["blobs"]}

        assert _collect_blob_refs(manifest["nodes"]) <= listed

    def test_no_orphan_blobs(self, tmp_path):
        """Every stored blob is referenced by something."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))
        listed = {entry["id"] for entry in manifest["blobs"]}

        assert listed == _collect_blob_refs(manifest["nodes"])

    def test_ids_are_unique(self, tmp_path):
        """Blob ids are not reused."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))
        ids = [entry["id"] for entry in manifest["blobs"]]

        assert len(ids) == len(set(ids))

    def test_every_listed_member_exists(self, tmp_path):
        """Each blob path in the table is a real member of the archive."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            members = set(zf.namelist())
            manifest = json.loads(zf.read(MANIFEST_NAME))

        assert {entry["path"] for entry in manifest["blobs"]} <= members

    def test_blobs_are_stored_uncompressed(self, tmp_path):
        """Blob members use ZIP_STORED, keeping them at known offsets."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            blob_infos = [i for i in zf.infolist() if i.filename != MANIFEST_NAME]
            manifest_info = zf.getinfo(MANIFEST_NAME)

        assert blob_infos, "expected at least one blob"
        assert all(i.compress_type == zipfile.ZIP_STORED for i in blob_infos)
        assert manifest_info.compress_type == zipfile.ZIP_DEFLATED

    def test_blob_table_names_the_node(self, tmp_path):
        """Each blob records which node it came from, for orientation."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))

        assert {"frame", "array"} <= {entry["node"] for entry in manifest["blobs"]}


def _collect_blob_refs(obj) -> set:
    """Return every blob id referenced anywhere within *obj*."""
    found = set()
    if isinstance(obj, dict):
        if "$blob" in obj and isinstance(obj["$blob"], int):
            found.add(obj["$blob"])
        for value in obj.values():
            found |= _collect_blob_refs(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= _collect_blob_refs(item)
    return found


class TestManifestReadability:
    """The efficient profile still describes every value in the manifest."""

    def test_shape_and_dtype_stay_inline(self, tmp_path):
        """A blob-backed array's shape and dtype are readable without decoding."""
        comp = Computation()
        comp.add_node("array", value=np.arange(10_000, dtype="float64"))
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))
        value = next(n["value"] for n in manifest["nodes"] if n["key"] == "array")

        assert value["shape"] == [10_000]
        assert value["dtype"] == "<f8"
        assert value["encoding"] == "npy"

    def test_manifest_stays_small(self, tmp_path):
        """A large frame does not put its data in the manifest."""
        n = 50_000
        comp = Computation()
        comp.add_node(
            "frame",
            value=pd.DataFrame(
                np.random.default_rng(0).standard_normal((n, 5)),
                index=pd.date_range("2020", periods=n, freq="min"),
            ),
        )
        comp.save(str(tmp_path / "c.loman"))

        manifest_bytes = len(_read_manifest_text(tmp_path / "c.loman", "zip"))

        assert manifest_bytes < 4096, f"manifest grew to {manifest_bytes} bytes"

    def test_small_values_stay_inline(self, tmp_path):
        """Below the threshold, values stay in the manifest and no blob is made."""
        comp = Computation()
        comp.add_node("small", value=np.array([1.0, 2.0, 3.0]))
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))

        assert manifest["blobs"] == []
        assert manifest["nodes"][0]["value"]["data"] == [1.0, 2.0, 3.0]


class TestContainerSelection:
    """Inference, explicit selection, and the one invalid combination."""

    def test_json_suffix_infers_json_container(self, tmp_path):
        """A .json path writes a single document."""
        assert infer_container_from_path(tmp_path / "x.json") == "json"

    def test_other_suffixes_infer_zip(self, tmp_path):
        """Anything else defaults to the .loman archive."""
        assert infer_container_from_path(tmp_path / "x.loman") == "zip"
        assert infer_container_from_path(tmp_path / "x") == "zip"

    def test_efficient_json_is_refused(self, tmp_path):
        """The efficient profile cannot be used with a single JSON document."""
        comp = _sample_computation()

        with pytest.raises(ValueError, match="nowhere to put them"):
            comp.save(str(tmp_path / "c.json"), profile="efficient")

    def test_error_names_the_way_out(self, tmp_path):
        """The refusal says what to do instead."""
        comp = _sample_computation()

        with pytest.raises(ValueError, match="container='zip'"):
            comp.save(str(tmp_path / "c.json"), profile="efficient")

    def test_unknown_container_is_refused(self, tmp_path):
        """An unrecognised container name raises rather than guessing."""
        comp = _sample_computation()

        with pytest.raises(ValueError, match="Unknown container"):
            comp.save(str(tmp_path / "c.loman"), container="tarball")

    def test_unknown_profile_is_refused(self):
        """An unrecognised profile name raises."""
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_profile("compact")

    def test_default_profile_is_efficient(self):
        """Saving without a profile picks the efficient one."""
        assert resolve_profile(None) is EFFICIENT

    def test_profile_instance_passes_through(self):
        """A profile instance is used as given."""
        profile = SerializationProfile(name="custom", inline_max_bytes=1)
        assert resolve_profile(profile) is profile


class TestSniffing:
    """A saved computation is identified by its contents, not its name."""

    def test_zip_is_detected(self, tmp_path):
        """A .loman archive is recognised by its magic number."""
        comp = _sample_computation()
        path = tmp_path / "named_anything"
        comp.save(str(path), container="zip")

        assert sniff_container(path) == "zip"

    def test_dir_is_detected(self, tmp_path):
        """A container directory is recognised."""
        comp = _sample_computation()
        path = tmp_path / "c_dir"
        comp.save(str(path), container="dir")

        assert sniff_container(path) == "dir"

    def test_json_is_detected(self, tmp_path):
        """A single document is recognised by its leading brace."""
        comp = _sample_computation()
        path = tmp_path / "c.json"
        comp.save(str(path), container="json")

        assert sniff_container(path) == "json"

    def test_dill_file_names_read_dill(self, tmp_path):
        """A write_dill file gets a useful message, not a JSON parse error."""
        import dill  # nosec B403

        path = tmp_path / "c.dill"
        with path.open("wb") as f:
            dill.dump({"not": "a computation"}, f)

        with pytest.raises(SerializationError, match="read_dill"):
            sniff_container(path)

    def test_unrecognised_file_is_refused(self, tmp_path):
        """Something else entirely raises, listing the accepted forms."""
        path = tmp_path / "c.txt"
        path.write_text("just some text")

        with pytest.raises(SerializationError, match=r"Expected a \.loman archive"):
            sniff_container(path)

    def test_missing_path_raises(self, tmp_path):
        """A path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            sniff_container(tmp_path / "nope.loman")

    def test_load_reads_any_container(self, tmp_path):
        """One load call handles all three containers."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "a.loman"), container="zip")
        comp.save(str(tmp_path / "a_dir"), container="dir")
        comp.save(str(tmp_path / "a.json"), container="json")

        for name in ("a.loman", "a_dir", "a.json"):
            assert Computation.load(str(tmp_path / name)).v.b == 3


class TestContainerErrors:
    """Malformed or hostile containers are refused clearly."""

    def test_zip_without_manifest(self, tmp_path):
        """A zip that is not a loman container says so."""
        path = tmp_path / "plain.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("hello.txt", "hi")

        with pytest.raises(SerializationError, match=f"no {MANIFEST_NAME}"):
            Computation.load(str(path))

    def test_directory_without_manifest(self, tmp_path):
        """A directory that is not a loman container says so."""
        path = tmp_path / "empty_dir"
        path.mkdir()

        with pytest.raises(SerializationError, match=f"has no {MANIFEST_NAME}"):
            Computation.load(str(path))

    def test_absolute_blob_path_is_refused(self):
        """A manifest cannot point a blob at an absolute path."""
        with pytest.raises(SerializationError, match="outside the container"):
            _validate_member_path("/etc/passwd")

    def test_traversing_blob_path_is_refused(self):
        """A manifest cannot climb out of the container with '..'."""
        with pytest.raises(SerializationError, match="outside the container"):
            _validate_member_path("../../etc/passwd")

    def test_empty_blob_path_is_refused(self):
        """An empty blob path is refused rather than resolving to the root."""
        with pytest.raises(SerializationError, match="empty"):
            _validate_member_path("")

    def test_ordinary_blob_path_is_accepted(self):
        """A normal blob path passes validation."""
        assert _validate_member_path("blobs/0000.npy") == "blobs/0000.npy"

    def test_dangling_blob_reference(self, tmp_path):
        """A reference to a blob the manifest does not list is refused."""
        comp = Computation()
        comp.add_node("array", value=np.arange(10_000, dtype="float64"))
        path = tmp_path / "c_dir"
        comp.save(str(path), container="dir")

        manifest_path = path / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["blobs"] = []
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(SerializationError, match="does not resolve"):
            Computation.load(str(path))

    def test_dir_container_over_a_file(self, tmp_path):
        """Writing a directory container onto an existing file is refused."""
        path = tmp_path / "occupied"
        path.write_text("existing file")
        comp = _sample_computation()

        with pytest.raises(SerializationError, match="over the existing file"):
            comp.save(str(path), container="dir")


class TestSaveOverwrites:
    """Saving over an existing container replaces it cleanly."""

    def test_zip_is_replaced(self, tmp_path):
        """A second save leaves no trace of the first."""
        path = tmp_path / "c.loman"
        comp = _sample_computation()
        comp.save(str(path))

        smaller = Computation()
        smaller.add_node("only", value=1)
        smaller.save(str(path))

        with zipfile.ZipFile(path) as zf:
            assert zf.namelist() == [MANIFEST_NAME]
        assert Computation.load(str(path)).v.only == 1

    def test_dir_blobs_are_replaced(self, tmp_path):
        """Stale blobs from a previous save are cleared."""
        path = tmp_path / "c_dir"
        comp = _sample_computation()
        comp.save(str(path), container="dir")
        assert list((path / "blobs").iterdir())

        smaller = Computation()
        smaller.add_node("only", value=1)
        smaller.save(str(path), container="dir")

        assert not (path / "blobs").exists() or not list((path / "blobs").iterdir())
        assert Computation.load(str(path)).v.only == 1


class TestManifestShape:
    """Fields the manifest is expected to carry."""

    def test_records_version_container_and_profile(self, tmp_path):
        """The manifest says how it was written."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))

        assert manifest["version"] == FORMAT_VERSION
        assert manifest["container"] == "zip"
        assert manifest["profile"] == "efficient"

    def test_readable_profile_writes_no_blobs(self, tmp_path):
        """The readable profile keeps everything in the manifest."""
        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"), profile="readable")

        manifest = json.loads(_read_manifest_text(tmp_path / "c.loman", "zip"))

        assert manifest["blobs"] == []
        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            assert zf.namelist() == [MANIFEST_NAME]

    def test_dumps_is_always_a_readable_document(self):
        """dumps() has nowhere to put blobs, so it never makes any."""
        comp = _sample_computation()

        manifest = json.loads(ComputationSerializer().dumps(comp))

        assert manifest["container"] == "json"
        assert manifest["profile"] == READABLE.name
        assert manifest["blobs"] == []


class TestAllowCodeThroughContainers:
    """allow_code=False holds for container loads too, not just documents."""

    def test_function_is_not_restored(self, tmp_path):
        """A .loman archive loaded without code has no node functions."""
        from loman.consts import NodeAttributes
        from loman.nodekey import parse_nodekey

        comp = _sample_computation()
        comp.save(str(tmp_path / "c.loman"))

        restored = Computation.load(str(tmp_path / "c.loman"), allow_code=False)

        assert restored.dag.nodes[parse_nodekey("b")][NodeAttributes.FUNC] is None
        assert restored.v.b == 3


class TestProfileOverrides:
    """Per-node override selectors."""

    def test_glob_selector_matches(self):
        """A node-key glob selects matching nodes."""
        profile = SerializationProfile(name="p", overrides={"market/*": {"codec": "parquet"}})

        assert profile.settings_for("market/prices") == {"codec": "parquet"}
        assert profile.settings_for("other") == {}

    def test_tag_selector_matches(self):
        """A tag selector selects nodes carrying that tag."""
        profile = SerializationProfile(name="p", overrides={"tag:raw": {"compression": "none"}})

        assert profile.settings_for("anything", frozenset({"raw"})) == {"compression": "none"}
        assert profile.settings_for("anything", frozenset({"cooked"})) == {}

    def test_later_selectors_win(self):
        """A more specific selector listed later overrides an earlier one."""
        profile = SerializationProfile(
            name="p",
            overrides={"*": {"codec": "npy"}, "market/*": {"codec": "parquet"}},
        )

        assert profile.settings_for("market/prices")["codec"] == "parquet"

    def test_no_node_matches_only_tags(self):
        """A blob with no owning node still matches tag selectors only."""
        profile = SerializationProfile(name="p", overrides={"*": {"codec": "npy"}})

        assert profile.settings_for(None) == {}


class TestThresholds:
    """When a value goes out of line."""

    def test_readable_profile_never_wants_blobs(self):
        """The readable profile keeps everything inline, however large."""
        assert READABLE.wants_blob(10**9) is False

    def test_efficient_profile_respects_the_threshold(self):
        """Small values stay inline; large ones do not."""
        assert EFFICIENT.wants_blob(10) is False
        assert EFFICIENT.wants_blob(10**7) is True

    def test_unknown_size_goes_out_of_line(self):
        """A transformer that cannot estimate its size is taken at its word."""
        assert EFFICIENT.wants_blob(None) is True


class TestNoStoreAvailable:
    """The single-document container has no blob storage of its own."""

    def test_writer_with_no_stores_accepts_nothing(self):
        """A writer with no stores reports that it cannot hold blobs."""
        from loman.serialization.blobs import BlobWriter

        writer = BlobWriter({})

        assert writer.accepts_blobs is False
        assert writer.can_store(None) is False
        assert writer.table() == []

    def test_put_names_the_missing_store(self):
        """Asking for a store that was not supplied says which one, and how."""
        from loman.serialization.blobs import BlobWriter

        writer = BlobWriter({})

        with pytest.raises(SerializationError, match="stores=") as excinfo:
            writer.put(b"x", codec="npy", node="prices", store="s3")
        assert "'s3'" in str(excinfo.value)

    def test_json_container_writes_no_blobs(self, tmp_path):
        """Values stay inline in a single document because no store exists."""
        comp = Computation()
        comp.add_node("array", value=np.arange(10_000, dtype="float64"))

        comp.save(str(tmp_path / "c.json"))

        manifest = json.loads(_read_manifest_text(tmp_path / "c.json", "json"))
        assert manifest["blobs"] == []
        assert manifest["nodes"][0]["value"].get("encoding") != "npy"


class TestMissingDestinationDirectory:
    """A destination whose parent does not exist is reported as such.

    Both containers build into a sibling ``.tmp`` first, so without the check
    the error names a temporary file the caller never asked for and never sees.
    """

    def test_zip_save_names_the_missing_directory(self, tmp_path):
        """Saving a .loman file names the directory, not the temporary."""
        comp = _sample_computation()
        target = tmp_path / "no_such_dir" / "c.loman"

        with pytest.raises(SerializationError, match="does not exist") as excinfo:
            comp.save(str(target))

        message = str(excinfo.value)
        assert "no_such_dir" in message
        assert ".tmp" not in message

    def test_dir_save_names_the_missing_directory(self, tmp_path):
        """The directory container reports it the same way."""
        comp = _sample_computation()
        target = tmp_path / "no_such_dir" / "c_dir"

        with pytest.raises(SerializationError, match="does not exist"):
            comp.save(str(target), container="dir")


class TestBlobReaderErrors:
    """Reading a blob the manifest describes but nothing can supply."""

    def test_missing_store_is_named(self):
        """A blob held in an unsupplied store reports which store, and the node."""
        from loman.serialization.blobs import BlobReader

        reader = BlobReader([{"id": 0, "path": "k", "store": "s3", "node": "prices"}], {})

        with pytest.raises(SerializationError, match="was not supplied") as excinfo:
            reader.get(0)
        message = str(excinfo.value)
        assert "'s3'" in message
        assert "prices" in message

    def test_unknown_id_is_refused(self):
        """A reference to a blob that is not listed is refused."""
        from loman.serialization.blobs import BlobReader

        with pytest.raises(SerializationError, match="does not resolve"):
            BlobReader([], {}).get(3)


class TestBlobScopeErrors:
    """The write and read scopes fail loudly when used out of context."""

    def test_offer_blob_outside_a_scope_is_false(self):
        """Outside a save, nothing is offered out-of-line storage."""
        from loman.serialization.computation import default_computation_transformer

        assert default_computation_transformer().offer_blob(nbytes=10**9) is False

    def test_put_blob_outside_a_scope_raises(self):
        """Calling put_blob with no sink open is a programming error, not a silent inline."""
        from loman.serialization.computation import default_computation_transformer

        with pytest.raises(RuntimeError, match="outside a write scope"):
            default_computation_transformer().put_blob(b"x", codec="npy")

    def test_get_blob_outside_a_scope_raises(self):
        """Decoding a blob reference with no container open says so clearly."""
        from loman.serialization.computation import default_computation_transformer

        with pytest.raises(RuntimeError, match="no blob store is open"):
            default_computation_transformer().get_blob({"$blob": 0})

    def test_reading_a_container_value_without_the_container(self, tmp_path):
        """A manifest lifted out of its container cannot decode its blobs."""
        comp = Computation()
        comp.add_node("array", value=np.arange(10_000, dtype="float64"))
        comp.save(str(tmp_path / "c_dir"), container="dir")

        manifest_text = (tmp_path / "c_dir" / MANIFEST_NAME).read_text()

        with pytest.raises(RuntimeError, match="no blob store is open"):
            ComputationSerializer().loads(manifest_text)


@pytest.mark.stress
class TestPerformanceGuard:
    """A large frame must not go back to being written one number at a time.

    Deselected from normal runs. It exists so that a regression to the
    element-wise encoding --- which cost 22 MB and most of a second for this
    frame --- fails loudly rather than being noticed months later.
    """

    def test_large_frame_saves_quickly_and_compactly(self, tmp_path):
        """100k x 10 floats save well under the element-wise cost."""
        import time

        n = 100_000
        frame = pd.DataFrame(
            np.random.default_rng(0).standard_normal((n, 10)),
            index=pd.date_range("2020-01-01", periods=n, freq="min"),
        )
        comp = Computation()
        comp.add_node("frame", value=frame)

        path = tmp_path / "big.loman"
        started = time.monotonic()
        comp.save(str(path))
        elapsed = time.monotonic() - started

        size = path.stat().st_size
        assert size < 12_000_000, f"saved {size:,} bytes; the element-wise encoding cost 22 MB"
        assert elapsed < 2.0, f"took {elapsed:.2f}s; the element-wise encoding took ~0.7s of pure formatting"
        assert Computation.load(str(path)).v.frame.equals(frame)

    def test_manifest_does_not_grow_with_the_data(self, tmp_path):
        """The manifest stays small however large the values are."""
        comp = Computation()
        comp.add_node("frame", value=pd.DataFrame(np.random.default_rng(0).standard_normal((200_000, 5))))
        comp.save(str(tmp_path / "big.loman"))

        assert len(_read_manifest_text(tmp_path / "big.loman", "zip")) < 4096
