"""Bring-your-own-storage: routing a node's values to a store you supply.

The store here stands in for S3 or a database. It is deliberately implemented
the way a user would implement one --- two methods, no loman internals --- so
that if this file needs to reach past the public interface, the interface is
wrong.
"""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from loman import Computation
from loman.exception import SerializationError
from loman.serialization import MemoryBlobStore, SerializationProfile
from loman.serialization.blobs import MANIFEST_NAME, BlobStore, _require_parent

LARGE = 20_000


class RecordingStore(BlobStore):
    """An in-memory store standing in for S3 or a database.

    Records the calls made to it, so tests can assert not just that a round-trip
    worked but that the bytes genuinely went somewhere else.
    """

    def __init__(self, prefix: str = "") -> None:
        """Keep blobs in a dict, optionally under a key prefix."""
        self.blobs: dict[str, bytes] = {}
        self.prefix = prefix
        self.writes: list[str] = []
        self.reads: list[str] = []

    def write_blob(self, key: str, data: bytes) -> None:
        """Record and store *data*."""
        self.writes.append(key)
        self.blobs[self.prefix + key] = data

    def read_blob(self, key: str) -> bytes:
        """Record and return the bytes for *key*."""
        self.reads.append(key)
        return self.blobs[self.prefix + key]


class NodeKeyedStore(BlobStore):
    """A store that lays its keys out by node name rather than by blob id."""

    def __init__(self) -> None:
        """Keep blobs in a dict."""
        self.blobs: dict[str, bytes] = {}

    def key_for(self, blob_id: int, codec: str, node: str | None) -> str:
        """Name each blob after the node it came from."""
        return f"{node or 'anon'}/{blob_id:03d}.{codec}"

    def write_blob(self, key: str, data: bytes) -> None:
        """Store *data* under *key*."""
        self.blobs[key] = data

    def read_blob(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        return self.blobs[key]


def _frame(n: int = LARGE) -> pd.DataFrame:
    """Return a frame large enough to be stored out of line."""
    return pd.DataFrame(
        {"px": np.arange(n, dtype="float64")},
        index=pd.date_range("2020-01-01", periods=n, freq="min"),
    )


def _manifest(path, container="zip"):
    """Return the manifest from a container."""
    if container == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if container == "dir":
        return json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(MANIFEST_NAME))


def test_payload_to_bytes_accepts_bytes():
    """Accept byte payloads without converting their contents."""
    from loman.serialization.blobs import _payload_to_bytes

    assert _payload_to_bytes(b"hello") == b"hello"


def test_read_blob_missing_key():
    """Raise a serialization error when a blob key is absent."""
    store = MemoryBlobStore()

    with pytest.raises(SerializationError, match="No blob stored under"):
        store.read_blob("missing")


def test_require_parent_missing_directory(tmp_path):
    """Report an error when the output parent directory is missing."""
    path = tmp_path / "does-not_exist" / "file.loman"

    with pytest.raises(SerializationError, match="does-not_exist"):
        _require_parent(path)


class TestNodeDeclaredStore:
    """A node names its store; save and load supply the implementation."""

    def test_roundtrip_through_an_external_store(self, tmp_path):
        """A node marked for a store round-trips through it."""
        frame = _frame()
        comp = Computation()
        comp.add_node("prices", value=frame, store="warehouse")

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert store.writes, "nothing was written to the external store"

        restored = Computation.load(str(tmp_path / "c.loman"), stores={"warehouse": store})
        assert restored.v.prices.equals(frame)
        assert store.reads

    def test_roundtrip_through_a_memoryblobstore(self, tmp_path):
        """A node aiming to achieve code coverage for the in-memory store."""
        frame = _frame()
        comp = Computation()
        comp.add_node("prices", value=frame, store="warehouse")

        store = MemoryBlobStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        restored = Computation.load(str(tmp_path / "c.loman"), stores={"warehouse": store})
        assert restored.v.prices.equals(frame)

    def test_bytes_are_not_in_the_container(self, tmp_path):
        """The archive holds the manifest only; the data went elsewhere."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": RecordingStore()})

        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            assert zf.namelist() == [MANIFEST_NAME]

    def test_manifest_records_the_store_name(self, tmp_path):
        """Each blob says which store holds it."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": RecordingStore()})

        entries = _manifest(tmp_path / "c.loman")["blobs"]
        assert entries
        assert all(entry["store"] == "warehouse" for entry in entries)

    def test_manifest_holds_no_store_configuration(self, tmp_path):
        """A saved file never contains a bucket, endpoint or credential."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": RecordingStore(prefix="s3://secret-bucket/")})

        text = json.dumps(_manifest(tmp_path / "c.loman"))
        assert "secret-bucket" not in text
        assert "s3://" not in text

    def test_the_node_store_survives_a_roundtrip(self, tmp_path):
        """A reloaded node still knows which store it belongs to."""
        from loman.consts import NodeAttributes
        from loman.nodekey import parse_nodekey

        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})
        restored = Computation.load(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert restored.dag.nodes[parse_nodekey("prices")][NodeAttributes.STORE] == "warehouse"

    def test_resaving_keeps_the_routing(self, tmp_path):
        """A reloaded computation saves back to the same store without re-marking."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        store = RecordingStore()
        comp.save(str(tmp_path / "one.loman"), stores={"warehouse": store})
        restored = Computation.load(str(tmp_path / "one.loman"), stores={"warehouse": store})

        second = RecordingStore()
        restored.save(str(tmp_path / "two.loman"), stores={"warehouse": second})

        assert second.writes
        with zipfile.ZipFile(tmp_path / "two.loman") as zf:
            assert zf.namelist() == [MANIFEST_NAME]


class TestMixedStores:
    """One save can span the container and an external store."""

    def test_only_marked_nodes_go_outside(self, tmp_path):
        """An unmarked node's blobs stay in the container."""
        comp = Computation()
        comp.add_node("remote", value=_frame(), store="warehouse")
        comp.add_node("local", value=np.arange(LARGE, dtype="float64"))

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        entries = _manifest(tmp_path / "c.loman")["blobs"]
        by_node = {entry["node"]: entry.get("store") for entry in entries}
        assert by_node["remote"] == "warehouse"
        assert by_node["local"] is None

        with zipfile.ZipFile(tmp_path / "c.loman") as zf:
            assert any(name.startswith("blobs/") for name in zf.namelist())

    def test_both_reload(self, tmp_path):
        """Values from both stores come back together."""
        frame = _frame()
        array = np.arange(LARGE, dtype="float64")
        comp = Computation()
        comp.add_node("remote", value=frame, store="warehouse")
        comp.add_node("local", value=array)

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})
        restored = Computation.load(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert restored.v.remote.equals(frame)
        assert np.array_equal(restored.v.local, array)

    def test_several_external_stores(self, tmp_path):
        """Different nodes can go to different stores."""
        comp = Computation()
        comp.add_node("a", value=_frame(), store="warehouse")
        comp.add_node("b", value=_frame(), store="archive")

        warehouse, archive = RecordingStore(), RecordingStore()
        stores = {"warehouse": warehouse, "archive": archive}
        comp.save(str(tmp_path / "c.loman"), stores=stores)

        assert warehouse.writes
        assert archive.writes

        restored = Computation.load(str(tmp_path / "c.loman"), stores=stores)
        assert restored.v.a.equals(restored.v.b)


class TestProfileOverridesTheNode:
    """The save can redirect a node the graph already marked."""

    def test_override_wins(self, tmp_path):
        """A profile override sends a marked node somewhere else."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        elsewhere = RecordingStore()
        profile = SerializationProfile(
            name="test",
            inline_max_bytes=1024,
            compression="zstd:1",
            overrides={"prices": {"store": "elsewhere"}},
        )
        comp.save(str(tmp_path / "c.loman"), profile=profile, stores={"elsewhere": elsewhere})

        assert elsewhere.writes
        assert all(e["store"] == "elsewhere" for e in _manifest(tmp_path / "c.loman")["blobs"])

    def test_override_can_send_a_node_to_the_container(self, tmp_path):
        """The same computation saves locally when the profile says so.

        This is the reason routing is a save-time decision as well as a node
        one: a test should not need a bucket.
        """
        frame = _frame()
        comp = Computation()
        comp.add_node("prices", value=frame, store="warehouse")

        profile = SerializationProfile(
            name="local",
            inline_max_bytes=1024,
            compression="zstd:1",
            overrides={"prices": {"store": None}},
        )
        comp.save(str(tmp_path / "c.loman"), profile=profile)

        assert all("store" not in entry for entry in _manifest(tmp_path / "c.loman")["blobs"])
        assert Computation.load(str(tmp_path / "c.loman")).v.prices.equals(frame)

    def test_tag_selector_routes(self, tmp_path):
        """A tag selector reaches the store setting.

        Regression: ``settings_for`` took tags but the only caller never passed
        them, so every ``tag:`` selector silently matched nothing.
        """
        comp = Computation()
        comp.add_node("prices", value=_frame(), tags=["bulky"])

        store = RecordingStore()
        profile = SerializationProfile(
            name="tagged",
            inline_max_bytes=1024,
            compression="zstd:1",
            overrides={"tag:bulky": {"store": "warehouse"}},
        )
        comp.save(str(tmp_path / "c.loman"), profile=profile, stores={"warehouse": store})

        assert store.writes, "the tag selector did not route anything"
        assert all(e["store"] == "warehouse" for e in _manifest(tmp_path / "c.loman")["blobs"])

    def test_tag_selector_leaves_untagged_nodes_alone(self, tmp_path):
        """A node without the tag is unaffected."""
        comp = Computation()
        comp.add_node("plain", value=np.arange(LARGE, dtype="float64"))

        store = RecordingStore()
        profile = SerializationProfile(
            name="tagged",
            inline_max_bytes=1024,
            compression="zstd:1",
            overrides={"tag:bulky": {"store": "warehouse"}},
        )
        comp.save(str(tmp_path / "c.loman"), profile=profile, stores={"warehouse": store})

        assert not store.writes


class TestExternalStoreWithReadableManifest:
    """A plain JSON manifest can sit alongside data held elsewhere."""

    def test_json_container_with_an_external_store(self, tmp_path):
        """Values go to the store; the manifest stays a readable document."""
        frame = _frame()
        comp = Computation()
        comp.add_node("prices", value=frame, store="warehouse")

        store = RecordingStore()
        profile = SerializationProfile(name="hybrid", inline_max_bytes=1024, compression="zstd:1")
        comp.save(str(tmp_path / "c.json"), profile=profile, container="json", stores={"warehouse": store})

        assert store.writes
        manifest = _manifest(tmp_path / "c.json", "json")
        assert manifest["blobs"][0]["store"] == "warehouse"

        restored = Computation.load(str(tmp_path / "c.json"), stores={"warehouse": store})
        assert restored.v.prices.equals(frame)

    def test_efficient_json_without_a_store_still_refused(self, tmp_path):
        """With nowhere at all to put them, the combination is still an error."""
        comp = Computation()
        comp.add_node("prices", value=_frame())

        with pytest.raises(ValueError, match="nowhere to put them"):
            comp.save(str(tmp_path / "c.json"), profile="efficient")


class TestCustomKeyLayout:
    """A store controls how its blobs are named."""

    def test_key_for_is_honoured(self, tmp_path):
        """Keys come from the store, and the manifest records what it chose."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        store = NodeKeyedStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert all(key.startswith("prices/") for key in store.blobs)
        assert all(entry["path"].startswith("prices/") for entry in _manifest(tmp_path / "c.loman")["blobs"])

    def test_custom_keys_reload(self, tmp_path):
        """Whatever the store named them, it is asked for the same keys back."""
        frame = _frame()
        comp = Computation()
        comp.add_node("prices", value=frame, store="warehouse")

        store = NodeKeyedStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert Computation.load(str(tmp_path / "c.loman"), stores={"warehouse": store}).v.prices.equals(frame)


class TestMissingStore:
    """Failing to supply a store is reported clearly at both ends."""

    def test_save_without_the_store(self, tmp_path):
        """Saving a node routed to a store nobody supplied names it."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        with pytest.raises(SerializationError, match="warehouse") as excinfo:
            comp.save(str(tmp_path / "c.loman"))
        assert "stores=" in str(excinfo.value)

    def test_load_without_the_store(self, tmp_path):
        """Loading a file whose blobs live elsewhere explains what is missing."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": RecordingStore()})

        with pytest.raises(SerializationError, match="was not supplied") as excinfo:
            Computation.load(str(tmp_path / "c.loman"))
        message = str(excinfo.value)
        assert "warehouse" in message
        assert "prices" in message

    def test_load_names_the_available_stores(self, tmp_path):
        """The error lists what was supplied, so a typo is obvious."""
        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": RecordingStore()})

        with pytest.raises(SerializationError, match="wharehouse") as excinfo:
            Computation.load(str(tmp_path / "c.loman"), stores={"wharehouse": RecordingStore()})
        assert "warehouse" in str(excinfo.value)


class TestStoreInteractionWithProfileFeatures:
    """Compression, dedup and checksums apply to external stores too."""

    def test_compression_applies(self, tmp_path):
        """Bytes reaching an external store are already compressed."""
        values = np.round(100 + np.cumsum(np.random.default_rng(0).standard_normal(200_000) * 0.01), 2)
        comp = Computation()
        comp.add_node("px", value=values, store="warehouse")

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        entry = _manifest(tmp_path / "c.loman")["blobs"][0]
        assert entry["compression"] != "none"
        stored = next(iter(store.blobs.values()))
        assert len(stored) == entry["stored_size"] < entry["size"]

    def test_dedup_applies(self, tmp_path):
        """Two nodes sharing an object write one blob to the store."""
        shared = np.arange(LARGE, dtype="float64")
        comp = Computation()
        comp.add_node("a", value=shared, store="warehouse")
        comp.add_node("b", value=shared, store="warehouse")

        store = RecordingStore()
        comp.save(str(tmp_path / "c.loman"), stores={"warehouse": store})

        assert len(store.blobs) == 1

    def test_checksums_apply(self, tmp_path):
        """Digests are recorded for external blobs as well."""
        import hashlib

        comp = Computation()
        comp.add_node("prices", value=_frame(), store="warehouse")

        store = RecordingStore()
        profile = SerializationProfile(name="ck", inline_max_bytes=1024, compression="zstd:1", checksums=True)
        comp.save(str(tmp_path / "c.loman"), profile=profile, stores={"warehouse": store})

        entry = _manifest(tmp_path / "c.loman")["blobs"][0]
        assert entry["sha256"] == hashlib.sha256(store.blobs[entry["path"]]).hexdigest()


class TestCustomTransformerWithAStore:
    """A user type, encoded by a user transformer, into a user store."""

    def test_end_to_end(self, tmp_path):
        """The three extension points compose."""
        from loman.serialization import CustomTransformer

        class Matrix:
            def __init__(self, data):
                self.data = np.asarray(data)

            def __eq__(self, other):
                return isinstance(other, Matrix) and np.array_equal(self.data, other.data)

        class MatrixTransformer(CustomTransformer):
            @property
            def name(self):
                return "matrix"

            def to_dict(self, transformer, o):
                head = {"shape": list(o.data.shape)}
                if transformer.offer_blob(nbytes=o.data.nbytes):
                    head["encoding"] = "npy"
                    head["data"] = transformer.put_blob(lambda f: np.save(f, o.data, allow_pickle=False), codec="npy")
                else:
                    head["data"] = transformer.to_dict(o.data.tolist())
                return head

            def from_dict(self, transformer, d):
                if d.get("encoding") == "npy":
                    import io

                    return Matrix(np.load(io.BytesIO(transformer.get_blob(d["data"])), allow_pickle=False))
                return Matrix(transformer.from_dict(d["data"]))

            @property
            def supported_direct_types(self):
                return [Matrix]

        from loman.serialization import ComputationSerializer

        matrix = Matrix(np.arange(LARGE, dtype="float64"))
        comp = Computation()
        comp.add_node("m", value=matrix, store="warehouse")

        serializer = ComputationSerializer()
        serializer.register(MatrixTransformer())
        store = RecordingStore()

        comp.save(str(tmp_path / "c.loman"), serializer=serializer, stores={"warehouse": store})
        restored = Computation.load(str(tmp_path / "c.loman"), serializer=serializer, stores={"warehouse": store})

        assert store.writes
        assert restored.v.m == matrix
