"""Blob storage: where a value's bytes live when they are too big to inline.

A saved computation has one logical layout::

    manifest.json
    blobs/0000.npy
    blobs/0001.npy

A ``.loman`` file is that tree inside a zip. A directory container is that tree
on disk. One specification, two ways of writing it down --- which is asserted by
a test comparing the manifests byte for byte, rather than left to discipline.

Blob keys are never derived from node keys. Zero-padded integer ids sidestep
``/`` in hierarchical keys, Windows reserved names, case-insensitive collisions
and unicode normalisation in one move; the blob table records which node a blob
came from so a human can still find their way around.

Two roles are kept apart here, which is what makes storage pluggable:

:class:`BlobStore` is *where bytes go*. It has two methods, and a user
implementing one for S3 or a database inherits compression, deduplication,
checksums and blob-table bookkeeping without writing any of it.

:class:`BlobWriter` and :class:`BlobReader` are *the bookkeeping*. One writer per
save, holding the blob table and routing each blob to whichever store its node
asked for. A single save can therefore span several stores --- most values in the
container, one node's frames in S3 --- and the manifest records which is which.

Bytes reaching a store are already compressed, if compressing them paid. The
container never compresses again: that would double-compress an already-compact
payload, and would put every member behind a decompression step, foreclosing a
future zero-copy read.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

from loman.exception import SerializationError

from .compression import compress_blob, decompress_blob

MANIFEST_NAME = "manifest.json"
BLOB_DIR = "blobs"

#: Name of the store that writes into the container itself. A blob entry without
#: a ``store`` field means this one, which keeps the common manifest short.
CONTAINER_STORE = "container"

# Key marking a blob reference inside an encoded value. Reserved, so a user dict
# containing it is escaped rather than mistaken for one.
BLOB_REF_KEY = "$blob"

# Zip timestamp written for every member. Zip stores no timezone and defaults to
# "now", which would make two saves of the same computation differ. A fixed
# stamp makes saves byte-reproducible, which is what lets them be compared,
# cached by content, or diffed.
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

BlobPayload = bytes | bytearray | memoryview | Callable[[BinaryIO], None]


def blob_ref(blob_id: int) -> dict[str, int]:
    """Return the encoded reference to blob *blob_id*."""
    return {BLOB_REF_KEY: blob_id}


def _payload_to_bytes(payload: BlobPayload) -> bytes:
    """Materialise *payload* as bytes, running a writer callable if given."""
    if callable(payload):
        writer = cast("Callable[[BinaryIO], None]", payload)
        buf = io.BytesIO()
        writer(buf)
        return buf.getvalue()
    return bytes(payload)


class BlobStore(ABC):
    """Where a blob's bytes are kept. Implement two methods.

    Everything else --- ids, compression, deduplication, checksums, the blob
    table --- belongs to :class:`BlobWriter` and is inherited, so a store for S3
    or a database is genuinely just these two::

        class S3Store(BlobStore):
            def __init__(self, bucket, prefix, client):
                self.bucket, self.prefix, self.client = bucket, prefix, client

            def write_blob(self, key, data):
                self.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}", Body=data)

            def read_blob(self, key):
                return self.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{key}")["Body"].read()

    Pass it by name to both ends::

        comp.save('run.loman', stores={'s3': S3Store(...)})
        Computation.load('run.loman', stores={'s3': S3Store(...)})

    The key is a short relative path such as ``blobs/0000.npy``. A store is free
    to place it under a prefix of its own, as above; it just has to hand the same
    bytes back for the same key.

    The manifest records the store's *name*, never its configuration, so a saved
    file never contains a bucket, a connection string or a credential. The
    consequence is that a file cannot resolve its own external blobs: whoever
    loads it supplies the matching store.
    """

    @abstractmethod
    def write_blob(self, key: str, data: bytes) -> None:
        """Store *data* under *key*."""

    @abstractmethod
    def read_blob(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""

    def key_for(self, blob_id: int, codec: str, node: str | None) -> str:
        """Return the key to store blob *blob_id* under.

        Overriding this lets a store lay its keys out differently --- by node
        name, or partitioned by date. The key is recorded in the manifest, so
        whatever is returned here is what :meth:`read_blob` is later asked for.
        """
        return f"{BLOB_DIR}/{blob_id:04d}.{codec}"


class ZipBlobStore(BlobStore):
    """Blobs as stored (uncompressed) members of a zip archive."""

    def __init__(self, zf: zipfile.ZipFile) -> None:
        """Read and write blobs in the already-open archive *zf*."""
        self._zf = zf

    def write_blob(self, key: str, data: bytes) -> None:
        """Write *data* as a stored zip member."""
        info = zipfile.ZipInfo(key, date_time=_FIXED_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        self._zf.writestr(info, data)

    def read_blob(self, key: str) -> bytes:
        """Return the raw bytes of a zip member."""
        return self._zf.read(_validate_member_path(key))


class DirBlobStore(BlobStore):
    """Blobs as files under a directory."""

    def __init__(self, root: Path) -> None:
        """Read and write blobs under *root*."""
        self._root = root

    def write_blob(self, key: str, data: bytes) -> None:
        """Write *data* to a file under the container root."""
        target = self._root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def read_blob(self, key: str) -> bytes:
        """Return the raw bytes of a file under the container root."""
        return (self._root / _validate_member_path(key)).read_bytes()


class MemoryBlobStore(BlobStore):
    """Blobs in a dict. Useful for tests, and as a worked example of the interface."""

    def __init__(self, data: dict[str, bytes] | None = None) -> None:
        """Keep blobs in *data*, or in a fresh dict."""
        self.data: dict[str, bytes] = {} if data is None else data

    def write_blob(self, key: str, data: bytes) -> None:
        """Store *data* under *key*."""
        self.data[key] = data

    def read_blob(self, key: str) -> bytes:
        """Return the bytes stored under *key*."""
        try:
            return self.data[key]
        except KeyError:
            msg = f"No blob stored under {key!r}"
            raise SerializationError(msg) from None


class BlobWriter:
    """Blob bookkeeping for one save: ids, compression, dedup, and the table.

    Routes each blob to a named :class:`BlobStore`. A save can use several ---
    the container for most values, an external store for the nodes that asked
    for one --- and the manifest records which store holds each blob.
    """

    def __init__(
        self,
        stores: dict[str, BlobStore],
        *,
        compression: str = "none",
        dedupe: str = "identity",
        checksums: bool = False,
    ) -> None:
        """Write blobs into *stores*, keyed by name."""
        self._stores = stores
        self._entries: list[dict[str, Any]] = []
        self._compression = compression
        self._dedupe = dedupe
        self._checksums = checksums
        self._by_identity: dict[int, int] = {}
        self._by_content: dict[str, int] = {}
        # Keeps deduplicated objects alive so their ids stay unique; see put().
        self._dedupe_refs: list[Any] = []

    def can_store(self, store: str | None) -> bool:
        """Return whether blobs can be written to the named store.

        ``None`` means the container's own store. A container that holds no
        blobs --- the single JSON document --- has none, but an external store
        named on a node still works there, which is how a readable manifest can
        sit alongside data in S3.
        """
        return (store or CONTAINER_STORE) in self._stores

    @property
    def accepts_blobs(self) -> bool:
        """Whether any store at all is available."""
        return bool(self._stores)

    def table(self) -> list[dict[str, Any]]:
        """Return the manifest's blob table."""
        return self._entries

    def put(
        self,
        payload: BlobPayload,
        *,
        codec: str,
        node: str | None,
        store: str | None = None,
        compressible: bool = True,
        dedupe_on: Any = None,
    ) -> int:
        """Store *payload*, compressing and deduplicating as configured."""
        store_name = store or CONTAINER_STORE
        target = self._stores.get(store_name)
        if target is None:
            known = sorted(self._stores) or ["(none)"]
            msg = (
                f"Node {node!r} asks for blob store {store_name!r}, which was not supplied. "
                f"Pass it as save(..., stores={{{store_name!r}: ...}}). Available: {known}"
            )
            raise SerializationError(msg)

        if dedupe_on is not None and self._dedupe == "identity":
            existing = self._by_identity.get(id(dedupe_on))
            if existing is not None:
                return existing

        raw = _payload_to_bytes(payload)
        stored, compression = compress_blob(raw, self._compression, compressible=compressible)

        digest = None
        if self._checksums or self._dedupe == "content":
            digest = hashlib.sha256(stored).hexdigest()
            if self._dedupe == "content":
                existing = self._by_content.get(digest)
                if existing is not None:
                    return existing

        blob_id = len(self._entries)
        key = target.key_for(blob_id, codec, node)
        target.write_blob(key, stored)

        entry: dict[str, Any] = {
            "id": blob_id,
            "path": key,
            "codec": codec,
            "compression": compression,
            "size": len(raw),
        }
        if store_name != CONTAINER_STORE:
            entry["store"] = store_name
        if compression != "none":
            entry["stored_size"] = len(stored)
        if node is not None:
            entry["node"] = node
        if self._checksums and digest is not None:
            entry["sha256"] = digest
        self._entries.append(entry)

        if dedupe_on is not None:
            # The object is kept alive alongside its id. Without that reference a
            # temporary --- a column's `.to_numpy()`, say --- would be collected
            # as soon as this returns, and CPython would hand the same id to the
            # next temporary, deduplicating two unrelated values onto one blob.
            self._by_identity[id(dedupe_on)] = blob_id
            self._dedupe_refs.append(dedupe_on)
        if digest is not None:
            self._by_content.setdefault(digest, blob_id)
        return blob_id


class BlobReader:
    """Resolves blob references from an already-written container."""

    def __init__(self, entries: list[dict[str, Any]], stores: dict[str, BlobStore]) -> None:
        """Resolve blobs listed in *entries* against *stores*."""
        self._entries = {int(entry["id"]): entry for entry in entries}
        self._stores = stores

    def get(self, blob_id: int) -> bytes:
        """Return the bytes of blob *blob_id*, decompressed."""
        entry = self._entries.get(int(blob_id))
        if entry is None:
            msg = f"Blob reference {blob_id} does not resolve: the manifest lists no such blob"
            raise SerializationError(msg)

        store_name = entry.get("store", CONTAINER_STORE)
        store = self._stores.get(store_name)
        if store is None:
            known = sorted(self._stores) or ["(none)"]
            node = entry.get("node")
            where = f" (node {node!r})" if node else ""
            msg = (
                f"Blob {blob_id}{where} is held in store {store_name!r}, which was not supplied. "
                f"Pass it as load(..., stores={{{store_name!r}: ...}}). Available: {known}. "
                "A saved file records a store's name but never its configuration, so it cannot "
                "resolve external blobs on its own."
            )
            raise SerializationError(msg)

        return decompress_blob(store.read_blob(entry["path"]), entry.get("compression", "none"))


def _validate_member_path(path: str) -> str:
    """Return *path* if it is safe to resolve inside a container.

    A manifest is data, and its blob paths are chosen by whoever wrote the file.
    An absolute path or one climbing out with ``..`` would read from anywhere on
    the machine, so both are refused before the path is used.
    """
    if not path:
        msg = "Blob path is empty"
        raise SerializationError(msg)
    normalised = posixpath.normpath(path)
    if posixpath.isabs(path) or normalised.startswith("..") or Path(path).is_absolute():
        msg = f"Refusing blob path outside the container: {path!r}"
        raise SerializationError(msg)
    return normalised


def write_zip_container(
    path: Path,
    build: Callable[[dict[str, BlobStore]], dict[str, Any]],
) -> None:
    """Write a ``.loman`` archive at *path*.

    *build* is handed the container's store and returns the manifest. Blobs are
    written first, as the manifest is only complete once every blob has an id ---
    so the manifest member is added last, even though it is read first.
    """
    _require_parent(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            manifest = build({CONTAINER_STORE: ZipBlobStore(zf)})
            info = zipfile.ZipInfo(MANIFEST_NAME, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, dump_manifest(manifest))
        tmp.replace(path)
    finally:
        if tmp.exists():  # pragma: no cover - only on a failed write
            tmp.unlink()


def write_dir_container(
    path: Path,
    build: Callable[[dict[str, BlobStore]], dict[str, Any]],
) -> None:
    """Write a directory container at *path*, replacing any existing one.

    Built alongside the target and swapped in at the end, so a save that fails
    part way leaves whatever was there before intact. Clearing the old blobs
    first would be simpler, but then an unserializable value in the middle of a
    graph would destroy the last good checkpoint --- losing data on the strength
    of an operation that did not even succeed.
    """
    if path.exists() and not path.is_dir():
        msg = f"Cannot write a directory container over the existing file {str(path)!r}"
        raise SerializationError(msg)
    _require_parent(path)

    staging = path.with_name(path.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        manifest = build({CONTAINER_STORE: DirBlobStore(staging)})
        (staging / MANIFEST_NAME).write_text(dump_manifest(manifest), encoding="utf-8")

        # Swap: move the old aside, put the new in place, then discard the old.
        # Two renames rather than one because a directory cannot be replaced
        # atomically the way a file can.
        previous = path.with_name(path.name + ".previous")
        shutil.rmtree(previous, ignore_errors=True)
        if path.exists():
            path.rename(previous)
        try:
            staging.rename(path)
        except OSError:  # pragma: no cover - put the old one back and re-raise
            if previous.exists():
                previous.rename(path)
            raise
        shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _require_parent(path: Path) -> None:
    """Fail early, and about the right path, when the destination has no parent.

    Both containers write to a sibling temporary first, so without this the
    error names a ``.tmp`` file the caller never asked for.
    """
    parent = path.parent
    if not parent.exists():
        msg = f"Cannot save to {str(path)!r}: the directory {str(parent)!r} does not exist"
        raise SerializationError(msg)


def dump_manifest(manifest: dict[str, Any]) -> str:
    """Return the manifest as JSON text.

    ``allow_nan`` is off so a non-finite float can never reach the file as a
    bare ``NaN`` token, which Python reads back and no other JSON parser will.
    """
    return json.dumps(manifest, allow_nan=False)


def read_zip_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Return the manifest from an open archive."""
    try:
        raw = zf.read(MANIFEST_NAME)
    except KeyError as exc:
        msg = f"Not a loman container: no {MANIFEST_NAME} inside the archive"
        raise SerializationError(msg) from exc
    return json.loads(raw.decode("utf-8"))


def read_dir_manifest(root: Path) -> dict[str, Any]:
    """Return the manifest from a directory container."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        msg = f"Not a loman container: {str(root)!r} has no {MANIFEST_NAME}"
        raise SerializationError(msg)
    return json.loads(manifest_path.read_text(encoding="utf-8"))
