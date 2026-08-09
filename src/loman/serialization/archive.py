"""The loman archive: a compact, partially-readable container for computations.

A JSON document holds everything inline, which is ideal for a small graph and
increasingly bad for a large one.  Numbers written as text cost roughly 2.7x
their in-memory size, every value must be parsed before any value can be read,
and the whole file has to be decoded to look at one node.

An archive is a zip holding:

- ``manifest.json`` — the same node/edge schema a JSON document uses, so the
  graph's structure stays greppable without unpacking anything.  Small values
  stay inline here.
- ``payloads/<id>.<ext>`` — one entry per large value, each in a format built
  for its type: parquet for frames and Series, ``.npy`` for arrays, JSON for
  anything else.

Because payloads are separate zip entries, :meth:`ArchiveSerializer.load` can
materialise a subset of nodes without reading the bytes of the rest.

Parquet needs pyarrow, which is an optional dependency (``pip install
'loman[archive]'``).  Without it, frames fall back to JSON payloads: the archive
is still written and still read, just larger.  The manifest always records which
encoding was used, so a reader missing a codec says so plainly rather than
failing somewhere deep in a decoder.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import zipfile
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import pandas as pd

from loman.exception import SerializationError

from .computation import ComputationSerializer

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

# Both extensions name the same format.  '.loman' is explicit; '.lm' is short.
ARCHIVE_EXTENSIONS = (".loman", ".lm")

MANIFEST_NAME = "manifest.json"
PAYLOAD_DIR = "payloads"

# Values estimated smaller than this stay inline in the manifest.  A graph of
# scalars should not produce a directory full of tiny zip entries.
DEFAULT_INLINE_THRESHOLD = 8192

PAYLOAD_MARKER = "__loman_payload__"

ENCODING_PARQUET = "parquet"
ENCODING_NPY = "npy"
ENCODING_JSON = "json"

_EXTENSIONS = {
    ENCODING_PARQUET: ".parquet",
    ENCODING_NPY: ".npy",
    ENCODING_JSON: ".json",
}

# Parquet compresses internally; deflating it again costs time and saves
# nothing.  npy and JSON are raw, so they are worth deflating.
_ZIP_COMPRESSION = {
    ENCODING_PARQUET: zipfile.ZIP_STORED,
    ENCODING_NPY: zipfile.ZIP_DEFLATED,
    ENCODING_JSON: zipfile.ZIP_DEFLATED,
}

# The column a Series is parked under inside its parquet payload.  Its real
# name is carried in the manifest, since it may be None or a non-string.
_SERIES_COLUMN = "__loman_series__"


def has_parquet_support() -> bool:
    """True when pyarrow is importable, so parquet payloads can be used."""
    return importlib.util.find_spec("pyarrow") is not None


def is_archive_path(path: str) -> bool:
    """True when *path* carries one of the archive extensions."""
    return path.lower().endswith(ARCHIVE_EXTENSIONS)


def _estimated_size(value: Any) -> int:
    """Roughly how many bytes *value* occupies in memory.

    Only used to decide inline versus payload, so being approximate is fine;
    being cheap is not optional, since this runs for every node.
    """
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(deep=True))
    return 0


class _PayloadWriter:
    """Accumulates out-of-line payloads while a manifest is being built."""

    def __init__(self, *, inline_threshold: int, use_parquet: bool) -> None:
        self._inline_threshold = inline_threshold
        self._use_parquet = use_parquet
        self._next_id = 0
        # id -> (encoding, bytes)
        self.payloads: dict[str, tuple[str, bytes]] = {}

    def _new_id(self) -> str:
        payload_id = f"p{self._next_id}"
        self._next_id += 1
        return payload_id

    def should_offload(self, value: Any) -> bool:
        """True when *value* is big enough to be worth its own zip entry."""
        return _estimated_size(value) >= self._inline_threshold

    def write(self, value: Any, transformer_encode: Any) -> dict[str, Any]:
        """Store *value* as a payload and return the manifest reference for it."""
        encoding, blob, extra = self._encode(value, transformer_encode)
        payload_id = self._new_id()
        self.payloads[payload_id] = (encoding, blob)
        ref: dict[str, Any] = {PAYLOAD_MARKER: True, "id": payload_id, "encoding": encoding}
        ref.update(extra)
        return ref

    def _encode(self, value: Any, transformer_encode: Any) -> tuple[str, bytes, dict[str, Any]]:
        """Pick the best available encoding for *value* and apply it."""
        if self._use_parquet and isinstance(value, pd.DataFrame):
            blob = _try_dataframe_to_parquet(value)
            if blob is not None:
                return ENCODING_PARQUET, blob, {"kind": "dataframe", **_index_extras(value.index)}

        if self._use_parquet and isinstance(value, pd.Series):
            blob = _try_dataframe_to_parquet(value.to_frame(name=_SERIES_COLUMN))
            if blob is not None:
                return (
                    ENCODING_PARQUET,
                    blob,
                    {
                        "kind": "series",
                        "name": transformer_encode(value.name),
                        **_index_extras(value.index),
                    },
                )

        if isinstance(value, np.ndarray) and value.dtype.kind not in ("O", "V"):
            buf = io.BytesIO()
            # allow_pickle stays off: a payload must never be able to execute
            # code when it is read back.
            np.save(buf, value, allow_pickle=False)
            return ENCODING_NPY, buf.getvalue(), {}

        # Everything else, including frames when pyarrow is unavailable.
        encoded = transformer_encode(value)
        return ENCODING_JSON, json.dumps(encoded).encode("utf-8"), {}


def _try_dataframe_to_parquet(df: pd.DataFrame) -> bytes | None:
    """Serialize a DataFrame to parquet bytes, or ``None`` if parquet cannot hold it.

    Parquet rejects duplicate column names, and arrow has no representation for
    columns of arbitrary Python objects.  Rather than fail the whole write, the
    caller falls back to a JSON payload — bigger, but it round-trips anything
    the transformer understands.
    """
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, engine="pyarrow", compression="zstd", index=True)
    except Exception:
        return None
    return buf.getvalue()


def _index_extras(index: pd.Index) -> dict[str, Any]:
    """Manifest fields carrying index detail that parquet does not preserve.

    A DatetimeIndex built by ``date_range`` has a frequency that is part of its
    identity — pandas compares two otherwise-equal indexes as different when one
    has lost it — and the parquet round trip drops it.
    """
    freq = getattr(index, "freqstr", None)
    return {"index_freq": freq} if freq is not None else {}


def _restore_index_freq(obj: Any, ref: dict[str, Any]) -> Any:
    """Re-apply a frequency recorded by :func:`_index_extras`."""
    freq = ref.get("index_freq")
    if freq is None:
        return obj
    # The values may no longer fit the frequency; not worth failing a read over.
    with contextlib.suppress(ValueError, TypeError, AttributeError):
        obj.index.freq = pd.tseries.frequencies.to_offset(freq)
    return obj


class _PayloadReader:
    """Reads payloads out of an open archive, on demand."""

    def __init__(self, zf: zipfile.ZipFile, transformer_decode: Any) -> None:
        self._zf = zf
        self._decode = transformer_decode

    def read(self, ref: dict[str, Any]) -> Any:
        """Materialise the value a manifest reference points at."""
        encoding = ref["encoding"]
        name = f"{PAYLOAD_DIR}/{ref['id']}{_EXTENSIONS.get(encoding, '')}"
        try:
            blob = self._zf.read(name)
        except KeyError as exc:
            msg = f"Archive is missing payload {name!r} referenced by its manifest."
            raise SerializationError(msg) from exc

        if encoding == ENCODING_PARQUET:
            return self._read_parquet(blob, ref)
        if encoding == ENCODING_NPY:
            return np.load(io.BytesIO(blob), allow_pickle=False)
        if encoding == ENCODING_JSON:
            return self._decode(json.loads(blob.decode("utf-8")))

        msg = f"Archive payload {ref['id']!r} uses unknown encoding {encoding!r}."
        raise SerializationError(msg)

    def _read_parquet(self, blob: bytes, ref: dict[str, Any]) -> Any:
        if not has_parquet_support():
            msg = (
                f"Archive payload {ref['id']!r} is stored as parquet, which needs pyarrow. "
                "Install it with:  pip install 'loman[archive]'"
            )
            raise SerializationError(msg)
        frame = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
        if ref.get("kind") == "series":
            series = frame[_SERIES_COLUMN]
            series.name = self._decode(ref.get("name"))
            return _restore_index_freq(series, ref)
        return _restore_index_freq(frame, ref)


class _ArchiveComputationSerializer(ComputationSerializer):
    """A ComputationSerializer that offloads large values to archive payloads.

    Constructed fresh for each dump or load by :class:`ArchiveSerializer`, and
    bound to that operation's writer or reader.  Binding at construction rather
    than stashing state on a long-lived object is what lets one
    :class:`ArchiveSerializer` be reused, and used concurrently, without two
    operations treading on each other.

    It borrows the transformer of the serializer it wraps — shared by reference
    and only read during an operation — so custom transformers keep working
    inside an archive.
    """

    def __init__(
        self,
        base: ComputationSerializer,
        *,
        writer: _PayloadWriter | None = None,
        reader: _PayloadReader | None = None,
    ) -> None:
        super().__init__(base._t, use_dill_for_functions=base._use_dill_for_functions)
        self._writer = writer
        self._reader = reader

    def _encode_value(self, node_key: Any, value: Any) -> Any:
        writer = self._writer
        if writer is not None and writer.should_offload(value):
            return writer.write(value, self._t.to_dict)
        return self._t.to_dict(value)

    def _decode_value(self, encoded: Any) -> Any:
        if isinstance(encoded, dict) and encoded.get(PAYLOAD_MARKER):
            if self._reader is None:  # pragma: no cover - guarded by callers
                msg = "Cannot decode an archive payload reference outside an archive."
                raise SerializationError(msg)
            return self._reader.read(encoded)
        return self._t.from_dict(encoded)


class ArchiveSerializer:
    """Read and write ``.loman`` / ``.lm`` archives.

    :param serializer: Optional :class:`ComputationSerializer` whose transformer
        configuration should be used.  Register custom transformers on it exactly
        as you would for :meth:`~loman.Computation.write_json`.
    :param inline_threshold: Values estimated at fewer than this many bytes stay
        inline in the manifest rather than becoming separate payloads.
    :param use_parquet: Force parquet on or off.  Defaults to using it when
        pyarrow is installed.  Setting it ``True`` without pyarrow raises.
    """

    def __init__(
        self,
        serializer: ComputationSerializer | None = None,
        *,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
        use_parquet: bool | None = None,
    ) -> None:
        """Initialise with an optional base serializer and payload settings."""
        self._base = serializer if serializer is not None else ComputationSerializer()
        self._inline_threshold = inline_threshold

        if use_parquet is None:
            use_parquet = has_parquet_support()
        elif use_parquet and not has_parquet_support():
            msg = "use_parquet=True but pyarrow is not installed. Install it with:  pip install 'loman[archive]'"
            raise SerializationError(msg)
        self._use_parquet = use_parquet

    def dump(self, comp: Any, fp: BinaryIO) -> None:
        """Write *comp* to *fp* as an archive."""
        writer = _PayloadWriter(inline_threshold=self._inline_threshold, use_parquet=self._use_parquet)
        manifest = _ArchiveComputationSerializer(self._base, writer=writer)._to_dict(comp)

        with zipfile.ZipFile(fp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, json.dumps(manifest))
            for payload_id, (encoding, blob) in writer.payloads.items():
                name = f"{PAYLOAD_DIR}/{payload_id}{_EXTENSIONS[encoding]}"
                zf.writestr(name, blob, compress_type=_ZIP_COMPRESSION[encoding])

    def load(self, fp: BinaryIO, *, nodes: Iterable[str] | None = None) -> Any:
        """Read a Computation from the archive in *fp*.

        When *nodes* is given, only those nodes' payloads are read from the zip;
        the rest are never decompressed at all.
        """
        try:
            zf = zipfile.ZipFile(fp)
        except zipfile.BadZipFile as exc:
            msg = "Not a loman archive: the file is not a zip container. Did you mean read_json?"
            raise SerializationError(msg) from exc

        with zf:
            try:
                manifest_bytes = zf.read(MANIFEST_NAME)
            except KeyError as exc:
                msg = f"Not a loman archive: no {MANIFEST_NAME} inside the zip."
                raise SerializationError(msg) from exc

            manifest = json.loads(manifest_bytes.decode("utf-8"))
            reader = _PayloadReader(zf, self._base._t.from_dict)
            inner = _ArchiveComputationSerializer(self._base, reader=reader)
            return inner._from_dict(manifest, only_nodes=None if nodes is None else set(nodes))

    def payload_summary(self, fp: BinaryIO) -> pd.DataFrame:
        """Describe an archive's contents without decoding any values.

        Returns a frame of one row per zip entry with its stored and compressed
        sizes — useful for finding which node is responsible for a large file.
        """
        with zipfile.ZipFile(fp) as zf:
            rows = [
                {
                    "name": info.filename,
                    "size": info.file_size,
                    "compressed": info.compress_size,
                }
                for info in zf.infolist()
            ]
        return pd.DataFrame(rows, columns=["name", "size", "compressed"])
