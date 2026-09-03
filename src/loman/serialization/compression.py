"""Blob compression.

Compression is a choice the caller makes, named on the profile as a codec and
level: ``"zstd:1"``, ``"zlib:6"``, or ``"none"``. Nothing here estimates,
samples or guesses.

There used to be an ``"auto"`` mode that compressed the first 256 KiB of a blob,
extrapolated, and skipped compression when the projection looked poor. It was
deleted rather than tuned, because it was wrong in both directions on data whose
character changes part way through --- which market data routinely does. On a
payload with a random head and a compressible tail it projected a 3.6% saving
against an actual 36.8%, and silently stored the blob raw.

The only reason to guess was that zlib rejects incompressible data at about
43 MB/s, so compressing to find out was expensive. zstd does the same at roughly
1 GB/s, and compresses real data better and faster besides, which is why it is a
required dependency rather than an extra. Compressing to find out now costs
about a second per gigabyte of incompressible data, so there is nothing left for
a heuristic to save.

One rule remains, and it is a measurement rather than a prediction: whichever of
the compressed and raw payloads is smaller is the one stored. A blob that did
not shrink is written as-is and recorded as ``"none"``, so no future read pays a
decompression step for nothing.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

from loman.exception import SerializationError

Codec = tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]


def _zlib_codec(level: int) -> Codec:
    """Return the compress/decompress pair for zlib at *level*."""
    return (lambda data: zlib.compress(data, level), zlib.decompress)


def _zstd_codec(level: int) -> Codec:
    """Return the compress/decompress pair for zstd at *level*."""
    import zstandard

    def compress(data: bytes) -> bytes:
        """Compress *data* at the level this codec was built for."""
        return zstandard.ZstdCompressor(level=level).compress(data)

    def decompress(data: bytes) -> bytes:
        """Decompress a zstd frame, whatever level wrote it."""
        return zstandard.ZstdDecompressor().decompress(data)

    return compress, decompress


_FAMILIES: dict[str, Callable[[int], Codec]] = {
    "zlib": _zlib_codec,
    "zstd": _zstd_codec,
}

# Level 1 for zstd: on measured data it gives 9.3x on a realistic price series
# at 568 MB/s, and rejects incompressible data at 1067 MB/s. Higher levels cost
# materially more for little further gain on numeric payloads.
_DEFAULT_LEVELS = {"zlib": 6, "zstd": 1}

#: What the efficient profile uses when the caller names nothing.
DEFAULT_COMPRESSION = "zstd:1"

#: Accepted when compression should not happen at all.
NO_COMPRESSION = "none"


def parse_spec(spec: str) -> tuple[str, int | None]:
    """Split a compression spec such as ``"zlib:1"`` into family and level."""
    family, _, level = spec.partition(":")
    if not level:
        return family, None
    try:
        return family, int(level)
    except ValueError:
        msg = f"Invalid compression level in {spec!r}: expected an integer after ':'"
        raise ValueError(msg) from None


def get_codec(spec: str) -> Codec:
    """Return the compress/decompress pair named by *spec*."""
    family, level = parse_spec(spec)
    factory = _FAMILIES.get(family)
    if factory is None:
        known = sorted([NO_COMPRESSION, *_FAMILIES])
        msg = f"Unknown compression {spec!r}; expected one of {known}"
        raise ValueError(msg)
    return factory(_DEFAULT_LEVELS[family] if level is None else level)


def register_codec(family: str, factory: Callable[[int], Codec], default_level: int) -> None:
    """Register a compression family, so a user can bring their own.

    :param family: Name used in a compression spec, before the ``:``.
    :param factory: Called with a level, returning ``(compress, decompress)``.
    :param default_level: Level used when a spec names no level.
    """
    _FAMILIES[family] = factory
    _DEFAULT_LEVELS[family] = default_level


def compress_blob(data: bytes, spec: str, *, compressible: bool = True) -> tuple[bytes, str]:
    """Return *data* compressed according to *spec*, and the spec actually used.

    The returned spec is ``"none"`` whenever the raw bytes are what got stored,
    whether because compression was not asked for or because it did not shrink
    them. That is what the manifest records, so a reader never has to know what
    was originally requested.

    :param data: The payload.
    :param spec: ``"none"``, or a family with an optional level.
    :param compressible: False for payloads that already compress themselves,
        such as parquet. Skips compression entirely, which is how
        double-compression is prevented.
    """
    if spec == NO_COMPRESSION or not compressible or not data:
        return data, NO_COMPRESSION

    compress, _ = get_codec(spec)
    compressed = compress(data)

    # Store whichever is smaller. Incompressible data comes back slightly larger
    # than it went in, since a codec adds framing, and keeping that would mean a
    # bigger file *and* a decompression step on every future read.
    if len(compressed) >= len(data):
        return data, NO_COMPRESSION
    return compressed, spec


def decompress_blob(data: bytes, spec: str) -> bytes:
    """Return *data* decompressed according to *spec*."""
    if spec in (NO_COMPRESSION, "", None):
        return data
    try:
        _, decompress = get_codec(spec)
    except ValueError as exc:
        msg = f"Cannot read a blob compressed with {spec!r}: {exc}"
        raise SerializationError(msg) from exc
    return decompress(data)
