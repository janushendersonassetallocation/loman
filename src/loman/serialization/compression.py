"""Blob compression, and deciding when it is worth doing.

Measured on this codebase, blanket compression is wrong in both directions. On
random float data, DEFLATE bought about 4% and cost three seconds for 128 MB. On
a realistic price series --- a random walk rounded to two decimals, which is what
financial data actually looks like --- zlib level 1 was **eight times** smaller
and took ten milliseconds.

So neither always-on nor always-off is defensible, and the choice cannot be made
from the codec or the dtype either: both of those examples are float64 arrays.
It has to come from the data. ``"auto"`` compresses a sample, extrapolates, and
keeps the result only if it is worth keeping.

Compression is applied here, before bytes reach the container, and the container
stores them without compressing again. That prevents double-compressing an
already-compact payload and keeps stored members at known offsets.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any

from loman.exception import SerializationError

#: Bytes of a payload compressed to decide whether compressing all of it pays.
SAMPLE_BYTES = 256 * 1024

#: A sampled payload must shrink by at least this fraction to be worth storing
#: compressed. Below it, the decompression cost on every future read outweighs
#: the saving.
MIN_SAVING = 0.10

#: Level used for the probe. Level 1 is the cheapest, and the question being
#: asked is "does this compress at all", not "how small can it get".
PROBE_LEVEL = 1

Codec = tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]


def _zlib_codec(level: int) -> Codec:
    """Return the compress/decompress pair for zlib at *level*."""
    return (lambda data: zlib.compress(data, level), zlib.decompress)


def _zstd_codec(level: int) -> Codec:
    """Return the compress/decompress pair for zstd at *level*."""
    from loman._extras import require

    zstandard = require("zstandard", "efficient")

    def compress(data: bytes) -> bytes:
        return zstandard.ZstdCompressor(level=level).compress(data)

    def decompress(data: bytes) -> bytes:
        return zstandard.ZstdDecompressor().decompress(data)

    return compress, decompress


_FAMILIES: dict[str, Callable[[int], Codec]] = {
    "zlib": _zlib_codec,
    "zstd": _zstd_codec,
}

_DEFAULT_LEVELS = {"zlib": 6, "zstd": 3}


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
        msg = f"Unknown compression {spec!r}; expected one of {sorted(['none', 'auto', *_FAMILIES])}"
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

    :param data: The payload.
    :param spec: ``"none"``, ``"auto"``, or a family with an optional level.
    :param compressible: False for payloads that already compress themselves,
        such as parquet. Short-circuits to no compression, which is how
        double-compression is prevented.
    """
    if spec == "none" or not compressible or not data:
        return data, "none"

    if spec == "auto":
        chosen = _probe(data)
        if chosen is None:
            return data, "none"
        spec = chosen

    compress, _ = get_codec(spec)
    compressed = compress(data)
    # A payload that grew is stored raw. Incompressible data does exist, and
    # storing it larger than it started would be perverse.
    if len(compressed) >= len(data):
        return data, "none"
    return compressed, spec


def decompress_blob(data: bytes, spec: str) -> bytes:
    """Return *data* decompressed according to *spec*."""
    if spec in ("none", "", None):
        return data
    try:
        _, decompress = get_codec(spec)
    except ValueError as exc:
        msg = f"Cannot read a blob compressed with {spec!r}: {exc}"
        raise SerializationError(msg) from exc
    return decompress(data)


def _probe(data: bytes) -> str | None:
    """Return a compression spec worth using for *data*, or ``None``.

    Compresses at most :data:`SAMPLE_BYTES` and extrapolates. Sampling rather
    than compressing everything is what makes the decision cheap enough to make
    per blob: the probe costs about a millisecond regardless of payload size.
    """
    sample = data[:SAMPLE_BYTES]
    compressed = zlib.compress(sample, PROBE_LEVEL)
    saving = 1.0 - (len(compressed) / len(sample))
    if saving < MIN_SAVING:
        return None
    return f"zlib:{PROBE_LEVEL}"


def describe_available() -> dict[str, Any]:
    """Return which compression families can be used in this environment."""
    available = {}
    for family in _FAMILIES:
        try:
            get_codec(family)
        except ImportError:
            available[family] = False
        else:
            available[family] = True
    return available
