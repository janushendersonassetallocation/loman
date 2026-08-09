"""Dtype-faithful encoding of one-dimensional arrays and pandas indexes.

Format version 1 encoded frames with ``DataFrame.values.tolist()``.  That has
three problems.  It goes through ``object`` whenever columns have mixed dtypes,
so per-column types are lost and have to be guessed back via ``astype``.  It
produces one Python object per element, which is both the slowest and the
largest possible JSON encoding.  And it silently fails outright on
``datetime64`` columns, because ``tolist()`` yields ``Timestamp`` objects that no
transformer handles.

This module encodes column-wise instead, dispatching on dtype.  Each encoding
is self-describing via its ``kind`` field, so a decoder never needs to know
which format version produced it — v1 payloads are recognised by their absence.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .transformer import Transformer

KEY_KIND = "kind"

KIND_PLAIN = "plain"
KIND_OBJECT = "object"
KIND_DATETIME = "datetime"
KIND_TIMEDELTA = "timedelta"
KIND_CATEGORY = "category"
KIND_MASKED = "masked"

KIND_INDEX = "index"
KIND_MULTIINDEX = "multiindex"

# numpy stores NaT as the smallest int64; we write JSON null instead, both
# because it survives a round trip through other tools and because a reader
# looking at the file can see what it means.
_NAT = np.iinfo(np.int64).min


def _is_json_native_dtype(dtype: np.dtype) -> bool:
    """True for dtypes whose elements JSON can hold without a wrapper."""
    return dtype.kind in ("i", "u", "f", "b")


def encode_1d(transformer: Transformer, values: Any, dtype: Any = None) -> dict[str, Any]:
    """Encode a 1-D array-like, preserving its dtype exactly.

    *values* may be a numpy array, a pandas Series, or a pandas extension
    array.  *dtype* overrides the inferred dtype, which matters for extension
    types whose values array does not carry it.
    """
    if isinstance(values, pd.Series):
        if dtype is None:
            dtype = values.dtype
        values = values.array if _is_extension(values.dtype) else values.to_numpy()
    if dtype is None:
        dtype = getattr(values, "dtype", None)

    # Timezone-aware datetimes are an extension dtype, but encoding them
    # element-wise would be both lossy and enormous.  Store the underlying UTC
    # integers and the zone name instead, so the branch below never sees them.
    if isinstance(dtype, pd.DatetimeTZDtype):
        utc = pd.DatetimeIndex(values).tz_convert("UTC").tz_localize(None)
        return {
            KEY_KIND: KIND_DATETIME,
            "dtype": f"datetime64[{dtype.unit}]",
            "tz": str(dtype.tz),
            "data": _encode_int64_with_nat(np.asarray(utc)),
        }

    # Categorical: store the categories once and integer codes per element,
    # which is both smaller and lossless about ordering.
    if isinstance(dtype, pd.CategoricalDtype):
        cat = values if isinstance(values, pd.Categorical) else pd.Categorical(values, dtype=dtype)
        return {
            KEY_KIND: KIND_CATEGORY,
            "categories": encode_1d(transformer, np.asarray(cat.categories)),
            "ordered": bool(cat.ordered),
            "codes": [int(c) for c in cat.codes],
        }

    # Pandas nullable extension dtypes (Int64, Float64, boolean, string).
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        arr = pd.array(values, dtype=dtype) if not isinstance(values, pd.api.extensions.ExtensionArray) else values
        return {
            KEY_KIND: KIND_MASKED,
            "dtype": str(dtype),
            "data": [None if v is pd.NA or v is None else transformer.to_dict(_py(v)) for v in arr],
        }

    arr = np.asarray(values)
    dtype = arr.dtype

    if dtype.kind == "M":
        return {
            KEY_KIND: KIND_DATETIME,
            "dtype": str(dtype),
            "data": _encode_int64_with_nat(arr),
        }

    if dtype.kind == "m":
        return {
            KEY_KIND: KIND_TIMEDELTA,
            "dtype": str(dtype),
            "data": _encode_int64_with_nat(arr),
        }

    if _is_json_native_dtype(dtype):
        return {KEY_KIND: KIND_PLAIN, "dtype": dtype.str, "data": arr.tolist()}

    # Strings, objects, anything else: hand each element to the transformer so
    # registered custom types keep working inside frames and arrays.
    return {
        KEY_KIND: KIND_OBJECT,
        "dtype": dtype.str,
        "data": [transformer.to_dict(v) for v in arr.tolist()],
    }


def _py(v: Any) -> Any:
    """Unwrap a numpy scalar to its Python equivalent."""
    return v.item() if isinstance(v, np.generic) else v


def _encode_int64_with_nat(arr: np.ndarray) -> list[int | None]:
    """Encode a datetime64/timedelta64 array as integers, with null for NaT."""
    ints = arr.view("int64")
    return [None if int(i) == _NAT else int(i) for i in ints]


def _decode_int64_with_nat(data: list[int | None], dtype: str) -> np.ndarray:
    """Rebuild a datetime64/timedelta64 array from integers and nulls."""
    ints = np.array([_NAT if v is None else int(v) for v in data], dtype="int64")
    return ints.view(np.dtype(dtype))


def decode_1d(transformer: Transformer, d: dict[str, Any]) -> Any:
    """Rebuild a 1-D array from :func:`encode_1d` output."""
    kind = d[KEY_KIND]

    if kind == KIND_PLAIN:
        return np.array(d["data"], dtype=np.dtype(d["dtype"]))

    if kind == KIND_DATETIME:
        naive = _decode_int64_with_nat(d["data"], d["dtype"])
        tz = d.get("tz")
        if tz is None:
            return naive
        return pd.DatetimeIndex(naive).tz_localize("UTC").tz_convert(tz).array

    if kind == KIND_TIMEDELTA:
        return _decode_int64_with_nat(d["data"], d["dtype"])

    if kind == KIND_CATEGORY:
        categories = decode_1d(transformer, d["categories"])
        codes = np.array(d["codes"], dtype="int64")
        return pd.Categorical.from_codes(codes, categories=categories, ordered=d["ordered"])

    if kind == KIND_MASKED:
        data = [None if v is None else transformer.from_dict(v) for v in d["data"]]
        try:
            return pd.array(data, dtype=d["dtype"])
        except (TypeError, ValueError):
            # The extension dtype names pandas recognises change between major
            # versions — 'str' is a dtype in pandas 3 but not in 2.  A file
            # written by a different pandas should still load with an inferred
            # dtype rather than fail the whole read.
            return pd.array(data)

    if kind == KIND_OBJECT:
        decoded = [transformer.from_dict(v) for v in d["data"]]
        dtype = np.dtype(d["dtype"])
        if dtype.kind != "O":
            # Fixed-width strings and the like: rebuild at the original dtype
            # rather than letting everything collapse to object.
            return np.array(decoded, dtype=dtype)
        out = np.empty(len(decoded), dtype=object)
        out[:] = decoded
        return out

    msg = f"Unknown array encoding kind {kind!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def encode_index(transformer: Transformer, index: pd.Index) -> dict[str, Any]:
    """Encode a pandas Index, including MultiIndex, without losing its dtype."""
    if isinstance(index, pd.MultiIndex):
        return {
            KEY_KIND: KIND_MULTIINDEX,
            "names": list(index.names),
            "levels": [encode_1d(transformer, np.asarray(lvl)) for lvl in index.levels],
            "codes": [[int(c) for c in level_codes] for level_codes in index.codes],
        }

    tz = getattr(index, "tz", None)
    encoded: dict[str, Any] = {
        KEY_KIND: KIND_INDEX,
        "name": transformer.to_dict(index.name),
        "data": encode_1d(transformer, index.values, dtype=index.dtype if _is_extension(index.dtype) else None),
    }
    if tz is not None:
        encoded["tz"] = str(tz)
    # A regular index (e.g. from date_range) carries a frequency that is part of
    # its identity — comparisons treat an index without it as different.
    freq = getattr(index, "freqstr", None)
    if freq is not None:
        encoded["freq"] = freq
    return encoded


def _is_extension(dtype: Any) -> bool:
    """True for pandas extension dtypes, which numpy cannot represent."""
    return isinstance(dtype, pd.api.extensions.ExtensionDtype)


def decode_index(transformer: Transformer, d: dict[str, Any]) -> pd.Index:
    """Rebuild a pandas Index from :func:`encode_index` output."""
    kind = d[KEY_KIND]

    if kind == KIND_MULTIINDEX:
        levels = [decode_1d(transformer, lvl) for lvl in d["levels"]]
        return pd.MultiIndex(levels=levels, codes=d["codes"], names=d["names"])

    values = decode_1d(transformer, d["data"])
    index = pd.Index(values, name=transformer.from_dict(d["name"]))

    tz = d.get("tz")
    if tz is not None and isinstance(index, pd.DatetimeIndex):
        index = index.tz_localize("UTC").tz_convert(tz)

    freq = d.get("freq")
    if freq is not None and isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        # Only regular index types carry a frequency, and only if the values
        # genuinely conform; a mismatch is not worth failing a whole read over.
        with contextlib.suppress(ValueError, TypeError):
            index.freq = pd.tseries.frequencies.to_offset(freq)
    return index
