"""Transformers for the everyday value types a computation holds.

These cover the types that reach a node value in practice but that the original
transformer set did not handle: dates and times, pandas indexes, numpy scalars,
sets, bytes and decimals. Before this module a DataFrame with a ``DatetimeIndex``
could not be serialized at all, because the index was encoded element by element
and no transformer claimed :class:`pandas.Timestamp`.

Two encoding principles run through the module:

*Exactness over readability for temporal types.* A timestamp is stored as an
integer nanosecond count plus a timezone rather than as a formatted string, so a
round-trip is bit-exact and does not depend on parsing rules.

*Whole indexes, not element sequences.* An index is encoded as an index, which
keeps a ``MultiIndex`` a ``MultiIndex`` and turns the default ``RangeIndex`` of a
100k-row frame into four numbers instead of 100k of them.
"""

import base64
import contextlib
import datetime
import decimal
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .transformer import CustomTransformer

if TYPE_CHECKING:
    from .transformer import Transformer


class DateTimeTransformer(CustomTransformer):
    """Transformer for :class:`datetime.datetime`.

    Encoded as an ISO 8601 string, which carries a UTC offset for aware values.
    The IANA zone name is kept alongside it when there is one, because an offset
    alone cannot distinguish ``Europe/London`` in winter from ``UTC``.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "datetime"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a datetime as ISO 8601, keeping the zone name when present."""
        assert isinstance(o, datetime.datetime)  # noqa: S101
        d: dict[str, Any] = {"iso": o.isoformat()}
        tzname = getattr(o.tzinfo, "key", None)  # zoneinfo.ZoneInfo
        if tzname is not None:
            d["tz"] = tzname
        return d

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a datetime, restoring its named zone when recorded."""
        dt = datetime.datetime.fromisoformat(d["iso"])
        tzname = d.get("tz")
        if tzname is not None:
            from zoneinfo import ZoneInfo

            dt = dt.astimezone(ZoneInfo(tzname))
        return dt

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled; pandas Timestamps go elsewhere."""
        return [datetime.datetime]


class DateTransformer(CustomTransformer):
    """Transformer for :class:`datetime.date`."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "date"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a date as an ISO 8601 string."""
        assert isinstance(o, datetime.date)  # noqa: S101
        return {"iso": o.isoformat()}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a date from its ISO 8601 string."""
        return datetime.date.fromisoformat(d["iso"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [datetime.date]


class TimeTransformer(CustomTransformer):
    """Transformer for :class:`datetime.time`."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "time"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a time as an ISO 8601 string."""
        assert isinstance(o, datetime.time)  # noqa: S101
        return {"iso": o.isoformat()}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a time from its ISO 8601 string."""
        return datetime.time.fromisoformat(d["iso"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [datetime.time]


class TimeDeltaTransformer(CustomTransformer):
    """Transformer for :class:`datetime.timedelta`.

    Stored as the three components the type is normalised into, rather than as
    total seconds, which would lose microsecond precision over long spans.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "timedelta"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a timedelta as its normalised day/second/microsecond parts."""
        assert isinstance(o, datetime.timedelta)  # noqa: S101
        return {"days": o.days, "seconds": o.seconds, "microseconds": o.microseconds}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a timedelta from its parts."""
        return datetime.timedelta(days=d["days"], seconds=d["seconds"], microseconds=d["microseconds"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [datetime.timedelta]


class TimestampTransformer(CustomTransformer):
    """Transformer for :class:`pandas.Timestamp`.

    :class:`pandas.Timestamp` subclasses :class:`datetime.datetime`, so it needs
    its own registration: exact-type dispatch would otherwise never reach the
    datetime transformer, and going through ISO strings would lose nanoseconds.
    The integer ``value`` is nanoseconds since the epoch, always UTC.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "timestamp"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a Timestamp as epoch nanoseconds plus timezone and resolution.

        ``Timestamp.value`` is always nanoseconds, but the resolution the value
        was held at is a separate property and is preserved so that a
        microsecond timestamp does not come back claiming nanosecond precision.
        """
        assert isinstance(o, pd.Timestamp)  # noqa: S101
        d: dict[str, Any] = {"value": o.value, "unit": o.unit}
        if o.tz is not None:
            d["tz"] = str(o.tz)
        return d

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a Timestamp, re-applying its timezone and resolution."""
        ts = pd.Timestamp(d["value"])
        tz = d.get("tz")
        if tz is not None:
            ts = ts.tz_localize("UTC").tz_convert(tz)
        unit = d.get("unit")
        if unit is not None:
            ts = ts.as_unit(unit)
        return ts

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [pd.Timestamp]


class PandasTimedeltaTransformer(CustomTransformer):
    """Transformer for :class:`pandas.Timedelta`, stored as nanoseconds."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "pd_timedelta"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a pandas Timedelta as a nanosecond count plus its resolution."""
        assert isinstance(o, pd.Timedelta)  # noqa: S101
        return {"value": o.value, "unit": o.unit}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a pandas Timedelta from its nanosecond count."""
        td = pd.Timedelta(d["value"])
        unit = d.get("unit")
        if unit is not None:
            td = td.as_unit(unit)
        return td

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [pd.Timedelta]


class NaTTransformer(CustomTransformer):
    """Transformer for :data:`pandas.NaT`.

    ``NaT`` is a singleton of its own type rather than a Timestamp, so nothing
    else claims it, and it would otherwise fail as an unknown type inside any
    datetime column holding a missing value.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "nat"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode NaT as an empty marker."""
        return {}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Return the NaT singleton."""
        return pd.NaT

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the NaT singleton's type."""
        return [type(pd.NaT)]


class NumpyScalarTransformer(CustomTransformer):
    """Transformer for numpy scalar types (:class:`numpy.generic`).

    Registered against the ``np.generic`` base so every width is covered by one
    transformer. The dtype string is kept so ``np.int32`` does not come back as
    ``np.int64``, and ``np.float64`` does not silently degrade to a plain float.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "npscalar"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a numpy scalar as its dtype plus a plain Python value."""
        assert isinstance(o, np.generic)  # noqa: S101
        dtype = o.dtype
        if dtype.kind in "Mm":
            # datetime64 / timedelta64: the integer tick count is exact, and
            # .item() would hand back a datetime whose unit had been forgotten.
            return {"dtype": dtype.str, "value": int(o.view("int64"))}
        return {"dtype": dtype.str, "value": transformer.to_dict(o.item())}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a numpy scalar of the recorded dtype."""
        dtype = np.dtype(d["dtype"])
        if dtype.kind in "Mm":
            return np.int64(d["value"]).view(dtype)
        return dtype.type(transformer.from_dict(d["value"]))

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match every numpy scalar type."""
        return [np.generic]


class SetTransformer(CustomTransformer):
    """Transformer for :class:`set` and :class:`frozenset`.

    Members are sorted by their encoded form where possible so that two equal
    sets serialize identically --- without it, a saved file would differ run to
    run with hash randomisation, defeating byte-level comparison of two saves.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "set"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a set as a list of encoded members plus its mutability."""
        assert isinstance(o, (set, frozenset))  # noqa: S101
        values = [transformer.to_dict(x) for x in o]
        with contextlib.suppress(TypeError):  # pragma: no cover - unorderable encodings
            values.sort(key=_sort_key)
        return {"values": values, "frozen": isinstance(o, frozenset)}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a set or frozenset from its members."""
        values = (transformer.from_dict(x) for x in d["values"])
        return frozenset(values) if d.get("frozen") else set(values)

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact types handled."""
        return [set, frozenset]


def _sort_key(encoded: Any) -> tuple[str, str]:
    """Return a total ordering key for an encoded value.

    Encoded members can be scalars or dicts, which do not compare with one
    another, so ordering falls back to the type name and the repr.
    """
    return (type(encoded).__name__, repr(encoded))


class BytesTransformer(CustomTransformer):
    """Transformer for :class:`bytes` and :class:`bytearray`, as base64."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "bytes"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a byte string as base64 ASCII."""
        assert isinstance(o, (bytes, bytearray))  # noqa: S101
        return {"b64": base64.b64encode(bytes(o)).decode("ascii"), "mutable": isinstance(o, bytearray)}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a byte string from base64."""
        raw = base64.b64decode(d["b64"].encode("ascii"))
        return bytearray(raw) if d.get("mutable") else raw

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact types handled."""
        return [bytes, bytearray]


class DecimalTransformer(CustomTransformer):
    """Transformer for :class:`decimal.Decimal`, stored as its exact string."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "decimal"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode a Decimal as the string that reproduces it exactly."""
        assert isinstance(o, decimal.Decimal)  # noqa: S101
        return {"value": str(o)}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a Decimal from its string form."""
        return decimal.Decimal(d["value"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the exact type handled."""
        return [decimal.Decimal]


class IndexTransformer(CustomTransformer):
    """Transformer for :class:`pandas.Index` and its subclasses.

    Encoding an index as an index rather than as a list of its elements is what
    keeps a ``MultiIndex`` a ``MultiIndex`` --- previously it came back as a flat
    ``Index`` of tuples --- and what stops the default ``RangeIndex`` of a large
    frame from being written out one integer at a time.

    Five shapes are recognised, discriminated by ``kind``: ``range``,
    ``datetime``, ``timedelta``, ``multi`` and ``base``. A ``MultiIndex``
    encodes its levels recursively as indexes, so a level that is itself a
    ``DatetimeIndex`` keeps its type and timezone.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "index"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode an index according to its concrete pandas type."""
        assert isinstance(o, pd.Index)  # noqa: S101
        if isinstance(o, pd.MultiIndex):
            return {
                "kind": "multi",
                "levels": [transformer.to_dict(level) for level in o.levels],
                "codes": [list(map(int, codes)) for codes in o.codes],
                "names": list(o.names),
            }
        if isinstance(o, pd.RangeIndex):
            return {
                "kind": "range",
                "start": int(o.start),
                "stop": int(o.stop),
                "step": int(o.step),
                "name": transformer.to_dict(o.name),
            }
        if isinstance(o, pd.DatetimeIndex):
            # asi8 counts ticks in the index's own resolution, which is
            # microseconds by default in pandas 3 and nanoseconds before it.
            # Recording the unit is what keeps the values from being reread at
            # the wrong scale.
            return {
                "kind": "datetime",
                "values": transformer.to_dict(o.asi8),
                "unit": o.unit,
                "tz": str(o.tz) if o.tz is not None else None,
                "freq": o.freqstr if o.freq is not None else None,
                "name": transformer.to_dict(o.name),
            }
        if isinstance(o, pd.TimedeltaIndex):
            return {
                "kind": "timedelta",
                "values": transformer.to_dict(o.asi8),
                "unit": o.unit,
                "freq": o.freqstr if o.freq is not None else None,
                "name": transformer.to_dict(o.name),
            }
        return {
            "kind": "base",
            "dtype": str(o.dtype),
            "values": _encode_index_values(transformer, o),
            "name": transformer.to_dict(o.name),
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct an index of the recorded kind."""
        kind = d["kind"]
        if kind == "multi":
            levels = [transformer.from_dict(level) for level in d["levels"]]
            return pd.MultiIndex(levels=levels, codes=d["codes"], names=d["names"])
        name = transformer.from_dict(d.get("name"))
        if kind == "range":
            return pd.RangeIndex(start=d["start"], stop=d["stop"], step=d["step"], name=name)
        if kind == "datetime":
            unit = d.get("unit", "ns")
            values = np.asarray(transformer.from_dict(d["values"]), dtype="int64").view(f"datetime64[{unit}]")
            idx = pd.DatetimeIndex(values, name=name)
            tz = d.get("tz")
            if tz is not None:
                idx = idx.tz_localize("UTC").tz_convert(tz)
            return _restore_freq(idx, d.get("freq"))
        if kind == "timedelta":
            unit = d.get("unit", "ns")
            values = np.asarray(transformer.from_dict(d["values"]), dtype="int64").view(f"timedelta64[{unit}]")
            return _restore_freq(pd.TimedeltaIndex(values, name=name), d.get("freq"))
        return pd.Index(transformer.from_dict(d["values"]), dtype=d["dtype"], name=name)

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match every pandas Index subclass."""
        return [pd.Index]


def _encode_index_values(transformer: "Transformer", index: "pd.Index") -> Any:
    """Encode a plain index's values, as an array where that is lossless.

    Same reasoning as for DataFrame columns: a numpy-backed index goes through
    the ndarray transformer, which lets a large one be written out of line
    rather than one number at a time in the manifest. Extension dtypes go
    element-wise, since their numpy form is not the same thing.
    """
    dtype = index.dtype
    if isinstance(dtype, np.dtype) and not dtype.hasobject:
        return transformer.to_dict(index.to_numpy())
    return transformer.to_dict(index.tolist())


def _restore_freq(index: Any, freq: str | None) -> Any:
    """Re-apply a recorded frequency to *index*, ignoring one that no longer fits.

    ``freq`` is a derived property: it describes a regular spacing the values
    already have. Setting it cannot change the data, and a value that does not
    match --- which pandas rejects --- is better dropped than fatal.
    """
    if freq is None:
        return index
    with contextlib.suppress(ValueError, TypeError):  # pragma: no cover - defensive
        index.freq = freq
    return index


VALUE_TRANSFORMERS: list[type[CustomTransformer]] = [
    DateTimeTransformer,
    DateTransformer,
    TimeTransformer,
    TimeDeltaTransformer,
    TimestampTransformer,
    PandasTimedeltaTransformer,
    NaTTransformer,
    NumpyScalarTransformer,
    SetTransformer,
    BytesTransformer,
    DecimalTransformer,
    IndexTransformer,
]


def register_value_transformers(t: "Transformer") -> None:
    """Register every everyday-value transformer on *t*."""
    for transformer_cls in VALUE_TRANSFORMERS:
        t.register(transformer_cls())
