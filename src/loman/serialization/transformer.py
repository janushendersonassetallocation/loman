"""Object serialization and transformation framework."""

import contextlib
import copy
import dataclasses
import datetime
import graphlib
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .arraycodec import decode_1d, decode_index, encode_1d, encode_index

try:
    import attrs

    HAS_ATTRS = True
except ImportError:  # pragma: no cover
    HAS_ATTRS = False

KEY_TYPE = "type"
# Marks a reference standing in for a value stored outside the document.
# Deliberately not a "type" discriminator: it is resolved before type dispatch,
# so it can never collide with a registered transformer name.
PAYLOAD_MARKER = "__loman_payload__"
KEY_CLASS = "class"
KEY_VALUES = "values"
KEY_DATA = "data"

TYPENAME_DICT = "dict"
TYPENAME_TUPLE = "tuple"
TYPENAME_TRANSFORMABLE = "transformable"
TYPENAME_ATTRS = "attrs"
TYPENAME_DATACLASS = "dataclass"


class UntransformableTypeError(Exception):
    """Exception raised when a type cannot be transformed for serialization."""

    pass


class UnrecognizedTypeError(Exception):
    """Exception raised when a type is not recognized during transformation."""

    pass


class MissingObject:
    """Sentinel object representing missing or unset values."""

    def __repr__(self) -> str:
        """Return string representation of missing object."""
        return "Missing"


def order_classes(classes: Iterable[type]) -> list[type]:
    """Order classes by inheritance hierarchy using topological sort."""
    graph: dict[type, set[type]] = {x: set() for x in classes}
    for x in classes:
        for y in classes:
            if issubclass(x, y) and x != y:
                graph[y].add(x)
    ts = graphlib.TopologicalSorter(graph)
    return list(ts.static_order())


class CustomTransformer(ABC):
    """Abstract base class for custom object transformers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return unique name identifier for this transformer."""
        pass  # pragma: no cover

    @abstractmethod
    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Convert object to dictionary representation."""
        pass  # pragma: no cover

    @abstractmethod
    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct object from dictionary representation."""
        pass  # pragma: no cover

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return types that this transformer handles directly."""
        return []

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Return base types whose subtypes this transformer can handle."""
        return []


class Transformable(ABC):
    """Abstract base class for objects that can transform themselves."""

    @abstractmethod
    def to_dict(self, transformer: "Transformer") -> dict[str, Any]:
        """Convert this object to dictionary representation."""
        pass  # pragma: no cover

    @classmethod
    @abstractmethod
    def from_dict(cls, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct object from dictionary representation."""
        pass  # pragma: no cover


class PayloadSink(ABC):
    """Somewhere bulky values can be stored outside the document.

    A transformer with a sink attached replaces every value the sink accepts
    with a small reference, wherever that value appears — a node's value, an
    element of a list, a field of a dataclass, a value in a dict.  Deciding
    this at the transformer rather than at the node keeps a frame buried
    three levels down from being inlined as JSON.
    """

    @abstractmethod
    def should_offload(self, o: object) -> bool:
        """True when *o* is worth storing outside the document."""
        pass  # pragma: no cover

    @abstractmethod
    def write(self, o: object) -> dict[str, Any]:
        """Store *o* and return the reference that stands in for it."""
        pass  # pragma: no cover


class PayloadSource(ABC):
    """The read side of a :class:`PayloadSink`."""

    @abstractmethod
    def read(self, ref: dict[str, Any]) -> object:
        """Materialise the value *ref* points at."""
        pass  # pragma: no cover


class Transformer:
    """Main transformer class for object serialization and deserialization."""

    def __init__(self, *, strict: bool = True) -> None:
        """Initialize transformer with strict mode setting."""
        self.strict = strict

        self._direct_type_map: dict[type, CustomTransformer] = {}
        self._subtype_order: list[type] = []
        self._subtype_map: dict[type, CustomTransformer] = {}
        self._transformers: dict[str, CustomTransformer] = {}
        self._transformable_types: dict[str, type[Transformable]] = {}
        self._attrs_types: dict[str, type] = {}
        self._dataclass_types: dict[str, type] = {}

        self._payload_sink: PayloadSink | None = None
        self._payload_source: PayloadSource | None = None

    def with_payloads(
        self,
        *,
        sink: PayloadSink | None = None,
        source: PayloadSource | None = None,
    ) -> "Transformer":
        """Return a view of this transformer that redirects bulky values.

        A shallow copy, so the type registries are shared rather than rebuilt —
        they are only read during an operation.  Returning a copy instead of
        mutating in place is what lets one serializer serve several concurrent
        reads and writes without them treading on each other.
        """
        clone = copy.copy(self)
        clone._payload_sink = sink
        clone._payload_source = source
        return clone

    def register(self, t: CustomTransformer | type[Transformable] | type) -> None:
        """Register a transformer, transformable type, or regular type."""
        if isinstance(t, CustomTransformer):
            self.register_transformer(t)
        elif isinstance(t, type) and issubclass(t, Transformable):
            self.register_transformable(t)
        elif HAS_ATTRS and isinstance(t, type) and attrs.has(t):
            self.register_attrs(t)
        elif isinstance(t, type) and dataclasses.is_dataclass(t):
            self.register_dataclass(t)
        else:
            msg = f"Unable to register {t}"
            raise ValueError(msg)

    def register_transformer(self, transformer: CustomTransformer) -> None:
        """Register a custom transformer for specific types."""
        assert transformer.name not in self._transformers  # noqa: S101
        for type_ in transformer.supported_direct_types:
            assert type_ not in self._direct_type_map  # noqa: S101
        for type_ in transformer.supported_subtypes:
            assert type_ not in self._subtype_map  # noqa: S101

        self._transformers[transformer.name] = transformer

        for type_ in transformer.supported_direct_types:
            self._direct_type_map[type_] = transformer

        contains_supported_subtypes = False
        for type_ in transformer.supported_subtypes:
            contains_supported_subtypes = True
            self._subtype_map[type_] = transformer
        if contains_supported_subtypes:
            self._subtype_order = order_classes(self._subtype_map.keys())

    def register_transformable(self, transformable_type: type[Transformable]) -> None:
        """Register a transformable type that can serialize itself."""
        name = transformable_type.__name__
        assert name not in self._transformable_types  # noqa: S101
        self._transformable_types[name] = transformable_type

    def register_attrs(self, attrs_type: type) -> None:
        """Register an attrs-decorated class for serialization."""
        name = attrs_type.__name__
        assert name not in self._attrs_types  # noqa: S101
        self._attrs_types[name] = attrs_type

    def register_dataclass(self, dataclass_type: type) -> None:
        """Register a dataclass for serialization."""
        name = dataclass_type.__name__
        assert name not in self._dataclass_types  # noqa: S101
        self._dataclass_types[name] = dataclass_type

    def get_transformer_for_obj(self, obj: object) -> CustomTransformer | None:
        """Get the appropriate transformer for a given object."""
        transformer = self._direct_type_map.get(type(obj))
        if transformer is not None:
            return transformer
        for tp in self._subtype_order:
            if isinstance(obj, tp):
                return self._subtype_map[tp]
        return None

    def get_transformer_for_name(self, name: str) -> CustomTransformer | None:
        """Get a transformer by its registered name."""
        transformer = self._transformers.get(name)
        return transformer

    def to_dict(self, o: object) -> Any:
        """Convert an object to a serializable dictionary representation."""
        # Checked before anything else so a bulky value is redirected wherever
        # it appears, not merely when it happens to be a whole node's value.
        # The attribute test is what keeps this free for the common case of no
        # sink at all, since this runs once per element of every container.
        if self._payload_sink is not None and self._payload_sink.should_offload(o):
            return self._payload_sink.write(o)

        if isinstance(o, str) or o is None or o is True or o is False or isinstance(o, (int, float)):
            return o
        elif isinstance(o, tuple):
            return {KEY_TYPE: TYPENAME_TUPLE, KEY_VALUES: [self.to_dict(x) for x in o]}
        elif isinstance(o, list):
            return [self.to_dict(x) for x in o]
        elif isinstance(o, dict):
            return self._dict_to_dict(o)
        # Check registered custom transformers before generic dataclass/attrs paths
        # so that explicitly registered types (e.g. NodeKey) take priority.
        elif self.get_transformer_for_obj(o) is not None:
            return self._to_dict_transformer(o)
        elif isinstance(o, Transformable):
            return {KEY_TYPE: TYPENAME_TRANSFORMABLE, KEY_CLASS: type(o).__name__, KEY_DATA: o.to_dict(self)}
        elif HAS_ATTRS and attrs.has(type(o)):
            return self._attrs_to_dict(o)
        elif dataclasses.is_dataclass(o) and not isinstance(o, type):
            return self._dataclass_to_dict(o)
        else:
            return self._to_dict_transformer(o)

    def _dict_to_dict(self, o: dict[Any, Any]) -> dict[str, Any]:
        """Convert a dictionary to serializable form."""
        d = {k: self.to_dict(v) for k, v in o.items()}
        if KEY_TYPE in o:
            return {KEY_TYPE: TYPENAME_DICT, KEY_DATA: d}
        else:
            return d

    def _attrs_to_dict(self, o: object) -> dict[str, Any]:
        """Convert an attrs object to serializable dictionary form."""
        data: dict[str, Any] = {}
        for a in o.__attrs_attrs__:  # type: ignore[attr-defined]
            data[a.name] = self.to_dict(o.__getattribute__(a.name))
        res: dict[str, Any] = {KEY_TYPE: TYPENAME_ATTRS, KEY_CLASS: type(o).__name__}
        if len(data) > 0:
            res[KEY_DATA] = data
        return res

    def _dataclass_to_dict(self, o: object) -> dict[str, Any]:
        """Convert a dataclass object to serializable dictionary form."""
        data: dict[str, Any] = {}
        for f in dataclasses.fields(o):  # type: ignore[arg-type]
            data[f.name] = self.to_dict(getattr(o, f.name))
        res: dict[str, Any] = {KEY_TYPE: TYPENAME_DATACLASS, KEY_CLASS: type(o).__name__}
        if len(data) > 0:
            res[KEY_DATA] = data
        return res

    def _to_dict_transformer(self, o: object) -> dict[str, Any] | None:
        """Convert an object using a registered custom transformer."""
        transformer = self.get_transformer_for_obj(o)
        if transformer is None:
            if self.strict:
                msg = f"Could not transform object of type {type(o).__name__}"
                raise UntransformableTypeError(msg)
            else:
                return None
        d = transformer.to_dict(self, o)
        d[KEY_TYPE] = transformer.name
        return d

    def from_dict(self, d: Any) -> Any:
        """Convert a dictionary representation back to the original object."""
        if isinstance(d, str) or d is None or d is True or d is False or isinstance(d, (int, float)):
            return d
        elif isinstance(d, list):
            return [self.from_dict(x) for x in d]
        elif isinstance(d, dict):
            # A payload reference carries no "type", so it has to be recognised
            # before the plain-dict branch would recurse into its fields.
            if d.get(PAYLOAD_MARKER):
                if self._payload_source is None:
                    msg = (
                        "This document stores values outside itself, but no payload source "
                        "was supplied to read them. Use Computation.read_archive rather than "
                        "read_json."
                    )
                    raise UnrecognizedTypeError(msg)
                return self._payload_source.read(d)

            type_ = d.get(KEY_TYPE)
            if type_ is None:
                return {k: self.from_dict(v) for k, v in d.items()}
            elif type_ == TYPENAME_TUPLE:
                return tuple(self.from_dict(x) for x in d[KEY_VALUES])
            elif type_ == TYPENAME_DICT:
                return {k: self.from_dict(v) for k, v in d[KEY_DATA].items()}
            elif type_ == TYPENAME_TRANSFORMABLE:
                return self._from_dict_transformable(d)
            elif type_ == TYPENAME_ATTRS:
                return self._from_attrs(d)
            elif type_ == TYPENAME_DATACLASS:
                return self._from_dataclass(d)
            else:
                return self._from_dict_transformer(type_, d)
        else:
            msg = "Unable to determine object type from dictionary"
            raise ValueError(msg)

    def _from_dict_transformable(self, d: dict[str, Any]) -> object:
        """Reconstruct a Transformable object from dictionary form."""
        classname = d[KEY_CLASS]
        cls = self._transformable_types.get(classname)
        if cls is None:
            if self.strict:
                msg = f"Unable to transform Transformable object of class {classname}"
                raise UnrecognizedTypeError(msg)
            else:
                return MissingObject()
        else:
            return cls.from_dict(self, d[KEY_DATA])

    def _from_attrs(self, d: dict[str, Any]) -> object:
        """Reconstruct an attrs object from dictionary form."""
        if not HAS_ATTRS:  # pragma: no cover
            if self.strict:
                msg = "attrs package not installed"
                raise UnrecognizedTypeError(msg)
            return MissingObject()
        cls = self._attrs_types.get(d[KEY_CLASS])
        if cls is None:
            if self.strict:
                msg = f"Unable to create attrs object of type {cls}"
                raise UnrecognizedTypeError(msg)
            else:
                return MissingObject()
        else:
            kwargs: dict[str, Any] = {}
            if KEY_DATA in d:
                for key, value in d[KEY_DATA].items():
                    kwargs[key] = self.from_dict(value)
            return cls(**kwargs)

    def _from_dataclass(self, d: dict[str, Any]) -> object:
        """Reconstruct a dataclass object from dictionary form."""
        cls = self._dataclass_types.get(d[KEY_CLASS])
        if cls is None:
            if self.strict:
                msg = f"Unable to create dataclass object of type {cls}"
                raise UnrecognizedTypeError(msg)
            else:
                return MissingObject()
        else:
            kwargs: dict[str, Any] = {}
            if KEY_DATA in d:
                for key, value in d[KEY_DATA].items():
                    kwargs[key] = self.from_dict(value)
            return cls(**kwargs)

    def _from_dict_transformer(self, type_: str, d: dict[str, Any]) -> object:
        """Reconstruct an object using a registered custom transformer."""
        transformer = self.get_transformer_for_name(type_)
        if transformer is None:
            if self.strict:
                msg = f"Unable to transform object of type {type_}"
                raise UnrecognizedTypeError(msg)
            else:
                return MissingObject()
        return transformer.from_dict(self, d)


class NdArrayTransformer(CustomTransformer):
    """Transformer for NumPy ndarray objects."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "ndarray"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Convert a numpy array to a dict with shape, dtype and encoded data.

        The data goes through the shared codec, so ``datetime64`` and
        ``timedelta64`` arrays encode as integers rather than failing on the
        ``Timestamp`` objects ``tolist()`` would produce.
        """
        assert isinstance(o, np.ndarray)  # noqa: S101
        return {
            "shape": list(o.shape),
            "dtype": o.dtype.str,
            "values": encode_1d(transformer, np.ravel(o)),
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a numpy array from either the v1 or the codec form."""
        if "values" not in d:
            # Format version 1: a flat JSON list, reshaped and cast.
            return np.array(transformer.from_dict(d["data"]), d["dtype"]).reshape(d["shape"])
        flat = np.asarray(decode_1d(transformer, d["values"]))
        return flat.reshape(d["shape"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return supported numpy array types."""
        return [np.ndarray]


class EnumTransformer(CustomTransformer):
    """Transformer for Enum subclasses.

    Enum classes must be registered via :meth:`register_enum` before use.
    """

    def __init__(self) -> None:
        """Initialise with an empty enum registry."""
        self._registry: dict[str, type[Enum]] = {}

    def register_enum(self, enum_class: type[Enum]) -> None:
        """Register an enum class so its members can be deserialized."""
        self._registry[enum_class.__qualname__] = enum_class

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "enum"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Convert an Enum member to a dict with class qualname and member name."""
        assert isinstance(o, Enum)  # noqa: S101
        return {"enum_class": type(o).__qualname__, "value": o.name}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct an Enum member from its serialized form."""
        enum_class = self._registry.get(d["enum_class"])
        if enum_class is None:
            msg = f"Unknown enum class: {d['enum_class']!r}. Register it with EnumTransformer.register_enum()."
            raise UnrecognizedTypeError(msg)
        return enum_class[d["value"]]

    @property
    def supported_subtypes(self) -> Iterable[type]:
        """Handle all Enum subclasses."""
        return [Enum]


class FunctionRefTransformer(CustomTransformer):
    """Transformer for importable callables (module-level functions and methods).

    Lambdas and closures (whose ``__qualname__`` contains ``<lambda>`` or
    ``<locals>``) are explicitly rejected with a :class:`ValueError`.
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "func_ref"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a callable as its module path and qualname."""
        if not callable(o):
            msg = f"Object {o!r} is not callable"
            raise TypeError(msg)
        qualname = getattr(o, "__qualname__", None)
        module = getattr(o, "__module__", None)
        if qualname is None or module is None:
            msg = f"Cannot serialize {o!r}: missing __qualname__ or __module__"
            raise ValueError(msg)
        if "<lambda>" in qualname:
            msg = f"Cannot serialize lambda function {o!r}: lambdas are not importable"
            raise ValueError(msg)
        if "<locals>" in qualname:
            msg = f"Cannot serialize closure/local function {o!r}: non-importable"
            raise ValueError(msg)
        # Verify the callable is actually reachable via import before committing.
        try:
            mod = importlib.import_module(module)
            obj: Any = mod
            for part in qualname.split("."):
                obj = getattr(obj, part)
            if obj is not o:
                msg = f"Cannot serialize {o!r}: import round-trip returned a different object"
                raise ValueError(msg)
        except (ImportError, AttributeError) as exc:
            msg = f"Cannot serialize {o!r}: not importable ({exc})"
            raise ValueError(msg) from exc
        return {"module": module, "qualname": qualname}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a callable from its module path and qualname."""
        module = importlib.import_module(d["module"])
        obj: Any = module
        for part in d["qualname"].split("."):
            obj = getattr(obj, part)
        return obj

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Register the built-in function types explicitly handled."""
        # We use supported_subtypes for the broad callable match instead,
        # but we must list at least one concrete type here to help dispatch.
        # The broad subtype match on Callable covers everything callable.
        return []

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match all callables via Callable ABC."""
        return [Callable]


class DillFunctionTransformer(CustomTransformer):
    """Transformer that serializes any callable — including lambdas and closures — using dill.

    The callable is serialized with :func:`dill.dumps` and the resulting bytes
    are stored as a base64-encoded string inside the JSON document.  On load the
    bytes are decoded and passed to :func:`dill.loads`.

    .. note::
        The embedded dill blob is **not** portable across Python versions and
        shares the same stability caveats as :meth:`~loman.Computation.write_dill`.
        Register this transformer when convenient lambda/closure round-trips matter
        more than portability.

    Example::

        from loman import Computation, ComputationSerializer
        from loman.serialization import DillFunctionTransformer

        s = ComputationSerializer(use_dill_for_functions=True)
        comp = Computation()
        comp.add_node('a', value=1)
        comp.add_node('b', lambda a: a + 1)
        comp.compute_all()
        comp.write_json('comp.json', serializer=s)
        comp2 = Computation.read_json('comp.json', serializer=s)
        assert comp2.v.b == 2
    """

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "dill_func"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a callable to a base64-encoded dill blob."""
        import base64

        import dill  # nosec B403  # dill is a trusted dependency for this specific use case and most likely be deprecated in the future in favor of a more portable solution, so we allow it here with a blanket nosec directive

        if not callable(o):
            msg = f"Object {o!r} is not callable"
            raise TypeError(msg)
        blob = dill.dumps(o)
        return {"blob": base64.b64encode(blob).decode("ascii")}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a callable from a base64-encoded dill blob."""
        import base64

        import dill  # nosec B403  # dill is a trusted dependency for this specific use case and most likely be deprecated in the future in favor of a more portable solution, so we allow it here with a blanket nosec directive

        blob = base64.b64decode(d["blob"].encode("ascii"))
        return dill.loads(blob)  # noqa: S301  # nosec B301

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """No direct type matches — rely on subtype matching."""
        return []

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match all callables via Callable ABC."""
        return [Callable]


class DataFrameTransformer(CustomTransformer):
    """Transformer for :class:`pandas.DataFrame` objects."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "dataframe"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a DataFrame column by column, preserving each column's dtype."""
        assert isinstance(o, pd.DataFrame)  # noqa: S101
        return {
            "columns": encode_index(transformer, o.columns),
            "index": encode_index(transformer, o.index),
            "cols": [encode_1d(transformer, o.iloc[:, i]) for i in range(o.shape[1])],
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a DataFrame from either the v1 or the column-wise form."""
        if "cols" not in d:
            return self._from_dict_v1(transformer, d)

        columns = decode_index(transformer, d["columns"])
        index = decode_index(transformer, d["index"])
        data = {i: decode_1d(transformer, col) for i, col in enumerate(d["cols"])}
        df = pd.DataFrame(data, index=index, copy=False)
        df.columns = columns
        return df

    def _from_dict_v1(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Decode the format-version-1 split-orientation encoding.

        Retained so files written before the column-wise encoding keep loading.
        Do not extend this; new dtype support belongs in the codec.
        """
        data = transformer.from_dict(d["data"])
        columns = d["columns"]
        index = transformer.from_dict(d["index"])
        dtypes = d.get("dtypes", {})
        df = pd.DataFrame(data, columns=columns, index=index)
        for col, dtype in dtypes.items():
            with contextlib.suppress(ValueError, TypeError):  # pragma: no cover
                df[col] = df[col].astype(dtype)
        return df

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return supported pandas DataFrame type."""
        return [pd.DataFrame]


class SeriesTransformer(CustomTransformer):
    """Transformer for :class:`pandas.Series` objects."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "series"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a Series with its name, index, and dtype-faithful values."""
        assert isinstance(o, pd.Series)  # noqa: S101
        return {
            "name": transformer.to_dict(o.name),
            "index": encode_index(transformer, o.index),
            "values": encode_1d(transformer, o),
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a Series from either the v1 or the codec form."""
        if "values" not in d:
            return self._from_dict_v1(transformer, d)

        values = decode_1d(transformer, d["values"])
        index = decode_index(transformer, d["index"])
        return pd.Series(values, index=index, name=transformer.from_dict(d["name"]), copy=False)

    def _from_dict_v1(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Decode the format-version-1 Series encoding.

        Retained so files written before the codec keep loading.  Do not extend
        this; new dtype support belongs in the codec.
        """
        data = transformer.from_dict(d["data"])
        index = transformer.from_dict(d["index"])
        s = pd.Series(data, index=index, name=d.get("name"))
        with contextlib.suppress(ValueError, TypeError):  # pragma: no cover
            s = s.astype(d["dtype"])
        return s

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return supported pandas Series type."""
        return [pd.Series]


class DateTimeTransformer(CustomTransformer):
    """Transformer for scalar date, time and datetime values.

    Covers :class:`datetime.datetime`, :class:`datetime.date`,
    :class:`datetime.time`, :class:`datetime.timedelta` and their pandas
    counterparts.  Without this, a lone timestamp — as a node value, or nested
    inside a list or dict — has no encoding at all.

    Values are stored as ISO 8601 strings, which stay readable in the JSON and
    are unambiguous about timezone offset.
    """

    _KIND_DATETIME = "datetime"
    _KIND_DATE = "date"
    _KIND_TIME = "time"
    _KIND_TIMEDELTA = "timedelta"
    _KIND_PD_TIMEDELTA = "pd_timedelta"
    _KIND_TIMESTAMP = "timestamp"

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "datetime"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a temporal scalar as an ISO 8601 string."""
        # pandas types first: Timestamp subclasses datetime, Timedelta
        # subclasses timedelta, and both round-trip better through pandas.
        if isinstance(o, pd.Timestamp):
            return {"kind": self._KIND_TIMESTAMP, "value": o.isoformat()}
        if isinstance(o, pd.Timedelta):
            return {"kind": self._KIND_PD_TIMEDELTA, "value": o.isoformat()}
        if isinstance(o, datetime.datetime):
            return {"kind": self._KIND_DATETIME, "value": o.isoformat()}
        if isinstance(o, datetime.date):
            return {"kind": self._KIND_DATE, "value": o.isoformat()}
        if isinstance(o, datetime.time):
            return {"kind": self._KIND_TIME, "value": o.isoformat()}
        if isinstance(o, datetime.timedelta):
            # Kept distinct from pd.Timedelta so the original type comes back.
            return {"kind": self._KIND_TIMEDELTA, "value": o.total_seconds()}
        msg = f"Cannot serialize temporal value {o!r} of type {type(o).__name__}"
        raise ValueError(msg)

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a temporal scalar from its ISO 8601 string."""
        kind = d["kind"]
        raw = d["value"]
        if kind == self._KIND_TIMESTAMP:
            return pd.Timestamp(raw)
        if kind == self._KIND_PD_TIMEDELTA:
            return pd.Timedelta(raw)
        if kind == self._KIND_TIMEDELTA:
            return datetime.timedelta(seconds=raw)
        if kind == self._KIND_DATETIME:
            return datetime.datetime.fromisoformat(raw)
        if kind == self._KIND_DATE:
            return datetime.date.fromisoformat(raw)
        if kind == self._KIND_TIME:
            return datetime.time.fromisoformat(raw)
        msg = f"Unknown temporal kind {kind!r}"
        raise UnrecognizedTypeError(msg)

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match all datetime-module and pandas temporal scalars.

        ``datetime`` is listed alongside ``date`` even though it is a subclass,
        because subtype dispatch is ordered most-derived-first and both need to
        reach this transformer.
        """
        return [datetime.datetime, datetime.date, datetime.time, datetime.timedelta]


class NodeKeyTransformer(CustomTransformer):
    """Transformer for :class:`~loman.nodekey.NodeKey` objects."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "nodekey"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a NodeKey as its path string."""
        return {"path": str(o)}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a NodeKey from its path string."""
        from loman.nodekey import parse_nodekey

        return parse_nodekey(d["path"])

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return supported NodeKey type."""
        from loman.nodekey import NodeKey

        return [NodeKey]
