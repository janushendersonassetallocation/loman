"""Object serialization and transformation framework."""

import contextlib
import dataclasses
import graphlib
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

try:
    import attrs

    HAS_ATTRS = True
except ImportError:  # pragma: no cover
    HAS_ATTRS = False

KEY_TYPE = "type"
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
        """Convert numpy array to dictionary with shape, dtype, and data."""
        assert isinstance(o, np.ndarray)  # noqa: S101
        return {"shape": list(o.shape), "dtype": o.dtype.str, "data": transformer.to_dict(o.ravel().tolist())}  # type: ignore[arg-type]

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct numpy array from dictionary."""
        return np.array(transformer.from_dict(d["data"]), d["dtype"]).reshape(d["shape"])

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

        import dill

        if not callable(o):
            msg = f"Object {o!r} is not callable"
            raise TypeError(msg)
        blob = dill.dumps(o)
        return {"blob": base64.b64encode(blob).decode("ascii")}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a callable from a base64-encoded dill blob."""
        import base64

        import dill

        blob = base64.b64decode(d["blob"].encode("ascii"))
        return dill.loads(blob)  # noqa: S301 — intentional: user-controlled data from their own write_json

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
        """Serialize a DataFrame using split orientation."""
        assert isinstance(o, pd.DataFrame)  # noqa: S101
        return {
            "columns": list(o.columns),
            "index": transformer.to_dict(list(o.index)),
            "data": transformer.to_dict(o.values.tolist()),
            "dtypes": {col: str(dtype) for col, dtype in o.dtypes.items()},
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a DataFrame from its serialized form."""
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
        """Serialize a Series with its name, dtype, index, and data."""
        assert isinstance(o, pd.Series)  # noqa: S101
        return {
            "name": o.name,
            "dtype": str(o.dtype),
            "index": transformer.to_dict(list(o.index)),
            "data": transformer.to_dict(o.tolist()),
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a Series from its serialized form."""
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
