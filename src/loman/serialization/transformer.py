"""Object serialization and transformation framework."""

import dataclasses
import graphlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np

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
    def supported_subtypes(self) -> Iterable[type]:
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
