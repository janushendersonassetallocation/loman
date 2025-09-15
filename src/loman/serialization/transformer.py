import graphlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

try:
    import attrs

    HAS_ATTRS = True
except ImportError:
    HAS_ATTRS = False

import dataclasses

KEY_TYPE = "type"
KEY_CLASS = "class"
KEY_VALUES = "values"
KEY_DATA = "data"

TYPENAME_DICT = "dict"
TYPENAME_TUPLE = "tuple"
TYPENAME_TRANSFORMABLE = "transformable"
TYPENAME_ATTRS = "attrs"
TYPENAME_DATACLASS = "dataclass"


class UntransformableTypeException(Exception):
    pass


class UnrecognizedTypeException(Exception):
    pass


class MissingObject:
    def __repr__(self):
        return "Missing"


def order_classes(classes):
    graph = {x: set() for x in classes}
    for x in classes:
        for y in classes:
            if issubclass(x, y) and x != y:
                graph[y].add(x)
    ts = graphlib.TopologicalSorter(graph)
    return list(ts.static_order())


class CustomTransformer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        pass

    @abstractmethod
    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        pass

    @property
    def supported_direct_types(self) -> Iterable[type]:
        return []

    @property
    def supported_subtypes(self) -> Iterable[type]:
        return []


class Transformable(ABC):
    @abstractmethod
    def to_dict(self, transformer: "Transformer") -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, transformer: "Transformer", d: dict) -> object:
        pass


class Transformer:
    def __init__(self, *, strict: bool = True):
        self.strict = strict

        self._direct_type_map = {}
        self._subtype_order = []
        self._subtype_map = {}
        self._transformers = {}
        self._transformable_types = {}
        self._attrs_types = {}
        self._dataclass_types = {}

    def register(self, t: CustomTransformer | type[Transformable] | type):
        if isinstance(t, CustomTransformer):
            self.register_transformer(t)
        elif issubclass(t, Transformable):
            self.register_transformable(t)
        elif HAS_ATTRS and attrs.has(t):
            self.register_attrs(t)
        elif dataclasses.is_dataclass(t):
            self.register_dataclass(t)
        else:
            raise ValueError(f"Unable to register {t}")

    def register_transformer(self, transformer: CustomTransformer):
        assert transformer.name not in self._transformers
        for type_ in transformer.supported_direct_types:
            assert type_ not in self._direct_type_map
        for type_ in transformer.supported_subtypes:
            assert type_ not in self._subtype_map

        self._transformers[transformer.name] = transformer

        for type_ in transformer.supported_direct_types:
            self._direct_type_map[type_] = transformer

        contains_supported_subtypes = False
        for type_ in transformer.supported_subtypes:
            contains_supported_subtypes = True
            self._subtype_map[type_] = transformer
        if contains_supported_subtypes:
            self._subtype_order = order_classes(self._subtype_map.keys())

    def register_transformable(self, transformable_type: type[Transformable]):
        name = transformable_type.__name__
        assert name not in self._transformable_types
        self._transformable_types[name] = transformable_type

    def register_attrs(self, attrs_type: type):
        name = attrs_type.__name__
        assert name not in self._attrs_types
        self._attrs_types[name] = attrs_type

    def register_dataclass(self, dataclass_type: type):
        name = dataclass_type.__name__
        assert name not in self._dataclass_types
        self._dataclass_types[name] = dataclass_type

    def get_transformer_for_obj(self, obj) -> CustomTransformer | None:
        transformer = self._direct_type_map.get(type(obj))
        if transformer is not None:
            return transformer
        for tp in self._subtype_order:
            if isinstance(obj, tp):
                return self._subtype_map[tp]

    def get_transformer_for_name(self, name) -> CustomTransformer | None:
        transformer = self._transformers.get(name)
        return transformer

    def to_dict(self, o):
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
        elif HAS_ATTRS and attrs.has(o):
            return self._attrs_to_dict(o)
        elif dataclasses.is_dataclass(o):
            return self._dataclass_to_dict(o)
        else:
            return self._to_dict_transformer(o)

    def _dict_to_dict(self, o):
        d = {k: self.to_dict(v) for k, v in o.items()}
        if KEY_TYPE in o:
            return {KEY_TYPE: TYPENAME_DICT, KEY_DATA: d}
        else:
            return d

    def _attrs_to_dict(self, o):
        data = {}
        for a in o.__attrs_attrs__:
            data[a.name] = self.to_dict(o.__getattribute__(a.name))
        res = {KEY_TYPE: TYPENAME_ATTRS, KEY_CLASS: type(o).__name__}
        if len(data) > 0:
            res[KEY_DATA] = data
        return res

    def _dataclass_to_dict(self, o):
        data = {}
        for f in dataclasses.fields(o):
            data[f.name] = self.to_dict(getattr(o, f.name))
        res = {KEY_TYPE: TYPENAME_DATACLASS, KEY_CLASS: type(o).__name__}
        if len(data) > 0:
            res[KEY_DATA] = data
        return res

    def _to_dict_transformer(self, o):
        transformer = self.get_transformer_for_obj(o)
        if transformer is None:
            if self.strict:
                raise UntransformableTypeException(f"Could not transform object of type {type(o).__name__}")
            else:
                return None
        d = transformer.to_dict(self, o)
        d[KEY_TYPE] = transformer.name
        return d

    def from_dict(self, d):
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
            raise Exception()

    def _from_dict_transformable(self, d):
        classname = d[KEY_CLASS]
        cls = self._transformable_types.get(classname)
        if cls is None:
            if self.strict:
                raise UnrecognizedTypeException(f"Unable to transform Transformable object of class {classname}")
            else:
                return MissingObject()
        else:
            return cls.from_dict(self, d[KEY_DATA])

    def _from_attrs(self, d):
        if not HAS_ATTRS:
            if self.strict:
                raise UnrecognizedTypeException("attrs package not installed")
            return MissingObject()
        cls = self._attrs_types.get(d[KEY_CLASS])
        if cls is None:
            if self.strict:
                raise UnrecognizedTypeException(f"Unable to create attrs object of type {cls}")
            else:
                return MissingObject()
        else:
            kwargs = {}
            if KEY_DATA in d:
                for key, value in d[KEY_DATA].items():
                    kwargs[key] = self.from_dict(value)
            return cls(**kwargs)

    def _from_dataclass(self, d):
        cls = self._dataclass_types.get(d[KEY_CLASS])
        if cls is None:
            if self.strict:
                raise UnrecognizedTypeException(f"Unable to create dataclass object of type {cls}")
            else:
                return MissingObject()
        else:
            kwargs = {}
            if KEY_DATA in d:
                for key, value in d[KEY_DATA].items():
                    kwargs[key] = self.from_dict(value)
            return cls(**kwargs)

    def _from_dict_transformer(self, type_, d):
        transformer = self.get_transformer_for_name(type_)
        if transformer is None:
            if self.strict:
                raise UnrecognizedTypeException(f"Unable to transform object of type {type_}")
            else:
                return MissingObject()
        return transformer.from_dict(self, d)


class NdArrayTransformer(CustomTransformer):
    @property
    def name(self):
        return "ndarray"

    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        assert isinstance(o, np.ndarray)
        return {"shape": list(o.shape), "dtype": o.dtype.str, "data": transformer.to_dict(o.ravel().tolist())}

    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        return np.array(transformer.from_dict(d["data"]), d["dtype"]).reshape(d["shape"])

    @property
    def supported_direct_types(self):
        return [np.ndarray]
