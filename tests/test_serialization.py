"""Tests for serialization and transformation functionality in Loman."""

from collections.abc import Iterable
from dataclasses import dataclass

import attrs
import numpy as np
import pytest

from loman.serialization import (
    CustomTransformer,
    MissingObject,
    NdArrayTransformer,
    Transformable,
    Transformer,
    UnrecognizedTypeException,
    UntransformableTypeException,
)

TEST_OBJS: list[object] = [
    1337,
    3.141,
    "Hello",
    ("Hello", "World"),
    ["Hello", "World"],
    {"Hello": "World"},
    {"type": "test", "foo": "bar"},
    [{"a": 1, "b": 2}, (1, 2, 3), "Hello"],
]

TEST_TRANSFORMED_OBJS: list[object] = [
    1337,
    3.141,
    "Hello",
    {"type": "tuple", "values": ["Hello", "World"]},
    ["Hello", "World"],
    {"Hello": "World"},
    {"data": {"foo": "bar", "type": "test"}, "type": "dict"},
    [{"a": 1, "b": 2}, {"type": "tuple", "values": [1, 2, 3]}, "Hello"],
]


@pytest.mark.parametrize("obj,obj_dict", zip(TEST_OBJS, TEST_TRANSFORMED_OBJS))
def test_serialization(obj, obj_dict):
    u = Transformer()
    assert u.to_dict(obj) == obj_dict


@pytest.mark.parametrize("obj,obj_dict", zip(TEST_OBJS, TEST_TRANSFORMED_OBJS))
def test_deserialization(obj, obj_dict):
    u = Transformer()
    assert u.from_dict(obj_dict) == obj


@pytest.mark.parametrize("obj", TEST_OBJS)
def test_serialization_roundtrip(obj):
    u = Transformer()
    d = u.to_dict(obj)
    obj_roundtrip = u.from_dict(d)
    assert obj_roundtrip == obj


TEST_OBJS_COMPLEX: list[object] = [complex(1, 2), {"a": 1, "b": complex(1, 2)}]


class ComplexTransformer(CustomTransformer):
    @property
    def name(self):
        return "complex"

    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        assert isinstance(o, complex)
        return {"real": o.real, "imag": o.imag}

    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        return complex(d["real"], d["imag"])

    @property
    def supported_direct_types(self):
        return [complex]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_COMPLEX)
def test_serialization_roundtrip_complex_transformer(obj):
    u = Transformer()
    u.register(ComplexTransformer())
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


class TestTransformable(Transformable):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def to_dict(self, transformer: "Transformer") -> dict:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, transformer: "Transformer", d: dict) -> object:
        return cls(d["a"], d["b"])


TEST_OBJS_TRANSFORMABLE: list[object] = [TestTransformable("Hello", "world")]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_TRANSFORMABLE)
def test_serialization_roundtrip_transformable(obj):
    u = Transformer()
    u.register(TestTransformable)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class TestAttrs:
    a: int
    b: str


TEST_OBJS_ATTRS: list[object] = [
    TestAttrs(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_ATTRS)
def test_serialization_roundtrip_attrs(obj):
    u = Transformer()
    u.register(TestAttrs)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class TestDataClass:
    a: int
    b: str


TEST_OBJS_DATACLASS: list[object] = [
    TestDataClass(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_DATACLASS)
def test_serialization_roundtrip_dataclass(obj):
    u = Transformer()
    u.register(TestDataClass)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class TestAttrsRecursive:
    a: object


def test_serialization_roundtrip_attrs_recursive():
    u = Transformer()
    u.register(TestAttrsRecursive)
    obj = TestAttrsRecursive(TestAttrsRecursive(TestAttrsRecursive(3)))
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class TestDataClassRecursive:
    a: object


def test_serialization_roundtrip_dataclass_recursive():
    u = Transformer()
    u.register(TestDataClassRecursive)
    obj = TestDataClassRecursive(TestDataClassRecursive(TestDataClassRecursive(3)))
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


def test_serialization_strictness():
    u = Transformer(strict=True)
    with pytest.raises(UntransformableTypeException):
        u.to_dict(TEST_OBJS_COMPLEX[0])
    with pytest.raises(UntransformableTypeException):
        u.to_dict(TEST_OBJS_COMPLEX[1])
    u = Transformer(strict=False)
    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[0])
    assert obj_dict is None
    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[1])
    assert obj_dict == {"a": 1, "b": None}


def test_deserialization_strictness():
    u = Transformer(strict=True)
    u.register(ComplexTransformer())
    v = Transformer(strict=True)

    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[0])
    with pytest.raises(UnrecognizedTypeException):
        obj_roundtrip = v.from_dict(obj_dict)

    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[1])
    with pytest.raises(UnrecognizedTypeException):
        obj_roundtrip = v.from_dict(obj_dict)

    w = Transformer(strict=False)
    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[0])
    obj_roundtrip = w.from_dict(obj_dict)
    assert isinstance(obj_roundtrip, MissingObject)

    obj_dict = u.to_dict(TEST_OBJS_COMPLEX[1])
    obj_roundtrip = w.from_dict(obj_dict)
    assert obj_roundtrip["a"] == 1
    assert isinstance(obj_roundtrip["b"], MissingObject)


def test_nd_array_transformer():
    t = Transformer(strict=False)
    t.register(NdArrayTransformer())
    arr = np.random.randn(4, 3, 2)
    d = t.to_dict(arr)
    arr2 = t.from_dict(d)
    assert np.all(arr == arr2)


class Foo:
    pass


class FooA(Foo):
    pass


class FooB(Foo):
    pass


class FooUnregistered(Foo):
    pass


class FooTransformer(CustomTransformer):
    @property
    def name(self) -> str:
        return "Foo"

    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        if type(o) is Foo:
            return {"v": "Foo"}
        elif type(o) is FooA:
            return {"v": "FooA"}
        elif type(o) is FooB:
            return {"v": "FooB"}
        else:
            raise ValueError(f"Cannot transform {o}")

    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        v = d["v"]
        if v == "Foo":
            return Foo()
        elif v == "FooA":
            return FooA()
        elif v == "FooB":
            return FooB()
        else:
            raise ValueError(f"Cannot transform {v}")

    @property
    def supported_subtypes(self) -> Iterable[type]:
        return [Foo]


def test_serialization_roundtrip_transformer_subtypes():
    u = Transformer()
    u.register(FooTransformer())
    o = Foo()
    d = u.to_dict(o)
    assert d == {"type": "Foo", "v": "Foo"}
    o2 = u.from_dict(d)
    assert type(o2) is Foo

    o = FooA()
    d = u.to_dict(o)
    assert d == {"type": "Foo", "v": "FooA"}
    o2 = u.from_dict(d)
    assert type(o2) is FooA

    o = FooB()
    d = u.to_dict(o)
    assert d == {"type": "Foo", "v": "FooB"}
    o2 = u.from_dict(d)
    assert type(o2) is FooB

    with pytest.raises(ValueError):
        u.to_dict(FooUnregistered())
