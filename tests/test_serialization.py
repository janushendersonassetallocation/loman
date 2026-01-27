"""Tests for serialization and transformation functionality in Loman."""

import io
from collections.abc import Iterable
from dataclasses import dataclass

import attrs
import numpy as np
import pytest

from loman import Computation, ComputationFactory, States, calc_node, input_node
from loman.computeengine import NodeData
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


class TransformableExample(Transformable):
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


TEST_OBJS_TRANSFORMABLE: list[object] = [TransformableExample("Hello", "world")]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_TRANSFORMABLE)
def test_serialization_roundtrip_transformable(obj):
    u = Transformer()
    u.register(TransformableExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class AttrsExample:
    a: int
    b: str


TEST_OBJS_ATTRS: list[object] = [
    AttrsExample(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_ATTRS)
def test_serialization_roundtrip_attrs(obj):
    u = Transformer()
    u.register(AttrsExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class DataClassExample:
    a: int
    b: str


TEST_OBJS_DATACLASS: list[object] = [
    DataClassExample(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_DATACLASS)
def test_serialization_roundtrip_dataclass(obj):
    u = Transformer()
    u.register(DataClassExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class AttrsRecursiveExample:
    a: object


def test_serialization_roundtrip_attrs_recursive():
    u = Transformer()
    u.register(AttrsRecursiveExample)
    obj = AttrsRecursiveExample(AttrsRecursiveExample(AttrsRecursiveExample(3)))
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class DataClassRecursiveExample:
    a: object


def test_serialization_roundtrip_dataclass_recursive():
    u = Transformer()
    u.register(DataClassRecursiveExample)
    obj = DataClassRecursiveExample(DataClassRecursiveExample(DataClassRecursiveExample(3)))
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


# =============================================================================
# Dill Serialization Tests (from test_dill_serialization.py)
# =============================================================================


def test_dill_serialization():
    def b(x):
        return x + 1

    def c(x):
        return 2 * x

    def d(x, y):
        return x + y

    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", b, kwds={"x": "a"})
    comp.add_node("c", c, kwds={"x": "a"})
    comp.add_node("d", d, kwds={"x": "b", "y": "c"})

    comp.insert("a", 1)
    comp.compute_all()
    f = io.BytesIO()
    comp.write_dill(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get("state", None) == foo.dag.nodes[n].get("state", None)
        assert comp.dag.nodes[n].get("value", None) == foo.dag.nodes[n].get("value", None)


def test_dill_serialization_skip_flag():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1, serialize=False)
    comp.add_node("c", lambda b: b + 1)

    comp.insert("a", 1)
    comp.compute_all()
    f = io.BytesIO()
    comp.write_dill(f)

    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.state("c") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2
    assert comp.value("c") == 3

    f.seek(0)
    comp2 = Computation.read_dill(f)
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.UNINITIALIZED
    assert comp2.state("c") == States.UPTODATE
    assert comp2.value("a") == 1
    assert comp2.value("c") == 3


def test_no_serialize_flag():
    comp = Computation()
    comp.add_node("a", serialize=False)
    comp.add_node("b", lambda a: a + 1)
    comp.insert("a", 1)
    comp.compute_all()

    f = io.BytesIO()
    comp.write_dill(f)
    f.seek(0)
    comp2 = Computation.read_dill(f)
    assert comp2.state("a") == States.UNINITIALIZED
    assert comp2["b"] == NodeData(States.UPTODATE, 2)


def test_serialize_nested_loman():
    @ComputationFactory
    class CompInner:
        a = input_node(value=3)

        @calc_node
        def b(self, a):
            return a + 1

    @ComputationFactory
    class CompOuter:
        COMP = input_node()

        @calc_node
        def out(self, comp):
            return comp.x.b + 10

    inner = CompInner()
    inner.compute_all()

    outer = CompOuter()
    outer.insert("COMP", inner)
    outer.compute_all()

    f = io.BytesIO()
    outer.write_dill(f)
    f.seek(0)
    outer2 = Computation.read_dill(f)

    assert outer2.v.COMP.v.b == outer.v.COMP.v.b
    assert outer2.v.out == outer.v.out


def test_roundtrip_old_dill():
    def b(x):
        return x + 1

    def c(x):
        return 2 * x

    def d(x, y):
        return x + y

    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", b, kwds={"x": "a"})
    comp.add_node("c", c, kwds={"x": "a"})
    comp.add_node("d", d, kwds={"x": "b", "y": "c"})

    comp.insert("a", 1)
    comp.compute_all()
    f = io.BytesIO()
    comp.write_dill(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get("state", None) == foo.dag.nodes[n].get("state", None)
        assert comp.dag.nodes[n].get("value", None) == foo.dag.nodes[n].get("value", None)


class UnserializableObject:
    def __init__(self):
        self.data = "This is some data"

    def __getstate__(self):
        raise TypeError(f"{self.__class__.__name__} is not serializable")


def test_serialize_nested_loman_with_unserializable_nodes():
    @ComputationFactory
    class CompInner:
        a = input_node(value=3)

        @calc_node
        def unserializable(self, a):
            return UnserializableObject()

    @ComputationFactory
    class CompOuter:
        COMP = input_node()

        @calc_node
        def out(self, comp):
            return comp.x.a + 10

    inner = CompInner()
    inner.compute_all()
    outer = CompOuter()
    outer.insert("COMP", inner)
    outer.compute_all()

    with pytest.raises(TypeError):
        f = io.BytesIO()
        outer.write_dill(f)

    outer.v.COMP.clear_tag("unserializable", "__serialize__")

    f = io.BytesIO()
    outer.write_dill(f)

    f.seek(0)
    outer2 = Computation.read_dill(f)

    assert outer2.v.COMP.v.a == outer.v.COMP.v.a
    assert outer2.v.out == outer.v.out
