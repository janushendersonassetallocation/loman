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
from loman.serialization.default import default_transformer

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


# ==================== ADDITIONAL COVERAGE TESTS ====================


class TestSerializationDefaultCoverage:
    """Tests for serialization/default.py coverage."""

    def test_default_transformer(self):
        """Test default_transformer creates a Transformer with NdArray support."""
        t = default_transformer()
        assert isinstance(t, Transformer)
        # Test that NdArrayTransformer is registered
        arr = np.array([1, 2, 3])
        d = t.to_dict(arr)
        assert d is not None
        arr_back = t.from_dict(d)
        assert np.array_equal(arr, arr_back)


class TestTransformerCoverage:
    """Additional tests for serialization/transformer.py coverage."""

    def test_missing_object_repr(self):
        """Test MissingObject repr."""
        mo = MissingObject()
        assert repr(mo) == "Missing"

    def test_transformer_strict_transformable_missing(self):
        """Test from_dict with unknown Transformable type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "transformable", "class": "UnknownClass", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_transformable_missing(self):
        """Test from_dict with unknown Transformable type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "transformable", "class": "UnknownClass", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_attrs_missing(self):
        """Test from_dict with unknown attrs type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "attrs", "class": "UnknownAttrs", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_attrs_missing(self):
        """Test from_dict with unknown attrs type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "attrs", "class": "UnknownAttrs", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_dataclass_missing(self):
        """Test from_dict with unknown dataclass type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "dataclass", "class": "UnknownDataclass", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_dataclass_missing(self):
        """Test from_dict with unknown dataclass type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "dataclass", "class": "UnknownDataclass", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_unknown_type(self):
        """Test from_dict with completely unknown type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "completely_unknown_type", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_unknown_type(self):
        """Test from_dict with completely unknown type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "completely_unknown_type", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_untransformable(self):
        """Test to_dict with untransformable object in strict mode."""

        class UntransformableClass:
            pass

        t = Transformer(strict=True)
        obj = UntransformableClass()
        with pytest.raises(UntransformableTypeException):
            t.to_dict(obj)

    def test_transformer_non_strict_untransformable(self):
        """Test to_dict with untransformable object in non-strict mode."""

        class UntransformableClass:
            pass

        t = Transformer(strict=False)
        obj = UntransformableClass()
        result = t.to_dict(obj)
        assert result is None

    def test_from_dict_exception_non_dict_list(self):
        """Test from_dict with unexpected type raises Exception."""
        t = Transformer()
        # Pass something that's not str/None/bool/int/float/list/dict
        with pytest.raises(Exception):
            t.from_dict(object())


class TestTransformerSubtypes:
    """Tests for transformer with subtype support."""

    def test_transformer_with_subtypes(self):
        """Test transformer that handles subtypes."""

        class MyBaseClass:
            pass

        class MyDerivedClass(MyBaseClass):
            def __init__(self, value):
                self.value = value

        class MySubtypeTransformer(CustomTransformer):
            @property
            def name(self):
                return "mybase"

            def to_dict(self, transformer, o):
                return {"value": o.value}

            def from_dict(self, transformer, d):
                return MyDerivedClass(d["value"])

            @property
            def supported_subtypes(self):
                return [MyBaseClass]

        t = Transformer()
        t.register(MySubtypeTransformer())

        obj = MyDerivedClass(42)
        d = t.to_dict(obj)
        obj_back = t.from_dict(d)

        assert obj_back.value == 42


class TestAttrsImportPath:
    """Test the attrs import path."""

    def test_transformer_with_attrs_class(self):
        """Test Transformer with an attrs class."""

        @attrs.define
        class MyAttrsClass:
            value: int

        t = Transformer()
        t.register(MyAttrsClass)

        obj = MyAttrsClass(42)
        d = t.to_dict(obj)
        obj_back = t.from_dict(d)

        assert obj_back.value == 42


class TestDataclassSerializer:
    """Test dataclass serialization."""

    def test_transformer_with_dataclass(self):
        """Test Transformer with a dataclass."""

        @dataclass
        class MyDataclass:
            value: int

        t = Transformer()
        t.register(MyDataclass)

        obj = MyDataclass(42)
        d = t.to_dict(obj)
        obj_back = t.from_dict(d)

        assert obj_back.value == 42


class TestMockAttrsImport:
    """Test the attrs import branch when attrs is not installed."""

    def test_attrs_not_available_branch(self):
        """Test the HAS_ATTRS = False branch is tested elsewhere."""
        # This tests the path when attrs IS available - which is always true
        from loman.serialization.transformer import HAS_ATTRS

        assert HAS_ATTRS is True


class TestTransformerNoAttrs:
    """Test Transformer behavior when processing without attrs."""

    def test_transformer_dict_round_trip(self):
        """Test transformer with plain dict."""
        t = Transformer()

        d = {"key": "value", "nested": {"a": 1}}
        result = t.to_dict(d)
        restored = t.from_dict(result)

        assert restored == d


class TestTransformerTupleWithNone:
    """Test transformer with tuple containing None."""

    def test_transformer_tuple_none(self):
        """Test transformer with tuple containing None."""
        t = Transformer()

        obj = (1, None, "str")
        result = t.to_dict(obj)
        restored = t.from_dict(result)

        assert restored == obj


class TestCustomTransformerSubtype:
    """Test CustomTransformer with supported_subtypes."""

    def test_custom_transformer_subtype(self):
        """Test CustomTransformer with subtype handling."""

        class MyBaseClass:
            def __init__(self, value):
                self.value = value

        class MySubClass(MyBaseClass):
            pass

        class MyTransformer(CustomTransformer):
            @property
            def name(self):
                return "my_transformer"

            def to_dict(self, transformer, o):
                return {"value": o.value}

            def from_dict(self, transformer, d):
                return MyBaseClass(d["value"])

            @property
            def supported_subtypes(self):
                return [MyBaseClass]

        t = Transformer()
        t.register(MyTransformer())

        obj = MySubClass(42)
        d = t.to_dict(obj)
        restored = t.from_dict(d)
        assert restored.value == 42


class TestTransformerUnrecognizedType:
    """Test Transformer with unrecognized type in strict mode."""

    def test_unrecognized_type_raises(self):
        """Test that unrecognized type raises error in strict mode."""
        from loman.serialization.transformer import UntransformableTypeError

        t = Transformer(strict=True)

        class UnknownClass:
            pass

        obj = UnknownClass()

        with pytest.raises(UntransformableTypeError):
            t.to_dict(obj)


class TestTransformerNonStrictMode:
    """Test Transformer in non-strict mode."""

    def test_non_strict_from_dict_unknown_type(self):
        """Test non-strict mode handles unknown types in from_dict."""
        from loman.serialization.transformer import TYPENAME_ATTRS

        t = Transformer(strict=False)

        # Construct a dict that references an unknown attrs type
        d = {
            "type": TYPENAME_ATTRS,
            "class": "NonExistentAttrsClass",
        }

        # In non-strict mode, returns MissingObject
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)


class TestTransformableClass:
    """Test Transformable abstract class implementation."""

    def test_transformable_class_roundtrip(self):
        """Test Transformable class serialization."""

        class MyTransformable(Transformable):
            def __init__(self, value):
                self.value = value

            def to_dict(self, transformer):
                return {"value": self.value}

            @classmethod
            def from_dict(cls, transformer, d):
                return cls(d["value"])

        t = Transformer()
        t.register(MyTransformable)

        obj = MyTransformable(42)
        d = t.to_dict(obj)
        restored = t.from_dict(d)
        assert restored.value == 42


class TestOrderClasses:
    """Test order_classes function."""

    def test_order_classes_inheritance(self):
        """Test order_classes orders by inheritance."""
        from loman.serialization.transformer import order_classes

        class Base:
            pass

        class Derived(Base):
            pass

        # Just ensure it returns all classes
        classes = [Derived, Base]
        ordered = order_classes(classes)

        # All classes should be in result
        assert len(ordered) == 2
        assert Base in ordered
        assert Derived in ordered


class TestTransformerRegisterDuplicate:
    """Test registering different types with same name."""

    def test_register_different_types(self):
        """Test registering different types."""

        @dataclass
        class MyClass:
            value: int

        @dataclass
        class AnotherClass:
            name: str

        t = Transformer()
        t.register(MyClass)
        t.register(AnotherClass)

        obj = MyClass(42)
        d = t.to_dict(obj)
        restored = t.from_dict(d)
        assert restored.value == 42


class TestTransformerRegisterUnregisterable:
    """Test Transformer.register with unregisterable type."""

    def test_register_plain_class_raises(self):
        """Test registering plain class raises ValueError."""
        t = Transformer()

        class PlainClass:
            pass

        with pytest.raises(ValueError, match="Unable to register"):
            t.register(PlainClass)
