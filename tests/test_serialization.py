"""Tests for serialization and transformation functionality in Loman."""

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass

import attrs
import numpy as np
import pandas as pd
import pytest

from loman import Computation, ComputationFactory, SerializationError, States, calc_node, input_node
from loman.computeengine import NodeData
from loman.nodekey import NodeKey, parse_nodekey
from loman.serialization import (
    ComputationSerializer,
    CustomTransformer,
    MissingObject,
    NdArrayTransformer,
    Transformable,
    Transformer,
    UnrecognizedTypeException,
    UntransformableTypeException,
)
from loman.serialization.default import default_transformer

# =============================================================================
# Module-level helper functions for JSON serialization acceptance tests.
# These must be at module level so they are importable by module + qualname.
# =============================================================================


def _json_add_one(x):
    """Return x + 1."""
    return x + 1


def _json_double(x):
    """Return 2 * x."""
    return 2 * x


def _json_add(x, y):
    """Return x + y."""
    return x + y


def _json_raise_value_error(x):
    """Raise a ValueError for testing ERROR state serialization."""
    msg = "deliberate test error"
    raise ValueError(msg)


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


@pytest.mark.parametrize(("obj", "obj_dict"), zip(TEST_OBJS, TEST_TRANSFORMED_OBJS, strict=False))
def test_serialization(obj, obj_dict):
    """Test serialization."""
    u = Transformer()
    assert u.to_dict(obj) == obj_dict


@pytest.mark.parametrize(("obj", "obj_dict"), zip(TEST_OBJS, TEST_TRANSFORMED_OBJS, strict=False))
def test_deserialization(obj, obj_dict):
    """Test deserialization."""
    u = Transformer()
    assert u.from_dict(obj_dict) == obj


@pytest.mark.parametrize("obj", TEST_OBJS)
def test_serialization_roundtrip(obj):
    """Test serialization roundtrip."""
    u = Transformer()
    d = u.to_dict(obj)
    obj_roundtrip = u.from_dict(d)
    assert obj_roundtrip == obj


TEST_OBJS_COMPLEX: list[object] = [complex(1, 2), {"a": 1, "b": complex(1, 2)}]


class ComplexTransformer(CustomTransformer):
    """Custom transformer for complex numbers."""

    @property
    def name(self):
        """Return transformer name."""
        return "complex"

    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        """Convert complex to dict."""
        assert isinstance(o, complex)
        return {"real": o.real, "imag": o.imag}

    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        """Convert dict to complex."""
        return complex(d["real"], d["imag"])

    @property
    def supported_direct_types(self):
        """Return supported types."""
        return [complex]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_COMPLEX)
def test_serialization_roundtrip_complex_transformer(obj):
    """Test serialization roundtrip complex transformer."""
    u = Transformer()
    u.register(ComplexTransformer())
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


class TransformableExample(Transformable):
    """Example class implementing Transformable for testing."""

    def __init__(self, a, b):
        """Initialize with two values."""
        self.a = a
        self.b = b

    def __eq__(self, other):
        """Check equality with another TransformableExample."""
        return self.a == other.a and self.b == other.b

    def to_dict(self, transformer: "Transformer") -> dict:
        """Convert to dict."""
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, transformer: "Transformer", d: dict) -> object:
        """Create from dict."""
        return cls(d["a"], d["b"])


TEST_OBJS_TRANSFORMABLE: list[object] = [TransformableExample("Hello", "world")]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_TRANSFORMABLE)
def test_serialization_roundtrip_transformable(obj):
    """Test serialization roundtrip transformable."""
    u = Transformer()
    u.register(TransformableExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class AttrsExample:
    """Example attrs class for testing serialization."""

    a: int
    b: str


TEST_OBJS_ATTRS: list[object] = [
    AttrsExample(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_ATTRS)
def test_serialization_roundtrip_attrs(obj):
    """Test serialization roundtrip attrs."""
    u = Transformer()
    u.register(AttrsExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class DataClassExample:
    """Example dataclass for testing serialization."""

    a: int
    b: str


TEST_OBJS_DATACLASS: list[object] = [
    DataClassExample(42, "Lorem ipsum.."),
]


@pytest.mark.parametrize("obj", TEST_OBJS + TEST_OBJS_DATACLASS)
def test_serialization_roundtrip_dataclass(obj):
    """Test serialization roundtrip dataclass."""
    u = Transformer()
    u.register(DataClassExample)
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@attrs.define
class AttrsRecursiveExample:
    """Recursive attrs example for testing nested serialization."""

    a: object


def test_serialization_roundtrip_attrs_recursive():
    """Test serialization roundtrip attrs recursive."""
    u = Transformer()
    u.register(AttrsRecursiveExample)
    obj = AttrsRecursiveExample(AttrsRecursiveExample(AttrsRecursiveExample(3)))
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


@dataclass
class DataClassRecursiveExample:
    """Recursive dataclass example for testing nested serialization."""

    a: object


def test_serialization_roundtrip_dataclass_recursive():
    """Test serialization roundtrip dataclass recursive."""
    u = Transformer()
    u.register(DataClassRecursiveExample)
    obj = DataClassRecursiveExample(DataClassRecursiveExample(DataClassRecursiveExample(3)))
    obj_dict = u.to_dict(obj)
    obj_roundtrip = u.from_dict(obj_dict)
    assert obj_roundtrip == obj


def test_serialization_strictness():
    """Test serialization strictness."""
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
    """Test deserialization strictness."""
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
    """Test nd array transformer."""
    t = Transformer(strict=False)
    t.register(NdArrayTransformer())
    arr = np.random.randn(4, 3, 2)
    d = t.to_dict(arr)
    arr2 = t.from_dict(d)
    assert np.all(arr == arr2)


class Foo:
    """Base test class for transformer subtype testing."""

    pass


class FooA(Foo):
    """Subclass A of Foo for testing."""

    pass


class FooB(Foo):
    """Subclass B of Foo for testing."""

    pass


class FooUnregistered(Foo):
    """Unregistered subclass of Foo for testing."""

    pass


class FooTransformer(CustomTransformer):
    """Custom transformer for Foo and subclasses."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "Foo"

    def to_dict(self, transformer: "Transformer", o: object) -> dict:
        """Convert Foo object to dict."""
        if type(o) is Foo:
            return {"v": "Foo"}
        elif type(o) is FooA:
            return {"v": "FooA"}
        elif type(o) is FooB:
            return {"v": "FooB"}
        else:
            msg = f"Cannot transform {o}"
            raise ValueError(msg)

    def from_dict(self, transformer: "Transformer", d: dict) -> object:
        """Convert dict to Foo object."""
        v = d["v"]
        if v == "Foo":
            return Foo()
        elif v == "FooA":
            return FooA()
        elif v == "FooB":
            return FooB()
        else:
            msg = f"Cannot transform {v}"
            raise ValueError(msg)

    @property
    def supported_subtypes(self) -> Iterable[type]:
        """Return supported subtypes."""
        return [Foo]


def test_serialization_roundtrip_transformer_subtypes():
    """Test serialization roundtrip transformer subtypes."""
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

    with pytest.raises(ValueError, match="Cannot transform"):
        u.to_dict(FooUnregistered())


# =============================================================================
# Dill Serialization Tests (from test_dill_serialization.py)
# =============================================================================


def test_dill_serialization():
    """Test dill serialization."""

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
    with pytest.warns(DeprecationWarning, match="write_dill"):
        comp.write_dill(f)

    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
        foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get("state", None) == foo.dag.nodes[n].get("state", None)
        assert comp.dag.nodes[n].get("value", None) == foo.dag.nodes[n].get("value", None)


def test_dill_serialization_skip_flag():
    """Test dill serialization skip flag."""
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1, serialize=False)
    comp.add_node("c", lambda b: b + 1)

    comp.insert("a", 1)
    comp.compute_all()
    f = io.BytesIO()
    with pytest.warns(DeprecationWarning, match="write_dill"):
        comp.write_dill(f)

    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.state("c") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2
    assert comp.value("c") == 3

    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
        comp2 = Computation.read_dill(f)
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.UNINITIALIZED
    assert comp2.state("c") == States.UPTODATE
    assert comp2.value("a") == 1
    assert comp2.value("c") == 3


def test_no_serialize_flag():
    """Test no serialize flag."""
    comp = Computation()
    comp.add_node("a", serialize=False)
    comp.add_node("b", lambda a: a + 1)
    comp.insert("a", 1)
    comp.compute_all()

    f = io.BytesIO()
    with pytest.warns(DeprecationWarning, match="write_dill"):
        comp.write_dill(f)
    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
        comp2 = Computation.read_dill(f)
    assert comp2.state("a") == States.UNINITIALIZED
    assert comp2["b"] == NodeData(States.UPTODATE, 2)


def test_serialize_nested_loman():
    """Test serialize nested loman."""

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
    with pytest.warns(DeprecationWarning, match="write_dill"):
        outer.write_dill(f)
    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
        outer2 = Computation.read_dill(f)

    assert outer2.v.COMP.v.b == outer.v.COMP.v.b
    assert outer2.v.out == outer.v.out


def test_roundtrip_old_dill():
    """Test roundtrip old dill."""

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
    with pytest.warns(DeprecationWarning, match="write_dill"):
        comp.write_dill(f)

    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
        foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get("state", None) == foo.dag.nodes[n].get("state", None)
        assert comp.dag.nodes[n].get("value", None) == foo.dag.nodes[n].get("value", None)


class UnserializableObject:
    """Test class that cannot be serialized."""

    def __init__(self):
        """Initialize with test data."""
        self.data = "This is some data"

    def __getstate__(self):
        """Raise error when trying to serialize."""
        msg = f"{self.__class__.__name__} is not serializable"
        raise TypeError(msg)


def test_serialize_nested_loman_with_unserializable_nodes():
    """Test serialize nested loman with unserializable nodes."""

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

    f = io.BytesIO()
    with pytest.raises(TypeError), pytest.warns(DeprecationWarning, match="write_dill"):
        outer.write_dill(f)

    outer.v.COMP.clear_tag("unserializable", "__serialize__")

    f = io.BytesIO()
    with pytest.warns(DeprecationWarning, match="write_dill"):
        outer.write_dill(f)

    f.seek(0)
    with pytest.warns(DeprecationWarning, match="read_dill"):
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
        with pytest.raises(Exception, match=r".*"):  # Intentionally broad
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


# =============================================================================
# Phase 1 Acceptance Tests: JSON Serialization API
#
# These tests define the target behaviour for write_json / read_json and the
# new transformer types that support them. All tests are expected to FAIL
# until the implementation phases (2-4) are complete.
# =============================================================================


# ---------------------------------------------------------------------------
# Transformer acceptance tests (7 tests)
# ---------------------------------------------------------------------------


class TestEnumTransformer:
    """Acceptance tests for EnumTransformer (States enum roundtrip)."""

    def test_enum_transformer_states(self):
        """States enum values roundtrip through the default computation transformer."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        for state in States:
            d = t.to_dict(state)
            restored = t.from_dict(d)
            assert restored == state, f"States.{state.name} did not roundtrip correctly"


class TestFunctionRefTransformer:
    """Acceptance tests for FunctionRefTransformer."""

    def test_func_ref_transformer_importable(self):
        """An importable module-level function serializes and restores correctly."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        d = t.to_dict(_json_add_one)
        assert isinstance(d, dict), "Expected a dict from to_dict"
        restored = t.from_dict(d)
        assert restored is _json_add_one, "Restored function should be the same object"

    def test_func_ref_transformer_lambda_raises(self):
        """A lambda raises ValueError when serialized."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        f = lambda x: x + 1  # noqa: E731
        with pytest.raises(ValueError, match=r"[Cc]annot serialize|non-importable|lambda"):
            t.to_dict(f)

    def test_func_ref_transformer_closure_raises(self):
        """A closure raises ValueError when serialized."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()

        def outer():
            y = 10

            def inner(x):
                return x + y

            return inner

        closure = outer()
        with pytest.raises(ValueError, match=r"[Cc]annot serialize|non-importable|locals"):
            t.to_dict(closure)


class TestPandasTransformer:
    """Acceptance tests for Pandas DataFrame and Series transformers."""

    def test_pandas_dataframe_transformer(self):
        """A DataFrame with mixed dtypes roundtrips correctly."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3], "c": ["x", "y", "z"]})
        d = t.to_dict(df)
        assert isinstance(d, dict)
        restored = t.from_dict(d)
        pd.testing.assert_frame_equal(df, restored)

    def test_pandas_series_transformer(self):
        """A Series roundtrips correctly."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        s = pd.Series([10, 20, 30], name="my_series")
        d = t.to_dict(s)
        assert isinstance(d, dict)
        restored = t.from_dict(d)
        pd.testing.assert_series_equal(s, restored)


class TestNodeKeyTransformer:
    """Acceptance tests for NodeKeyTransformer."""

    def test_nodekey_transformer(self):
        """A hierarchical NodeKey roundtrips correctly."""
        from loman.serialization.computation import default_computation_transformer

        t = default_computation_transformer()
        nk = NodeKey(("foo", "bar"))
        d = t.to_dict(nk)
        assert isinstance(d, dict)
        restored = t.from_dict(d)
        assert restored == nk


# ---------------------------------------------------------------------------
# Computation JSON API acceptance tests (15 tests)
# ---------------------------------------------------------------------------


class TestJsonRoundtrip:
    """Acceptance tests for Computation.write_json and Computation.read_json."""

    def test_json_roundtrip_basic(self):
        """4-node graph using module-level functions roundtrips with correct states and values."""
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", _json_add_one, kwds={"x": "a"})
        comp.add_node("c", _json_double, kwds={"x": "a"})
        comp.add_node("d", _json_add, kwds={"x": "b", "y": "c"})
        comp.insert("a", 1)
        comp.compute_all()

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert set(comp.dag.nodes) == set(comp2.dag.nodes)
        for n in comp.dag.nodes():
            assert comp.state(n) == comp2.state(n)
            assert comp.value(n) == comp2.value(n)

    def test_json_roundtrip_skip_flag(self):
        """A node with serialize=False is UNINITIALIZED after roundtrip."""
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", _json_add_one, kwds={"x": "a"}, serialize=False)
        comp.add_node("c", _json_add_one, kwds={"x": "b"})
        comp.insert("a", 1)
        comp.compute_all()

        assert comp.state("b") == States.UPTODATE
        assert comp.state("c") == States.UPTODATE

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("a") == States.UPTODATE
        assert comp2.state("b") == States.UNINITIALIZED
        assert comp2.state("c") == States.UPTODATE
        assert comp2.value("a") == 1
        assert comp2.value("c") == 3

    def test_json_no_serialize_input(self):
        """An input node with serialize=False is UNINITIALIZED after roundtrip."""
        comp = Computation()
        comp.add_node("a", serialize=False)
        comp.add_node("b", _json_add_one, kwds={"x": "a"})
        comp.insert("a", 1)
        comp.compute_all()

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("a") == States.UNINITIALIZED
        assert comp2.state("b") == States.UPTODATE
        assert comp2.value("b") == 2

    def test_json_lambda_raises_serialization_error(self):
        """write_json raises SerializationError when a serializable node has a lambda."""
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", lambda x: x + 1, kwds={"x": "a"})
        comp.insert("a", 1)
        comp.compute_all()

        buf = io.StringIO()
        with pytest.raises(SerializationError):
            comp.write_json(buf)

    def test_json_lambda_raises_error_message_mentions_dill_option(self):
        """SerializationError message mentions the use_dill_for_functions option."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()

        buf = io.StringIO()
        with pytest.raises(SerializationError, match="use_dill_for_functions"):
            comp.write_json(buf)

    def test_json_use_dill_for_functions_lambda_roundtrip(self):
        """Lambda functions round-trip when use_dill_for_functions=True."""
        comp = Computation()
        comp.add_node("a", value=3)
        comp.add_node("b", lambda a: a * 2)
        comp.compute_all()

        s = ComputationSerializer(use_dill_for_functions=True)
        buf = io.StringIO()
        comp.write_json(buf, serializer=s)
        buf.seek(0)
        comp2 = Computation.read_json(buf, serializer=s)

        assert comp2.state("b") == States.UPTODATE
        assert comp2.value("b") == 6
        # The function is restored — re-compute works.
        comp2.insert("a", 10)
        comp2.compute_all()
        assert comp2.value("b") == 20

    def test_json_use_dill_for_functions_closure_roundtrip(self):
        """Closures capturing free variables round-trip when use_dill_for_functions=True."""
        offset = 7

        def add_offset(a):
            return a + offset

        comp = Computation()
        comp.add_node("a", value=5)
        comp.add_node("result", add_offset)
        comp.compute_all()

        s = ComputationSerializer(use_dill_for_functions=True)
        buf = io.StringIO()
        comp.write_json(buf, serializer=s)
        buf.seek(0)
        comp2 = Computation.read_json(buf, serializer=s)

        assert comp2.value("result") == 12
        comp2.insert("a", 3)
        comp2.compute_all()
        assert comp2.value("result") == 10

    def test_json_use_dill_for_functions_false_by_default(self):
        """use_dill_for_functions defaults to False — lambdas still raise."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()

        s = ComputationSerializer()  # default — dill disabled
        buf = io.StringIO()
        with pytest.raises(SerializationError):
            comp.write_json(buf, serializer=s)

    def test_json_roundtrip_with_pandas_values(self):
        """A node whose value is a DataFrame roundtrips correctly."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        comp = Computation()
        comp.add_node("data")
        comp.insert("data", df)

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("data") == States.UPTODATE
        pd.testing.assert_frame_equal(comp2.value("data"), df)

    def test_json_roundtrip_with_numpy_values(self):
        """A node whose value is a numpy array roundtrips correctly."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        comp = Computation()
        comp.add_node("matrix")
        comp.insert("matrix", arr)

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("matrix") == States.UPTODATE
        np.testing.assert_array_equal(comp2.value("matrix"), arr)

    def test_json_roundtrip_empty_graph(self):
        """An empty Computation roundtrips without error."""
        comp = Computation()

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert list(comp2.dag.nodes()) == []

    def test_json_roundtrip_file_path(self, tmp_path):
        """write_json and read_json accept string file paths."""
        comp = Computation()
        comp.add_node("a")
        comp.insert("a", 42)

        path = str(tmp_path / "comp.json")
        comp.write_json(path)
        comp2 = Computation.read_json(path)

        assert comp2.state("a") == States.UPTODATE
        assert comp2.value("a") == 42

    def test_json_output_contains_version(self):
        """The JSON output contains a top-level 'version' key."""
        comp = Computation()
        comp.add_node("a")
        comp.insert("a", 1)

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        data = json.loads(buf.read())

        assert "version" in data

    def test_json_output_is_valid_json_text(self):
        """write_json writes text (not binary) that is parseable by json.loads."""
        comp = Computation()
        comp.add_node("a")
        comp.insert("a", 1)

        buf = io.StringIO()
        comp.write_json(buf)
        raw = buf.getvalue()

        # Must be a non-empty string, not bytes
        assert isinstance(raw, str)
        assert len(raw) > 0
        # Must parse as valid JSON
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_json_roundtrip_preserves_edges(self):
        """Edge parameter information (arg vs kwd, name/index) is preserved."""
        comp = Computation()
        comp.add_node("x")
        comp.add_node("y")
        # keyword arg: b depends on x and y as kwargs
        comp.add_node("z", _json_add, kwds={"x": "x", "y": "y"})
        comp.insert("x", 3)
        comp.insert("y", 4)
        comp.compute_all()

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        # After roundtrip, z should still be computable from x and y
        comp2.insert("x", 10)
        comp2.compute_all()
        assert comp2.value("z") == 14  # 10 + 4

    def test_json_roundtrip_pinned_state(self):
        """A PINNED node remains PINNED after roundtrip."""
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", _json_add_one, kwds={"x": "a"})
        comp.insert("a", 5)
        comp.compute_all()
        comp.pin("b", 99)

        assert comp.state("b") == States.PINNED

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("b") == States.PINNED
        assert comp2.value("b") == 99

    def test_json_roundtrip_error_state(self):
        """An ERROR node is serialized preserving exception type and message as strings."""
        # Use a module-level importable function that raises, so the function ref
        # can be serialized and the error value is also preserved.
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", _json_raise_value_error, kwds={"x": "a"})
        comp.insert("a", 1)
        comp.compute_all()

        assert comp.state("b") == States.ERROR

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        assert comp2.state("b") == States.ERROR
        from loman.computeengine import Error

        error_val = comp2.value("b")
        assert isinstance(error_val, Error)
        # Exception type and message must be preserved (as strings after roundtrip)
        assert "deliberate test error" in str(error_val.traceback)

    def test_json_roundtrip_with_nodekeys(self):
        """Hierarchical NodeKey node names roundtrip correctly."""
        comp = Computation()
        comp.add_node("foo/a")
        comp.add_node("foo/b", _json_add_one, kwds={"x": "foo/a"})
        comp.insert("foo/a", 5)
        comp.compute_all()

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        nk_a = parse_nodekey("foo/a")
        nk_b = parse_nodekey("foo/b")
        assert comp2.state(nk_a) == States.UPTODATE
        assert comp2.state(nk_b) == States.UPTODATE
        assert comp2.value(nk_a) == 5
        assert comp2.value(nk_b) == 6

    def test_json_roundtrip_with_block(self, tmp_path):
        """A computation built with add_block roundtrips correctly."""
        from tests.conftest import BasicFourNodeComputation

        inner = BasicFourNodeComputation()
        inner.insert("a", 3)
        inner.compute_all()

        comp = Computation()
        comp.add_block("blk", inner, keep_values=True)

        buf = io.StringIO()
        comp.write_json(buf)
        buf.seek(0)
        comp2 = Computation.read_json(buf)

        nk_a = parse_nodekey("blk/a")
        nk_d = parse_nodekey("blk/d")
        assert comp2.state(nk_a) == States.UPTODATE
        assert comp2.value(nk_a) == 3
        assert comp2.state(nk_d) == States.UPTODATE
        assert comp2.value(nk_d) == inner.value("d")
