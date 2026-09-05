"""Error and fallback branches of :mod:`loman.serialization.transformer`.

The happy paths of the transformer live in ``test_serialization.py``, which
round-trips values through a fully populated :class:`Transformer`. This module
covers what that suite never reaches: the registrations that must be *refused*,
the container subclasses that fall past exact-type dispatch, and the messages
:class:`DefinitionMethodTransformer` raises when a bound method cannot be
resolved back to the class it came from.

These are all raise sites with user-facing text, which is the reason they are
worth a test of their own. ``DuplicateRegistrationError`` in particular exists
because the previous ``AssertionError`` vanished under ``python -O``, silently
turning a duplicate registration into an overwrite; nothing asserted it fires.
"""

import re
import types
from collections import OrderedDict, namedtuple
from dataclasses import dataclass

import attrs
import numpy as np
import pytest

from loman.serialization.transformer import (
    KEY_TYPE,
    KEY_VALUES,
    TYPENAME_TUPLE,
    DefinitionMethodTransformer,
    DuplicateRegistrationError,
    NdArrayTransformer,
    SimpleTransformer,
    Transformable,
    Transformer,
)


class Point(Transformable):
    """A minimal :class:`Transformable` used as a registration subject."""

    def __init__(self, x=0):
        """Store the single coordinate."""
        self.x = x

    def to_dict(self, transformer):
        """Encode the coordinate."""
        return {"x": self.x}

    @classmethod
    def from_dict(cls, transformer, d):
        """Rebuild from the encoded coordinate."""
        return cls(d["x"])


@attrs.define
class AttrsPoint:
    """An attrs class used as a registration subject."""

    x: int = 0


@dataclass
class DataclassPoint:
    """A dataclass used as a registration subject."""

    x: int = 0


class Definition:
    """A module-level definition class, resolvable by import path."""

    def method(self):
        """Return a constant, so the method has something to be."""
        return 1

    def other(self):
        """Stand in for a method the transformer should not resolve to."""
        return 2


NOT_A_CLASS = 42


def _simple(name, type_, *, subtypes=False):
    """Build a throwaway :class:`SimpleTransformer` for *type_*."""
    return SimpleTransformer(
        name,
        type_,
        to_dict=lambda o: {"v": o},
        from_dict=lambda d: d["v"],
        subtypes=subtypes,
    )


class TestRegisterTransformerRejectsDuplicates:
    """A second registration must be refused rather than silently winning."""

    def test_duplicate_name_is_refused(self):
        """Two transformers may not share a discriminator name."""
        transformer = Transformer()
        transformer.register_transformer(_simple("point", Point))

        with pytest.raises(DuplicateRegistrationError, match="named 'point' is already registered"):
            transformer.register_transformer(_simple("point", AttrsPoint))

    def test_duplicate_direct_type_is_refused(self):
        """Two transformers may not both claim a type directly."""
        transformer = Transformer()
        transformer.register_transformer(_simple("first", Point))

        with pytest.raises(DuplicateRegistrationError, match="already handled directly by transformer 'first'"):
            transformer.register_transformer(_simple("second", Point))

    def test_duplicate_subtype_is_refused(self):
        """Two transformers may not both claim a base type's subtypes."""
        transformer = Transformer()
        transformer.register_transformer(_simple("first", Point, subtypes=True))

        with pytest.raises(DuplicateRegistrationError, match="already handled by transformer 'first'"):
            transformer.register_transformer(_simple("second", Point, subtypes=True))


class TestRegisterTypeRejectsDuplicates:
    """The by-class-name registries refuse a second registration too."""

    def test_transformable_registered_twice_is_refused(self):
        """A ``Transformable`` class may only be registered once."""
        transformer = Transformer()
        transformer.register_transformable(Point)

        with pytest.raises(DuplicateRegistrationError, match="Transformable class named 'Point'"):
            transformer.register_transformable(Point)

    def test_attrs_class_registered_twice_is_refused(self):
        """An attrs class may only be registered once."""
        transformer = Transformer()
        transformer.register_attrs(AttrsPoint)

        with pytest.raises(DuplicateRegistrationError, match="attrs class named 'AttrsPoint'"):
            transformer.register_attrs(AttrsPoint)

    def test_dataclass_registered_twice_is_refused(self):
        """A dataclass may only be registered once."""
        transformer = Transformer()
        transformer.register_dataclass(DataclassPoint)

        with pytest.raises(DuplicateRegistrationError, match="dataclass named 'DataclassPoint'"):
            transformer.register_dataclass(DataclassPoint)


class TestBlobStoreNameOutsideAWrite:
    """Blob routing is a property of a save, not of the transformer."""

    def test_no_store_without_a_write_scope(self):
        """Asked outside a save, the transformer names no store."""
        assert Transformer().blob_store_name() is None


class TestContainerSubclassesEncodeAsBuiltins:
    """A subclass that nothing else claims is encoded as the plain builtin.

    Dispatch is on the *exact* type, so a namedtuple, a list subclass and an
    ``OrderedDict`` all fall past the fast path and land in the fallback, where
    they take the encoding they had before exact-type dispatch existed.
    """

    def test_namedtuple_encodes_as_a_tuple(self):
        """A tuple subclass takes the plain tuple encoding."""
        pair = namedtuple("pair", ["a", "b"])
        encoded = Transformer().to_dict(pair(1, 2))

        assert encoded == {KEY_TYPE: TYPENAME_TUPLE, KEY_VALUES: [1, 2]}

    def test_list_subclass_encodes_as_a_list(self):
        """A list subclass takes the plain list encoding."""

        class MyList(list):
            """A list subclass with no encoding of its own."""

        assert Transformer().to_dict(MyList([1, 2])) == [1, 2]

    def test_dict_subclass_encodes_as_a_dict(self):
        """A dict subclass takes the plain dict encoding."""
        assert Transformer().to_dict(OrderedDict(a=1)) == {"a": 1}


class TestObjectArraysStayInline:
    """An object array cannot go out of line without pickling its elements."""

    def test_object_dtype_keeps_its_data_in_the_head(self):
        """The elements are encoded inline, with no blob offered."""
        array = np.array([1, 2], dtype=object)
        encoded = NdArrayTransformer().to_dict(Transformer(), array)

        assert encoded["data"] == [1, 2]
        assert "encoding" not in encoded


class TestDefinitionMethodRefusesWhatItCannotResolve:
    """Saving reports an unresolvable method, rather than the load doing so."""

    def test_a_plain_object_is_not_a_bound_method(self):
        """Something with no ``__func__`` is refused by name."""
        with pytest.raises(ValueError, match="not a bound method"):
            DefinitionMethodTransformer().to_dict(Transformer(), object())

    def test_a_method_without_a_class_qualified_name_is_refused(self):
        """A function bound by hand has no class to resolve it by."""

        def loose(self):
            """Stand in for a function never defined on a class."""
            return 1

        loose.__qualname__ = "loose"
        bound = types.MethodType(loose, Definition())

        with pytest.raises(ValueError, match="no class-qualified name"):
            DefinitionMethodTransformer().to_dict(Transformer(), bound)

    def test_a_method_defined_inside_a_function_is_refused(self):
        """A class defined in a function body is not importable."""

        class Local:
            """A definition class that exists only in this frame."""

            def method(self):
                """Return a constant."""
                return 1

        with pytest.raises(ValueError, match="defined inside a function"):
            DefinitionMethodTransformer().to_dict(Transformer(), Local().method)

    def test_a_method_that_no_longer_resolves_back_is_refused(self, monkeypatch):
        """A name that now resolves elsewhere is caught while saving."""
        bound = Definition().method
        monkeypatch.setattr(Definition, "method", Definition.other)

        with pytest.raises(ValueError, match="does not resolve back to this method"):
            DefinitionMethodTransformer().to_dict(Transformer(), bound)


class TestDefinitionMethodResolutionFailures:
    """Rebuilding names what could not be resolved, and where it stopped."""

    def test_unknown_method_on_a_known_class(self):
        """The class resolves, the method on it does not."""
        with pytest.raises(ValueError, match="no such method on the definition class"):
            DefinitionMethodTransformer()._resolve(__name__, "Definition", "missing")

    def test_unimportable_module(self):
        """A module that cannot be imported is reported with its name."""
        with pytest.raises(ValueError, match=re.escape("Cannot resolve loman._no_such_module.Definition")):
            DefinitionMethodTransformer()._definition_class("loman._no_such_module", "Definition")

    def test_unknown_attribute_on_an_importable_module(self):
        """A name absent from a real module is reported the same way."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            DefinitionMethodTransformer()._definition_class(__name__, "NoSuchName")

    def test_name_that_resolves_to_something_other_than_a_class(self):
        """A name resolving to a non-class is not a definition class."""
        with pytest.raises(ValueError, match="not a class or a computation factory"):
            DefinitionMethodTransformer()._definition_class(__name__, "NOT_A_CLASS")
