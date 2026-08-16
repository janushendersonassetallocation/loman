"""Object serialization and transformation framework."""

import contextlib
import contextvars
import dataclasses
import graphlib
import importlib
import math
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from enum import Enum
from typing import Any, NamedTuple, cast

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
KEY_ITEMS = "items"

TYPENAME_DICT = "dict"
TYPENAME_TUPLE = "tuple"
TYPENAME_TRANSFORMABLE = "transformable"
TYPENAME_ATTRS = "attrs"
TYPENAME_DATACLASS = "dataclass"
TYPENAME_FLOAT = "float"

# Keys that carry meaning in an encoded value. A user dict containing any of
# them is written in the escaped ``{"type": "dict", ...}`` form so that its
# contents can never be mistaken for encoding metadata on the way back in.
# "$blob" is here for the same reason "type" is: a dict that happens to contain
# it would otherwise be read back as a reference to someone else's bytes.
RESERVED_KEYS = frozenset({KEY_TYPE, "$blob"})

# Non-finite floats have no JSON literal. Writing them as bare NaN / Infinity
# tokens --- json.dump's default --- produces a file Python can read back and
# nothing else can, so they are tagged and ``allow_nan=False`` is set, which
# makes an invalid document structurally impossible rather than merely unlikely.
_NONFINITE_TO_NAME = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
_NAME_TO_FLOAT = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


class _WriteScope(NamedTuple):
    """Ambient state for one encoding pass: where blobs go, and for which node."""

    sink: Any
    profile: Any
    node: str | None
    tags: frozenset[str]
    store: str | None


# Set by Transformer.writing() / .reading(). ContextVars rather than instance
# attributes so that two threads encoding through the same Transformer cannot
# see each other's sink.
_WRITE_SCOPE: contextvars.ContextVar[_WriteScope | None] = contextvars.ContextVar("loman_blob_write", default=None)
_READ_SCOPE: contextvars.ContextVar[Any] = contextvars.ContextVar("loman_blob_read", default=None)


class UntransformableTypeError(Exception):
    """Exception raised when a type cannot be transformed for serialization."""

    pass


class UnrecognizedTypeError(Exception):
    """Exception raised when a type is not recognized during transformation."""

    pass


class DuplicateRegistrationError(ValueError):
    """Exception raised when a name or type is registered on a Transformer twice.

    Subclasses :class:`ValueError` because that is what :meth:`Transformer.register`
    already raises for a type it cannot register at all.
    """

    pass


class MissingObject:
    """Sentinel object representing missing or unset values."""

    def __repr__(self) -> str:
        """Return string representation of missing object."""
        return "Missing"


def _nonfinite_name(value: float) -> str:
    """Return the JSON-safe name for a non-finite float."""
    return _NONFINITE_TO_NAME[repr(value)]


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


class SimpleTransformer(CustomTransformer):
    """A :class:`CustomTransformer` built from two plain functions.

    Implementing :class:`CustomTransformer` directly means a subclass with three
    members. When the encoding is just "turn it into a dict and back", this saves
    the ceremony::

        point_transformer = SimpleTransformer(
            "point", Point,
            to_dict=lambda p: {"x": p.x, "y": p.y},
            from_dict=lambda d: Point(d["x"], d["y"]),
        )

    The callables receive and return plain values; nested values are *not*
    transformed automatically. Subclass :class:`CustomTransformer` directly when
    the encoding needs to recurse through the transformer, or to write bytes
    out-of-line.
    """

    def __init__(
        self,
        name: str,
        type_: type,
        *,
        to_dict: Callable[[Any], dict[str, Any]],
        from_dict: Callable[[dict[str, Any]], Any],
        subtypes: bool = False,
    ) -> None:
        """Build a transformer for *type_* from *to_dict* and *from_dict*.

        :param name: Unique discriminator written as the ``"type"`` field.
        :param type_: The type this transformer handles.
        :param to_dict: Callable taking an instance and returning a plain dict.
        :param from_dict: Callable taking that dict and returning an instance.
        :param subtypes: When true, also handle subclasses of *type_*.
        """
        self._name = name
        self._type = type_
        self._to_dict = to_dict
        self._from_dict = from_dict
        self._subtypes = subtypes

    @property
    def name(self) -> str:
        """Return the transformer's discriminator name."""
        return self._name

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Encode *o* by delegating to the supplied callable."""
        return self._to_dict(o)

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Decode *d* by delegating to the supplied callable."""
        return self._from_dict(d)

    @property
    def supported_direct_types(self) -> Iterable[type]:
        """Return the handled type, unless subtype matching was requested."""
        return [] if self._subtypes else [self._type]

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Return the handled base type when subtype matching was requested."""
        return [self._type] if self._subtypes else []


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
        if transformer.name in self._transformers:
            msg = f"A transformer named {transformer.name!r} is already registered"
            raise DuplicateRegistrationError(msg)
        for type_ in transformer.supported_direct_types:
            if type_ in self._direct_type_map:
                msg = f"Type {type_!r} is already handled directly by transformer {self._direct_type_map[type_].name!r}"
                raise DuplicateRegistrationError(msg)
        for type_ in transformer.supported_subtypes:
            if type_ in self._subtype_map:
                msg = f"Subtype {type_!r} is already handled by transformer {self._subtype_map[type_].name!r}"
                raise DuplicateRegistrationError(msg)

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
        if name in self._transformable_types:
            msg = f"A Transformable class named {name!r} is already registered"
            raise DuplicateRegistrationError(msg)
        self._transformable_types[name] = transformable_type

    def register_attrs(self, attrs_type: type) -> None:
        """Register an attrs-decorated class for serialization."""
        name = attrs_type.__name__
        if name in self._attrs_types:
            msg = f"An attrs class named {name!r} is already registered"
            raise DuplicateRegistrationError(msg)
        self._attrs_types[name] = attrs_type

    def register_dataclass(self, dataclass_type: type) -> None:
        """Register a dataclass for serialization."""
        name = dataclass_type.__name__
        if name in self._dataclass_types:
            msg = f"A dataclass named {name!r} is already registered"
            raise DuplicateRegistrationError(msg)
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

    # ------------------------------------------------------------------
    # Out-of-line bytes.
    #
    # A transformer is *offered* a blob sink rather than being asked to
    # implement a second encoding: it calls offer_blob() to ask whether this
    # save wants bytes out of line, and put_blob() to hand them over. Nothing
    # about to_dict's signature changes, so a transformer written before any of
    # this existed keeps working --- it never calls offer_blob, so it always
    # inlines, which is exactly what it did before.
    #
    # The sink lives in a ContextVar rather than being threaded through every
    # call. That keeps user code free of plumbing, and makes concurrent use safe
    # for free, which matters because loman runs nodes on executors.
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def writing(
        self,
        sink: Any,
        profile: Any = None,
        node: str | None = None,
        tags: frozenset[str] = frozenset(),
        store: str | None = None,
    ) -> Iterator[None]:
        """Scope in which ``offer_blob`` and ``put_blob`` write to *sink*.

        :param node: Node key, for the blob table and for glob selectors.
        :param tags: The node's tags, for ``tag:`` selectors.
        :param store: The store named on the node itself, which a profile
            override for this node can replace.
        """
        token = _WRITE_SCOPE.set(_WriteScope(sink=sink, profile=profile, node=node, tags=tags, store=store))
        try:
            yield
        finally:
            _WRITE_SCOPE.reset(token)

    @contextlib.contextmanager
    def reading(self, source: Any) -> Iterator[None]:
        """Scope in which ``get_blob`` resolves references against *source*."""
        token = _READ_SCOPE.set(source)
        try:
            yield
        finally:
            _READ_SCOPE.reset(token)

    def offer_blob(self, nbytes: int | None = None) -> bool:
        """Return whether a value of *nbytes* should be written out of line.

        Outside a write scope, or in a container with no blob storage, this is
        always false --- so a transformer never has to know which container it
        is writing into.

        :param nbytes: Estimated encoded size. ``None`` means "large, but I
            cannot say how large", and is taken at its word.
        """
        scope = _WRITE_SCOPE.get()
        if scope is None or scope.profile is None:
            return False
        # Asks the writer about *this node's* store, not just whether any store
        # exists. A node routed to an external store can go out of line even in
        # the single-document container, which is how a readable manifest can
        # sit alongside data held in S3.
        store = self.blob_store_name()
        if not scope.sink.can_store(store):
            if store is not None:
                # The node asked for a specific store and it was not supplied.
                # Falling back to inline would put the data in the file while
                # the caller believed it had gone to their bucket, so this is an
                # error rather than a quiet substitution.
                from loman.exception import SerializationError

                msg = (
                    f"Node {scope.node!r} is routed to blob store {store!r}, which was not supplied. "
                    f"Pass it as save(..., stores={{{store!r}: ...}}), or override the routing for "
                    "this save with a profile."
                )
                raise SerializationError(msg)
            return False
        return bool(scope.profile.wants_blob(nbytes))

    def blob_store_name(self) -> str | None:
        """Return the store the value being encoded should be written to.

        The node's own declaration is the default; a profile override matching
        this node replaces it, so the same computation can be saved to S3 in
        production and to a plain container in a test.
        """
        scope = _WRITE_SCOPE.get()
        if scope is None:
            return None
        return self.blob_setting("store", scope.store)

    def put_blob(
        self,
        payload: Any,
        *,
        codec: str,
        compressible: bool = True,
        dedupe_on: Any = None,
    ) -> dict[str, int]:
        """Store *payload* out of line and return the reference to embed.

        :param payload: ``bytes``, a buffer, or a callable taking a binary file
            object. The callable form lets an encoder stream straight into the
            container instead of building the whole payload in memory first.
        :param codec: How the bytes are encoded, e.g. ``"npy"``. Recorded in the
            blob table as the file extension and as metadata for other tools; it
            is not what dispatch keys off on the way back in.
        :param compressible: Pass false for a payload that already compresses
            itself, such as parquet. Skips the compression probe entirely, which
            is how double-compression is prevented.
        :param dedupe_on: The object being stored, when two nodes holding the
            same object should share one blob. Pass the object itself, not its
            ``id()``: the store keeps a reference to it, which is what stops a
            short-lived temporary's id from being reused by the next one and
            silently deduplicating two unrelated values onto one blob.
        """
        scope = _WRITE_SCOPE.get()
        if scope is None:
            msg = "put_blob() called outside a write scope; use Transformer.writing()"
            raise RuntimeError(msg)
        from .blobs import blob_ref

        blob_id = scope.sink.put(
            payload,
            codec=codec,
            node=scope.node,
            store=self.blob_store_name(),
            compressible=compressible,
            dedupe_on=dedupe_on,
        )
        return blob_ref(blob_id)

    def blob_setting(self, name: str, default: Any = None) -> Any:
        """Return a profile setting for the value currently being encoded.

        Per-node overrides win over the profile's own value, so one save can
        treat some nodes differently from the rest. Outside a write scope the
        default is returned, which is what keeps transformers working when they
        are driven directly.
        """
        scope = _WRITE_SCOPE.get()
        if scope is None or scope.profile is None:
            return default
        overrides = scope.profile.settings_for(scope.node, scope.tags)
        if name in overrides:
            return overrides[name]
        return getattr(scope.profile, name, default)

    def get_blob(self, ref: dict[str, int]) -> bytes:
        """Return the bytes for the blob reference *ref*."""
        source = _READ_SCOPE.get()
        if source is None:
            msg = "This value's data is stored out of line, but no blob store is open for reading"
            raise RuntimeError(msg)
        from .blobs import BLOB_REF_KEY

        return source.get(ref[BLOB_REF_KEY])

    def to_dict(self, o: object) -> Any:
        """Convert an object to a serializable dictionary representation.

        Split in two along the line that matters. This half handles values whose
        type is *exactly* a builtin, which is the overwhelming majority of what a
        graph holds and so the path worth keeping short; everything else goes to
        :meth:`_to_dict_object`.

        The checks are exact-type, not ``isinstance``, deliberately: numpy
        scalars and ``IntEnum`` members are instances of ``int`` or ``float``, so
        an ``isinstance`` fast path would return them as bare numbers and
        silently discard the type. Falling through is what lets the registered
        transformers see them.
        """
        type_ = type(o)
        if type_ is str or o is None or o is True or o is False or type_ is int:
            return o
        if type_ is float:
            # cast: an exact-type check carries no narrowing for a type checker.
            number = cast("float", o)
            return number if math.isfinite(number) else {KEY_TYPE: TYPENAME_FLOAT, KEY_VALUES: _nonfinite_name(number)}
        if type_ is tuple:
            return {KEY_TYPE: TYPENAME_TUPLE, KEY_VALUES: [self.to_dict(x) for x in cast("tuple[Any, ...]", o)]}
        if type_ is list:
            return [self.to_dict(x) for x in cast("list[Any]", o)]
        if type_ is dict:
            return self._dict_to_dict(cast("dict[Any, Any]", o))
        return self._to_dict_object(o)

    def _to_dict_object(self, o: object) -> Any:
        """Encode a value that is not exactly a builtin scalar or container.

        Order is significant. Registered transformers come first so an
        explicitly registered type --- ``NodeKey``, say --- wins over the generic
        attrs and dataclass handling that would otherwise also claim it.
        """
        if self.get_transformer_for_obj(o) is not None:
            return self._to_dict_transformer(o)
        if isinstance(o, Transformable):
            return {KEY_TYPE: TYPENAME_TRANSFORMABLE, KEY_CLASS: type(o).__name__, KEY_DATA: o.to_dict(self)}
        if HAS_ATTRS and attrs.has(type(o)):
            return self._attrs_to_dict(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return self._dataclass_to_dict(o)

        # Subclasses of the builtin containers, reached only when nothing above
        # claimed them: a namedtuple, an OrderedDict, a list subclass. Encoded as
        # the plain builtin form, which is what they did before exact-type
        # dispatch was introduced.
        if isinstance(o, tuple):
            return {KEY_TYPE: TYPENAME_TUPLE, KEY_VALUES: [self.to_dict(x) for x in o]}
        if isinstance(o, list):
            return [self.to_dict(x) for x in o]
        if isinstance(o, dict):
            return self._dict_to_dict(o)
        return self._to_dict_transformer(o)

    def _dict_to_dict(self, o: dict[Any, Any]) -> dict[str, Any]:
        """Convert a dictionary to serializable form.

        A dict whose keys are all plain strings, none of them reserved, is
        written as a JSON object --- the readable form, and the overwhelmingly
        common case. Anything else is written as a list of encoded key/value
        pairs, because a JSON object key can only be a string: the previous
        encoding turned ``{1: 'a'}`` into ``{'1': 'a'}`` and handed back a
        different dict than it was given.
        """
        if all(type(k) is str and k not in RESERVED_KEYS for k in o):
            return {k: self.to_dict(v) for k, v in o.items()}
        return {
            KEY_TYPE: TYPENAME_DICT,
            KEY_ITEMS: [[self.to_dict(k), self.to_dict(v)] for k, v in o.items()],
        }

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
            elif type_ == TYPENAME_FLOAT:
                return _NAME_TO_FLOAT[d[KEY_VALUES]]
            elif type_ == TYPENAME_TUPLE:
                return tuple(self.from_dict(x) for x in d[KEY_VALUES])
            elif type_ == TYPENAME_DICT:
                return self._from_escaped_dict(d)
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

    def _from_escaped_dict(self, d: dict[str, Any]) -> dict[Any, Any]:
        """Reconstruct a dict written in the escaped form.

        Two layouts are accepted. ``items`` is the current one, a list of encoded
        key/value pairs that supports keys of any transformable type. ``data`` is
        the older layout, a JSON object that could only ever hold string keys;
        files written before this change still use it.
        """
        if KEY_ITEMS in d:
            return {self.from_dict(k): self.from_dict(v) for k, v in d[KEY_ITEMS]}
        return {k: self.from_dict(v) for k, v in d[KEY_DATA].items()}

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
        """Convert numpy array to dictionary with shape, dtype, and data.

        The shape and dtype stay inline whichever way the data goes, so the
        manifest still describes every array in the graph without a single blob
        being decoded.
        """
        assert isinstance(o, np.ndarray)  # noqa: S101
        head: dict[str, Any] = {"shape": list(o.shape), "dtype": o.dtype.str}
        if o.dtype.hasobject:
            # An object array's elements are arbitrary Python values; .npy would
            # have to pickle them, so it stays on the inline path.
            head["data"] = transformer.to_dict(o.ravel().tolist())  # type: ignore[arg-type]
            return head
        if transformer.offer_blob(nbytes=o.nbytes):
            head["encoding"] = "npy"
            head["data"] = transformer.put_blob(
                lambda f: np.save(f, o, allow_pickle=False),
                codec="npy",
                dedupe_on=o,
            )
        else:
            head["data"] = transformer.to_dict(o.ravel().tolist())  # type: ignore[arg-type]
        return head

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct numpy array from dictionary."""
        if d.get("encoding") == "npy":
            import io

            return np.load(io.BytesIO(transformer.get_blob(d["data"])), allow_pickle=False)
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


class DefinitionMethodTransformer(CustomTransformer):
    """Transformer for calc nodes declared on a ``@ComputationFactory`` class.

    A computation built from a class stores each calc node's function as a method
    *bound to the definition object* --- the instance the factory created. That
    has no importable path of its own:
    :class:`FunctionRefTransformer` looks up ``module.Portfolio.signal``, but
    after decoration the name ``Portfolio`` refers to the factory function rather
    than to the class, so the lookup fails.

    The consequence, before this transformer existed, was that the library's own
    primary idiom produced graphs whose functions were silently dropped. Values
    reloaded; nodes could never recompute.

    The route back is through the factory. ``functools.wraps(cls)`` leaves the
    class on the factory as ``__wrapped__``, so the method is stored as the class
    name plus the method name, and rebuilt by importing the module, recovering
    the class, and binding the method to a definition object.

    .. note::
        That definition object is a **new** instance, not the one the graph was
        built with. For the ordinary case --- a class whose body only declares
        nodes --- the two are indistinguishable. A class whose ``__init__``
        computes state, or whose methods mutate ``self`` at run time, will not
        see that state after a round-trip. A class that cannot be constructed
        without arguments cannot be restored at all, and its node falls back to
        being stored as ``null`` with a warning.

        One instance is created per class per transformer, so all of a
        computation's nodes share a definition object, as they did originally.
    """

    def __init__(self) -> None:
        """Start with no definition objects built."""
        self._instances: dict[type, object] = {}

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "definition_method"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a bound method as its class name and method name."""
        func = getattr(o, "__func__", None)
        if func is None:
            msg = f"Cannot serialize {o!r}: not a bound method"
            raise ValueError(msg)

        qualname = getattr(func, "__qualname__", "") or ""
        module = getattr(func, "__module__", None)
        if module is None or "." not in qualname:
            msg = f"Cannot serialize {o!r}: no class-qualified name to resolve it by"
            raise ValueError(msg)
        if "<locals>" in qualname:
            msg = f"Cannot serialize {o!r}: defined inside a function, so it is not importable"
            raise ValueError(msg)

        class_qualname, _, method = qualname.rpartition(".")

        # Resolve now rather than on load. A method that cannot be rebuilt should
        # be reported while the graph is still in front of the person saving it,
        # not when someone else opens the file.
        rebuilt = self._resolve(module, class_qualname, method)
        if getattr(rebuilt, "__func__", None) is not func:
            msg = f"Cannot serialize {o!r}: {module}.{qualname} does not resolve back to this method"
            raise ValueError(msg)

        return {"module": module, "class_qualname": class_qualname, "method": method}

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Rebuild the bound method from its class and method names."""
        return self._resolve(d["module"], d["class_qualname"], d["method"])

    def _resolve(self, module: str, class_qualname: str, method: str) -> object:
        """Return the bound method for *method* on the definition class."""
        cls = self._definition_class(module, class_qualname)
        instance = self._instances.get(cls)
        if instance is None:
            try:
                instance = cls()
            except TypeError as exc:
                msg = (
                    f"Cannot rebuild {class_qualname}: its definition class cannot be constructed "
                    f"without arguments ({exc})"
                )
                raise ValueError(msg) from exc
            self._instances[cls] = instance

        bound = getattr(instance, method, None)
        if bound is None or not callable(bound):
            msg = f"Cannot rebuild {class_qualname}.{method}: no such method on the definition class"
            raise ValueError(msg)
        return bound

    @staticmethod
    def _definition_class(module: str, class_qualname: str) -> type:
        """Return the class named by *class_qualname*, seeing through the factory."""
        try:
            obj: Any = importlib.import_module(module)
            for part in class_qualname.split("."):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as exc:
            msg = f"Cannot resolve {module}.{class_qualname}: {exc}"
            raise ValueError(msg) from exc

        # After @ComputationFactory the name refers to the factory function;
        # functools.wraps left the class on it as __wrapped__.
        if not isinstance(obj, type):
            obj = getattr(obj, "__wrapped__", obj)
        if not isinstance(obj, type):
            # ValueError, not TypeError: the complaint is about what this name
            # resolved to, not about an argument the caller passed. Every other
            # unresolvable-callable path raises ValueError too, and the callers
            # that turn this into a warning key off that.
            msg = f"Cannot resolve {module}.{class_qualname}: it is not a class or a computation factory"
            raise ValueError(msg)  # noqa: TRY004
        return obj

    @property
    def supported_subtypes(self) -> Iterable[Any]:
        """Match bound methods, which are more specific than plain callables."""
        return [types.MethodType]


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


def _encode_frame_as_parquet(transformer: "Transformer", frame: "pd.DataFrame") -> dict[str, Any] | None:
    """Encode *frame* as a parquet blob, or return ``None`` to use the default path.

    Parquet is opt-in via ``frame_encoding="parquet"`` on the profile. It buys
    columnar compression and a file other tools can read, at the cost of an
    optional pyarrow dependency.

    Returns ``None`` --- meaning "not this way" --- when parquet was not asked
    for, when the value is too small to be worth a blob, or when pyarrow cannot
    represent this particular frame. That last case is why the conversion is
    attempted before anything is written: a frame with duplicate column names or
    an exotic dtype should fall back to an encoding that works, not fail the
    save.
    """
    if transformer.blob_setting("frame_encoding", "npy") != "parquet":
        return None

    estimated = int(frame.memory_usage(deep=False).sum())
    if not transformer.offer_blob(nbytes=estimated):
        return None

    try:
        from loman._extras import require

        pa = require("pyarrow", "efficient")
        pq = require("pyarrow.parquet", "efficient")
        table = pa.Table.from_pandas(frame, preserve_index=True)
    except Exception:
        return None

    def write(f: Any) -> None:
        pq.write_table(table, f, compression="zstd")

    return {
        "shape": list(frame.shape),
        "encoding": "parquet",
        # Parquet already compresses; compressing the blob again would cost time
        # and achieve nothing.
        "data": transformer.put_blob(write, codec="parquet", compressible=False, dedupe_on=frame),
    }


def _encode_column(transformer: "Transformer", column: "pd.Series") -> Any:
    """Encode one DataFrame column, as an array where that is lossless.

    A column backed by a plain numpy dtype is handed to the ndarray transformer,
    which means it inherits out-of-line storage for free --- this is what keeps a
    large numeric frame from being written one decimal string at a time.

    Extension dtypes (categorical, nullable integers, timezone-aware datetimes,
    pandas strings) are *not* sent that way. Their numpy representation loses the
    thing that makes them distinct --- a tz-aware column flattens to naive UTC,
    and reading it back as a local wall time would silently shift every value ---
    so they go through the element-wise path and are restored from their
    recorded dtype.
    """
    dtype = column.dtype
    if isinstance(dtype, np.dtype) and not dtype.hasobject:
        return transformer.to_dict(column.to_numpy())
    return transformer.to_dict(column.tolist())


class DataFrameTransformer(CustomTransformer):
    """Transformer for :class:`pandas.DataFrame` objects."""

    @property
    def name(self) -> str:
        """Return transformer name."""
        return "dataframe"

    def to_dict(self, transformer: "Transformer", o: object) -> dict[str, Any]:
        """Serialize a DataFrame using split orientation.

        Columns are encoded per column rather than through ``o.values``, which
        would push every cell through one shared dtype --- object for a frame
        mixing numbers and strings --- and lose the per-column types on the way.
        Both axes go through the transformer as indexes, so a ``MultiIndex`` or a
        ``DatetimeIndex`` on either axis survives the round-trip.
        """
        assert isinstance(o, pd.DataFrame)  # noqa: S101
        parquet = _encode_frame_as_parquet(transformer, o)
        if parquet is not None:
            return parquet
        return {
            "columns": transformer.to_dict(o.columns),
            "index": transformer.to_dict(o.index),
            "data": [_encode_column(transformer, o.iloc[:, i]) for i in range(o.shape[1])],
            "orient": "columns",
            "dtypes": [str(dtype) for dtype in o.dtypes],
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a DataFrame from its serialized form."""
        # A parquet blob carries its own schema, columns and index in the file
        # footer, so it is decoded before anything else is read from the entry.
        if d.get("encoding") == "parquet":
            import io

            from loman._extras import require

            pq = require("pyarrow.parquet", "efficient")
            return pq.read_table(io.BytesIO(transformer.get_blob(d["data"]))).to_pandas()

        columns = transformer.from_dict(d["columns"])
        index = transformer.from_dict(d["index"])
        dtypes = d.get("dtypes", {})

        if d.get("orient") == "columns":
            data = {i: transformer.from_dict(col) for i, col in enumerate(d["data"])}
            df = pd.DataFrame(data, index=index)
            df.columns = pd.Index(columns) if not isinstance(columns, pd.Index) else columns
            for i, dtype in enumerate(dtypes):
                with contextlib.suppress(ValueError, TypeError):
                    df.isetitem(i, df.iloc[:, i].astype(dtype))
            return df

        # Row-major layout written before the per-column encoding existed.
        df = pd.DataFrame(transformer.from_dict(d["data"]), columns=columns, index=index)
        for col, dtype in (dtypes or {}).items():
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
            "name": transformer.to_dict(o.name),
            "dtype": str(o.dtype),
            "index": transformer.to_dict(o.index),
            "data": transformer.to_dict(o.tolist()),
        }

    def from_dict(self, transformer: "Transformer", d: dict[str, Any]) -> object:
        """Reconstruct a Series from its serialized form."""
        data = transformer.from_dict(d["data"])
        index = transformer.from_dict(d["index"])
        s = pd.Series(data, index=index, name=transformer.from_dict(d.get("name")))
        with contextlib.suppress(ValueError, TypeError):
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
