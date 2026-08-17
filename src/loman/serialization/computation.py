"""Serialization for Computation graphs to/from JSON."""

from __future__ import annotations

import contextvars
import io
import json
import warnings
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TextIO

from loman.consts import EdgeAttributes, NodeAttributes, States, SystemTags
from loman.exception import SerializationError
from loman.nodekey import parse_nodekey

from .blobs import (
    CONTAINER_STORE,
    MANIFEST_NAME,
    BlobReader,
    BlobStore,
    BlobWriter,
    DirBlobStore,
    ZipBlobStore,
    read_dir_manifest,
    read_zip_manifest,
    write_dir_container,
    write_zip_container,
)
from .profile import READABLE, SerializationProfile, resolve_profile
from .transformer import (
    DataFrameTransformer,
    DefinitionMethodTransformer,
    DillFunctionTransformer,
    EnumTransformer,
    FunctionRefTransformer,
    NdArrayTransformer,
    NodeKeyTransformer,
    SeriesTransformer,
    Transformer,
    UntransformableTypeError,
)
from .values import register_value_transformers

if TYPE_CHECKING:
    pass

# Serialization format version — bump when the schema changes.
#
# This is version 1: the first version of the format that is meant to be kept.
# The shapes written by pre-release versions of loman are still read, on a
# best-effort basis, by the reader branches marked "legacy" below --- an escaped
# dict under "data" rather than "items", a row-major DataFrame, an index written
# as a list of elements. Those predate any published guarantee and are not
# themselves versioned, which is why the number restarts here rather than
# continuing a sequence that never meant anything to anyone outside the repo.
#
# Note that the reader has never inspected this field, so a bump cannot make an
# older release reject a newer file. Forward compatibility comes from keeping
# changes additive, not from this number.
FORMAT_VERSION = 1

# Sentinel for a constant the policy dropped, distinct from a legitimate None.
_DROPPED = object()

_CONSTANT_POLICIES = frozenset({"raise", "drop"})


class UnserializableConstantWarning(UserWarning):
    """A constant argument was dropped, so the saved graph cannot be recalculated."""


class UnserializableFunctionWarning(UserWarning):
    """A node's function could not be encoded, so that node cannot be recalculated.

    The node's *value* is still saved. What is lost is the ability to recompute
    it: the reloaded node holds what it last produced and stays out of date
    forever, because there is nothing left to call.

    The common cause is a function that is not importable by name --- a lambda, a
    closure, or a bound method of a class built by
    :func:`~loman.computeengine.computation_factory`.
    """


CONTAINERS = ("zip", "dir", "json")


class _SaveState(NamedTuple):
    """The blob writer and profile in force for one save."""

    writer: Any
    profile: Any


# Per-save state lives in a ContextVar rather than on the serializer, because a
# serializer is a natural thing to build once and reuse --- and instance
# attributes would mean two concurrent saves through the same one silently
# writing into each other's archive. Measured: 11 of 12 threads failed that way.
_SAVE_STATE: contextvars.ContextVar[_SaveState | None] = contextvars.ContextVar("loman_save_state", default=None)

# Zip local file header. Every .loman archive starts with it.
_ZIP_MAGIC = b"PK\x03\x04"

# Pickle protocol 2+ opening opcode, as written by dill. Recognised only so a
# write_dill file gets a useful error instead of "not valid JSON".
_PICKLE_MAGIC = b"\x80"


def infer_container_from_path(path: Path) -> str:
    """Return the container implied by *path*'s suffix.

    ``.json`` means the single readable document; everything else means a
    ``.loman`` zip, which is the default because a single file is what people
    move around.
    """
    return "json" if path.suffix.lower() == ".json" else "zip"


def _resolve_profile_for_container(
    profile: str | SerializationProfile | None,
    container: str,
    stores: dict[str, Any] | None = None,
) -> SerializationProfile:
    """Return the profile to use, rejecting the one impossible combination."""
    if container not in CONTAINERS:
        msg = f"Unknown container {container!r}; expected one of {list(CONTAINERS)}"
        raise ValueError(msg)

    if container == "json":
        if profile is None:
            return READABLE
        resolved = resolve_profile(profile)
        # A single JSON document holds no blobs of its own. It can still carry
        # out-of-line values when an external store is supplied, so the refusal
        # only applies when there is genuinely nowhere for them to go.
        if resolved.inline_max_bytes is not None and not stores:
            msg = (
                f"The {resolved.name!r} profile writes values out of line, and a single JSON "
                "document has nowhere to put them. Use container='zip' (or a .loman path) to "
                "keep this profile, profile='readable' to keep the single document, or pass "
                "stores={...} to hold the values somewhere of your own."
            )
            raise ValueError(msg)
        return resolved

    return resolve_profile(profile)


def sniff_container(path: Path) -> str:
    """Return which container *path* holds, by looking at it.

    Order matters: a directory is checked first because it cannot be read as
    bytes, then the zip magic number, then a leading brace for the single
    document. A dill pickle is recognised only to say so, since "not valid JSON"
    would be a poor description of that mistake.
    """
    if path.is_dir():
        return "dir"
    if not path.exists():
        msg = f"No such file or directory: {str(path)!r}"
        raise FileNotFoundError(msg)

    with path.open("rb") as f:
        head = f.read(4)

    if head.startswith(_ZIP_MAGIC):
        return "zip"
    stripped = head.lstrip()
    if stripped.startswith(b"{"):
        return "json"
    if head.startswith(_PICKLE_MAGIC):
        msg = (
            f"{str(path)!r} looks like a pickle written by write_dill, not a loman container. "
            "Use Computation.read_dill to read it."
        )
        raise SerializationError(msg)

    msg = (
        f"Cannot tell what {str(path)!r} is. Expected a .loman archive, a directory "
        f"containing {MANIFEST_NAME}, or a JSON document."
    )
    raise SerializationError(msg)


def default_computation_transformer() -> Transformer:
    """Create a Transformer pre-registered with all types needed for Computation serialization."""
    t = Transformer()

    # Numeric arrays
    t.register(NdArrayTransformer())

    # Enums: register the States enum so node states roundtrip correctly.
    enum_t = EnumTransformer()
    enum_t.register_enum(States)
    t.register(enum_t)

    # Importable callables (module-level functions).  Lambdas / closures raise.
    t.register(FunctionRefTransformer())

    # Calc nodes declared on a @ComputationFactory class, which are bound methods
    # of the definition object and so have no importable path of their own.
    t.register(DefinitionMethodTransformer())

    # Pandas
    t.register(DataFrameTransformer())
    t.register(SeriesTransformer())

    # NodeKey (hierarchical node names)
    t.register(NodeKeyTransformer())

    # Dates, indexes, numpy scalars, sets, bytes, decimals.
    register_value_transformers(t)

    # Per-node execution timing. Imported here rather than at module scope
    # because computeengine reaches back into this module to serialize.
    from loman.computeengine import TimingData

    t.register(TimingData)

    return t


def dill_computation_transformer() -> Transformer:
    """Create a Transformer that serializes all callables — including lambdas and closures — via dill.

    Identical to :func:`default_computation_transformer` except that
    :class:`~loman.serialization.transformer.DillFunctionTransformer` replaces
    :class:`~loman.serialization.transformer.FunctionRefTransformer`, so lambdas
    and locally-defined closures are serialized as base64-encoded dill blobs
    rather than raising :class:`~loman.exception.SerializationError`.
    """
    t = Transformer()

    t.register(NdArrayTransformer())

    enum_t = EnumTransformer()
    enum_t.register_enum(States)
    t.register(enum_t)

    # Dill-based callable serializer — handles lambdas and closures.
    t.register(DillFunctionTransformer())

    t.register(DataFrameTransformer())
    t.register(SeriesTransformer())
    t.register(NodeKeyTransformer())

    # Dates, indexes, numpy scalars, sets, bytes, decimals.
    register_value_transformers(t)

    # Per-node execution timing. Imported here rather than at module scope
    # because computeengine reaches back into this module to serialize.
    from loman.computeengine import TimingData

    t.register(TimingData)

    return t


class ComputationSerializer:
    """Serialize and deserialize a :class:`~loman.computeengine.Computation` graph to JSON.

    The serialized format is a JSON object with the following top-level keys:

    - ``version``: integer format version
    - ``nodes``: list of node objects
    - ``edges``: list of edge objects

    Each **node** object has:

    - ``key``: string representation of the NodeKey
    - ``state``: name of the :class:`~loman.consts.States` enum member (or ``null``)
    - ``value``: transformer-encoded value (or ``null`` when absent / not serialized)
    - ``has_value``: bool — false when the node has no meaningful value to restore
    - ``func``: transformer-encoded callable (or ``null``)
    - ``args``: transformer-encoded constant positional arguments, keyed by
      stringified positional index
    - ``kwds``: transformer-encoded constant keyword arguments, keyed by parameter name
    - ``serialize``: bool — whether the node has the ``__serialize__`` tag
    - ``tags``: list of non-system tags

    Each **edge** object has:

    - ``src``: string key of the source node
    - ``dst``: string key of the destination node
    - ``param_type``: ``"arg"`` or ``"kwd"``
    - ``param``: positional index (int) for args, parameter name (str) for kwds

    Arguments taken from other nodes are recorded on **edges**, while arguments
    given as :class:`~loman.computeengine.ConstantValue` are held on the node
    itself and recorded in ``args`` and ``kwds``. Both are needed to call a node's
    function, so a graph that dropped its constants would raise a
    :class:`TypeError` the next time the node was calculated.

    Parameters
    ----------
    transformer:
        Custom :class:`~loman.serialization.transformer.Transformer` instance.
        If ``None``, a default transformer is built based on *use_dill_for_functions*.
    use_dill_for_functions:
        When ``True``, lambdas and closures are serialized as base64-encoded dill
        blobs rather than raising :class:`~loman.exception.SerializationError`.
        Has no effect when a custom *transformer* is supplied.  Defaults to ``False``.
    on_unserializable_constant:
        What to do when a constant argument cannot be encoded. ``"raise"``, the
        default, refuses to write a graph that could not be recalculated.
        ``"drop"`` omits the constant and emits
        :class:`UnserializableConstantWarning`, restoring the behaviour of
        releases before constants were recorded — where such a graph saved
        silently and then raised :class:`TypeError` from the missing argument on
        the first recalculation. It exists so an existing codebase can keep
        writing files while it is fixed, not as a setting to leave in place.
    """

    # States that never carry a value worth writing. Everything else is
    # serialized when the node actually holds a value --- notably STALE, whose
    # value the in-memory computation keeps and whose intermediates are the
    # whole point of saving a graph for post-mortem inspection.
    _VALUELESS_STATES: ClassVar[set[States]] = {States.PLACEHOLDER, States.UNINITIALIZED}

    def __init__(
        self,
        transformer: Transformer | None = None,
        *,
        use_dill_for_functions: bool = False,
        on_unserializable_constant: str = "raise",
    ) -> None:
        """Initialise with an optional custom transformer."""
        if on_unserializable_constant not in _CONSTANT_POLICIES:
            msg = (
                f"on_unserializable_constant must be one of {sorted(_CONSTANT_POLICIES)}, "
                f"got {on_unserializable_constant!r}"
            )
            raise ValueError(msg)
        if transformer is None:
            transformer = (
                dill_computation_transformer() if use_dill_for_functions else default_computation_transformer()
            )
        self._t = transformer
        self._use_dill_for_functions = use_dill_for_functions
        self._on_unserializable_constant = on_unserializable_constant
        # Set per load() / loads() call; see the allow_code parameter there.
        self._allow_code = True

    def register(self, t: Any) -> None:
        """Register a transformer or type with this serializer's transformer.

        Accepts anything :meth:`~loman.serialization.transformer.Transformer.register`
        accepts: a :class:`~loman.serialization.transformer.CustomTransformer`
        instance, a :class:`~loman.serialization.transformer.Transformable`
        subclass, an attrs class, or a dataclass.

        The same serializer instance must be used for both writing and reading,
        since the registration lives on the instance::

            s = ComputationSerializer()
            s.register(my_transformer)
            comp.write_json('comp.json', serializer=s)
            comp2 = Computation.read_json('comp.json', serializer=s)
        """
        self._t.register(t)

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------

    def save(
        self,
        comp: Any,
        path: str | Path,
        *,
        profile: str | SerializationProfile | None = None,
        container: str | None = None,
        stores: dict[str, BlobStore] | None = None,
    ) -> None:
        """Write *comp* to *path*.

        :param path: Destination. A ``.json`` suffix implies the single-document
            container; anything else defaults to a ``.loman`` zip.
        :param profile: ``"readable"``, ``"efficient"``, or a
            :class:`~loman.serialization.profile.SerializationProfile`. Defaults
            to efficient, except in the ``json`` container where only readable is
            possible.
        :param container: ``"zip"``, ``"dir"`` or ``"json"``. Inferred from
            *path* when omitted.
        :param stores: Named :class:`~loman.serialization.blobs.BlobStore`
            instances that nodes may be routed to. A node names a store through
            ``add_node(store=...)`` or a profile override.
        """
        path = Path(path)
        container = container or infer_container_from_path(path)
        resolved = _resolve_profile_for_container(profile, container, stores)
        external = dict(stores or {})

        if container == "json":
            with path.open("w", encoding="utf-8") as f:
                self._dump_document(comp, f, resolved, external)
            return

        writer = write_zip_container if container == "zip" else write_dir_container
        writer(
            path,
            lambda container_stores: self._build_manifest(comp, {**container_stores, **external}, resolved, container),
        )

    @staticmethod
    def load_path(
        path: str | Path,
        *,
        serializer: ComputationSerializer | None = None,
        allow_code: bool = True,
        stores: dict[str, BlobStore] | None = None,
    ) -> Any:
        """Read a computation from *path*, whatever container it uses.

        :param stores: Named stores for blobs held outside the container. A
            saved file records a store's name but never its configuration, so a
            file with external blobs cannot resolve them unaided.
        """
        s = serializer if serializer is not None else ComputationSerializer()
        path = Path(path)
        container = sniff_container(path)
        external = dict(stores or {})

        if container == "json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            return s._read_manifest(manifest, external, allow_code=allow_code)

        if container == "dir":
            manifest = read_dir_manifest(path)
            all_stores = {CONTAINER_STORE: DirBlobStore(path), **external}
            return s._read_manifest(manifest, all_stores, allow_code=allow_code)

        with zipfile.ZipFile(path) as zf:
            manifest = read_zip_manifest(zf)
            all_stores = {CONTAINER_STORE: ZipBlobStore(zf), **external}
            return s._read_manifest(manifest, all_stores, allow_code=allow_code)

    def _read_manifest(self, manifest: dict[str, Any], stores: dict[str, BlobStore], *, allow_code: bool) -> Any:
        """Rebuild a computation from *manifest*, resolving blobs against *stores*."""
        reader = BlobReader(manifest.get("blobs", []), stores)
        with self._t.reading(reader):
            return self._from_dict(manifest, allow_code=allow_code)

    def _build_manifest(
        self,
        comp: Any,
        stores: dict[str, BlobStore],
        profile: SerializationProfile,
        container: str,
    ) -> dict[str, Any]:
        """Return the manifest for *comp*, writing any blobs into *stores*."""
        writer = BlobWriter(
            stores,
            compression=profile.compression,
            dedupe=profile.dedupe,
            checksums=profile.checksums,
        )
        token = _SAVE_STATE.set(_SaveState(writer=writer, profile=profile))
        try:
            manifest = self._to_dict(comp)
        finally:
            _SAVE_STATE.reset(token)
        manifest["container"] = container
        manifest["profile"] = profile.name
        manifest["blobs"] = writer.table()
        return manifest

    def _dump_document(
        self,
        comp: Any,
        fp: TextIO,
        profile: SerializationProfile,
        stores: dict[str, BlobStore] | None = None,
    ) -> None:
        """Write the single-document form.

        The container itself holds no blobs, but an external store still can, so
        a readable manifest can sit alongside data held elsewhere.
        """
        data = self._build_manifest(comp, dict(stores or {}), profile, "json")
        json.dump(data, fp, allow_nan=False)

    def dump(self, comp: Any, fp: TextIO) -> None:
        """Serialize *comp* to *fp* (a text-mode file-like object)."""
        self._dump_document(comp, fp, READABLE)

    def dumps(self, comp: Any) -> str:
        """Serialize *comp* and return a JSON string.

        Always the readable single-document form: a string has nowhere to put
        out-of-line bytes.
        """
        buf = io.StringIO()
        self._dump_document(comp, buf, READABLE)
        return buf.getvalue()

    def _serialize_node_value(self, node_key: Any, state: States | None, node_data: dict[str, Any]) -> tuple[Any, bool]:
        """Return ``(encoded_value, has_value)`` for a node that should be serialized.

        Raises :class:`~loman.exception.SerializationError` if the value cannot
        be encoded.
        """
        from loman.computeengine import Error

        if state in self._VALUELESS_STATES or NodeAttributes.VALUE not in node_data:
            return None, False

        raw_value = node_data[NodeAttributes.VALUE]
        # Keyed off the value's type, not the node's state. A node that failed
        # and then went STALE --- because one of its inputs was replaced --- is no
        # longer in ERROR state but still holds the Error it produced. Testing
        # the state instead sent that value down the generic path, where an
        # exception object has no encoding, and failed the entire save.
        if isinstance(raw_value, Error):
            exception_type = type(raw_value.exception)
            return (
                {
                    "__loman_error__": True,
                    "exception_type": exception_type.__name__,
                    "exception_module": exception_type.__module__,
                    "exception_str": str(raw_value.exception),
                    "traceback": raw_value.traceback,
                },
                True,
            )

        try:
            return self._t.to_dict(raw_value), True
        except (UntransformableTypeError, ValueError) as exc:
            msg = f"Cannot serialize value of node {node_key!r}: {exc}"
            raise SerializationError(msg) from exc

    def _serialize_node_func(self, node_key: Any, raw_func: Any) -> Any:
        """Return the encoded function for a node, or ``None`` if it cannot be serialized.

        Lambdas raise :class:`~loman.exception.SerializationError` unless
        ``use_dill_for_functions`` is enabled.  Other non-importable callables
        (e.g. framework closures from ``add_block``) are silently stored as ``null``.
        """
        qualname = getattr(raw_func, "__qualname__", "") or ""
        if not self._use_dill_for_functions and "<lambda>" in qualname:
            msg = (
                f"Cannot serialize lambda function on node {node_key!r}. "
                "Use a module-level importable function, serialize=False, "
                "or ComputationSerializer(use_dill_for_functions=True)."
            )
            raise SerializationError(msg)
        try:
            return self._t.to_dict(raw_func)
        except (UntransformableTypeError, ValueError, TypeError) as exc:
            # The value is still saved; only the ability to recalculate is lost.
            # That used to happen silently, so a graph could be reloaded, look
            # complete, and never update again with nothing to explain why.
            warnings.warn(
                f"Cannot serialize the function on node {node_key!r} ({exc}). Its value is still "
                "saved, but the reloaded node will have no function and so can never be "
                "recalculated. Use a module-level importable function, or "
                "ComputationSerializer(use_dill_for_functions=True).",
                UnserializableFunctionWarning,
                stacklevel=2,
            )
            return None

    def _serialize_constant(self, node_key: Any, param: Any, value: Any) -> Any:
        """Encode one constant argument held on a node.

        Unlike a node function, a constant argument has no fallback: dropping it
        would leave the node callable with the wrong number of arguments, so an
        unrepresentable constant is an error rather than a ``null``.
        """
        try:
            return self._t.to_dict(value)
        except (UntransformableTypeError, ValueError, TypeError) as e:
            msg = (
                f"Cannot serialize constant argument {param!r} on node {node_key!r} ({e}). "
                "Constant arguments are needed to call the node's function, so they cannot "
                "be skipped: register a transformer for the type, set serialize=False on the "
                "node, or use ComputationSerializer(use_dill_for_functions=True) for callables."
            )
            if self._on_unserializable_constant == "raise":
                raise SerializationError(msg) from e
            warnings.warn(
                f"{msg} Dropping it, because on_unserializable_constant='drop'. The saved "
                "graph will raise TypeError from the missing argument when this node is "
                "recalculated.",
                UnserializableConstantWarning,
                stacklevel=2,
            )
            return _DROPPED

    def _serialize_node_constants(
        self, node_key: Any, node_data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the encoded constant positional and keyword arguments for a node.

        A constant the policy drops is omitted from the mapping entirely, so the
        reloaded node looks exactly as it did before constants were recorded.
        """
        encoded_args = {
            str(index): encoded
            for index, value in node_data.get(NodeAttributes.ARGS, {}).items()
            if (encoded := self._serialize_constant(node_key, index, value)) is not _DROPPED
        }
        encoded_kwds = {
            name: encoded
            for name, value in node_data.get(NodeAttributes.KWDS, {}).items()
            if (encoded := self._serialize_constant(node_key, name, value)) is not _DROPPED
        }
        return encoded_args, encoded_kwds

    def _serialize_node(self, node_key: Any, node_data: dict[str, Any]) -> dict[str, Any]:
        """Return the serialized dict for a single node.

        The whole node is encoded inside one write scope, so any value that asks
        for out-of-line storage --- a node's value, or a constant argument that
        happens to be a large array --- is attributed to this node in the blob
        table.
        """
        state = _SAVE_STATE.get()
        if state is None:
            # save()/dumps() establish this; only reached if a caller drives
            # _serialize_node directly.
            return self._serialize_node_inner(node_key, node_data)  # pragma: no cover
        tags: set[str] = node_data.get(NodeAttributes.TAG, set())
        with self._t.writing(
            state.writer,
            state.profile,
            node=str(node_key),
            tags=frozenset(tags),
            store=node_data.get(NodeAttributes.STORE),
        ):
            return self._serialize_node_inner(node_key, node_data)

    def _serialize_node_inner(self, node_key: Any, node_data: dict[str, Any]) -> dict[str, Any]:
        """Return the serialized dict for a single node, inside a write scope."""
        state: States | None = node_data.get(NodeAttributes.STATE)
        tags: set[str] = node_data.get(NodeAttributes.TAG, set())
        serialize_flag: bool = SystemTags.SERIALIZE in tags

        serialized_state: States | None
        if not serialize_flag:
            serialized_state = States.UNINITIALIZED
            encoded_value = None
            has_value = False
        else:
            serialized_state = state
            encoded_value, has_value = self._serialize_node_value(node_key, state, node_data)

        raw_func = node_data.get(NodeAttributes.FUNC)
        encoded_func = (
            self._serialize_node_func(node_key, raw_func) if raw_func is not None and serialize_flag else None
        )
        encoded_args, encoded_kwds = (
            self._serialize_node_constants(node_key, node_data) if encoded_func is not None else ({}, {})
        )

        user_tags = [t for t in tags if not t.startswith("__")]

        out = {
            "key": str(node_key),
            "state": serialized_state.name if serialized_state is not None else None,
            "value": encoded_value,
            "has_value": has_value,
            "func": encoded_func,
            "args": encoded_args,
            "kwds": encoded_kwds,
            "serialize": serialize_flag,
            "tags": user_tags,
        }
        out.update(self._serialize_node_attributes(node_key, node_data, serialize_flag=serialize_flag))
        return out

    def _serialize_node_attributes(
        self, node_key: Any, node_data: dict[str, Any], *, serialize_flag: bool
    ) -> dict[str, Any]:
        """Return the presentational and execution attributes of a node.

        Group, style and executor are plain strings describing how a node is
        drawn and where it runs. They were previously dropped on load and
        rebuilt as ``None``, so a reloaded graph rendered differently from the
        one that was saved. The converter is a callable and follows the same
        rules as the node's function; timing is written for the record and
        restored as data.
        """
        attributes: dict[str, Any] = {}
        for field, attr in (("group", NodeAttributes.GROUP), ("style", NodeAttributes.STYLE)):
            value = node_data.get(attr)
            if value is not None:
                attributes[field] = self._t.to_dict(value)

        executor = node_data.get(NodeAttributes.EXECUTOR)
        if executor is not None:
            attributes["executor"] = self._t.to_dict(executor)

        store = node_data.get(NodeAttributes.STORE)
        if store is not None:
            attributes["store"] = self._t.to_dict(store)

        converter = node_data.get(NodeAttributes.CONVERTER)
        if converter is not None and serialize_flag:
            encoded = self._serialize_node_func(node_key, converter)
            if encoded is not None:
                attributes["converter"] = encoded

        timing = node_data.get(NodeAttributes.TIMING)
        if timing is not None:
            attributes["timing"] = self._t.to_dict(timing)

        return attributes

    def _restore_node_attributes(self, node_info: dict[str, Any], node_data: dict[str, Any]) -> None:
        """Apply the presentational and execution attributes from *node_info*."""
        node_data[NodeAttributes.GROUP] = self._t.from_dict(node_info.get("group"))
        node_data[NodeAttributes.STYLE] = self._t.from_dict(node_info.get("style"))
        node_data[NodeAttributes.EXECUTOR] = self._t.from_dict(node_info.get("executor"))
        node_data[NodeAttributes.STORE] = self._t.from_dict(node_info.get("store"))

        encoded_converter = node_info.get("converter")
        node_data[NodeAttributes.CONVERTER] = self._decode_callable(encoded_converter)

        encoded_timing = node_info.get("timing")
        if encoded_timing is not None:
            node_data[NodeAttributes.TIMING] = self._t.from_dict(encoded_timing)

    def _serialize_edge(self, src: Any, dst: Any, edge_data: dict[str, Any]) -> dict[str, Any]:
        """Return the serialized dict for a single edge."""
        param = edge_data.get(EdgeAttributes.PARAM)
        if param is None:
            return {"src": str(src), "dst": str(dst), "param_type": None, "param": None}

        from loman.computeengine import _ParameterType

        param_type, param_val = param
        return {
            "src": str(src),
            "dst": str(dst),
            "param_type": "kwd" if param_type == _ParameterType.KWD else "arg",
            "param": param_val,
        }

    def _to_dict(self, comp: Any) -> dict[str, Any]:
        """Convert a Computation to a JSON-serializable dict."""
        nodes_out = [self._serialize_node(k, comp.dag.nodes[k]) for k in comp.dag.nodes()]
        edges_out = [self._serialize_edge(src, dst, data) for src, dst, data in comp.dag.edges(data=True)]
        return {
            "version": FORMAT_VERSION,
            "nodes": nodes_out,
            "edges": edges_out,
            "metadata": self._serialize_metadata(comp),
        }

    def _serialize_metadata(self, comp: Any) -> dict[str, Any]:
        """Return the computation's per-node metadata, keyed by node-key string.

        Metadata is held on the computation rather than on dag nodes, so it needs
        a map of its own. The root computation's own metadata is keyed by the
        empty string.
        """
        from loman.nodekey import NodeKey

        out: dict[str, Any] = {}
        for node_key, metadata in getattr(comp, "_metadata", {}).items():
            key = "" if node_key == NodeKey.root() else str(node_key)
            out[key] = self._t.to_dict(metadata)
        return out

    def _restore_metadata(self, comp: Any, encoded: dict[str, Any]) -> None:
        """Apply a serialized metadata map back onto *comp*."""
        from loman.nodekey import NodeKey

        for key, metadata in encoded.items():
            node_key = NodeKey.root() if key == "" else parse_nodekey(key)
            comp._metadata[node_key] = self._t.from_dict(metadata)

    def load(self, fp: TextIO, *, allow_code: bool = True) -> Any:
        """Deserialize a Computation from *fp* (a text-mode file-like object).

        :param allow_code: When false, node functions and converters are not
            restored. See :meth:`loads`.
        """
        data = json.load(fp)
        return self._from_dict(data, allow_code=allow_code)

    def loads(self, s: str, *, allow_code: bool = True) -> Any:
        """Deserialize a Computation from a JSON string.

        :param allow_code: When false, encoded callables are skipped rather than
            resolved, and every node's function and converter comes back as
            ``None``. Restoring a callable means importing the module the file
            names, or unpickling a dill blob out of it --- both of which run code
            the file chose. Values, structure, states and tags still load, which
            is enough to inspect a graph from an untrusted source. Defaults to
            true, preserving existing behaviour.
        """
        data = json.loads(s)
        return self._from_dict(data, allow_code=allow_code)

    def _decode_callable(self, encoded: Any) -> Any:
        """Resolve an encoded callable, or ``None`` when code loading is refused."""
        if encoded is None or not self._allow_code:
            return None
        return self._t.from_dict(encoded)

    def _decode_error_value(self, encoded: dict[str, Any]) -> Any:
        """Rebuild an ``Error`` from its recorded exception.

        Builtin exception types are reconstructed so that ``except ValueError``
        still matches after a round-trip. Anything else becomes a
        :class:`~loman.exception.DeserializedError` carrying the original name:
        importing the module a file names in order to rebuild its exception
        would be running code chosen by the file.
        """
        import builtins

        from loman.computeengine import Error
        from loman.exception import DeserializedError

        type_name = encoded.get("exception_type", "Exception")
        module = encoded.get("exception_module")
        message = encoded["exception_str"]

        exception: Exception | None = None
        if module in (None, "builtins"):
            candidate = getattr(builtins, type_name, None)
            # Exception, not BaseException: rebuilding a KeyboardInterrupt or a
            # SystemExit as a node value would be a surprising thing to hand back.
            if isinstance(candidate, type) and issubclass(candidate, Exception):
                try:
                    exception = candidate(message)
                except Exception:  # pragma: no cover - exotic __init__
                    exception = None
        if exception is None:
            exception = DeserializedError(message, exception_type=type_name, exception_module=module)

        return Error(exception=exception, traceback=encoded["traceback"])

    def _from_dict(self, data: dict[str, Any], *, allow_code: bool = True) -> Any:
        """Reconstruct a Computation from a deserialized dict."""
        from loman.computeengine import Computation, _ParameterType

        self._allow_code = allow_code
        comp = Computation()

        for node_info in data["nodes"]:
            raw_key = node_info["key"]
            node_key = parse_nodekey(raw_key)
            state_name = node_info["state"]
            state = States[state_name] if state_name is not None else None
            serialize_flag: bool = node_info.get("serialize", True)
            has_value: bool = node_info.get("has_value", False)
            user_tags: list[str] = node_info.get("tags", [])

            func = self._decode_callable(node_info.get("func"))

            encoded_value = node_info.get("value")
            if has_value:
                # Decode even when the encoded value is null: a node whose value
                # is legitimately None is not the same as a node with no value,
                # and has_value is what distinguishes them.
                if isinstance(encoded_value, dict) and encoded_value.get("__loman_error__"):
                    value = self._decode_error_value(encoded_value)
                else:
                    value = self._t.from_dict(encoded_value)
            else:
                value = None

            comp.dag.add_node(node_key)
            node_data = comp.dag.nodes[node_key]
            node_data[NodeAttributes.STATE] = state if state is not None else States.UNINITIALIZED
            node_data[NodeAttributes.VALUE] = value if has_value else None
            node_data[NodeAttributes.FUNC] = func
            node_data[NodeAttributes.ARGS] = {
                int(index): self._t.from_dict(encoded) for index, encoded in node_info.get("args", {}).items()
            }
            node_data[NodeAttributes.KWDS] = {
                name: self._t.from_dict(encoded) for name, encoded in node_info.get("kwds", {}).items()
            }
            node_data[NodeAttributes.TAG] = set()
            self._restore_node_attributes(node_info, node_data)

            if serialize_flag:
                node_data[NodeAttributes.TAG].add(SystemTags.SERIALIZE)
            for tag in user_tags:
                node_data[NodeAttributes.TAG].add(tag)

        for edge_info in data["edges"]:
            src_key = parse_nodekey(edge_info["src"])
            dst_key = parse_nodekey(edge_info["dst"])
            param_type_str = edge_info.get("param_type")
            param_val = edge_info.get("param")

            if param_type_str is not None:
                param_type = _ParameterType.KWD if param_type_str == "kwd" else _ParameterType.ARG
                comp.dag.add_edge(src_key, dst_key, **{EdgeAttributes.PARAM: (param_type, param_val)})
            else:
                comp.dag.add_edge(src_key, dst_key)

        self._restore_metadata(comp, data.get("metadata", {}))
        comp._refresh_maps()

        return comp
