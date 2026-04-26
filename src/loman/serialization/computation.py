"""Serialization for Computation graphs to/from JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

from loman.consts import EdgeAttributes, NodeAttributes, States, SystemTags
from loman.exception import SerializationError
from loman.nodekey import parse_nodekey

from .transformer import (
    DataFrameTransformer,
    DillFunctionTransformer,
    EnumTransformer,
    FunctionRefTransformer,
    NdArrayTransformer,
    NodeKeyTransformer,
    SeriesTransformer,
    Transformer,
    UntransformableTypeError,
)

if TYPE_CHECKING:
    pass

# Serialization format version — bump when the schema changes.
FORMAT_VERSION = 1


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

    # Pandas
    t.register(DataFrameTransformer())
    t.register(SeriesTransformer())

    # NodeKey (hierarchical node names)
    t.register(NodeKeyTransformer())

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
    - ``serialize``: bool — whether the node has the ``__serialize__`` tag
    - ``tags``: list of non-system tags

    Each **edge** object has:

    - ``src``: string key of the source node
    - ``dst``: string key of the destination node
    - ``param_type``: ``"arg"`` or ``"kwd"``
    - ``param``: positional index (int) for args, parameter name (str) for kwds

    Parameters
    ----------
    transformer:
        Custom :class:`~loman.serialization.transformer.Transformer` instance.
        If ``None``, a default transformer is built based on *use_dill_for_functions*.
    use_dill_for_functions:
        When ``True``, lambdas and closures are serialized as base64-encoded dill
        blobs rather than raising :class:`~loman.exception.SerializationError`.
        Has no effect when a custom *transformer* is supplied.  Defaults to ``False``.
    """

    # States whose nodes carry a meaningful value that should be preserved.
    _VALUE_STATES: ClassVar[set[States]] = {States.UPTODATE, States.PINNED, States.ERROR}

    def __init__(
        self,
        transformer: Transformer | None = None,
        *,
        use_dill_for_functions: bool = False,
    ) -> None:
        """Initialise with an optional custom transformer."""
        if transformer is None:
            transformer = (
                dill_computation_transformer() if use_dill_for_functions else default_computation_transformer()
            )
        self._t = transformer
        self._use_dill_for_functions = use_dill_for_functions

    def dump(self, comp: Any, fp: TextIO) -> None:
        """Serialize *comp* to *fp* (a text-mode file-like object)."""
        data = self._to_dict(comp)
        json.dump(data, fp)

    def dumps(self, comp: Any) -> str:
        """Serialize *comp* and return a JSON string."""
        return json.dumps(self._to_dict(comp))

    def _serialize_node_value(self, node_key: Any, state: States | None, node_data: dict[str, Any]) -> tuple[Any, bool]:
        """Return ``(encoded_value, has_value)`` for a node that should be serialized.

        Raises :class:`~loman.exception.SerializationError` if the value cannot
        be encoded.
        """
        from loman.computeengine import Error

        if state not in self._VALUE_STATES:
            return None, False

        raw_value = node_data.get(NodeAttributes.VALUE)
        if state == States.ERROR and isinstance(raw_value, Error):
            return (
                {
                    "__loman_error__": True,
                    "exception_type": type(raw_value.exception).__name__,
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
        except (UntransformableTypeError, ValueError, TypeError):
            # Non-importable callable (e.g. framework closure) — store null.
            return None

    def _serialize_node(self, node_key: Any, node_data: dict[str, Any]) -> dict[str, Any]:
        """Return the serialized dict for a single node."""
        state: States | None = node_data.get(NodeAttributes.STATE)
        tags: set[str] = node_data.get(NodeAttributes.TAG, set())
        serialize_flag: bool = SystemTags.SERIALIZE in tags

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

        user_tags = [t for t in tags if not t.startswith("__")]

        return {
            "key": str(node_key),
            "state": serialized_state.name if serialized_state is not None else None,
            "value": encoded_value,
            "has_value": has_value,
            "func": encoded_func,
            "serialize": serialize_flag,
            "tags": user_tags,
        }

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
        return {"version": FORMAT_VERSION, "nodes": nodes_out, "edges": edges_out}

    def load(self, fp: TextIO) -> Any:
        """Deserialize a Computation from *fp* (a text-mode file-like object)."""
        data = json.load(fp)
        return self._from_dict(data)

    def loads(self, s: str) -> Any:
        """Deserialize a Computation from a JSON string."""
        data = json.loads(s)
        return self._from_dict(data)

    def _from_dict(self, data: dict[str, Any]) -> Any:
        """Reconstruct a Computation from a deserialized dict."""
        from loman.computeengine import Computation, Error, _ParameterType

        comp = Computation()

        for node_info in data["nodes"]:
            raw_key = node_info["key"]
            node_key = parse_nodekey(raw_key)
            state_name = node_info["state"]
            state = States[state_name] if state_name is not None else None
            serialize_flag: bool = node_info.get("serialize", True)
            has_value: bool = node_info.get("has_value", False)
            user_tags: list[str] = node_info.get("tags", [])

            encoded_func = node_info.get("func")
            func = self._t.from_dict(encoded_func) if encoded_func is not None else None

            encoded_value = node_info.get("value")
            if has_value and encoded_value is not None:
                if isinstance(encoded_value, dict) and encoded_value.get("__loman_error__"):
                    value = Error(
                        exception=Exception(encoded_value["exception_str"]),
                        traceback=encoded_value["traceback"],
                    )
                else:
                    value = self._t.from_dict(encoded_value)
            else:
                value = None

            comp.dag.add_node(node_key)
            node_data = comp.dag.nodes[node_key]
            node_data[NodeAttributes.STATE] = state if state is not None else States.UNINITIALIZED
            node_data[NodeAttributes.VALUE] = value if has_value else None
            node_data[NodeAttributes.FUNC] = func
            node_data[NodeAttributes.ARGS] = {}
            node_data[NodeAttributes.KWDS] = {}
            node_data[NodeAttributes.TAG] = set()
            node_data[NodeAttributes.STYLE] = None
            node_data[NodeAttributes.GROUP] = None
            node_data[NodeAttributes.EXECUTOR] = None
            node_data[NodeAttributes.CONVERTER] = None

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

        comp._refresh_maps()

        return comp
