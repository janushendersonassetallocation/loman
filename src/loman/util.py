"""Utility functions and classes for loman computation graphs."""

from __future__ import annotations

import itertools
import types
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from loman.computeengine import Computation
    from loman.nodekey import Name, NodeKey
else:
    Computation = Any
    Name = Any
    NodeKey = Any

T = TypeVar("T")
R = TypeVar("R")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class FanOut(Generic[K]):
    """Wire one source node to a relative input in every repeated block.

    ``source`` normally names a single outer node feeding every block. Passing a
    callable instead resolves a source node per key, as ``source(key)``, so each
    block can read from a different outer node.
    """

    source: Name | Callable[[K], Name]
    target: Name
    transform: Callable[[Any, K], Any] | None = None


@dataclass(frozen=True)
class FanIn(Generic[K]):
    """Collect one relative output from every repeated block."""

    source: Name
    result: Name
    combine: Callable[[Mapping[K, Any]], Any] | None = None


@dataclass(frozen=True)
class BuiltRepeatedBlocks(Generic[K]):
    """Node paths created by :meth:`RepeatedBlocks.add_to`."""

    blocks: dict[K, NodeKey]
    results: dict[Name, NodeKey]
    id_nodes: dict[K, NodeKey] = field(default_factory=dict)


@dataclass(frozen=True)
class RepeatedBlocks(Generic[K]):
    """Reusable definition for keyed copies of a computation block.

    ``fan_out`` entries connect outer computation nodes to relative inputs in
    every block. ``fan_in`` entries collect relative block outputs into outer
    result nodes.

    ``id_node``, if given, names a node created inside every block holding that
    block's own key, so block functions can depend on their key by name.

    ``keep_values`` defaults to ``False``, so only the structure of ``block`` is
    copied. This differs from :meth:`Computation.add_block`, which copies values
    by default: that call adds one specific block, which may be a sub-model that
    has already been populated, whereas this one stamps out many copies of a
    template. To give every copy the same value, broadcast it with a
    :class:`FanOut` that has no ``transform``, which keeps one outer node as the
    single place to set it.
    """

    block: Computation
    keys: Sequence[K]
    base_path: Name
    fan_out: Sequence[FanOut[K]] = ()
    fan_in: Sequence[FanIn[K]] = ()
    id_node: Name | None = None
    keep_values: bool = False

    def __post_init__(self) -> None:
        """Freeze collection inputs as tuples for reusable definitions."""
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "fan_out", tuple(self.fan_out))
        object.__setattr__(self, "fan_in", tuple(self.fan_in))

    def add_to(self, comp: Computation) -> BuiltRepeatedBlocks[K]:
        """Add this repeated-block definition to a computation."""
        return _add_repeated_blocks_definition(comp, self)


def _apply_keyed_transform(value: Any, key: Hashable, transform: Callable[[Any, Hashable], Any]) -> Any:
    """Apply a fan-out transform to a value and target key."""
    return transform(value, key)


def _combine_keyed_values(
    keys: tuple[Hashable, ...], combine: Callable[[Mapping[Hashable, Any]], Any] | None, *values: Any
) -> Any:
    """Build a keyed value mapping and optionally combine it."""
    keyed_values = dict(zip(keys, values, strict=True))
    if combine is None:
        return keyed_values
    return combine(keyed_values)


def _to_node_name(name: Name, description: str) -> NodeKey:
    """Convert a node name to a key, rejecting a callable given by mistake.

    ``Name`` admits any hashable, so a callable would otherwise silently become a
    node whose key is the function object itself. Only a fan-out source may be a
    callable, where it resolves a different source node for each key.
    """
    from loman.nodekey import to_nodekey

    if callable(name):
        msg = f"{description} must be a node name, not a callable: {name!r}"
        raise TypeError(msg)
    return to_nodekey(name)


def _resolve_fan_out_sources(source: Name | Callable[[K], Name], keys: Iterable[K]) -> dict[K, NodeKey]:
    """Resolve a fan-out source to one source node per key.

    A callable is applied to each key, so every target can read from a different
    node. Any other value names one node broadcast to every key.
    """
    if not callable(source):
        return dict.fromkeys(keys, _to_node_name(source, "Fan-out source"))
    resolve = cast("Callable[[K], Name]", source)
    return {key: _to_node_name(resolve(key), f"Fan-out source for key {key!r}") for key in keys}


def _repeated_block_paths(keys: Iterable[K], base_path: Name) -> dict[K, NodeKey]:
    """Build and validate the paths for repeated block keys."""
    from loman.nodekey import to_nodekey

    base_path_node_key = to_nodekey(base_path)
    blocks: dict[K, NodeKey] = {}
    for key in keys:
        if key in blocks:
            msg = f"Duplicate repeated block key: {key!r}"
            raise ValueError(msg)
        blocks[key] = base_path_node_key.join_parts(key)
    return blocks


def _validate_block_template(comp: Computation, block: Computation) -> None:
    """Ensure a block template is not the computation being added to."""
    if block is comp:
        msg = "Repeated block template must be a different computation"
        raise ValueError(msg)


def _is_placeholder(comp: Computation, node_key: NodeKey) -> bool:
    """Return whether a node exists only as an unfulfilled forward reference."""
    from loman.consts import NodeAttributes, States

    return comp.dag.nodes[node_key][NodeAttributes.STATE] == States.PLACEHOLDER


def _is_defined(comp: Computation, node_key: NodeKey) -> bool:
    """Return whether a node already has a definition or a value in a computation.

    Placeholder nodes do not count as defined: they record that another node
    refers to a name that has not been defined yet, so generated nodes are free
    to supply that definition.
    """
    return comp.has_node(node_key) and not _is_placeholder(comp, node_key)


def _validate_repeated_block_nodes(comp: Computation, block: Computation, blocks: Mapping[K, NodeKey]) -> None:
    """Ensure repeated blocks will not replace existing nodes."""
    generated_nodes = [block_path.join(node_name) for block_path in blocks.values() for node_name in block.nodes()]
    collisions = [node_key for node_key in generated_nodes if _is_defined(comp, node_key)]
    if collisions:
        msg = f"Repeated blocks would replace existing nodes: {collisions!r}"
        raise ValueError(msg)


def _validate_acyclic_edges(comp: Computation, edges: Iterable[tuple[NodeKey, NodeKey]]) -> None:
    """Ensure proposed dependency edges preserve the computation DAG."""
    import networkx as nx

    proposed_graph = nx.DiGraph(comp.dag)
    proposed_graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(proposed_graph):
        msg = "Generated computation utilities would create a cycle"
        raise ValueError(msg)


def add_repeated_blocks(
    comp: Computation,
    block: Computation,
    keys: Iterable[K],
    *,
    base_path: Name,
    keep_values: bool = False,
) -> dict[K, NodeKey]:
    """Add one copy of a computation block for each key.

    Each key becomes one path segment below ``base_path``. Block values are not
    copied by default, making the supplied block a reusable calculation
    template. This differs from :meth:`Computation.add_block`, which copies
    values by default; see :class:`RepeatedBlocks` for why the defaults differ.

    Args:
        comp: Computation to add the blocks to.
        block: Computation used as the block template.
        keys: Unique keys identifying the block instances.
        base_path: Parent path for all generated blocks.
        keep_values: Whether to copy current values from the template block.
            Prefer a broadcast :func:`add_fan_out` over ``True`` when every copy
            needs the same value.

    Returns:
        A mapping from each key to its generated block path.

    Raises:
        ValueError: If ``keys`` contains a duplicate, or ``block`` is the
            computation being added to.
    """
    _validate_block_template(comp, block)
    blocks = _repeated_block_paths(keys, base_path)
    _validate_repeated_block_nodes(comp, block, blocks)
    for block_path in blocks.values():
        comp.add_block(block_path, block, keep_values=keep_values)
    return blocks


def add_fan_out(
    comp: Computation,
    source: Name | Callable[[K], Name],
    targets: Mapping[K, Name],
    *,
    transform: Callable[[Any, K], Any] | None = None,
) -> dict[K, NodeKey]:
    """Connect one or more source nodes to a keyed collection of target nodes.

    With no ``transform``, each target receives its source value unchanged. If
    supplied, ``transform(value, key)`` is evaluated independently for each
    target when the target is computed.

    ``source`` normally names one node broadcast to every target. Passing a
    callable instead resolves a source node per key, as ``source(key)``, so each
    target can read from a different node.

    Args:
        comp: Computation to add the fan-out nodes to.
        source: Source node to broadcast, or a function of the key returning one.
        targets: Mapping from target keys to target node names.
        transform: Optional keyed transformation applied at computation time.

    Returns:
        A mapping from each key to its target node key.

    Raises:
        ValueError: If targets are repeated, replace calculation nodes, or a
            transformed target is also its own source.
        TypeError: If a node name is a callable, or ``source`` resolves one.
    """
    from loman.computeengine import C
    from loman.consts import NodeAttributes

    source_node_keys = _resolve_fan_out_sources(source, targets)
    target_node_keys = {key: _to_node_name(target, "Fan-out target") for key, target in targets.items()}
    if len(set(target_node_keys.values())) != len(target_node_keys):
        msg = "Fan-out targets must be unique"
        raise ValueError(msg)
    if transform is not None and any(source_node_keys[key] == target_node_keys[key] for key in target_node_keys):
        msg = "A transformed fan-out target cannot also be the source node"
        raise ValueError(msg)
    for target_node_key in target_node_keys.values():
        if comp.has_node(target_node_key) and (
            comp.dag.nodes[target_node_key].get(NodeAttributes.FUNC) is not None
            or next(comp.dag.predecessors(target_node_key), None) is not None
        ):
            msg = f"Fan-out target must be an input or placeholder node: {target_node_key!r}"
            raise ValueError(msg)
    _validate_acyclic_edges(comp, ((source_node_keys[key], target) for key, target in target_node_keys.items()))

    for key, target_node_key in target_node_keys.items():
        if transform is None:
            comp.link(target_node_key, source_node_keys[key])
        else:
            comp.add_node(
                target_node_key,
                _apply_keyed_transform,
                args=[source_node_keys[key], C(key), C(transform)],
                inspect=False,
            )
    return target_node_keys


def add_fan_in(
    comp: Computation,
    result: Name,
    sources: Mapping[K, Name],
    *,
    combine: Callable[[Mapping[K, Any]], Any] | None = None,
) -> NodeKey:
    """Collect keyed source nodes into one result node.

    Source values are assembled into an insertion-ordered mapping when the
    result is computed. With no ``combine`` function, that mapping is the
    result. Otherwise, ``combine(mapping)`` produces the result value.

    Args:
        comp: Computation to add the fan-in node to.
        result: Name of the generated result node.
        sources: Mapping from source keys to source node names.
        combine: Optional function that combines the keyed values.

    Returns:
        The generated result node key.

    Raises:
        ValueError: If source nodes are repeated, the result already exists, or
            the result is also a source.
        TypeError: If a node name is a callable.
    """
    from loman.computeengine import C

    result_node_key = _to_node_name(result, "Fan-in result")
    source_node_keys = [_to_node_name(source, "Fan-in source") for source in sources.values()]
    if len(set(source_node_keys)) != len(source_node_keys):
        msg = "Fan-in source nodes must be unique"
        raise ValueError(msg)
    if result_node_key in source_node_keys:
        msg = "A fan-in result cannot also be a source node"
        raise ValueError(msg)
    if _is_defined(comp, result_node_key):
        msg = f"Fan-in result node already exists: {result_node_key!r}"
        raise ValueError(msg)

    comp.add_node(
        result_node_key,
        _combine_keyed_values,
        args=[C(tuple(sources)), C(combine), *source_node_keys],
        inspect=False,
    )
    return result_node_key


def add_id_nodes(comp: Computation, blocks: Mapping[K, NodeKey], name: Name) -> dict[K, NodeKey]:
    """Give each block a node holding its own key.

    Block functions can then depend on their key by name, to look data up or to
    branch on it, without the key being wired in from outside. Unlike a fan-out,
    the generated nodes have no predecessors: each simply holds its key as a
    value.

    Args:
        comp: Computation holding the blocks.
        blocks: Mapping from each key to its block path.
        name: Relative node name to create inside every block.

    Returns:
        A mapping from each key to its generated identifier node key.

    Raises:
        ValueError: If a generated node would replace a calculation node.
        TypeError: If ``name`` is a callable.
    """
    from loman.consts import NodeAttributes

    id_node_key = _to_node_name(name, "Identifier node name")
    id_nodes = {key: block_path.join(id_node_key) for key, block_path in blocks.items()}
    for node_key in id_nodes.values():
        if comp.has_node(node_key) and (
            comp.dag.nodes[node_key].get(NodeAttributes.FUNC) is not None
            or next(comp.dag.predecessors(node_key), None) is not None
        ):
            msg = f"Identifier node must be an input or placeholder node: {node_key!r}"
            raise ValueError(msg)
    for key, node_key in id_nodes.items():
        comp.add_node(node_key, value=key)
    return id_nodes


def _add_repeated_blocks_definition(comp: Computation, definition: RepeatedBlocks[K]) -> BuiltRepeatedBlocks[K]:
    """Validate and add a :class:`RepeatedBlocks` definition atomically."""
    from loman.consts import NodeAttributes

    _validate_block_template(comp, definition.block)
    blocks = _repeated_block_paths(definition.keys, definition.base_path)
    _validate_repeated_block_nodes(comp, definition.block, blocks)
    generated_nodes = {
        block_path.join(node_name) for block_path in blocks.values() for node_name in definition.block.nodes()
    }

    id_node_key = None if definition.id_node is None else _to_node_name(definition.id_node, "Repeated block id_node")
    id_nodes: dict[K, NodeKey] = {}
    if id_node_key is not None:
        if definition.block.has_node(id_node_key):
            id_template_node = definition.block.dag.nodes[id_node_key]
            if (
                id_template_node.get(NodeAttributes.FUNC) is not None
                or next(definition.block.dag.predecessors(id_node_key), None) is not None
            ):
                msg = f"Repeated block id_node must be an input node: {id_node_key!r}"
                raise ValueError(msg)
        id_nodes = {key: path.join(id_node_key) for key, path in blocks.items()}
        generated_nodes.update(id_nodes.values())

    fan_outs: list[tuple[FanOut[K], dict[K, NodeKey], dict[K, NodeKey]]] = []
    target_nodes: set[NodeKey] = set()
    for fan_out in definition.fan_out:
        target_node_key = _to_node_name(fan_out.target, "Repeated block fan-out target")
        if not definition.block.has_node(target_node_key):
            msg = f"Repeated block fan-out target does not exist: {target_node_key!r}"
            raise ValueError(msg)
        if target_node_key == id_node_key:
            msg = f"Repeated block fan-out target cannot also be the id_node: {target_node_key!r}"
            raise ValueError(msg)
        target_node = definition.block.dag.nodes[target_node_key]
        if (
            target_node.get(NodeAttributes.FUNC) is not None
            or next(definition.block.dag.predecessors(target_node_key), None) is not None
        ):
            msg = f"Repeated block fan-out target must be an input node: {target_node_key!r}"
            raise ValueError(msg)
        targets = {key: path.join(target_node_key) for key, path in blocks.items()}
        duplicate_targets = target_nodes.intersection(targets.values())
        if duplicate_targets:
            duplicate_target_names = sorted(str(node_key) for node_key in duplicate_targets)
            msg = f"Repeated block fan-out targets must be unique: {duplicate_target_names!r}"
            raise ValueError(msg)
        target_nodes.update(targets.values())
        fan_outs.append((fan_out, _resolve_fan_out_sources(fan_out.source, blocks), targets))

    fan_ins: list[tuple[FanIn[K], NodeKey, dict[K, NodeKey]]] = []
    result_nodes: set[NodeKey] = set()
    for fan_in in definition.fan_in:
        source_node_key = _to_node_name(fan_in.source, "Repeated block fan-in source")
        if not definition.block.has_node(source_node_key):
            msg = f"Repeated block fan-in source does not exist: {source_node_key!r}"
            raise ValueError(msg)
        result_node_key = _to_node_name(fan_in.result, "Repeated block fan-in result")
        if result_node_key in result_nodes:
            msg = f"Repeated block fan-in result must be unique: {result_node_key!r}"
            raise ValueError(msg)
        if _is_defined(comp, result_node_key) or result_node_key in generated_nodes:
            msg = f"Repeated block fan-in result node already exists: {result_node_key!r}"
            raise ValueError(msg)
        result_nodes.add(result_node_key)
        sources = {key: path.join(source_node_key) for key, path in blocks.items()}
        fan_ins.append((fan_in, result_node_key, sources))

    generated_edges = [
        (block_path.join(source_node), block_path.join(target_node))
        for block_path in blocks.values()
        for source_node, target_node in definition.block.dag.edges()
    ]
    for _fan_out, source_node_keys, targets in fan_outs:
        generated_edges.extend((source_node_keys[key], target) for key, target in targets.items())
    for _fan_in, result_node_key, sources in fan_ins:
        generated_edges.extend((source, result_node_key) for source in sources.values())
    _validate_acyclic_edges(comp, generated_edges)

    for block_path in blocks.values():
        comp.add_block(block_path, definition.block, keep_values=definition.keep_values)

    if id_node_key is not None:
        add_id_nodes(comp, blocks, id_node_key)

    results: dict[Name, NodeKey] = {}
    for fan_in, result_node_key, sources in fan_ins:
        results[fan_in.result] = add_fan_in(comp, result_node_key, sources, combine=fan_in.combine)
    for fan_out, source_node_keys, targets in fan_outs:
        add_fan_out(comp, source_node_keys.__getitem__, targets, transform=fan_out.transform)
    return BuiltRepeatedBlocks(blocks, results, id_nodes)


@overload
def apply1(f: Callable[..., R], xs: list[T], *args: Any, **kwds: Any) -> list[R]: ...


@overload
def apply1(f: Callable[..., R], xs: T, *args: Any, **kwds: Any) -> R: ...


@overload
def apply1(f: Callable[..., R], xs: Generator[T, None, None], *args: Any, **kwds: Any) -> Generator[R, None, None]: ...


def apply1(
    f: Callable[..., R], xs: T | list[T] | Generator[T, None, None], *args: Any, **kwds: Any
) -> R | list[R] | Generator[R, None, None]:
    """Apply function f to xs, handling generators, lists, and single values."""
    if isinstance(xs, types.GeneratorType):
        return (f(x, *args, **kwds) for x in xs)
    if isinstance(xs, list):
        return [f(x, *args, **kwds) for x in xs]
    return f(xs, *args, **kwds)


def as_iterable(xs: T | Iterable[T]) -> Iterable[T]:
    """Convert input to iterable form if not already iterable."""
    if isinstance(xs, (types.GeneratorType, list, set)):
        return xs  # type: ignore[return-value]
    return (xs,)  # type: ignore[return-value]


def apply_n(f: Callable[..., Any], *xs: Any, **kwds: Any) -> None:
    """Apply function f to the cartesian product of iterables xs."""
    for p in itertools.product(*[as_iterable(x) for x in xs]):
        f(*p, **kwds)


class AttributeView:
    """Provides attribute-style access to dynamic collections."""

    def __init__(
        self,
        get_attribute_list: Callable[[], Iterable[str]],
        get_attribute: Callable[[str], Any],
        get_item: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize with functions to get attribute list and individual attributes.

        Args:
            get_attribute_list: Function that returns list of available attributes
            get_attribute: Function that takes an attribute name and returns its value
            get_item: Optional function for item access, defaults to get_attribute
        """
        self.get_attribute_list = get_attribute_list
        self.get_attribute = get_attribute
        self.get_item: Callable[[Any], Any] = get_item if get_item is not None else get_attribute

    def __dir__(self) -> list[str]:
        """Return list of available attributes."""
        return list(self.get_attribute_list())

    def __getattr__(self, attr: str) -> Any:
        """Get attribute by name, raising AttributeError if not found."""
        try:
            return self.get_attribute(attr)
        except KeyError as e:
            raise AttributeError(attr) from e

    def __getitem__(self, key: Any) -> Any:
        """Get item by key."""
        return self.get_item(key)

    def __getstate__(self) -> dict[str, Any]:
        """Prepare object for serialization."""
        return {
            "get_attribute_list": self.get_attribute_list,
            "get_attribute": self.get_attribute,
            "get_item": self.get_item,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore object from serialized state."""
        self.get_attribute_list = state["get_attribute_list"]
        self.get_attribute = state["get_attribute"]
        self.get_item = state["get_item"]
        if self.get_item is None:
            self.get_item = self.get_attribute

    @staticmethod
    def from_dict(d: dict[Any, Any], use_apply1: bool = True) -> AttributeView:
        """Create an AttributeView from a dictionary."""
        if use_apply1:

            def get_attribute(xs: Any) -> Any:
                """Get attribute value from dictionary with apply1 support."""
                return apply1(d.get, xs)
        else:
            get_attribute = d.get
        return AttributeView(d.keys, get_attribute)


pandas_types = (pd.Series, pd.DataFrame)


def value_eq(a: Any, b: Any) -> bool:
    """Compare two values for equality, handling pandas and numpy objects safely.

    - Uses .equals for pandas Series/DataFrame
    - For numpy arrays, returns a single boolean using np.array_equal (treats NaNs as equal)
    - Falls back to == and coerces to bool when possible
    """
    if a is b:
        return True

    # pandas objects: use robust equality
    if isinstance(a, pandas_types):
        return bool(a.equals(b))
    if isinstance(b, pandas_types):  # pragma: no cover
        return bool(b.equals(a))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except Exception:
            return False

    # Default comparison; ensure a single boolean
    try:
        result = a == b
        # If result is an array-like truth value, reduce safely
        if isinstance(result, (np.ndarray,)):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False
