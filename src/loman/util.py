"""Utility functions and classes for loman computation graphs."""

from __future__ import annotations

import itertools
import types
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

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
class RepeatedPipeline(Generic[K]):
    """Nodes created by :func:`add_repeated_pipeline`."""

    blocks: dict[K, NodeKey]
    result: NodeKey


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


def _validate_repeated_block_nodes(comp: Computation, block: Computation, blocks: Mapping[K, NodeKey]) -> None:
    """Ensure repeated blocks will not replace existing nodes."""
    generated_nodes = [block_path.join(node_name) for block_path in blocks.values() for node_name in block.nodes()]
    collisions = [node_key for node_key in generated_nodes if comp.has_node(node_key)]
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
    template.

    Args:
        comp: Computation to add the blocks to.
        block: Computation used as the block template.
        keys: Unique keys identifying the block instances.
        base_path: Parent path for all generated blocks.
        keep_values: Whether to copy current values from the template block.

    Returns:
        A mapping from each key to its generated block path.

    Raises:
        ValueError: If ``keys`` contains a duplicate.
    """
    blocks = _repeated_block_paths(keys, base_path)
    _validate_repeated_block_nodes(comp, block, blocks)
    for block_path in blocks.values():
        comp.add_block(block_path, block, keep_values=keep_values)
    return blocks


def add_fan_out(
    comp: Computation,
    source: Name,
    targets: Mapping[K, Name],
    *,
    transform: Callable[[Any, K], Any] | None = None,
) -> dict[K, NodeKey]:
    """Connect one source node to a keyed collection of target nodes.

    With no ``transform``, each target receives the source value unchanged. If
    supplied, ``transform(value, key)`` is evaluated independently for each
    target when the target is computed.

    Args:
        comp: Computation to add the fan-out nodes to.
        source: Source node to broadcast or transform.
        targets: Mapping from target keys to target node names.
        transform: Optional keyed transformation applied at computation time.

    Returns:
        A mapping from each key to its target node key.

    Raises:
        ValueError: If targets are repeated, replace calculation nodes, or a
            transformed target is also the source node.
    """
    from loman.computeengine import C
    from loman.consts import NodeAttributes
    from loman.nodekey import to_nodekey

    source_node_key = to_nodekey(source)
    target_node_keys = {key: to_nodekey(target) for key, target in targets.items()}
    if len(set(target_node_keys.values())) != len(target_node_keys):
        msg = "Fan-out targets must be unique"
        raise ValueError(msg)
    if transform is not None and source_node_key in target_node_keys.values():
        msg = "A transformed fan-out target cannot also be the source node"
        raise ValueError(msg)
    for target_node_key in target_node_keys.values():
        if comp.has_node(target_node_key) and (
            comp.dag.nodes[target_node_key].get(NodeAttributes.FUNC) is not None
            or next(comp.dag.predecessors(target_node_key), None) is not None
        ):
            msg = f"Fan-out target must be an input or placeholder node: {target_node_key!r}"
            raise ValueError(msg)
    _validate_acyclic_edges(comp, ((source_node_key, target) for target in target_node_keys.values()))

    for key, target_node_key in target_node_keys.items():
        if transform is None:
            comp.link(target_node_key, source_node_key)
        else:
            comp.add_node(
                target_node_key,
                _apply_keyed_transform,
                args=[source_node_key, C(key), C(transform)],
                inspect=False,
            )
    return target_node_keys


def add_fan_in(
    comp: Computation,
    result: Name,
    sources: Mapping[K, Name],
    *,
    combine: Callable[[Mapping[K, Any]], R] | None = None,
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
    """
    from loman.computeengine import C
    from loman.nodekey import to_nodekey

    result_node_key = to_nodekey(result)
    source_node_keys = [to_nodekey(source) for source in sources.values()]
    if len(set(source_node_keys)) != len(source_node_keys):
        msg = "Fan-in source nodes must be unique"
        raise ValueError(msg)
    if result_node_key in source_node_keys:
        msg = "A fan-in result cannot also be a source node"
        raise ValueError(msg)
    if comp.has_node(result_node_key):
        msg = f"Fan-in result node already exists: {result_node_key!r}"
        raise ValueError(msg)

    comp.add_node(
        result_node_key,
        _combine_keyed_values,
        args=[C(tuple(sources)), C(combine), *source_node_keys],
        inspect=False,
    )
    return result_node_key


def add_repeated_pipeline(
    comp: Computation,
    block: Computation,
    keys: Iterable[K],
    *,
    base_path: Name,
    source: Name,
    block_input: Name,
    block_output: Name,
    result: Name,
    transform: Callable[[Any, K], Any] | None = None,
    combine: Callable[[Mapping[K, Any]], R] | None = None,
    keep_values: bool = False,
) -> RepeatedPipeline[K]:
    """Create a keyed fan-out, repeated block, and fan-in pipeline.

    Args:
        comp: Computation to add the pipeline to.
        block: Computation used as the repeated block template.
        keys: Unique keys identifying block instances.
        base_path: Parent path for all generated blocks.
        source: Node fanned out to each block.
        block_input: Input node path relative to each block.
        block_output: Output node path relative to each block.
        result: Name of the fan-in result node.
        transform: Optional ``transform(value, key)`` for each block input.
        combine: Optional ``combine(mapping)`` for block outputs.
        keep_values: Whether to copy current values from the block template.

    Returns:
        The generated block paths and result node key.
    """
    from loman.consts import NodeAttributes
    from loman.nodekey import to_nodekey

    blocks = _repeated_block_paths(keys, base_path)
    _validate_repeated_block_nodes(comp, block, blocks)
    block_input_node_key = to_nodekey(block_input)
    block_output_node_key = to_nodekey(block_output)
    if not block.has_node(block_input_node_key):
        msg = f"Repeated pipeline block input does not exist: {block_input_node_key!r}"
        raise ValueError(msg)
    if block.dag.nodes[block_input_node_key].get(NodeAttributes.FUNC) is not None:
        msg = f"Repeated pipeline block input must be an input node: {block_input_node_key!r}"
        raise ValueError(msg)
    if not block.has_node(block_output_node_key):
        msg = f"Repeated pipeline block output does not exist: {block_output_node_key!r}"
        raise ValueError(msg)
    result_node_key = to_nodekey(result)
    if comp.has_node(result_node_key) or result_node_key in {
        block_path.join(node_name) for block_path in blocks.values() for node_name in block.nodes()
    }:
        msg = f"Repeated pipeline result node already exists: {result_node_key!r}"
        raise ValueError(msg)
    source_node_key = to_nodekey(source)
    if source_node_key == result_node_key:
        msg = "Repeated pipeline source cannot also be the result node"
        raise ValueError(msg)
    targets = {key: path.join(block_input_node_key) for key, path in blocks.items()}
    if transform is not None and source_node_key in targets.values():
        msg = "A transformed fan-out target cannot also be the source node"
        raise ValueError(msg)
    generated_edges = [
        (block_path.join(source_node), block_path.join(target_node))
        for block_path in blocks.values()
        for source_node, target_node in block.dag.edges()
    ]
    generated_edges.extend((source_node_key, target) for target in targets.values())
    generated_edges.extend((path.join(block_output_node_key), result_node_key) for path in blocks.values())
    _validate_acyclic_edges(comp, generated_edges)

    for block_path in blocks.values():
        comp.add_block(block_path, block, keep_values=keep_values)
    add_fan_out(comp, source, targets, transform=transform)
    sources = {key: path.join(block_output_node_key) for key, path in blocks.items()}
    result_node_key = add_fan_in(comp, result_node_key, sources, combine=combine)
    return RepeatedPipeline(blocks, result_node_key)


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
