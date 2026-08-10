"""Utility functions and classes for loman computation graphs."""

from __future__ import annotations

import itertools
import types
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast, overload

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


_NO_VALUE = object()


@dataclass(frozen=True)
class PlannedNode:
    """One node a feature intends to create, described rather than created.

    Features return these instead of changing the computation, so the builder can
    check every node and edge of a definition before any of it is applied. A node
    with no ``func`` is an input node holding ``value``; otherwise ``func`` is
    called with ``args``, where each argument is either a :class:`NodeKey` to
    depend on or a :class:`ConstantValue` to pass through unchanged.
    """

    node_key: NodeKey
    func: Callable[..., Any] | None = None
    args: tuple[Any, ...] = ()
    value: Any = _NO_VALUE
    label: Name | None = None

    @classmethod
    def input_node(cls, node_key: NodeKey, value: Any, label: Name | None = None) -> PlannedNode:
        """Plan an input node holding a fixed value."""
        return cls(node_key, value=value, label=label)

    @classmethod
    def link(cls, node_key: NodeKey, source: NodeKey, label: Name | None = None) -> PlannedNode:
        """Plan a node that takes its value from another node unchanged."""
        from loman.computeengine import identity_function

        return cls(node_key, identity_function, (source,), label=label)

    @classmethod
    def calc(
        cls, node_key: NodeKey, func: Callable[..., Any], args: Sequence[Any], label: Name | None = None
    ) -> PlannedNode:
        """Plan a calculation node."""
        return cls(node_key, func, tuple(args), label=label)

    @property
    def predecessors(self) -> tuple[NodeKey, ...]:
        """Return the nodes this planned node would depend on."""
        from loman.computeengine import ConstantValue

        return tuple(arg for arg in self.args if not isinstance(arg, ConstantValue))

    def apply_to(self, comp: Computation) -> NodeKey:
        """Create this node in a computation."""
        if self.func is None:
            comp.add_node(self.node_key, value=self.value)
        else:
            comp.add_node(self.node_key, self.func, args=list(self.args), inspect=False)
        return self.node_key


@dataclass(frozen=True)
class BlockContext(Generic[K]):
    """What a feature is given when it plans its nodes.

    ``blocks`` maps each key to the path of its generated block. ``block`` is the
    template, so a feature can check that a relative name it was given really is a
    node, or an input node, of the block being repeated.
    """

    comp: Computation
    block: Computation
    base_path: NodeKey
    blocks: Mapping[K, NodeKey]
    definition_object: object = None
    ignore_self: bool = False
    planned: set[NodeKey] = field(default_factory=set)

    def require_block_node(self, name: Name, description: str) -> NodeKey:
        """Resolve a relative name that must be a node inside every block.

        The name may come from the template, or from a node an earlier feature
        already planned, so one feature can build on another's output. Features
        are planned in the order they are declared.
        """
        node_key = _to_node_name(name, description)
        if not self.block.has_node(node_key) and not self._is_planned_in_every_block(node_key):
            msg = f"{description} does not exist in the block: {node_key!r}"
            raise ValueError(msg)
        return node_key

    def _is_planned_in_every_block(self, node_key: NodeKey) -> bool:
        """Return whether an earlier feature planned this node in every block."""
        return all(block_path.join(node_key) in self.planned for block_path in self.blocks.values())

    def require_block_input(self, name: Name, description: str, *, create: bool = False) -> NodeKey:
        """Resolve a relative name that must be an input node inside every block.

        A node the template declares must be an input: replacing a calculation
        would silently discard it. A node an earlier feature planned is accepted
        as-is, since the template has nothing to say about it. A name the template
        never mentions is rejected, because it is usually a typo that would
        otherwise add a dead node to every block — pass ``create=True`` to allow
        it deliberately.
        """
        from loman.consts import NodeAttributes

        node_key = _to_node_name(name, description)
        if not self.block.has_node(node_key):
            if self._is_planned_in_every_block(node_key) or create:
                return node_key
            msg = (
                f"{description} does not exist in the block: {node_key!r}. "
                "Pass create=True to add it to every block anyway."
            )
            raise ValueError(msg)
        node = self.block.dag.nodes[node_key]
        if node.get(NodeAttributes.FUNC) is not None or next(self.block.dag.predecessors(node_key), None) is not None:
            msg = f"{description} must be an input node: {node_key!r}"
            raise ValueError(msg)
        return node_key

    def bind(self, func: Any) -> Any:
        """Bind a callback to the class a computation factory was defined from.

        Returns anything that is not callable unchanged, so a plain node name
        passes through.
        """
        from loman.computeengine import _bind_self

        return _bind_self(func, self.definition_object, self.ignore_self)


class BlockFeature(Protocol):
    """One wiring pattern applied to every copy of a repeated block.

    A feature never changes the computation itself. It describes the nodes it
    wants, and the builder validates every feature's plan together before
    applying any of it, so a definition that fails leaves the graph untouched.
    Implement this protocol to add a wiring pattern of your own.
    """

    def plan(self, ctx: BlockContext[Any]) -> Iterable[PlannedNode]:
        """Describe the nodes to create, without changing anything."""
        ...


@dataclass(frozen=True)
class FanOut(Generic[K]):
    """Wire one source node to a relative input in every repeated block.

    ``source`` normally names a single outer node feeding every block. Passing a
    callable instead resolves a source node per key, as ``source(key)``, so each
    block can read from a different outer node. With a ``transform``, each target
    is calculated as ``transform(value, key)``.

    ``target`` is a name inside each block. It must be something the template
    declares or refers to, so a typo does not quietly add a dead node to every
    block; set ``create=True`` to feed a name the template never mentions.
    """

    source: Name | Callable[[K], Name]
    target: Name
    transform: Callable[[Any, K], Any] | None = None
    create: bool = False

    def plan(self, ctx: BlockContext[K]) -> Iterable[PlannedNode]:
        """Plan one target node per key, linked or transformed from its source."""
        from loman.computeengine import C

        target = ctx.require_block_input(self.target, "Fan-out target", create=self.create)
        sources = _resolve_fan_out_sources(ctx.bind(self.source), ctx.blocks)
        transform = ctx.bind(self.transform)
        for key, block_path in ctx.blocks.items():
            node_key = block_path.join(target)
            if transform is None:
                yield PlannedNode.link(node_key, sources[key])
            else:
                yield PlannedNode.calc(node_key, transform, (sources[key], C(key)))


@dataclass(frozen=True)
class Positional:
    """Adapt an aggregator that takes positional arguments to a keyed ``combine``.

    ``combine`` receives an ordered mapping so keys stay attached to values, which
    is usually what you want. Where an existing function takes the values
    positionally, wrap it rather than repeating ``lambda m: fn(*m.values())``::

        FanIn("value", "total", combine=Positional(df_hconcat))

    Keys are discarded, so prefer a keyed aggregator where one exists — for
    dataframes, ``lambda m: pd.concat(m, axis=1)`` keeps the keys as column
    labels. Like any callable that is not an importable module-level function,
    this needs ``use_dill_for_functions=True`` to serialize.
    """

    func: Callable[..., Any]

    def __call__(self, values: Mapping[Any, Any]) -> Any:
        """Call the wrapped function with the mapping's values, in order."""
        return self.func(*values.values())


@dataclass(frozen=True)
class FanIn(Generic[K]):
    """Collect one relative output from every repeated block into a result node.

    ``source`` is a name inside each block. ``result`` is **not** relative to the
    definition's ``base_path``: it names a node in the outer computation, so the
    aggregate can live wherever it belongs rather than being forced under the
    blocks. ``BuiltRepeatedBlocks.named`` reports the key that was created.
    """

    source: Name
    result: Name
    combine: Callable[[Mapping[K, Any]], Any] | None = None

    def plan(self, ctx: BlockContext[K]) -> Iterable[PlannedNode]:
        """Plan one result node gathering the same relative node from every block."""
        from loman.computeengine import C

        source = ctx.require_block_node(self.source, "Fan-in source")
        result = _to_node_name(self.result, "Fan-in result")
        sources = [block_path.join(source) for block_path in ctx.blocks.values()]
        yield PlannedNode.calc(
            result,
            _combine_keyed_values,
            (C(tuple(ctx.blocks)), C(ctx.bind(self.combine)), *sources),
            label=self.result,
        )


@dataclass(frozen=True)
class IdNode:
    """Give every block a node holding its own key.

    Block functions can then depend on their key by name, to look data up or to
    branch on it, without the key being wired in from outside.
    """

    name: Name

    def plan(self, ctx: BlockContext[K]) -> Iterable[PlannedNode]:
        """Plan one value node per key, holding that key."""
        name = _to_node_name(self.name, "Identifier node")
        if ctx.block.has_node(name):
            ctx.require_block_input(name, "Identifier node")
        for key, block_path in ctx.blocks.items():
            yield PlannedNode.input_node(block_path.join(name), key)


@dataclass(frozen=True)
class InputValue:
    """Give every block the same constant value for one relative input.

    Unlike :class:`FanIn`'s ``result``, the shared node this creates **is**
    relative to the definition's ``base_path``, landing at ``<base_path>/<name>``
    so that two definitions with different base paths do not collide.
    """

    name: Name
    value: Any

    def plan(self, ctx: BlockContext[K]) -> Iterable[PlannedNode]:
        """Plan one shared outer node, linked into every block."""
        name = ctx.require_block_input(self.name, "Input value")
        shared = ctx.base_path.join(name)
        yield PlannedNode.input_node(shared, self.value, label=self.name)
        for block_path in ctx.blocks.values():
            yield PlannedNode.link(block_path.join(name), shared)


@dataclass(frozen=True)
class BuiltRepeatedBlocks(Generic[K]):
    """Node paths created by :meth:`RepeatedBlocks.add_to`.

    ``named`` collects the nodes that features chose to label, so a fan-in result
    can be looked up by the name it was declared with.
    """

    blocks: dict[K, NodeKey]
    nodes: tuple[NodeKey, ...]
    named: dict[Name, NodeKey]


@dataclass(frozen=True)
class RepeatedBlocks(Generic[K]):
    """Reusable definition for keyed copies of a computation block.

    ``features`` describe how data flows in and out of every copy: see
    :class:`FanOut`, :class:`FanIn`, :class:`IdNode` and :class:`InputValue`, or
    write your own against :class:`BlockFeature`. Features are applied in the
    order given, and every feature's plan is validated before any of it is
    applied.

    ``keep_values`` defaults to ``False``, so only the structure of ``block`` is
    copied. This differs from :meth:`Computation.add_block`, which copies values
    by default: that call adds one specific block, which may be a sub-model that
    has already been populated, whereas this one stamps out many copies of a
    template. To give every copy the same value, use an :class:`InputValue`, or a
    :class:`FanOut` with no ``transform`` to broadcast an existing node.
    """

    block: Computation
    keys: Sequence[K]
    base_path: Name
    features: Sequence[BlockFeature] = ()
    keep_values: bool = False

    def __post_init__(self) -> None:
        """Freeze collection inputs as tuples for reusable definitions."""
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "features", tuple(self.features))

    def add_to(
        self, comp: Computation, definition_object: object = None, ignore_self: bool = False
    ) -> BuiltRepeatedBlocks[K]:
        """Add this repeated-block definition to a computation."""
        return _add_repeated_blocks_definition(comp, self, definition_object, ignore_self)


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
            comp.add_node(target_node_key, transform, args=[source_node_keys[key], C(key)], inspect=False)
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


def _validate_planned_nodes(comp: Computation, planned: Sequence[PlannedNode], block_nodes: set[NodeKey]) -> None:
    """Ensure a set of planned nodes can all be created without conflict."""
    seen: set[NodeKey] = set()
    for planned_node in planned:
        node_key = planned_node.node_key
        if node_key in seen:
            msg = f"Repeated block features would write the same node twice: {node_key!r}"
            raise ValueError(msg)
        seen.add(node_key)
        if node_key not in block_nodes and _is_defined(comp, node_key):
            msg = f"Repeated block feature node already exists: {node_key!r}"
            raise ValueError(msg)


def _add_repeated_blocks_definition(
    comp: Computation,
    definition: RepeatedBlocks[K],
    definition_object: object = None,
    ignore_self: bool = False,
) -> BuiltRepeatedBlocks[K]:
    """Validate and add a :class:`RepeatedBlocks` definition atomically."""
    _validate_block_template(comp, definition.block)
    base_path = _to_node_name(definition.base_path, "Repeated block base_path")
    blocks = _repeated_block_paths(definition.keys, base_path)
    _validate_repeated_block_nodes(comp, definition.block, blocks)
    block_nodes = {
        block_path.join(node_name) for block_path in blocks.values() for node_name in definition.block.nodes()
    }

    ctx: BlockContext[K] = BlockContext(comp, definition.block, base_path, blocks, definition_object, ignore_self)
    planned: list[PlannedNode] = []
    for feature in definition.features:
        feature_nodes = list(feature.plan(ctx))
        planned.extend(feature_nodes)
        ctx.planned.update(planned_node.node_key for planned_node in feature_nodes)
    _validate_planned_nodes(comp, planned, block_nodes)

    generated_edges = [
        (block_path.join(source_node), block_path.join(target_node))
        for block_path in blocks.values()
        for source_node, target_node in definition.block.dag.edges()
    ]
    for planned_node in planned:
        generated_edges.extend((predecessor, planned_node.node_key) for predecessor in planned_node.predecessors)
    _validate_acyclic_edges(comp, generated_edges)

    for block_path in blocks.values():
        comp.add_block(block_path, definition.block, keep_values=definition.keep_values)

    named: dict[Name, NodeKey] = {}
    for planned_node in planned:
        planned_node.apply_to(comp)
        if planned_node.label is not None:
            named[planned_node.label] = planned_node.node_key
    return BuiltRepeatedBlocks(blocks, tuple(planned_node.node_key for planned_node in planned), named)


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
