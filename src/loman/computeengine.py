"""Core computation engine for dependency-aware calculation graphs.

:class:`Computation` is the top of a stack of layers, each in its own module
and each calling only the layers beneath it — lowest first:

=========================  =====================================================
:mod:`loman.base`          the shared state all layers operate on
:mod:`loman.state`         node state transitions and the state index
:mod:`loman.query`         read-only accessors over the graph
:mod:`loman.attributes`    tags, styles and metadata
:mod:`loman.execution`     scheduling and running calculations
:mod:`loman.persistence`   copying, and file round-trips
:mod:`loman.mutation`      adding, removing, rewiring and populating nodes
=========================  =====================================================

What is left in this module is the object's construction, the class-definition
DSL that builds computations out of decorated classes, and the delegation to
planning and visualization.
"""

import functools
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

import decorator
import networkx as nx

from .compat import get_signature
from .consts import NodeTransformations, States, SystemTags
from .exception import NonExistentNodeException
from .mutation import MutationMixin, identity_function
from .nodekey import Name, Names, NodeKey, names_to_node_keys, to_nodekey
from .planning import ExecutionPlan, ValidationReport, create_execution_plan, validate_graph
from .util import AttributeView
from .values import C, ConstantValue, Error, NodeData, TimingData

# Re-exported under their historical names. These are not merely a convenience:
# instances of them are stored inside the DAG (``Error`` as a node value,
# ``_ParameterType`` inside every edge's PARAM attribute), and pickle records
# the module path it first saw. Dropping these names would make dill files
# written by earlier versions unreadable by ``Computation.read_dill``.
from .values import _ParameterItem as _ParameterItem
from .values import _ParameterType as _ParameterType
from .visualization import GraphView, NodeFormatter

LOG = logging.getLogger("loman.computeengine")

F = TypeVar("F", bound=Callable[..., Any])

# Re-exported for backwards compatibility: these types moved to loman.values
# and loman.mutation when Computation was split into mixins, but they have
# always been importable from loman.computeengine.
__all__ = [
    "Block",
    "C",
    "CalcNode",
    "Computation",
    "ConstantValue",
    "Error",
    "InputNode",
    "Node",
    "NodeData",
    "NodeKey",
    "NullObject",
    "TimingData",
    "block",
    "calc_node",
    "computation_factory",
    "identity_function",
    "input_node",
    "node",
    "populate_computation_from_class",
]


def _node(func: Callable[..., Any], *args: Any, **kws: Any) -> Any:  # pragma: no cover
    """Internal wrapper function for node decorator."""
    return func(*args, **kws)


def node(comp: "Computation", name: Name | None = None, *args: Any, **kw: Any) -> Callable[[F], F]:
    """Decorator to add a function as a node to a computation graph."""

    def inner(f: F) -> F:
        """Inner decorator that registers the function as a node."""
        if name is None:
            comp.add_node(f.__name__, f, *args, **kw)
        else:
            comp.add_node(name, f, *args, **kw)
        result: F = decorator.decorate(f, _node)
        return result

    return inner


class Node:
    """Base class for computation graph nodes."""

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool) -> None:
        """Add this node to the computation graph."""
        raise NotImplementedError()


@dataclass
class InputNode(Node):
    """A node representing input data in the computation graph."""

    args: tuple[Any, ...] = field(default_factory=tuple)
    kwds: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initialize an input node with arguments and keyword arguments."""
        self.args = args
        self.kwds = kwds

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool) -> None:
        """Add this input node to the computation graph."""
        comp.add_node(name, **self.kwds)


input_node = InputNode


@dataclass
class CalcNode(Node):
    """A node representing a calculation in the computation graph."""

    f: Callable[..., Any]
    kwds: dict[str, Any] = field(default_factory=dict)

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool) -> None:
        """Add this calculation node to the computation graph."""
        kwds = self.kwds.copy()
        ignore_self = ignore_self or kwds.get("ignore_self", False)
        f = self.f
        if ignore_self:
            signature = get_signature(self.f)
            if len(signature.kwd_params) > 0 and signature.kwd_params[0] == "self":
                f = f.__get__(obj, obj.__class__)  # type: ignore[attr-defined]
        if "ignore_self" in kwds:
            del kwds["ignore_self"]
        comp.add_node(name, f, **kwds)


@overload
def calc_node(f: F, **kwds: Any) -> F: ...


@overload
def calc_node(f: None = None, **kwds: Any) -> Callable[[F], F]: ...


def calc_node(f: F | None = None, **kwds: Any) -> F | Callable[[F], F]:
    """Decorator to mark a function as a calculation node."""

    def wrap(func: F) -> F:
        """Wrap function with node info attribute."""
        func._loman_node_info = CalcNode(func, kwds)
        return func

    if f is None:
        return wrap
    return wrap(f)


@dataclass
class Block(Node):
    """A node representing a computational block or subgraph."""

    block: "Callable[[], Computation] | Computation"
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwds: dict[str, Any] = field(default_factory=dict)

    def __init__(self, block: "Callable[[], Computation] | Computation", *args: Any, **kwds: Any) -> None:
        """Initialize a block node with a computation block and arguments."""
        self.block = block
        self.args = args
        self.kwds = kwds

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool) -> None:
        """Add this block node to the computation graph."""
        if isinstance(self.block, Computation):
            comp.add_block(name, self.block, *self.args, **self.kwds)
        elif callable(self.block):
            block0 = self.block()
            comp.add_block(name, block0, *self.args, **self.kwds)
        else:
            msg = f"Block {self.block} must be callable or Computation"
            raise TypeError(msg)


block = Block


def populate_computation_from_class(comp: "Computation", cls: type, obj: object, ignore_self: bool = True) -> None:
    """Populate a computation from class methods with node decorators."""
    for name, member in inspect.getmembers(cls):
        node_: Node | None = None
        if isinstance(member, Node):
            node_ = member
        elif hasattr(member, "_loman_node_info"):
            node_ = member._loman_node_info
        if node_ is not None:
            node_.add_to_comp(comp, name, obj, ignore_self)


def computation_factory(
    maybe_cls: type | None = None, *, ignore_self: bool = True
) -> Callable[..., "Computation"] | Callable[[type], Callable[..., "Computation"]]:
    """Factory function to create computations from class definitions."""

    def wrap(cls: type) -> Callable[..., "Computation"]:
        """Wrap class to create computation factory function."""

        @functools.wraps(cls, updated=())
        def create_computation(*args: Any, **kwargs: Any) -> "Computation":
            """Create a computation instance from the wrapped class."""
            obj = cls()
            comp = Computation(*args, **kwargs)
            comp._definition_object = obj  # type: ignore[attr-defined]
            populate_computation_from_class(comp, cls, obj, ignore_self)
            return comp

        return create_computation

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


class NullObject:
    """Debug helper object that raises exceptions for all attribute/item access."""

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for any attribute access."""
        print(f"__getattr__: {name}")
        msg = f"'NullObject' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Raise AttributeError for any attribute assignment."""
        print(f"__setattr__: {name}")
        msg = f"'NullObject' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Raise AttributeError for any attribute deletion."""
        print(f"__delattr__: {name}")
        msg = f"'NullObject' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Raise TypeError when called as a function."""
        print(f"__call__: {args}, {kwargs}")
        msg = "'NullObject' object is not callable"
        raise TypeError(msg)

    def __getitem__(self, key: Any) -> Any:
        """Raise KeyError for any item access."""
        print(f"__getitem__: {key}")
        msg = f"'NullObject' object has no item with key '{key}'"
        raise KeyError(msg)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Raise KeyError for any item assignment."""
        print(f"__setitem__: {key}")
        msg = f"'NullObject' object cannot have items set with key '{key}'"
        raise KeyError(msg)

    def __repr__(self) -> str:
        """Return string representation of NullObject."""
        print(f"__repr__: {object.__getattribute__(self, '__dict__')}")
        return "<NullObject>"


class Computation(MutationMixin):
    """A computation graph that manages dependencies and calculations.

    The Computation class provides a framework for building and executing
    computation graphs where nodes represent data or calculations, and edges
    represent dependencies between them.
    """

    def __init__(
        self,
        *,
        default_executor: Executor | None = None,
        executor_map: dict[str, Executor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new Computation.

        :param default_executor: An executor
        :type default_executor: concurrent.futures.Executor, default ThreadPoolExecutor(max_workers=1)
        """
        if default_executor is None:
            self.default_executor: Executor = ThreadPoolExecutor(1)
        else:
            self.default_executor = default_executor
        if executor_map is None:
            self.executor_map: dict[str, Executor] = {}
        else:
            self.executor_map = executor_map
        self.dag: nx.DiGraph = nx.DiGraph()
        self._metadata: dict[NodeKey, Any] = {}
        if metadata is not None:
            self._metadata[NodeKey.root()] = metadata

        self.v = self.get_attribute_view_for_path(NodeKey.root(), self._value_one, self.value)
        self.s = self.get_attribute_view_for_path(NodeKey.root(), self._state_one, self.state)
        self.i = self.get_attribute_view_for_path(NodeKey.root(), self._get_inputs_one_names, self.get_inputs)
        self.o = self.get_attribute_view_for_path(NodeKey.root(), self._get_outputs_one, self.get_outputs)
        self.t = self.get_attribute_view_for_path(NodeKey.root(), self._tag_one, self.tags)
        self.style = self.get_attribute_view_for_path(NodeKey.root(), self._style_one, self.styles)
        self.tim = self.get_attribute_view_for_path(NodeKey.root(), self._get_timing_one, self.get_timing)
        self.x = self.get_attribute_view_for_path(
            NodeKey.root(), self.compute_and_get_value, self.compute_and_get_value
        )
        self.src = self.get_attribute_view_for_path(NodeKey.root(), self.print_source, self.print_source)
        self._tag_map: defaultdict[str, set[NodeKey]] = defaultdict(set)
        self._state_map: dict[States, set[NodeKey]] = {state: set() for state in States}

    def get_attribute_view_for_path(
        self, nodekey: NodeKey, get_one_func: Callable[[Name], Any], get_many_func: Callable[[Name | Names], Any]
    ) -> AttributeView:
        """Create an attribute view for a specific node path."""

        def node_func() -> Iterable[str]:
            """Return list of child node names for this path."""
            return [str(n) for n in self.get_tree_list_children(nodekey)]

        def get_one_func_for_path(name: str) -> Any:
            """Get value for a single node at this path."""
            nk = to_nodekey(name)
            new_nk = nk.prepend(nodekey)
            if self.has_node(new_nk):
                return get_one_func(new_nk)
            elif self.tree_has_path(new_nk):
                return self.get_attribute_view_for_path(new_nk, get_one_func, get_many_func)
            else:
                msg = f"Path {new_nk} does not exist"
                raise KeyError(msg)  # pragma: no cover

        def get_many_func_for_path(name: Name | Names) -> Any:
            """Get values for one or more nodes at this path."""
            if isinstance(name, list):
                return [get_one_func_for_path(str(n)) for n in name]
            else:
                return get_one_func_for_path(str(name))

        return AttributeView(node_func, get_one_func_for_path, get_many_func_for_path)

    def validate(self) -> ValidationReport:
        """Inspect the entire graph for structural and readiness problems.

        Validation does not execute functions or mutate the computation.
        """
        return validate_graph(self.dag, self.executor_map)

    def plan(self, targets: Name | Names | None = None) -> ExecutionPlan:
        """Describe the work needed to compute one or more targets.

        Passing ``None`` plans the whole graph. Planning does not execute
        functions or mutate the computation.

        :param targets: Target node, list of target nodes, or ``None`` for all nodes.
        """
        target_node_keys = None if targets is None else names_to_node_keys(targets)
        if target_node_keys is not None:
            for node_key in target_node_keys:
                if not self.dag.has_node(node_key):
                    msg = f"Node {node_key} does not exist"
                    raise NonExistentNodeException(msg)
        return create_execution_plan(self.dag, self.executor_map, target_node_keys)

    def _repr_svg_(self) -> str | None:
        """Return SVG representation for Jupyter notebook display."""
        return GraphView(self).svg()

    def draw(
        self,
        root: NodeKey | None = None,
        *,
        node_transformations: dict[Name, str] | None = None,
        cmap: Any = None,
        colors: str = "state",
        shapes: str | None = None,
        graph_attr: dict[str, Any] | None = None,
        node_attr: dict[str, Any] | None = None,
        edge_attr: dict[str, Any] | None = None,
        show_expansion: bool = False,
        collapse_all: bool = True,
    ) -> GraphView:
        """Draw a computation's current state using the GraphViz utility.

        :param root: Optional PathType. Sub-block to draw
        :param cmap: Default: None
        :param colors: 'state' - colors indicate state. 'timing' - colors indicate execution time. Default: 'state'.
        :param shapes: None - ovals. 'type' - shapes indicate type. Default: None.
        :param graph_attr: Mapping of (attribute, value) pairs for the graph. For example
            ``graph_attr={'size': '"10,8"'}`` can control the size of the output graph
        :param node_attr: Mapping of (attribute, value) pairs set for all nodes.
        :param edge_attr: Mapping of (attribute, value) pairs set for all edges.
        :param collapse_all: Whether to collapse all blocks that aren't explicitly expanded.
        """
        node_formatter = NodeFormatter.create(cmap, colors, shapes)
        node_transformations_copy: dict[Name, str] = (
            node_transformations.copy() if node_transformations is not None else {}
        )
        if not show_expansion:
            for nodekey in self.nodes_by_tag(SystemTags.EXPANSION):
                node_transformations_copy[nodekey] = NodeTransformations.CONTRACT
        v = GraphView(
            self,
            root=root,
            node_formatter=node_formatter,
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
            node_transformations=node_transformations_copy,
            collapse_all=collapse_all,
        )
        return v

    def view(self, cmap: Any = None, colors: str = "state", shapes: str | None = None) -> None:
        """Create and display a visualization of the computation graph."""
        node_formatter = NodeFormatter.create(cmap, colors, shapes)
        v = GraphView(self, node_formatter=node_formatter)
        v.view()

    def print_errors(self) -> None:
        """Print tracebacks for every node with state "ERROR" in a Computation."""
        for n in self.nodes():
            if self.s[n] == States.ERROR:
                print(f"{n}")
                print("=" * len(str(n)))
                print()
                print(self.v[n].traceback)
                print()

    @classmethod
    def from_class(cls, definition_class: type, ignore_self: bool = True) -> "Computation":
        """Create a computation from a class with decorated methods."""
        comp = cls()
        obj = definition_class()
        populate_computation_from_class(comp, definition_class, obj, ignore_self=ignore_self)
        return comp
