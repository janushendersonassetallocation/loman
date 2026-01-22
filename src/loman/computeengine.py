"""Core computation engine for dependency-aware calculation graphs."""

import inspect
import logging
import traceback
import types
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Type, Union  # noqa: UP035

import decorator
import dill
import networkx as nx
import pandas as pd

from .compat import get_signature
from .consts import EdgeAttributes, NodeAttributes, NodeTransformations, States, SystemTags
from .exception import (
    CannotInsertToPlaceholderNodeException,
    ComputationError,
    LoopDetectedException,
    MapException,
    NodeAlreadyExistsException,
    NonExistentNodeException,
)
from .graph_utils import topological_sort
from .nodekey import Name, Names, NodeKey, names_to_node_keys, node_keys_to_names, to_nodekey
from .util import AttributeView, apply1, apply_n, as_iterable, value_eq
from .visualization import GraphView, NodeFormatter

LOG = logging.getLogger("loman.computeengine")


@dataclass
class Error:
    """Container for error information during computation."""

    exception: Exception
    traceback: inspect.Traceback


@dataclass
class NodeData:
    """Data associated with a computation node."""

    state: States
    value: object


@dataclass
class TimingData:
    """Timing information for computation execution."""

    start: datetime
    end: datetime
    duration: float


class _ParameterType(Enum):
    ARG = 1
    KWD = 2


@dataclass
class _ParameterItem:
    type: object
    name: int | str
    value: object


def _node(func, *args, **kws):  # pragma: no cover
    return func(*args, **kws)


def node(comp, name=None, *args, **kw):
    """Decorator to add a function as a node to a computation graph."""

    def inner(f):
        if name is None:
            comp.add_node(f.__name__, f, *args, **kw)
        else:
            comp.add_node(name, f, *args, **kw)
        return decorator.decorate(f, _node)

    return inner


@dataclass()
class ConstantValue:
    """Container for constant values in computations."""

    value: object


C = ConstantValue


class Node:
    """Base class for computation graph nodes."""

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool):
        """Add this node to the computation graph."""
        raise NotImplementedError()


@dataclass
class InputNode(Node):
    """A node representing input data in the computation graph."""

    args: tuple[Any, ...] = field(default_factory=tuple)
    kwds: dict = field(default_factory=dict)

    def __init__(self, *args, **kwds):
        """Initialize an input node with arguments and keyword arguments."""
        self.args = args
        self.kwds = kwds

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool):
        """Add this input node to the computation graph."""
        comp.add_node(name, **self.kwds)


input_node = InputNode


@dataclass
class CalcNode(Node):
    """A node representing a calculation in the computation graph."""

    f: Callable
    kwds: dict = field(default_factory=dict)

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool):
        """Add this calculation node to the computation graph."""
        kwds = self.kwds.copy()
        ignore_self = ignore_self or kwds.get("ignore_self", False)
        f = self.f
        if ignore_self:
            signature = get_signature(self.f)
            if len(signature.kwd_params) > 0 and signature.kwd_params[0] == "self":
                f = f.__get__(obj, obj.__class__)
        if "ignore_self" in kwds:
            del kwds["ignore_self"]
        comp.add_node(name, f, **kwds)


def calc_node(f=None, **kwds):
    """Decorator to mark a function as a calculation node."""

    def wrap(func):
        func._loman_node_info = CalcNode(func, kwds)
        return func

    if f is None:
        return wrap
    return wrap(f)


@dataclass
class Block(Node):
    """A node representing a computational block or subgraph."""

    block: Union[Callable, "Computation"]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwds: dict = field(default_factory=dict)

    def __init__(self, block, *args, **kwds):
        """Initialize a block node with a computation block and arguments."""
        self.block = block
        self.args = args
        self.kwds = kwds

    def add_to_comp(self, comp: "Computation", name: str, obj: object, ignore_self: bool):
        """Add this block node to the computation graph."""
        if isinstance(self.block, Computation):
            comp.add_block(name, self.block, *self.args, **self.kwds)
        elif callable(self.block):
            block0 = self.block()
            comp.add_block(name, block0, *self.args, **self.kwds)
        else:
            raise TypeError(f"Block {self.block} must be callable or Computation")


block = Block


def populate_computation_from_class(comp, cls, obj, ignore_self=True):
    """Populate a computation from class methods with node decorators."""
    for name, member in inspect.getmembers(cls):
        node_ = None
        if isinstance(member, Node):
            node_ = member
        elif hasattr(member, "_loman_node_info"):
            node_ = getattr(member, "_loman_node_info")
        if node_ is not None:
            node_.add_to_comp(comp, name, obj, ignore_self)


def computation_factory(maybe_cls=None, *, ignore_self=True) -> Type["Computation"]:  # noqa: UP006
    """Factory function to create computations from class definitions."""

    def wrap(cls):
        def create_computation(*args, **kwargs):
            obj = cls()
            comp = Computation(*args, **kwargs)
            comp._definition_object = obj
            populate_computation_from_class(comp, cls, obj, ignore_self)
            return comp

        return create_computation

    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


def _eval_node(name, f, args, kwds, raise_exceptions):
    """To make multiprocessing work, this function must be standalone so that pickle works."""
    exc, tb = None, None
    start_dt = datetime.now(UTC)
    try:
        logging.debug("Running " + str(name))
        value = f(*args, **kwds)
        logging.debug("Completed " + str(name))
    except Exception as e:
        value = None
        exc = e
        tb = traceback.format_exc()
        if raise_exceptions:
            raise
    end_dt = datetime.now(UTC)
    return value, exc, tb, start_dt, end_dt


_MISSING_VALUE_SENTINEL = object()


class NullObject:
    """Debug helper object that raises exceptions for all attribute/item access."""

    def __getattr__(self, name):
        """Raise AttributeError for any attribute access."""
        print(f"__getattr__: {name}")
        raise AttributeError(f"'NullObject' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Raise AttributeError for any attribute assignment."""
        print(f"__setattr__: {name}")
        raise AttributeError(f"'NullObject' object has no attribute '{name}'")

    def __delattr__(self, name):
        """Raise AttributeError for any attribute deletion."""
        print(f"__delattr__: {name}")
        raise AttributeError(f"'NullObject' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        """Raise TypeError when called as a function."""
        print(f"__call__: {args}, {kwargs}")
        raise TypeError("'NullObject' object is not callable")

    def __getitem__(self, key):
        """Raise KeyError for any item access."""
        print(f"__getitem__: {key}")
        raise KeyError(f"'NullObject' object has no item with key '{key}'")

    def __setitem__(self, key, value):
        """Raise KeyError for any item assignment."""
        print(f"__setitem__: {key}")
        raise KeyError(f"'NullObject' object cannot have items set with key '{key}'")

    def __repr__(self):
        """Return string representation of NullObject."""
        print(f"__repr__: {self.__dict__}")
        return "<NullObject>"


def identity_function(x):
    """Return the input value unchanged."""
    return x


class Computation:
    """A computation graph that manages dependencies and calculations.

    The Computation class provides a framework for building and executing
    computation graphs where nodes represent data or calculations, and edges
    represent dependencies between them.
    """

    def __init__(self, *, default_executor=None, executor_map=None, metadata=None):
        """Initialize a new Computation.

        :param default_executor: An executor
        :type default_executor: concurrent.futures.Executor, default ThreadPoolExecutor(max_workers=1)
        """
        if default_executor is None:
            self.default_executor = ThreadPoolExecutor(1)
        else:
            self.default_executor = default_executor
        if executor_map is None:
            self.executor_map = {}
        else:
            self.executor_map = executor_map
        self.dag = nx.DiGraph()
        self._metadata = {}
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
        self._tag_map = defaultdict(set)
        self._state_map = {state: set() for state in States}

    def get_attribute_view_for_path(self, nodekey: NodeKey, get_one_func: callable, get_many_func: callable):
        """Create an attribute view for a specific node path."""

        def node_func():
            return self.get_tree_list_children(nodekey)

        def get_one_func_for_path(name: Name):
            nk = to_nodekey(name)
            new_nk = nk.prepend(nodekey)
            if self.has_node(new_nk):
                return get_one_func(new_nk)
            elif self.tree_has_path(new_nk):
                return self.get_attribute_view_for_path(new_nk, get_one_func, get_many_func)
            else:
                raise KeyError(f"Path {new_nk} does not exist")  # pragma: no cover

        def get_many_func_for_path(name: Name | Names):
            if isinstance(name, list):
                return [get_one_func_for_path(n) for n in name]
            else:
                return get_one_func_for_path(name)

        return AttributeView(node_func, get_one_func_for_path, get_many_func_for_path)

    def _get_names_for_state(self, state: States):
        return set(node_keys_to_names(self._state_map[state]))

    def _get_tags_for_state(self, tag: str):
        return set(node_keys_to_names(self._tag_map[tag]))

    def _process_function_args(self, node_key, node, args):
        """Process positional arguments for a function node."""
        args_count = 0
        if args:
            args_count = len(args)
            for i, arg in enumerate(args):
                if isinstance(arg, ConstantValue):
                    node[NodeAttributes.ARGS][i] = arg.value
                else:
                    input_vertex_name = arg
                    input_vertex_node_key = to_nodekey(input_vertex_name)
                    if not self.dag.has_node(input_vertex_node_key):
                        self.dag.add_node(input_vertex_node_key, **{NodeAttributes.STATE: States.PLACEHOLDER})
                        self._state_map[States.PLACEHOLDER].add(input_vertex_node_key)
                    self.dag.add_edge(
                        input_vertex_node_key, node_key, **{EdgeAttributes.PARAM: (_ParameterType.ARG, i)}
                    )
        return args_count

    def _build_param_map(self, func, node_key, args_count, kwds, inspect):
        """Build parameter map for function node."""
        param_map = {}
        default_names = []

        if inspect:
            signature = get_signature(func)
            if not signature.has_var_args:
                for param_name in signature.kwd_params[args_count:]:
                    if kwds is not None and param_name in kwds:
                        param_source = kwds[param_name]
                    else:
                        param_source = node_key.parent.join_parts(param_name)
                    param_map[param_name] = param_source
            if signature.has_var_kwds and kwds is not None:
                for param_name, param_source in kwds.items():
                    param_map[param_name] = param_source
            default_names = signature.default_params
        else:
            if kwds is not None:
                for param_name, param_source in kwds.items():
                    param_map[param_name] = param_source

        return param_map, default_names

    def _process_function_kwds(self, node_key, node, param_map, default_names):
        """Process keyword arguments for a function node."""
        for param_name, param_source in param_map.items():
            if isinstance(param_source, ConstantValue):
                node[NodeAttributes.KWDS][param_name] = param_source.value
            else:
                in_node_name = param_source
                in_node_key = to_nodekey(in_node_name)
                if not self.dag.has_node(in_node_key):
                    if param_name in default_names:
                        continue
                    else:
                        self.dag.add_node(in_node_key, **{NodeAttributes.STATE: States.PLACEHOLDER})
                        self._state_map[States.PLACEHOLDER].add(in_node_key)
                self.dag.add_edge(in_node_key, node_key, **{EdgeAttributes.PARAM: (_ParameterType.KWD, param_name)})

    def add_node(
        self,
        name: Name,
        func=None,
        *,
        args=None,
        kwds=None,
        value=_MISSING_VALUE_SENTINEL,
        converter=None,
        serialize=True,
        inspect=True,
        group=None,
        tags=None,
        style=None,
        executor=None,
        metadata=None,
    ):
        """Adds or updates a node in a computation.

        :param name: Name of the node to add. This may be any hashable object.
        :param func: Function to use to calculate the node if the node is a calculation node. By default, the input
            nodes to the function will be implied from the names of the function parameters. For example, a
            parameter called ``a`` would be taken from the node called ``a``. This can be modified with the
            ``kwds`` parameter.
        :type func: Function, default None
        :param args: Specifies a list of nodes that will be used to populate arguments of the function positionally
            for a calculation node. e.g. If args is ``['a', 'b', 'c']`` then the function would be called with
            three parameters, taken from the nodes 'a', 'b' and 'c' respectively.
        :type args: List, default None
        :param kwds: Specifies a mapping from parameter name to the node that should be used to populate that
            parameter when calling the function for a calculation node. e.g. If args is ``{'x': 'a', 'y': 'b'}``
            then the function would be called with parameters named 'x' and 'y', and their values would be taken
            from nodes 'a' and 'b' respectively. Each entry in the dictionary can be read as "take parameter
            [key] from node [value]".
        :type kwds: Dictionary, default None
        :param value: If given, the value is inserted into the node, and the node state set to UPTODATE.
        :type value: default None
        :param serialize: Whether the node should be serialized. Some objects cannot be serialized, in which
            case, set serialize to False
        :type serialize: boolean, default True
        :param inspect: Whether to use introspection to determine the arguments of the function, which can be
            slow. If this is not set, kwds and args must be set for the function to obtain parameters.
        :type inspect: boolean, default True
        :param group: Subgraph to render node in
        :type group: default None
        :param tags: Set of tags to apply to node
        :type tags: Iterable
        :param styles: Style to apply to node
        :type styles: String, default None
        :param executor: Name of executor to run node on
        :type executor: string
        """
        node_key = to_nodekey(name)
        LOG.debug(f"Adding node {node_key}")
        has_value = value is not _MISSING_VALUE_SENTINEL
        if value is _MISSING_VALUE_SENTINEL:
            value = None
        if tags is None:
            tags = []

        self.dag.add_node(node_key)
        pred_edges = [(p, node_key) for p in self.dag.predecessors(node_key)]
        self.dag.remove_edges_from(pred_edges)
        node = self.dag.nodes[node_key]

        if metadata is None:
            if node_key in self._metadata:
                del self._metadata[node_key]
        else:
            self._metadata[node_key] = metadata

        self._set_state_and_literal_value(node_key, States.UNINITIALIZED, None, require_old_state=False)

        node[NodeAttributes.TAG] = set()
        node[NodeAttributes.STYLE] = style
        node[NodeAttributes.GROUP] = group
        node[NodeAttributes.ARGS] = {}
        node[NodeAttributes.KWDS] = {}
        node[NodeAttributes.FUNC] = None
        node[NodeAttributes.EXECUTOR] = executor
        node[NodeAttributes.CONVERTER] = converter

        if func:
            node[NodeAttributes.FUNC] = func
            args_count = self._process_function_args(node_key, node, args)
            param_map, default_names = self._build_param_map(func, node_key, args_count, kwds, inspect)
            self._process_function_kwds(node_key, node, param_map, default_names)
            self._set_descendents(node_key, States.STALE)

        if has_value:
            self._set_uptodate(node_key, value)
        if node[NodeAttributes.STATE] == States.UNINITIALIZED:
            self._try_set_computable(node_key)
        self.set_tag(node_key, tags)
        if serialize:
            self.set_tag(node_key, SystemTags.SERIALIZE)

    def _refresh_maps(self):
        self._tag_map.clear()
        for state in States:
            self._state_map[state].clear()
        for node_key in self._node_keys():
            state = self.dag.nodes[node_key][NodeAttributes.STATE]
            self._state_map[state].add(node_key)
            tags = self.dag.nodes[node_key].get(NodeAttributes.TAG, set())
            for tag in tags:
                self._tag_map[tag].add(node_key)

    def _set_tag_one(self, name: Name, tag):
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.TAG].add(tag)
        self._tag_map[tag].add(node_key)

    def set_tag(self, name: Names, tag):
        """Set tags on a node or nodes. Ignored if tags are already set.

        :param name: Node or nodes to set tag for
        :param tag: Tag to set
        """
        apply_n(self._set_tag_one, name, tag)

    def _clear_tag_one(self, name: Name, tag):
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.TAG].discard(tag)
        self._tag_map[tag].discard(node_key)

    def clear_tag(self, name: Names, tag):
        """Clear tag on a node or nodes. Ignored if tags are not set.

        :param name: Node or nodes to clear tags for
        :param tag: Tag to clear
        """
        apply_n(self._clear_tag_one, name, tag)

    def _set_style_one(self, name: Name, style):
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.STYLE] = style

    def set_style(self, name: Names, style):
        """Set styles on a node or nodes.

        :param name: Node or nodes to set style for
        :param style: Style to set
        """
        apply_n(self._set_style_one, name, style)

    def _clear_style_one(self, name):
        node_key = to_nodekey(name)
        self.dag.nodes[node_key][NodeAttributes.STYLE] = None

    def clear_style(self, name):
        """Clear style on a node or nodes.

        :param name: Node or nodes to clear styles for
        """
        apply_n(self._clear_style_one, name)

    def metadata(self, name):
        """Get metadata for a node."""
        node_key = to_nodekey(name)
        if self.tree_has_path(name):
            if node_key not in self._metadata:
                self._metadata[node_key] = {}
            return self._metadata[node_key]
        else:
            raise NonExistentNodeException(f"Node {node_key} does not exist.")

    def delete_node(self, name):
        """Delete a node from a computation.

        When nodes are explicitly deleted with ``delete_node``, but are still depended on by other nodes, then they
        will be set to PLACEHOLDER status. In this case, if the nodes that depend on a PLACEHOLDER node are deleted,
        then the PLACEHOLDER node will also be deleted.

        :param name: Name of the node to delete. If the node does not exist, a ``NonExistentNodeException`` will
            be raised.
        """
        node_key = to_nodekey(name)
        LOG.debug(f"Deleting node {node_key}")

        if not self.dag.has_node(node_key):
            raise NonExistentNodeException(f"Node {node_key} does not exist")

        if node_key in self._metadata:
            del self._metadata[node_key]

        if len(self.dag.succ[node_key]) == 0:
            preds = self.dag.predecessors(node_key)
            state = self.dag.nodes[node_key][NodeAttributes.STATE]
            self.dag.remove_node(node_key)
            self._state_map[state].remove(node_key)
            for n in preds:
                if self.dag.nodes[n][NodeAttributes.STATE] == States.PLACEHOLDER:
                    self.delete_node(n)
        else:
            self._set_state(node_key, States.PLACEHOLDER)

    def rename_node(self, old_name: Name | Mapping[Name, Name], new_name: Name | None = None):
        """Rename a node in a computation.

        :param old_name: Node to rename, or a dictionary of nodes to rename, with existing names as keys, and
            new names as values
        :param new_name: New name for node.
        """
        if hasattr(old_name, "__getitem__") and not isinstance(old_name, str):
            for k, v in old_name.items():
                LOG.debug(f"Renaming node {k} to {v}")
            if new_name is not None:
                raise ValueError("new_name must not be set if rename_node is passed a dictionary")
            else:
                name_mapping = old_name
        else:
            LOG.debug(f"Renaming node {old_name} to {new_name}")
            old_node_key = to_nodekey(old_name)
            if not self.dag.has_node(old_node_key):
                raise NonExistentNodeException(f"Node {old_name} does not exist")
            new_node_key = to_nodekey(new_name)
            if self.dag.has_node(new_node_key):
                raise NodeAlreadyExistsException(f"Node {new_name} already exists")
            name_mapping = {old_name: new_name}

        node_key_mapping = {to_nodekey(old_name): to_nodekey(new_name) for old_name, new_name in name_mapping.items()}
        nx.relabel_nodes(self.dag, node_key_mapping, copy=False)

        for old_node_key, new_node_key in node_key_mapping.items():
            if old_node_key in self._metadata:
                self._metadata[new_node_key] = self._metadata[old_node_key]
                del self._metadata[old_node_key]
            else:
                if new_node_key in self._metadata:  # pragma: no cover
                    del self._metadata[new_node_key]

        self._refresh_maps()

    def repoint(self, old_name: Name, new_name: Name):
        """Changes all nodes that use old_name as an input to use new_name instead.

        Note that if old_name is an input to new_name, then that will not be changed, to try to avoid introducing
        circular dependencies, but other circular dependencies will not be checked.

        If new_name does not exist, then it will be created as a PLACEHOLDER node.

        :param old_name:
        :param new_name:
        :return:
        """
        old_node_key = to_nodekey(old_name)
        new_node_key = to_nodekey(new_name)
        if old_node_key == new_node_key:
            return

        changed_names = list(self.dag.successors(old_node_key))

        if len(changed_names) > 0 and not self.dag.has_node(new_node_key):
            self.dag.add_node(new_node_key, **{NodeAttributes.STATE: States.PLACEHOLDER})
            self._state_map[States.PLACEHOLDER].add(new_node_key)

        for name in changed_names:
            if name == new_node_key:
                continue
            edge_data = self.dag.get_edge_data(old_node_key, name)
            self.dag.add_edge(new_node_key, name, **edge_data)
            self.dag.remove_edge(old_node_key, name)

        for name in changed_names:
            self.set_stale(name)

    def insert(self, name: Name, value, force=False):
        """Insert a value into a node of a computation.

        Following insertation, the node will have state UPTODATE, and all its descendents will be COMPUTABLE or STALE.

        If an attempt is made to insert a value into a node that does not exist, a ``NonExistentNodeException``
        will be raised.

        :param name: Name of the node to add.
        :param value: The value to be inserted into the node.
        :param force: Whether to force recalculation of descendents if node value and state would not be changed
        """
        node_key = to_nodekey(name)
        LOG.debug(f"Inserting value into node {node_key}")

        if not self.dag.has_node(node_key):
            raise NonExistentNodeException(f"Node {node_key} does not exist")

        state = self._state_one(name)
        if state == States.PLACEHOLDER:
            raise CannotInsertToPlaceholderNodeException(
                "Cannot insert into placeholder node. Use add_node to create the node first"
            )

        if not force:
            if state == States.UPTODATE:
                current_value = self._value_one(name)
                if value_eq(value, current_value):
                    return

        self._set_state_and_value(node_key, States.UPTODATE, value)
        self._set_descendents(node_key, States.STALE)
        for n in self.dag.successors(node_key):
            self._try_set_computable(n)

    def insert_many(self, name_value_pairs: Iterable[tuple[Name, object]]):
        """Insert values into many nodes of a computation simultaneously.

        Following insertation, the nodes will have state UPTODATE, and all their descendents will be COMPUTABLE
        or STALE. In the case of inserting many nodes, some of which are descendents of others, this ensures that
        the inserted nodes have correct status, rather than being set as STALE when their ancestors are inserted.

        If an attempt is made to insert a value into a node that does not exist, a ``NonExistentNodeException`` will be
        raised, and none of the nodes will be inserted.

        :param name_value_pairs: Each tuple should be a pair (name, value), where name is the name of the node to
            insert the value into.
        :type name_value_pairs: List of tuples
        """
        node_key_value_pairs = [(to_nodekey(name), value) for name, value in name_value_pairs]
        LOG.debug(f"Inserting value into nodes {', '.join(str(name) for name, value in node_key_value_pairs)}")

        for name, value in node_key_value_pairs:
            if not self.dag.has_node(name):
                raise NonExistentNodeException(f"Node {name} does not exist")

        stale = set()
        computable = set()
        for name, value in node_key_value_pairs:
            self._set_state_and_value(name, States.UPTODATE, value)
            stale.update(nx.dag.descendants(self.dag, name))
            computable.update(self.dag.successors(name))
        names = set([name for name, value in node_key_value_pairs])
        stale.difference_update(names)
        computable.difference_update(names)
        for name in stale:
            self._set_state(name, States.STALE)
        for name in computable:
            self._try_set_computable(name)

    def insert_from(self, other, nodes: Iterable[Name] | None = None):
        """Insert values into another Computation object into this Computation object.

        :param other: The computation object to take values from
        :type Computation:
        :param nodes: Only populate the nodes with the names provided in this list. By default, all nodes from the
            other Computation object that have corresponding nodes in this Computation object will be inserted
        :type nodes: List, default None
        """
        if nodes is None:
            nodes = set(self.dag.nodes)
            nodes.intersection_update(other.dag.nodes())
        name_value_pairs = [(name, other.value(name)) for name in nodes]
        self.insert_many(name_value_pairs)

    def _set_state(self, node_key: NodeKey, state: States):
        node = self.dag.nodes[node_key]
        old_state = node[NodeAttributes.STATE]
        self._state_map[old_state].remove(node_key)
        node[NodeAttributes.STATE] = state
        self._state_map[state].add(node_key)

    def _set_state_and_value(
        self, node_key: NodeKey, state: States, value: object, *, throw_conversion_exception: bool = True
    ):
        node = self.dag.nodes[node_key]
        converter = node.get(NodeAttributes.CONVERTER)
        if converter is None:
            self._set_state_and_literal_value(node_key, state, value)
        else:
            try:
                converted_value = converter(value)
                self._set_state_and_literal_value(node_key, state, converted_value)
            except Exception as e:
                tb = traceback.format_exc()
                self._set_error(node_key, e, tb)
                if throw_conversion_exception:
                    raise e

    def _set_state_and_literal_value(
        self, node_key: NodeKey, state: States, value: object, require_old_state: bool = True
    ):
        node = self.dag.nodes[node_key]
        try:
            old_state = node[NodeAttributes.STATE]
            self._state_map[old_state].remove(node_key)
        except KeyError:
            if require_old_state:
                raise  # pragma: no cover
        node[NodeAttributes.STATE] = state
        node[NodeAttributes.VALUE] = value
        self._state_map[state].add(node_key)

    def _set_states(self, node_keys: Iterable[NodeKey], state: States):
        for name in node_keys:
            node = self.dag.nodes[name]
            old_state = node[NodeAttributes.STATE]
            self._state_map[old_state].remove(name)
            node[NodeAttributes.STATE] = state
        self._state_map[state].update(node_keys)

    def set_stale(self, name: Name):
        """Set the state of a node and all its dependencies to STALE.

        :param name: Name of the node to set as STALE.
        """
        node_key = to_nodekey(name)
        node_keys = [node_key]
        node_keys.extend(nx.dag.descendants(self.dag, node_key))
        self._set_states(node_keys, States.STALE)
        self._try_set_computable(node_key)

    def pin(self, name: Name, value=None):
        """Set the state of a node to PINNED.

        :param name: Name of the node to set as PINNED.
        :param value: Value to pin to the node, if provided.
        :type value: default None
        """
        node_key = to_nodekey(name)
        if value is not None:
            self.insert(node_key, value)
        self._set_states([node_key], States.PINNED)

    def unpin(self, name):
        """Unpin a node (state of node and all descendents will be set to STALE).

        :param name: Name of the node to set as PINNED.
        """
        node_key = to_nodekey(name)
        self.set_stale(node_key)

    def _get_descendents(self, node_key: NodeKey, stop_states: set[States] | None = None) -> set[NodeKey]:
        if stop_states is None:
            stop_states = set()
        if self.dag.nodes[node_key][NodeAttributes.STATE] in stop_states:
            return set()
        visited = set()
        to_visit = {node_key}
        while to_visit:
            n = to_visit.pop()
            visited.add(n)
            for n1 in self.dag.successors(n):
                if n1 in visited:
                    continue
                if self.dag.nodes[n1][NodeAttributes.STATE] in stop_states:
                    continue
                to_visit.add(n1)
        visited.remove(node_key)
        return visited

    def _set_descendents(self, node_key: NodeKey, state):
        descendents = self._get_descendents(node_key, {States.PINNED})
        self._set_states(descendents, state)

    def _set_uninitialized(self, node_key: NodeKey):
        self._set_states([node_key], States.UNINITIALIZED)
        self.dag.nodes[node_key].pop(NodeAttributes.VALUE, None)

    def _set_uptodate(self, node_key: NodeKey, value: object):
        self._set_state_and_value(node_key, States.UPTODATE, value)
        self._set_descendents(node_key, States.STALE)
        for n in self.dag.successors(node_key):
            self._try_set_computable(n)

    def _set_error(self, node_key: NodeKey, exc: Exception, tb: inspect.Traceback):
        self._set_state_and_literal_value(node_key, States.ERROR, Error(exc, tb))
        self._set_descendents(node_key, States.STALE)

    def _try_set_computable(self, node_key: NodeKey):
        if self.dag.nodes[node_key][NodeAttributes.STATE] == States.PINNED:
            return
        if self.dag.nodes[node_key].get(NodeAttributes.FUNC) is not None:
            for n in self.dag.predecessors(node_key):
                if not self.dag.has_node(n):
                    return  # pragma: no cover
                if self.dag.nodes[n][NodeAttributes.STATE] != States.UPTODATE:
                    return
            self._set_state(node_key, States.COMPUTABLE)

    def _get_parameter_data(self, node_key: NodeKey):
        for arg, value in self.dag.nodes[node_key][NodeAttributes.ARGS].items():
            yield _ParameterItem(_ParameterType.ARG, arg, value)
        for param_name, value in self.dag.nodes[node_key][NodeAttributes.KWDS].items():
            yield _ParameterItem(_ParameterType.KWD, param_name, value)
        for in_node_name in self.dag.predecessors(node_key):
            param_value = self.dag.nodes[in_node_name][NodeAttributes.VALUE]
            edge = self.dag[in_node_name][node_key]
            param_type, param_name = edge[EdgeAttributes.PARAM]
            yield _ParameterItem(param_type, param_name, param_value)

    def _get_func_args_kwds(self, node_key: NodeKey):
        node0 = self.dag.nodes[node_key]
        f = node0[NodeAttributes.FUNC]
        executor_name = node0.get(NodeAttributes.EXECUTOR)
        args, kwds = [], {}
        for param in self._get_parameter_data(node_key):
            if param.type == _ParameterType.ARG:
                idx = param.name
                while len(args) <= idx:
                    args.append(None)
                args[idx] = param.value
            elif param.type == _ParameterType.KWD:
                kwds[param.name] = param.value
            else:  # pragma: no cover
                raise Exception(f"Unexpected param type: {param.type}")
        return f, executor_name, args, kwds

    def get_definition_args_kwds(self, name: Name) -> tuple[list, dict]:
        """Get the arguments and keyword arguments for a node's function definition."""
        res_args = []
        res_kwds = {}
        node_key = to_nodekey(name)
        node_data = self.dag.nodes[node_key]
        if NodeAttributes.ARGS in node_data:
            for idx, value in node_data[NodeAttributes.ARGS].items():
                while len(res_args) <= idx:
                    res_args.append(None)
                res_args[idx] = C(value)
        if NodeAttributes.KWDS in node_data:
            for param_name, value in node_data[NodeAttributes.KWDS].items():
                res_kwds[param_name] = C(value)
        for in_node_key in self.dag.predecessors(node_key):
            edge = self.dag[in_node_key][node_key]
            if EdgeAttributes.PARAM in edge:
                param_type, param_name = edge[EdgeAttributes.PARAM]
                if param_type == _ParameterType.ARG:
                    idx: int = param_name
                    while len(res_args) <= idx:
                        res_args.append(None)
                    res_args[idx] = in_node_key.name
                elif param_type == _ParameterType.KWD:
                    res_kwds[param_name] = in_node_key.name
                else:  # pragma: no cover
                    raise Exception(f"Unexpected param type: {param_type}")
        return res_args, res_kwds

    def _compute_nodes(self, node_keys: Iterable[NodeKey], raise_exceptions: bool = False):
        LOG.debug(f"Computing nodes {node_keys}")

        futs = {}

        def run(name):
            f, executor_name, args, kwds = self._get_func_args_kwds(name)
            if executor_name is None:
                executor = self.default_executor
            else:
                executor = self.executor_map[executor_name]
            fut = executor.submit(_eval_node, name, f, args, kwds, raise_exceptions)
            futs[fut] = name

        computed = set()

        for node_key in node_keys:
            node0 = self.dag.nodes[node_key]
            state = node0[NodeAttributes.STATE]
            if state == States.COMPUTABLE:
                run(node_key)

        while len(futs) > 0:
            done, not_done = wait(futs.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                node_key = futs.pop(fut)
                node0 = self.dag.nodes[node_key]
                try:
                    value, exc, tb, start_dt, end_dt = fut.result()
                except Exception as e:
                    exc = e
                    tb = traceback.format_exc()
                    self._set_error(node_key, exc, tb)
                    raise
                delta = (end_dt - start_dt).total_seconds()
                if exc is None:
                    self._set_state_and_value(node_key, States.UPTODATE, value, throw_conversion_exception=False)
                    node0[NodeAttributes.TIMING] = TimingData(start_dt, end_dt, delta)
                    self._set_descendents(node_key, States.STALE)
                    for n in self.dag.successors(node_key):
                        logging.debug(str(node_key) + " " + str(n) + " " + str(computed))
                        if n in computed:
                            raise LoopDetectedException(f"Calculating {node_key} for the second time")
                        self._try_set_computable(n)
                        node0 = self.dag.nodes[n]
                        state = node0[NodeAttributes.STATE]
                        if state == States.COMPUTABLE and n in node_keys:
                            run(n)
                else:
                    self._set_error(node_key, exc, tb)
                computed.add(node_key)

    def _get_calc_node_keys(self, node_key: NodeKey) -> list[NodeKey]:
        g = nx.DiGraph()
        g.add_nodes_from(self.dag.nodes)
        g.add_edges_from(self.dag.edges)
        for n in nx.ancestors(g, node_key):
            node = self.dag.nodes[n]
            state = node[NodeAttributes.STATE]
            if state == States.UPTODATE or state == States.PINNED:
                g.remove_node(n)

        ancestors = nx.ancestors(g, node_key)
        for n in ancestors:
            if state == States.UNINITIALIZED and len(self.dag.pred[n]) == 0:
                raise Exception(f"Cannot compute {node_key} because {n} uninitialized")
            if state == States.PLACEHOLDER:
                raise Exception(f"Cannot compute {node_key} because {n} is placeholder")

        ancestors.add(node_key)
        nodes_sorted = topological_sort(g)
        return [n for n in nodes_sorted if n in ancestors]

    def _get_calc_node_names(self, name: Name) -> Names:
        node_key = to_nodekey(name)
        return node_keys_to_names(self._get_calc_node_keys(node_key))

    def compute(self, name: Name | Iterable[Name], raise_exceptions=False):
        """Compute a node and all necessary predecessors.

        Following the computation, if successful, the target node, and all necessary ancestors that were not already
        UPTODATE will have been calculated and set to UPTODATE. Any node that did not need to be calculated will not
        have been recalculated.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an
        object containing the exception object, as well as a traceback. This will not halt the computation, which
        will proceed as far as it can, until no more nodes that would be required to calculate the target are
        COMPUTABLE.

        :param name: Name of the node to compute
        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """
        if isinstance(name, (types.GeneratorType, list)):
            calc_nodes = set()
            for name0 in name:
                node_key = to_nodekey(name0)
                for n in self._get_calc_node_keys(node_key):
                    calc_nodes.add(n)
        else:
            node_key = to_nodekey(name)
            calc_nodes = self._get_calc_node_keys(node_key)
        self._compute_nodes(calc_nodes, raise_exceptions=raise_exceptions)

    def compute_all(self, raise_exceptions=False):
        """Compute all nodes of a computation that can be computed.

        Nodes that are already UPTODATE will not be recalculated. Following the computation, if successful, all
        nodes will have state UPTODATE, except UNINITIALIZED input nodes and PLACEHOLDER nodes.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an
        object containing the exception object, as well as a traceback. This will not halt the computation, which
        will proceed as far as it can, until no more nodes are COMPUTABLE.

        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """
        self._compute_nodes(self._node_keys(), raise_exceptions=raise_exceptions)

    def _node_keys(self) -> list[NodeKey]:
        """Get a list of nodes in this computation.

        :return: List of nodes.
        """
        return list(self.dag.nodes)

    def nodes(self) -> list[Name]:
        """Get a list of nodes in this computation.

        :return: List of nodes.
        """
        return list(n.name for n in self.dag.nodes)

    def get_tree_list_children(self, name: Name) -> set[Name]:
        """Get a list of nodes in this computation.

        :return: List of nodes.
        """
        node_key = to_nodekey(name)
        idx = len(node_key.parts)
        result = set()
        for n in self.dag.nodes:
            if n.is_descendent_of(node_key):
                result.add(n.parts[idx])
        return result

    def has_node(self, name: Name):
        """Check if a node with the given name exists in the computation."""
        node_key = to_nodekey(name)
        return node_key in self.dag.nodes

    def tree_has_path(self, name: Name):
        """Check if a hierarchical path exists in the computation tree."""
        node_key = to_nodekey(name)
        if node_key.is_root:
            return True
        if self.has_node(node_key):
            return True
        for n in self.dag.nodes:
            if n.is_descendent_of(node_key):
                return True
        return False

    def get_tree_descendents(
        self, name: Name | None = None, *, include_stem: bool = True, graph_nodes_only: bool = False
    ) -> set[Name]:
        """Get a list of descendent blocks and nodes.

        Returns blocks and nodes that are descendents of the input node,
        e.g. for node 'foo', might return ['foo/bar', 'foo/baz'].

        :param name: Name of node to get descendents for
        :return: List of descendent node names
        """
        node_key = NodeKey.root() if name is None else to_nodekey(name)
        stemsize = len(node_key.parts)
        result = set()
        for n in self.dag.nodes:
            if n.is_descendent_of(node_key):
                if graph_nodes_only:
                    nodes = [n]
                else:
                    nodes = n.ancestors()
                for n2 in nodes:
                    if n2.is_descendent_of(node_key):
                        if include_stem:
                            nm = n2.name
                        else:
                            nm = NodeKey(tuple(n2.parts[stemsize:])).name
                        result.add(nm)
        return result

    def _state_one(self, name: Name):
        node_key = to_nodekey(name)
        return self.dag.nodes[node_key][NodeAttributes.STATE]

    def state(self, name: Name | Names):
        """Get the state of a node.

        This can also be accessed using the attribute-style accessor ``s`` if ``name`` is a valid Python
        attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.state('foo')
            <States.UPTODATE: 4>
            >>> comp.s.foo
            <States.UPTODATE: 4>

        :param name: Name or names of the node to get state for
        :type name: Name or Names
        """
        return apply1(self._state_one, name)

    def _value_one(self, name: Name):
        node_key = to_nodekey(name)
        return self.dag.nodes[node_key][NodeAttributes.VALUE]

    def value(self, name: Name | Names):
        """Get the current value of a node.

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python
        attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.value('foo')
            1
            >>> comp.v.foo
            1

        :param name: Name or names of the node to get the value of
        :type name: Name or Names
        """
        return apply1(self._value_one, name)

    def compute_and_get_value(self, name: Name):
        """Get the current value of a node.

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python
        attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', lambda foo: foo + 1)
            >>> comp.compute_and_get_value('bar')
            2
            >>> comp.x.bar
            2

        :param name: Name or names of the node to get the value of
        :type name: Name
        """
        name = to_nodekey(name)
        if self.state(name) == States.UPTODATE:
            return self.value(name)
        self.compute(name, raise_exceptions=True)
        if self.state(name) == States.UPTODATE:
            return self.value(name)
        raise ComputationError(f"Unable to compute node {name}")

    def _tag_one(self, name: Name):
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        return node[NodeAttributes.TAG]

    def tags(self, name: Name | Names):
        """Get the tags associated with a node.

            >>> comp = Computation()
            >>> comp.add_node('a', tags=['foo', 'bar'])
            >>> sorted(comp.t.a)
            ['__serialize__', 'bar', 'foo']

        :param name: Name or names of the node to get the tags of
        :return:
        """
        return apply1(self._tag_one, name)

    def nodes_by_tag(self, tag) -> set[Name]:
        """Get the names of nodes with a particular tag or tags.

        :param tag: Tag or tags for which to retrieve nodes
        :return: Names of the nodes with those tags
        """
        nodes = set()
        for tag1 in as_iterable(tag):
            nodes1 = self._tag_map.get(tag1)
            if nodes1 is not None:
                nodes.update(nodes1)
        return set(n.name for n in nodes)

    def _style_one(self, name: Name):
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        return node.get(NodeAttributes.STYLE)

    def styles(self, name: Name | Names):
        """Get the tags associated with a node.

            >>> comp = Computation()
            >>> comp.add_node('a', style='dot')
            >>> comp.style.a
            'dot'

        :param name: Name or names of the node to get the tags of
        :return:
        """
        return apply1(self._style_one, name)

    def _get_item_one(self, name: Name):
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        return NodeData(node[NodeAttributes.STATE], node[NodeAttributes.VALUE])

    def __getitem__(self, name: Name | Names):
        """Get the state and current value of a node.

        :param name: Name of the node to get the state and value of
        """
        return apply1(self._get_item_one, name)

    def _get_timing_one(self, name: Name):
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        return node.get(NodeAttributes.TIMING, None)

    def get_timing(self, name: Name | Names):
        """Get the timing information for a node.

        :param name: Name or names of the node to get the timing information of
        :return:
        """
        return apply1(self._get_timing_one, name)

    def to_df(self):
        """Get a dataframe containing the states and value of all nodes of computation.

        ::

            >>> import loman
            >>> comp = loman.Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', value=2)
            >>> comp.to_df()  # doctest: +NORMALIZE_WHITESPACE
                           state  value
            foo  States.UPTODATE      1
            bar  States.UPTODATE      2
        """
        df = pd.DataFrame(index=topological_sort(self.dag))
        df[NodeAttributes.STATE] = pd.Series(nx.get_node_attributes(self.dag, NodeAttributes.STATE))
        df[NodeAttributes.VALUE] = pd.Series(nx.get_node_attributes(self.dag, NodeAttributes.VALUE))
        df_timing = pd.DataFrame.from_dict(nx.get_node_attributes(self.dag, "timing"), orient="index")
        df = pd.merge(df, df_timing, left_index=True, right_index=True, how="left")
        df.index = pd.Index([nk.name for nk in df.index])
        return df

    def to_dict(self):
        """Get a dictionary containing the values of all nodes of a computation.

        ::

            >>> import loman
            >>> comp = loman.Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', value=2)
            >>> comp.to_dict()  # doctest: +ELLIPSIS
            {NodeKey('foo'): 1, NodeKey('bar'): 2}
        """
        return nx.get_node_attributes(self.dag, NodeAttributes.VALUE)

    def _get_inputs_one_node_keys(self, node_key: NodeKey) -> list[NodeKey]:
        args_dict = {}
        kwds = []
        max_arg_index = -1
        for input_node in self.dag.predecessors(node_key):
            input_edge = self.dag[input_node][node_key]
            input_type, input_param = input_edge[EdgeAttributes.PARAM]
            if input_type == _ParameterType.ARG:
                idx = input_param
                max_arg_index = max(max_arg_index, idx)
                args_dict[idx] = input_node
            elif input_type == _ParameterType.KWD:
                kwds.append(input_node)
        if max_arg_index >= 0:
            args = [None] * (max_arg_index + 1)
            for idx, input_node in args_dict.items():
                args[idx] = input_node
            return args + kwds
        else:
            return kwds

    def _get_inputs_one_names(self, name: Name) -> Names:
        node_key = to_nodekey(name)
        return node_keys_to_names(self._get_inputs_one_node_keys(node_key))

    def get_inputs(self, name: Name | Names) -> list[Names]:
        """Get a list of the inputs for a node or set of nodes.

        :param name: Name or names of nodes to get inputs for
        :return: If name is scalar, return a list of upstream nodes used as input. If name is a list, return a
            list of list of inputs.
        """
        return apply1(self._get_inputs_one_names, name)

    def _get_ancestors_node_keys(self, node_keys: Iterable[NodeKey], include_self=True) -> set[NodeKey]:
        ancestors = set()
        for n in node_keys:
            if include_self:
                ancestors.add(n)
            for ancestor in nx.ancestors(self.dag, n):
                ancestors.add(ancestor)
        return ancestors

    def get_ancestors(self, names: Name | Names, include_self=True) -> Names:
        """Get all ancestor nodes of the specified nodes."""
        node_keys = names_to_node_keys(names)
        ancestor_node_keys = self._get_ancestors_node_keys(node_keys, include_self)
        return node_keys_to_names(ancestor_node_keys)

    def _get_original_inputs_node_keys(self, node_keys: list[NodeKey] | None) -> Names:
        if node_keys is None:
            node_keys = self._node_keys()
        else:
            node_keys = self._get_ancestors_node_keys(node_keys)
        return [n for n in node_keys if self.dag.nodes[n].get(NodeAttributes.FUNC) is None]

    def get_original_inputs(self, names: Name | Names | None = None) -> Names:
        """Get a list of the original non-computed inputs for a node or set of nodes.

        :param names: Name or names of nodes to get inputs for
        :return: Return a list of original non-computed inputs that are ancestors of the input nodes
        """
        if names is None:
            node_keys = None
        else:
            node_keys = names_to_node_keys(names)

        node_keys = self._get_original_inputs_node_keys(node_keys)

        return node_keys_to_names(node_keys)

    def _get_outputs_one(self, name: Name) -> Names:
        node_key = to_nodekey(name)
        output_node_keys = list(self.dag.successors(node_key))
        return node_keys_to_names(output_node_keys)

    def get_outputs(self, name: Name | Names) -> Names | list[Names]:
        """Get a list of the outputs for a node or set of nodes.

        :param name: Name or names of nodes to get outputs for
        :return: If name is scalar, return a list of downstream nodes used as output. If name is a list, return a
            list of list of outputs.

        """
        return apply1(self._get_outputs_one, name)

    def _get_descendents_node_keys(self, node_keys: Iterable[NodeKey], include_self: bool = True) -> Names:
        ancestor_node_keys = set()
        for node_key in node_keys:
            if include_self:
                ancestor_node_keys.add(node_key)
            for ancestor in nx.descendants(self.dag, node_key):
                ancestor_node_keys.add(ancestor)
        return ancestor_node_keys

    def get_descendents(self, names: Name | Names, include_self: bool = True) -> Names:
        """Get all descendent nodes of the specified nodes."""
        node_keys = names_to_node_keys(names)
        descendent_node_keys = self._get_descendents_node_keys(node_keys, include_self)
        return node_keys_to_names(descendent_node_keys)

    def get_final_outputs(self, names: Name | Names | None = None):
        """Get final output nodes (nodes with no descendants) from the specified nodes."""
        if names is None:
            node_keys = self._node_keys()
        else:
            node_keys = names_to_node_keys(names)
            node_keys = self._get_descendents_node_keys(node_keys)
        output_node_keys = [n for n in node_keys if len(nx.descendants(self.dag, n)) == 0]
        return node_keys_to_names(output_node_keys)

    def get_source(self, name: Name) -> str:
        """Get the source code for a node."""
        node_key = to_nodekey(name)
        func = self.dag.nodes[node_key].get(NodeAttributes.FUNC, None)
        if func is not None:
            file = inspect.getsourcefile(func)
            _, lineno = inspect.getsourcelines(func)
            source = inspect.getsource(func)
            return f"{file}:{lineno}\n\n{source}"
        else:
            return "NOT A CALCULATED NODE"

    def print_source(self, name: Name):
        """Print the source code for a computation node."""
        print(self.get_source(name))

    def restrict(self, output_names: Name | Names, input_names: Name | Names | None = None):
        """Restrict a computation to the ancestors of a set of output nodes.

        Excludes ancestors of a set of input nodes.

        If the set of input_nodes that is specified is not sufficient for the set of output_nodes then additional
        nodes that are ancestors of the output_nodes will be included, but the input nodes specified will be input
        nodes of the modified Computation.

        :param output_nodes:
        :param input_nodes:
        :return: None - modifies existing computation in place
        """
        if input_names is not None:
            for name in input_names:
                nodedata = self._get_item_one(name)
                self.add_node(name)
                self._set_state_and_literal_value(to_nodekey(name), nodedata.state, nodedata.value)
        output_node_keys = names_to_node_keys(output_names)
        ancestor_node_keys = self._get_ancestors_node_keys(output_node_keys)
        self.dag.remove_nodes_from([n for n in self.dag if n not in ancestor_node_keys])

    def __getstate__(self):
        """Prepare computation for serialization by removing non-serializable nodes."""
        node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
        obj = self.copy()
        for name, tags in node_serialize.items():
            if SystemTags.SERIALIZE not in tags:
                obj._set_uninitialized(name)
        return {"dag": obj.dag}

    def __setstate__(self, state):
        """Restore computation from serialized state."""
        self.__init__()
        self.dag = state["dag"]
        self._refresh_maps()

    def write_dill_old(self, file_):
        """Serialize a computation to a file or file-like object.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        warnings.warn("write_dill_old is deprecated, use write_dill instead", DeprecationWarning, stacklevel=2)
        original_getstate = self.__class__.__getstate__
        original_setstate = self.__class__.__setstate__

        try:
            del self.__class__.__getstate__
            del self.__class__.__setstate__

            node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
            obj = self.copy()
            obj.executor_map = None
            obj.default_executor = None
            for name, tags in node_serialize.items():
                if SystemTags.SERIALIZE not in tags:
                    obj._set_uninitialized(name)

            if isinstance(file_, str):
                with open(file_, "wb") as f:
                    dill.dump(obj, f)
            else:
                dill.dump(obj, file_)
        finally:
            self.__class__.__getstate__ = original_getstate
            self.__class__.__setstate__ = original_setstate

    def write_dill(self, file_):
        """Serialize a computation to a file or file-like object.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        if isinstance(file_, str):
            with open(file_, "wb") as f:
                dill.dump(self, f)
        else:
            dill.dump(self, file_)

    @staticmethod
    def read_dill(file_):
        """Deserialize a computation from a file or file-like object.

        .. warning::
            This method uses dill.load() which can execute arbitrary code.
            Only load files from trusted sources. Never load data from
            untrusted or unauthenticated sources as it may lead to arbitrary
            code execution.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        if isinstance(file_, str):
            with open(file_, "rb") as f:
                obj = dill.load(f)
        else:
            obj = dill.load(file_)
        if isinstance(obj, Computation):
            return obj
        else:
            raise Exception()

    def copy(self):
        """Create a copy of a computation.

        The copy is shallow. Any values in the new Computation's DAG will be the same object as this Computation's
        DAG. As new objects will be created by any further computations, this should not be an issue.

        :rtype: Computation
        """
        obj = Computation()
        obj.dag = nx.DiGraph(self.dag)
        obj._tag_map = {tag: nodes.copy() for tag, nodes in self._tag_map.items()}
        obj._state_map = {state: nodes.copy() for state, nodes in self._state_map.items()}
        return obj

    def add_named_tuple_expansion(self, name, namedtuple_type, group=None):
        """Automatically add nodes to extract each element of a named tuple type.

        It is often convenient for a calculation to return multiple values, and it is polite to do this a namedtuple
        rather than a regular tuple, so that later users have same name to identify elements of the tuple. It can
        also help make a computation clearer if a downstream computation depends on one element of such a tuple,
        rather than the entire tuple. This does not affect the computation per se, but it does make the intention
        clearer.

        To avoid having to create many boiler-plate node definitions to expand namedtuples, the
        ``add_named_tuple_expansion`` method automatically creates new nodes for each element of a tuple. The
        convention is that an element called 'element', in a node called 'node' will be expanded into a new node
        called 'node.element', and that this will be applied for each element.

        Example::

            >>> from collections import namedtuple
            >>> Coordinate = namedtuple('Coordinate', ['x', 'y'])
            >>> comp = Computation()
            >>> comp.add_node('c', value=Coordinate(1, 2))
            >>> comp.add_named_tuple_expansion('c', Coordinate)
            >>> comp.compute_all()
            >>> comp.value('c.x')
            1
            >>> comp.value('c.y')
            2

        :param name: Node to cera
        :param namedtuple_type: Expected type of the node
        :type namedtuple_type: namedtuple class
        """

        def make_f(field_name):
            def get_field_value(tuple):
                return getattr(tuple, field_name)

            return get_field_value

        for field_name in namedtuple_type._fields:
            node_name = f"{name}.{field_name}"
            self.add_node(node_name, make_f(field_name), kwds={"tuple": name}, group=group)
            self.set_tag(node_name, SystemTags.EXPANSION)

    def add_map_node(self, result_node, input_node, subgraph, subgraph_input_node, subgraph_output_node):
        """Apply a graph to each element of iterable.

        In turn, each element in the ``input_node`` of this graph will be inserted in turn into the subgraph's
        ``subgraph_input_node``, then the subgraph's ``subgraph_output_node`` calculated. The resultant list, with
        an element or each element in ``input_node``, will be inserted into ``result_node`` of this graph. In this
        way ``add_map_node`` is similar to ``map`` in functional programming.

        :param result_node: The node to place a list of results in **this** graph
        :param input_node: The node to get a list input values from **this** graph
        :param subgraph: The graph to use to perform calculation for each element
        :param subgraph_input_node: The node in **subgraph** to insert each element in turn
        :param subgraph_output_node: The node in **subgraph** to read the result for each element
        """

        def f(xs):
            results = []
            is_error = False
            for x in xs:
                subgraph.insert(subgraph_input_node, x)
                subgraph.compute(subgraph_output_node)
                if subgraph.state(subgraph_output_node) == States.UPTODATE:
                    results.append(subgraph.value(subgraph_output_node))
                else:
                    is_error = True
                    results.append(subgraph.copy())
            if is_error:
                raise MapException(f"Unable to calculate {result_node}", results)
            return results

        self.add_node(result_node, f, kwds={"xs": input_node})

    def prepend_path(self, path, prefix_path: NodeKey):
        """Prepend a prefix path to a node path."""
        if isinstance(path, ConstantValue):
            return path
        path = to_nodekey(path)
        return prefix_path.join(path)

    def add_block(
        self,
        base_path: Name,
        block: "Computation",
        *,
        keep_values: bool | None = True,
        links: dict | None = None,
        metadata: dict | None = None,
    ):
        """Add a computation block as a subgraph to this computation."""
        base_path = to_nodekey(base_path)
        for node_name in block.nodes():
            node_key = to_nodekey(node_name)
            node_data = block.dag.nodes[node_key]
            tags = node_data.get(NodeAttributes.TAG, None)
            style = node_data.get(NodeAttributes.STYLE, None)
            group = node_data.get(NodeAttributes.GROUP, None)
            args, kwds = block.get_definition_args_kwds(node_key)
            args = [self.prepend_path(arg, base_path) for arg in args]
            kwds = {k: self.prepend_path(v, base_path) for k, v in kwds.items()}
            func = node_data.get(NodeAttributes.FUNC, None)
            executor = node_data.get(NodeAttributes.EXECUTOR, None)
            converter = node_data.get(NodeAttributes.CONVERTER, None)
            new_node_name = self.prepend_path(node_name, base_path)
            self.add_node(
                new_node_name,
                func,
                args=args,
                kwds=kwds,
                converter=converter,
                serialize=False,
                inspect=False,
                group=group,
                tags=tags,
                style=style,
                executor=executor,
            )
            if keep_values and NodeAttributes.VALUE in node_data:
                new_node_key = to_nodekey(new_node_name)
                self._set_state_and_literal_value(
                    new_node_key, node_data[NodeAttributes.STATE], node_data[NodeAttributes.VALUE]
                )
        if links is not None:
            for target, source in links.items():
                self.link(base_path.join_parts(target), source)
        if metadata is not None:
            self._metadata[base_path] = metadata
        else:
            if base_path in self._metadata:
                del self._metadata[base_path]

    def link(self, target: Name, source: Name):
        """Create a link between two nodes in the computation graph."""
        target = to_nodekey(target)
        source = to_nodekey(source)
        if target == source:
            return

        target_style = self._style_one(target) if self.has_node(target) else None
        source_style = self._style_one(source) if self.has_node(source) else None
        style = target_style if target_style else source_style

        self.add_node(target, identity_function, kwds={"x": source}, style=style)

    def _repr_svg_(self):
        return GraphView(self).svg()

    def draw(
        self,
        root: NodeKey | None = None,
        *,
        node_transformations: dict | None = None,
        cmap=None,
        colors="state",
        shapes=None,
        graph_attr=None,
        node_attr=None,
        edge_attr=None,
        show_expansion=False,
        collapse_all=True,
    ):
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
        node_transformations = node_transformations.copy() if node_transformations is not None else {}
        if not show_expansion:
            for nodekey in self.nodes_by_tag(SystemTags.EXPANSION):
                node_transformations[nodekey] = NodeTransformations.CONTRACT
        v = GraphView(
            self,
            root=root,
            node_formatter=node_formatter,
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
            node_transformations=node_transformations,
            collapse_all=collapse_all,
        )
        return v

    def view(self, cmap=None, colors="state", shapes=None):
        """Create and display a visualization of the computation graph."""
        node_formatter = NodeFormatter.create(cmap, colors, shapes)
        v = GraphView(self, node_formatter=node_formatter)
        v.view()

    def print_errors(self):
        """Print tracebacks for every node with state "ERROR" in a Computation."""
        for n in self.nodes():
            if self.s[n] == States.ERROR:
                print(f"{n}")
                print("=" * len(n))
                print()
                print(self.v[n].traceback)
                print()

    @classmethod
    def from_class(cls, definition_class, ignore_self=True):
        """Create a computation from a class with decorated methods."""
        comp = cls()
        obj = definition_class()
        populate_computation_from_class(comp, definition_class, obj, ignore_self=ignore_self)
        return comp

    def inject_dependencies(self, dependencies: dict, *, force: bool = False):
        """Injects dependencies into the nodes of the current computation where nodes are in a placeholder state.

        (or all possible nodes when the 'force' parameter is set to True), using values
        provided in the 'dependencies' dictionary.

        Each key in the 'dependencies' dictionary corresponds to a node identifier, and the associated
        value is the dependency object to inject. If the value is a callable, it will be added as a calc node.

        :param dependencies: A dictionary where each key-value pair consists of a node identifier and
                             its corresponding dependency object or a callable that returns the dependency object.
        :param force: A boolean flag that, when set to True, forces the replacement of existing node values
                      with the ones provided in 'dependencies', regardless of their current state. Defaults to False.
        :return: None
        """
        for n in self.nodes():
            if force or self.s[n] == States.PLACEHOLDER:
                obj = dependencies.get(n)
                if obj is None:
                    continue
                if callable(obj):
                    self.add_node(n, obj)
                else:
                    self.add_node(n, value=obj)
