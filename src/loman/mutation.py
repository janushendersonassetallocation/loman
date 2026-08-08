"""Structural changes to a computation graph.

Everything that adds, removes, rewires or populates nodes lives here:
:meth:`~MutationMixin.add_node` and its parameter-binding helpers, deletion and
renaming, value insertion, block composition, and the convenience constructors
built on top of them.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import networkx as nx

from .compat import get_signature
from .consts import EdgeAttributes, NodeAttributes, States, SystemTags
from .exception import (
    CannotInsertToPlaceholderNodeException,
    MapException,
    NodeAlreadyExistsException,
    NonExistentNodeException,
)
from .nodekey import Name, Names, NodeKey, names_to_node_keys, to_nodekey
from .persistence import PersistenceMixin
from .subscription import _notifies_subscribers
from .util import as_iterable, value_eq
from .values import ConstantValue, _ParameterType

if TYPE_CHECKING:
    from .computeengine import Computation

# Deliberately the pre-split logger name: this refactor moved code between
# modules and should not silently rename the channel people filter on.
LOG = logging.getLogger("loman.computeengine")

_MISSING_VALUE_SENTINEL = object()


def identity_function(x: Any) -> Any:
    """Return the input value unchanged."""
    return x


class MutationMixin(PersistenceMixin):
    """Adding, removing, rewiring and populating nodes."""

    def _process_function_args(self, node_key: NodeKey, node: dict[str, Any], args: list[Any] | None) -> int:
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

    def _build_param_map(
        self,
        func: Callable[..., Any],
        node_key: NodeKey,
        args_count: int,
        kwds: dict[str, Any] | None,
        inspect: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build parameter map for function node."""
        param_map: dict[str, Any] = {}
        default_names: list[str] = []

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

    def _process_function_kwds(
        self, node_key: NodeKey, node: dict[str, Any], param_map: dict[str, Any], default_names: list[str]
    ) -> None:
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

    @_notifies_subscribers(graph_changed=True)
    def add_node(
        self,
        name: Name,
        func: Callable[..., Any] | None = None,
        *,
        args: list[Any] | None = None,
        kwds: dict[str, Any] | None = None,
        value: Any = _MISSING_VALUE_SENTINEL,
        converter: Callable[[Any], Any] | None = None,
        serialize: bool = True,
        inspect: bool = True,
        group: str | None = None,
        tags: Iterable[str] | None = None,
        style: str | None = None,
        executor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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

    @_notifies_subscribers(graph_changed=True)
    def delete_node(self, name: Name) -> None:
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
            msg = f"Node {node_key} does not exist"
            raise NonExistentNodeException(msg)

        if node_key in self._metadata:
            del self._metadata[node_key]

        if len(self.dag.succ[node_key]) == 0:
            preds = self.dag.predecessors(node_key)
            state = self.dag.nodes[node_key][NodeAttributes.STATE]
            self.dag.remove_node(node_key)
            self._state_map[state].remove(node_key)
            self._mark_changed(node_key)
            for n in preds:
                if self.dag.nodes[n][NodeAttributes.STATE] == States.PLACEHOLDER:
                    self.delete_node(n)
        else:
            self._set_state(node_key, States.PLACEHOLDER)

    @_notifies_subscribers(graph_changed=True)
    def rename_node(self, old_name: Name | Mapping[Name, Name], new_name: Name | None = None) -> None:
        """Rename a node in a computation.

        :param old_name: Node to rename, or a dictionary of nodes to rename, with existing names as keys, and
            new names as values
        :param new_name: New name for node.
        """
        name_mapping: dict[Name, Name]
        if isinstance(old_name, Mapping) and not isinstance(old_name, str):
            for k, v in old_name.items():
                LOG.debug(f"Renaming node {k} to {v}")
            if new_name is not None:
                msg = "new_name must not be set if rename_node is passed a dictionary"
                raise ValueError(msg)
            else:
                name_mapping = dict(old_name)  # type: ignore[arg-type]
        else:
            LOG.debug(f"Renaming node {old_name} to {new_name}")
            old_node_key = to_nodekey(old_name)
            if not self.dag.has_node(old_node_key):
                msg = f"Node {old_name} does not exist"
                raise NonExistentNodeException(msg)
            assert new_name is not None  # noqa: S101
            new_node_key = to_nodekey(new_name)
            if self.dag.has_node(new_node_key):
                msg = f"Node {new_name} already exists"
                raise NodeAlreadyExistsException(msg)
            name_mapping = {old_name: new_name}

        node_key_mapping = {to_nodekey(on): to_nodekey(nn) for on, nn in name_mapping.items()}
        nx.relabel_nodes(self.dag, node_key_mapping, copy=False)

        for old_nk, new_nk in node_key_mapping.items():
            if old_nk in self._metadata:
                self._metadata[new_nk] = self._metadata[old_nk]
                del self._metadata[old_nk]
            else:
                if new_nk in self._metadata:  # pragma: no cover
                    del self._metadata[new_nk]

        self._mark_changed(*node_key_mapping.keys(), *node_key_mapping.values())
        self._refresh_maps()

    @_notifies_subscribers(graph_changed=True)
    def repoint(self, old_name: Name, new_name: Name) -> None:
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

        for n in changed_names:
            self.set_stale(n)

    @_notifies_subscribers()
    def insert(self, name: Name, value: Any, force: bool = False) -> None:
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
            msg = f"Node {node_key} does not exist"
            raise NonExistentNodeException(msg)

        state = self._state_one(name)
        if state == States.PLACEHOLDER:
            msg = "Cannot insert into placeholder node. Use add_node to create the node first"
            raise CannotInsertToPlaceholderNodeException(msg)

        if not force and state == States.UPTODATE:
            current_value = self._value_one(name)
            if value_eq(value, current_value):
                return

        self._set_state_and_value(node_key, States.UPTODATE, value)
        self._set_descendents(node_key, States.STALE)
        for n in self.dag.successors(node_key):
            self._try_set_computable(n)

    @_notifies_subscribers()
    def insert_many(self, name_value_pairs: Iterable[tuple[Name, object]]) -> None:
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

        for name, _value in node_key_value_pairs:
            if not self.dag.has_node(name):
                msg = f"Node {name} does not exist"
                raise NonExistentNodeException(msg)

        stale = set()
        computable = set()
        for name, value in node_key_value_pairs:
            self._set_state_and_value(name, States.UPTODATE, value)
            stale.update(nx.dag.descendants(self.dag, name))
            computable.update(self.dag.successors(name))
        names = {name for name, value in node_key_value_pairs}
        stale.difference_update(names)
        computable.difference_update(names)
        for name in stale:
            self._set_state(name, States.STALE)
        for name in computable:
            self._try_set_computable(name)

    @_notifies_subscribers()
    def insert_from(self, other: "Computation", nodes: Iterable[Name] | None = None) -> None:
        """Insert values into another Computation object into this Computation object.

        :param other: The computation object to take values from
        :type Computation:
        :param nodes: Only populate the nodes with the names provided in this list. By default, all nodes from the
            other Computation object that have corresponding nodes in this Computation object will be inserted
        :type nodes: List, default None
        """
        if nodes is None:
            nodes_set: set[Any] = set(self.dag.nodes)
            nodes_set.intersection_update(other.dag.nodes())
            nodes = nodes_set
        name_value_pairs = [(name, other.value(name)) for name in nodes]
        self.insert_many(name_value_pairs)

    @_notifies_subscribers()
    def pin(self, name: Name, value: Any = None) -> None:
        """Set the state of a node to PINNED.

        :param name: Name of the node to set as PINNED.
        :param value: Value to pin to the node, if provided.
        :type value: default None
        """
        node_key = to_nodekey(name)
        if value is not None:
            self.insert(node_key, value)
        self._set_states([node_key], States.PINNED)

    @_notifies_subscribers(graph_changed=True)
    def restrict(self, output_names: Name | Names, input_names: Name | Names | None = None) -> None:
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
            for n in as_iterable(input_names):
                nodedata = self._get_item_one(n)
                self.add_node(n)
                self._set_state_and_literal_value(to_nodekey(n), nodedata.state, nodedata.value)
        output_node_keys = names_to_node_keys(output_names)
        ancestor_node_keys = self._get_ancestors_node_keys(output_node_keys)
        removed = [n for n in self.dag if n not in ancestor_node_keys]
        self.dag.remove_nodes_from(removed)
        self._mark_changed(*removed)

    @_notifies_subscribers(graph_changed=True)
    def add_named_tuple_expansion(self, name: Name, namedtuple_type: type, group: str | None = None) -> None:
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
            >>> from loman import Computation
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

        def make_f(field_name: str) -> Callable[[Any], Any]:
            """Create a function to extract a field from a namedtuple."""

            def get_field_value(tuple_val: Any) -> Any:
                """Extract field value from the namedtuple."""
                return getattr(tuple_val, field_name)

            return get_field_value

        for field_name in namedtuple_type._fields:  # type: ignore[attr-defined]
            node_name = f"{name}.{field_name}"
            self.add_node(node_name, make_f(field_name), kwds={"tuple_val": name}, group=group)
            self.set_tag(node_name, SystemTags.EXPANSION)

    @_notifies_subscribers(graph_changed=True)
    def add_map_node(
        self,
        result_node: Name,
        input_node: Name,
        subgraph: "Computation",
        subgraph_input_node: Name,
        subgraph_output_node: Name,
    ) -> None:
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

        def f(xs: Iterable[Any]) -> list[Any]:
            """Apply subgraph computation to each element in the input."""
            results: list[Any] = []
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
                msg = f"Unable to calculate {result_node}"
                raise MapException(msg, results)
            return results

        self.add_node(result_node, f, kwds={"xs": input_node})

    def prepend_path(self, path: Name | ConstantValue, prefix_path: NodeKey) -> NodeKey | ConstantValue:
        """Prepend a prefix path to a node path."""
        if isinstance(path, ConstantValue):
            return path
        nk = to_nodekey(path)
        return prefix_path.join(nk)

    @_notifies_subscribers(graph_changed=True)
    def add_block(
        self,
        base_path: Name,
        block: "Computation",
        *,
        keep_values: bool | None = True,
        links: dict[str, Name] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a computation block as a subgraph to this computation."""
        base_path_nk = to_nodekey(base_path)
        for node_name in block.nodes():
            node_key = to_nodekey(node_name)
            node_data = block.dag.nodes[node_key]
            tags = node_data.get(NodeAttributes.TAG, None)
            # strip the serialize tag from the original node: add_block explicitly
            # sets serialize=False, meaning "don't serialize the function".
            # Value serialization is controlled separately via keep_values below.
            if tags is not None:
                tags = tags - {SystemTags.SERIALIZE}
            style = node_data.get(NodeAttributes.STYLE, None)
            group = node_data.get(NodeAttributes.GROUP, None)
            args_def, kwds_def = block.get_definition_args_kwds(node_key)
            args_prepended = [self.prepend_path(arg, base_path_nk) for arg in args_def]
            kwds_prepended = {k: self.prepend_path(v, base_path_nk) for k, v in kwds_def.items()}
            func = node_data.get(NodeAttributes.FUNC, None)
            executor = node_data.get(NodeAttributes.EXECUTOR, None)
            converter = node_data.get(NodeAttributes.CONVERTER, None)
            new_node_name = self.prepend_path(node_name, base_path_nk)
            self.add_node(
                new_node_name,
                func,
                args=args_prepended,
                kwds=kwds_prepended,
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
                # The node has a concrete value — mark it serializable so the
                # value survives a JSON roundtrip even though the function is not.
                self._set_tag_one(new_node_key, SystemTags.SERIALIZE)
        if links is not None:
            for target, source in links.items():
                self.link(base_path_nk.join_parts(target), source)
        if metadata is not None:
            self._metadata[base_path_nk] = metadata
        else:
            if base_path_nk in self._metadata:
                del self._metadata[base_path_nk]

    @_notifies_subscribers(graph_changed=True)
    def link(self, target: Name, source: Name) -> None:
        """Create a link between two nodes in the computation graph."""
        target_nk = to_nodekey(target)
        source_nk = to_nodekey(source)
        if target_nk == source_nk:
            return

        target_style = self._style_one(target_nk) if self.has_node(target_nk) else None
        source_style = self._style_one(source_nk) if self.has_node(source_nk) else None
        style = target_style if target_style else source_style

        self.add_node(target_nk, identity_function, kwds={"x": source_nk}, style=style)

    @_notifies_subscribers()
    def inject_dependencies(self, dependencies: dict[Name, Any], *, force: bool = False) -> None:
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
