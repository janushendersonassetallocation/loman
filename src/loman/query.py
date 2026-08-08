"""Read-only accessors over a computation graph.

Nothing in this mixin mutates the graph. It covers node listing and tree
navigation, per-node state/value/timing lookups, the tabular and dictionary
exports, and the input/output/ancestor/descendent relations.
"""

import inspect
from collections.abc import Iterable
from typing import Any, overload

import networkx as nx
import pandas as pd

from .consts import EdgeAttributes, NodeAttributes, States
from .graph_utils import topological_sort
from .nodekey import Name, Names, NodeKey, names_to_node_keys, node_keys_to_names, to_nodekey
from .state import StateMixin
from .util import apply1
from .values import NodeData, TimingData, _ParameterType


class QueryMixin(StateMixin):
    """Read-only queries over the computation graph."""

    def _get_names_for_state(self, state: States) -> set[Name]:
        """Get node names that have a specific state."""
        return set(node_keys_to_names(self._state_map[state]))

    def _node_keys(self) -> list[NodeKey]:
        """Get a list of nodes in this computation.

        :return: List of nodes.
        """
        return list(self.dag.nodes)

    def nodes(self) -> list[Name]:
        """Get a list of nodes in this computation.

        :return: List of nodes.
        """
        return [n.name for n in self.dag.nodes]

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

    def has_node(self, name: Name) -> bool:
        """Check if a node with the given name exists in the computation."""
        node_key = to_nodekey(name)
        return node_key in self.dag.nodes

    def tree_has_path(self, name: Name) -> bool:
        """Check if a hierarchical path exists in the computation tree."""
        node_key = to_nodekey(name)
        if node_key.is_root:
            return True
        if self.has_node(node_key):
            return True
        return any(n.is_descendent_of(node_key) for n in self.dag.nodes)

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
                nodes = [n] if graph_nodes_only else n.ancestors()
                for n2 in nodes:
                    if n2.is_descendent_of(node_key):
                        nm = n2.name if include_stem else NodeKey(tuple(n2.parts[stemsize:])).name
                        result.add(nm)
        return result

    def _state_one(self, name: Name) -> States:
        """Get the state of a single node."""
        node_key = to_nodekey(name)
        state: States = self.dag.nodes[node_key][NodeAttributes.STATE]
        return state

    @overload
    def state(self, name: Name) -> States: ...

    @overload
    def state(self, name: Names) -> list[States]: ...

    def state(self, name: Name | Names) -> States | list[States]:
        """Get the state of a node.

        This can also be accessed using the attribute-style accessor ``s`` if ``name`` is a valid Python
        attribute name::

            >>> from loman import Computation
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

    def _value_one(self, name: Name) -> Any:
        """Get the value of a single node."""
        node_key = to_nodekey(name)
        return self.dag.nodes[node_key][NodeAttributes.VALUE]

    @overload
    def value(self, name: Name) -> Any: ...

    @overload
    def value(self, name: Names) -> list[Any]: ...

    def value(self, name: Name | Names) -> Any | list[Any]:
        """Get the current value of a node.

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python
        attribute name::

            >>> from loman import Computation
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

    def _get_item_one(self, name: Name) -> NodeData:
        """Get state and value data for a single node."""
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        return NodeData(node[NodeAttributes.STATE], node[NodeAttributes.VALUE])

    @overload
    def __getitem__(self, name: Name) -> NodeData: ...

    @overload
    def __getitem__(self, name: Names) -> list[NodeData]: ...

    def __getitem__(self, name: Name | Names) -> NodeData | list[NodeData]:
        """Get the state and current value of a node.

        :param name: Name of the node to get the state and value of
        """
        return apply1(self._get_item_one, name)

    def _get_timing_one(self, name: Name) -> TimingData | None:
        """Get timing data for a single node."""
        node_key = to_nodekey(name)
        node = self.dag.nodes[node_key]
        timing: TimingData | None = node.get(NodeAttributes.TIMING, None)
        return timing

    @overload
    def get_timing(self, name: Name) -> TimingData | None: ...

    @overload
    def get_timing(self, name: Names) -> list[TimingData | None]: ...

    def get_timing(self, name: Name | Names) -> TimingData | None | list[TimingData | None]:
        """Get the timing information for a node.

        :param name: Name or names of the node to get the timing information of
        :return:
        """
        return apply1(self._get_timing_one, name)

    def to_df(self) -> pd.DataFrame:
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

    def to_dict(self) -> dict[NodeKey, Any]:
        """Get a dictionary containing the values of all nodes of a computation.

        ::

            >>> import loman
            >>> comp = loman.Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', value=2)
            >>> comp.to_dict()  # doctest: +ELLIPSIS
            {NodeKey('foo'): 1, NodeKey('bar'): 2}
        """
        result: dict[NodeKey, Any] = nx.get_node_attributes(self.dag, NodeAttributes.VALUE)
        return result

    def _get_inputs_one_node_keys(self, node_key: NodeKey) -> list[NodeKey | None]:
        """Get input node keys for a single node."""
        args_dict: dict[int, NodeKey] = {}
        kwds: list[NodeKey | None] = []
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
            args: list[NodeKey | None] = [None] * (max_arg_index + 1)
            for idx, input_node in args_dict.items():
                args[idx] = input_node
            result: list[NodeKey | None] = args + kwds
            return result
        else:
            return kwds

    def _get_inputs_one_names(self, name: Name) -> Names:
        """Get input node names for a single node."""
        node_key = to_nodekey(name)
        return node_keys_to_names([nk for nk in self._get_inputs_one_node_keys(node_key) if nk is not None])

    @overload
    def get_inputs(self, name: Name) -> Names: ...

    @overload
    def get_inputs(self, name: Names) -> list[Names]: ...

    def get_inputs(self, name: Name | Names) -> Names | list[Names]:
        """Get a list of the inputs for a node or set of nodes.

        :param name: Name or names of nodes to get inputs for
        :return: If name is scalar, return a list of upstream nodes used as input. If name is a list, return a
            list of list of inputs.
        """
        return apply1(self._get_inputs_one_names, name)

    def _get_ancestors_node_keys(self, node_keys: Iterable[NodeKey], include_self: bool = True) -> set[NodeKey]:
        """Get all ancestor node keys for a set of nodes."""
        ancestors: set[NodeKey] = set()
        for n in node_keys:
            if include_self:
                ancestors.add(n)
            for ancestor in nx.ancestors(self.dag, n):
                ancestors.add(ancestor)
        return ancestors

    def get_ancestors(self, names: Name | Names, include_self: bool = True) -> Names:
        """Get all ancestor nodes of the specified nodes."""
        node_keys = names_to_node_keys(names)
        ancestor_node_keys = self._get_ancestors_node_keys(node_keys, include_self)
        return node_keys_to_names(ancestor_node_keys)

    def _get_original_inputs_node_keys(self, node_keys: list[NodeKey] | None) -> list[NodeKey]:
        """Get original input node keys that have no computation function."""
        resolved_node_keys: Iterable[NodeKey]
        resolved_node_keys = self._node_keys() if node_keys is None else self._get_ancestors_node_keys(node_keys)
        return [n for n in resolved_node_keys if self.dag.nodes[n].get(NodeAttributes.FUNC) is None]

    def get_original_inputs(self, names: Name | Names | None = None) -> Names:
        """Get a list of the original non-computed inputs for a node or set of nodes.

        :param names: Name or names of nodes to get inputs for
        :return: Return a list of original non-computed inputs that are ancestors of the input nodes
        """
        nks = None if names is None else names_to_node_keys(names)

        result_nks = self._get_original_inputs_node_keys(nks)

        return node_keys_to_names(result_nks)

    def _get_outputs_one(self, name: Name) -> Names:
        """Get output node names for a single node."""
        node_key = to_nodekey(name)
        output_node_keys = list(self.dag.successors(node_key))
        return node_keys_to_names(output_node_keys)

    @overload
    def get_outputs(self, name: Name) -> Names: ...

    @overload
    def get_outputs(self, name: Names) -> list[Names]: ...

    def get_outputs(self, name: Name | Names) -> Names | list[Names]:
        """Get a list of the outputs for a node or set of nodes.

        :param name: Name or names of nodes to get outputs for
        :return: If name is scalar, return a list of downstream nodes used as output. If name is a list, return a
            list of list of outputs.

        """
        return apply1(self._get_outputs_one, name)

    def _get_descendents_node_keys(self, node_keys: Iterable[NodeKey], include_self: bool = True) -> set[NodeKey]:
        """Get all descendant node keys for a set of nodes."""
        descendent_node_keys: set[NodeKey] = set()
        for node_key in node_keys:
            if include_self:
                descendent_node_keys.add(node_key)
            for descendent in nx.descendants(self.dag, node_key):
                descendent_node_keys.add(descendent)
        return descendent_node_keys

    def get_descendents(self, names: Name | Names, include_self: bool = True) -> Names:
        """Get all descendent nodes of the specified nodes."""
        node_keys = names_to_node_keys(names)
        descendent_node_keys = self._get_descendents_node_keys(node_keys, include_self)
        return node_keys_to_names(descendent_node_keys)

    def get_final_outputs(self, names: Name | Names | None = None) -> Names:
        """Get final output nodes (nodes with no descendants) from the specified nodes."""
        final_node_keys: Iterable[NodeKey]
        if names is None:
            final_node_keys = self._node_keys()
        else:
            nks = names_to_node_keys(names)
            final_node_keys = self._get_descendents_node_keys(nks)
        output_node_keys = [n for n in final_node_keys if len(nx.descendants(self.dag, n)) == 0]
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

    def print_source(self, name: Name) -> None:
        """Print the source code for a computation node."""
        print(self.get_source(name))
