"""Node state transitions for the computation engine.

Every change to a node's state goes through this mixin, which keeps
``dag.nodes[...][NodeAttributes.STATE]`` and the ``_state_map`` index in step.
"""

import traceback
from collections.abc import Iterable

import networkx as nx

from .base import ComputationBase
from .consts import NodeAttributes, States
from .nodekey import Name, NodeKey, to_nodekey
from .values import Error


class StateMixin(ComputationBase):
    """State transitions and the state index they maintain."""

    def _refresh_maps(self) -> None:
        """Refresh internal tag and state maps from node data."""
        self._tag_map.clear()
        for state in States:
            self._state_map[state].clear()
        for node_key in self.dag.nodes:
            state = self.dag.nodes[node_key][NodeAttributes.STATE]
            self._state_map[state].add(node_key)
            tags = self.dag.nodes[node_key].get(NodeAttributes.TAG, set())
            for tag in tags:
                self._tag_map[tag].add(node_key)

    def _set_state(self, node_key: NodeKey, state: States) -> None:
        """Set the state of a node without changing its value."""
        node = self.dag.nodes[node_key]
        old_state = node[NodeAttributes.STATE]
        self._state_map[old_state].remove(node_key)
        node[NodeAttributes.STATE] = state
        self._state_map[state].add(node_key)

    def _set_state_and_value(
        self, node_key: NodeKey, state: States, value: object, *, throw_conversion_exception: bool = True
    ) -> None:
        """Set state and value of a node, applying any converter."""
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
                    raise

    def _set_state_and_literal_value(
        self, node_key: NodeKey, state: States, value: object, require_old_state: bool = True
    ) -> None:
        """Set state and literal value of a node without conversion."""
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

    def _set_states(self, node_keys: Iterable[NodeKey], state: States) -> None:
        """Set the state of multiple nodes at once."""
        for name in node_keys:
            node = self.dag.nodes[name]
            old_state = node[NodeAttributes.STATE]
            self._state_map[old_state].remove(name)
            node[NodeAttributes.STATE] = state
        self._state_map[state].update(node_keys)

    def set_stale(self, name: Name) -> None:
        """Set the state of a node and all its dependencies to STALE.

        :param name: Name of the node to set as STALE.
        """
        node_key = to_nodekey(name)
        node_keys: list[NodeKey] = [node_key]
        node_keys.extend(nx.dag.descendants(self.dag, node_key))
        self._set_states(node_keys, States.STALE)
        self._try_set_computable(node_key)

    def unpin(self, name: Name) -> None:
        """Unpin a node (state of node and all descendents will be set to STALE).

        :param name: Name of the node to set as PINNED.
        """
        node_key = to_nodekey(name)
        self.set_stale(node_key)

    def _get_descendents(self, node_key: NodeKey, stop_states: set[States] | None = None) -> set[NodeKey]:
        """Get all descendant nodes, optionally stopping at certain states."""
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

    def _set_descendents(self, node_key: NodeKey, state: States) -> None:
        """Set the state of all descendant nodes."""
        descendents = self._get_descendents(node_key, {States.PINNED})
        self._set_states(descendents, state)

    def _set_uninitialized(self, node_key: NodeKey) -> None:
        """Set a node to uninitialized state and clear its value."""
        self._set_states([node_key], States.UNINITIALIZED)
        self.dag.nodes[node_key].pop(NodeAttributes.VALUE, None)

    def _set_uptodate(self, node_key: NodeKey, value: object) -> None:
        """Set a node to up-to-date state with a value."""
        self._set_state_and_value(node_key, States.UPTODATE, value)
        self._set_descendents(node_key, States.STALE)
        for n in self.dag.successors(node_key):
            self._try_set_computable(n)

    def _set_error(self, node_key: NodeKey, exc: Exception, tb: str) -> None:
        """Set a node to error state with exception information."""
        self._set_state_and_literal_value(node_key, States.ERROR, Error(exc, tb))
        self._set_descendents(node_key, States.STALE)

    def _try_set_computable(self, node_key: NodeKey) -> None:
        """Set node to computable if all predecessors are up-to-date."""
        if self.dag.nodes[node_key][NodeAttributes.STATE] == States.PINNED:
            return
        if self.dag.nodes[node_key].get(NodeAttributes.FUNC) is not None:
            for n in self.dag.predecessors(node_key):
                if not self.dag.has_node(n):
                    return  # pragma: no cover
                if self.dag.nodes[n][NodeAttributes.STATE] not in (States.UPTODATE, States.PINNED):
                    return
            self._set_state(node_key, States.COMPUTABLE)
