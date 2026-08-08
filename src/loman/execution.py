"""Scheduling and running the calculations in a computation graph.

This mixin turns node definitions into calls: it assembles each function's
arguments from its predecessors, works out which nodes a target depends on,
and drives the executors that run them.
"""

import logging
import traceback
import types
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from .attributes import AttributeMixin
from .consts import EdgeAttributes, NodeAttributes, States
from .exception import ComputationError, LoopDetectedException, ValidationError
from .graph_utils import topological_sort
from .nodekey import Name, Names, NodeKey, names_to_node_keys, node_keys_to_names, to_nodekey
from .subscription import _notifies_subscribers
from .values import C, TimingData, _ParameterItem, _ParameterType

# Deliberately the pre-split logger name: this refactor moved code between
# modules and should not silently rename the channel people filter on.
LOG = logging.getLogger("loman.computeengine")


def _eval_node(
    name: NodeKey,
    f: Callable[..., Any],
    args: list[Any],
    kwds: dict[str, Any],
    raise_exceptions: bool,
) -> tuple[Any, Exception | None, str | None, datetime, datetime]:
    """To make multiprocessing work, this function must be standalone so that pickle works."""
    exc: Exception | None = None
    tb: str | None = None
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


class ExecutionMixin(AttributeMixin):
    """Assembling function calls and running them."""

    def _get_parameter_data(self, node_key: NodeKey) -> Iterable[_ParameterItem]:
        """Get all parameter data for a node's function call."""
        for arg, value in self.dag.nodes[node_key][NodeAttributes.ARGS].items():
            yield _ParameterItem(_ParameterType.ARG, arg, value)
        for param_name, value in self.dag.nodes[node_key][NodeAttributes.KWDS].items():
            yield _ParameterItem(_ParameterType.KWD, param_name, value)
        for in_node_name in self.dag.predecessors(node_key):
            param_value = self.dag.nodes[in_node_name][NodeAttributes.VALUE]
            edge = self.dag[in_node_name][node_key]
            param_type, param_name = edge[EdgeAttributes.PARAM]
            yield _ParameterItem(param_type, param_name, param_value)

    def _get_func_args_kwds(
        self, node_key: NodeKey
    ) -> tuple[Callable[..., Any], str | None, list[Any], dict[str, Any]]:
        """Get function, executor name, args and kwargs for a node."""
        node0 = self.dag.nodes[node_key]
        f = node0[NodeAttributes.FUNC]
        executor_name = node0.get(NodeAttributes.EXECUTOR)
        args: list[Any] = []
        kwds: dict[str, Any] = {}
        for param in self._get_parameter_data(node_key):
            if param.type == _ParameterType.ARG:
                idx = param.name
                assert isinstance(idx, int)  # noqa: S101
                while len(args) <= idx:
                    args.append(None)
                args[idx] = param.value
            elif param.type == _ParameterType.KWD:
                assert isinstance(param.name, str)  # noqa: S101
                kwds[param.name] = param.value
            else:  # pragma: no cover
                msg = f"Unexpected param type: {param.type}"
                raise ValidationError(msg)
        return f, executor_name, args, kwds

    def get_definition_args_kwds(self, name: Name) -> tuple[list[Any], dict[str, Any]]:
        """Get the arguments and keyword arguments for a node's function definition."""
        res_args: list[Any] = []
        res_kwds: dict[str, Any] = {}
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
                    idx = param_name
                    assert isinstance(idx, int)  # noqa: S101
                    while len(res_args) <= idx:
                        res_args.append(None)
                    res_args[idx] = in_node_key.name
                elif param_type == _ParameterType.KWD:
                    res_kwds[param_name] = in_node_key.name
                else:  # pragma: no cover
                    msg = f"Unexpected param type: {param_type}"
                    raise ValidationError(msg)
        return res_args, res_kwds

    def _compute_nodes(self, node_keys: Iterable[NodeKey], raise_exceptions: bool = False) -> None:
        """Compute multiple nodes, handling dependencies and parallel execution."""
        LOG.debug(f"Computing nodes {node_keys}")

        futs: dict[Any, NodeKey] = {}
        node_keys_set = set(node_keys)

        def run(name: NodeKey) -> None:
            """Submit a node computation to an executor."""
            f, executor_name, args, kwds = self._get_func_args_kwds(name)
            executor = self.default_executor if executor_name is None else self.executor_map[executor_name]
            fut = executor.submit(_eval_node, name, f, args, kwds, raise_exceptions)
            futs[fut] = name

        computed: set[NodeKey] = set()

        for node_key in node_keys_set:
            node0 = self.dag.nodes[node_key]
            state = node0[NodeAttributes.STATE]
            if state == States.COMPUTABLE:
                run(node_key)

        while len(futs) > 0:
            done, _not_done = wait(futs.keys(), return_when=FIRST_COMPLETED)
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
                            msg = f"Calculating {node_key} for the second time"
                            raise LoopDetectedException(msg)
                        self._try_set_computable(n)
                        node0 = self.dag.nodes[n]
                        state = node0[NodeAttributes.STATE]
                        if state == States.COMPUTABLE and n in node_keys_set:
                            run(n)
                else:
                    assert tb is not None  # noqa: S101
                    self._set_error(node_key, exc, tb)
                computed.add(node_key)

    def _get_calc_node_keys(self, node_key: NodeKey) -> list[NodeKey]:
        """Get node keys that need to be computed for a target node."""
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
            node = self.dag.nodes[n]
            state = node[NodeAttributes.STATE]

            if state == States.UNINITIALIZED and len(self.dag.pred[n]) == 0:
                msg = f"Cannot compute {node_key} because {n} uninitialized"
                raise ValidationError(msg)
            if state == States.PLACEHOLDER:
                msg = f"Cannot compute {node_key} because {n} is placeholder"
                raise ValidationError(msg)

        ancestors.add(node_key)
        g = g.subgraph(ancestors).copy()
        nodes_sorted = topological_sort(g)
        return list(nodes_sorted)

    def _get_calc_node_names(self, name: Name) -> Names:
        """Get node names that need to be computed for a target node."""
        node_key = to_nodekey(name)
        return node_keys_to_names(self._get_calc_node_keys(node_key))

    @_notifies_subscribers()
    def compute(self, name: Name | Iterable[Name], raise_exceptions: bool = False) -> None:
        """Compute a node or block and all necessary predecessors.

        Following the computation, if successful, the target node, and all necessary ancestors that were not already
        UPTODATE will have been calculated and set to UPTODATE. Any node that did not need to be calculated will not
        have been recalculated.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an
        object containing the exception object, as well as a traceback. This will not halt the computation, which
        will proceed as far as it can, until no more nodes that would be required to calculate the target are
        COMPUTABLE.

        A block name computes every node below that path. Multiple node and block
        names may be supplied in a list or generator.

        :param name: Name of the node or block to compute
        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """
        calc_nodes: set[NodeKey] = set()
        names = name if isinstance(name, (types.GeneratorType, list)) else [name]
        for name0 in names:
            node_key = to_nodekey(name0)
            targets = (
                [node_key]
                if self.has_node(node_key)
                else names_to_node_keys(self.get_tree_descendents(node_key, graph_nodes_only=True))
            )
            if not targets:
                targets = [node_key]
            for target in targets:
                calc_nodes.update(self._get_calc_node_keys(target))
        self._compute_nodes(calc_nodes, raise_exceptions=raise_exceptions)

    @_notifies_subscribers()
    def compute_all(self, raise_exceptions: bool = False) -> None:
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

    def compute_and_get_value(self, name: Name) -> Any:
        """Get the current value of a node.

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python
        attribute name::

            >>> from loman import Computation
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
        nk = to_nodekey(name)
        if self.state(nk) == States.UPTODATE:
            return self.value(nk)
        self.compute(nk, raise_exceptions=True)
        if self.state(nk) == States.UPTODATE:
            return self.value(nk)
        msg = f"Unable to compute node {nk}"
        raise ComputationError(msg)
