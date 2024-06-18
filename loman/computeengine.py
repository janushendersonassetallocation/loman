import logging
import os
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime
from enum import Enum

import inspect
import decorator
import dill
import networkx as nx
import pandas as pd
import types

from .consts import NodeAttributes, EdgeAttributes, SystemTags, States
from .graph_utils import contract_node
from .visualization import create_viz_dag, to_pydot
from .compat import get_signature
from .util import AttributeView, apply_n, apply1, as_iterable

LOG = logging.getLogger('loman.computeengine')


@dataclass
class Error:
    exception: Exception
    traceback: inspect.Traceback


@dataclass
class NodeData:
    state: States
    value: object


@dataclass
class TimingData:
    start: object
    end: object
    duration: object


class ComputationException(Exception):
    pass


class MapException(ComputationException):
    def __init__(self, message, results):
        super(MapException, self).__init__(message)
        self.results = results


class LoopDetectedException(ComputationException):
    pass


class NonExistentNodeException(ComputationException):
    pass

class NodeAlreadyExistsException(ComputationException):
    pass


class _ParameterType(Enum):
    ARG = 1
    KWD = 2


@dataclass
class _ParameterItem:
    type: object
    name: str
    value: object


def _node(func, *args, **kws):
    return func(*args, **kws)


def node(comp, name=None, *args, **kw):
    def inner(f):
        if name is None:
            comp.add_node(f.__name__, f, *args, **kw)
        else:
            comp.add_node(name, f, *args, **kw)
        return decorator.decorate(f, _node)
    return inner


class ConstantValue(object):
    def __init__(self, value):
        self.value = value


C = ConstantValue


class InputNode(object):
    def __init__(self, args, kwds):
        self.args = args
        self.kwds = kwds


def input_node(*args, **kwds):
    return InputNode(args, kwds)


class CalcNode(object):
    def __init__(self, f, args, kwds):
        self.f = f
        self.args = args
        self.kwds = kwds


def calc_node(f, *args, **kwds):
    return CalcNode(f, args, kwds)


class ComputationFactory(object):
    def __init__(self, definition_class):
        self.definition_class = definition_class
    def __call__(self, *args, **kwargs):
        return Computation(self.definition_class, *args, **kwargs)


def _eval_node(name, f, args, kwds, raise_exceptions):
    """ To make multiprocessing work, this function must be standalone so that pickle works """
    exc, tb = None, None
    start_dt = datetime.utcnow()
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
    end_dt = datetime.utcnow()
    return value, exc, tb, start_dt, end_dt


class Computation(object):
    def __init__(self, definition_class=None, default_executor=None, executor_map=None):
        """

        :param definition_class: A class with methods defining the nodes of the Computation
        :type definition_class: type
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
        self.v = AttributeView(self.nodes, self.value, self.value)
        self.s = AttributeView(self.nodes, self.state, self.state)
        self.i = AttributeView(self.nodes, self.get_inputs, self.get_inputs)
        self.o = AttributeView(self.nodes, self.get_outputs, self.get_outputs)
        self.t = AttributeView(self.nodes, self.tags, self.tags)
        self.tim = AttributeView(self.nodes, self.get_timing, self.get_timing)
        self.x = AttributeView(self.nodes, self.compute_and_get_value)
        self._tag_map = defaultdict(set)
        self._state_map = {state: set() for state in States}
        if definition_class is not None:
            self.add_nodes_from_class(definition_class)

    def add_node(self, name, func=None, **kwargs):
        """
        Adds or updates a node in a computation

        :param name: Name of the node to add. This may be any hashable object.
        :param func: Function to use to calculate the node if the node is a calculation node. By default, the input nodes to the function will be implied from the names of the function    parameters. For example, a parameter called ``a`` would be taken from the node called ``a``. This can be modified with the ``kwds`` parameter.
        :type func: Function, default None
        :param args: Specifies a list of nodes that will be used to populate arguments of the function positionally for a calculation node. e.g. If args is ``['a', 'b', 'c']`` then the function would be called with three parameters, taken from the nodes 'a', 'b' and 'c' respectively.
        :type args: List, default None
        :param kwds: Specifies a mapping from parameter name to the node that should be used to populate that parameter when calling the function for a calculation node. e.g. If args is ``{'x': 'a', 'y': 'b'}`` then the function would be called with parameters named 'x' and 'y', and their values would be taken from nodes 'a' and 'b' respectively. Each entry in the dictionary can be read as "take parameter [key] from node [value]".
        :type kwds: Dictionary, default None
        :param value: If given, the value is inserted into the node, and the node state set to UPTODATE.
        :type value: default None
        :param serialize: Whether the node should be serialized. Some objects cannot be serialized, in which case, set serialize to False
        :type serialize: boolean, default True
        :param inspect: Whether to use introspection to determine the arguments of the function, which can be slow. If this is not set, kwds and args must be set for the function to obtain parameters.
        :type inspect: boolean, default True
        :param group: Subgraph to render node in
        :type group: default None
        :param tags: Set of tags to apply to node
        :type tags: Iterable
        :param executor: Name of executor to run node on
        :type executor: string
        """
        LOG.debug('Adding node {}'.format(str(name)))
        args = kwargs.get('args', None)
        kwds = kwargs.get('kwds', None)
        has_value = 'value' in kwargs
        value = kwargs.get('value', None)
        serialize = kwargs.get('serialize', True)
        inspect = kwargs.get('inspect', True)
        group = kwargs.get('group', None)
        tags = kwargs.get('tags', [])
        executor = kwargs.get('executor', None)

        self.dag.add_node(name)
        pred_edges = [(p, name) for p in self.dag.predecessors(name)]
        self.dag.remove_edges_from(pred_edges)
        node = self.dag.nodes[name]

        self._set_state_and_value(name, States.UNINITIALIZED, None, require_old_state=False)

        node[NodeAttributes.TAG] = set()
        node[NodeAttributes.GROUP] = group
        node[NodeAttributes.ARGS] = {}
        node[NodeAttributes.KWDS] = {}
        node[NodeAttributes.FUNC] = None
        node[NodeAttributes.EXECUTOR] = executor

        if func:
            node[NodeAttributes.FUNC] = func
            args_count = 0
            if args:
                args_count = len(args)
                for i, arg in enumerate(args):
                    if isinstance(arg, ConstantValue):
                        node[NodeAttributes.ARGS][i] = arg.value
                    else:
                        input_vertex_name = arg
                        if not self.dag.has_node(input_vertex_name):
                            self.dag.add_node(input_vertex_name, **{NodeAttributes.STATE: States.PLACEHOLDER})
                            self._state_map[States.PLACEHOLDER].add(input_vertex_name)
                        self.dag.add_edge(input_vertex_name, name, **{EdgeAttributes.PARAM: (_ParameterType.ARG, i)})
            if inspect:
                signature = get_signature(func)
                param_names = set()
                if not signature.has_var_args:
                    param_names.update(signature.kwd_params[args_count:])
                if signature.has_var_kwds and kwds is not None:
                    param_names.update(kwds.keys())
                default_names = signature.default_params
            else:
                if kwds is None:
                    param_names = []
                else:
                    param_names = kwds.keys()
                default_names = []
            for param_name in param_names:
                value_source = kwds.get(param_name, param_name) if kwds else param_name
                if isinstance(value_source, ConstantValue):
                    node[NodeAttributes.KWDS][param_name] = value_source.value
                else:
                    in_node_name = value_source
                    if not self.dag.has_node(in_node_name):
                        if param_name in default_names:
                            continue
                        else:
                            self.dag.add_node(in_node_name, **{NodeAttributes.STATE: States.PLACEHOLDER})
                            self._state_map[States.PLACEHOLDER].add(in_node_name)
                    self.dag.add_edge(in_node_name, name, **{EdgeAttributes.PARAM: (_ParameterType.KWD, param_name)})

        if func or value is not None:
            self._set_descendents(name, States.STALE)
        if has_value:
            self._set_uptodate(name, value)
        if node[NodeAttributes.STATE] == States.UNINITIALIZED:
            self._try_set_computable(name)
        self.set_tag(name, tags)
        if serialize:
            self.set_tag(name, SystemTags.SERIALIZE)

    def add_nodes_from_class(self, cls):
        for name, node in inspect.getmembers(cls, lambda o: isinstance(o, InputNode)):
            self.add_node(name, *node.args, **node.kwds)
        for name, node in inspect.getmembers(cls, lambda o: isinstance(o, CalcNode)):
            self.add_node(name, node.f, *node.args, **node.kwds)

    def _refresh_maps(self):
        self._tag_map.clear()
        for state in States:
            self._state_map[state].clear()
        for name in self.nodes():
            self._state_map[self.state(name)].add(name)
            for tag in self.tags(name):
                self._tag_map[tag].add(name)

    def _set_tag_one(self, name, tag):
        self.dag.nodes[name][NodeAttributes.TAG].add(tag)
        self._tag_map[tag].add(name)

    def set_tag(self, name, tag):
        """
        Set tags on a node or nodes. Ignored if tags are already set.
        
        :param name: Node or nodes to set tag for
        :param tag: Tag to set
        """
        apply_n(self._set_tag_one, name, tag)

    def _clear_tag_one(self, name, tag):
        self.dag.nodes[name][NodeAttributes.TAG].discard(tag)
        self._tag_map[tag].discard(name)

    def clear_tag(self, name, tag):
        """
        Clear tag on a node or nodes. Ignored if tags are not set.

        :param name: Node or nodes to clear tags for
        :param tag: Tag to clear
        """
        apply_n(self._clear_tag_one, name, tag)

    def delete_node(self, name):
        """
        Delete a node from a computation

        When nodes are explicitly deleted with ``delete_node``, but are still depended on by other nodes, then they will be set to PLACEHOLDER status. In this case, if the nodes that depend on a PLACEHOLDER node are deleted, then the PLACEHOLDER node will also be deleted.

        :param name: Name of the node to delete. If the node does not exist, a ``NonExistentNodeException`` will be raised.
        """
        LOG.debug('Deleting node {}'.format(str(name)))

        if not self.dag.has_node(name):
            raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        if len(self.dag.succ[name]) == 0:
            preds = self.dag.predecessors(name)
            state = self.dag.nodes[name][NodeAttributes.STATE]
            self.dag.remove_node(name)
            self._state_map[state].remove(name)
            for n in preds:
                if self.dag.nodes[n][NodeAttributes.STATE] == States.PLACEHOLDER:
                    self.delete_node(n)
        else:
            self._set_state(name, States.PLACEHOLDER)

    def rename_node(self, old_name, new_name=None):
        """
        Rename a node in a computation
        :param old_name: Node to rename, or a dictionary of nodes to rename, with existing names as keys, and new names as values
        :param new_name: New name for node
        """
        if hasattr(old_name, '__getitem__') and not isinstance(old_name, str):
            for k, v in old_name.items():
                LOG.debug('Renaming node {} to {}'.format(str(k), str(v)))
            if new_name is not None:
                raise ValueError("new_name must not be set if rename_node is passed a dictionary")
            else:
                mapping = old_name
        else:
            LOG.debug('Renaming node {} to {}'.format(str(old_name), str(new_name)))
            if not self.dag.has_node(old_name):
                raise NonExistentNodeException('Node {} does not exist'.format(str(old_name)))
            if self.dag.has_node(new_name):
                raise NodeAlreadyExistsException('Node {} does not exist'.format(str(old_name)))
            mapping = {old_name: new_name}

        nx.relabel_nodes(self.dag, mapping, copy=False)

        self._refresh_maps()

    def repoint(self, old_name, new_name):
        """
        Changes all nodes that use old_name as an input to use new_name instead.

        Note that if old_name is an input to new_name, then that will not be changed, to try to avoid introducing
        circular dependencies, but other circular dependencies will not be checked.

        If new_name does not exist, then it will be created as a PLACEHOLDER node.

        :param old_name:
        :param new_name:
        :return:
        """
        if old_name == new_name:
            return

        changed_names = list(self.dag.successors(old_name))

        if len(changed_names) > 0 and not self.dag.has_node(new_name):
            self.dag.add_node(new_name, **{NodeAttributes.STATE: States.PLACEHOLDER})
            self._state_map[States.PLACEHOLDER].add(new_name)

        for name in changed_names:
            if name == new_name:
                continue
            edge_data = self.dag.get_edge_data(old_name, name)
            self.dag.add_edge(new_name, name, **edge_data)
            self.dag.remove_edge(old_name, name)

        for name in changed_names:
            self.set_stale(name)

    def insert(self, name, value, force=False):
        """
        Insert a value into a node of a computation

        Following insertation, the node will have state UPTODATE, and all its descendents will be COMPUTABLE or STALE.

        If an attempt is made to insert a value into a node that does not exist, a ``NonExistentNodeException`` will be raised.

        :param name: Name of the node to add.
        :param value: The value to be inserted into the node.
        :param force: Whether to force recalculation of descendents if node value and state would not be changed
        """
        LOG.debug('Inserting value into node {}'.format(str(name)))

        if not self.dag.has_node(name):
            raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        if not force:
            try:
                current_state, current_value = self.__getitem__(name)
                if current_state == States.UPTODATE and current_value == value:
                    return
            except:
                pass

        self._set_state_and_value(name, States.UPTODATE, value)
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

    def insert_many(self, name_value_pairs):
        """
        Insert values into many nodes of a computation simultaneously

        Following insertation, the nodes will have state UPTODATE, and all their descendents will be COMPUTABLE or STALE. In the case of inserting many nodes, some of which are descendents of others, this ensures that the inserted nodes have correct status, rather than being set as STALE when their ancestors are inserted.

        If an attempt is made to insert a value into a node that does not exist, a ``NonExistentNodeException`` will be raised, and none of the nodes will be inserted.

        :param name_value_pairs: Each tuple should be a pair (name, value), where name is the name of the node to insert the value into.
        :type name_value_pairs: List of tuples
        """
        LOG.debug('Inserting value into nodes {}'.format(", ".join(str(name) for name, value in name_value_pairs)))

        for name, value in name_value_pairs:
            if not self.dag.has_node(name):
                raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        stale = set()
        computable = set()
        for name, value in name_value_pairs:
            self._set_state_and_value(name, States.UPTODATE, value)
            stale.update(nx.dag.descendants(self.dag, name))
            computable.update(self.dag.successors(name))
        names = set([name for name, value in name_value_pairs])
        stale.difference_update(names)
        computable.difference_update(names)
        for name in stale:
            self._set_state(name, States.STALE)
        for name in computable:
            self._try_set_computable(name)

    def insert_from(self, other, nodes=None):
        """
        Insert values into another Computation object into this Computation object

        :param other: The computation object to take values from
        :type Computation:
        :param nodes: Only populate the nodes with the names provided in this list. By default, all nodes from the other Computation object that have corresponding nodes in this Computation object will be inserted
        :type nodes: List, default None
        """
        if nodes is None:
            nodes = set(self.dag.nodes)
            nodes.intersection_update(other.dag.nodes())
        name_value_pairs = [(name, other.value(name)) for name in nodes]
        self.insert_many(name_value_pairs)

    def _set_state(self, name, state):
        node = self.dag.nodes[name]
        old_state = node[NodeAttributes.STATE]
        self._state_map[old_state].remove(name)
        node[NodeAttributes.STATE] = state
        self._state_map[state].add(name)

    def _set_state_and_value(self, name, state, value, require_old_state=True):
        node = self.dag.nodes[name]
        try:
            old_state = node[NodeAttributes.STATE]
            self._state_map[old_state].remove(name)
        except KeyError:
            if require_old_state:
                raise
        node[NodeAttributes.STATE] = state
        node[NodeAttributes.VALUE] = value
        self._state_map[state].add(name)

    def _set_states(self, names, state):
        for name in names:
            node = self.dag.nodes[name]
            old_state = node[NodeAttributes.STATE]
            self._state_map[old_state].remove(name)
            node[NodeAttributes.STATE] = state
        self._state_map[state].update(names)

    def set_stale(self, name):
        """
        Set the state of a node and all its dependencies to STALE

        :param name: Name of the node to set as STALE.
        """
        names = [name]
        names.extend(nx.dag.descendants(self.dag, name))
        self._set_states(names, States.STALE)
        self._try_set_computable(name)

    def pin(self, name, value=None):
        """
        Set the state of a node to PINNED
        
        :param name: Name of the node to set as PINNED.
        :param value: Value to pin to the node, if provided.
        :type value: default None
        """
        if value is not None:
            self.insert(name, value)
        self._set_states(name, States.PINNED)

    def unpin(self, name):
        """
        Unpin a node (state of node and all descendents will be set to STALE)

        :param name: Name of the node to set as PINNED.
        """
        self.set_stale(name)

    def _get_descendents(self, name, stop_states=None):
        if self.dag.nodes[name][NodeAttributes.STATE] in stop_states:
            return set()
        if stop_states is None:
            stop_states = []
        visited = set()
        to_visit = {name}
        while to_visit:
            n = to_visit.pop()
            visited.add(n)
            for n1 in self.dag.successors(n):
                if n1 in visited:
                    continue
                if self.dag.nodes[n1][NodeAttributes.STATE] in stop_states:
                    continue
                to_visit.add(n1)
        visited.remove(name)
        return visited

    def _set_descendents(self, name, state):
        descendents = self._get_descendents(name, set([States.PINNED]))
        self._set_states(descendents, state)

    def _set_uninitialized(self, name):
        self._set_states([name], States.UNINITIALIZED)
        self.dag.nodes[name].pop(NodeAttributes.VALUE, None)

    def _set_uptodate(self, name, value):
        self._set_state_and_value(name, States.UPTODATE, value)
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

    def _set_error(self, name, error):
        self._set_state_and_value(name, States.ERROR, error)
        self._set_descendents(name, States.STALE)

    def _try_set_computable(self, name):
        if self.dag.nodes[name][NodeAttributes.STATE] == States.PINNED:
            return
        if self.dag.nodes[name].get(NodeAttributes.FUNC) is not None:
            for n in self.dag.predecessors(name):
                if not self.dag.has_node(n):
                    return
                if self.dag.nodes[n][NodeAttributes.STATE] != States.UPTODATE:
                    return
            self._set_state(name, States.COMPUTABLE)

    def _get_parameter_data(self, name):
        for arg, value in self.dag.nodes[name][NodeAttributes.ARGS].items():
            yield _ParameterItem(_ParameterType.ARG, arg, value)
        for param_name, value in self.dag.nodes[name][NodeAttributes.KWDS].items():
            yield _ParameterItem(_ParameterType.KWD, param_name, value)
        for in_node_name in self.dag.predecessors(name):
            param_value = self.dag.nodes[in_node_name][NodeAttributes.VALUE]
            edge = self.dag[in_node_name][name]
            param_type, param_name = edge[EdgeAttributes.PARAM]
            yield _ParameterItem(param_type, param_name, param_value)

    def _get_func_args_kwds(self, name):
        node0 = self.dag.nodes[name]
        f = node0[NodeAttributes.FUNC]
        executor_name = node0.get(NodeAttributes.EXECUTOR)
        args, kwds = [], {}
        for param in self._get_parameter_data(name):
            if param.type == _ParameterType.ARG:
                idx = param.name
                while len(args) <= idx:
                    args.append(None)
                args[idx] = param.value
            elif param.type == _ParameterType.KWD:
                kwds[param.name] = param.value
            else:
                raise Exception("Unexpected param type: {}".format(param.type))
        return f, executor_name, args, kwds

    def _compute_nodes(self, names, raise_exceptions=False):
        LOG.debug('Computing nodes {}'.format(list(map(str, names))))

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

        for name in names:
            node0 = self.dag.nodes[name]
            state = node0[NodeAttributes.STATE]
            if state == States.COMPUTABLE:
                run(name)

        while len(futs) > 0:
            done, not_done = wait(futs.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                name = futs.pop(fut)
                node0 = self.dag.nodes[name]
                value, exc, tb, start_dt, end_dt = fut.result()
                delta = (end_dt - start_dt).total_seconds()
                if exc is None:
                    self._set_state_and_value(name, States.UPTODATE, value)
                    node0[NodeAttributes.TIMING] = TimingData(start_dt, end_dt, delta)
                    self._set_descendents(name, States.STALE)
                    for n in self.dag.successors(name):
                        logging.debug(str(name) + ' ' + str(n) + ' ' + str(computed))
                        if n in computed:
                            raise LoopDetectedException("Calculating {} for the second time".format(name))
                        self._try_set_computable(n)
                        node0 = self.dag.nodes[n]
                        state = node0[NodeAttributes.STATE]
                        if state == States.COMPUTABLE and n in names:
                            run(n)
                else:
                    self._set_state_and_value(name, States.ERROR, Error(exc, tb))
                    self._set_descendents(name, States.STALE)
                computed.add(name)

    def _get_calc_nodes(self, name):
        g = nx.DiGraph()
        g.add_nodes_from(self.dag.nodes)
        g.add_edges_from(self.dag.edges)
        for n in nx.ancestors(g, name):
            node = self.dag.nodes[n]
            state = node[NodeAttributes.STATE]
            if state == States.UPTODATE or state == States.PINNED:
                g.remove_node(n)

        ancestors = nx.ancestors(g, name)
        for n in ancestors:
            if state == States.UNINITIALIZED and len(self.dag.pred[n]) == 0:
                raise Exception("Cannot compute {} because {} uninitialized".format(name, n))
            if state == States.PLACEHOLDER:
                raise Exception("Cannot compute {} because {} is placeholder".format(name, n))

        ancestors.add(name)
        nodes_sorted = nx.topological_sort(g)
        return [n for n in nodes_sorted if n in ancestors]

    def compute(self, name, raise_exceptions=False):
        """
        Compute a node and all necessary predecessors

        Following the computation, if successful, the target node, and all necessary ancestors that were not already UPTODATE will have been calculated and set to UPTODATE. Any node that did not need to be calculated will not have been recalculated.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an object containing the exception object, as well as a traceback. This will not halt the computation, which will proceed as far as it can, until no more nodes that would be required to calculate the target are COMPUTABLE.

        :param name: Name of the node to compute
        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """

        if isinstance(name, (types.GeneratorType, list)):
            calc_nodes = set()
            for name0 in name:
                for n in self._get_calc_nodes(name0):
                    calc_nodes.add(n)
        else:
            calc_nodes = self._get_calc_nodes(name)
        self._compute_nodes(calc_nodes, raise_exceptions=raise_exceptions)

    def compute_all(self, raise_exceptions=False):
        """Compute all nodes of a computation that can be computed

        Nodes that are already UPTODATE will not be recalculated. Following the computation, if successful, all nodes will have state UPTODATE, except UNINITIALIZED input nodes and PLACEHOLDER nodes.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an object containing the exception object, as well as a traceback. This will not halt the computation, which will proceed as far as it can, until no more nodes are COMPUTABLE.

        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """
        self._compute_nodes(self.nodes(), raise_exceptions=raise_exceptions)

    def nodes(self):
        """
        Get a list of nodes in this computation
        :return: List of nodes
        """
        return list(self.dag.nodes)

    def _state_one(self, name):
        return self.dag.nodes[name][NodeAttributes.STATE]

    def state(self, name):
        """
        Get the state of a node

        This can also be accessed using the attribute-style accessor ``s`` if ``name`` is a valid Python attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.state('foo')
            <States.UPTODATE: 4>
            >>> comp.s.foo
            <States.UPTODATE: 4>

        :param name: Name or names of the node to get state for
        :type name: Key or [Keys]
        """
        return apply1(self._state_one, name)

    def _value_one(self, name):
        return self.dag.nodes[name][NodeAttributes.VALUE]

    def value(self, name):
        """
        Get the current value of a node

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.value('foo')
            1
            >>> comp.v.foo
            1

        :param name: Name or names of the node to get the value of
        :type name: Key or [Keys]
        """
        return apply1(self._value_one, name)

    def compute_and_get_value(self, name):
        """
        Get the current value of a node

        This can also be accessed using the attribute-style accessor ``v`` if ``name`` is a valid Python attribute name::

            >>> comp = Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', lambda foo: foo + 1)
            >>> comp.compute_and_get_value('bar')
            2
            >>> comp.x.bar
            2

        :param name: Name or names of the node to get the value of
        :type name: Key or [Keys]
        """
        if self.state(name) == States.UPTODATE:
            return self.value(name)
        self.compute(name, raise_exceptions=True)
        if self.state(name) == States.UPTODATE:
            return self.value(name)
        raise ComputationException(f"Unable to compute node {name}")

    def _tag_one(self, name):
        node = self.dag.nodes[name]
        return node[NodeAttributes.TAG]

    def tags(self, name):
        """
        Get the tags associated with a node
        
            >>> comp = Computation()
            >>> comp.add_node('a', tags=['foo', 'bar'])
            >>> comp.t.a
            {'__serialize__', 'bar', 'foo'}
        :param name: Name or names of the node to get the tags of
        :return: 
        """
        return apply1(self._tag_one, name)

    def nodes_by_tag(self, tag):
        """
        Get the names of nodes with a particular tag or tags
        
        :param tag: Tag or tags for which to retrieve nodes 
        :return: Names of the nodes with those tags 
        """
        nodes = set()
        for tag1 in as_iterable(tag):
            nodes1 = self._tag_map.get(tag1)
            if nodes1 is not None:
                nodes.update(nodes1)
        return nodes

    def _get_item_one(self, name):
        node = self.dag.nodes[name]
        return NodeData(node[NodeAttributes.STATE], node[NodeAttributes.VALUE])

    def __getitem__(self, name):
        """
        Get the state and current value of a node

        :param name: Name of the node to get the state and value of
        """
        return apply1(self._get_item_one, name)

    def _get_timing_one(self, name):
        node = self.dag.nodes[name]
        return node.get(NodeAttributes.TIMING, None)

    def get_timing(self, name):
        """
        Get the timing information for a node
        
        :param name: Name or names of the node to get the timing information of
        :return: 
        """
        return apply1(self._get_timing_one, name)

    def to_df(self):
        """
        Get a dataframe containing the states and value of all nodes of computation

        ::

            >>> comp = loman.Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', value=2)
            >>> comp.to_df()
                           state  value  is_expansion
            bar  States.UPTODATE      2           NaN
            foo  States.UPTODATE      1           NaN
        """
        df = pd.DataFrame(index=nx.topological_sort(self.dag))
        df[NodeAttributes.STATE] = pd.Series(nx.get_node_attributes(self.dag, NodeAttributes.STATE))
        df[NodeAttributes.VALUE] = pd.Series(nx.get_node_attributes(self.dag, NodeAttributes.VALUE))
        df_timing = pd.DataFrame.from_dict(nx.get_node_attributes(self.dag, 'timing'), orient='index')
        df = pd.merge(df, df_timing, left_index=True, right_index=True, how='left')
        return df

    def to_dict(self):
        """
        Get a dictionary containing the values of all nodes of a computation

        ::

            >>> comp = loman.Computation()
            >>> comp.add_node('foo', value=1)
            >>> comp.add_node('bar', value=2)
            >>> comp.to_dict()
            {'bar': 2, 'foo': 1}
        """
        return nx.get_node_attributes(self.dag, NodeAttributes.VALUE)

    def _get_inputs_one(self, name):
        args_dict = {}
        kwds = []
        max_arg_index = -1
        for input_node in self.dag.predecessors(name):
            input_edge = self.dag[input_node][name]
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

    def get_inputs(self, name):
        """
        Get a list of the inputs for a node or set of nodes
        
        :param name: Name or names of nodes to get inputs for 
        :return: If name is scalar, return a list of upstream nodes used as input. If name is a list, return a list of list of inputs.
        """
        return apply1(self._get_inputs_one, name)

    def get_ancestors(self, names, include_self=True):
        ancestors = set()
        for n in as_iterable(names):
            if include_self:
                ancestors.add(n)
            for ancestor in nx.ancestors(self.dag, n):
                ancestors.add(ancestor)
        return ancestors

    def get_original_inputs(self, names=None):
        """
        Get a list of the original non-computed inputs for a node or set of nodes

        :param names: Name or names of nodes to get inputs for
        :return: Return a list of original non-computed inputs that are ancestors of the input nodes
        """
        if names is None:
            nodes = self.nodes()
        else:
            nodes = self.get_ancestors(names)
        return [n for n in nodes if self.dag.nodes[n].get(NodeAttributes.FUNC) is None]

    def _get_outputs_one(self, name):
        return list(self.dag.successors(name))

    def get_outputs(self, name):
        """
        Get a list of the outputs for a node or set of nodes

        :param name: Name or names of nodes to get outputs for
        :return: If name is scalar, return a list of downstream nodes used as output. If name is a list, return a list of list of outputs.

        """
        return apply1(self._get_outputs_one, name)

    def get_descendents(self, names, include_self=True):
        ancestors = set()
        for n in as_iterable(names):
            if include_self:
                ancestors.add(n)
            for ancestor in nx.descendants(self.dag, n):
                ancestors.add(ancestor)
        return ancestors

    def get_final_outputs(self, names=None):
        if names is None:
            nodes = self.nodes()
        else:
            nodes = self.get_descendents(names)
        return [n for n in nodes if len(nx.descendants(self.dag, n))==0]

    def restrict(self, output_nodes, input_nodes=None):
        """
        Restrict a computation to the ancestors of a set of output nodes, excluding ancestors of a set of input nodes

        If the set of input_nodes that is specified is not sufficient for the set of output_nodes then additional nodes that are ancestors of the output_nodes will be included, but the input nodes specified will be input nodes of the modified Computation.

        :param output_nodes:
        :param input_nodes:
        :return: None - modifies existing computation in place
        """
        if input_nodes is not None:
            for n in input_nodes:
                nodedata = self._get_item_one(n)
                self.add_node(n)
                self._set_state_and_value(n, nodedata.state, nodedata.value)
        nodes = self.get_ancestors(output_nodes)
        self.dag.remove_nodes_from([n for n in self.dag if n not in nodes])

    def write_dill(self, file_):
        """
        Serialize a computation to a file or file-like object

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
        obj = self.copy()
        obj.executor_map = None
        obj.default_executor = None
        for name, tags in node_serialize.items():
            if SystemTags.SERIALIZE not in tags:
                obj._set_uninitialized(name)

        if isinstance(file_, str):
            with open(file_, 'wb') as f:
                dill.dump(obj, f)
        else:
            dill.dump(obj, file_)

    @staticmethod
    def read_dill(file_):
        """
        Deserialize a computation from a file or file-like object

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        if isinstance(file_, str):
            with open(file_, 'rb') as f:
                return dill.load(f)
        else:
            return dill.load(file_)

    def copy(self):
        """
        Create a copy of a computation

        The copy is shallow. Any values in the new Computation's DAG will be the same object as this Computation's DAG. As new objects will be created by any further computations, this should not be an issue.

        :rtype: Computation
        """
        obj = Computation()
        obj.dag = nx.DiGraph(self.dag)
        obj._tag_map = {tag: nodes.copy() for tag, nodes in self._tag_map.items()}
        obj._state_map = {state: nodes.copy() for state, nodes in self._state_map.items()}
        return obj

    def add_named_tuple_expansion(self, name, namedtuple_type, group=None):
        """
        Automatically add nodes to extract each element of a named tuple type

        It is often convenient for a calculation to return multiple values, and it is polite to do this a namedtuple rather than a regular tuple, so that later users have same name to identify elements of the tuple. It can also help make a computation clearer if a downstream computation depends on one element of such a tuple, rather than the entire tuple. This does not affect the computation per se, but it does make the intention clearer.

        To avoid having to create many boiler-plate node definitions to expand namedtuples, the ``add_named_tuple_expansion`` method automatically creates new nodes for each element of a tuple. The convention is that an element called 'element', in a node called 'node' will be expanded into a new node called 'node.element', and that this will be applied for each element.

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
        def make_f(field):
            def get_field_value(tuple):
                return getattr(tuple, field)
            return get_field_value
        for field in namedtuple_type._fields:
            node_name = "{}.{}".format(name, field)
            self.add_node(node_name, make_f(field), kwds={'tuple': name}, group=group)
            self.set_tag(node_name, SystemTags.EXPANSION)

    def add_map_node(self, result_node, input_node, subgraph, subgraph_input_node, subgraph_output_node):
        """
        Apply a graph to each element of iterable

        In turn, each element in the ``input_node`` of this graph will be inserted in turn into the subgraph's ``subgraph_input_node``, then the subgraph's ``subgraph_output_node`` calculated. The resultant list, with an element or each element in ``input_node``, will be inserted into ``result_node`` of this graph. In this way ``add_map_node`` is similar to ``map`` in functional programming.

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
                raise MapException("Unable to calculate {}".format(result_node), results)
            return results
        self.add_node(result_node, f, kwds={'xs': input_node})

    def _repr_svg_(self):
        return self.to_pydot().create_svg().decode('utf-8')

    def to_pydot(self, colors='state', cmap=None, graph_attr=None, node_attr=None, edge_attr=None, show_expansion=False):
        struct_dag = nx.DiGraph(self.dag)
        if not show_expansion:
            hide_nodes = set(struct_dag.nodes())
            for name1, name2 in struct_dag.edges():
                if not show_expansion and SystemTags.EXPANSION in self.tags(name2):
                    continue
                hide_nodes.discard(name1)
                hide_nodes.discard(name2)
            contract_node(struct_dag, hide_nodes)
        viz_dag = create_viz_dag(struct_dag, colors=colors, cmap=cmap)
        viz_dot = to_pydot(viz_dag, graph_attr, node_attr, edge_attr)
        return viz_dot

    def draw(self, colors='state', cmap=None, graph_attr=None, node_attr=None, edge_attr=None, show_expansion=False):
        """
        Draw a computation's current state using the GraphViz utility

        :param graph_attr: Mapping of (attribute, value) pairs for the graph. For example ``graph_attr={'size': '"10,8"'}`` can control the size of the output graph
        :param node_attr: Mapping of (attribute, value) pairs set for all nodes.
        :param edge_attr: Mapping of (attribute, value) pairs set for all edges.
        :param show_expansion: Whether to show expansion nodes (i.e. named tuple expansion nodes) if they are not referenced by other nodes
        """
        d = self.to_pydot(colors=colors, cmap=cmap, graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr,
                          show_expansion=show_expansion)

        def repr_svg(self):
            return self.create_svg().decode('utf-8')

        d._repr_svg_ = types.MethodType(repr_svg, d)
        return d

    def view(self, colors='state', cmap=None):
        d = self.to_pydot(colors=colors, cmap=cmap)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(d.create_pdf())
            os.startfile(f.name)

    def print_errors(self):
        """
        Print tracebacks for every node with state "ERROR" in a Computation 
        """
        for n in self.nodes():
            if self.s[n] == States.ERROR:
                print("{}".format(n))
                print("=" * len(n))
                print()
                print(self.v[n].traceback)
                print()


