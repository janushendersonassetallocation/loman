import networkx as nx
from enum import Enum
from collections import OrderedDict, deque, namedtuple
import inspect
import decorator
import dill
import six
import pandas as pd
import traceback
import graphviz
import logging


LOG = logging.getLogger('loman.computeengine')


class States(Enum):
    """Possible states for a computation node"""
    PLACEHOLDER = 0
    UNINITIALIZED = 1
    STALE = 2
    COMPUTABLE = 3
    UPTODATE = 4
    ERROR = 5


state_colors = {
    None: '#ffffff', # xkcd white
    States.PLACEHOLDER: '#f97306', # xkcd orange
    States.UNINITIALIZED: '#0343df', # xkcd blue
    States.STALE: '#ffff14', # xkcd yellow
    States.COMPUTABLE: '#9dff00', # xkcd bright yellow green
    States.UPTODATE: '#15b01a', # xkcd green
    States.ERROR: '#e50000'
}

Error = namedtuple('Error', ['exception', 'traceback'])
NodeData = namedtuple('NodeData', ['state', 'value'])


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


class _ParameterType(Enum):
    ARG = 1
    KWD = 2

_ParameterItem = namedtuple('ParameterItem', ['type', 'name', 'value'])


class _ComputationAttributeView(object):
    def __init__(self, get_attribute_list, get_attribute, get_item=None):
        self.get_attribute_list = get_attribute_list
        self.get_attribute = get_attribute
        self.get_item = get_item
        if self.get_item is None:
            self.get_item = get_attribute

    def __dir__(self):
        return self.get_attribute_list()

    def __getattr__(self, attr):
        try:
            return self.get_attribute(attr)
        except KeyError:
            raise AttributeError()

    def __getitem__(self, key):
        return self.get_item(key)

    def __getstate__(self):
        return {
            'get_attribute_list': self.get_attribute_list,
            'get_attribute': self.get_attribute,
            'get_item': self.get_item
        }

    def __setstate__(self, state):
        self.get_attribute_list = state['get_attribute_list']
        self.get_attribute = state['get_attribute']
        self.get_item = state['get_item']

_Signature = namedtuple('_Signature', ['kwd_params', 'has_var_args', 'has_var_kwds'])


def _get_signature(func):
    if six.PY3:
        sig = inspect.signature(func)
        pk = inspect._ParameterKind
        has_var_args = any(p.kind == pk.VAR_POSITIONAL for p in sig.parameters.values())
        has_var_kwds = any(p.kind == pk.VAR_KEYWORD for p in sig.parameters.values())
        all_keyword_params = [param_name for param_name, param in sig.parameters.items()
                              if param.kind in (pk.POSITIONAL_OR_KEYWORD, pk.KEYWORD_ONLY)]
    elif six.PY2:
        argspec = inspect.getargspec(func)
        has_var_args = argspec.varargs is not None
        has_var_kwds = argspec.keywords is not None
        all_keyword_params = argspec.args
    else:
        raise Exception("Only Pythons 2 and 3 supported")
    return _Signature(all_keyword_params, has_var_args, has_var_kwds)


class Computation(object):
    def __init__(self):
        self.dag = nx.DiGraph()
        self.v = _ComputationAttributeView(self.nodes, self.value, self.value)
        self.s = _ComputationAttributeView(self.nodes, self.state, self.state)
        self.i = _ComputationAttributeView(self.nodes, self.get_inputs, self.get_inputs)

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
        """
        LOG.debug('Adding node {}'.format(str(name)))
        args = kwargs.get('args', None)
        kwds = kwargs.get('kwds', None)
        value = kwargs.get('value', None)
        serialize = kwargs.get('serialize', True)
        inspect = kwargs.get('inspect', True)
        group = kwargs.get('group', None)

        self.dag.add_node(name)
        self.dag.remove_edges_from((p, name) for p in self.dag.predecessors(name))
        node = self.dag.node[name]

        node['state'] = States.UNINITIALIZED
        node['value'] = None
        node['serialize'] = serialize
        node['group'] = group

        if value is not None:
            node['state'] = States.UPTODATE
            node['value'] = value

        if func:
            node['func'] = func
            args_count = 0
            if args:
                args_count = len(args)
                for i, param_name in enumerate(args):
                    if not self.dag.has_node(param_name):
                        self.dag.add_node(param_name, state=States.PLACEHOLDER)
                    self.dag.add_edge(param_name, name, param=(_ParameterType.ARG, i))
            if inspect:
                signature = _get_signature(func)
                param_names = set()
                if not signature.has_var_args:
                    param_names.update(signature.kwd_params[args_count:])
                if signature.has_var_kwds and kwds is not None:
                    param_names.update(kwds.keys())
            else:
                param_names = kwds.keys()
            for param_name in param_names:
                in_node_name = kwds.get(param_name, param_name) if kwds else param_name
                if not self.dag.has_node(in_node_name):
                    self.dag.add_node(in_node_name, state=States.PLACEHOLDER)
                self.dag.add_edge(in_node_name, name, param=(_ParameterType.KWD, param_name))
            self._set_descendents(name, States.STALE)
            if node['state'] == States.UNINITIALIZED:
                self._try_set_computable(name)

    def delete_node(self, name):
        """
        Delete a node from a computation

        When nodes are explicitly deleted with ``delete_node``, but are still depended on by other nodes, then they will be set to PLACEHOLDER status. In this case, if the nodes that depend on a PLACEHOLDER node are deleted, then the PLACEHOLDER node will also be deleted.

        :param name: Name of the node to delete. If the node does not exist, a ``NonExistentNodeException`` will be raised.
        """
        LOG.debug('Deleting node {}'.format(str(name)))

        if name not in self.dag:
            raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        if len(self.dag.successors(name)) == 0:
            preds = self.dag.predecessors(name)
            self.dag.remove_node(name)
            for n in preds:
                if self.dag.node[n]['state'] == States.PLACEHOLDER:
                    self.delete_node(n)
        else:
            self.dag.node[name]['state'] = States.PLACEHOLDER

    def insert(self, name, value):
        """
        Insert a value into a node of a computation

        Following insertation, the node will have state UPTODATE, and all its descendents will be COMPUTABLE or STALE.

        If an attempt is made to insert a value into a node that does not exist, a ``NonExistentNodeException`` will be raised.

        :param name: Name of the node to add.
        :param value: The value to be inserted into the node.
        """
        LOG.debug('Inserting value into node {}'.format(str(name)))

        if name not in self.dag:
            raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        node = self.dag.node[name]
        node['value'] = value
        node['state'] = States.UPTODATE
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
            if name not in self.dag:
                raise NonExistentNodeException('Node {} does not exist'.format(str(name)))

        stale = set()
        computable = set()
        for name, value in name_value_pairs:
            self.dag.node[name]['value'] = value
            self.dag.node[name]['state'] = States.UPTODATE
            stale.update(nx.dag.descendants(self.dag, name))
            computable.update(self.dag.successors(name))
        names = set([name for name, value in name_value_pairs])
        stale.difference_update(names)
        computable.difference_update(names)
        for name in stale:
            self.dag.node[name]['state'] = States.STALE
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
            nodes = set(self.dag.nodes())
            nodes.intersection_update(other.dag.nodes())
        name_value_pairs = [(name, other.value(name)) for name in nodes]
        self.insert_many(name_value_pairs)

    def set_stale(self, name):
        """
        Set the state of a node and all its dependencies to STALE

        :param name: Name of the node to set as STALE.
        """
        self.dag.node[name]['state'] = States.STALE
        for n in nx.dag.descendants(self.dag, name):
            self.dag.node[n]['state'] = States.STALE
        self._try_set_computable(name)

    def _set_descendents(self, name, state):
        for n in nx.dag.descendants(self.dag, name):
            self.dag.node[n]['state'] = state

    def _set_uninitialized(self, name):
        n = self.dag.node[name]
        n['state'] = States.UNINITIALIZED
        n.pop('value', None)

    def _set_uptodate(self, name, value):
        node = self.dag.node[name]
        node['state'] = States.UPTODATE
        node['value'] = value
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

    def _set_error(self, name, error):
        node = self.dag.node[name]
        node['state'] = States.ERROR
        node['value'] = error
        self._set_descendents(name, States.STALE)

    def _try_set_computable(self, name):
        if 'func' in self.dag.node[name]:
            for n in self.dag.predecessors(name):
                if not self.dag.has_node(n):
                    return
                if self.dag.node[n]['state'] != States.UPTODATE:
                    return
            self.dag.node[name]['state'] = States.COMPUTABLE

    def _get_parameter_data(self, name):
        for in_node_name in self.dag.predecessors(name):
            param_value = self.dag.node[in_node_name]['value']
            edge = self.dag.edge[in_node_name][name]
            param_type, param_name = edge['param']
            yield _ParameterItem(param_type, param_name, param_value)

    def _compute_node(self, name, raise_exceptions=False):
        LOG.debug('Computing node {}'.format(str(name)))
        node = self.dag.node[name]
        f = node['func']
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
        try:
            value = f(*args, **kwds)
            node['state'] = States.UPTODATE
            node['value'] = value
            self._set_descendents(name, States.STALE)
            for n in self.dag.successors(name):
                self._try_set_computable(n)
        except Exception as e:
            node['state'] = States.ERROR
            tb = traceback.format_exc()
            node['value'] = Error(e, tb)
            self._set_descendents(name, States.STALE)
            if raise_exceptions:
                raise

    def _get_calc_nodes(self, name):
        g = nx.DiGraph()
        g.add_nodes_from(self.dag.nodes_iter())
        g.add_edges_from(self.dag.edges_iter())
        for n in nx.ancestors(g, name):
            node = self.dag.node[n]
            state = node['state']
            if state == States.UPTODATE:
                g.remove_node(n)
            if state == States.UNINITIALIZED and len(g.predecessors(n)) == 0:
                raise Exception("Cannot compute {} because {} uninitialized".format(name, n))
            if state == States.PLACEHOLDER:
                raise Exception("Cannot compute {} because {} is placeholder".format(name, n))

        ancestors = nx.ancestors(g, name)
        ancestors.add(name)
        nodes_sorted = nx.topological_sort(g, ancestors)
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
        calc_nodes = self._get_calc_nodes(name)
        for n in calc_nodes:
            preds = self.dag.predecessors(n)
            if all(self.dag.node[n1]['state'] == States.UPTODATE for n1 in preds):
                self._compute_node(n, raise_exceptions=raise_exceptions)

    def _get_computable_nodes_iter(self):
        for n, node in self.dag.nodes_iter(data=True):

            if node['state'] == States.COMPUTABLE:
                yield n

    def compute_all(self, raise_exceptions=False):
        """Compute all nodes of a computation that can be computed

        Nodes that are already UPTODATE will not be recalculated. Following the computation, if successful, all nodes will have state UPTODATE, except UNINITIALIZED input nodes and PLACEHOLDER nodes.

        If any nodes raises an exception, then the state of that node will be set to ERROR, and its value set to an object containing the exception object, as well as a traceback. This will not halt the computation, which will proceed as far as it can, until no more nodes are COMPUTABLE.

        :param raise_exceptions: Whether to pass exceptions raised by node computations back to the caller
        :type raise_exceptions: Boolean, default False
        """
        computed = set()
        while True:
            computable = self._get_computable_nodes_iter()
            any_computable = False
            for n in computable:
                if n in computed:
                    raise LoopDetectedException("compute_all is calculating {} for the second time".format(n))
                any_computable = True
                self._compute_node(n, raise_exceptions=raise_exceptions)
                computed.add(n)
            if not any_computable:
                break

    def nodes(self):
        """
        Get a list of nodes in this computation
        :return: List of nodes
        """
        return self.dag.nodes()

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
        if isinstance(name, list):
            return [self.dag.node[n]['state'] for n in name]
        return self.dag.node[name]['state']

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
        if isinstance(name, list):
            return [self.dag.node[n]['value'] for n in name]
        return self.dag.node[name]['value']

    def __getitem__(self, name):
        """
        Get the state and current value of a node

        :param name: Name of the node to get the state and value of
        """
        node = self.dag.node[name]
        return NodeData(node['state'], node['value'])

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
        df = pd.DataFrame(index=nx.topological_sort_recursive(self.dag))
        df['state'] = pd.Series(nx.get_node_attributes(self.dag, 'state'))
        df['value'] = pd.Series(nx.get_node_attributes(self.dag, 'value'))
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
        return nx.get_node_attributes(self.dag, 'value')

    def get_inputs(self, name):
        """
        Get a list of the inputs for a node or set of nodes
        
        :return: 
        """
        if isinstance(name, list):
            return [self.dag.predecessors(n) for n in name]
        return self.dag.predecessors(name)

    def write_dill(self, file_):
        """
        Serialize a computation to a file or file-like object

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        node_serialize = nx.get_node_attributes(self.dag, 'serialize')
        if all(serialize for name, serialize in six.iteritems(node_serialize)):
            obj = self
        else:
            obj = self.copy()
            for name, serialize in six.iteritems(node_serialize):
                if not serialize:
                    obj._set_uninitialized(name)

        if isinstance(file_, six.string_types):
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
        if isinstance(file_, six.string_types):
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
            self.dag.node[node_name]['is_expansion'] = True

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

    @staticmethod
    def _add_nodes(g, nodes, node_index_map):
        for name, n in nodes:
            short_name = node_index_map[name]
            node_color = state_colors[n.get('state', None)]
            g.node(short_name, str(name), style='filled', fillcolor=node_color)

    @staticmethod
    def _add_edges(g, edges, node_index_map):
        for name1, name2 in edges:
            short_name1, short_name2 = node_index_map[name1], node_index_map[name2]
            g.edge(short_name1, short_name2)

    def draw(self, graph_attr=None, node_attr=None, edge_attr=None, show_expansion=False):
        """
        Draw a computation's current state using the GraphViz utility

        :param graph_attr: Mapping of (attribute, value) pairs for the graph. For example ``graph_attr={'size': '10,8'}`` can control the size of the output graph
        :param node_attr: Mapping of (attribute, value) pairs set for all nodes.
        :param edge_attr: Mapping of (attribute, value) pairs set for all edges.
        :param show_expansion: Whether to show expansion nodes (i.e. named tuple expansion nodes) if they are not referenced by other nodes
        """
        nodes = [("n{}".format(i), name, data) for i, (name, data) in enumerate(self.dag.nodes(data=True))]
        node_index_map = {name: short_name for short_name, name, data in nodes}

        show_nodes = set()
        for name1, name2, n in self.dag.edges_iter(data=True):
            if not show_expansion and (self.dag.node[name2].get('is_expansion', False)):
                continue
            show_nodes.add(name1)
            show_nodes.add(name2)

        node_groups = {}
        for node, group in six.iteritems(nx.get_node_attributes(self.dag, 'group')):
            node_groups.setdefault(group, []).append(node)

        edge_groups = {}
        for name1, name2 in self.dag.edges_iter():
            group1 = self.dag.node[name1].get('group')
            group2 = self.dag.node[name2].get('group')
            group = group1 if group1 == group2 else None
            edge_groups.setdefault(group, []).append((name1, name2))

        g = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)

        for group, names in six.iteritems(node_groups):
            if group is None:
                continue
            c = graphviz.Digraph('cluster_' + str(group))

            nodes = ((name, self.dag.node[name]) for name in names if name in show_nodes)
            self._add_nodes(c, nodes, node_index_map)

            edges = ((name1, name2) for name1, name2 in edge_groups.get(group, []) if
                     name1 in show_nodes and name2 in show_nodes)
            self._add_edges(c, edges, node_index_map)

            c.body.append('label = "{}"'.format(str(group)))
            g.subgraph(c)

        nodes = ((name, self.dag.node[name]) for name in node_groups.get(None, []) if name in show_nodes)
        self._add_nodes(g, nodes, node_index_map)

        edges = ((name1, name2) for name1, name2 in edge_groups.get(None, []) if
                 name1 in show_nodes and name2 in show_nodes)
        self._add_edges(g, edges, node_index_map)

        return g
