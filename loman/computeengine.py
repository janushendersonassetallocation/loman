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


class States(Enum):
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


class _ParameterType(Enum):
    ARG = 1
    KWD = 2

_ParameterItem = namedtuple('ParameterItem', ['type', 'name', 'value'])


class Computation(object):
    def __init__(self):
        self.dag = nx.DiGraph()

    def add_node(self, name, func=None, **kwargs):
        args = kwargs.get('args', None)
        kwds = kwargs.get('kwds', None)
        value = kwargs.get('value', None)
        serialize = kwargs.get('serialize', True)
        self.dag.add_node(name)
        self.dag.remove_edges_from((p, name) for p in self.dag.predecessors(name))
        node = self.dag.node[name]

        node['state'] = States.UNINITIALIZED
        node['value'] = None
        node['serialize'] = serialize

        if value is not None:
            node['state'] = States.UPTODATE
            node['value'] = value

        if func:
            node['func'] = func
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
            if args:
                for i, param_name in enumerate(args):
                    if not self.dag.has_node(param_name):
                        self.dag.add_node(param_name, state=States.PLACEHOLDER)
                    self.dag.add_edge(param_name, name, param=(_ParameterType.ARG, i))
            param_names = set()
            if not has_var_args:
                param_names.update(all_keyword_params)
            if has_var_kwds:
                if kwds:
                    param_names.update(kwds.keys())
            for param_name in param_names:
                in_node_name = kwds.get(param_name, param_name) if kwds else param_name
                if not self.dag.has_node(in_node_name):
                    self.dag.add_node(in_node_name, state=States.PLACEHOLDER)
                self.dag.add_edge(in_node_name, name, param=(_ParameterType.KWD, param_name))
            self._set_descendents(name, States.STALE)
            if node['state'] == States.UNINITIALIZED:
                self._try_set_computable(name)

    def delete_node(self, name):
        if len(self.dag.successors(name)) == 0:
            preds = self.dag.predecessors(name)
            self.dag.remove_node(name)
            for n in preds:
                if self.dag.node[n]['state'] == States.PLACEHOLDER:
                    self.delete_node(n)
        else:
            self.dag.node[name]['state'] = States.PLACEHOLDER

    def insert(self, name, value):
        node = self.dag.node[name]
        node['value'] = value
        node['state'] = States.UPTODATE
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

    def insert_many(self, name_value_pairs):
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
        if nodes is None:
            nodes = set(self.dag.nodes())
            nodes.intersection_update(other.dag.nodes())
        name_value_pairs = [(name, other.value(name)) for name in nodes]
        self.insert_many(name_value_pairs)

    def set_stale(self, name):
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

    def _compute_node(self, name):
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

    def _get_calc_nodes(self, name):
        process_nodes = deque([name])
        seen = set(process_nodes)
        calc_nodes = []
        while process_nodes:
            n = process_nodes.popleft()
            seen.add(n)
            node = self.dag.node[n]
            state = node['state']
            if state == States.UPTODATE:
                continue
            preds = self.dag.predecessors(n)
            if state == States.UNINITIALIZED and len(preds) == 0:
                raise Exception("Cannot compute {} because {} uninitialized".format(name, n))
            if state == States.PLACEHOLDER:
                raise Exception("Cannot compute {} because {} is placeholder".format(name, n))
            calc_nodes.append(n)
            for n1 in self.dag.predecessors(n):
                if n1 not in seen:
                    process_nodes.append(n1)
        calc_nodes.reverse()
        return calc_nodes

    def compute(self, name):
        for n in self._get_calc_nodes(name):
            if all(self.dag.node[n1]['state'] == States.UPTODATE for n1 in self.dag.predecessors(n)):
                self._compute_node(n)

    def _get_computable_nodes_iter(self):
        for n, node in self.dag.nodes_iter(data=True):

            if node['state'] == States.COMPUTABLE:
                yield n

    def compute_all(self):
        while True:
            computable = self._get_computable_nodes_iter()
            any_computable = False
            for n in computable:
                any_computable = True
                self._compute_node(n)
            if not any_computable:
                break

    def state(self, name):
        return self.dag.node[name]['state']

    def value(self, name):
        return self.dag.node[name]['value']

    def __getitem__(self, name):
        node = self.dag.node[name]
        return NodeData(node['state'], node['value'])

    def write_dill(self, file_):
        node_serialize = nx.get_node_attributes(self.dag, 'serialize')
        if all(serialize for name, serialize in node_serialize.items()):
            obj = self
        else:
            obj = self.copy()
            for name, serialize in node_serialize.items():
                if not serialize:
                    obj._set_uninitialized(name)

        if isinstance(file_, six.string_types):
            with open(file_, 'wb') as f:
                dill.dump(obj, f)
        else:
            dill.dump(obj, file_)

    @staticmethod
    def read_dill(file_):
        if isinstance(file_, six.string_types):
            with open(file_, 'rb') as f:
                return dill.load(f)
        else:
            return dill.load(file_)

    def copy(self):
        obj = Computation()
        obj.dag = nx.DiGraph(self.dag)
        return obj

    def add_named_tuple_expansion(self, name, namedtuple_type):
        def make_f(field):
            def get_field_value(tuple):
                return getattr(tuple, field)
            return get_field_value
        for field in namedtuple_type._fields:
            node_name = "{}.{}".format(name, field)
            self.add_node(node_name, make_f(field), kwds={'tuple': name})
            self.dag.node[node_name]['is_expansion'] = True

    def get_df(self):
        df = pd.DataFrame(index=nx.topological_sort_recursive(self.dag))
        df['state'] = pd.Series(nx.get_node_attributes(self.dag, 'state'))
        df['value'] = pd.Series(nx.get_node_attributes(self.dag, 'value'))
        df['is_expansion'] = pd.Series(nx.get_node_attributes(self.dag, 'is_expansion'))
        return df

    def get_value_dict(self):
        return nx.get_node_attributes(self.dag, 'value')

    def add_map_node(self, result_node, input_node, subgraph, subgraph_input_node, subgraph_output_node):
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


    def draw_nx(self, show_values=True):
        if show_values:
            labels = {k: "{}: {}".format(k, v.get('value')) for k, v in self.dag.node.items()}
        else:
            labels = {k: "{}".format(k) for k, v in self.dag.node.items()}
        node_color = [state_colors[n.get('state', None)] for name, n in self.dag.node.iteritems()]
        nx.draw(self.dag, with_labels=True, arrows=True, labels=labels, node_shape='s', node_color=node_color)

    def draw_graphviz(self, graph_attr=None, node_attr=None, edge_attr=None, show_expansion=False):
        nodes = [("n{}".format(i), name, data) for i, (name, data) in enumerate(self.dag.nodes(data=True))]
        node_index_map = {name: short_name for short_name, name, data in nodes}
        show_nodes = set()
        g = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
        for name1, name2, n in self.dag.edges_iter(data=True):
            if not show_expansion and (self.dag.node[name2].get('is_expansion', False)):
                continue
            show_nodes.add(name1)
            show_nodes.add(name2)
        for name, n in self.dag.nodes_iter(data=True):
            if name in show_nodes:
                short_name = node_index_map[name]
                node_color = state_colors[n.get('state', None)]
                g.node(short_name, str(name), style='filled', fillcolor=node_color)
        for name1, name2, n in self.dag.edges_iter(data=True):
            if name1 in show_nodes and name2 in show_nodes:
                short_name1, short_name2 = node_index_map[name1], node_index_map[name2]
                g.edge(short_name1, short_name2)
        return g
