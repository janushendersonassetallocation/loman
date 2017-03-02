import networkx as nx
from enum import Enum
from collections import OrderedDict, deque, namedtuple
import inspect
import decorator
import dill
import six
import seaborn as sns
import pandas as pd
import traceback
import graphviz
import py.path
import os
from IPython.display import Image


class States(Enum):
    UNINITIALIZED = 1
    STALE = 2
    COMPUTABLE = 3
    UPTODATE = 4
    ERROR = 5


state_colors = {
    None: sns.xkcd_rgb['white'],
    States.UNINITIALIZED: sns.xkcd_rgb['blue'],
    States.STALE: sns.xkcd_rgb['yellow'],
    States.COMPUTABLE: sns.xkcd_rgb['bright yellow green'],
    States.UPTODATE: sns.xkcd_rgb['green'],
    States.ERROR: sns.xkcd_rgb['red']
}

Error = namedtuple('Error', ['exception', 'traceback'])
NodeData = namedtuple('NodeData', ['state', 'value'])


class Computation(object):
    def __init__(self):
        self.dag = nx.DiGraph()

    def add_node(self, name, func=None, sources=None, serialize=True):
        self.dag.add_node(name)
        self.dag.remove_edges_from((p, name) for p in self.dag.predecessors(name))
        node = self.dag.node[name]
        node.clear()

        node['state'] = States.UNINITIALIZED
        node['value'] = None
        node['serialize'] = serialize

        if func:
            node['func'] = func
            argspec = inspect.getargspec(func)
            for arg in argspec.args:
                if sources:
                    source = sources.get(arg, arg)
                else:
                    source = arg
                if not self.dag.has_node(source):
                    raise Exception("No such node: {}".format(source))
                self.dag.add_edge(source, name, arg_name=arg)
        for n in nx.dag.descendants(self.dag, name):
            if self.dag.node[n]['state'] in (States.COMPUTABLE, States.ERROR, States.UPTODATE):
                self.dag.node[n]['state'] = States.STALE
        self._try_set_computable(name)

    def insert(self, name, value):
        node = self.dag.node[name]
        node['value'] = value
        node['state'] = States.UPTODATE
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

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

    def _compute_node(self, name):
        node = self.dag.node[name]
        f = node['func']
        params = {}
        for n in self.dag.predecessors(name):
            value = self.value(n)
            edge_data = self.dag.get_edge_data(n, name)
            arg_name = edge_data['arg_name']
            params[arg_name] = value
        try:
            value = f(**params)
            self._set_uptodate(name, value)
        except Exception as e:
            self._set_error(name, Error(e, traceback.format_exc()))

    def _get_calc_nodes(self, name):
        process_nodes = deque([name])
        seen = set(process_nodes)
        calc_nodes = []
        while process_nodes:
            n = process_nodes.popleft()
            seen.add(n)
            node = self.dag.node[n]
            state = node['state']
            if state == States.UNINITIALIZED:
                raise Exception()
            elif state == States.STALE or state == States.COMPUTABLE:
                calc_nodes.append(n)
                for n1 in self.dag.predecessors(n):
                    if n1 not in seen:
                        process_nodes.append(n1)
            elif state == States.UPTODATE:
                pass
        calc_nodes.reverse()
        return calc_nodes

    def compute(self, name):
        for n in self._get_calc_nodes(name):
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
            self.add_node(node_name, make_f(field), {'tuple': name})
            self.dag.node[node_name]['is_expansion'] = True

    def get_df(self):
        df = pd.DataFrame(index=nx.topological_sort_recursive(self.dag))
        df['state'] = pd.Series(nx.get_node_attributes(self.dag, 'state'))
        df['value'] = pd.Series(nx.get_node_attributes(self.dag, 'value'))
        df['is_expansion'] = pd.Series(nx.get_node_attributes(self.dag, 'is_expansion'))
        return df

    def get_value_dict(self):
        return nx.get_node_attributes(self.dag, 'value')

    def insert_from(self, other, nodes=None):
        if nodes is None:
            nodes = set(self.dag.nodes())
            nodes.intersection_update(other.dag.nodes())
        for name in nodes:
            self.insert(name, other.value(name))

    def draw(self, show_values=True):
        if show_values:
            labels = {k: "{}: {}".format(k, v.get('value')) for k, v in self.dag.node.items()}
        else:
            labels = {k: "{}".format(k) for k, v in self.dag.node.items()}
        node_color = [state_colors[n.get('state', None)] for name, n in self.dag.node.iteritems()]
        nx.draw(self.dag, with_labels=True, arrows=True, labels=labels, node_shape='s', node_color=node_color)

    def draw2(self, graph_attr=None, node_attr=None, edge_attr=None, show_expansion=False):
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
                g.node(short_name, name, style='filled', fillcolor=node_color)
        for name1, name2, n in self.dag.edges_iter(data=True):
            if name1 in show_nodes and name2 in show_nodes:
                short_name1, short_name2 = node_index_map[name1], node_index_map[name2]
                g.edge(short_name1, short_name2)
        with open('tmp.dot', 'w') as f:
            f.write(g.source)
        os.system('dot tmp.dot -Tpng -o test.png')
        return Image(filename='test.png')
