import networkx as nx
from enum import Enum
from collections import OrderedDict, deque, namedtuple
import inspect
import decorator
import dill
import six
import seaborn as sns


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

class Computation(object):
    def __init__(self):
        self.dag = nx.DiGraph()

    def add_node(self, name, func=None, sources=None):
        self.dag.add_node(name)
        self.dag.node[name]['state'] = States.UNINITIALIZED
        if func:
            self.dag.node[name]['func'] = func
            argspec = inspect.getargspec(func)
            for arg in argspec.args:
                if sources:
                    source = sources.get(arg, arg)
                else:
                    source = arg
                self.dag.add_edge(source, name, arg_name=arg)
        self._try_set_computable(name)

    def draw(self, show_values=True):
        if show_values:
            labels = {k: "{}: {}".format(k, v.get('value')) for k, v in self.dag.node.items()}
        else:
            labels = {k: "{}".format(k) for k, v in self.dag.node.items()}
        node_color = [state_colors[n.get('state', None)] for name, n in self.dag.node.iteritems()]
        nx.draw(self.dag, with_labels=True, arrows=True, labels=labels, node_shape='s', node_color=node_color)

    def insert(self, name, value):
        self.dag.node[name]['value'] = value
        self.dag.node[name]['state'] = States.UPTODATE
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

    def _set_all(self, state):
        for n in self.dag.nodes():
            self.dag.node[n]['state'] = state

    def _set_descendents(self, name, state):
        for n in nx.dag.descendants(self.dag, name):
            self.dag.node[n]['state'] = state

    def _try_set_computable(self, name):
        if 'func' in self.dag.node[name]:
            for n in self.dag.predecessors(name):
                if self.dag.node[n]['state'] != States.UPTODATE:
                    return
            self.dag.node[name]['state'] = States.COMPUTABLE

    def _compute_node(self, name):
        f = self.dag.node[name]['func']
        params = {}
        for n in self.dag.predecessors(name):
            value = self.value(n)
            edge_data = self.dag.get_edge_data(n, name)
            arg_name = edge_data['arg_name']
            params[arg_name] = value
        try:
            value = f(**params)
            self.dag.node[name]['state'] = States.UPTODATE
            self.dag.node[name]['value'] = value
            self.dag.node[name].pop('exception', None)
            self._set_descendents(name, States.STALE)
            for n in self.dag.successors(name):
                self._try_set_computable(n)
        except Exception as e:
            self.dag.node[name]['state'] = States.ERROR
            self.dag.node[name].pop('value', None)
            self.dag.node[name]['exception'] = e
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

    def value(self, name):
        return self.dag.node[name]['value']

    def state(self, name):
        return self.dag.node[name]['state']

    def exception(self, name):
        return self.dag.node[name]['exception']

    def write_pickle(self, file_):
        if isinstance(file_, six.string_types):
            with open(file_, 'wb') as f:
                dill.dump(self, f)
        else:
            dill.dump(self, file_)

    @staticmethod
    def read_pickle(file_):
        if isinstance(file_, six.string_types):
            with open(file_, 'rb') as f:
                return dill.load(f)
        else:
            return dill.load(file_)

    def add_named_tuple_expansion(self, name, namedtuple_type):
        def make_f(field):
            def get_field_value(tuple):
                return getattr(tuple, field)
            return get_field_value
        for field in namedtuple_type._fields:
            node_name = "{}.{}".format(name, field)
            self.add_node(node_name, make_f(field), {'tuple': name})