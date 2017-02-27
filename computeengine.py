import networkx as nx
from enum import Enum
from collections import OrderedDict, deque, namedtuple
import inspect
import decorator


class States(Enum):
    UNINITIALIZED = 1
    STALE = 2
    COMPUTABLE = 3
    UPTODATE = 4


class Computation(object):
    def __init__(self):
        self.dag = nx.DiGraph()

    def add_node(self, name, func=None, sources=None):
        self.dag.add_node(name)
        self.dag.node[name]['state'] = States.UNINITIALIZED
        if func:
            self.dag.node[name]['func'] = func
        if sources:
            for source in sources:
                self.dag.add_edge(source, name)

    def draw(self):
        labels = {k: "{}: {}".format(k, v.get('value')) for k,v in self.dag.node.items()}
        nx.draw(self.dag, with_labels=True, arrows=True, labels=labels, node_shape='s')

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
        for n in self.dag.predecessors(name):
            if self.dag.node[n]['state'] != States.UPTODATE:
                return
        self.dag.node[name]['state'] = States.COMPUTABLE

    def _compute_node(self, name):
        f = self.dag.node[name]['func']
        params = {n: self.dag.node[n].get('value') for n in self.dag.predecessors(name)}
        value = f(**params)
        self.dag.node[name]['state'] = States.UPTODATE
        self.dag.node[name]['value'] = value
        self._set_descendents(name, States.STALE)
        for n in self.dag.successors(name):
            self._try_set_computable(n)

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
        return (n for n, node in self.dag.nodes_iter(data=True) if node['state'] == States.COMPUTABLE)

    def compute_all(self):
        while True:
            computable = self._get_computable_nodes_iter()
            any_computable = False
            for n in computable:
                any_computable = True
                self._compute_node(n)
            if not any_computable:
                break
