from itertools import tee

import networkx as nx

import loman.visualization
from loman import Computation, States
import loman.computeengine
from collections import namedtuple
from loman.consts import SystemTags, NodeTransformations
from loman.path_parser import Path, to_path
from loman.structs import NodeKey
from test.standard_test_computations import create_example_block_computation


def node_set(nodes):
    s = set()
    for n in nodes:
        nodekey = NodeKey.from_name(n)
        s.add(nodekey)
    return s


def edges_set(edges):
    s = set()
    for a, b in edges:
        nk_a = NodeKey.from_name(a)
        nk_b = NodeKey.from_name(a)
        el = frozenset((nk_a, nk_b))
        s.add(el)
    return s


def edges_from_chain(chain_iter):
    a, b = tee(chain_iter)
    next(b, None)
    return zip(a, b)


def check_graph(g, expected_chains):
    expected_nodes = set()
    for chain in expected_chains:
        for node in chain:
            expected_nodes.add(node)

    expected_edges = set()
    for chain in expected_chains:
        for node_a, node_b in edges_from_chain(chain):
            expected_edges.add((node_a, node_b))

    assert node_set(expected_nodes) == node_set(g.nodes)
    assert edges_set(expected_edges) == edges_set(g.edges)


def test_simple():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda b, c: b + c)

    v = loman.visualization.GraphView(comp)

    node = get_label_to_node_mapping(v)
    assert node['a']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    assert node['d']['attributes']['style'] == 'filled'

    comp.insert('a', 1)

    v.refresh()
    node = get_label_to_node_mapping(v)
    assert node['a']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.COMPUTABLE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.COMPUTABLE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.STALE]
    assert node['d']['attributes']['style'] == 'filled'

    comp.compute_all()

    v.refresh()
    node = get_label_to_node_mapping(v)
    assert node['a']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['d']['attributes']['style'] == 'filled'


def get_label_to_node_mapping(v):
    nodes = v.viz_dot.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in nodes.items()}
    node = {label: nodes[name][0] for label, name in label_to_name_mapping.items()}
    return node


def get_path_to_node_mapping(v):
    d = {}
    for name, node_obj in v.viz_dag.nodes(data=True):
        label = node_obj['label']
        group = node_obj.get('_group')
        path = Path((label,), is_absolute_path=False) if group is None else group.join(label)
        d[path] = node_obj
    return d


def test_with_groups():
    comp = Computation()
    comp.add_node('a', group='foo')
    comp.add_node('b', lambda a: a + 1, group='foo')
    comp.add_node('c', lambda a: 2 * a, group='bar')
    comp.add_node('d', lambda b, c: b + c, group='bar')
    v = loman.visualization.GraphView(comp)


def test_show_expansion():
    Coordinate = namedtuple('Coordinate', ['x', 'y'])
    comp = Computation()
    comp.add_node('c', value=Coordinate(1, 2))
    comp.add_node('foo', lambda x: x + 1, kwds={'x': 'c.x'})
    comp.add_named_tuple_expansion('c', Coordinate)
    comp.compute_all()

    view_uncontracted = comp.draw(show_expansion=True)
    view_uncontracted.refresh()
    labels = nx.get_node_attributes(view_uncontracted.viz_dag, 'label')
    assert set(labels.values()) == {'c', 'c.x', 'c.y', 'foo'}

    view_contracted = comp.draw(show_expansion=False)
    view_contracted.refresh()
    labels = nx.get_node_attributes(view_contracted.viz_dag, 'label')
    assert set(labels.values()) == {'c', 'foo'}


def test_with_visualization_blocks():
    comp = create_example_block_computation()

    comp.compute_all()

    v = comp.draw()
    check_graph(v.struct_dag, [
        ('input_foo', 'foo/a', 'foo/b', 'foo/d', 'output'),
        ('foo/a', 'foo/c', 'foo/d'),
        ('input_bar', 'bar/a', 'bar/b', 'bar/d', 'output'),
        ('bar/a', 'bar/c', 'bar/d'),
    ])


def test_with_visualization_view_subblocks():
    comp = create_example_block_computation()

    comp.compute_all()

    v_foo = comp.draw('/foo')
    check_graph(v_foo.struct_dag,[('a', 'b', 'd'), ('a', 'c', 'd')])

    v_bar = comp.draw('/bar')
    check_graph(v_bar.struct_dag, [('a', 'b', 'd'), ('a', 'c', 'd')])


def test_with_visualization_collapsed_blocks():
    comp = create_example_block_computation()

    comp.compute_all()

    node_transformations = {'foo': NodeTransformations.COLLAPSE, 'bar': NodeTransformations.COLLAPSE}

    v = comp.draw(node_transformations=node_transformations)
    check_graph(v.struct_dag, [
        ('input_foo', 'foo', 'output'),
        ('input_bar', 'bar', 'output')
    ])
    node = get_label_to_node_mapping(v)
    assert node['foo']['attributes']['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    assert node['foo']['attributes']['shape'] == 'rect'
    assert node['foo']['attributes']['peripheries'] == 2


def test_with_visualization_single_element_collapsed_blocks():
    comp = loman.Computation()
    comp.add_node('foo1/bar1/baz1/a')

    v = comp.draw(node_transformations={'foo1': NodeTransformations.COLLAPSE})
    d = get_path_to_node_mapping(v)
    assert d[to_path('foo1')]['shape'] == 'rect'
    assert d[to_path('foo1')]['peripheries'] == 2

    v = comp.draw(node_transformations={'foo1/bar1': NodeTransformations.COLLAPSE})
    d = get_path_to_node_mapping(v)
    assert d[to_path('foo1/bar1')]['shape'] == 'rect'
    assert d[to_path('foo1/bar1')]['peripheries'] == 2

    v = comp.draw(node_transformations={'foo1/bar1/baz1': NodeTransformations.COLLAPSE})
    d = get_path_to_node_mapping(v)
    assert d[to_path('foo1/bar1/baz1')]['shape'] == 'rect'
    assert d[to_path('foo1/bar1/baz1')]['peripheries'] == 2


def test_sub_blocks_collapse_with_group():
    comp = loman.Computation()
    comp.add_node('a')
    comp.add_node('foo/bar/b', lambda a: a + 1, kwds={'a': 'a'})
    comp.add_node('foo/bar/c', lambda a: a + 1, kwds={'a': 'a'})
    v = comp.draw(node_transformations={'foo/bar': NodeTransformations.COLLAPSE})
    d = get_path_to_node_mapping(v)
    assert d[to_path('foo/bar')]['shape'] == 'rect'


def test_with_visualization_collapsed_blocks_uniform_sate():
    comp = loman.Computation()
    comp.add_node('a')
    comp.add_node('foo/bar/b', lambda a: a + 1, kwds={'a': 'a'})
    comp.add_node('foo/bar/c', lambda a: a + 1, kwds={'a': 'a'})
    v = comp.draw(node_transformations={'foo/bar': NodeTransformations.COLLAPSE})
    d = get_path_to_node_mapping(v)
    assert d[to_path('foo/bar')]['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
