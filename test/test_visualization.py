import itertools
from itertools import tee

import networkx as nx

import loman.visualization
from loman import Computation, States, node
import loman.computeengine
from collections import namedtuple
from loman.consts import NodeTransformations
from loman.nodekey import NodeKey, to_nodekey
from test.standard_test_computations import create_example_block_computation, BasicFourNodeComputation


def node_set(nodes):
    s = set()
    for n in nodes:
        nodekey = to_nodekey(n)
        s.add(nodekey)
    return s


def edges_set(edges):
    s = set()
    for a, b in edges:
        nk_a = to_nodekey(a)
        nk_b = to_nodekey(a)
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
        path = NodeKey((label,)) if group is None else group.join_parts(label)
        d[path] = node_obj
    return d


def test_simple():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda b, c: b + c)

    v = loman.visualization.GraphView(comp, collapse_all=False)

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


def test_with_groups():
    comp = Computation()
    comp.add_node('a', group='foo')
    comp.add_node('b', lambda a: a + 1, group='foo')
    comp.add_node('c', lambda a: 2 * a, group='bar')
    comp.add_node('d', lambda b, c: b + c, group='bar')
    v = loman.visualization.GraphView(comp, collapse_all=False)


def test_show_expansion():
    Coordinate = namedtuple('Coordinate', ['x', 'y'])
    comp = Computation()
    comp.add_node('c', value=Coordinate(1, 2))
    comp.add_node('foo', lambda x: x + 1, kwds={'x': 'c.x'})
    comp.add_named_tuple_expansion('c', Coordinate)
    comp.compute_all()

    view_uncontracted = comp.draw(show_expansion=True, collapse_all=False)
    view_uncontracted.refresh()
    labels = nx.get_node_attributes(view_uncontracted.viz_dag, 'label')
    assert set(labels.values()) == {'c', 'c.x', 'c.y', 'foo'}

    view_contracted = comp.draw(show_expansion=False, collapse_all=False)
    view_contracted.refresh()
    labels = nx.get_node_attributes(view_contracted.viz_dag, 'label')
    assert set(labels.values()) == {'c', 'foo'}


def test_with_visualization_blocks():
    comp = create_example_block_computation()

    comp.compute_all()

    v = comp.draw(collapse_all=False)
    check_graph(v.struct_dag, [
        ('input_foo', 'foo/a', 'foo/b', 'foo/d', 'output'),
        ('foo/a', 'foo/c', 'foo/d'),
        ('input_bar', 'bar/a', 'bar/b', 'bar/d', 'output'),
        ('bar/a', 'bar/c', 'bar/d'),
    ])


def test_with_visualization_view_subblocks():
    comp = create_example_block_computation()

    comp.compute_all()

    v_foo = comp.draw('/foo', collapse_all=False)
    check_graph(v_foo.struct_dag,[('a', 'b', 'd'), ('a', 'c', 'd')])

    v_bar = comp.draw('/bar', collapse_all=False)
    check_graph(v_bar.struct_dag, [('a', 'b', 'd'), ('a', 'c', 'd')])


def test_with_visualization_collapsed_blocks():
    comp = create_example_block_computation()

    comp.compute_all()

    node_transformations = {'foo': NodeTransformations.COLLAPSE, 'bar': NodeTransformations.COLLAPSE}

    v = comp.draw(node_transformations=node_transformations, collapse_all=False)
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

    v = comp.draw(node_transformations={'foo1': NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey('foo1')]['shape'] == 'rect'
    assert d[to_nodekey('foo1')]['peripheries'] == 2

    v = comp.draw(node_transformations={'foo1/bar1': NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey('foo1/bar1')]['shape'] == 'rect'
    assert d[to_nodekey('foo1/bar1')]['peripheries'] == 2

    v = comp.draw(node_transformations={'foo1/bar1/baz1': NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey('foo1/bar1/baz1')]['shape'] == 'rect'
    assert d[to_nodekey('foo1/bar1/baz1')]['peripheries'] == 2


def test_sub_blocks_collapse_with_group():
    comp = loman.Computation()
    comp.add_node('a')
    comp.add_node('foo/bar/b', lambda a: a + 1, kwds={'a': 'a'})
    comp.add_node('foo/bar/c', lambda a: a + 1, kwds={'a': 'a'})
    v = comp.draw(node_transformations={'foo/bar': NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey('foo/bar')]['shape'] == 'rect'


def test_with_visualization_collapsed_blocks_uniform_sate():
    comp = loman.Computation()
    comp.add_node('a')
    comp.add_node('foo/bar/b', lambda a: a + 1, kwds={'a': 'a'})
    comp.add_node('foo/bar/c', lambda a: a + 1, kwds={'a': 'a'})
    v = comp.draw(node_transformations={'foo/bar': NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey('foo/bar')]['fillcolor'] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]


def test_with_visualization_view_default_collapsing():
    comp = create_example_block_computation()

    comp.compute_all()

    v_foo = comp.draw()
    check_graph(v_foo.struct_dag,[('input_foo', 'foo', 'output'), ('input_bar', 'bar', 'output')])


def test_with_visualization_view_subblocks_default_collapsing():
    comp = Computation()
    comp.add_node('foo/a')
    comp.add_node('foo/b', lambda a: a + 1)
    comp.add_node('foo/c', lambda a: 2 * a)
    comp.add_node('foo/d', lambda b, c: b + c)
    v = comp.draw('foo')
    check_graph(v.struct_dag, [('a', 'b', 'd'), ('a', 'c', 'd')])


def test_draw_expanded_block():
    comp = Computation()
    comp.add_node('foo/bar/baz/a')
    comp.add_node('foo/bar/baz/b', lambda a: a + 1)
    comp.add_node('foo/bar/baz/c', lambda a: 2 * a)
    comp.add_node('foo/bar/baz/d', lambda b, c: b + c)

    v = comp.draw(node_transformations={'foo/bar/baz': 'expand'})
    nodes = get_path_to_node_mapping(v)

    assert to_nodekey('foo/bar/baz/a') in nodes
    assert to_nodekey('foo/bar/baz/b') in nodes
    assert to_nodekey('foo/bar/baz/c') in nodes
    assert to_nodekey('foo/bar/baz/d') in nodes


def test_draw_expanded_block_with_wildcard():
    comp = Computation()
    comp.add_node('foo/bar/baz/a')
    comp.add_node('foo/bar/baz/b', lambda a: a + 1)
    comp.add_node('foo/bar/baz/c', lambda a: 2 * a)
    comp.add_node('foo/bar/baz/d', lambda b, c: b + c)

    v = comp.draw(node_transformations={'**': 'expand'})
    nodes = get_path_to_node_mapping(v)

    assert to_nodekey('foo/bar/baz/a') in nodes
    assert to_nodekey('foo/bar/baz/b') in nodes
    assert to_nodekey('foo/bar/baz/c') in nodes
    assert to_nodekey('foo/bar/baz/d') in nodes


def test_draw_expanded_block_with_wildcard_2():
    comp_inner = BasicFourNodeComputation()
    comp = Computation()
    for x, y, z in itertools.product(range(1, 3), range(1, 3), range(1, 3)):
        comp.add_block(f'foo{x}/bar{y}/baz{z}', comp_inner, keep_values=False, links={'a': 'input_a'})
    comp.add_node('input_a', value=7)
    comp.compute_all()

    v = comp.draw(node_transformations={'**': 'expand'})
    nodes = get_path_to_node_mapping(v)
    expected = ['input_a',
                'foo1/bar1/baz1/a', 'foo1/bar1/baz1/b', 'foo1/bar1/baz1/c', 'foo1/bar1/baz1/d',
                'foo1/bar1/baz2/a', 'foo1/bar1/baz2/b', 'foo1/bar1/baz2/c', 'foo1/bar1/baz2/d',
                'foo1/bar2/baz1/a', 'foo1/bar2/baz1/b', 'foo1/bar2/baz1/c', 'foo1/bar2/baz1/d',
                'foo1/bar2/baz2/a', 'foo1/bar2/baz2/b', 'foo1/bar2/baz2/c', 'foo1/bar2/baz2/d',
                'foo2/bar1/baz1/a', 'foo2/bar1/baz1/b', 'foo2/bar1/baz1/c', 'foo2/bar1/baz1/d',
                'foo2/bar1/baz2/a', 'foo2/bar1/baz2/b', 'foo2/bar1/baz2/c', 'foo2/bar1/baz2/d',
                'foo2/bar2/baz1/a', 'foo2/bar2/baz1/b', 'foo2/bar2/baz1/c', 'foo2/bar2/baz1/d',
                'foo2/bar2/baz2/a', 'foo2/bar2/baz2/b', 'foo2/bar2/baz2/c', 'foo2/bar2/baz2/d']
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={'foo1/bar1/**': 'expand'})
    nodes = get_path_to_node_mapping(v)
    expected = ['input_a',
                'foo1/bar1/baz1/a', 'foo1/bar1/baz1/b', 'foo1/bar1/baz1/c', 'foo1/bar1/baz1/d',
                'foo1/bar1/baz2/a', 'foo1/bar1/baz2/b', 'foo1/bar1/baz2/c', 'foo1/bar1/baz2/d',
                'foo1/bar2', 'foo2']
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={'foo2/bar2/**': 'expand'})
    nodes = get_path_to_node_mapping(v)
    expected = ['input_a',
                'foo1', 'foo2/bar1',
                'foo2/bar2/baz1/a', 'foo2/bar2/baz1/b', 'foo2/bar2/baz1/c', 'foo2/bar2/baz1/d',
                'foo2/bar2/baz2/a', 'foo2/bar2/baz2/b', 'foo2/bar2/baz2/c', 'foo2/bar2/baz2/d']
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={'*': 'expand'})
    nodes = get_path_to_node_mapping(v)
    assert nodes.keys() == {to_nodekey(n) for n in ['input_a', 'foo1/bar1', 'foo1/bar2', 'foo2/bar1', 'foo2/bar2']}

    v = comp.draw(node_transformations={'*/*': 'expand'})
    nodes = get_path_to_node_mapping(v)
    assert nodes.keys() == {to_nodekey(n) for n in ['input_a',
                                                    'foo1/bar1/baz1', 'foo1/bar1/baz2', 'foo1/bar2/baz1', 'foo1/bar2/baz2',
                                                    'foo2/bar1/baz1', 'foo2/bar1/baz2', 'foo2/bar2/baz1', 'foo2/bar2/baz2']}

def test_style_preservation_with_block_links():
    def build_comp():
        comp = Computation()
        comp.add_node("a", style="dot")
        comp.add_node("b", style="dot")
        comp.add_node('e', style='dot')

        @node(comp, "c")
        def comp_c(a, b):
            return a + b

        comp.add_node("d", lambda a: a + 1, style="small")
        return comp

    full_comp = Computation()
    full_comp.add_node("params/a", value=1, style="dot")
    full_comp.add_node("params/b", value=1, style="dot")
    full_comp.add_node("params/c", value=1, style="dot")
    full_comp.add_block("comp", build_comp(), links={
        "a": "params/a",
        "b": "params/b",
        "e": "params/c"
    })

    expected_styles = ['dot'] * 4 + [None, 'small']
    actual_styles = full_comp.styles(["params/a", 'params/b', 'comp/a', 'comp/b', 'comp/c', 'comp/d'])

    assert expected_styles == actual_styles
