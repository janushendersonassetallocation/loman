from loman import Computation, States, MapException, LoopDetectedException, NonExistentNodeException, node, C
import loman.computeengine
import six


def test_simple():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda b, c: b + c)

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in six.iteritems(nodes)}
    node = {label: nodes[name][0] for label, name in six.iteritems(label_to_name_mapping)}
    assert node['a']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UNINITIALIZED]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UNINITIALIZED]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UNINITIALIZED]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UNINITIALIZED]
    assert node['d']['attributes']['style'] == 'filled'

    comp.insert('a', 1)

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in six.iteritems(nodes)}
    node = {label: nodes[name][0] for label, name in six.iteritems(label_to_name_mapping)}
    assert node['a']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.COMPUTABLE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.COMPUTABLE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.STALE]
    assert node['d']['attributes']['style'] == 'filled'

    comp.compute_all()

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in six.iteritems(nodes)}
    node = {label: nodes[name][0] for label, name in six.iteritems(label_to_name_mapping)}
    assert node['a']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UPTODATE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UPTODATE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.computeengine._state_colors[States.UPTODATE]
    assert node['d']['attributes']['style'] == 'filled'
