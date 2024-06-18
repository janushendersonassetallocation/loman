import loman.visualization
from loman import Computation, States
import loman.computeengine


def test_simple():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda b, c: b + c)

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in nodes.items()}
    node = {label: nodes[name][0] for label, name in label_to_name_mapping.items()}
    assert node['a']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UNINITIALIZED]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UNINITIALIZED]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UNINITIALIZED]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UNINITIALIZED]
    assert node['d']['attributes']['style'] == 'filled'

    comp.insert('a', 1)

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in nodes.items()}
    node = {label: nodes[name][0] for label, name in label_to_name_mapping.items()}
    assert node['a']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.state_colors[States.COMPUTABLE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.state_colors[States.COMPUTABLE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.state_colors[States.STALE]
    assert node['d']['attributes']['style'] == 'filled'

    comp.compute_all()

    d = comp.to_pydot()

    nodes = d.obj_dict['nodes']
    label_to_name_mapping = {v[0]['attributes']['label']: k for k, v in nodes.items()}
    node = {label: nodes[name][0] for label, name in label_to_name_mapping.items()}
    assert node['a']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UPTODATE]
    assert node['a']['attributes']['style'] == 'filled'
    assert node['b']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UPTODATE]
    assert node['b']['attributes']['style'] == 'filled'
    assert node['c']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UPTODATE]
    assert node['c']['attributes']['style'] == 'filled'
    assert node['d']['attributes']['fillcolor'] == loman.visualization.state_colors[States.UPTODATE]
    assert node['d']['attributes']['style'] == 'filled'
