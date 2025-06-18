import loman as lm

def test_simple_node_metadata():
    comp = lm.Computation()
    comp.add_node('foo', metadata={'test': 'working'})
    assert comp.metadata('foo')['test'] == 'working'

def test_simple_computation_metadata():
    comp = lm.Computation(metadata={'test': 'working'})
    assert comp.metadata('')['test'] == 'working'
