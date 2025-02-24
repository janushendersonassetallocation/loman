from loman import *

def test_simple_block():
    comp_inner = Computation()
    comp_inner.add_node('a')
    comp_inner.add_node('b', lambda a: a + 1)
    comp_inner.add_node('c', lambda a: 2 * a)
    comp_inner.add_node('d', lambda b, c: b + c)

    comp = Computation()
    comp.add_block('foo', comp_inner)
    comp.add_block('bar', comp_inner)
    comp.add_node('input_foo')
    comp.add_node('input_bar')
    comp.link('foo/a', 'input_foo')
    comp.link('bar/a', 'input_bar')
    comp.add_node('output', lambda x, y: x + y, kwds={'x': 'foo/d', 'y': 'bar/d'})

    comp.insert('input_foo', value=7)
    comp.insert('input_bar', value=10)

    comp.compute_all()

    assert comp.v['foo/d'] == 22
    assert comp.v['bar/d'] == 31
    assert comp.v.output == 22 + 31
