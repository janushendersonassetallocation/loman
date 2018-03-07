import random

from loman import Computation
from loman.test.standard_test_computations import BasicFourNodeComputation


def test_get_inputs():
    comp = BasicFourNodeComputation()
    assert set(comp.get_inputs('a')) == set()
    assert set(comp.get_inputs('b')) == {'a'}
    assert set(comp.get_inputs('c')) == {'a'}
    assert set(comp.get_inputs('d')) == {'c', 'b'}
    assert list(map(set, comp.get_inputs(['a', 'b', 'c', 'd']))) == [set(), {'a'}, {'a'}, {'b', 'c'}]


def test_attribute_i():
    comp = BasicFourNodeComputation()
    assert set(comp.i.a) == set()
    assert set(comp.i.b) == {'a'}
    assert set(comp.i.c) == {'a'}
    assert set(comp.i.d) == {'c', 'b'}
    assert set(comp.i['a']) == set()
    assert set(comp.i['b']) == {'a'}
    assert set(comp.i['c']) == {'a'}
    assert set(comp.i['d']) == {'c', 'b'}
    assert list(map(set, comp.i[['a', 'b', 'c', 'd']])) == [set(), {'a'}, {'a'}, {'b', 'c'}]


def test_get_inputs_order():
    comp = Computation()
    input_nodes = list(('inp', i) for i in range(100))
    comp.add_node(input_node for input_node in input_nodes)
    random.shuffle(input_nodes)
    comp.add_node('res', lambda *args: args, args=input_nodes, inspect=False)
    assert comp.i.res == input_nodes


def test_get_original_inputs():
    comp = BasicFourNodeComputation()
    assert set(comp.get_original_inputs()) == {'a'}
    assert set(comp.get_original_inputs('a')) == {'a'}
    assert set(comp.get_original_inputs('b')) == {'a'}
    assert set(comp.get_original_inputs(['b', 'c'])) == {'a'}

    comp.add_node('a', lambda: 1)
    assert set(comp.get_original_inputs()) == set()
