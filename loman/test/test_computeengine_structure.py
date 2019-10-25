import random

from loman import Computation, States
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


def test_get_outputs():
    comp = BasicFourNodeComputation()
    assert set(comp.get_outputs('a')) == {'c', 'b'}
    assert set(comp.get_outputs('b')) == {'d'}
    assert set(comp.get_outputs('c')) == {'d'}
    assert set(comp.get_outputs('d')) == set()
    assert list(map(set, comp.get_outputs(['a', 'b', 'c', 'd']))) == [{'b', 'c'}, {'d'}, {'d'}, set()]


def test_attribute_o():
    comp = BasicFourNodeComputation()
    assert set(comp.o.a) == {'c', 'b'}
    assert set(comp.o.b) == {'d'}
    assert set(comp.o.c) == {'d'}
    assert set(comp.o.d) == set()
    assert set(comp.o['a']) == {'c', 'b'}
    assert set(comp.o['b']) == {'d'}
    assert set(comp.o['c']) == {'d'}
    assert set(comp.o['d']) == set()
    assert list(map(set, comp.o[['a', 'b', 'c', 'd']])) == [{'b', 'c'}, {'d'}, {'d'}, set()]


def test_get_final_outputs():
    comp = BasicFourNodeComputation()
    assert set(comp.get_final_outputs()) == {'d'}
    assert set(comp.get_final_outputs('a')) == {'d'}
    assert set(comp.get_final_outputs('b')) == {'d'}
    assert set(comp.get_final_outputs(['b', 'c'])) == {'d'}


def test_restrict_1():
    comp = BasicFourNodeComputation()
    comp.restrict('c')
    assert set(comp.nodes()) == {'a', 'c'}


def test_restrict_2():
    comp = BasicFourNodeComputation()
    comp.restrict(['b', 'c'])
    assert set(comp.nodes()) == {'a', 'b', 'c'}


def test_restrict_3():
    comp = BasicFourNodeComputation()
    comp.restrict('d', ['b', 'c'])
    assert set(comp.nodes()) == {'b', 'c', 'd'}


def test_rename_nodes():
    comp = BasicFourNodeComputation()
    comp.insert('a', 10)
    comp.compute('b')

    comp.rename_node('a', 'alpha')
    comp.rename_node('b', 'beta')
    comp.rename_node('c', 'gamma')
    comp.rename_node('d', 'delta')
    assert comp.s[['alpha', 'beta', 'gamma', 'delta']] == \
           [States.UPTODATE, States.UPTODATE, States.COMPUTABLE, States.STALE]

    comp.compute('delta')
    assert comp.s[['alpha', 'beta', 'gamma', 'delta']] == \
           [States.UPTODATE, States.UPTODATE, States.UPTODATE, States.UPTODATE]


def test_rename_nodes_with_dict():
    comp = BasicFourNodeComputation()
    comp.insert('a', 10)
    comp.compute('b')

    comp.rename_node({'a': 'alpha', 'b': 'beta', 'c': 'gamma', 'd': 'delta'})
    assert comp.s[['alpha', 'beta', 'gamma', 'delta']] == \
           [States.UPTODATE, States.UPTODATE, States.COMPUTABLE, States.STALE]

    comp.compute('delta')
    assert comp.s[['alpha', 'beta', 'gamma', 'delta']] == \
           [States.UPTODATE, States.UPTODATE, States.UPTODATE, States.UPTODATE]


def test_state_map_updated_with_placeholder():
    comp = Computation()
    comp.add_node('b', lambda a: a+1)
    assert comp.s.a == States.PLACEHOLDER
    assert 'a' in comp._state_map[States.PLACEHOLDER]


def test_state_map_updated_with_placeholder_kwds():
    comp = Computation()
    comp.add_node('b', lambda x: x+1, kwds={'x': 'a'})
    assert comp.s.a == States.PLACEHOLDER
    assert 'a' in comp._state_map[States.PLACEHOLDER]


def test_state_map_updated_with_placeholder_args():
    comp = Computation()
    comp.add_node('b', lambda x: x+1, args=['a'])
    assert comp.s.a == States.PLACEHOLDER
    assert 'a' in comp._state_map[States.PLACEHOLDER]