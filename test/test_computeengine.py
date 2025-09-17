from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep

import pandas as pd

from loman import (Computation, States, MapException, LoopDetectedException, NonExistentNodeException, node, C,
                   CannotInsertToPlaceholderNodeException)
from collections import namedtuple
import random
import pytest

from loman.computeengine import NodeData, NodeKey
from loman.nodekey import to_nodekey


def test_basic():
    def b(a):
        return a + 1

    def c(a):
        return 2 * a

    def d(b, c):
        return b + c

    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", b)
    comp.add_node("c", c)
    comp.add_node("d", d)

    assert comp.state('a') == States.UNINITIALIZED
    assert comp.state('c') == States.UNINITIALIZED
    assert comp.state('b') == States.UNINITIALIZED
    assert comp.state('d') == States.UNINITIALIZED
    assert comp._get_names_for_state(States.UNINITIALIZED) == set(['a', 'b', 'c', 'd'])

    comp.insert("a", 1)
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.COMPUTABLE
    assert comp.state('c') == States.COMPUTABLE
    assert comp.state('d') == States.STALE
    assert comp.value('a') == 1

    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.state('d') == States.UPTODATE
    assert comp.value('a') == 1
    assert comp.value('b') == 2
    assert comp.value('c') == 2
    assert comp.value('d') == 4

    comp.insert("a", 2)
    comp.compute("b")
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.COMPUTABLE
    assert comp.state('d') == States.STALE
    assert comp.value('a') == 2
    assert comp.value('b') == 3

    assert set(comp._get_calc_node_names("d")) == set(['c', 'd'])


def test_parameter_mapping():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda x: x + 1, kwds={'x': 'a'})
    comp.insert('a', 1)
    comp.compute_all()
    assert comp.state('b') == States.UPTODATE
    assert comp.value('b') == 2


def test_parameter_mapping_2():
    def b(x):
        return x + 1

    def c(x):
        return 2 * x

    def d(x, y):
        return x + y

    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", b, kwds={'x': 'a'})
    comp.add_node("c", c, kwds={'x': 'a'})
    comp.add_node("d", d, kwds={'x': 'b', 'y': 'c'})

    comp.insert("a", 1)
    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.state('d') == States.UPTODATE
    assert comp.value('a') == 1
    assert comp.value('b') == 2
    assert comp.value('c') == 2
    assert comp.value('d') == 4


def test_namedtuple_expansion():
    comp = Computation()
    Coordinate = namedtuple("Coordinate", ['x', 'y'])
    comp.add_node("a")
    comp.add_named_tuple_expansion("a", Coordinate)
    comp.insert("a", Coordinate(1, 2))
    comp.compute_all()
    assert comp.value("a.x") == 1
    assert comp.value("a.y") == 2


def test_zero_parameter_functions():
    comp = Computation()

    def a():
        return 1
    comp.add_node('a', a)
    assert comp.state('a') == States.COMPUTABLE

    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.value('a') == 1


def test_change_structure():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda c: 10 * c)
    comp.insert('a', 10)
    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, 200)

    comp.add_node('d', lambda b: 5 * b)
    assert comp.state('d') == States.COMPUTABLE

    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, 55)


def test_exceptions():
    def b(a):
        raise Exception("Infinite sadness")
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', b)
    comp.insert('a', 1)
    comp.compute_all()

    assert comp.state('b') == States.ERROR
    assert str(comp.value('b').exception) == "Infinite sadness"


def test_exceptions_2():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b+1)

    comp.insert('a', 1)
    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE
    assert comp.value('a') == 1

    comp.add_node('b', lambda a: a+1)
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.COMPUTABLE
    assert comp.state('c') == States.STALE
    assert comp.value('a') == 1

    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.value('a') == 1
    assert comp.value('b') == 2
    assert comp.value('c') == 3


def test_exception_compute_all():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


def test_raising_exception_compute():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    with pytest.raises(ZeroDivisionError):
        comp.compute_all(raise_exceptions=True)


def test_exception_compute():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    comp.compute('c')
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


def test_raising_exception_compute_all():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    with pytest.raises(ZeroDivisionError):
        comp.compute('c', raise_exceptions=True)


def test_update_function():
    def b1(a):
        return a + 1
    def b2(a):
        return a + 2
    def c(b):
        return 10 * b
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', b1)
    comp.add_node('c', c)

    comp.insert('a', 1)

    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.value('a') == 1
    assert comp.value('b') == 2
    assert comp.value('c') == 20

    comp.add_node('b', b2)
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.COMPUTABLE
    assert comp.state('c') == States.STALE
    assert comp.value('a') == 1

    comp.compute_all()
    assert comp.state('a') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.value('a') == 1
    assert comp.value('b') == 3
    assert comp.value('c') == 30


def test_update_function_with_structure_change():
    def b1(a1):
        return a1 + 1
    def b2(a2):
        return a2 + 2
    def c(b):
        return 10 * b
    comp = Computation()
    comp.add_node('a1')
    comp.add_node('a2')
    comp.add_node('b', b1)
    comp.add_node('c', c)

    comp.insert('a1', 1)
    comp.insert('a2', 2)

    comp.compute_all()
    assert comp.state('a1') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.value('a1') == 1
    assert comp.value('b') == 2
    assert comp.value('c') == 20

    comp.add_node('b', b2)
    assert comp.state('a2') == States.UPTODATE
    assert comp.state('b') == States.COMPUTABLE
    assert comp.state('c') == States.STALE
    assert comp.value('a2') == 2

    comp.compute_all()
    assert comp.state('a2') == States.UPTODATE
    assert comp.state('b') == States.UPTODATE
    assert comp.state('c') == States.UPTODATE
    assert comp.value('a2') == 2
    assert comp.value('b') == 4
    assert comp.value('c') == 40


def test_copy():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1)

    comp.insert("a", 1)
    comp.compute_all()
    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2

    comp2 = comp.copy()
    comp2.insert("a", 5)
    comp2.compute_all()
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.UPTODATE
    assert comp2.value("a") == 5
    assert comp2.value("b") == 6

    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2


def test_copy_2():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)

    comp2 = comp.copy()
    assert comp2['a'] == NodeData(States.UPTODATE, 1)
    assert comp2.state('b') == States.COMPUTABLE

    comp2.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.COMPUTABLE
    assert comp2['a'] == NodeData(States.UPTODATE, 1)
    assert comp2['b'] == NodeData(States.UPTODATE, 2)


def test_insert_many():
    comp = Computation()
    l = list(range(100))
    random.shuffle(l)
    prev = None
    for x in l:
        if prev is None:
            comp.add_node(x)
        else:
            comp.add_node(x, lambda n: n+1, kwds={'n': prev})
        prev = x
    comp.insert_many([(x, x) for x in range(100)])
    for x in range(100):
        assert comp[x] == NodeData(States.UPTODATE, x)


def test_insert_from():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1, serialize=False)
    comp.add_node("c", lambda b: b + 1)

    comp.insert("a", 1)
    comp2 = comp.copy()

    comp.compute_all()
    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.state("c") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2
    assert comp.value("c") == 3
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.COMPUTABLE
    assert comp2.state("c") == States.STALE
    assert comp2.value("a") == 1

    comp2.insert_from(comp, ['a', 'c'])
    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.state("c") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2
    assert comp.value("c") == 3
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.COMPUTABLE
    assert comp2.state("c") == States.UPTODATE
    assert comp2.value("a") == 1
    assert comp2.value("c") == 3


def test_insert_from_large():
    def make_chain(comp, f, l):
        prev = None
        for i in l:
            if prev is None:
                comp.add_node(i)
            else:
                comp.add_node(i, f, kwds={"x": prev})
            prev = i

    def add_one(x):
        return x + 1

    comp1 = Computation()
    make_chain(comp1, add_one, range(100))
    comp1.insert(0, 0)
    comp1.compute_all()

    for i in range(100):
        assert comp1.state(i) == States.UPTODATE
        assert comp1.value(i) == i

    comp2 = Computation()
    l1 = list(range(100))
    random.shuffle(l1)
    make_chain(comp2, add_one, l1)

    comp2.insert_from(comp1)
    for i in range(100):
        assert comp2.state(i) == States.UPTODATE
        assert comp2.value(i) == i


def test_to_df():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    df = comp.to_df()

    assert df.loc['a', 'value'] == 1
    assert df.loc['a', 'state'] == States.UPTODATE
    assert df.loc['b', 'value'] == 2
    assert df.loc['b', 'state'] == States.UPTODATE


def test_tuple_node_key():
    def add(a, b):
        return a + b

    comp = Computation()
    comp.add_node(('fib', 1))
    comp.add_node(('fib', 2))
    for i in range(3, 11):
        comp.add_node(('fib', i), add, kwds={'a': ('fib', i - 2), 'b': ('fib', i - 1)})

    comp.insert(('fib', 1), 0)
    comp.insert(('fib', 2), 1)
    comp.compute_all()

    assert comp.value(('fib', 10)) == 34


def test_get_item():
    comp = Computation()
    comp.add_node('a', lambda: 1)
    comp.add_node('b', lambda a: a + 1)
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)


def test_set_stale():
    comp = Computation()
    comp.add_node('a', lambda: 1)
    comp.add_node('b', lambda a: a + 1)
    comp.compute_all()

    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)

    comp.set_stale('a')
    assert comp.state('a') == States.COMPUTABLE
    assert comp.state('b') == States.STALE

    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)


def test_error_stops_compute_all():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


def test_error_stops_compute():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
    comp.compute('c')
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


def test_map_graph():
    subcomp = Computation()
    subcomp.add_node('a')
    subcomp.add_node('b', lambda a: 2*a)
    comp = Computation()
    comp.add_node('inputs')
    comp.add_map_node('results', 'inputs', subcomp, 'a', 'b')
    comp.insert('inputs', [1, 2, 3])
    comp.compute_all()
    assert comp['results'] == NodeData(States.UPTODATE, [2, 4, 6])


def test_map_graph_error():
    subcomp = Computation()
    subcomp.add_node('a')
    subcomp.add_node('b', lambda a: 1/(a-2))
    comp=Computation()
    comp.add_node('inputs')
    comp.add_map_node('results', 'inputs', subcomp, 'a', 'b')
    comp.insert('inputs', [1, 2, 3])
    comp.compute_all()
    assert comp.state('results') == States.ERROR
    assert isinstance(comp.value('results').exception, MapException)
    results = comp.value('results').exception.results
    assert results[0] == -1
    assert results[2] == 1
    assert isinstance(results[1], Computation)
    failed_graph = results[1]
    assert failed_graph.state('b') == States.ERROR

def test_placeholder():
    comp = Computation()
    comp.add_node('b', lambda a: a + 1)
    assert comp.state('a') == States.PLACEHOLDER
    assert comp.state('b') == States.UNINITIALIZED
    comp.add_node('a')
    assert comp.state('a') == States.UNINITIALIZED
    assert comp.state('b') == States.UNINITIALIZED
    comp.insert('a', 1)
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp.state('b') == States.COMPUTABLE
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)


def test_delete_predecessor():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)
    comp.delete_node('a')
    assert comp.state('a') == States.PLACEHOLDER
    assert comp['b'] == NodeData(States.UPTODATE, 2)
    comp.delete_node('b')
    assert list(comp.dag.nodes()) == []


def test_delete_successor():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert comp['b'] == NodeData(States.UPTODATE, 2)
    comp.delete_node('b')
    assert comp['a'] == NodeData(States.UPTODATE, 1)
    assert list(comp.nodes()) == ['a']
    comp.delete_node('a')
    assert list(comp.nodes()) == []


def test_value():
    comp = Computation()
    comp.add_node('a', value=10)
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda c: 10 * c)
    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, 200)


def test_args():
    def f(*args):
        return sum(args)
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', value=1)
    comp.add_node('c', value=1)
    comp.add_node('d', f, args=['a', 'b', 'c'])
    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, 3)


def test_kwds():
    def f(**kwds):
        return set(kwds.keys()), sum(kwds.values())
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', value=1)
    comp.add_node('c', value=1)
    comp.add_node('d', f, kwds={'a': 'a', 'b': 'b', 'c': 'c'})
    assert comp.state('d') == States.COMPUTABLE
    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, ({'a', 'b', 'c'}, 3))


def test_args_and_kwds():
    def f(a, b, c, *args, **kwds):
        return locals()
    comp = Computation()
    comp.add_node('a', value='a')
    comp.add_node('b', value='b')
    comp.add_node('c', value='c')
    comp.add_node('p', value='p')
    comp.add_node('q', value='q')
    comp.add_node('r', value='r')
    comp.add_node('x', value='x')
    comp.add_node('y', value='y')
    comp.add_node('z', value='z')
    comp.add_node('res', func=f, args=['a', 'b', 'c', 'p', 'q', 'r'], kwds={'x': 'x', 'y': 'y', 'z': 'z'})
    comp.compute_all()
    assert comp.value('res') == {'a': 'a', 'b': 'b', 'c': 'c',
                                 'args': ('p', 'q', 'r'),
                                 'kwds': {'x': 'x', 'y': 'y', 'z': 'z'}}


def test_avoid_infinite_loop_compute_all():
    comp = Computation()
    comp.add_node('a', lambda c: c+1)
    comp.add_node('b', lambda a: a+1)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
    with pytest.raises(LoopDetectedException):
        comp.compute_all()


def test_views():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1)
    comp.add_node("c", lambda a: 2 * a)
    comp.add_node("d", lambda b, c: b + c)

    assert comp.s.a == States.UNINITIALIZED
    assert comp.s.b == States.UNINITIALIZED
    assert comp.s.c == States.UNINITIALIZED
    assert comp.s.d == States.UNINITIALIZED

    comp.insert("a", 1)
    assert comp.s.a == States.UPTODATE
    assert comp.s.b == States.COMPUTABLE
    assert comp.s.c == States.COMPUTABLE
    assert comp.s.d == States.STALE
    assert comp.v.a == 1

    comp.compute_all()
    assert comp.s.a == States.UPTODATE
    assert comp.s.b == States.UPTODATE
    assert comp.s.c == States.UPTODATE
    assert comp.s.d == States.UPTODATE
    assert comp.v.a == 1
    assert comp.v.b == 2
    assert comp.v.c == 2
    assert comp.v.d == 4


def test_view_x():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1)
    comp.add_node("c", lambda a: 2 * a)
    comp.add_node("d", lambda b, c: b + c)

    assert comp.s.a == States.UNINITIALIZED
    assert comp.s.b == States.UNINITIALIZED
    assert comp.s.c == States.UNINITIALIZED
    assert comp.s.d == States.UNINITIALIZED

    comp.insert("a", 10)
    assert comp.x.d == 31


def test_view_x_exception():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1)
    comp.add_node("c", lambda a: a / 0)
    comp.add_node("d", lambda b, c: b + c)

    assert comp.s.a == States.UNINITIALIZED
    assert comp.s.b == States.UNINITIALIZED
    assert comp.s.c == States.UNINITIALIZED
    assert comp.s.d == States.UNINITIALIZED

    comp.insert("a", 10)
    with pytest.raises(ZeroDivisionError):
        assert comp.x.d == 31


def test_delete_nonexistent_causes_exception():
    comp = Computation()
    with pytest.raises(NonExistentNodeException):
        comp.delete_node('a')


def test_insert_nonexistent_causes_exception():
    comp = Computation()
    with pytest.raises(NonExistentNodeException):
        comp.insert('a', 1)


def test_insert_many_nonexistent_causes_exception():
    comp = Computation()
    comp.add_node('a')
    comp.insert('a', 0)

    with pytest.raises(NonExistentNodeException):
        comp.insert_many([('a', 1), ('b', 2)])

    assert comp.v.a == 0


def test_no_inspect():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1, kwds={'a': 'a'}, inspect=False)
    comp.add_node('c', lambda a: 2 * a, kwds={'a': 'a'}, inspect=False)
    comp.add_node('d', lambda b, c: b + c, kwds={'b': 'b', 'c': 'c'}, inspect=False)
    comp.insert('a', 10)
    comp.compute_all()
    assert comp['b'] == NodeData(States.UPTODATE, 11)
    assert comp['c'] == NodeData(States.UPTODATE, 20)
    assert comp['d'] == NodeData(States.UPTODATE, 31)


def test_compute_fib_5():
    n = 5

    comp = Computation()

    def add(x, y):
        return x + y

    comp.add_node(0, value=1)
    comp.add_node(1, value=1)

    for i in range(2, n + 1):
        comp.add_node(i, add, kwds={'x': i - 2, 'y': i - 1}, inspect=False)

    comp.compute(n)
    assert comp.state(n) == States.UPTODATE


def test_multiple_values():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1, kwds={'a': 'a'}, inspect=False)
    comp.add_node('c', lambda a: 2 * a, kwds={'a': 'a'}, inspect=False)
    comp.add_node('d', lambda b, c: b + c, kwds={'b': 'b', 'c': 'c'}, inspect=False)
    comp.insert('a', 10)
    comp.compute_all()
    assert comp.value(['d', 'b']) == [31, 11]
    assert comp.v[['d', 'b']] == [31, 11]


def test_compute_with_unpicklable_object():
    class Unpicklable:
        def __getstate__(self):
            raise Exception("UNPICKLABLE")
    comp = Computation()
    comp.add_node('a', value=Unpicklable())
    comp.add_node('b', lambda a: None)
    comp.compute('b')


def test_compute_with_args():
    comp = Computation()
    comp.add_node('a', value=1)

    def f(foo):
        return foo + 1

    comp.add_node('b', f, args=['a'])

    assert set(comp.nodes()) == {'a', 'b'}
    assert set(comp.dag.edges()) == {(to_nodekey('a'), to_nodekey('b'))}

    comp.compute_all()
    assert comp['b'] == NodeData(States.UPTODATE, 2)


def test_default_args():
    comp = Computation()
    comp.add_node('x', value=1)
    def bar(x, some_default=0):
        return x + some_default
    comp.add_node('foo', bar)
    assert set(comp.i.foo) == {'x'}
    comp.compute('foo')
    assert comp.v.foo == 1


def test_default_args_2():
    comp = Computation()
    comp.add_node('x', value=1)
    comp.add_node('some_default', value=1)
    def bar(x, some_default=0):
        return x + some_default
    comp.add_node('foo', bar)
    assert set(comp.i.foo) == {'x', 'some_default'}
    comp.compute('foo')
    assert comp.v.foo == 2


def test_add_node_with_value_sets_descendents_stale():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', value=2)
    comp.add_node('c', lambda a, b: a + b)
    comp.compute_all()
    assert comp.s.a == States.UPTODATE
    assert comp.s.b == States.UPTODATE
    assert comp.s.c == States.UPTODATE
    comp.add_node('a', value=3)
    assert comp.s.a == States.UPTODATE
    assert comp.s.b == States.UPTODATE
    assert comp.s.c == States.COMPUTABLE


def test_tags():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.set_tag('a', 'foo')
    assert 'foo' in comp.t.a
    assert 'foo' in comp.tags('a')
    assert 'a' in comp.nodes_by_tag('foo')
    comp.clear_tag('a', 'foo')
    assert 'foo' not in comp.t.a
    assert 'foo' not in comp.tags('a')
    assert 'a' not in comp.nodes_by_tag('foo')
    comp.set_tag('a', ['foo'])
    assert 'foo' in comp.t.a
    assert 'foo' in comp.tags('a')
    assert 'a' in comp.nodes_by_tag('foo')
    comp.clear_tag('a', ['foo'])
    assert 'foo' not in comp.t.a
    assert 'foo' not in comp.tags('a')
    assert 'a' not in comp.nodes_by_tag('foo')

    # This should not throw
    comp.clear_tag('a', 'bar')
    comp.clear_tag('a', ['bar'])

    # This should not throw
    comp.set_tag('a', ['foo'])
    comp.set_tag('a', ['foo'])
    assert 'foo' in comp.t.a

    # This should not throw
    comp.clear_tag('a', ['foo'])
    comp.clear_tag('a', ['foo'])
    assert 'foo' not in comp.t.a

    assert comp.nodes_by_tag('baz') == set()

    comp.set_tag(['a', 'b'], ['foo', 'bar'])
    assert 'foo' in comp.t.a
    assert 'bar' in comp.t.a
    assert 'foo' in comp.t.b
    assert 'bar' in comp.t.b
    assert {'a', 'b'} == comp.nodes_by_tag('foo')
    assert {'a', 'b'} == comp.nodes_by_tag('bar')
    assert {'a', 'b'} == comp.nodes_by_tag(['foo', 'bar'])

    comp.add_node('c', lambda a: 2 * a, tags=['baz'])
    assert 'baz' in comp.t.c
    assert {'c'} == comp.nodes_by_tag('baz')


def test_set_and_clear_multiple_tags():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.set_tag(['a', 'b'], ['foo', 'bar'])
    assert 'foo' in comp.t.a
    assert 'foo' in comp.t.b
    assert 'bar' in comp.t.a
    assert 'bar' in comp.t.b
    comp.clear_tag(['a', 'b'], ['foo', 'bar'])
    assert 'foo' not in comp.t.a
    assert 'foo' not in comp.t.b
    assert 'bar' not in comp.t.a
    assert 'bar' not in comp.t.b


def test_decorator():
    comp = Computation()
    comp.add_node('a', value=1)

    @node(comp)
    def b(a):
        return a + 1

    comp.compute_all()
    assert comp['b'] == NodeData(States.UPTODATE, 2)

    @node(comp, name='c', args=['b'])
    def foo(x):
        return x + 1

    comp.compute_all()
    assert comp['c'] == NodeData(States.UPTODATE, 3)

    @node(comp, kwds={'x': 'b', 'y': 'c'})
    def d(x, y):
        return x + y

    comp.compute_all()
    assert comp['d'] == NodeData(States.UPTODATE, 5)


def test_with_uptodate_predecessors_but_stale_ancestors():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a + 1)
    comp.compute_all()
    assert comp['b'] == NodeData(States.UPTODATE, 2)
    comp.dag.nodes[to_nodekey('a')]['state'] = States.UNINITIALIZED # This can happen due to serialization
    comp.add_node('c', lambda b: b + 1)
    comp.compute('c')
    assert comp['b'] == NodeData(States.UPTODATE, 2)
    assert comp['c'] == NodeData(States.UPTODATE, 3)


def test_constant_values():
    comp = Computation()
    comp.add_node('a', value=1)

    def add(x, y):
        return x + y

    comp.add_node('b', add, args=['a', C(2)])
    comp.add_node('c', add, args=[C(3), 'a'])
    comp.add_node('d', add, kwds={'x': C(4), 'y': 'a'})
    comp.add_node('e', add, kwds={'y': C(5), 'x': 'a'})

    comp.compute_all()

    assert comp.dag.nodes[to_nodekey('b')]['args'] == {1: 2}

    assert comp['b'] == NodeData(States.UPTODATE, 3)
    assert comp['c'] == NodeData(States.UPTODATE, 4)
    assert comp['d'] == NodeData(States.UPTODATE, 5)
    assert comp['e'] == NodeData(States.UPTODATE, 6)


def test_compute_multiple():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda b, c: b + c)
    comp.compute(['b', 'c'])
    assert comp.s.b == States.UPTODATE
    assert comp.s.c == States.UPTODATE
    assert comp.s.d != States.UPTODATE


def test_state_map_with_adding_existing_node():
    comp = Computation()
    comp.add_node('a', lambda: 1)
    assert comp._get_names_for_state(States.COMPUTABLE) == {'a'}
    assert comp._get_names_for_state(States.UPTODATE) == set()
    comp.compute('a')
    assert comp._get_names_for_state(States.COMPUTABLE) == set()
    assert comp._get_names_for_state(States.UPTODATE) == {'a'}
    comp.add_node('a', lambda: 1)
    assert comp._get_names_for_state(States.COMPUTABLE) == {'a'}
    assert comp._get_names_for_state(States.UPTODATE) == set()
    comp.compute('a')
    assert comp._get_names_for_state(States.COMPUTABLE) == set()
    assert comp._get_names_for_state(States.UPTODATE) == {'a'}


def test_pinning():
    def add_one(x):
        return x + 1

    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', add_one, args=['a'])
    comp.add_node('c', add_one, args=['b'])
    comp.add_node('d', add_one, args=['c'])
    comp.compute_all()

    assert comp.v[['a', 'b', 'c', 'd']] == [1, 2, 3, 4]

    comp.pin('c')
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.PINNED, States.UPTODATE]

    comp.insert('a', 11)
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.COMPUTABLE, States.PINNED, States.UPTODATE]
    comp.compute_all()
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.PINNED, States.UPTODATE]
    assert comp.v[['a', 'b', 'c', 'd']] == [11, 12, 3, 4]

    comp.pin('c', 20)
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.PINNED, States.COMPUTABLE]
    assert comp.v.c == 20
    comp.compute_all()
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.PINNED, States.UPTODATE]
    assert comp.v[['a', 'b', 'c', 'd']] == [11, 12, 20, 21]

    comp.unpin('c')
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.COMPUTABLE, States.STALE]

    comp.compute_all()
    assert comp.s[['a', 'b', 'c', 'd']] == [States.UPTODATE, States.UPTODATE, States.UPTODATE, States.UPTODATE]
    assert comp.v[['a', 'b', 'c', 'd']] == [11, 12, 13, 14]


def test_add_node_with_none_value():
    comp = Computation()
    comp.add_node('a', value=None)
    assert comp.s.a == States.UPTODATE
    assert comp.v.a is None


def test_add_node_with_value_replacing_calculation_node():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a+1)
    comp.compute_all()
    comp.add_node('b', value=10)
    comp.compute_all()
    assert comp.s.b == States.UPTODATE
    assert comp.v.b == 10


def test_thread_pool_executor():
    sleep_time = 0.2
    n = 10
    def wait(c):
        sleep(sleep_time)
        return c

    comp = Computation(default_executor=ThreadPoolExecutor(n))
    start_dt = datetime.utcnow()
    for c in range(n):
        comp.add_node(c, wait, kwds={'c': C(c)})
    comp.compute_all()
    end_dt = datetime.utcnow()
    delta = (end_dt - start_dt).total_seconds()
    assert delta < (n-1) * sleep_time


def test_node_specific_thread_pool_executor():
    sleep_time = 0.2
    n = 10
    def wait(c):
        sleep(sleep_time)
        return c

    executor_map = {'foo': ThreadPoolExecutor(n)}
    comp = Computation(executor_map=executor_map)
    start_dt = datetime.utcnow()
    for c in range(n):
        comp.add_node(c, wait, kwds={'c': C(c)}, executor='foo')
    comp.compute_all()
    end_dt = datetime.utcnow()
    delta = (end_dt - start_dt).total_seconds()
    assert delta < (n-1) * sleep_time


def test_delete_node_with_placeholder_parent():
    comp = Computation()
    comp.add_node('b', lambda a: a)
    comp.delete_node('b')
    

def test_repoint():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a+1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp.v.b == 2
    
    comp.add_node('5a', lambda a: 5*a)
    comp.repoint('a', '5a')
    comp.compute_all()
    assert comp.v.b == 5*1+1


def test_repoint_missing_node():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a+1)
    comp.insert('a', 1)
    
    comp.repoint('a', 'new_a')
    assert comp.s.new_a == States.PLACEHOLDER


def test_insert_same_value_int():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a+1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp.s.b == States.UPTODATE

    comp.insert('a', 1)
    assert comp.s.b == States.UPTODATE

    comp.insert('a', 1, force=True)
    assert comp.s.b != States.UPTODATE

    comp.compute_all()
    assert comp.s.b == States.UPTODATE


def test_insert_same_value_df():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a+1)
    comp.insert('a', pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))
    comp.compute_all()
    assert comp.s.b == States.UPTODATE

    comp.insert('a', pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))
    assert comp.s.b == States.UPTODATE

    comp.insert('a', pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']), force=True)
    assert comp.s.b != States.UPTODATE

    comp.compute_all()
    assert comp.s.b == States.UPTODATE


def test_link():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b')
    comp.link('b', 'a')
    comp.insert('a', 5)
    comp.compute_all()
    assert comp.v.b == 5


def test_self_link():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a+1)
    comp.insert('a', 5)
    comp.compute_all()
    assert comp.v.b == 6
    comp.link('b', 'b')
    comp.compute_all()
    assert comp.v.b == 6


def test_self_link_with_paths():
    comp = Computation()
    comp.add_node('foo/a')
    comp.add_node('foo/b', lambda a: a+1)
    comp.insert('foo/a', 5)
    comp.compute_all()
    assert comp.v['foo/b'] == 6
    comp.link('foo/b', 'foo/b')
    comp.compute_all()
    assert comp.v['foo/b'] == 6
    comp.link(NodeKey(('foo', 'b')), 'foo/b')
    comp.compute_all()
    assert comp.v['foo/b'] == 6
    comp.link('foo/b', NodeKey(('foo', 'b')))
    comp.compute_all()
    assert comp.v['foo/b'] == 6


def test_args_kwds():
    comp = Computation()
    comp.add_node('a', value=1)

    def add(x, y):
        return x + y

    comp.add_node('b', add, args=['a', C(2)])
    comp.add_node('c', add, args=[C(3), 'a'])
    comp.add_node('d', add, kwds={'x': C(4), 'y': 'a'})
    comp.add_node('e', add, kwds={'y': C(5), 'x': 'a'})

    comp.compute_all()

    assert comp.get_definition_args_kwds('b') == (['a', C(2)], {})
    assert comp.get_definition_args_kwds('c') == ([C(3), 'a'], {})
    assert comp.get_definition_args_kwds('d') == ([], {'x': C(4), 'y': 'a'})
    assert comp.get_definition_args_kwds('e') == ([], {'y': C(5), 'x': 'a'})


def test_insert_fails_for_placeholder():
    comp = Computation()
    comp.add_node('b', lambda a: a + 1)
    with pytest.raises(CannotInsertToPlaceholderNodeException):
        comp.insert('a', value=1)


def test_get_source():
    comp = Computation()
    comp.add_node('a', value=1)

    @node(comp)
    def b(a):
        return a + 1

    src = comp.get_source('b')
    src_lines = [line.strip() for line in src.split('\n')]
    assert src_lines[1:] == ['', '@node(comp)', 'def b(a):', 'return a + 1', '']

def test_compute_and_get_value_raises_error_and_sets_state():
    comp = Computation()
    comp.add_node('a', value=1)

    @node(comp)
    def b(a):
        """Will error"""
        return a / 0
    with pytest.raises(ZeroDivisionError):
        b = comp.compute_and_get_value("b")

    assert comp.state("b") == States.ERROR

def test_view_x_raises_error_and_sets_state():
    comp = Computation()
    comp.add_node('a', value=1)

    @node(comp)
    def b(a):
        """Will error"""
        return a / 0

    with pytest.raises(ZeroDivisionError):
        b = comp.x.b

    assert comp.s.b == States.ERROR
