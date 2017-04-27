from loman import Computation, States, MapException, LoopDetectedException, NonExistentNodeException, node, C
import six
from collections import namedtuple
import random
from nose.tools import raises, assert_raises


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

    assert set(comp._get_calc_nodes("d")) == set(['c', 'd'])


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



def test_serialization():
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
    f = six.BytesIO()
    comp.write_dill(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(comp.dag.nodes()) == set(foo.dag.nodes())
    for n in comp.dag.nodes():
        assert comp.dag.node[n].get('state', None) == foo.dag.node[n].get('state', None)
        assert comp.dag.node[n].get('value', None) == foo.dag.node[n].get('value', None)


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
    assert comp['d'] == (States.UPTODATE, 200)

    comp.add_node('d', lambda b: 5 * b)
    assert comp.state('d') == States.COMPUTABLE

    comp.compute_all()
    assert comp['d'] == (States.UPTODATE, 55)


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
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


@raises(ZeroDivisionError)
def test_raising_exception_compute():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    comp.compute_all(raise_exceptions=True)


def test_exception_compute():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
    comp.compute('c')
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


@raises(ZeroDivisionError)
def test_raising_exception_compute_all():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b)
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
    assert comp2['a'] == (States.UPTODATE, 1)
    assert comp2.state('b') == States.COMPUTABLE

    comp2.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.state('b') == States.COMPUTABLE
    assert comp2['a'] == (States.UPTODATE, 1)
    assert comp2['b'] == (States.UPTODATE, 2)



def test_serialization_skip_flag():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1, serialize=False)
    comp.add_node("c", lambda b: b + 1)

    comp.insert("a", 1)
    comp.compute_all()
    f = six.BytesIO()
    comp.write_dill(f)

    assert comp.state("a") == States.UPTODATE
    assert comp.state("b") == States.UPTODATE
    assert comp.state("c") == States.UPTODATE
    assert comp.value("a") == 1
    assert comp.value("b") == 2
    assert comp.value("c") == 3

    f.seek(0)
    comp2 = Computation.read_dill(f)
    assert comp2.state("a") == States.UPTODATE
    assert comp2.state("b") == States.UNINITIALIZED
    assert comp2.state("c") == States.UPTODATE
    assert comp2.value("a") == 1
    assert comp2.value("c") == 3


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
        assert comp[x] == (States.UPTODATE, x)


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
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)


def test_set_stale():
    comp = Computation()
    comp.add_node('a', lambda: 1)
    comp.add_node('b', lambda a: a + 1)
    comp.compute_all()

    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)

    comp.set_stale('a')
    assert comp.state('a') == States.COMPUTABLE
    assert comp.state('b') == States.STALE

    comp.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)


def test_error_stops_compute_all():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.state('b') == States.ERROR
    assert comp.state('c') == States.STALE


def test_error_stops_compute():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a/0)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
    comp.compute('c')
    assert comp['a'] == (States.UPTODATE, 1)
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
    assert comp['results'] == (States.UPTODATE, [2, 4, 6])


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
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.state('b') == States.COMPUTABLE
    comp.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)


def test_delete_predecessor():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)
    comp.delete_node('a')
    assert comp.state('a') == States.PLACEHOLDER
    assert comp['b'] == (States.UPTODATE, 2)
    comp.delete_node('b')
    assert comp.dag.nodes() == []


def test_delete_successor():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp['b'] == (States.UPTODATE, 2)
    comp.delete_node('b')
    assert comp['a'] == (States.UPTODATE, 1)
    assert comp.dag.nodes() == ['a']
    comp.delete_node('a')
    assert comp.dag.nodes() == []


def test_no_serialize_flag():
    comp = Computation()
    comp.add_node('a', serialize=False)
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()

    f = six.BytesIO()
    comp.write_dill(f)
    f.seek(0)
    comp2 = Computation.read_dill(f)
    assert comp2.state('a') == States.UNINITIALIZED
    assert comp2['b'] == (States.UPTODATE, 2)


def test_value():
    comp = Computation()
    comp.add_node('a', value=10)
    comp.add_node('b', lambda a: a + 1)
    comp.add_node('c', lambda a: 2 * a)
    comp.add_node('d', lambda c: 10 * c)
    comp.compute_all()
    assert comp['d'] == (States.UPTODATE, 200)


def test_args():
    def f(*args):
        return sum(args)
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', value=1)
    comp.add_node('c', value=1)
    comp.add_node('d', f, args=['a', 'b', 'c'])
    comp.compute_all()
    assert comp['d'] == (States.UPTODATE, 3)


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
    assert comp['d'] == (States.UPTODATE, (set(['a', 'b', 'c']), 3))


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


@raises(LoopDetectedException)
def test_avoid_infinite_loop_compute_all():
    comp = Computation()
    comp.add_node('a', lambda c: c+1)
    comp.add_node('b', lambda a: a+1)
    comp.add_node('c', lambda b: b+1)
    comp.insert('a', 1)
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


@raises(NonExistentNodeException)
def test_delete_nonexistent_causes_exception():
    comp = Computation()
    comp.delete_node('a')


@raises(NonExistentNodeException)
def test_insert_nonexistent_causes_exception():
    comp = Computation()
    comp.insert('a', 1)


def test_insert_many_nonexistent_causes_exception():
    comp = Computation()
    comp.add_node('a')
    comp.insert('a', 0)

    with assert_raises(NonExistentNodeException):
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
    assert comp['b'] == (States.UPTODATE, 11)
    assert comp['c'] == (States.UPTODATE, 20)
    assert comp['d'] == (States.UPTODATE, 31)


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


def test_get_inputs():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1, kwds={'a': 'a'}, inspect=False)
    comp.add_node('c', lambda a: 2 * a, kwds={'a': 'a'}, inspect=False)
    comp.add_node('d', lambda b, c: b + c, kwds={'b': 'b', 'c': 'c'}, inspect=False)
    assert set(comp.get_inputs('a')) == set()
    assert set(comp.get_inputs('b')) == {'a'}
    assert set(comp.get_inputs('c')) == {'a'}
    assert set(comp.get_inputs('d')) == {'c', 'b'}
    assert list(map(set, comp.get_inputs(['a', 'b', 'c', 'd']))) == [set(), {'a'}, {'a'}, {'b', 'c'}]
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


def test_compute_with_unpicklable_object():
    class Unpicklable(object):
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
    assert set(comp.dag.edges_iter()) == {('a', 'b')}

    comp.compute_all()
    assert comp['b'] == (States.UPTODATE, 2)


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
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.set_tag('a', 'foo')
    assert 'foo' in comp.t.a
    assert 'foo' in comp.tags('a')
    comp.clear_tag('a', 'foo')
    assert 'foo' not in comp.t.a
    assert 'foo' not in comp.tags('a')
    comp.set_tags('a', ['foo'])
    assert 'foo' in comp.t.a
    assert 'foo' in comp.tags('a')
    comp.clear_tags('a', ['foo'])
    assert 'foo' not in comp.t.a
    assert 'foo' not in comp.tags('a')

    # This should not throw
    comp.clear_tag('a', 'bar')
    comp.clear_tags('a', ['bar'])

    # This should not throw
    comp.set_tags('a', ['foo'])
    comp.set_tags('a', ['foo'])
    assert 'foo'  in comp.t.a

    # This should not throw
    comp.clear_tags('a', ['foo'])
    comp.clear_tags('a', ['foo'])
    assert 'foo' not in comp.t.a

    comp.set_tags(['a', 'b'], ['foo', 'bar'])
    assert 'foo' in comp.t.a
    assert 'bar' in comp.t.a
    assert 'foo' in comp.t.b
    assert 'bar' in comp.t.b

    comp.add_node('c', lambda a: 2 * a, tags=['baz'])
    assert 'baz' in comp.t.c


def test_decorator():
    comp = Computation()
    comp.add_node('a', value=1)

    @node(comp)
    def b(a):
        return a + 1

    comp.compute_all()
    assert comp['b'] == (States.UPTODATE, 2)

    @node(comp, name='c', args=['b'])
    def foo(x):
        return x + 1

    comp.compute_all()
    assert comp['c'] == (States.UPTODATE, 3)

    @node(comp, kwds={'x': 'b', 'y': 'c'})
    def d(x, y):
        return x + y

    comp.compute_all()
    assert comp['d'] == (States.UPTODATE, 5)


def test_with_uptodate_predecessors_but_stale_ancestors():
    comp = Computation()
    comp.add_node('a', value=1)
    comp.add_node('b', lambda a: a + 1)
    comp.compute_all()
    assert comp['b'] == (States.UPTODATE, 2)
    comp.dag.node['a']['state'] = States.UNINITIALIZED # This can happen due to serialization
    comp.add_node('c', lambda b: b + 1)
    comp.compute('c')
    assert comp['b'] == (States.UPTODATE, 2)
    assert comp['c'] == (States.UPTODATE, 3)


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

    assert comp.dag.node['b']['args'] == {1: 2}

    assert comp['b'] == (States.UPTODATE, 3)
    assert comp['c'] == (States.UPTODATE, 4)
    assert comp['d'] == (States.UPTODATE, 5)
    assert comp['e'] == (States.UPTODATE, 6)

