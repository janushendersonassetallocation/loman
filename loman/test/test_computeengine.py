from loman import Computation, States, MapException
import six
from collections import namedtuple
import random


def test_basic():
    def b(a):
        return a + 1

    def c(a):
        return 2 * a

    def d(b, c):
        return b + c

    cpu = Computation()
    cpu.add_node("a")
    cpu.add_node("b", b)
    cpu.add_node("c", c)
    cpu.add_node("d", d)

    assert cpu.state('a') == States.UNINITIALIZED
    assert cpu.state('c') == States.UNINITIALIZED
    assert cpu.state('b') == States.UNINITIALIZED
    assert cpu.state('d') == States.UNINITIALIZED

    cpu.insert("a", 1)
    assert cpu.state('a') == States.UPTODATE
    assert cpu.state('b') == States.COMPUTABLE
    assert cpu.state('c') == States.COMPUTABLE
    assert cpu.state('d') == States.STALE
    assert cpu.value('a') == 1

    cpu.compute_all()
    assert cpu.state('a') == States.UPTODATE
    assert cpu.state('b') == States.UPTODATE
    assert cpu.state('c') == States.UPTODATE
    assert cpu.state('d') == States.UPTODATE
    assert cpu.value('a') == 1
    assert cpu.value('b') == 2
    assert cpu.value('c') == 2
    assert cpu.value('d') == 4

    cpu.insert("a", 2)
    cpu.compute("b")
    assert cpu.state('a') == States.UPTODATE
    assert cpu.state('b') == States.UPTODATE
    assert cpu.state('c') == States.COMPUTABLE
    assert cpu.state('d') == States.STALE
    assert cpu.value('a') == 2
    assert cpu.value('b') == 3

    assert set(cpu._get_calc_nodes("d")) == set(['c', 'd'])


def test_defined_sources():
    def b(x):
        return x + 1

    def c(x):
        return 2 * x

    def d(x, y):
        return x + y

    cpu = Computation()
    cpu.add_node("a")
    cpu.add_node("b", b, {'x': 'a'})
    cpu.add_node("c", c, {'x': 'a'})
    cpu.add_node("d", d, {'x': 'b', 'y': 'c'})

    cpu.insert("a", 1)
    cpu.compute_all()
    assert cpu.state('a') == States.UPTODATE
    assert cpu.state('b') == States.UPTODATE
    assert cpu.state('c') == States.UPTODATE
    assert cpu.state('d') == States.UPTODATE
    assert cpu.value('a') == 1
    assert cpu.value('b') == 2
    assert cpu.value('c') == 2
    assert cpu.value('d') == 4


def test_serialization():
    def b(x):
        return x + 1

    def c(x):
        return 2 * x

    def d(x, y):
        return x + y

    cpu = Computation()
    cpu.add_node("a")
    cpu.add_node("b", b, {'x': 'a'})
    cpu.add_node("c", c, {'x': 'a'})
    cpu.add_node("d", d, {'x': 'b', 'y': 'c'})

    cpu.insert("a", 1)
    cpu.compute_all()
    f = six.BytesIO()
    cpu.write_dill(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(cpu.dag.nodes()) == set(foo.dag.nodes())
    for n in cpu.dag.nodes():
        assert cpu.dag.node[n].get('state', None) == foo.dag.node[n].get('state', None)
        assert cpu.dag.node[n].get('value', None) == foo.dag.node[n].get('value', None)


def test_namedtuple_expansion():
    cpu = Computation()
    Coordinate = namedtuple("Coordinate", ['x', 'y'])
    cpu.add_node("a")
    cpu.add_named_tuple_expansion("a", Coordinate)
    cpu.insert("a", Coordinate(1, 2))
    cpu.compute_all()
    assert cpu.value("a.x") == 1
    assert cpu.value("a.y") == 2


def test_zero_parameter_functions():
    cpu = Computation()

    def a():
        return 1
    cpu.add_node('a', a)
    assert cpu.state('a') == States.COMPUTABLE

    cpu.compute_all()
    assert cpu.state('a') == States.UPTODATE
    assert cpu.value('a') == 1


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
                comp.add_node(i, f, {"x": prev})
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


def test_get_df():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()
    df = comp.get_df()

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
        comp.add_node(('fib', i), add, {'a': ('fib', i - 2), 'b': ('fib', i - 1)})

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
    comp=Computation()
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