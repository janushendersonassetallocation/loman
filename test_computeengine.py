from computeengine import Computation, States
import six
from collections import namedtuple


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
    f = six.StringIO()
    cpu.write_pickle(f)

    f.seek(0)
    foo = Computation.read_pickle(f)

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
    assert comp.exception('b').message == "Infinite sadness"