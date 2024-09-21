import io

from loman import Computation, States, ComputationFactory, input_node, calc_node
from loman.computeengine import NodeData


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
    f = io.BytesIO()
    comp.write_dill(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get('state', None) == foo.dag.nodes[n].get('state', None)
        assert comp.dag.nodes[n].get('value', None) == foo.dag.nodes[n].get('value', None)


def test_serialization_skip_flag():
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1, serialize=False)
    comp.add_node("c", lambda b: b + 1)

    comp.insert("a", 1)
    comp.compute_all()
    f = io.BytesIO()
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


def test_no_serialize_flag():
    comp = Computation()
    comp.add_node('a', serialize=False)
    comp.add_node('b', lambda a: a + 1)
    comp.insert('a', 1)
    comp.compute_all()

    f = io.BytesIO()
    comp.write_dill(f)
    f.seek(0)
    comp2 = Computation.read_dill(f)
    assert comp2.state('a') == States.UNINITIALIZED
    assert comp2['b'] == NodeData(States.UPTODATE, 2)


def test_serialize_nested_loman():
    @ComputationFactory
    class CompInner:
        a = input_node(value=3)

        @calc_node
        def b(self, a):
            return a + 1

    @ComputationFactory
    class CompOuter:
        COMP = input_node()

        @calc_node
        def out(self, COMP):
            return COMP.x.b + 10

    inner = CompInner()
    inner.compute_all()

    outer = CompOuter()
    outer.insert('COMP', inner)
    outer.compute_all()

    f = io.BytesIO()
    outer.write_dill(f)
    f.seek(0)
    outer2 = Computation.read_dill(f)

    assert outer2.v.COMP.v.b == outer.v.COMP.v.b
    assert outer2.v.out == outer.v.out


def test_roundtrip_old_dill():
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
    f = io.BytesIO()
    comp.write_dill_old(f)

    f.seek(0)
    foo = Computation.read_dill(f)

    assert set(comp.dag.nodes) == set(foo.dag.nodes)
    for n in comp.dag.nodes():
        assert comp.dag.nodes[n].get('state', None) == foo.dag.nodes[n].get('state', None)
        assert comp.dag.nodes[n].get('value', None) == foo.dag.nodes[n].get('value', None)