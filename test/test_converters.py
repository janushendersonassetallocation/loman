from loman import Computation, States


def test_conversion_on_add_node():
    comp = Computation()
    comp.add_node('a', value=1, converter=float)
    assert comp.s.a == States.UPTODATE and isinstance(comp.v.a, float) and comp.v.a == 1.0


def test_conversion_on_insert():
    comp = Computation()
    comp.add_node('a', converter=float)
    comp.insert('a', 1)
    assert comp.s.a == States.UPTODATE and isinstance(comp.v.a, float) and comp.v.a == 1.0


def test_conversion_on_computation():
    comp = Computation()
    comp.add_node('a')
    comp.add_node('b', lambda a: a + 1, converter=float)
    comp.insert('a', 1)
    comp.compute_all()
    assert comp.s.b == States.UPTODATE and isinstance(comp.v.b, float) and comp.v.b == 2.0
