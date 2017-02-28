from computeengine import Computation, States


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
