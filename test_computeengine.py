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
    cpu.add_node("b", b, ["a"])
    cpu.add_node("c", c, ["a"])
    cpu.add_node("d", d, ["b", "c"])

    assert cpu.dag.node['a']['state'] == States.UNINITIALIZED
    assert cpu.dag.node['b']['state'] == States.UNINITIALIZED
    assert cpu.dag.node['c']['state'] == States.UNINITIALIZED
    assert cpu.dag.node['d']['state'] == States.UNINITIALIZED

    cpu.insert("a", 1)
    assert cpu.dag.node['a']['state'] == States.UPTODATE
    assert cpu.dag.node['b']['state'] == States.COMPUTABLE
    assert cpu.dag.node['c']['state'] == States.COMPUTABLE
    assert cpu.dag.node['d']['state'] == States.STALE
    assert cpu.dag.node['a']['value'] == 1

    cpu.compute_all()
    assert cpu.dag.node['a']['state'] == States.UPTODATE
    assert cpu.dag.node['b']['state'] == States.UPTODATE
    assert cpu.dag.node['c']['state'] == States.UPTODATE
    assert cpu.dag.node['d']['state'] == States.UPTODATE
    assert cpu.dag.node['a']['value'] == 1
    assert cpu.dag.node['b']['value'] == 2
    assert cpu.dag.node['c']['value'] == 2
    assert cpu.dag.node['d']['value'] == 4

    cpu.insert("a", 2)
    cpu.compute("b")
    assert cpu.dag.node['a']['state'] == States.UPTODATE
    assert cpu.dag.node['b']['state'] == States.UPTODATE
    assert cpu.dag.node['c']['state'] == States.COMPUTABLE
    assert cpu.dag.node['d']['state'] == States.STALE
    assert cpu.dag.node['a']['value'] == 2
    assert cpu.dag.node['b']['value'] == 3

    assert set(cpu._get_calc_nodes("d")) == set(['c', 'd'])