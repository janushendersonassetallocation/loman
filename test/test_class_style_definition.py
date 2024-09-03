import pytest
from loman import (Computation, States, MapException, LoopDetectedException, NonExistentNodeException, node, C,
                   input_node, calc_node, ComputationFactory)


@pytest.mark.xfail
def test_class_style_definition():
    class FooComp():
        a = input_node(value=3)

        @calc_node
        def b(a):
            return a + 1

        @calc_node
        def c(a):
            return 2 * a

        @calc_node
        def d(b, c):
            return b + c

    comp = Computation(FooComp)
    comp.compute_all()

    assert comp.v.d == 10


@pytest.mark.xfail
def test_class_style_definition_as_decorator():
    @Computation
    class FooComp():
        a = input_node(value=3)

        @calc_node
        def b(a):
            return a + 1

        @calc_node
        def c(a):
            return 2 * a

        @calc_node
        def d(b, c):
            return b + c

    FooComp.compute_all()

    assert FooComp.v.d == 10


def test_class_style_definition_as_factory_decorator():
    @ComputationFactory
    class FooComp():
        a = input_node(value=3)

        @calc_node
        def b(a):
            return a + 1

        @calc_node
        def c(a):
            return 2 * a

        @calc_node
        def d(b, c):
            return b + c

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_class_style_definition_as_factory_decorator_with_args():
    @ComputationFactory()
    class FooComp():
        a = input_node(value=3)

        @calc_node
        def b(a):
            return a + 1

        @calc_node
        def c(a):
            return 2 * a

        @calc_node
        def d(b, c):
            return b + c

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_computation_factory_methods_ignore_self_by_default():
    @ComputationFactory
    class FooComp:
        a = input_node(value=3)

        @calc_node
        def b(self, a):
            return a + 1

        @calc_node
        def c(self, a):
            return 2 * a

        @calc_node
        def d(self, b, c):
            return b + c

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_computation_factory_methods_explicitly_use_self():
    @ComputationFactory(ignore_self=False)
    class FooComp:
        a = input_node(value=3)

        @calc_node
        def b(self, a):
            return a + 1

        @calc_node
        def c(self, a):
            return 2 * a

        @calc_node
        def d(self, b, c):
            return b + c

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UNINITIALIZED

    comp.add_node('self', value=None)
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_standard_computation_does_not_ignore_self():
    def b(self, a):
        return a + 1

    def c(self, a):
        return 2 * a

    def d(self, b, c):
        return b + c

    comp = Computation()
    comp.add_node("a", value=3)
    comp.add_node("b", b)
    comp.add_node("c", c)
    comp.add_node("d", d)

    comp.compute_all()
    assert comp.s.d == States.UNINITIALIZED

    comp.add_node('self', value=1)
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_computation_factory_methods_calc_node_ignore_self():
    @ComputationFactory(ignore_self=False)
    class FooComp:
        a = input_node(value=3)

        @calc_node
        def b(a):
            return a + 1

        @calc_node(ignore_self=True)
        def c(self, a):
            return 2 * a

        @calc_node
        def d(b, c):
            return b + c

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10


def test_computation_factory_methods_calling_methods_on_self():
    @ComputationFactory
    class FooComp:
        a = input_node(value=3)

        def add(self, x, y):
            return x + y

        @calc_node
        def b(self, a):
            return self.add(a, 1)

        @calc_node
        def c(self, a):
            return 2 * a

        @calc_node
        def d(self, b, c):
            return self.add(b, c)

    comp = FooComp()
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10
