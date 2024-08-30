from loman import (Computation, States, MapException, LoopDetectedException, NonExistentNodeException, node, C,
                   input_node, calc_node, ComputationFactory)


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

    assert comp.v.d == 10


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
    comp.add_node("a")
    comp.add_node("b", b)
    comp.add_node("c", c)
    comp.add_node("d", d)

    comp.compute_all()
    assert comp.s.d == States.UNINITIALIZED

    comp.add_node('self', value=None)
    comp.compute_all()
    assert comp.s.d == States.UPTODATE and comp.v.d == 10
