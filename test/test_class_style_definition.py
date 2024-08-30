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
