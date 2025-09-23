from loman import (
    Computation,
    ComputationFactory,
    States,
    calc_node,
    input_node,
)


def test_class_style_definition():
    class FooComp:
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

    comp = Computation.from_class(FooComp)
    comp.compute_all()

    assert comp.v.d == 10


def test_class_style_definition_as_decorator():
    @Computation.from_class
    class FooComp:
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
    class FooComp:
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
    class FooComp:
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

    comp.add_node("self", value=None)
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

    comp.add_node("self", value=1)
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
    comp.add_node("self", value=None)  # Provide self node as required when ignore_self=False
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


def test_computation_factory_methods_calling_methods_on_self_recursively():
    @ComputationFactory
    class FooComp:
        a = input_node(value=3)

        def really_add(self, x, y):
            return x + y

        def add(self, x, y):
            return self.really_add(x, y)

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


def test_computation_factory_calc_node_no_args():
    @ComputationFactory
    class FooComp:
        @calc_node
        def a():
            return 3

    comp = FooComp()
    comp.compute_all()
    assert comp.s.a == States.UPTODATE and comp.v.a == 3
