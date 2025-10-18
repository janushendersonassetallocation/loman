import pytest

from loman import *
from loman.computeengine import Block


def test_simple_block():
    comp_inner = Computation()
    comp_inner.add_node("a")
    comp_inner.add_node("b", lambda a: a + 1)
    comp_inner.add_node("c", lambda a: 2 * a)
    comp_inner.add_node("d", lambda b, c: b + c)

    comp = Computation()
    comp.add_block("foo", comp_inner)
    comp.add_block("bar", comp_inner)
    comp.add_node("input_foo")
    comp.add_node("input_bar")
    comp.link("foo/a", "input_foo")
    comp.link("bar/a", "input_bar")
    comp.add_node("output", lambda x, y: x + y, kwds={"x": "foo/d", "y": "bar/d"})

    comp.insert("input_foo", value=7)
    comp.insert("input_bar", value=10)

    comp.compute_all()

    assert comp.v["foo/d"] == 22
    assert comp.v["bar/d"] == 31
    assert comp.v.output == 22 + 31


def test_add_node_for_block_definition():
    comp = Computation()
    comp.add_node("foo/a")
    comp.add_node("foo/b", lambda a: a + 1)
    comp.add_node("foo/c", lambda a: 2 * a)
    comp.add_node("foo/d", lambda b, c: b + c)
    comp.insert("foo/a", value=7)
    comp.compute_all()
    assert comp.v["foo/d"] == 22


def test_add_node_for_block_definition_with_kwds():
    comp = Computation()
    comp.add_node("foo_a/a")
    comp.add_node("foo_b/b", lambda a: a + 1, kwds={"a": "foo_a/a"})
    comp.add_node("foo_c/c", lambda a: 2 * a, kwds={"a": "foo_a/a"})
    comp.add_node("foo_d/d", lambda b, c: b + c, kwds={"b": "foo_b/b", "c": "foo_c/c"})
    comp.insert("foo_a/a", value=7)
    comp.compute_all()
    assert comp.v["foo_d/d"] == 22


def test_add_block_with_links():
    comp_inner = Computation()
    comp_inner.add_node("a")
    comp_inner.add_node("b", lambda a: a + 1)
    comp_inner.add_node("c", lambda a: 2 * a)
    comp_inner.add_node("d", lambda b, c: b + c)

    comp = Computation()
    comp.add_block("foo", comp_inner, links={"a": "input_foo"})
    comp.add_block("bar", comp_inner, links={"a": "input_bar"})
    comp.add_node("output", lambda x, y: x + y, kwds={"x": "foo/d", "y": "bar/d"})

    comp.add_node("input_foo", value=7)
    comp.add_node("input_bar", value=10)

    comp.compute_all()

    assert comp.v["foo/d"] == 22
    assert comp.v["bar/d"] == 31
    assert comp.v.output == 22 + 31


def test_add_block_with_keep_values_false():
    comp_inner = Computation()
    comp_inner.add_node("a", value=7)
    comp_inner.add_node("b", lambda a: a + 1)
    comp_inner.add_node("c", lambda a: 2 * a)
    comp_inner.add_node("d", lambda b, c: b + c)
    comp_inner.compute_all()

    comp = Computation()
    comp.add_block("foo", comp_inner, keep_values=False, links={"a": "input_foo"})
    comp.add_block("bar", comp_inner, keep_values=False, links={"a": "input_bar"})
    comp.add_node("output", lambda x, y: x + y, kwds={"x": "foo/d", "y": "bar/d"})

    comp.add_node("input_foo", value=7)
    comp.add_node("input_bar", value=10)

    comp.compute_all()

    assert comp.v["foo/d"] == 22
    assert comp.v["bar/d"] == 31
    assert comp.v.output == 22 + 31


def test_add_block_with_keep_values_true():
    comp_inner = Computation()
    comp_inner.add_node("a", value=7)
    comp_inner.add_node("b", lambda a: a + 1)
    comp_inner.add_node("c", lambda a: 2 * a)
    comp_inner.add_node("d", lambda b, c: b + c)
    comp_inner.compute_all()

    comp = Computation()
    comp.add_block("foo", comp_inner, keep_values=True)
    comp.add_block("bar", comp_inner, keep_values=True, links={"a": "input_bar"})
    comp.add_node("output", lambda x, y: x + y, kwds={"x": "foo/d", "y": "bar/d"})

    comp.add_node("input_bar", value=10)

    assert comp.v["foo/d"] == 22
    assert comp.v["bar/d"] == 22

    comp.compute_all()

    assert comp.v["foo/d"] == 22
    assert comp.v["bar/d"] == 31
    assert comp.v.output == 22 + 31


def test_block_accessors():
    comp = Computation()
    comp.add_node("foo1/bar1/baz1/a", value=1)

    assert comp.v.foo1.bar1.baz1.a == 1
    assert comp.v["foo1/bar1/baz1/a"] == 1
    assert comp.v["foo1"].bar1.baz1.a == 1
    assert comp.v.foo1["bar1"].baz1.a == 1

    with pytest.raises(AttributeError):
        comp.v.foo1.bar1.baz1.nonexistent

    comp.add_node("foo1/bar1/baz1", value=2)

    assert comp.v.foo1.bar1.baz1 == 2
    with pytest.raises(AttributeError):
        comp.v.foo1.bar1.nonexistent


def test_computation_factory_with_blocks():
    @ComputationFactory
    class InnerComputation:
        a = input_node()

        @calc_node
        def b(self, a):
            return a + 1

        @calc_node
        def c(self, a):
            return 2 * a

        @calc_node
        def d(self, b, c):
            return b + c

    @ComputationFactory
    class OuterComputation:
        input_foo = input_node()
        input_bar = input_node()

        foo = block(InnerComputation, links={"a": "input_foo"})
        bar = block(InnerComputation, links={"a": "input_bar"})

        @calc_node(kwds={"a": "foo/d", "b": "bar/d"})
        def output(self, a, b):
            return a + b

    comp = OuterComputation()

    comp.insert("input_foo", value=7)
    comp.insert("input_bar", value=10)

    comp.compute_all()

    assert comp.v.foo.d == 22
    assert comp.v.bar.d == 31
    assert comp.v.output == 22 + 31


def test_block_add_to_comp():
    inner_comp = Computation()
    inner_comp.add_node("a", value=10)
    inner_comp.add_node("b", lambda a: a * 2)
    outer_comp = Computation()
    Block(inner_comp).add_to_comp(outer_comp, 'blk', None, True)
    outer_comp.compute('blk/b')
    assert outer_comp.nodes() == ['blk/a', 'blk/b']
    assert outer_comp.v['blk/b'] == 20
