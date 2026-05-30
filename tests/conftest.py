"""Pytest configuration and fixtures for loman tests.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validating conditions
- S603/S607 (subprocess usage): Any subprocess calls use controlled inputs in test environments
"""

from loman import Computation, ComputationFactory, calc_node, input_node


@ComputationFactory
class BasicFourNodeComputation:
    """Basic computation with four nodes for testing."""

    a = input_node()

    @calc_node
    def b(a):  # noqa: N805
        """Calculate b = a + 1."""
        return a + 1

    @calc_node
    def c(a):  # noqa: N805
        """Calculate c = 2 * a."""
        return 2 * a

    @calc_node
    def d(b, c):  # noqa: N805
        """Calculate d = b + c."""
        return b + c


def create_example_block_computation():
    """Create an example computation with nested blocks for testing."""
    comp_inner = BasicFourNodeComputation()
    comp_inner.insert("a", value=7)
    comp_inner.compute_all()
    comp = Computation()
    comp.add_block("foo", comp_inner, keep_values=False, links={"a": "input_foo"})
    comp.add_block("bar", comp_inner, keep_values=False, links={"a": "input_bar"})
    comp.add_node("output", lambda x, y: x + y, kwds={"x": "foo/d", "y": "bar/d"})
    comp.add_node("input_foo", value=7)
    comp.add_node("input_bar", value=10)
    return comp
