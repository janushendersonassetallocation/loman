"""Pytest configuration and fixtures for loman tests.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validating conditions
- S603/S607 (subprocess usage): Any subprocess calls use controlled inputs in test environments
"""

import shutil

import pytest

from loman import Computation, ComputationFactory, calc_node, input_node

# Rendering a graph or a widget goes through pydotplus, which shells out to
# graphviz's `dot`. `make install` provisions that binary via the pre-install hook
# in .rhiza/make.d/00-additional-deps.mk, so it is always there for `make test` and
# for local work -- but CI jobs that call `uv run pytest` directly never trigger
# make, and every render-dependent assertion then fails on the same
# InvocationException. Skip those modules rather than reporting a missing system
# package as ~106 test failures.
requires_dot = pytest.mark.skipif(
    shutil.which("dot") is None,
    reason="graphviz 'dot' is not on PATH; install it with scripts/install-graphviz.sh",
)


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
