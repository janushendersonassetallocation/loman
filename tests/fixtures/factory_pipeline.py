"""Computation classes used by the ``@ComputationFactory`` serialization tests.

These live in an importable module for the same reason
:mod:`tests.fixtures.pipeline` does: a calc node declared on a factory class is
stored by the class's importable path, so a class defined inside a test function
--- whose qualified name contains ``<locals>`` --- cannot be resolved and would
make the tests pass for the wrong reason.
"""

from loman import ComputationFactory, calc_node, input_node


@ComputationFactory
class Portfolio:
    """An ordinary factory-declared computation: the case that must round-trip."""

    prices = input_node()
    multiplier = input_node()

    @calc_node
    def signal(self, prices, multiplier):
        """Scale the prices."""
        return prices * multiplier

    @calc_node
    def total(self, signal):
        """Sum the signal."""
        return float(signal.sum().sum())


@ComputationFactory
class StatefulPortfolio:
    """A factory class whose methods read state set in ``__init__``.

    The restored graph binds to a *fresh* definition object, so state computed in
    ``__init__`` is rebuilt rather than carried across. That is fine here ---
    ``scale`` is a constant --- and the tests use this class to pin exactly what
    is and is not preserved.
    """

    def __init__(self):
        """Set the state the calc node reads."""
        self.scale = 10.0

    prices = input_node()

    @calc_node
    def scaled(self, prices):
        """Scale the prices by the instance's own factor."""
        return prices * self.scale


class RequiresArguments:
    """A definition class that cannot be constructed without arguments.

    Not decorated: it is used directly to show the fallback, where the method
    cannot be rebuilt and the node is stored without a function.
    """

    def __init__(self, factor):
        """Take a factor that has no default."""
        self.factor = factor

    def scaled(self, prices):
        """Scale the prices by the instance's factor."""
        return prices * self.factor
