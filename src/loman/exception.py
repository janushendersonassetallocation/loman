"""Exception classes for the loman computation engine."""
from contextlib import contextmanager

import networkx as nx


class ComputationError(Exception):
    """Base exception for computation-related errors."""

    pass


class MapError(ComputationError):
    """Exception raised during map operations with partial results."""

    def __init__(self, message, results):
        """Initialize MapError with message and partial results."""
        super().__init__(message)
        self.results = results


class LoopDetectedError(ComputationError):
    """Exception raised when a dependency loop is detected."""

    pass

@contextmanager
def translate_nx_exceptions(g: nx.Graph = None):
    """Transalates nx exceptions to loman API ones.

    :param g: if specified, LoopDetectedError will include DAG loop details
    :raises: LoopDetectedError for nx.NetworkXUnfeasible
    """
    try:
        yield
    except nx.NetworkXUnfeasible as e:
        cycle_lst = None
        if g is not None:
            try:
                cycle_lst = nx.find_cycle(g)
            except nx.NetworkXNoCycle:
                # there must non-cycle reason NetworkXUnfeasible, leave as is
                raise e
        args = []
        if cycle_lst:
            lst = [f"{n_src}->{n_tgt}" for n_src, n_tgt in cycle_lst]
            args = [f"DAG cycle: {', '.join(lst)}"]
        raise LoopDetectedError(*args) from e



class NonExistentNodeError(ComputationError):
    """Exception raised when trying to access a non-existent node."""

    pass


class NodeAlreadyExistsError(ComputationError):
    """Exception raised when trying to create a node that already exists."""

    pass


class CannotInsertToPlaceholderNodeError(ComputationError):
    """Exception raised when trying to insert into a placeholder node."""

    pass


# Backward compatibility aliases
MapException = MapError
LoopDetectedException = LoopDetectedError
NonExistentNodeException = NonExistentNodeError
NodeAlreadyExistsException = NodeAlreadyExistsError
CannotInsertToPlaceholderNodeException = CannotInsertToPlaceholderNodeError
