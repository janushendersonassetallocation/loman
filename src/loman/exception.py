"""Exception classes for the loman computation engine."""


class ComputationError(Exception):
    """Base exception for computation-related errors."""

    pass


class MapException(ComputationError):
    """Exception raised during map operations with partial results."""

    def __init__(self, message, results):
        """Initialize MapException with message and partial results."""
        super().__init__(message)
        self.results = results


class LoopDetectedException(ComputationError):
    """Exception raised when a dependency loop is detected."""

    pass


class NonExistentNodeException(ComputationError):
    """Exception raised when trying to access a non-existent node."""

    pass


class NodeAlreadyExistsException(ComputationError):
    """Exception raised when trying to create a node that already exists."""

    pass


class CannotInsertToPlaceholderNodeException(ComputationError):
    """Exception raised when trying to insert into a placeholder node."""

    pass
