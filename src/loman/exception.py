"""Exception classes for the loman computation engine."""


class ComputationError(Exception):
    """Base exception for computation-related errors."""

    pass


class MapError(ComputationError):
    """Exception raised during map operations with partial results."""

    def __init__(self, message: str, results: list[object]) -> None:
        """Initialize MapError with message and partial results."""
        super().__init__(message)
        self.results = results


class LoopDetectedError(ComputationError):
    """Exception raised when a dependency loop is detected."""

    pass


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
