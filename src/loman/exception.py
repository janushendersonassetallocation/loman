"""Exception classes for the loman computation engine."""


class ComputationError(Exception):
    """Base exception for computation-related errors."""


class MapError(ComputationError):
    """Exception raised during map operations with partial results."""

    def __init__(self, message: str, results: list[object]) -> None:
        """Initialize MapError with message and partial results."""
        super().__init__(message)
        self.results = results


class LoopDetectedError(ComputationError):
    """Exception raised when a dependency loop is detected."""


class NonExistentNodeError(ComputationError):
    """Exception raised when trying to access a non-existent node."""


class NodeAlreadyExistsError(ComputationError):
    """Exception raised when trying to create a node that already exists."""


class CannotInsertToPlaceholderNodeError(ComputationError):
    """Exception raised when trying to insert into a placeholder node."""


class InvalidBlockTypeError(TypeError, ComputationError):
    """Exception raised when a block is not callable or a Computation."""


class FittingError(ComputationError):
    """Exception raised when curve fitting exceeds error tolerance."""


class ValidationError(ComputationError):
    """Exception raised during computation validation."""


class SerializationError(ComputationError):
    """Exception raised during serialization/deserialization."""


# Backward compatibility aliases
MapException = MapError
LoopDetectedException = LoopDetectedError
NonExistentNodeException = NonExistentNodeError
NodeAlreadyExistsException = NodeAlreadyExistsError
CannotInsertToPlaceholderNodeException = CannotInsertToPlaceholderNodeError
