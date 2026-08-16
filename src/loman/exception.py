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


class DeserializedError(ComputationError):
    """Stand-in for an exception whose original class could not be rebuilt.

    A saved ERROR node records the name of the exception that produced it.
    Rebuilding an arbitrary one would mean importing whatever module the file
    names, which is executing code chosen by the file, so only builtin exception
    types are reconstructed. Everything else becomes one of these, carrying the
    original identity as data for post-mortem reading.

    :ivar exception_type: Name of the exception class that was originally raised.
    :ivar exception_module: Module that class came from, when it was recorded.
    """

    def __init__(self, message: str, exception_type: str, exception_module: str | None = None) -> None:
        """Record the original exception's identity alongside its message."""
        super().__init__(message)
        self.exception_type = exception_type
        self.exception_module = exception_module

    def __repr__(self) -> str:
        """Show the original exception's type so it is not mistaken for this one."""
        qualified = f"{self.exception_module}.{self.exception_type}" if self.exception_module else self.exception_type
        return f"DeserializedError({qualified}: {super().__str__()!r})"


# Backward compatibility aliases
MapException = MapError
LoopDetectedException = LoopDetectedError
NonExistentNodeException = NonExistentNodeError
NodeAlreadyExistsException = NodeAlreadyExistsError
CannotInsertToPlaceholderNodeException = CannotInsertToPlaceholderNodeError
