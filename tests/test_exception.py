"""Tests for the exception module.

This module tests:
- ComputationError base exception
- MapError with partial results
- LoopDetectedError for cycle detection
- NonExistentNodeError for missing nodes
- NodeAlreadyExistsError for duplicate nodes
- CannotInsertToPlaceholderNodeError for placeholder restrictions
- Backward compatibility aliases
"""

import pytest

from loman.exception import (
    CannotInsertToPlaceholderNodeError,
    CannotInsertToPlaceholderNodeException,
    ComputationError,
    LoopDetectedError,
    LoopDetectedException,
    MapError,
    MapException,
    NodeAlreadyExistsError,
    NodeAlreadyExistsException,
    NonExistentNodeError,
    NonExistentNodeException,
)


class TestComputationError:
    """Test ComputationError base exception."""

    def test_computation_error_is_exception(self):
        """Test that ComputationError is an Exception."""
        assert issubclass(ComputationError, Exception)

    def test_computation_error_can_be_raised(self):
        """Test that ComputationError can be raised with message."""
        test_msg = "test message"
        with pytest.raises(ComputationError, match="test message"):
            raise ComputationError(test_msg)


class TestMapError:
    """Test MapError exception."""

    def test_map_error_is_computation_error(self):
        """Test that MapError inherits from ComputationError."""
        assert issubclass(MapError, ComputationError)

    def test_map_error_stores_results(self):
        """Test that MapError stores partial results."""
        partial_results = {"a": 1, "b": 2}
        error = MapError("Partial failure", partial_results)

        assert error.results == partial_results
        assert str(error) == "Partial failure"

    def test_map_error_can_be_raised(self):
        """Test that MapError can be raised and caught."""
        partial_results = [1, 2, 3]
        test_msg = "Map failed"
        with pytest.raises(MapError) as exc_info:
            raise MapError(test_msg, partial_results)

        assert exc_info.value.results == partial_results


class TestLoopDetectedError:
    """Test LoopDetectedError exception."""

    def test_loop_detected_error_is_computation_error(self):
        """Test that LoopDetectedError inherits from ComputationError."""
        assert issubclass(LoopDetectedError, ComputationError)

    def test_loop_detected_error_can_be_raised(self):
        """Test that LoopDetectedError can be raised with message."""
        test_msg = "cycle detected"
        with pytest.raises(LoopDetectedError, match="cycle detected"):
            raise LoopDetectedError(test_msg)


class TestNonExistentNodeError:
    """Test NonExistentNodeError exception."""

    def test_non_existent_node_error_is_computation_error(self):
        """Test that NonExistentNodeError inherits from ComputationError."""
        assert issubclass(NonExistentNodeError, ComputationError)

    def test_non_existent_node_error_can_be_raised(self):
        """Test that NonExistentNodeError can be raised with message."""
        test_msg = "node not found"
        with pytest.raises(NonExistentNodeError, match="node not found"):
            raise NonExistentNodeError(test_msg)


class TestNodeAlreadyExistsError:
    """Test NodeAlreadyExistsError exception."""

    def test_node_already_exists_error_is_computation_error(self):
        """Test that NodeAlreadyExistsError inherits from ComputationError."""
        assert issubclass(NodeAlreadyExistsError, ComputationError)

    def test_node_already_exists_error_can_be_raised(self):
        """Test that NodeAlreadyExistsError can be raised with message."""
        test_msg = "node exists"
        with pytest.raises(NodeAlreadyExistsError, match="node exists"):
            raise NodeAlreadyExistsError(test_msg)


class TestCannotInsertToPlaceholderNodeError:
    """Test CannotInsertToPlaceholderNodeError exception."""

    def test_cannot_insert_to_placeholder_is_computation_error(self):
        """Test that CannotInsertToPlaceholderNodeError inherits from ComputationError."""
        assert issubclass(CannotInsertToPlaceholderNodeError, ComputationError)

    def test_cannot_insert_to_placeholder_can_be_raised(self):
        """Test that CannotInsertToPlaceholderNodeError can be raised with message."""
        with pytest.raises(CannotInsertToPlaceholderNodeError, match="placeholder"):
            raise CannotInsertToPlaceholderNodeError("placeholder")


class TestBackwardCompatibilityAliases:
    """Test backward compatibility aliases."""

    def test_map_exception_alias(self):
        """Test that MapException is an alias for MapError."""
        assert MapException is MapError

    def test_loop_detected_exception_alias(self):
        """Test that LoopDetectedException is an alias for LoopDetectedError."""
        assert LoopDetectedException is LoopDetectedError

    def test_non_existent_node_exception_alias(self):
        """Test that NonExistentNodeException is an alias for NonExistentNodeError."""
        assert NonExistentNodeException is NonExistentNodeError

    def test_node_already_exists_exception_alias(self):
        """Test that NodeAlreadyExistsException is an alias for NodeAlreadyExistsError."""
        assert NodeAlreadyExistsException is NodeAlreadyExistsError

    def test_cannot_insert_to_placeholder_exception_alias(self):
        """Test that CannotInsertToPlaceholderNodeException is an alias."""
        assert CannotInsertToPlaceholderNodeException is CannotInsertToPlaceholderNodeError
