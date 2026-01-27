"""Tests for the consts module.

This module tests:
- States enumeration values and behavior
- NodeAttributes constants
- EdgeAttributes constants
- SystemTags constants
- NodeTransformations constants
"""

from loman.consts import (
    EdgeAttributes,
    NodeAttributes,
    NodeTransformations,
    States,
    SystemTags,
)


class TestStates:
    """Test States enumeration."""

    def test_states_values(self):
        """Test that States enum has expected values."""
        assert States.PLACEHOLDER.value == 0
        assert States.UNINITIALIZED.value == 1
        assert States.STALE.value == 2
        assert States.COMPUTABLE.value == 3
        assert States.UPTODATE.value == 4
        assert States.ERROR.value == 5
        assert States.PINNED.value == 6

    def test_states_count(self):
        """Test that States enum has exactly 7 states."""
        assert len(States) == 7

    def test_states_are_unique(self):
        """Test that all States values are unique."""
        values = [s.value for s in States]
        assert len(values) == len(set(values))


class TestNodeAttributes:
    """Test NodeAttributes constants."""

    def test_node_attribute_values(self):
        """Test that NodeAttributes has expected string values."""
        assert NodeAttributes.VALUE == "value"
        assert NodeAttributes.STATE == "state"
        assert NodeAttributes.FUNC == "func"
        assert NodeAttributes.GROUP == "group"
        assert NodeAttributes.TAG == "tag"
        assert NodeAttributes.STYLE == "style"
        assert NodeAttributes.ARGS == "args"
        assert NodeAttributes.KWDS == "kwds"
        assert NodeAttributes.TIMING == "timing"
        assert NodeAttributes.EXECUTOR == "executor"
        assert NodeAttributes.CONVERTER == "converter"


class TestEdgeAttributes:
    """Test EdgeAttributes constants."""

    def test_edge_attribute_values(self):
        """Test that EdgeAttributes has expected string values."""
        assert EdgeAttributes.PARAM == "param"


class TestSystemTags:
    """Test SystemTags constants."""

    def test_system_tag_values(self):
        """Test that SystemTags has expected string values."""
        assert SystemTags.SERIALIZE == "__serialize__"
        assert SystemTags.EXPANSION == "__expansion__"


class TestNodeTransformations:
    """Test NodeTransformations constants."""

    def test_node_transformation_values(self):
        """Test that NodeTransformations has expected string values."""
        assert NodeTransformations.CONTRACT == "contract"
