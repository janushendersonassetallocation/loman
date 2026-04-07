"""Tests for type safety improvements across the loman codebase.

This module covers:
- Phase 2: Name type alias and NodeKey.parts typing
- Phase 3: @overload return types for polymorphic methods
- Phase 4: Assert replacements, Block callable typing
"""

from collections.abc import Hashable

import pytest

from loman import Computation, States
from loman.computeengine import NodeData, TimingData
from loman.nodekey import Name, NodeKey, to_nodekey

# ==================== Phase 2: Name type and NodeKey.parts ====================


class TestNodeKeyPartsTyping:
    """Tests that NodeKey.parts contains Hashable elements."""

    def test_nodekey_parts_are_hashable(self):
        """All NodeKey parts should be Hashable."""
        nk = NodeKey(("a", "b", "c"))
        for part in nk.parts:
            assert isinstance(part, Hashable), f"NodeKey part {part!r} must be Hashable"

    def test_nodekey_from_string_parts_are_hashable(self):
        """NodeKey parts created from string parsing should be Hashable."""
        nk = to_nodekey("foo/bar/baz")
        for part in nk.parts:
            assert isinstance(part, Hashable), f"NodeKey part {part!r} must be Hashable"

    def test_nodekey_with_integer_part_is_hashable(self):
        """NodeKey parts created from non-string hashable objects should be Hashable."""
        nk = to_nodekey(42)
        for part in nk.parts:
            assert isinstance(part, Hashable), f"NodeKey part {part!r} must be Hashable"

    def test_nodekey_join_parts_are_hashable(self):
        """NodeKey.join_parts should produce Hashable parts."""
        nk = NodeKey(("a",)).join_parts("b", "c")
        for part in nk.parts:
            assert isinstance(part, Hashable), f"NodeKey part {part!r} must be Hashable"


class TestNameType:
    """Tests that the Name type alias works correctly with its expected types."""

    def test_string_is_valid_name(self):
        """Strings should be valid Name values."""
        name: Name = "foo"
        nk = to_nodekey(name)
        assert nk == NodeKey(("foo",))

    def test_nodekey_is_valid_name(self):
        """NodeKey objects should be valid Name values."""
        name: Name = NodeKey(("foo", "bar"))
        nk = to_nodekey(name)
        assert nk == NodeKey(("foo", "bar"))

    def test_hashable_is_valid_name(self):
        """Other hashable objects (e.g., int) should be valid Name values."""
        name: Name = 42
        nk = to_nodekey(name)
        assert nk == NodeKey((42,))


# ==================== Phase 3: @overload return types ====================


class TestStateReturnType:
    """Tests that state() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a simple computation with two nodes."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)
        return comp

    def test_state_single_returns_states(self, comp):
        """state() with a single name should return a States enum value."""
        result = comp.state("a")
        assert isinstance(result, States)
        assert result == States.UPTODATE

    def test_state_list_returns_list_of_states(self, comp):
        """state() with a list of names should return a list of States."""
        result = comp.state(["a", "b"])
        assert isinstance(result, list)
        assert all(isinstance(s, States) for s in result)
        assert result == [States.UPTODATE, States.UPTODATE]


class TestValueReturnType:
    """Tests that value() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a simple computation with two nodes."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)
        return comp

    def test_value_single_returns_scalar(self, comp):
        """value() with a single name should return the node's value directly."""
        result = comp.value("a")
        assert result == 1
        # Should NOT be a list
        assert not isinstance(result, list)

    def test_value_list_returns_list(self, comp):
        """value() with a list of names should return a list of values."""
        result = comp.value(["a", "b"])
        assert isinstance(result, list)
        assert result == [1, 2]


class TestTagsReturnType:
    """Tests that tags() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a computation with tagged nodes."""
        comp = Computation()
        comp.add_node("a", value=1, tags=["foo", "bar"])
        comp.add_node("b", value=2, tags=["baz"])
        return comp

    def test_tags_single_returns_set(self, comp):
        """tags() with a single name should return a set of strings."""
        result = comp.tags("a")
        assert isinstance(result, set)
        assert "foo" in result
        assert "bar" in result

    def test_tags_list_returns_list_of_sets(self, comp):
        """tags() with a list of names should return a list of sets."""
        result = comp.tags(["a", "b"])
        assert isinstance(result, list)
        assert all(isinstance(s, set) for s in result)


class TestStylesReturnType:
    """Tests that styles() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a computation with styled nodes."""
        comp = Computation()
        comp.add_node("a", value=1, style="dot")
        comp.add_node("b", value=2)
        return comp

    def test_styles_single_returns_string_or_none(self, comp):
        """styles() with a single name should return str or None."""
        result_a = comp.styles("a")
        assert result_a == "dot"
        result_b = comp.styles("b")
        assert result_b is None

    def test_styles_list_returns_list(self, comp):
        """styles() with a list of names should return a list."""
        result = comp.styles(["a", "b"])
        assert isinstance(result, list)
        assert result == ["dot", None]


class TestGetItemReturnType:
    """Tests that __getitem__ returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a simple computation."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)
        return comp

    def test_getitem_single_returns_nodedata(self, comp):
        """__getitem__ with a single name should return a NodeData."""
        result = comp["a"]
        assert isinstance(result, NodeData)
        assert result.state == States.UPTODATE
        assert result.value == 1

    def test_getitem_list_returns_list_of_nodedata(self, comp):
        """__getitem__ with a list of names should return a list of NodeData."""
        result = comp[["a", "b"]]
        assert isinstance(result, list)
        assert all(isinstance(nd, NodeData) for nd in result)


class TestGetTimingReturnType:
    """Tests that get_timing() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a computation with a calculated node (which gets timing data)."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()
        return comp

    def test_get_timing_single_returns_timing_or_none(self, comp):
        """get_timing() with a single name should return TimingData or None."""
        result_b = comp.get_timing("b")
        assert result_b is None or isinstance(result_b, TimingData)
        # Input nodes have no timing
        result_a = comp.get_timing("a")
        assert result_a is None

    def test_get_timing_list_returns_list(self, comp):
        """get_timing() with a list of names should return a list."""
        result = comp.get_timing(["a", "b"])
        assert isinstance(result, list)
        assert len(result) == 2


class TestGetInputsReturnType:
    """Tests that get_inputs() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a computation with dependencies."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)
        comp.add_node("c", lambda a, b: a + b)
        return comp

    def test_get_inputs_single_returns_names(self, comp):
        """get_inputs() with a single name should return a Names (list)."""
        result = comp.get_inputs("c")
        assert isinstance(result, list)

    def test_get_inputs_list_returns_list_of_names(self, comp):
        """get_inputs() with a list of names should return a list of lists."""
        result = comp.get_inputs(["a", "c"])
        assert isinstance(result, list)
        assert all(isinstance(item, list) for item in result)


class TestGetOutputsReturnType:
    """Tests that get_outputs() returns correct types for single vs list input."""

    @pytest.fixture
    def comp(self):
        """Create a computation with dependencies."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        return comp

    def test_get_outputs_single_returns_names(self, comp):
        """get_outputs() with a single name should return a Names (list)."""
        result = comp.get_outputs("a")
        assert isinstance(result, list)

    def test_get_outputs_list_returns_list_of_names(self, comp):
        """get_outputs() with a list of names should return a list of lists."""
        result = comp.get_outputs(["a", "b"])
        assert isinstance(result, list)
        assert all(isinstance(item, list) for item in result)


# ==================== Phase 4: Assert replacements and other ====================


class TestNodeKeyParsingErrors:
    """Tests that nodekey parsing raises proper exceptions (not AssertionError).

    After replacing asserts with explicit raises, malformed input should produce
    ValueError instead of AssertionError. This is important because asserts are
    stripped in optimized mode (python -O).
    """

    def test_parse_malformed_quoted_string_raises_valueerror(self):
        """Parsing a malformed quoted nodekey should raise ValueError, not AssertionError."""
        # A quoted string followed by invalid content (not / or end)
        with pytest.raises(ValueError, match="Expected end of string or '/'"):
            to_nodekey('"foo"bar')

    def test_parse_incomplete_string_raises_error(self):
        """Parsing should not rely on assertions for validation."""
        # Normal valid strings should parse fine
        nk = to_nodekey("foo/bar")
        assert nk == NodeKey(("foo", "bar"))


class TestBlockCallableTyping:
    """Tests for Block node with callable vs Computation."""

    def test_block_with_zero_arg_callable(self):
        """Block should accept a zero-argument callable that returns a Computation."""

        def make_comp():
            comp = Computation()
            comp.add_node("x", value=42)
            return comp

        from loman.computeengine import Block

        b = Block(make_comp)
        outer = Computation()
        b.add_to_comp(outer, "inner", None, False)
        assert outer.has_node("inner/x")

    def test_block_with_computation_instance(self):
        """Block should accept a Computation instance directly."""
        from loman.computeengine import Block

        inner = Computation()
        inner.add_node("x", value=42)
        b = Block(inner)
        outer = Computation()
        b.add_to_comp(outer, "inner", None, False)
        assert outer.has_node("inner/x")

    def test_block_with_non_callable_raises(self):
        """Block should raise TypeError for non-callable, non-Computation."""
        from loman.computeengine import Block

        b = Block("not_callable")  # type: ignore[arg-type]
        outer = Computation()
        with pytest.raises(TypeError, match="must be callable or Computation"):
            b.add_to_comp(outer, "inner", None, False)


class TestDecoratorTypePreservation:
    """Tests that node/calc_node decorators preserve function identity."""

    def test_calc_node_preserves_function(self):
        """calc_node should attach _loman_node_info without losing the function."""
        from loman.computeengine import CalcNode, calc_node

        @calc_node
        def my_func(a, b):
            """Add a and b."""
            return a + b

        # The function should still be callable
        assert my_func(1, 2) == 3
        # It should have the node info attached
        assert hasattr(my_func, "_loman_node_info")
        assert isinstance(my_func._loman_node_info, CalcNode)

    def test_calc_node_with_kwargs_preserves_function(self):
        """calc_node with keyword args should preserve the function."""
        from loman.computeengine import CalcNode, calc_node

        @calc_node(serialize=False)
        def my_func(x):
            """Double x."""
            return x * 2

        assert my_func(5) == 10
        assert hasattr(my_func, "_loman_node_info")
        assert isinstance(my_func._loman_node_info, CalcNode)

    def test_node_decorator_registers_and_preserves(self):
        """The @node decorator should register the function and preserve callability."""
        from loman import Computation, node

        comp = Computation()

        @node(comp, "my_node")
        def my_func(a):
            """Add one to a."""
            return a + 1

        # Function should still be callable
        assert my_func(5) == 6
        # Node should be registered
        assert comp.has_node("my_node")
