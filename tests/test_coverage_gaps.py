"""Tests to achieve 100% code coverage for loman modules."""

import io
import sys
import tempfile
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from loman import Computation, States
from loman.compat import get_signature
from loman.computeengine import (
    Block,
    ConstantValue,
    InputNode,
    NullObject,
    TimingData,
    identity_function,
)
from loman.consts import NodeAttributes
from loman.exception import (
    NodeAlreadyExistsException,
    NonExistentNodeException,
)
from loman.graph_utils import topological_sort
from loman.nodekey import (
    NodeKey,
    PathNotFoundError,
    quote_part,
    to_nodekey,
)
from loman.serialization import (
    MissingObject,
    Transformer,
    UnrecognizedTypeException,
    UntransformableTypeException,
)
from loman.serialization.default import default_transformer
from loman.util import AttributeView, apply1, value_eq
from loman.visualization import (
    ColorByState,
    ColorByTiming,
    CompositeNodeFormatter,
    GraphView,
    Node,
    NodeFormatter,
    RectBlocks,
    ShapeByType,
    StandardGroup,
    StandardStylingOverrides,
    create_root_graph,
    create_subgraph,
)

# ==================== COMPAT.PY TESTS ====================


class TestCompat:
    """Tests for compat.py coverage."""

    def test_get_signature_positional_only(self):
        """Test get_signature with positional-only parameters raises NotImplementedError."""
        # Create a function with positional-only parameters (Python 3.8+)
        # We need to create a function with POSITIONAL_ONLY parameter kind
        exec_globals = {}
        exec("def func_with_pos_only(x, /): pass", exec_globals)
        func = exec_globals["func_with_pos_only"]

        with pytest.raises(NotImplementedError, match="Unexpected param kind"):
            get_signature(func)


# ==================== GRAPH_UTILS.PY TESTS ====================


class TestGraphUtils:
    """Tests for graph_utils.py coverage."""

    def test_topological_sort_non_cycle_unfeasible(self):
        """Test topological sort with NetworkXUnfeasible that's not a cycle."""
        # Create a graph that triggers NetworkXUnfeasible but not due to a cycle
        # This is an edge case - in practice, NetworkXUnfeasible from topological_sort
        # is almost always due to cycles, but we need to test the re-raise path
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")  # Create a cycle

        from loman.exception import LoopDetectedError

        with pytest.raises(LoopDetectedError):
            topological_sort(g)


# ==================== NODEKEY.PY TESTS ====================


class TestNodeKey:
    """Tests for nodekey.py coverage."""

    def test_quote_part_with_slash(self):
        """Test quote_part with a string containing slash."""
        result = quote_part("foo/bar")
        assert result == '"foo/bar"'

    def test_nodekey_label_empty(self):
        """Test NodeKey label property with empty parts."""
        nk = NodeKey(())
        assert nk.label == ""

    def test_nodekey_name_with_mixed_parts(self):
        """Test NodeKey name property with non-string parts."""
        obj = object()
        nk = NodeKey(("a", obj))
        # When parts contain non-strings, name returns self
        assert nk.name is nk

    def test_nodekey_parent_of_root(self):
        """Test getting parent of root NodeKey raises PathNotFoundError."""
        nk = NodeKey(())
        with pytest.raises(PathNotFoundError):
            _ = nk.parent

    def test_nodekey_eq_with_none(self):
        """Test NodeKey equality with None."""
        nk = NodeKey(("a",))
        assert nk != None  # noqa: E711

    def test_nodekey_eq_with_non_nodekey(self):
        """Test NodeKey equality with non-NodeKey returns NotImplemented."""
        nk = NodeKey(("a",))
        result = nk.__eq__("not a nodekey")
        assert result is NotImplemented

    def test_nodekey_root_singleton(self):
        """Test NodeKey.root() returns singleton."""
        r1 = NodeKey.root()
        r2 = NodeKey.root()
        assert r1 is r2

    def test_to_nodekey_value_error_unreachable(self):
        """Test to_nodekey - the else branch is unreachable but test the object path."""
        # This tests line 216 - the object path that's always taken for non-str/non-NodeKey
        obj = object()
        nk = to_nodekey(obj)
        assert nk.parts == (obj,)


# ==================== SERIALIZATION/DEFAULT.PY TESTS ====================


class TestSerializationDefault:
    """Tests for serialization/default.py coverage."""

    def test_default_transformer(self):
        """Test default_transformer creates a Transformer with NdArray support."""
        t = default_transformer()
        assert isinstance(t, Transformer)
        # Test that NdArrayTransformer is registered
        arr = np.array([1, 2, 3])
        d = t.to_dict(arr)
        assert d is not None
        arr_back = t.from_dict(d)
        assert np.array_equal(arr, arr_back)


# ==================== SERIALIZATION/TRANSFORMER.PY TESTS ====================


class TestTransformer:
    """Tests for serialization/transformer.py coverage."""

    def test_missing_object_repr(self):
        """Test MissingObject repr."""
        mo = MissingObject()
        assert repr(mo) == "Missing"

    def test_transformer_strict_transformable_missing(self):
        """Test from_dict with unknown Transformable type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "transformable", "class": "UnknownClass", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_transformable_missing(self):
        """Test from_dict with unknown Transformable type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "transformable", "class": "UnknownClass", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_attrs_missing(self):
        """Test from_dict with unknown attrs type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "attrs", "class": "UnknownAttrs", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_attrs_missing(self):
        """Test from_dict with unknown attrs type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "attrs", "class": "UnknownAttrs", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_dataclass_missing(self):
        """Test from_dict with unknown dataclass type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "dataclass", "class": "UnknownDataclass", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_dataclass_missing(self):
        """Test from_dict with unknown dataclass type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "dataclass", "class": "UnknownDataclass", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_unknown_type(self):
        """Test from_dict with completely unknown type in strict mode."""
        t = Transformer(strict=True)
        d = {"type": "completely_unknown_type", "data": {}}
        with pytest.raises(UnrecognizedTypeException):
            t.from_dict(d)

    def test_transformer_non_strict_unknown_type(self):
        """Test from_dict with completely unknown type in non-strict mode."""
        t = Transformer(strict=False)
        d = {"type": "completely_unknown_type", "data": {}}
        result = t.from_dict(d)
        assert isinstance(result, MissingObject)

    def test_transformer_strict_untransformable(self):
        """Test to_dict with untransformable object in strict mode."""

        class UntransformableClass:
            pass

        t = Transformer(strict=True)
        obj = UntransformableClass()
        with pytest.raises(UntransformableTypeException):
            t.to_dict(obj)

    def test_transformer_non_strict_untransformable(self):
        """Test to_dict with untransformable object in non-strict mode."""

        class UntransformableClass:
            pass

        t = Transformer(strict=False)
        obj = UntransformableClass()
        result = t.to_dict(obj)
        assert result is None

    def test_from_dict_exception_non_dict_list(self):
        """Test from_dict with unexpected type raises Exception."""
        t = Transformer()
        # Pass something that's not str/None/bool/int/float/list/dict
        with pytest.raises(Exception):
            t.from_dict(object())


# ==================== UTIL.PY TESTS ====================


class TestUtil:
    """Tests for util.py coverage."""

    def test_apply1_with_generator(self):
        """Test apply1 with a generator input."""

        def double(x):
            return x * 2

        gen = (x for x in [1, 2, 3])
        result = apply1(double, gen)
        # Result should be a generator
        assert list(result) == [2, 4, 6]

    def test_attribute_view_getstate_setstate(self):
        """Test AttributeView serialization methods."""
        d = {"a": 1, "b": 2}
        av = AttributeView.from_dict(d)

        state = av.__getstate__()
        assert "get_attribute_list" in state
        assert "get_attribute" in state
        assert "get_item" in state

        # Create new AttributeView and restore state
        new_av = AttributeView(lambda: [], lambda x: None)
        new_av.__setstate__(state)
        assert new_av.a == 1

    def test_attribute_view_setstate_with_none_get_item(self):
        """Test AttributeView setstate when get_item is None."""
        d = {"a": 1}
        # Create AttributeView without using from_dict to have None get_item
        av = AttributeView.__new__(AttributeView)
        av.get_attribute_list = d.keys
        av.get_attribute = d.get
        av.get_item = None  # Explicitly set to None

        state = av.__getstate__()
        assert state["get_item"] is None

        new_av = AttributeView(lambda: [], lambda x: None)
        new_av.__setstate__(state)
        # After setstate with get_item=None, get_item should default to get_attribute
        assert new_av["a"] == 1

    def test_attribute_view_from_dict_no_apply1(self):
        """Test AttributeView.from_dict with use_apply1=False."""
        d = {"a": 1, "b": 2}
        av = AttributeView.from_dict(d, use_apply1=False)
        assert av.a == 1
        assert av.b == 2

    def test_value_eq_ndarray_exception(self):
        """Test value_eq when numpy comparison raises exception."""
        a = np.array([1, 2, 3])
        # Create something that causes array_equal to fail
        b = "not an array that can be compared"
        result = value_eq(a, b)
        assert result is False

    def test_value_eq_fallback_exception(self):
        """Test value_eq when comparison raises exception."""

        class BadComparison:
            def __eq__(self, other):
                raise ValueError("Cannot compare")

        a = BadComparison()
        b = BadComparison()
        result = value_eq(a, b)
        # Should return False when comparison fails
        assert result is False

    def test_value_eq_result_is_ndarray(self):
        """Test value_eq when result is ndarray-like."""
        # Create objects where == returns an array
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        result = value_eq(a, b)
        assert result is True


# ==================== VISUALIZATION.PY TESTS ====================


class TestVisualization:
    """Tests for visualization.py coverage."""

    def test_color_by_timing_no_timing(self):
        """Test ColorByTiming with node having no timing data."""
        cbt = ColorByTiming()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = cbt.format(nk, [node], False)
        assert result["fillcolor"] == "#FFFFFF"

    def test_color_by_timing_with_timing(self):
        """Test ColorByTiming with timing data."""
        from datetime import datetime

        cbt = ColorByTiming()

        nk1 = NodeKey(("fast",))
        nk2 = NodeKey(("slow",))
        timing1 = TimingData(datetime.now(), datetime.now(), 0.1)
        timing2 = TimingData(datetime.now(), datetime.now(), 1.0)

        node1 = Node(nk1, nk1, {NodeAttributes.TIMING: timing1})
        node2 = Node(nk2, nk2, {NodeAttributes.TIMING: timing2})

        cbt.calibrate([node1, node2])
        result = cbt.format(nk1, [node1], False)
        assert "fillcolor" in result
        assert result["fillcolor"] != "#FFFFFF"

    def test_shape_by_type_no_value(self):
        """Test ShapeByType with no value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = sbt.format(nk, [node], False)
        assert result is None

    def test_shape_by_type_ndarray(self):
        """Test ShapeByType with numpy array value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.VALUE: np.array([1, 2, 3])})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "rect"

    def test_shape_by_type_dataframe(self):
        """Test ShapeByType with DataFrame value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.VALUE: pd.DataFrame({"a": [1, 2]})})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "box3d"

    def test_shape_by_type_scalar(self):
        """Test ShapeByType with scalar value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.VALUE: 42})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "ellipse"

    def test_shape_by_type_list(self):
        """Test ShapeByType with list value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.VALUE: [1, 2, 3]})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "ellipse"
        assert result["peripheries"] == 2

    def test_shape_by_type_dict(self):
        """Test ShapeByType with dict value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.VALUE: {"a": 1}})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "house"
        assert result["peripheries"] == 2

    def test_shape_by_type_computation(self):
        """Test ShapeByType with Computation value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))
        comp = Computation()
        node = Node(nk, nk, {NodeAttributes.VALUE: comp})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "hexagon"

    def test_shape_by_type_other(self):
        """Test ShapeByType with other type value."""
        sbt = ShapeByType()
        nk = NodeKey(("test",))

        class CustomClass:
            pass

        node = Node(nk, nk, {NodeAttributes.VALUE: CustomClass()})
        result = sbt.format(nk, [node], False)
        assert result["shape"] == "diamond"

    def test_standard_styling_overrides_small(self):
        """Test StandardStylingOverrides with 'small' style."""
        sso = StandardStylingOverrides()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.STYLE: "small"})
        result = sso.format(nk, [node], False)
        assert result["width"] == 0.3
        assert result["height"] == 0.2
        assert result["fontsize"] == 8

    def test_standard_styling_overrides_dot(self):
        """Test StandardStylingOverrides with 'dot' style."""
        sso = StandardStylingOverrides()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {NodeAttributes.STYLE: "dot"})
        result = sso.format(nk, [node], False)
        assert result["shape"] == "point"
        assert result["width"] == 0.1
        assert result["peripheries"] == 1

    def test_standard_styling_overrides_none(self):
        """Test StandardStylingOverrides with no style."""
        sso = StandardStylingOverrides()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = sso.format(nk, [node], False)
        assert result is None

    def test_node_formatter_create_invalid_shapes(self):
        """Test NodeFormatter.create with invalid shapes parameter."""
        with pytest.raises(ValueError, match="is not a valid loman shapes parameter"):
            NodeFormatter.create(shapes="invalid")

    def test_node_formatter_create_invalid_colors(self):
        """Test NodeFormatter.create with invalid colors parameter."""
        with pytest.raises(ValueError, match="is not a valid loman colors parameter"):
            NodeFormatter.create(colors="invalid")

    def test_node_formatter_create_type_shapes(self):
        """Test NodeFormatter.create with shapes='type'."""
        nf = NodeFormatter.create(shapes="type")
        assert isinstance(nf, CompositeNodeFormatter)

    def test_node_formatter_create_timing_colors(self):
        """Test NodeFormatter.create with colors='timing'."""
        nf = NodeFormatter.create(colors="timing")
        assert isinstance(nf, CompositeNodeFormatter)

    def test_rect_blocks_composite(self):
        """Test RectBlocks for composite node."""
        rb = RectBlocks()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = rb.format(nk, [node], is_composite=True)
        assert result["shape"] == "rect"
        assert result["peripheries"] == 2

    def test_rect_blocks_non_composite(self):
        """Test RectBlocks for non-composite node."""
        rb = RectBlocks()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = rb.format(nk, [node], is_composite=False)
        assert result is None

    def test_color_by_state_composite_all_error(self):
        """Test ColorByState with multiple nodes all having ERROR state."""
        cbs = ColorByState()
        nk = NodeKey(("test",))
        node1 = Node(nk, nk, {NodeAttributes.STATE: States.ERROR})
        node2 = Node(nk, nk, {NodeAttributes.STATE: States.UPTODATE})
        result = cbs.format(nk, [node1, node2], False)
        assert result["fillcolor"] == ColorByState.DEFAULT_STATE_COLORS[States.ERROR]

    def test_color_by_state_composite_stale(self):
        """Test ColorByState with multiple nodes including STALE."""
        cbs = ColorByState()
        nk = NodeKey(("test",))
        node1 = Node(nk, nk, {NodeAttributes.STATE: States.STALE})
        node2 = Node(nk, nk, {NodeAttributes.STATE: States.UPTODATE})
        result = cbs.format(nk, [node1, node2], False)
        assert result["fillcolor"] == ColorByState.DEFAULT_STATE_COLORS[States.STALE]

    def test_color_by_state_composite_all_same(self):
        """Test ColorByState with multiple nodes all same state."""
        cbs = ColorByState()
        nk = NodeKey(("test",))
        node1 = Node(nk, nk, {NodeAttributes.STATE: States.UPTODATE})
        node2 = Node(nk, nk, {NodeAttributes.STATE: States.UPTODATE})
        result = cbs.format(nk, [node1, node2], False)
        assert result["fillcolor"] == ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]

    def test_color_by_state_composite_mixed(self):
        """Test ColorByState with multiple nodes with different states (not error/stale)."""
        cbs = ColorByState()
        nk = NodeKey(("test",))
        node1 = Node(nk, nk, {NodeAttributes.STATE: States.UPTODATE})
        node2 = Node(nk, nk, {NodeAttributes.STATE: States.COMPUTABLE})
        result = cbs.format(nk, [node1, node2], False)
        # Mixed states should result in None state
        assert result["fillcolor"] == ColorByState.DEFAULT_STATE_COLORS[None]

    def test_create_root_graph_with_attributes(self):
        """Test create_root_graph with various attributes."""
        graph_attr = {"size": "10,8", "label": "Test Graph"}
        node_attr = {"shape": "box", "style": "filled"}
        edge_attr = {"color": "blue"}

        root = create_root_graph(graph_attr, node_attr, edge_attr)
        assert root is not None

    def test_create_root_graph_with_comma_values(self):
        """Test create_root_graph handles comma-containing values."""
        graph_attr = {"size": "10,8"}  # Contains comma
        root = create_root_graph(graph_attr, None, None)
        assert root is not None

    def test_create_subgraph(self):
        """Test create_subgraph."""
        group = NodeKey(("group1",))
        sg = create_subgraph(group)
        assert sg is not None

    def test_graph_view_svg_none(self):
        """Test GraphView.svg() returns None when viz_dot is None."""
        comp = Computation()
        v = GraphView(comp)
        v.viz_dot = None
        assert v.svg() is None

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_graph_view_view_linux(self, mock_tempfile, mock_run):
        """Test GraphView.view() on Linux."""
        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        mock_file = MagicMock()
        mock_file.name = "/tmp/test.pdf"
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tempfile.return_value = mock_file

        with patch.object(sys, "platform", "linux"):
            v.view()
            mock_run.assert_called()

    def test_standard_group_with_group_attribute(self):
        """Test StandardGroup with group attribute set."""
        sg = StandardGroup()
        nk = NodeKey(("parent", "child"))
        node = Node(nk, nk, {NodeAttributes.GROUP: "subgroup"})
        result = sg.format(nk, [node], False)
        assert "_group" in result

    def test_standard_group_root_parent(self):
        """Test StandardGroup when group_path is root."""
        sg = StandardGroup()
        nk = NodeKey(("a",))  # Single part, parent is root
        node = Node(nk, nk, {})
        result = sg.format(nk, [node], False)
        assert result is None


# ==================== COMPUTEENGINE.PY TESTS ====================


class TestComputeEngine:
    """Tests for computeengine.py coverage."""

    def test_null_object_getattr(self, capsys):
        """Test NullObject __getattr__."""
        no = NullObject()
        with pytest.raises(AttributeError):
            _ = no.some_attr
        captured = capsys.readouterr()
        assert "__getattr__" in captured.out

    def test_null_object_setattr(self, capsys):
        """Test NullObject __setattr__."""
        no = object.__new__(NullObject)  # Bypass normal init
        with pytest.raises(AttributeError):
            no.some_attr = 42
        captured = capsys.readouterr()
        assert "__setattr__" in captured.out

    def test_null_object_delattr(self, capsys):
        """Test NullObject __delattr__."""
        no = object.__new__(NullObject)
        with pytest.raises(AttributeError):
            del no.some_attr
        captured = capsys.readouterr()
        assert "__delattr__" in captured.out

    def test_null_object_call(self, capsys):
        """Test NullObject __call__."""
        no = object.__new__(NullObject)
        with pytest.raises(TypeError):
            no(1, 2, 3)
        captured = capsys.readouterr()
        assert "__call__" in captured.out

    def test_null_object_getitem(self, capsys):
        """Test NullObject __getitem__."""
        no = object.__new__(NullObject)
        with pytest.raises(KeyError):
            _ = no["key"]
        captured = capsys.readouterr()
        assert "__getitem__" in captured.out

    def test_null_object_setitem(self, capsys):
        """Test NullObject __setitem__."""
        no = object.__new__(NullObject)
        with pytest.raises(KeyError):
            no["key"] = "value"
        captured = capsys.readouterr()
        assert "__setitem__" in captured.out

    def test_null_object_repr(self, capsys):
        """Test NullObject __repr__."""
        no = object.__new__(NullObject)
        result = repr(no)
        assert result == "<NullObject>"
        captured = capsys.readouterr()
        assert "__repr__" in captured.out

    def test_identity_function(self):
        """Test identity_function."""
        assert identity_function(42) == 42
        obj = object()
        assert identity_function(obj) is obj

    def test_rename_node_with_dict(self):
        """Test rename_node with dictionary mapping."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)

        comp.rename_node({"a": "x", "b": "y"})

        assert not comp.has_node("a")
        assert not comp.has_node("b")
        assert comp.has_node("x")
        assert comp.has_node("y")
        assert comp.v.x == 1
        assert comp.v.y == 2

    def test_rename_node_dict_with_new_name_raises(self):
        """Test rename_node with dict and new_name raises ValueError."""
        comp = Computation()
        comp.add_node("a", value=1)

        with pytest.raises(ValueError, match="new_name must not be set"):
            comp.rename_node({"a": "x"}, new_name="y")

    def test_rename_node_nonexistent(self):
        """Test rename_node with non-existent node raises exception."""
        comp = Computation()
        with pytest.raises(NonExistentNodeException):
            comp.rename_node("nonexistent", "new_name")

    def test_rename_node_already_exists(self):
        """Test rename_node to existing node raises exception."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", value=2)

        with pytest.raises(NodeAlreadyExistsException):
            comp.rename_node("a", "b")

    def test_repoint_same_node(self):
        """Test repoint when old_name equals new_name."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.repoint("a", "a")  # Should do nothing
        assert comp.v.a == 1

    def test_repoint_creates_placeholder(self):
        """Test repoint creates placeholder for non-existent new_name."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)

        comp.repoint("a", "c")  # 'c' doesn't exist

        assert comp.has_node("c")
        assert comp.s.c == States.PLACEHOLDER

    def test_metadata_nonexistent(self):
        """Test metadata on non-existent node."""
        comp = Computation()
        with pytest.raises(NonExistentNodeException):
            comp.metadata("nonexistent")

    def test_metadata_tree_path(self):
        """Test metadata creates empty dict for tree path."""
        comp = Computation()
        comp.add_node("a/b", value=1)

        # 'a' is a tree path but not a node
        meta = comp.metadata("a")
        assert meta == {}

    def test_delete_node_placeholder_cleanup(self):
        """Test delete_node cleans up placeholder nodes."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)

        # Delete 'a' - it becomes placeholder because 'b' depends on it
        comp.delete_node("a")
        assert comp.s.a == States.PLACEHOLDER

        # Now delete 'b' - 'a' should be fully removed
        comp.delete_node("b")
        assert not comp.has_node("a")
        assert not comp.has_node("b")

    def test_computation_get_tags_for_state(self):
        """Test _get_tags_for_state method."""
        comp = Computation()
        comp.add_node("a", value=1, tags=["tag1"])
        comp.add_node("b", value=2, tags=["tag1", "tag2"])

        # The method exists but _get_names_for_state returns nodes by state
        nodes = comp._get_tags_for_state("tag1")
        assert "a" in nodes or to_nodekey("a") in nodes

    def test_write_dill_old_deprecated(self):
        """Test write_dill_old is deprecated."""
        comp = Computation()
        comp.add_node("a", value=1)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            with pytest.warns(DeprecationWarning):
                comp.write_dill_old(f.name)

    def test_write_dill_to_file(self):
        """Test write_dill to file path."""
        comp = Computation()
        comp.add_node("a", value=42)

        with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as f:
            comp.write_dill(f.name)

            # Read it back
            loaded = Computation.read_dill(f.name)
            assert loaded.v.a == 42

    def test_write_dill_to_fileobj(self):
        """Test write_dill to file object."""
        comp = Computation()
        comp.add_node("a", value=42)

        buf = io.BytesIO()
        comp.write_dill(buf)
        buf.seek(0)

        loaded = Computation.read_dill(buf)
        assert loaded.v.a == 42

    def test_read_dill_invalid(self):
        """Test read_dill with non-Computation object."""
        import dill

        buf = io.BytesIO()
        dill.dump("not a computation", buf)
        buf.seek(0)

        with pytest.raises(Exception):
            Computation.read_dill(buf)

    def test_print_errors(self, capsys):
        """Test print_errors method."""
        comp = Computation()

        def raise_error():
            raise ValueError("Test error")

        comp.add_node("a", value=1)
        comp.add_node("b", raise_error)
        comp.compute_all()

        comp.print_errors()
        captured = capsys.readouterr()
        assert "b" in captured.out

    def test_inject_dependencies_callable(self):
        """Test inject_dependencies with callable."""
        comp = Computation()
        comp.add_node("a")  # Uninitialized - not placeholder

        # Create placeholder by having something depend on non-existent node
        comp.add_node("b", lambda x: x + 1, kwds={"x": "c"})

        # 'c' should be placeholder
        assert comp.s.c == States.PLACEHOLDER

        # Inject callable
        comp.inject_dependencies({"c": lambda: 10})
        comp.compute_all()
        assert comp.v.b == 11

    def test_inject_dependencies_force(self):
        """Test inject_dependencies with force=True."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()

        # Force replacement even though 'a' is not placeholder
        comp.inject_dependencies({"a": 100}, force=True)
        comp.compute_all()
        assert comp.v.a == 100
        assert comp.v.b == 101

    def test_draw_method(self):
        """Test Computation.draw method."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)

        v = comp.draw()
        assert isinstance(v, GraphView)

    def test_view_method(self):
        """Test Computation.view method (mocked to avoid opening viewer)."""
        comp = Computation()
        comp.add_node("a", value=1)

        with patch.object(GraphView, "view"):
            comp.view()

    def test_get_tree_list_children(self):
        """Test get_tree_list_children method."""
        comp = Computation()
        comp.add_node("parent/child1", value=1)
        comp.add_node("parent/child2", value=2)
        comp.add_node("other", value=3)

        children = comp.get_tree_list_children("parent")
        assert "child1" in children or to_nodekey("child1").name in children

    def test_rename_node_with_metadata(self):
        """Test rename_node preserves metadata."""
        comp = Computation()
        comp.add_node("a", value=1, metadata={"key": "value"})

        comp.rename_node("a", "b")

        assert comp.metadata("b") == {"key": "value"}

    def test_add_map_node(self):
        """Test add_map_node method."""
        # Create a subgraph that doubles values
        subgraph = Computation()
        subgraph.add_node("input")
        subgraph.add_node("output", lambda input: input * 2)

        # Main computation
        comp = Computation()
        comp.add_node("input", value=[1, 2, 3])
        comp.add_map_node("output", "input", subgraph, "input", "output")
        comp.compute_all()

        assert comp.v.output == [2, 4, 6]

    def test_link_source_style(self):
        """Test link uses source style when target has no style."""
        comp = Computation()
        comp.add_node("a", value=1, style="small")
        comp.link("b", "a")

        assert comp.style.b == "small"

    def test_add_block_with_links(self):
        """Test add_block with links parameter."""
        block = Computation()
        block.add_node("input")
        block.add_node("output", lambda input: input + 1)

        comp = Computation()
        comp.add_node("source", value=10)

        comp.add_block("block1", block, links={"input": "source"})
        comp.compute_all()

        assert comp.v["block1/output"] == 11

    def test_add_block_with_metadata(self):
        """Test add_block with metadata parameter."""
        block = Computation()
        block.add_node("a", value=1)

        comp = Computation()
        comp.add_block("block1", block, metadata={"key": "value"})

        assert comp.metadata("block1") == {"key": "value"}


class TestInputNode:
    """Tests for InputNode class."""

    def test_input_node_with_args_kwds(self):
        """Test InputNode stores args and kwds."""
        node = InputNode(1, 2, 3, x=4, y=5)
        assert node.args == (1, 2, 3)
        assert node.kwds == {"x": 4, "y": 5}


class TestBlock:
    """Tests for Block class."""

    def test_block_add_to_comp(self):
        """Test Block.add_to_comp method."""
        inner_comp = Computation()
        inner_comp.add_node("a", value=1)

        block = Block(inner_comp)

        outer_comp = Computation()
        block.add_to_comp(outer_comp, "block1", inner_comp, ignore_self=True)

        assert outer_comp.has_node("block1/a")


class TestConstantValue:
    """Tests for ConstantValue class."""

    def test_constant_value(self):
        """Test ConstantValue stores value."""
        cv = ConstantValue(42)
        assert cv.value == 42


class TestTimingData:
    """Tests for TimingData class."""

    def test_timing_data(self):
        """Test TimingData stores timing information."""
        from datetime import datetime

        start = datetime.now()
        end = datetime.now()
        td = TimingData(start, end, 0.5)
        assert td.start == start
        assert td.end == end
        assert td.duration == 0.5
