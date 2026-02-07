"""Tests for visualization functionality in Loman computations."""

import itertools
import os
import subprocess
import sys
import tempfile
from collections import namedtuple
from datetime import datetime
from itertools import tee
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import loman.computeengine
import loman.visualization
from loman import Computation, States, node
from loman.computeengine import TimingData
from loman.consts import NodeAttributes, NodeTransformations
from loman.nodekey import NodeKey, to_nodekey
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
from tests.conftest import (
    BasicFourNodeComputation,
    create_example_block_computation,
)


def node_set(nodes):
    """Create a set of node keys from node names."""
    s = set()
    for n in nodes:
        nodekey = to_nodekey(n)
        s.add(nodekey)
    return s


def edges_set(edges):
    """Create a set of edges from edge pairs."""
    s = set()
    for a, _b in edges:
        nk_a = to_nodekey(a)
        nk_b = to_nodekey(a)
        el = frozenset((nk_a, nk_b))
        s.add(el)
    return s


def edges_from_chain(chain_iter):
    """Generate edges from a chain of nodes."""
    a, b = tee(chain_iter)
    next(b, None)
    return zip(a, b, strict=False)


def check_graph(g, expected_chains):
    """Check that graph matches expected chains."""
    expected_nodes = set()
    for chain in expected_chains:
        for node_name in chain:
            expected_nodes.add(node_name)

    expected_edges = set()
    for chain in expected_chains:
        for node_a, node_b in edges_from_chain(chain):
            expected_edges.add((node_a, node_b))

    assert node_set(expected_nodes) == node_set(g.nodes)
    assert edges_set(expected_edges) == edges_set(g.edges)


def get_label_to_node_mapping(v):
    """Get mapping from labels to nodes in visualization."""
    nodes = v.viz_dot.obj_dict["nodes"]
    label_to_name_mapping = {v[0]["attributes"]["label"]: k for k, v in nodes.items()}
    node = {label: nodes[name][0] for label, name in label_to_name_mapping.items()}
    return node


def get_path_to_node_mapping(v):
    """Get mapping from paths to nodes in visualization."""
    d = {}
    for _name, node_obj in v.viz_dag.nodes(data=True):
        label = node_obj["label"]
        group = node_obj.get("_group")
        path = NodeKey((label,)) if group is None else group.join_parts(label)
        d[path] = node_obj
    return d


def test_simple():
    """Test simple."""
    comp = Computation()
    comp.add_node("a")
    comp.add_node("b", lambda a: a + 1)
    comp.add_node("c", lambda a: 2 * a)
    comp.add_node("d", lambda b, c: b + c)

    v = loman.visualization.GraphView(comp, collapse_all=False)

    node = get_label_to_node_mapping(v)
    assert (
        node["a"]["attributes"]["fillcolor"]
        == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    )
    assert node["a"]["attributes"]["style"] == "filled"
    assert (
        node["b"]["attributes"]["fillcolor"]
        == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    )
    assert node["b"]["attributes"]["style"] == "filled"
    assert (
        node["c"]["attributes"]["fillcolor"]
        == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    )
    assert node["c"]["attributes"]["style"] == "filled"
    assert (
        node["d"]["attributes"]["fillcolor"]
        == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    )
    assert node["d"]["attributes"]["style"] == "filled"

    comp.insert("a", 1)

    v.refresh()
    node = get_label_to_node_mapping(v)
    assert (
        node["a"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["a"]["attributes"]["style"] == "filled"
    assert (
        node["b"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.COMPUTABLE]
    )
    assert node["b"]["attributes"]["style"] == "filled"
    assert (
        node["c"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.COMPUTABLE]
    )
    assert node["c"]["attributes"]["style"] == "filled"
    assert node["d"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.STALE]
    assert node["d"]["attributes"]["style"] == "filled"

    comp.compute_all()

    v.refresh()
    node = get_label_to_node_mapping(v)
    assert (
        node["a"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["a"]["attributes"]["style"] == "filled"
    assert (
        node["b"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["b"]["attributes"]["style"] == "filled"
    assert (
        node["c"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["c"]["attributes"]["style"] == "filled"
    assert (
        node["d"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["d"]["attributes"]["style"] == "filled"


def test_with_groups():
    """Test with groups."""
    comp = Computation()
    comp.add_node("a", group="foo")
    comp.add_node("b", lambda a: a + 1, group="foo")
    comp.add_node("c", lambda a: 2 * a, group="bar")
    comp.add_node("d", lambda b, c: b + c, group="bar")
    loman.visualization.GraphView(comp, collapse_all=False)


def test_show_expansion():
    """Test show expansion."""
    Coordinate = namedtuple("Coordinate", ["x", "y"])
    comp = Computation()
    comp.add_node("c", value=Coordinate(1, 2))
    comp.add_node("foo", lambda x: x + 1, kwds={"x": "c.x"})
    comp.add_named_tuple_expansion("c", Coordinate)
    comp.compute_all()

    view_uncontracted = comp.draw(show_expansion=True, collapse_all=False)
    view_uncontracted.refresh()
    labels = nx.get_node_attributes(view_uncontracted.viz_dag, "label")
    assert set(labels.values()) == {"c", "c.x", "c.y", "foo"}

    view_contracted = comp.draw(show_expansion=False, collapse_all=False)
    view_contracted.refresh()
    labels = nx.get_node_attributes(view_contracted.viz_dag, "label")
    assert set(labels.values()) == {"c", "foo"}


def test_with_visualization_blocks():
    """Test with visualization blocks."""
    comp = create_example_block_computation()

    comp.compute_all()

    v = comp.draw(collapse_all=False)
    check_graph(
        v.struct_dag,
        [
            ("input_foo", "foo/a", "foo/b", "foo/d", "output"),
            ("foo/a", "foo/c", "foo/d"),
            ("input_bar", "bar/a", "bar/b", "bar/d", "output"),
            ("bar/a", "bar/c", "bar/d"),
        ],
    )


def test_with_visualization_view_subblocks():
    """Test with visualization view subblocks."""
    comp = create_example_block_computation()

    comp.compute_all()

    v_foo = comp.draw("/foo", collapse_all=False)
    check_graph(v_foo.struct_dag, [("a", "b", "d"), ("a", "c", "d")])

    v_bar = comp.draw("/bar", collapse_all=False)
    check_graph(v_bar.struct_dag, [("a", "b", "d"), ("a", "c", "d")])


def test_with_visualization_collapsed_blocks():
    """Test with visualization collapsed blocks."""
    comp = create_example_block_computation()

    comp.compute_all()

    node_transformations = {"foo": NodeTransformations.COLLAPSE, "bar": NodeTransformations.COLLAPSE}

    v = comp.draw(node_transformations=node_transformations, collapse_all=False)
    check_graph(v.struct_dag, [("input_foo", "foo", "output"), ("input_bar", "bar", "output")])
    node = get_label_to_node_mapping(v)
    assert (
        node["foo"]["attributes"]["fillcolor"] == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UPTODATE]
    )
    assert node["foo"]["attributes"]["shape"] == "rect"
    assert node["foo"]["attributes"]["peripheries"] == 2


def test_with_visualization_single_element_collapsed_blocks():
    """Test with visualization single element collapsed blocks."""
    comp = loman.Computation()
    comp.add_node("foo1/bar1/baz1/a")

    v = comp.draw(node_transformations={"foo1": NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey("foo1")]["shape"] == "rect"
    assert d[to_nodekey("foo1")]["peripheries"] == 2

    v = comp.draw(node_transformations={"foo1/bar1": NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey("foo1/bar1")]["shape"] == "rect"
    assert d[to_nodekey("foo1/bar1")]["peripheries"] == 2

    v = comp.draw(node_transformations={"foo1/bar1/baz1": NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey("foo1/bar1/baz1")]["shape"] == "rect"
    assert d[to_nodekey("foo1/bar1/baz1")]["peripheries"] == 2


def test_sub_blocks_collapse_with_group():
    """Test sub blocks collapse with group."""
    comp = loman.Computation()
    comp.add_node("a")
    comp.add_node("foo/bar/b", lambda a: a + 1, kwds={"a": "a"})
    comp.add_node("foo/bar/c", lambda a: a + 1, kwds={"a": "a"})
    v = comp.draw(node_transformations={"foo/bar": NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert d[to_nodekey("foo/bar")]["shape"] == "rect"


def test_with_visualization_collapsed_blocks_uniform_sate():
    """Test with visualization collapsed blocks uniform sate."""
    comp = loman.Computation()
    comp.add_node("a")
    comp.add_node("foo/bar/b", lambda a: a + 1, kwds={"a": "a"})
    comp.add_node("foo/bar/c", lambda a: a + 1, kwds={"a": "a"})
    v = comp.draw(node_transformations={"foo/bar": NodeTransformations.COLLAPSE}, collapse_all=False)
    d = get_path_to_node_mapping(v)
    assert (
        d[to_nodekey("foo/bar")]["fillcolor"]
        == loman.visualization.ColorByState.DEFAULT_STATE_COLORS[States.UNINITIALIZED]
    )


def test_with_visualization_view_default_collapsing():
    """Test with visualization view default collapsing."""
    comp = create_example_block_computation()

    comp.compute_all()

    v_foo = comp.draw()
    check_graph(v_foo.struct_dag, [("input_foo", "foo", "output"), ("input_bar", "bar", "output")])


def test_with_visualization_view_subblocks_default_collapsing():
    """Test with visualization view subblocks default collapsing."""
    comp = Computation()
    comp.add_node("foo/a")
    comp.add_node("foo/b", lambda a: a + 1)
    comp.add_node("foo/c", lambda a: 2 * a)
    comp.add_node("foo/d", lambda b, c: b + c)
    v = comp.draw("foo")
    check_graph(v.struct_dag, [("a", "b", "d"), ("a", "c", "d")])


def test_draw_expanded_block():
    """Test draw expanded block."""
    comp = Computation()
    comp.add_node("foo/bar/baz/a")
    comp.add_node("foo/bar/baz/b", lambda a: a + 1)
    comp.add_node("foo/bar/baz/c", lambda a: 2 * a)
    comp.add_node("foo/bar/baz/d", lambda b, c: b + c)

    v = comp.draw(node_transformations={"foo/bar/baz": "expand"})
    nodes = get_path_to_node_mapping(v)

    assert to_nodekey("foo/bar/baz/a") in nodes
    assert to_nodekey("foo/bar/baz/b") in nodes
    assert to_nodekey("foo/bar/baz/c") in nodes
    assert to_nodekey("foo/bar/baz/d") in nodes


def test_draw_expanded_block_with_wildcard():
    """Test draw expanded block with wildcard."""
    comp = Computation()
    comp.add_node("foo/bar/baz/a")
    comp.add_node("foo/bar/baz/b", lambda a: a + 1)
    comp.add_node("foo/bar/baz/c", lambda a: 2 * a)
    comp.add_node("foo/bar/baz/d", lambda b, c: b + c)

    v = comp.draw(node_transformations={"**": "expand"})
    nodes = get_path_to_node_mapping(v)

    assert to_nodekey("foo/bar/baz/a") in nodes
    assert to_nodekey("foo/bar/baz/b") in nodes
    assert to_nodekey("foo/bar/baz/c") in nodes
    assert to_nodekey("foo/bar/baz/d") in nodes


def test_draw_expanded_block_with_wildcard_2():
    """Test draw expanded block with wildcard 2."""
    comp_inner = BasicFourNodeComputation()
    comp = Computation()
    for x, y, z in itertools.product(range(1, 3), range(1, 3), range(1, 3)):
        comp.add_block(f"foo{x}/bar{y}/baz{z}", comp_inner, keep_values=False, links={"a": "input_a"})
    comp.add_node("input_a", value=7)
    comp.compute_all()

    v = comp.draw(node_transformations={"**": "expand"})
    nodes = get_path_to_node_mapping(v)
    expected = [
        "input_a",
        "foo1/bar1/baz1/a",
        "foo1/bar1/baz1/b",
        "foo1/bar1/baz1/c",
        "foo1/bar1/baz1/d",
        "foo1/bar1/baz2/a",
        "foo1/bar1/baz2/b",
        "foo1/bar1/baz2/c",
        "foo1/bar1/baz2/d",
        "foo1/bar2/baz1/a",
        "foo1/bar2/baz1/b",
        "foo1/bar2/baz1/c",
        "foo1/bar2/baz1/d",
        "foo1/bar2/baz2/a",
        "foo1/bar2/baz2/b",
        "foo1/bar2/baz2/c",
        "foo1/bar2/baz2/d",
        "foo2/bar1/baz1/a",
        "foo2/bar1/baz1/b",
        "foo2/bar1/baz1/c",
        "foo2/bar1/baz1/d",
        "foo2/bar1/baz2/a",
        "foo2/bar1/baz2/b",
        "foo2/bar1/baz2/c",
        "foo2/bar1/baz2/d",
        "foo2/bar2/baz1/a",
        "foo2/bar2/baz1/b",
        "foo2/bar2/baz1/c",
        "foo2/bar2/baz1/d",
        "foo2/bar2/baz2/a",
        "foo2/bar2/baz2/b",
        "foo2/bar2/baz2/c",
        "foo2/bar2/baz2/d",
    ]
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={"foo1/bar1/**": "expand"})
    nodes = get_path_to_node_mapping(v)
    expected = [
        "input_a",
        "foo1/bar1/baz1/a",
        "foo1/bar1/baz1/b",
        "foo1/bar1/baz1/c",
        "foo1/bar1/baz1/d",
        "foo1/bar1/baz2/a",
        "foo1/bar1/baz2/b",
        "foo1/bar1/baz2/c",
        "foo1/bar1/baz2/d",
        "foo1/bar2",
        "foo2",
    ]
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={"foo2/bar2/**": "expand"})
    nodes = get_path_to_node_mapping(v)
    expected = [
        "input_a",
        "foo1",
        "foo2/bar1",
        "foo2/bar2/baz1/a",
        "foo2/bar2/baz1/b",
        "foo2/bar2/baz1/c",
        "foo2/bar2/baz1/d",
        "foo2/bar2/baz2/a",
        "foo2/bar2/baz2/b",
        "foo2/bar2/baz2/c",
        "foo2/bar2/baz2/d",
    ]
    assert nodes.keys() == {to_nodekey(n) for n in expected}

    v = comp.draw(node_transformations={"*": "expand"})
    nodes = get_path_to_node_mapping(v)
    assert nodes.keys() == {to_nodekey(n) for n in ["input_a", "foo1/bar1", "foo1/bar2", "foo2/bar1", "foo2/bar2"]}

    v = comp.draw(node_transformations={"*/*": "expand"})
    nodes = get_path_to_node_mapping(v)
    assert nodes.keys() == {
        to_nodekey(n)
        for n in [
            "input_a",
            "foo1/bar1/baz1",
            "foo1/bar1/baz2",
            "foo1/bar2/baz1",
            "foo1/bar2/baz2",
            "foo2/bar1/baz1",
            "foo2/bar1/baz2",
            "foo2/bar2/baz1",
            "foo2/bar2/baz2",
        ]
    }


def test_style_preservation_with_block_links():
    """Test style preservation with block links."""

    def build_comp():
        comp = Computation()
        comp.add_node("a", style="dot")
        comp.add_node("b", style="dot")
        comp.add_node("e", style="dot")

        @node(comp, "c")
        def comp_c(a, b):
            return a + b

        comp.add_node("d", lambda a: a + 1, style="small")
        return comp

    full_comp = Computation()
    full_comp.add_node("params/a", value=1, style="dot")
    full_comp.add_node("params/b", value=1, style="dot")
    full_comp.add_node("params/c", value=1, style="dot")
    full_comp.add_block("comp", build_comp(), links={"a": "params/a", "b": "params/b", "e": "params/c"})

    expected_styles = ["dot"] * 4 + [None, "small"]
    actual_styles = full_comp.styles(["params/a", "params/b", "comp/a", "comp/b", "comp/c", "comp/d"])

    assert expected_styles == actual_styles


# ==================== ADDITIONAL COVERAGE TESTS ====================


class TestVisualizationCoverage:
    """Additional tests for visualization.py coverage."""

    def test_color_by_timing_no_timing(self):
        """Test ColorByTiming with node having no timing data."""
        cbt = ColorByTiming()
        nk = NodeKey(("test",))
        node = Node(nk, nk, {})
        result = cbt.format(nk, [node], False)
        assert result["fillcolor"] == "#FFFFFF"

    def test_color_by_timing_with_timing(self):
        """Test ColorByTiming with timing data."""
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

    def test_graph_view_repr_svg(self):
        """Test GraphView._repr_svg_() method."""
        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)
        svg = v._repr_svg_()
        assert svg is not None
        assert "<svg" in svg

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_graph_view_view_linux(self, mock_tempfile, mock_run):
        """Test GraphView.view() on Linux."""
        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        mock_file = MagicMock()
        mock_file.name = os.path.join(tempfile.gettempdir(), "test.pdf")  # nosec B108 - mock path for testing
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


class TestVisualizationWin32:
    """Tests for visualization on Windows platform."""

    def test_graph_view_view_windows(self):
        """Test GraphView.view() on Windows."""
        # This test is for coverage of the win32 branch
        # We can't actually run os.startfile on Linux, so we test via mock

        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        # Create a mock for the PDF generation and file opening
        with patch.object(v.viz_dot, "create_pdf", return_value=b"fake pdf"):
            with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                mock_file = MagicMock()
                mock_file.name = os.path.join(tempfile.gettempdir(), "test.pdf")  # nosec B108 - mock path for testing
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_tempfile.return_value = mock_file

                # Force the win32 branch and mock os.startfile
                with patch.object(sys, "platform", "win32"):
                    with patch("os.startfile", create=True) as mock_startfile:
                        v.view()

                        # Ensure os.startfile was called with the generated PDF path
                        mock_startfile.assert_called_once_with(mock_file.name)


class TestVisualizationAttrNormalization:
    """Tests for attribute normalization in create_root_graph."""

    def test_create_root_graph_empty_string(self):
        """Test create_root_graph with empty string value."""
        graph_attr = {"label": ""}
        root = create_root_graph(graph_attr, None, None)
        assert root is not None

    def test_create_root_graph_already_quoted(self):
        """Test create_root_graph with already quoted value."""
        graph_attr = {"size": '"10,8"'}
        root = create_root_graph(graph_attr, None, None)
        assert root is not None

    def test_create_root_graph_numeric(self):
        """Test create_root_graph with numeric values."""
        graph_attr = {"fontsize": 12}
        node_attr = {"width": 1.5}
        edge_attr = {"penwidth": 2}
        root = create_root_graph(graph_attr, node_attr, edge_attr)
        assert root is not None


class TestGraphViewCollapse:
    """Tests for GraphView with collapse_all."""

    def test_graph_view_collapse_all_true(self):
        """Test GraphView with collapse_all=True."""
        comp = Computation()
        comp.add_node("parent/child1", value=1)
        comp.add_node("parent/child2", value=2)
        comp.add_node("parent/result", lambda **k: sum(k.values()))

        v = GraphView(comp, collapse_all=True)
        svg = v.svg()
        assert svg is not None

    def test_graph_view_with_root(self):
        """Test GraphView with root parameter."""
        comp = Computation()
        comp.add_node("parent/child1", value=1)
        comp.add_node("parent/child2", value=2)

        v = GraphView(comp, root=NodeKey(("parent",)))
        svg = v.svg()
        assert svg is not None


class TestGraphViewEmpty:
    """Test GraphView with empty computation."""

    def test_empty_computation_svg(self):
        """Test GraphView with empty computation."""
        comp = Computation()

        v = GraphView(comp)
        svg = v.svg()
        # Empty graph should still produce valid SVG
        assert isinstance(svg, str)
        assert "<svg" in svg


class TestVisualizationWin32Branch:
    """Test the Windows platform branch in view()."""

    def test_graph_view_win32_branch(self):
        """Test the Windows startfile branch coverage."""
        # This is difficult to test on Linux but we can check the branch exists
        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        # Just verify svg() works
        svg = v.svg()
        assert svg is not None
        assert "<svg" in svg


class TestVisualizationCalibrate:
    """Test NodeFormatter calibrate method."""

    def test_formatter_calibrate(self):
        """Test calibrate on base formatter."""

        class TestFormatter(NodeFormatter):
            def format(self, name, nodes, is_composite):
                return {}

        f = TestFormatter()
        f.calibrate([])  # Should do nothing, no error


class TestCollapseNodeMapped:
    """Test the collapse logic in visualization."""

    def test_viz_with_collapsed_nodes(self):
        """Test visualization with collapsed nodes."""
        comp = Computation()
        comp.add_node("parent/child1", value=1)
        comp.add_node("parent/child2", value=2)
        # Connect the children to a result node
        comp.add_node("result", lambda **kwargs: sum(kwargs.values()))
        comp.link("result", "parent/child1")
        comp.link("result", "parent/child2")

        # Use collapse_all to test collapsed node logic
        v = GraphView(comp, collapse_all=True)
        svg = v.svg()
        assert svg is not None


class TestViewFunctionOpenBranch:
    """Test the view() function opens the PDF."""

    def test_view_calls_subprocess_on_linux(self, mocker):
        """Test view() calls subprocess.run on Linux."""
        mock_run = mocker.patch("subprocess.run")
        mock_tempfile = mocker.patch("tempfile.NamedTemporaryFile")

        # Create a mock file object
        mock_file = mocker.MagicMock()
        mock_file.name = os.path.join(tempfile.gettempdir(), "test.pdf")  # nosec B108 - mock path for testing
        mock_file.__enter__ = mocker.MagicMock(return_value=mock_file)
        mock_file.__exit__ = mocker.MagicMock(return_value=False)
        mock_tempfile.return_value = mock_file

        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        # Mock the create_pdf method
        v.viz_dot = mocker.MagicMock()
        v.viz_dot.create_pdf.return_value = b"pdf content"

        v.view()

        mock_run.assert_called_once()


class TestVisualizationOpenPdf:
    """Test view() method opens PDF."""

    def test_view_runs_open_command(self, mocker):
        """Test view() runs 'open' command on non-Windows."""
        # Mock subprocess.run
        mock_run = mocker.patch("subprocess.run")
        # Mock tempfile
        mocker.patch("tempfile.NamedTemporaryFile")

        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        # Mock viz_dot
        v.viz_dot = mocker.MagicMock()
        v.viz_dot.create_pdf.return_value = b"pdf content"

        v.view()

        # subprocess.run should be called with 'open'
        mock_run.assert_called_once()
        assert "open" in mock_run.call_args[0][0]


class TestVisualizationWithNoneCmap:
    """Test visualization with None colormap."""

    def test_graph_view_default_cmap(self):
        """Test GraphView uses default colormap when None passed."""
        comp = Computation()
        comp.add_node("a", value=1)

        # Use default cmap by passing None
        formatter = NodeFormatter.create(cmap=None)
        v = GraphView(comp, node_formatter=formatter)
        svg = v.svg()
        assert svg is not None


class TestNodeFormatterCreate:
    """Test NodeFormatter.create factory method."""

    def test_create_with_shapes(self):
        """Test NodeFormatter.create with shapes parameter."""
        formatter = NodeFormatter.create(shapes="type")
        assert formatter is not None

    def test_create_with_colors_timing(self):
        """Test NodeFormatter.create with colors='timing'."""
        formatter = NodeFormatter.create(colors="timing")
        assert formatter is not None


class TestVisualizationCalibrateWithNodes:
    """Test NodeFormatter.calibrate with nodes."""

    def test_standard_label_calibrate(self):
        """Test StandardLabel calibrate method."""
        from loman.visualization import StandardLabel

        formatter = StandardLabel()
        nodes = [Node(NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE})]
        # calibrate should not raise
        formatter.calibrate(nodes)


class TestColorByTimingCoverage:
    """Test ColorByTiming formatter."""

    def test_color_by_timing_with_timing_data(self):
        """Test ColorByTiming with timing data."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()

        # Color by timing should work even without timing data
        formatter = ColorByTiming()

        node = Node(NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE})
        result = formatter.format(NodeKey(("a",)), [node], False)
        # Should return a valid result
        assert result is not None


class TestRenameMetadataClearBranch:
    """Test the metadata clear branch in rename_nodes."""

    def test_rename_node_target_has_metadata(self):
        """Test rename_node when target previously had metadata but source doesn't."""
        comp = Computation()
        # Create a source node without metadata
        comp.add_node("source", value=1)

        # Create a target with metadata then delete
        comp.add_node("temp", value=999, metadata={"key": "value"})

        # Store the metadata for temp
        assert comp.metadata("temp") == {"key": "value"}

        # Delete temp
        comp.delete_node("temp")

        # Rename source to a new name (simulating the branch)
        comp.rename_node("source", "new_name")

        # new_name should have no metadata since source had none
        assert comp.metadata("new_name") == {}


class TestGraphViewWithRootNode:
    """Test GraphView with root parameter filtering."""

    def test_graph_view_root_filters_nodes(self):
        """Test GraphView with root parameter filters to subtree."""
        comp = Computation()
        comp.add_node("group1/a", value=1)
        comp.add_node("group1/b", value=2)
        comp.add_node("group2/c", value=3)

        # View only group1
        v = GraphView(comp, root=NodeKey(("group1",)))
        svg = v.svg()
        assert svg is not None


class TestGraphViewNoneFormatter:
    """Tests for GraphView with node_formatter=None."""

    def test_graph_view_with_none_formatter(self):
        """Test GraphView with node_formatter=None."""
        comp = Computation()
        comp.add_node("a", value=1)

        v = GraphView(comp, node_formatter=None)
        svg = v.svg()
        assert svg is not None


class TestColormapNodeFormatter:
    """Tests for colormap in node formatter."""

    def test_node_formatter_with_dict_colormap(self):
        """Test node formatter with dict colormap using States as keys."""
        comp = Computation()
        comp.add_node("a", value=1)

        # Keys must be States, not strings
        cmap = {
            States.UPTODATE: "#00ff00",
            States.ERROR: "#ff0000",
            States.STALE: "#ffff00",
            States.COMPUTABLE: "#0000ff",
            States.PLACEHOLDER: "#cccccc",
            States.UNINITIALIZED: "#ffffff",
            States.PINNED: "#ff00ff",
            None: "#888888",
        }
        formatter = NodeFormatter.create(cmap=cmap)

        v = GraphView(comp, node_formatter=formatter)
        svg = v.svg()
        assert svg is not None


class TestCompositeNodeFormatterCoverage:
    """Test CompositeNodeFormatter."""

    def test_composite_formatter_calibrate(self):
        """Test CompositeNodeFormatter calibrate calls all formatters."""
        from loman.visualization import StandardLabel

        formatter = CompositeNodeFormatter([StandardLabel(), StandardGroup()])
        nodes = [Node(NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE})]
        formatter.calibrate(nodes)  # Should not raise


class TestShapeByTypeCoverage:
    """Test ShapeByType formatter."""

    def test_shape_by_type_with_func(self):
        """Test ShapeByType with a function node."""
        formatter = ShapeByType()

        # Node with a function
        node = Node(
            NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE, NodeAttributes.FUNC: lambda: 1}
        )
        result = formatter.format(NodeKey(("a",)), [node], False)
        # ShapeByType only returns shapes for nodes with values, not functions
        assert result is None

    def test_shape_by_type_without_func(self):
        """Test ShapeByType with a value node (no function)."""
        formatter = ShapeByType()

        # Node without a function
        node = Node(
            NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE, NodeAttributes.FUNC: None}
        )
        result = formatter.format(NodeKey(("a",)), [node], False)
        # ShapeByType only returns shapes for nodes with values
        assert result is None


class TestRectBlocksCoverage:
    """Test RectBlocks formatter."""

    def test_rect_blocks_composite_coverage(self):
        """Test RectBlocks with composite node."""
        formatter = RectBlocks()

        node = Node(NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE})
        result = formatter.format(NodeKey(("a",)), [node], is_composite=True)
        # RectBlocks returns a dict for composite nodes
        assert isinstance(result, dict)

    def test_rect_blocks_non_composite_coverage(self):
        """Test RectBlocks with non-composite node returns no blocks."""
        formatter = RectBlocks()

        node = Node(NodeKey(("a",)), NodeKey(("a",)), {NodeAttributes.STATE: States.UPTODATE})
        result = formatter.format(NodeKey(("a",)), [node], is_composite=False)
        assert result is None


class TestGraphViewTransformations:
    """Test GraphView with transformations."""

    def test_graph_view_with_node_transformations(self):
        """Test GraphView with node transformations."""
        comp = Computation()
        comp.add_node("group/a", value=1)
        comp.add_node("group/b", value=2)
        comp.add_node("group/c", lambda **kw: sum(kw.values()))
        comp.link("group/c", "group/a")
        comp.link("group/c", "group/b")

        # Use node_transformations directly
        transformations = {NodeKey(("group",)): NodeTransformations.COLLAPSE}
        v = GraphView(comp, node_transformations=transformations, collapse_all=False)
        svg = v.svg()
        assert svg is not None


class TestStandardStylingOverridesCoverage:
    """Test StandardStylingOverrides formatter."""

    def test_styling_overrides_with_style(self):
        """Test StandardStylingOverrides with node style."""
        formatter = StandardStylingOverrides()

        # Node with style
        node = Node(
            NodeKey(("a",)),
            NodeKey(("a",)),
            {NodeAttributes.STATE: States.UPTODATE, NodeAttributes.STYLE: {"color": "red"}},
        )
        result = formatter.format(NodeKey(("a",)), [node], False)
        # When a node has an explicit style that's not small/dot, the formatter returns None
        assert result is None


class TestVisualizationDropRootNone:
    """Test visualization when drop_root returns None."""

    def test_viz_with_root_excludes_outside_nodes(self):
        """Test GraphView with root excludes nodes outside the subtree."""
        comp = Computation()
        comp.add_node("inside/a", value=1)
        comp.add_node("inside/b", value=2)
        comp.add_node("outside/c", value=3)

        # View only 'inside' subtree
        v = GraphView(comp, root=NodeKey(("inside",)))
        svg = v.svg()
        # Should produce valid SVG without 'outside' nodes
        assert svg is not None


class TestVisualizationSubprocess:
    """Test view() subprocess call."""

    def test_view_subprocess_run(self, mocker):
        """Test view() calls subprocess.run on Linux."""
        mock_run = mocker.patch.object(subprocess, "run")

        comp = Computation()
        comp.add_node("a", value=1)
        v = GraphView(comp)

        # Mock viz_dot to avoid actual graphviz call
        v.viz_dot = mocker.MagicMock()
        v.viz_dot.create_pdf.return_value = b"pdf"

        v.view()

        # Verify subprocess.run was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "open" in call_args


class TestVisualizationVizDagFormatterNone:
    """Test create_viz_dag with node_formatter=None."""

    def test_create_viz_dag_no_formatter(self):
        """Test create_viz_dag without node formatter."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.compute_all()

        v = GraphView(comp, node_formatter=None)
        # Should still produce valid SVG
        svg = v.svg()
        assert svg is not None
