"""Python-side integration tests for the live computation widget."""

from pathlib import Path

from loman import Computation, States
from loman.nodekey import to_nodekey
from loman.ui import ComputationWidget
from tests.conftest import create_example_block_computation


def make_widget() -> tuple[Computation, ComputationWidget]:
    """Create a small live graph and its widget."""
    comp = Computation()
    comp.add_node("price", value=10.0)
    comp.add_node("quantity", value=2)
    comp.add_node("value", lambda price, quantity: price * quantity)
    return comp, comp.widget(collapse_all=False)


def test_widget_follows_state_changes_without_relayout():
    """Computation events update state traits while retaining the SVG layout."""
    comp, widget = make_widget()
    try:
        svg = widget.graph_svg
        value_id = widget._view.node_index_map[to_nodekey("value")]
        assert widget.node_states[value_id] == States.COMPUTABLE.name

        comp.compute_all()

        assert widget.node_states[value_id] == States.UPTODATE.name
        assert widget.graph_svg == svg
        assert widget.revision == comp.revision
    finally:
        widget.close()


def test_widget_relayouts_after_graph_change_and_unsubscribes_on_close():
    """Structural events rebuild SVG, and a closed widget stops following."""
    comp, widget = make_widget()
    old_svg = widget.graph_svg

    comp.add_node("tax", lambda value: value * 0.2)

    assert widget.graph_svg != old_svg
    assert "tax" in widget.graph_svg
    widget.close()
    closed_revision = widget.revision
    comp.insert("price", 12.0)
    assert widget.revision == closed_revision


def test_selection_preserves_non_string_node_identity():
    """Opaque rendered IDs resolve back to real, possibly colliding names."""
    comp = Computation()
    comp.add_node(1, value="integer")
    comp.add_node("1", value="string")
    widget = comp.widget(collapse_all=False)
    try:
        integer_id = widget._view.node_index_map[to_nodekey(1)]
        string_id = widget._view.node_index_map[to_nodekey("1")]
        widget.selected_id = integer_id
        assert widget.selected_name == 1
        assert widget.detail["value"]["value"] == "integer"
        widget.selected_id = string_id
        assert widget.selected_name == "1"
        assert widget.detail["value"]["value"] == "string"
    finally:
        widget.close()


def test_widget_edits_input_and_computes_target():
    """Trait requests exercise the same round-trip as browser controls."""
    comp, widget = make_widget()
    try:
        price_id = widget._view.node_index_map[to_nodekey("price")]
        value_id = widget._view.node_index_map[to_nodekey("value")]
        widget.edit_request = {
            "id": price_id,
            "value": {"kind": "scalar", "type": "float", "value": 12.5},
            "sequence": 1,
        }
        assert comp.value("price") == 12.5
        assert comp.state("value") == States.COMPUTABLE

        widget.compute_request = {"id": value_id, "sequence": 2}
        assert comp.value("value") == 25.0
        assert widget.detail == {}
        widget.selected_id = value_id
        assert widget.detail["value"]["value"] == 25.0
    finally:
        widget.close()


def test_widget_expands_collapsed_block():
    """A composite-node request expands the existing GraphView hierarchy."""
    comp = create_example_block_computation()
    widget = comp.widget()
    try:
        foo_id = widget._view.node_index_map[to_nodekey("foo")]
        assert foo_id in widget.composite_ids

        widget.toggle_request = {"id": foo_id, "sequence": 1}

        visible = {str(node) for node in widget._view.node_index_map}
        assert {"foo/a", "foo/b", "foo/c", "foo/d"}.issubset(visible)
        assert widget.status == "Opened foo"
        widget.toggle_request = {"collapse_all": True, "sequence": 2}
        assert to_nodekey("foo") in widget._view.composite_nodes
        assert widget.status == "Collapsed all blocks"
    finally:
        widget.close()


def test_widget_clears_selection_when_relayout_reuses_rendered_id():
    """Selection cannot silently jump when Graphviz IDs are reused."""
    comp = create_example_block_computation()
    widget = comp.widget()
    try:
        foo_id = widget._view.node_index_map[to_nodekey("foo")]
        widget.toggle_request = {"id": foo_id, "request_id": "expand"}
        leaf_id = widget._view.node_index_map[to_nodekey("foo/a")]
        widget.selected_id = leaf_id
        assert widget.selected_name == "foo/a"

        widget.toggle_request = {"collapse_all": True, "request_id": "collapse"}

        assert widget.selected_id == ""
        assert widget.selected_name is None
        assert widget.detail == {}
    finally:
        widget.close()


def test_read_only_widget_rejects_compute_requests():
    """A crafted trait request cannot bypass the read-only UI."""
    comp = Computation()
    comp.add_node("input", value=2)
    comp.add_node("output", lambda input: input + 1)
    widget = comp.widget(collapse_all=False, editable=False)
    try:
        widget.compute_request = {"all": True, "request_id": "read-only"}

        assert comp.state("output") == States.COMPUTABLE
        assert widget.status == "Compute failed: this widget is read-only"
    finally:
        widget.close()


def test_widget_rejects_stale_derived_traits_from_browser():
    """Reconnect echoes cannot corrupt the canonical Python-side view model."""
    comp, widget = make_widget()
    try:
        price_id = widget._view.node_index_map[to_nodekey("price")]
        widget.selected_id = price_id
        expected_detail = widget.detail
        expected_svg = widget.graph_svg
        expected_composites = widget.composite_ids
        expected_states = widget.node_states

        widget.composite_ids = ["stale"]
        widget.detail = {"id": price_id, "name": "wrong node"}
        widget.graph_svg = "<svg>stale</svg>"
        widget.node_states = {"stale": States.ERROR.name}
        widget.revision = -1

        assert widget.composite_ids == expected_composites
        assert widget.detail == expected_detail
        assert widget.graph_svg == expected_svg
        assert widget.node_states == expected_states
        assert widget.revision == comp.revision
    finally:
        widget.close()


def test_widget_assets_cover_python_trait_contract():
    """The hand-written frontend references every interactive trait name."""
    static = Path(__file__).parents[1] / "src/loman/ui/static"
    javascript = (static / "widget.js").read_text()
    css = (static / "widget.css").read_text()
    assert "function render({ model, el })" in javascript
    assert "const signal = controller.signal" in javascript
    assert "return cleanup" in javascript
    assert 'send("toggle_request", { id })' in javascript
    assert 'data-action="zoom-in"' in javascript
    assert "globalThis.crypto?.randomUUID" in javascript
    assert 'computeAll.disabled = !model.get("editable")' in javascript
    assert 'model.on("change:composite_ids", renderGraph)' not in javascript
    assert "height: clamp(320px, 45vh, 480px)" in css
    for trait in (
        "graph_svg",
        "node_states",
        "state_colors",
        "composite_ids",
        "selected_id",
        "detail",
        "status",
        "revision",
        "edit_request",
        "compute_request",
        "toggle_request",
    ):
        assert trait in javascript
