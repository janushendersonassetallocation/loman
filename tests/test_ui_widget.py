"""Python-side integration tests for the live computation widget.

This module tests:
- Automatic following of a computation through its subscription
- Selection identity across relayouts
- Editing, computing and expanding via the browser-facing traits
- Every status message the widget can report
- Rejection of stale derived traits and replayed requests
- Widget lifecycle: close, unsubscribe and garbage collection
"""

import gc
import re
import weakref
from pathlib import Path

import pandas as pd
import pytest

from loman import Computation, States
from loman.nodekey import to_nodekey
from loman.ui import ComputationWidget
from loman.ui.viewmodel import MIXED_STATE_LABEL
from loman.visualization import GraphView
from tests.conftest import create_example_block_computation

STATIC = Path(__file__).parents[1] / "src" / "loman" / "ui" / "static"


def make_widget() -> tuple[Computation, ComputationWidget]:
    """Create a small live graph and its widget."""
    comp = Computation()
    comp.add_node("price", value=10.0)
    comp.add_node("quantity", value=2)
    comp.add_node("value", lambda price, quantity: price * quantity)
    return comp, comp.widget(collapse_all=False)


def node_id(widget: ComputationWidget, name) -> str:
    """Return the rendered ID standing for a computation node."""
    return widget._view.node_index_map[to_nodekey(name)]


class TestFollowingTheComputation:
    """The widget tracks its computation without being told to."""

    def test_state_change_repaints_without_relayout(self):
        """Computation events update state traits while retaining the SVG."""
        comp, widget = make_widget()
        try:
            svg = widget.graph_svg
            value_id = node_id(widget, "value")
            assert widget.node_states[value_id] == States.COMPUTABLE.name

            comp.compute_all()

            assert widget.node_states[value_id] == States.UPTODATE.name
            assert widget.graph_svg == svg
            assert widget.revision == comp.revision
        finally:
            widget.close()

    def test_structural_change_rebuilds_the_layout(self):
        """A new node forces a fresh Graphviz layout."""
        comp, widget = make_widget()
        try:
            old_svg = widget.graph_svg

            comp.add_node("tax", lambda value: value * 0.2)

            assert widget.graph_svg != old_svg
            assert "tax" in widget.graph_svg
        finally:
            widget.close()

    def test_detail_refreshes_for_the_selected_node(self):
        """The open detail panel follows the value it is showing."""
        comp, widget = make_widget()
        try:
            widget.selected_id = node_id(widget, "price")
            assert widget.detail["value"]["value"] == 10.0

            comp.insert("price", 11.0)

            assert widget.detail["value"]["value"] == 11.0
        finally:
            widget.close()

    def test_non_state_colouring_relayouts_on_every_change(self):
        """Timing colours depend on values, so they cannot repaint in place."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        widget = comp.widget(collapse_all=False, colors="timing")
        try:
            assert widget.repaint_states is False

            comp.compute_all()

            assert widget.revision == comp.revision
        finally:
            widget.close()


class TestSelection:
    """Mapping rendered shapes back to real Loman names."""

    def test_selection_preserves_non_string_node_identity(self):
        """Opaque rendered IDs resolve back to real, possibly colliding names."""
        comp = Computation()
        comp.add_node(1, value="integer")
        comp.add_node("1", value="string")
        widget = comp.widget(collapse_all=False)
        try:
            widget.selected_id = node_id(widget, 1)
            assert widget.selected_name == 1
            assert widget.detail["value"]["value"] == "integer"

            widget.selected_id = node_id(widget, "1")
            assert widget.selected_name == "1"
            assert widget.detail["value"]["value"] == "string"
        finally:
            widget.close()

    def test_selected_is_an_alias_for_selected_name(self):
        """Both spellings are documented, so both must work."""
        _comp, widget = make_widget()
        try:
            widget.selected_id = node_id(widget, "price")
            assert widget.selected == widget.selected_name == "price"
        finally:
            widget.close()

    def test_nothing_selected_reports_nothing(self):
        """An empty selection has no name and no members."""
        _comp, widget = make_widget()
        try:
            assert widget.selected_name is None
            assert widget.selected_names == []
            assert widget.detail == {}
        finally:
            widget.close()

    def test_unknown_selection_id_reports_nothing(self):
        """A rendered ID the current layout does not contain resolves to nothing."""
        _comp, widget = make_widget()
        try:
            widget.selected_id = "n999"
            assert widget.selected_names == []
            assert widget.selected_name is None
        finally:
            widget.close()

    def test_collapsed_block_selection_reports_the_block_path(self):
        """A composite has many members but one meaningful name."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.selected_id = node_id(widget, "foo")

            assert widget.selected_name == "foo"
            assert set(widget.selected_names) == {"foo/a", "foo/b", "foo/c", "foo/d"}
            assert widget.detail["composite"] is True
        finally:
            widget.close()

    def test_selection_clears_when_relayout_reuses_rendered_id(self):
        """Selection cannot silently jump when Graphviz IDs are reused."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "expand"}
            widget.selected_id = node_id(widget, "foo/a")
            assert widget.selected_name == "foo/a"

            widget.toggle_request = {"collapse_all": True, "request_id": "collapse"}

            assert widget.selected_id == ""
            assert widget.selected_name is None
            assert widget.detail == {}
        finally:
            widget.close()


class TestEditing:
    """The scalar edit control and everything that can go wrong with it."""

    def test_edit_updates_the_computation(self):
        """Trait requests exercise the same round-trip as browser controls."""
        comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "price"),
                "value": {"kind": "scalar", "type": "float", "value": 12.5},
                "request_id": "e1",
            }

            assert comp.value("price") == 12.5
            assert comp.state("value") == States.COMPUTABLE
            assert widget.status == "Updated price"
        finally:
            widget.close()

    def test_read_only_widget_rejects_edits(self):
        """A crafted trait request cannot bypass the read-only UI."""
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(collapse_all=False, editable=False)
        try:
            widget.edit_request = {
                "id": node_id(widget, "a"),
                "value": {"kind": "scalar", "type": "int", "value": 99},
                "request_id": "e1",
            }

            assert comp.value("a") == 1
            assert widget.status == "Edit failed: this widget is read-only"
        finally:
            widget.close()

    def test_collapsed_block_cannot_be_edited(self):
        """A block has no single value to replace."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "foo"),
                "value": {"kind": "scalar", "type": "int", "value": 1},
                "request_id": "e1",
            }

            assert widget.status == "Edit failed: collapsed blocks cannot be edited"
        finally:
            widget.close()

    def test_calculated_node_cannot_be_edited(self):
        """Editing a calculated node would be overwritten by the next compute."""
        _comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "value"),
                "value": {"kind": "scalar", "type": "float", "value": 1.0},
                "request_id": "e1",
            }

            assert widget.status == "Edit failed: this node is not an editable scalar input"
        finally:
            widget.close()

    def test_malformed_edit_payload_reports_a_status(self):
        """A bad wire value surfaces in the UI, not only in the kernel log."""
        comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "price"),
                "value": {"kind": "scalar", "type": "float", "value": "not-a-number"},
                "request_id": "e1",
            }

            assert widget.status.startswith("Edit failed: ValueWireError")
            assert comp.value("price") == 10.0
        finally:
            widget.close()

    def test_unknown_node_in_edit_reports_a_status(self):
        """A stale rendered ID cannot raise out of a traitlets observer."""
        _comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": "n999",
                "value": {"kind": "scalar", "type": "int", "value": 1},
                "request_id": "e1",
            }

            assert widget.status.startswith("Edit failed: KeyError")
        finally:
            widget.close()

    def test_cleared_edit_request_is_ignored(self):
        """An empty payload must not be treated as a request."""
        _comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "price"),
                "value": {"kind": "scalar", "type": "float", "value": 12.5},
                "request_id": "e1",
            }
            settled = widget.status

            widget.edit_request = {}

            assert widget.status == settled
        finally:
            widget.close()


class TestComputing:
    """The compute controls."""

    def test_compute_target_computes_only_what_is_needed(self):
        """Computing a node runs it and its predecessors."""
        comp, widget = make_widget()
        try:
            widget.compute_request = {"id": node_id(widget, "value"), "request_id": "c1"}

            assert comp.value("value") == 20.0
            assert widget.status == "Computed value"
        finally:
            widget.close()

    def test_compute_all_computes_the_whole_graph(self):
        """The toolbar button drives compute_all()."""
        comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True, "request_id": "c1"}

            assert comp.value("value") == 20.0
            assert widget.status == "Computed all available nodes"
        finally:
            widget.close()

    def test_read_only_widget_rejects_compute_requests(self):
        """A crafted trait request cannot bypass the read-only UI."""
        comp = Computation()
        comp.add_node("input", value=2)
        comp.add_node("output", lambda input: input + 1)
        widget = comp.widget(collapse_all=False, editable=False)
        try:
            widget.compute_request = {"all": True, "request_id": "c1"}

            assert comp.state("output") == States.COMPUTABLE
            assert widget.status == "Compute failed: this widget is read-only"
        finally:
            widget.close()

    def test_compute_of_unknown_node_reports_a_status(self):
        """A stale rendered ID cannot raise out of a traitlets observer."""
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"id": "n999", "request_id": "c1"}

            assert widget.status.startswith("Compute failed: KeyError")
        finally:
            widget.close()

    def test_compute_failure_reports_a_status(self):
        """An uncomputable target is explained rather than silently ignored."""
        comp = Computation()
        comp.add_node("a")
        comp.add_node("b", lambda a: a + 1)
        widget = comp.widget(collapse_all=False)
        try:
            widget.compute_request = {"id": node_id(widget, "b"), "request_id": "c1"}

            assert widget.status.startswith("Compute failed:")
        finally:
            widget.close()

    def test_node_error_leaves_an_error_state_not_a_failed_status(self):
        """A node that raises becomes an ERROR state, which is the honest presentation."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: 1 / 0)
        widget = comp.widget(collapse_all=False)
        try:
            widget.compute_request = {"all": True, "request_id": "c1"}

            assert widget.status == "Computed all available nodes"
            assert widget.node_states[node_id(widget, "b")] == States.ERROR.name
            widget.selected_id = node_id(widget, "b")
            assert "ZeroDivisionError" in widget.detail["error"]
        finally:
            widget.close()


class TestExpandAndCollapse:
    """Drilling into blocks, which stays available even when read-only."""

    def test_expanding_a_block_reveals_its_members(self):
        """A composite-node request expands the existing GraphView hierarchy."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            foo_id = node_id(widget, "foo")
            assert foo_id in widget.composite_ids

            widget.toggle_request = {"id": foo_id, "request_id": "t1"}

            visible = {str(node) for node in widget._view.node_index_map}
            assert {"foo/a", "foo/b", "foo/c", "foo/d"}.issubset(visible)
            assert widget.status == "Opened foo"

            widget.toggle_request = {"collapse_all": True, "request_id": "t2"}

            assert to_nodekey("foo") in widget._view.composite_nodes
            assert widget.status == "Collapsed all blocks"
        finally:
            widget.close()

    def test_one_block_can_be_closed_without_collapsing_the_rest(self):
        """Opening is one click, so closing must not cost a full collapse.

        An open block is drawn as a Graphviz cluster rather than a node, so it
        is identified by path rather than by a rendered node ID.
        """
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "open-foo"}
            widget.toggle_request = {"id": node_id(widget, "bar"), "request_id": "open-bar"}
            assert widget.expanded_paths == ["bar", "foo"]

            widget.toggle_request = {"path": "foo", "collapse": True, "request_id": "close-foo"}

            assert widget.status == "Closed foo"
            assert widget.expanded_paths == ["bar"]
            assert to_nodekey("foo") in widget._view.composite_nodes
            assert to_nodekey("bar") not in widget._view.composite_nodes
        finally:
            widget.close()

    def test_closing_a_block_that_is_not_open_is_refused(self):
        """A stale close request from the browser cannot corrupt the view."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"path": "foo", "collapse": True, "request_id": "t1"}

            assert widget.status == "Expand/collapse failed: that block is not open"
            assert widget.status_severity == "error"
        finally:
            widget.close()

    def test_closing_a_block_also_closes_blocks_open_inside_it(self):
        """Expansions nested inside a closed block must not linger unseen."""
        comp = Computation()
        inner = Computation()
        inner.add_node("x", value=1)
        outer = Computation()
        outer.add_block("mid", inner, keep_values=True)
        comp.add_block("top", outer, keep_values=True)
        widget = comp.widget()
        try:
            widget._expanded = {to_nodekey("top"), to_nodekey("top/mid")}
            widget.refresh()
            assert widget.expanded_paths == ["top", "top/mid"]

            widget.toggle_request = {"path": "top", "collapse": True, "request_id": "t1"}

            assert widget.expanded_paths == []
        finally:
            widget.close()

    def test_expanded_paths_survive_a_browser_echo(self):
        """The open-block list is Python's, like the other derived traits."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "t1"}

            widget.expanded_paths = ["not-a-block"]

            assert widget.expanded_paths == ["foo"]
        finally:
            widget.close()

    def test_read_only_widget_still_allows_navigation(self):
        """Opening a block inspects the graph; it does not mutate it."""
        comp = create_example_block_computation()
        widget = comp.widget(editable=False)
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "t1"}

            assert widget.status == "Opened foo"
        finally:
            widget.close()

    def test_expanding_a_plain_node_is_refused(self):
        """Only collapsed blocks have anything to open."""
        _comp, widget = make_widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "price"), "request_id": "t1"}

            assert widget.status == "Expand/collapse failed: only collapsed blocks can be expanded"
        finally:
            widget.close()

    def test_unknown_node_in_toggle_reports_a_status(self):
        """A stale rendered ID cannot raise out of a traitlets observer."""
        _comp, widget = make_widget()
        try:
            widget.toggle_request = {"id": "n999", "request_id": "t1"}

            assert widget.status.startswith("Expand/collapse failed: KeyError")
        finally:
            widget.close()

    def test_oversized_block_is_not_opened(self):
        """One click must not hang the kernel rendering thousands of nodes."""
        comp = create_example_block_computation()
        widget = comp.widget(max_rendered_nodes=4)
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "t1"}

            assert "over the limit of 4" in widget.status
            assert to_nodekey("foo") in widget._view.composite_nodes
        finally:
            widget.close()

    def test_limit_does_not_cap_the_initial_view(self):
        """What the caller asked to draw is drawn, however large."""
        comp = Computation()
        for i in range(10):
            comp.add_node(f"n{i}", value=i)
        widget = comp.widget(collapse_all=False, max_rendered_nodes=2)
        try:
            assert len(widget._view.node_index_map) == 10
        finally:
            widget.close()

    def test_cleared_toggle_request_is_ignored(self):
        """An empty payload must not be treated as a request."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "t1"}
            settled = widget.status

            widget.toggle_request = {}

            assert widget.status == settled
        finally:
            widget.close()


class TestRenderFailure:
    """Graphviz is a subprocess, and subprocesses fail."""

    def test_render_failure_reports_a_readable_status(self, mocker):
        """A missing dot binary must not produce an opaque traceback."""
        comp, widget = make_widget()
        try:
            good_svg = widget.graph_svg
            mocker.patch.object(GraphView, "svg", side_effect=OSError("dot not found"))

            comp.add_node("tax", lambda value: value * 0.2)

            assert widget.status.startswith("Unable to render graph: OSError")
            assert widget.graph_svg == good_svg
        finally:
            widget.close()

    def test_failed_refresh_reports_failure_to_its_caller(self, mocker):
        """Expanding a block must not claim success when rendering failed."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            foo_id = node_id(widget, "foo")
            mocker.patch.object(GraphView, "svg", side_effect=OSError("dot not found"))

            widget.toggle_request = {"id": foo_id, "request_id": "t1"}

            assert widget.status.startswith("Unable to render graph: OSError")
            assert widget.refresh() is False
        finally:
            widget.close()

    def test_empty_svg_is_tolerated(self, mocker):
        """GraphView.svg() returns None for an unrendered view."""
        _comp, widget = make_widget()
        try:
            mocker.patch.object(GraphView, "svg", return_value=None)

            assert widget.refresh() is True
            assert widget.graph_svg == ""
        finally:
            widget.close()


class TestWidgetWithoutAGraph:
    """A widget whose very first render failed, as with no dot binary installed.

    It has no view at all, so every interaction has to degrade to a status
    message rather than raising out of a traitlets observer.
    """

    @pytest.fixture
    def broken(self, mocker):
        """Build a widget whose initial Graphviz render fails."""
        mocker.patch.object(GraphView, "svg", side_effect=OSError("dot not found"))
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(collapse_all=False)
        yield comp, widget
        widget.close()

    def test_construction_reports_the_failure(self, broken):
        """The widget exists and explains itself rather than raising."""
        _comp, widget = broken
        assert widget.graph_svg == ""
        assert widget.status.startswith("Unable to render graph: OSError")

    def test_selection_reports_nothing(self, broken):
        """With no layout there are no rendered nodes to name."""
        _comp, widget = broken
        widget.selected_id = "n0"
        assert widget.selected_names == []
        assert widget.detail == {}

    def test_edit_reports_a_status(self, broken):
        """An edit against a graph that was never drawn is refused."""
        _comp, widget = broken
        widget.edit_request = {"id": "n0", "value": {"kind": "scalar", "type": "int", "value": 2}}
        assert widget.status == "Edit failed: the graph is not rendered"

    def test_compute_of_a_target_reports_a_status(self, broken):
        """Computing a specific node needs a layout to resolve its ID."""
        _comp, widget = broken
        widget.compute_request = {"id": "n0"}
        assert widget.status == "Compute failed: the graph is not rendered"

    def test_compute_all_still_works(self, broken):
        """Computing everything needs no node identity, so it is still allowed."""
        comp, widget = broken
        comp.add_node("b", lambda a: a + 1)

        widget.compute_request = {"all": True}

        assert comp.value("b") == 2

    def test_expanding_reports_a_status(self, broken):
        """There are no blocks to open without a layout."""
        _comp, widget = broken
        widget.toggle_request = {"id": "n0"}
        assert widget.status == "Expand/collapse failed: the graph is not rendered"

    def test_browser_echo_check_is_skipped(self, broken):
        """With no canonical view there is nothing to compare an echo against."""
        _comp, widget = broken
        widget.node_states = {"stale": "ERROR"}
        assert widget.node_states == {"stale": "ERROR"}


class TestBrowserEchoes:
    """The browser model is not the source of truth."""

    def test_stale_derived_traits_are_put_back(self):
        """Reconnect echoes cannot corrupt the canonical Python-side view model."""
        comp, widget = make_widget()
        try:
            widget.selected_id = node_id(widget, "price")
            expected_detail = widget.detail
            expected_svg = widget.graph_svg
            expected_composites = widget.composite_ids
            expected_states = widget.node_states

            widget.composite_ids = ["stale"]
            widget.detail = {"id": "n0", "name": "wrong node"}
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

    def test_status_survives_a_request_carrying_a_stale_status(self):
        """The browser's cached status must not revert the one Python just set.

        This is the real front-end path, and it is not the same as assigning
        traits one at a time. ipywidgets applies an incoming message inside a
        single ``hold_trait_notifications`` block, so the browser's stale copy of
        ``status`` lands *after* the observer that handled the request has
        already set a new one.
        """
        comp, widget = make_widget()
        try:
            widget.set_state({"compute_request": {"all": True, "request_id": "c1"}, "status": "stale from browser"})

            assert widget.status == "Computed all available nodes"
            assert comp.value("value") == 20.0
        finally:
            widget.close()

    def test_stale_derived_traits_in_a_request_message_are_put_back(self):
        """The same applies to every trait Python owns, not just status."""
        comp, widget = make_widget()
        try:
            expected_svg = widget.graph_svg

            widget.set_state(
                {
                    "compute_request": {"all": True, "request_id": "c1"},
                    "graph_svg": "<svg>stale</svg>",
                    "node_states": {"stale": "ERROR"},
                    "revision": -1,
                }
            )

            assert widget.graph_svg == expected_svg
            assert widget.node_states != {"stale": "ERROR"}
            assert widget.revision == comp.revision
        finally:
            widget.close()

    def test_replayed_edit_request_is_ignored(self):
        """A recreated front-end model must not re-apply an edit it still holds."""
        comp, widget = make_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "price"),
                "value": {"kind": "scalar", "type": "float", "value": 12.5},
                "request_id": "same-nonce",
            }
            assert comp.value("price") == 12.5

            widget.edit_request = {
                "id": node_id(widget, "price"),
                "value": {"kind": "scalar", "type": "float", "value": 99.0},
                "request_id": "same-nonce",
            }

            assert comp.value("price") == 12.5
        finally:
            widget.close()

    def test_replayed_compute_request_is_ignored(self):
        """Replaying a compute must not silently re-run the graph."""
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True, "request_id": "same-nonce"}
            assert widget.status == "Computed all available nodes"

            widget.compute_request = {"id": node_id(widget, "value"), "request_id": "same-nonce"}

            assert widget.status == "Computed all available nodes"
        finally:
            widget.close()

    def test_distinct_requests_are_both_honoured(self):
        """The replay guard must not swallow a genuine repeat of an action."""
        comp, widget = make_widget()
        try:
            widget.compute_request = {"id": node_id(widget, "value"), "request_id": "c1"}
            assert widget.status == "Computed value"
            comp.insert("price", 11.0)

            widget.compute_request = {"all": True, "request_id": "c2"}

            assert widget.status == "Computed all available nodes"
            assert comp.value("value") == 22.0
        finally:
            widget.close()

    def test_requests_without_a_nonce_still_work(self):
        """Python-side callers and tests need not invent request IDs."""
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True}
            assert widget.status == "Computed all available nodes"
        finally:
            widget.close()


class TestLifecycle:
    """Subscription lifetime, which decides whether a notebook leaks widgets."""

    def test_close_stops_the_widget_following(self):
        """A closed widget unsubscribes and stops updating."""
        comp, widget = make_widget()
        widget.close()
        closed_revision = widget.revision

        comp.insert("price", 12.0)

        assert widget.revision == closed_revision

    def test_close_is_idempotent(self):
        """Closing twice must not raise."""
        _comp, widget = make_widget()
        widget.close()
        widget.close()

    def test_computation_holds_the_widget_weakly(self):
        """Subscribing must not make the computation an owner of the widget.

        ipywidgets registers every open widget in a process-wide table of its
        own, so ``close()`` is what ultimately releases one. What Loman controls
        is not adding a second, permanent retention path on top of that, which
        is why the subscription stores a weak reference.
        """
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(collapse_all=False)
        try:
            subscription = comp._subscriptions[0]

            assert subscription.resolve() == widget._on_computation_event
            assert isinstance(subscription._weak, weakref.WeakMethod)
            assert subscription._strong is None
        finally:
            widget.close()

    def test_close_removes_the_subscription(self):
        """Nothing is left behind on the computation after close."""
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(collapse_all=False)
        assert len(comp._subscriptions) == 1

        widget.close()
        gc.collect()

        assert comp._subscriptions == []


class TestAssetContract:
    """The hand-written frontend and the Python traits must stay in step."""

    def test_javascript_references_every_synced_trait(self):
        """A trait renamed on one side only is the realistic failure here."""
        javascript = (STATIC / "widget.js").read_text()
        synced = [
            name
            for name, trait in ComputationWidget.class_own_traits().items()
            if trait.metadata.get("sync") and not name.startswith("_")
        ]
        assert synced, "expected the widget to declare synced traits"
        for name in synced:
            assert name in javascript, f"trait {name} is never referenced by widget.js"

    def test_javascript_only_writes_traits_python_owns_are_left_alone(self):
        """The frontend must not set a trait Python treats as canonical."""
        javascript = (STATIC / "widget.js").read_text()
        for own in ("graph_svg", "node_states", "composite_ids", "detail", "status", "revision"):
            assert f'model.set("{own}"' not in javascript

    def test_every_state_has_a_colour_the_browser_can_use(self):
        """A new state can never render unstyled."""
        _comp, widget = make_widget()
        try:
            for state in States:
                assert state.name in widget.state_colors
            assert MIXED_STATE_LABEL in widget.state_colors
        finally:
            widget.close()

    def test_module_uses_the_portable_anywidget_entry_point(self):
        """render({model, el}) returning a cleanup callback is valid everywhere."""
        javascript = (STATIC / "widget.js").read_text()
        assert "function render({ model, el })" in javascript
        assert "return cleanup" in javascript
        assert "export default { render }" in javascript

    def test_stylesheet_supports_both_colour_schemes(self):
        """Marimo and JupyterLab both have dark themes."""
        css = (STATIC / "widget.css").read_text()
        assert "@media (prefers-color-scheme: dark)" in css

    def test_the_graph_paper_surface_is_never_used_behind_text(self):
        """``--loman-canvas`` is white in both themes, so ink vanishes on it.

        It stays white because Graphviz paints a white background and black
        labels into the SVG itself. Any other element that borrows it as a
        background and then sets ``--loman-ink`` renders white on white in dark
        mode, which is exactly what happened to the edit fields.
        """
        css = (STATIC / "widget.css").read_text()
        blocks = re.findall(r"([^{}]*)\{([^{}]*)\}", css)
        offenders = [
            selector.strip()
            for selector, body in blocks
            if "background: var(--loman-canvas)" in body and "color:" in body
        ]
        assert offenders == [], f"these set ink on the graph paper surface: {offenders}"

    def test_graph_is_not_scaled_to_fit_the_pane(self):
        """Sizing the SVG in percentages is what made labels shrink.

        Graphviz emits a fixed viewBox, so a percentage width makes the browser
        scale the whole graph down as soon as a block is opened. Measured at
        5.2 px labels on a 32-node graph before this was changed.
        """
        css = (STATIC / "widget.css").read_text()
        javascript = (STATIC / "widget.js").read_text()
        stage_svg = css.split(".loman-stage svg {")[1].split("}")[0]
        assert "%" not in stage_svg
        assert "naturalSize.w * zoom" in javascript

    def test_hover_and_selection_are_styled_differently(self):
        """Identical styling made it impossible to see what was selected."""
        css = (STATIC / "widget.css").read_text()
        hover = css.split("g.node:hover > :not(title, text) {")[1].split("}")[0]
        selected = css.split("g.node.loman-selected > :not(title, text) {")[1].split("}")[0]
        assert hover.strip() != selected.strip()

    def test_frontend_declares_a_busy_state(self):
        """A synchronous compute must not look like nothing happened."""
        javascript = (STATIC / "widget.js").read_text()
        css = (STATIC / "widget.css").read_text()
        assert "setBusy(true" in javascript
        assert 'data-severity="busy"' in css or '[data-severity="busy"]' in css

    def test_legend_names_every_state_on_screen(self):
        """State is colour-only in the graph, and the colours are not CVD-safe."""
        javascript = (STATIC / "widget.js").read_text()
        assert "renderLegend" in javascript
        assert "label.textContent = state" in javascript


@pytest.mark.parametrize("colors", ["state", "timing"])
def test_widget_builds_for_each_colouring(colors):
    """Both colour modes construct and render a graph."""
    comp = Computation()
    comp.add_node("a", value=1)
    comp.add_node("b", lambda a: a + 1)
    comp.compute_all()
    widget = comp.widget(collapse_all=False, colors=colors)
    try:
        assert "<svg" in widget.graph_svg
    finally:
        widget.close()


class TestCellEditing:
    """Editing one cell of a tabular input node."""

    @staticmethod
    def make_frame_widget():
        """Create a computation whose input node holds a DataFrame."""
        comp = Computation()
        comp.add_node("book", value=pd.DataFrame({"ccy": ["USD", "GBP"], "notional": [1.0, 2.0]}))
        comp.add_node("total", lambda book: book["notional"].sum())
        comp.compute_all()
        return comp, comp.widget(collapse_all=False)

    def test_cell_edit_updates_the_frame_and_downstream(self):
        """A cell edit is an ordinary insert, so the graph reacts as usual."""
        comp, widget = self.make_frame_widget()
        try:
            original = comp.value("book")
            widget.edit_request = {
                "id": node_id(widget, "book"),
                "cell": {"row": 1, "column": 1},
                "value": {"kind": "scalar", "type": "float", "value": 9.0},
                "request_id": "c1",
            }

            assert comp.value("book")["notional"].tolist() == [1.0, 9.0]
            assert original["notional"].tolist() == [1.0, 2.0], "the previous value must not be mutated"
            assert comp.state("total") == States.COMPUTABLE
            assert widget.status == "Updated book [1, 1]"
        finally:
            widget.close()

    def test_cell_edit_is_refused_on_a_calculated_node(self):
        """The next compute would silently discard it."""
        _comp, widget = self.make_frame_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "total"),
                "cell": {"row": 0, "column": 0},
                "value": {"kind": "scalar", "type": "float", "value": 1.0},
                "request_id": "c1",
            }

            assert widget.status == "Edit failed: this node's cells are not editable"
            assert widget.status_severity == "error"
        finally:
            widget.close()

    def test_cell_edit_is_refused_on_a_read_only_widget(self):
        """A crafted request cannot bypass editable=False."""
        comp = Computation()
        comp.add_node("book", value=pd.DataFrame({"x": [1]}))
        widget = comp.widget(collapse_all=False, editable=False)
        try:
            widget.edit_request = {
                "id": node_id(widget, "book"),
                "cell": {"row": 0, "column": 0},
                "value": {"kind": "scalar", "type": "int", "value": 5},
                "request_id": "c1",
            }

            assert widget.status == "Edit failed: this widget is read-only"
            assert comp.value("book")["x"].tolist() == [1]
        finally:
            widget.close()

    def test_out_of_range_cell_reports_a_status(self):
        """A stale window from the browser cannot write outside the frame."""
        _comp, widget = self.make_frame_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "book"),
                "cell": {"row": 99, "column": 0},
                "value": {"kind": "scalar", "type": "str", "value": "x"},
                "request_id": "c1",
            }

            assert widget.status.startswith("Edit failed: ValueWireError")
            assert "outside" in widget.status
        finally:
            widget.close()

    def test_wrong_type_for_the_column_reports_a_status(self):
        """Editing must not silently change a column's dtype."""
        comp, widget = self.make_frame_widget()
        try:
            widget.edit_request = {
                "id": node_id(widget, "book"),
                "cell": {"row": 0, "column": 1},
                "value": {"kind": "scalar", "type": "str", "value": "lots"},
                "request_id": "c1",
            }

            assert widget.status.startswith("Edit failed: ValueWireError")
            assert comp.value("book")["notional"].tolist() == [1.0, 2.0]
        finally:
            widget.close()

    def test_frontend_sends_cell_edits(self):
        """The table renderer and the Python handler must agree on the payload."""
        javascript = (STATIC / "widget.js").read_text()
        assert "cell: { row, column }" in javascript
        assert "openCellEditor" in javascript

    def test_read_only_state_is_not_undone_by_clearing_busy(self):
        """Busy and editable both disable controls, so they must be applied together.

        Clearing the busy state used to blanket-enable the toolbar, which
        re-enabled Compute all on a read-only widget. Python still refused the
        request, but the button looked live.
        """
        javascript = (STATIC / "widget.js").read_text()
        body = javascript.split("const applyEnabledState = () => {")[1].split("};")[0]
        assert 'model.get("editable")' in body
        assert "applyEnabledState();" in javascript.split("const setBusy")[1]
        # Nothing may flip the toolbar's disabled state outside that one helper.
        assert javascript.count('buttons("compute-all").disabled') == 1


class TestAcknowledgement:
    """Every browser request is answered, including the ones that change nothing.

    The front end shows an optimistic busy state and waits for Python. A request
    whose outcome is "nothing changed" fires no other trait, so without an
    explicit acknowledgement the widget spins for ever -- which is what happened
    when Collapse all was pressed on an already-collapsed graph.
    """

    def test_a_request_that_changes_nothing_still_acknowledges(self):
        """The reported bug: collapse all, twice, on an already-collapsed graph."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"collapse_all": True, "request_id": "t1"}
            settled = (widget.graph_svg, widget.status, widget.ack)

            widget.toggle_request = {"collapse_all": True, "request_id": "t2"}

            assert widget.graph_svg == settled[0], "nothing about the picture changed"
            assert widget.status == settled[1], "nor the message"
            assert widget.ack > settled[2], "but the request was still acknowledged"
        finally:
            widget.close()

    def test_recomputing_an_up_to_date_graph_acknowledges(self):
        """compute_all on a settled graph publishes no event either."""
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True, "request_id": "c1"}
            settled = widget.ack

            widget.compute_request = {"all": True, "request_id": "c2"}

            assert widget.ack > settled
        finally:
            widget.close()

    def test_failed_requests_acknowledge_too(self):
        """A refusal is an outcome; the browser must stop waiting for it."""
        _comp, widget = make_widget()
        try:
            before = widget.ack

            widget.edit_request = {"id": "n999", "value": {"kind": "scalar", "type": "int", "value": 1}}

            assert widget.status.startswith("Edit failed")
            assert widget.ack > before
        finally:
            widget.close()

    def test_replayed_requests_acknowledge_too(self):
        """A replay is dropped, but the browser still has to stop waiting.

        A replay carries a nonce already seen with a different payload, so the
        observer runs and declines. An identical payload does not reach the
        observer at all, because traitlets does not fire when nothing changed --
        and in that case nothing was waiting either.
        """
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True, "request_id": "same"}
            after_first = widget.ack

            widget.compute_request = {"id": node_id(widget, "value"), "request_id": "same"}

            assert widget.ack > after_first
            assert widget.status == "Computed all available nodes", "the replay was declined"
        finally:
            widget.close()

    def test_acknowledgement_survives_a_browser_echo(self):
        """A stale ack from the browser must not un-answer a request."""
        _comp, widget = make_widget()
        try:
            widget.compute_request = {"all": True, "request_id": "c1"}
            expected = widget.ack

            widget.ack = 0

            assert widget.ack == expected
        finally:
            widget.close()

    def test_frontend_releases_the_busy_state_on_acknowledgement(self):
        """The Python counter is only useful if the browser listens for it."""
        javascript = (STATIC / "widget.js").read_text()
        assert 'model.on("change:ack", renderStatus)' in javascript
        assert "setBusy(false)" in javascript.split("const renderStatus")[1]


class TestLayoutDirection:
    """The graph reads left to right by default, and the toolbar toggles it."""

    def test_defaults_to_left_to_right(self):
        """A plain widget lays the graph out with rankdir=LR."""
        _comp, widget = make_widget()
        try:
            assert widget.rankdir == "LR"
            assert "rankdir=LR" in widget._view.viz_dot.to_string()
        finally:
            widget.close()

    def test_explicit_rankdir_argument_is_honoured(self):
        """comp.widget(rankdir=...) starts in the direction asked for."""
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(rankdir="TB")
        try:
            assert widget.rankdir == "TB"
            assert "rankdir=TB" in widget._view.viz_dot.to_string()
        finally:
            widget.close()

    def test_rankdir_in_graph_attr_takes_precedence(self):
        """A direction set the old way, through graph_attr, still wins."""
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(graph_attr={"rankdir": "BT"})
        try:
            assert widget.rankdir == "BT"
            assert "rankdir=BT" in widget._view.viz_dot.to_string()
        finally:
            widget.close()

    def test_toggle_relayouts_in_the_new_direction(self):
        """A layout request re-runs Graphviz with the new rankdir."""
        _comp, widget = make_widget()
        try:
            svg = widget.graph_svg

            widget.layout_request = {"rankdir": "TB", "request_id": "l1"}

            assert widget.rankdir == "TB"
            assert widget.status == "Layout direction TB"
            assert "rankdir=TB" in widget._view.viz_dot.to_string()
            assert widget.graph_svg != svg
        finally:
            widget.close()

    def test_an_invalid_direction_is_refused(self):
        """A nonsense rankdir leaves the picture and the trait alone."""
        _comp, widget = make_widget()
        try:
            widget.layout_request = {"rankdir": "sideways", "request_id": "l1"}

            assert widget.status.startswith("Layout failed")
            assert widget.status_severity == "error"
            assert widget.rankdir == "LR"
        finally:
            widget.close()

    def test_rankdir_survives_a_browser_echo(self):
        """The direction is Python's own, like the other derived traits."""
        _comp, widget = make_widget()
        try:
            widget.layout_request = {"rankdir": "TB", "request_id": "l1"}

            widget.rankdir = "LR"

            assert widget.rankdir == "TB"
        finally:
            widget.close()

    def test_the_toolbar_wires_the_toggle(self):
        """The button exists and drives a layout_request."""
        javascript = (STATIC / "widget.js").read_text()
        assert 'data-action="layout"' in javascript
        assert '"layout_request"' in javascript

    def test_an_empty_direction_is_refused(self):
        """The message has to read sensibly when nothing was supplied."""
        _comp, widget = make_widget()
        try:
            widget.layout_request = {"rankdir": "", "request_id": "l1"}

            assert widget.status == "Layout failed: (empty) is not a valid rankdir"
        finally:
            widget.close()

    def test_a_failed_relayout_keeps_the_direction_on_screen(self, mocker):
        """Rankdir must describe the picture, not the request that failed."""
        _comp, widget = make_widget()
        try:
            mocker.patch.object(GraphView, "svg", side_effect=OSError("dot not found"))

            widget.layout_request = {"rankdir": "TB", "request_id": "l1"}

            assert widget.rankdir == "LR", "the picture is still left to right"
            assert widget.status.startswith("Unable to render graph")
        finally:
            widget.close()


class TestFocus:
    """Re-rooting the view onto one block, which nesting makes worthwhile."""

    def test_focusing_a_block_reroots_the_view(self):
        """Focusing shows only the chosen block's own contents."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            assert widget.status == "Focused on foo"
            visible = {str(node) for node in widget._view.node_index_map}
            assert {"a", "b", "c", "d"}.issubset(visible)
            assert "bar" not in visible
            assert widget.focus_trail == [
                {"label": "Reset", "path": ""},
                {"label": "foo", "path": "foo"},
            ]
        finally:
            widget.close()

    def test_focus_drills_into_nested_blocks(self):
        """Once focused on a block, its own nested blocks become focusable."""
        comp = Computation()
        inner = Computation()
        inner.add_node("x", value=1)
        outer = Computation()
        outer.add_block("mid", inner, keep_values=True)
        comp.add_block("top", outer, keep_values=True)
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "top"), "request_id": "f1"}
            widget.focus_request = {"id": node_id(widget, "mid"), "request_id": "f2"}

            assert widget.status == "Focused on top/mid"
            assert widget.focus_trail == [
                {"label": "Reset", "path": ""},
                {"label": "top", "path": "top"},
                {"label": "mid", "path": "top/mid"},
            ]
            assert "x" in {str(node) for node in widget._view.node_index_map}
        finally:
            widget.close()

    def test_a_breadcrumb_step_climbs_back_out(self):
        """An empty path returns focus to the whole graph."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            widget.focus_request = {"path": "", "request_id": "f2"}

            assert widget.status == "Showing the whole graph"
            assert to_nodekey("foo") in widget._view.composite_nodes
            assert widget.focus_trail == [{"label": "Reset", "path": ""}]
        finally:
            widget.close()

    def test_only_blocks_can_be_focused(self):
        """Focusing a plain node is refused rather than re-rooting on nothing."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "output"), "request_id": "f1"}

            assert widget.status.startswith("Focus failed")
            assert widget.status_severity == "error"
        finally:
            widget.close()

    def test_focusing_drops_expansions_outside_the_new_root(self):
        """A block opened elsewhere is no longer reachable once focus moves."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "bar"), "request_id": "t1"}
            assert widget.expanded_paths == ["bar"]

            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            assert widget.expanded_paths == []
        finally:
            widget.close()

    def test_focus_trail_survives_a_browser_echo(self):
        """The breadcrumb is Python's own, like the other derived traits."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            widget.focus_trail = [{"label": "stale", "path": "x"}]

            assert widget.focus_trail == [
                {"label": "Reset", "path": ""},
                {"label": "foo", "path": "foo"},
            ]
        finally:
            widget.close()

    def test_the_inspector_and_breadcrumb_are_wired(self):
        """The Focus control and the breadcrumb both drive a focus_request."""
        javascript = (STATIC / "widget.js").read_text()
        assert '"focus_request"' in javascript
        assert "loman-breadcrumb" in javascript
        assert "renderBreadcrumb" in javascript


class TestFullView:
    """Handing a node to the notebook to render in full.

    The widget only ever holds a window onto a large value, and it cannot call
    the host's renderers without depending on that host. So it publishes which
    node was asked for and the notebook renders it.
    """

    @staticmethod
    def frame_widget():
        """Create a computation whose input node holds a DataFrame."""
        comp = Computation()
        comp.add_node("book", value=pd.DataFrame({"n": range(200)}))
        return comp, comp.widget(collapse_all=False)

    def test_requesting_a_node_publishes_its_name(self):
        """The notebook reads widget.full_view to know what to render."""
        _comp, widget = self.frame_widget()
        try:
            widget.full_view_request = {"id": node_id(widget, "book"), "request_id": "f1"}

            assert widget.full_view == "book"
            assert widget.status == "Showing book in full below"
        finally:
            widget.close()

    def test_the_full_value_is_reachable_without_indexing(self):
        """The convenience property saves guarding the empty case."""
        comp, widget = self.frame_widget()
        try:
            assert widget.full_view_value is None

            widget.full_view_request = {"id": node_id(widget, "book"), "request_id": "f1"}

            assert widget.full_view_value.equals(comp.value("book"))
        finally:
            widget.close()

    def test_an_empty_request_closes_the_full_view(self):
        """The button toggles, so it has to be able to clear the selection."""
        _comp, widget = self.frame_widget()
        try:
            widget.full_view_request = {"id": node_id(widget, "book"), "request_id": "f1"}

            widget.full_view_request = {"request_id": "f2"}

            assert widget.full_view == ""
            assert widget.status == "Closed the full view"
        finally:
            widget.close()

    def test_a_collapsed_block_has_no_single_value(self):
        """A block stands for many nodes, so there is nothing to render."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.full_view_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            assert widget.status == "Show full failed: a collapsed block has no single value"
            assert widget.full_view == ""
        finally:
            widget.close()

    def test_an_unknown_node_reports_a_status(self):
        """A stale rendered ID cannot raise out of a traitlets observer."""
        _comp, widget = self.frame_widget()
        try:
            widget.full_view_request = {"id": "n999", "request_id": "f1"}

            assert widget.status.startswith("Show full failed: KeyError")
        finally:
            widget.close()

    def test_the_published_name_survives_a_browser_echo(self):
        """full_view is Python's, like the other derived traits."""
        _comp, widget = self.frame_widget()
        try:
            widget.full_view_request = {"id": node_id(widget, "book"), "request_id": "f1"}

            widget.full_view = "something else"

            assert widget.full_view == "book"
        finally:
            widget.close()

    def test_the_frontend_offers_the_button_for_windowed_values(self):
        """Only values the panel cannot show whole are worth opening elsewhere."""
        javascript = (STATIC / "widget.js").read_text()
        assert "buildShowFull" in javascript
        assert 'send(\n      "full_view_request"' in javascript or '"full_view_request"' in javascript
        assert 'value.kind === "table" || value.kind === "tree" || value.kind === "repr"' in javascript


class TestHostIntegration:
    """The widget wears the host page's backdrop rather than its own."""

    def test_the_frontend_samples_the_host_background(self):
        """A widget that looks pasted on does not read as part of the page."""
        javascript = (STATIC / "widget.js").read_text()
        assert "hostBackground" in javascript
        assert "--loman-backdrop" in javascript

    def test_the_host_palette_is_adopted_only_once_it_is_confirmed(self):
        """--background is a common name; another design system may own it.

        Adopting the host's tokens blindly would repaint the widget in whatever
        those names happen to mean there, so they are used only when the host's
        declared background matches the backdrop it actually paints.
        """
        javascript = (STATIC / "widget.js").read_text()
        assert "resolveHostToken" in javascript
        assert "sameColour" in javascript
        assert "dataset.hostTokens" in javascript
        # The Graphviz canvas is not the host's to theme: it paints a white
        # background and black labels into the SVG itself.
        mapping = javascript.split("const HOST_TOKEN_MAP = [")[1].split("];")[0]
        assert "--loman-canvas" not in mapping
        # Every adopted token keeps a fallback, for a host defining only some of
        # them. --background is the exception: it is the one that was verified.
        for line in mapping.strip().splitlines():
            if "var(" not in line or 'var(--background)"' in line:
                continue
            assert "," in line.split("var(", 1)[1], f"no fallback for: {line.strip()}"

    def test_the_sampled_theme_beats_the_os_preference(self):
        """A notebook's own light/dark toggle never touches prefers-color-scheme.

        The stamped attribute has to outrank the media query, or a dark notebook
        on a light desktop would render the widget light.
        """
        javascript = (STATIC / "widget.js").read_text()
        css = (STATIC / "widget.css").read_text()
        assert "relativeLuminance" in javascript
        assert "dataset.hostTheme" in javascript
        for theme in ("light", "dark"):
            assert f'.loman-widget[data-host-theme="{theme}"]' in css
        # Attribute selectors outrank the bare class used inside the media query.
        assert css.index("@media (prefers-color-scheme: dark)") < css.index('[data-host-theme="dark"]')


class TestUnrenderedWidgetRequests:
    """Every request path degrades when the first Graphviz render failed."""

    @pytest.fixture
    def broken(self, mocker):
        """Build a widget whose initial render fails, then a block computation."""
        mocker.patch.object(GraphView, "svg", side_effect=OSError("dot not found"))
        comp = create_example_block_computation()
        widget = comp.widget()
        yield comp, widget
        widget.close()

    def test_focus_reports_a_status(self, broken):
        """There is no rendered block to focus on."""
        _comp, widget = broken
        widget.focus_request = {"id": "n0", "request_id": "f1"}
        assert widget.status == "Focus failed: the graph is not rendered"

    def test_show_full_reports_a_status(self, broken):
        """There is no rendered node to resolve."""
        _comp, widget = broken
        widget.full_view_request = {"id": "n0", "request_id": "f1"}
        assert widget.status == "Show full failed: the graph is not rendered"


class TestRequestsThatAreIgnored:
    """An empty payload is the trait's default, not a click."""

    @pytest.mark.parametrize(
        "trait",
        ["edit_request", "compute_request", "toggle_request", "layout_request", "focus_request", "full_view_request"],
    )
    def test_a_cleared_request_is_ignored_but_acknowledged(self, trait):
        """Clearing a request must not act, but must still release the spinner."""
        _comp, widget = make_widget()
        try:
            setattr(widget, trait, {"request_id": "seed", "id": node_id(widget, "price")})
            settled_status = widget.status
            settled_ack = widget.ack

            setattr(widget, trait, {})

            assert widget.status == settled_status
            assert widget.ack > settled_ack
        finally:
            widget.close()


class TestFocusScope:
    """A focused widget cannot be steered outside its own root."""

    def test_focusing_outside_the_widget_root_is_refused(self):
        """A section-scoped widget must stay inside its section."""
        comp = create_example_block_computation()
        widget = comp.widget(root=to_nodekey("foo"))
        try:
            widget.focus_request = {"path": "bar", "request_id": "f1"}

            assert widget.status.startswith("Focus failed: ValueError")
            assert "not within this widget's root" in widget.status
        finally:
            widget.close()

    def test_focusing_the_widget_root_itself_is_allowed(self):
        """Climbing the breadcrumb back to the root is not going outside it."""
        comp = create_example_block_computation()
        widget = comp.widget(root=to_nodekey("foo"))
        try:
            widget.focus_request = {"path": "foo", "request_id": "f1"}

            assert widget.status == "Focused on foo"
        finally:
            widget.close()


class TestFitOnRender:
    """Auto-fitting, for when the shape matters more than the labels."""

    def test_it_is_off_by_default(self):
        """A graph fitted into a notebook pane is unreadable past a few nodes."""
        _comp, widget = make_widget()
        try:
            assert widget.fit_on_render is False
        finally:
            widget.close()

    def test_the_option_reaches_the_front_end(self):
        """It is the browser that measures the pane, so it has to be synced."""
        comp = Computation()
        comp.add_node("a", value=1)
        widget = comp.widget(fit_on_render=True)
        try:
            assert widget.fit_on_render is True
            assert widget.trait_metadata("fit_on_render", "sync") is True
        finally:
            widget.close()

    def test_the_front_end_only_ever_shrinks(self):
        """Blowing a small graph up to fill the pane is not what fit means."""
        javascript = (STATIC / "widget.js").read_text()
        body = javascript.split("const fitIfRequested = () => {")[1].split("};")[0]
        assert 'model.get("fit_on_render")' in body
        assert "if (scale < 1) fitToPane();" in body


class TestFrameClickFocus:
    """A block's frame isolates it; its interior opens it."""

    def test_the_front_end_distinguishes_frame_from_interior(self):
        """One shape opens the block, the ring around it re-roots the view."""
        javascript = (STATIC / "widget.js").read_text()
        assert "const onFrame" in javascript
        activate = javascript.split("const activate = (id, composite, event) => {")[1].split("\n  };")[0]
        assert '"focus_request"' in activate
        assert '"toggle_request"' in activate

    def test_focusing_by_rendered_id_isolates_the_block(self):
        """That request is what a frame click sends."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}

            assert widget.status == "Focused on foo"
            assert [entry["label"] for entry in widget.focus_trail] == ["Reset", "foo"]
            visible = {str(node) for node in widget._view.node_index_map}
            assert "bar" not in visible, "the rest of the graph is out of view"
        finally:
            widget.close()

    def test_the_breadcrumb_root_reads_as_a_reset(self):
        """It is an action, not a place, so it is not labelled after the graph."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.focus_request = {"id": node_id(widget, "foo"), "request_id": "f1"}
            assert widget.focus_trail[0]["label"] == "Reset"

            widget.focus_request = {"path": "", "request_id": "f2"}

            assert widget.status == "Showing the whole graph"
            assert widget.focus_trail == [{"label": "Reset", "path": ""}]
        finally:
            widget.close()

    def test_focusing_a_plain_node_is_refused(self):
        """Only blocks contain anything to isolate."""
        _comp, widget = make_widget()
        try:
            widget.focus_request = {"id": node_id(widget, "price"), "request_id": "f1"}

            assert widget.status.startswith("Focus failed")
            assert "only blocks can be focused" in widget.status
        finally:
            widget.close()


class TestFocusFromAnExpandedView:
    """Focus has to be reachable once blocks are open, not only while collapsed.

    A collapsed block is a node with a frame; an open one is a Graphviz cluster
    with a frame. Wiring only the first meant that after opening a few levels
    there was no way to focus at all, and so no way to reach the breadcrumb and
    its Reset.
    """

    def test_the_frontend_wires_open_block_frames_too(self):
        """The cluster frame sends the same focus request a node frame does."""
        javascript = (STATIC / "widget.js").read_text()
        wiring = javascript.split("const wireOpenBlocks = () => {")[1].split("\n  };")[0]
        assert "loman-block-frame" in wiring
        assert '"focus_request"' in wiring
        # The label still closes the block; the frame must not steal that.
        assert '"toggle_request"' in wiring

    def test_svg_tooltips_are_built_with_append_child(self):
        """Node.append returns undefined, so assigning to its result throws.

        That mistake silently took the rest of the handler with it, leaving open
        blocks with neither a close nor a focus target.
        """
        javascript = (STATIC / "widget.js").read_text()
        helper = javascript.split("const svgTooltip = (element, text) => {")[1].split("};")[0]
        assert "appendChild" in helper
        assert ".append(document.createElementNS" not in javascript

    def test_focusing_by_path_works_while_blocks_are_open(self):
        """This is the request an open block's frame sends."""
        comp = Computation()
        inner = Computation()
        inner.add_node("x", value=1)
        mid = Computation()
        mid.add_block("cds", inner, keep_values=True)
        comp.add_block("credit", mid, keep_values=True)
        widget = comp.widget()
        try:
            widget._expanded = {to_nodekey("credit"), to_nodekey("credit/cds")}
            widget.refresh()

            widget.focus_request = {"path": "credit", "request_id": "f1"}

            assert widget.status == "Focused on credit"
            assert [entry["label"] for entry in widget.focus_trail] == ["Reset", "credit"]
            # Expansions under the new root survive; the rest are gone.
            assert widget.expanded_paths == ["credit/cds"]
        finally:
            widget.close()

    def test_reset_returns_to_the_top_and_clears_expansions(self):
        """Reset means the whole graph, not merely an unfocused deep view."""
        comp = create_example_block_computation()
        widget = comp.widget()
        try:
            widget.toggle_request = {"id": node_id(widget, "foo"), "request_id": "t1"}
            widget.focus_request = {"id": node_id(widget, "bar"), "request_id": "f1"}
            assert widget.focus_trail[-1]["label"] == "bar"

            widget.focus_request = {"path": "", "request_id": "f2"}

            assert widget.status == "Showing the whole graph"
            assert widget.focus_trail == [{"label": "Reset", "path": ""}]
            assert widget.expanded_paths == []
        finally:
            widget.close()
