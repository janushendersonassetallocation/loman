"""Tests for the pure view-model builders behind the notebook widget."""

from loman import Computation, States
from loman.nodekey import to_nodekey
from loman.ui.viewmodel import MIXED_STATE_LABEL, build_detail, node_states, state_colors, state_label
from loman.visualization import ColorByState
from tests.conftest import create_example_block_computation


class TestStateLabel:
    """Reducing member states to the label shown for one rendered node."""

    def test_single_member_reports_its_own_state(self):
        """An ordinary node displays exactly its own state."""
        assert state_label([States.COMPUTABLE]) == "COMPUTABLE"

    def test_error_beats_every_other_state(self):
        """One failed member makes the whole block read as an error."""
        assert state_label([States.UPTODATE, States.ERROR, States.STALE]) == "ERROR"

    def test_stale_beats_agreement_among_the_rest(self):
        """A stale member outranks otherwise up-to-date members."""
        assert state_label([States.UPTODATE, States.STALE, States.UPTODATE]) == "STALE"

    def test_unanimous_members_report_the_common_state(self):
        """A block whose members agree displays that shared state."""
        assert state_label([States.UPTODATE, States.UPTODATE]) == "UPTODATE"

    def test_disagreeing_members_report_mixed(self):
        """Members in genuinely different states have no single state."""
        assert state_label([States.UPTODATE, States.COMPUTABLE]) == MIXED_STATE_LABEL

    def test_empty_membership_reports_mixed(self):
        """A rendered node with no members degrades rather than raising."""
        assert state_label([]) == MIXED_STATE_LABEL

    def test_label_matches_the_colour_the_graph_paints(self):
        """The widget label and the Graphviz fill colour cannot disagree.

        Both derive from ``aggregate_states``; this pins the shared behaviour so
        that a change to one has to change the other.
        """
        colors = state_colors()
        formatter = ColorByState()
        for states in (
            [States.UPTODATE, States.ERROR],
            [States.UPTODATE, States.STALE],
            [States.UPTODATE, States.UPTODATE],
            [States.UPTODATE, States.COMPUTABLE],
        ):
            nodes = [_FakeNode(state) for state in states]
            painted = formatter.format(to_nodekey("x"), nodes, is_composite=True)
            assert colors[state_label(states)] == painted["fillcolor"]


class _FakeNode:
    """Minimal stand-in for the node objects a NodeFormatter receives."""

    def __init__(self, state):
        """Record the state this node should report."""
        self.data = {"state": state}


class TestStateColors:
    """Exposing Loman's colour map to the browser."""

    def test_every_state_has_a_colour(self):
        """A new state can never render unstyled in the widget."""
        colors = state_colors()
        for state in States:
            assert state.name in colors
        assert MIXED_STATE_LABEL in colors

    def test_custom_colour_map_is_honoured(self):
        """A caller-supplied cmap replaces the defaults, keys and all."""
        colors = state_colors({States.UPTODATE: "#000000", None: "#ffffff"})
        assert colors == {"UPTODATE": "#000000", MIXED_STATE_LABEL: "#ffffff"}


class TestNodeStates:
    """The compact repaint payload."""

    def test_one_entry_per_rendered_node(self):
        """Every rendered shape gets a state, keyed by its rendered ID."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        view = comp.draw(collapse_all=False)

        states = node_states(view)

        assert set(states) == set(view.node_index_map.values())
        assert states[view.node_index_map[to_nodekey("a")]] == "UPTODATE"
        assert states[view.node_index_map[to_nodekey("b")]] == "COMPUTABLE"

    def test_collapsed_block_reports_one_aggregate_state(self):
        """A block collapses to a single entry, not one per member."""
        comp = create_example_block_computation()
        view = comp.draw()

        states = node_states(view)

        foo = to_nodekey("foo")
        assert len(view.original_nodes[foo]) == 4
        assert states[view.node_index_map[foo]] == "STALE"
        assert len(states) == len(view.node_index_map)


class TestBuildDetail:
    """The lazily populated detail panel."""

    def test_unknown_rendered_id_yields_nothing(self):
        """A stale ID from the browser produces an empty payload, not an error."""
        comp = Computation()
        comp.add_node("a", value=1)

        assert build_detail(comp.draw(collapse_all=False), "no-such-id", editable=True) == {}

    def test_scalar_input_is_reported_editable(self):
        """An uncomputed scalar input offers the edit control."""
        comp = Computation()
        comp.add_node("a", value=1)
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("a")], editable=True)

        assert detail["editable"] is True
        assert detail["value"] == {"kind": "scalar", "type": "int", "value": 1}
        assert detail["source"] == "NOT A CALCULATED NODE"

    def test_calculated_node_is_never_editable(self):
        """Editing a calculated node would be overwritten by the next compute."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        comp.compute_all()
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("b")], editable=True)

        assert detail["editable"] is False
        assert detail["inputs"] == ["a"]
        assert detail["timing"]["duration"] >= 0

    def test_non_scalar_value_is_not_editable(self):
        """Only the scalar wire format round-trips, so only scalars can be edited."""
        comp = Computation()
        comp.add_node("a", value=[1, 2, 3])
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("a")], editable=True)

        assert detail["editable"] is False
        assert detail["value"]["kind"] == "repr"

    def test_read_only_widget_reports_nothing_editable(self):
        """The editable flag gates the payload, not just the browser control."""
        comp = Computation()
        comp.add_node("a", value=1)
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("a")], editable=False)

        assert detail["editable"] is False

    def test_placeholder_node_is_not_editable(self):
        """A placeholder has no value to replace, and insert would raise."""
        comp = Computation()
        comp.add_node("b", lambda a: a + 1)
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("a")], editable=True)

        assert detail["state"] == "PLACEHOLDER"
        assert detail["editable"] is False

    def test_error_node_carries_its_traceback(self):
        """A failed node shows why it failed, not just a red shape."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: 1 / 0)
        comp.compute_all()
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("b")], editable=True)

        assert detail["state"] == "ERROR"
        assert "ZeroDivisionError" in detail["error"]
        assert detail["editable"] is False

    def test_collapsed_block_lists_members_and_stops_there(self):
        """A block has no single value, timing or source to report."""
        comp = create_example_block_computation()
        view = comp.draw()

        detail = build_detail(view, view.node_index_map[to_nodekey("foo")], editable=True)

        assert detail["composite"] is True
        assert set(detail["members"]) == {"foo/a", "foo/b", "foo/c", "foo/d"}
        assert detail["editable"] is False
        assert "value" not in detail
        assert "source" not in detail

    def test_source_is_reported_when_it_can_be_recovered(self):
        """A normally defined function shows its source in the panel."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", _double)
        view = comp.draw(collapse_all=False)

        detail = build_detail(view, view.node_index_map[to_nodekey("b")], editable=True)

        assert "return a * 2" in detail["source"]

    def test_unrecoverable_source_degrades_gracefully(self, mocker):
        """Lambdas from a REPL or restored from dill have no recoverable source."""
        comp = Computation()
        comp.add_node("a", value=1)
        comp.add_node("b", lambda a: a + 1)
        view = comp.draw(collapse_all=False)
        mocker.patch.object(Computation, "get_source", side_effect=OSError("could not get source code"))

        detail = build_detail(view, view.node_index_map[to_nodekey("b")], editable=True)

        assert detail["source"] == "Source unavailable for this callable (OSError)"

    def test_caller_supplied_id_map_is_used(self):
        """The widget passes in the map it already keeps rather than rebuilding it."""
        comp = Computation()
        comp.add_node("a", value=1)
        view = comp.draw(collapse_all=False)
        node_id = view.node_index_map[to_nodekey("a")]

        supplied = build_detail(view, node_id, editable=True, id_to_visible={node_id: to_nodekey("a")})
        rebuilt = build_detail(view, node_id, editable=True)

        assert supplied == rebuilt


def _double(a):
    """Return twice its argument."""
    return a * 2
