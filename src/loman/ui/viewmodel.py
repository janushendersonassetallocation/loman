"""Pure view-model builders shared by the widget and its tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loman.computeengine import Error
from loman.consts import NodeAttributes, States
from loman.nodekey import NodeKey
from loman.visualization import ColorByState, GraphView

from .value import to_wire

if TYPE_CHECKING:
    from loman.computeengine import Computation


def aggregate_state(states: list[States]) -> str:
    """Return the display state for one rendered node or collapsed block."""
    if len(states) == 1:
        return states[0].name
    if States.ERROR in states:
        return States.ERROR.name
    if States.STALE in states:
        return States.STALE.name
    if states and all(state == states[0] for state in states):
        return states[0].name
    return "MIXED"


def state_colors(cmap: dict[States | None, str] | None = None) -> dict[str, str]:
    """Convert Loman's state-colour mapping to JSON-safe string keys."""
    colors = ColorByState.DEFAULT_STATE_COLORS if cmap is None else cmap
    return {("MIXED" if state is None else state.name): color for state, color in colors.items()}


def node_states(view: GraphView) -> dict[str, str]:
    """Build the small rendered-ID to state map used for repainting."""
    result: dict[str, str] = {}
    for visible_key, node_id in view.node_index_map.items():
        members = view.original_nodes[visible_key]
        states = [view.computation.dag.nodes[node][NodeAttributes.STATE] for node in members]
        result[node_id] = aggregate_state(states)
    return result


def _safe_source(computation: Computation, node_key: NodeKey) -> str:
    """Return source text, tolerating interactive and restored callables."""
    try:
        return computation.get_source(node_key)
    except (OSError, TypeError):
        return "Source unavailable for this callable"


def build_detail(view: GraphView, node_id: str, *, editable: bool) -> dict[str, Any]:
    """Build the lazily populated detail panel for one rendered node."""
    id_to_visible = {value: key for key, value in view.node_index_map.items()}
    visible_key = id_to_visible.get(node_id)
    if visible_key is None:
        return {}
    computation = view.computation
    members = view.original_nodes[visible_key]
    member_states = [computation.dag.nodes[node][NodeAttributes.STATE] for node in members]
    detail: dict[str, Any] = {
        "id": node_id,
        "name": str(visible_key),
        "state": aggregate_state(member_states),
        "members": [str(member) for member in members],
        "composite": visible_key in view.composite_nodes,
        "editable": False,
    }
    if len(members) != 1:
        return detail

    node_key = members[0]
    node = computation.dag.nodes[node_key]
    value = node.get(NodeAttributes.VALUE)
    value_wire = to_wire(value)
    timing = node.get(NodeAttributes.TIMING)
    detail.update(
        {
            "name": str(node_key),
            "value": value_wire,
            "timing": None
            if timing is None
            else {"start": timing.start.isoformat(), "end": timing.end.isoformat(), "duration": timing.duration},
            "source": _safe_source(computation, node_key),
            "inputs": [str(name) for name in computation.get_inputs(node_key)],
            "outputs": [str(name) for name in computation.get_outputs(node_key)],
        }
    )
    if isinstance(value, Error):
        detail["error"] = value.traceback
    detail["editable"] = bool(
        editable
        and node.get(NodeAttributes.FUNC) is None
        and node.get(NodeAttributes.STATE) != States.PLACEHOLDER
        and value_wire["kind"] == "scalar"
    )
    return detail
