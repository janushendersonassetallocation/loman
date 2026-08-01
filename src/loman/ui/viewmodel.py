"""Pure view-model builders shared by the widget and its tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loman.computeengine import Error
from loman.consts import NodeAttributes, States
from loman.nodekey import NodeKey
from loman.visualization import ColorByState, GraphView, aggregate_states

from .value import to_wire

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from loman.computeengine import Computation

#: Label used where the graph renderer uses ``None``: a collapsed block whose
#: members are in genuinely different states.
MIXED_STATE_LABEL = "MIXED"


def state_label(states: Sequence[States | None]) -> str:
    """Return the display label for one rendered node or collapsed block.

    Thin wrapper over :func:`loman.visualization.aggregate_states` that renames
    the mixed marker from ``None`` to a JSON-safe string, so the widget and the
    Graphviz picture always agree on what a rendered node represents.

    :param states: States of the members behind one rendered node.
    :return: A :class:`~loman.consts.States` name, or ``MIXED``.
    """
    state = aggregate_states(states)
    return MIXED_STATE_LABEL if state is None else state.name


def state_colors(cmap: dict[States | None, str] | None = None) -> dict[str, str]:
    """Convert Loman's state-colour mapping to JSON-safe string keys.

    :param cmap: Custom state colours, or ``None`` for Loman's defaults.
    :return: Colours keyed by the labels :func:`state_label` produces.
    """
    colors = ColorByState.DEFAULT_STATE_COLORS if cmap is None else cmap
    return {(MIXED_STATE_LABEL if state is None else state.name): color for state, color in colors.items()}


def node_states(view: GraphView) -> dict[str, str]:
    """Build the small rendered-ID to state map used for repainting.

    :param view: The graph view whose rendered nodes should be described.
    :return: One state label per rendered node ID.
    """
    result: dict[str, str] = {}
    for visible_key, node_id in view.node_index_map.items():
        members = view.original_nodes[visible_key]
        states = [view.computation.dag.nodes[node][NodeAttributes.STATE] for node in members]
        result[node_id] = state_label(states)
    return result


def _safe_source(computation: Computation, node_key: NodeKey) -> str:
    """Return source text, tolerating interactive and restored callables.

    Loman users define lambdas constantly, and :func:`inspect.getsource` cannot
    recover the text of one typed at a REPL or rehydrated from dill. The detail
    panel says so rather than failing.

    :param computation: Computation owning the node.
    :param node_key: Node whose source is wanted.
    :return: Source text, or a short explanation of why it is unavailable.
    """
    try:
        return computation.get_source(node_key)
    except (OSError, TypeError, SyntaxError) as exc:
        return f"Source unavailable for this callable ({type(exc).__name__})"


def build_detail(
    view: GraphView, node_id: str, *, editable: bool, id_to_visible: Mapping[str, NodeKey] | None = None
) -> dict[str, Any]:
    """Build the lazily populated detail panel for one rendered node.

    :param view: The graph view the rendered node belongs to.
    :param node_id: Rendered node ID, as carried by the SVG title element.
    :param editable: Whether the widget permits edits at all.
    :param id_to_visible: Reverse of ``view.node_index_map``. The widget keeps
        one and passes it in; omit it and this rebuilds it.
    :return: The detail payload, or an empty dict for an unknown ID.
    """
    if id_to_visible is None:
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
        "state": state_label(member_states),
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
