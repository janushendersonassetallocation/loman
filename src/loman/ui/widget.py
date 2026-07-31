"""AnyWidget implementation for live Loman computation graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import anywidget
import traitlets

from loman.computeengine import ComputationEvent
from loman.consts import NodeTransformations
from loman.nodekey import Name, NodeKey, to_nodekey

from .value import from_wire
from .viewmodel import build_detail, node_states, state_colors

if TYPE_CHECKING:
    from collections.abc import Callable

    from loman.computeengine import Computation
    from loman.visualization import GraphView


_STATIC = Path(__file__).parent / "static"


class _DrawOptions(TypedDict):
    """Keyword options forwarded to :meth:`Computation.draw`."""

    cmap: Any
    colors: str
    shapes: str | None
    graph_attr: dict[str, Any] | None
    node_attr: dict[str, Any] | None
    edge_attr: dict[str, Any] | None
    show_expansion: bool
    collapse_all: bool


class ComputationWidget(anywidget.AnyWidget):
    """Interactive graph view that automatically follows a computation."""

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    graph_svg = traitlets.Unicode("").tag(sync=True)
    node_states = traitlets.Dict(default_value={}).tag(sync=True)
    state_colors = traitlets.Dict(default_value={}).tag(sync=True)
    composite_ids = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    selected_id = traitlets.Unicode("").tag(sync=True)
    detail = traitlets.Dict(default_value={}).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    editable = traitlets.Bool(True).tag(sync=True)
    repaint_states = traitlets.Bool(True).tag(sync=True)
    revision = traitlets.Int(0).tag(sync=True)

    edit_request = traitlets.Dict(default_value={}).tag(sync=True)
    compute_request = traitlets.Dict(default_value={}).tag(sync=True)
    toggle_request = traitlets.Dict(default_value={}).tag(sync=True)

    def __init__(
        self,
        computation: Computation,
        root: NodeKey | None = None,
        *,
        node_transformations: dict[Name, str] | None = None,
        cmap: Any = None,
        colors: str = "state",
        shapes: str | None = None,
        graph_attr: dict[str, Any] | None = None,
        node_attr: dict[str, Any] | None = None,
        edge_attr: dict[str, Any] | None = None,
        show_expansion: bool = False,
        collapse_all: bool = True,
        editable: bool = True,
    ) -> None:
        """Create a widget and subscribe it to ``computation``."""
        self.computation = computation
        self._root = root
        self._base_transformations = {} if node_transformations is None else node_transformations.copy()
        self._expanded: set[NodeKey] = set()
        self._draw_options: _DrawOptions = {
            "cmap": cmap,
            "colors": colors,
            "shapes": shapes,
            "graph_attr": graph_attr,
            "node_attr": node_attr,
            "edge_attr": edge_attr,
            "show_expansion": show_expansion,
            "collapse_all": collapse_all,
        }
        self._view: GraphView | None = None
        self._id_to_visible: dict[str, NodeKey] = {}
        self._canonical_graph_svg = ""
        self._unsubscribe: Callable[[], None] | None = None
        super().__init__(editable=editable, repaint_states=colors == "state")
        custom_colors = cmap if colors == "state" and isinstance(cmap, dict) else None
        self.state_colors = state_colors(custom_colors)
        self.refresh()
        self._unsubscribe = computation.subscribe(self._on_computation_event)

    @property
    def selected_names(self) -> list[Name]:
        """Return the real Loman names represented by the selected shape."""
        if self._view is None:
            return []
        visible = self._id_to_visible.get(self.selected_id)
        if visible is None:
            return []
        return [node.name for node in self._view.original_nodes[visible]]

    @property
    def selected_name(self) -> Name | None:
        """Return the selected Loman name, or block path for a composite."""
        names = self.selected_names
        if len(names) == 1:
            return names[0]
        visible = self._id_to_visible.get(self.selected_id)
        return None if visible is None else self._full_visible_key(visible).name

    @property
    def selected(self) -> Name | None:
        """Alias for :attr:`selected_name`."""
        return self.selected_name

    def _full_visible_key(self, visible: NodeKey) -> NodeKey:
        """Restore a rooted view key to its full computation path."""
        return visible if self._root is None else to_nodekey(self._root).join(visible)

    def _make_view(self) -> GraphView:
        """Create the current GraphView, including interactive expansions."""
        transformations = self._base_transformations.copy()
        transformations.update(dict.fromkeys(self._expanded, NodeTransformations.EXPAND))
        return self.computation.draw(
            self._root,
            node_transformations=transformations,
            **self._draw_options,
        )

    def refresh(self) -> None:
        """Force a full graph refresh, including SVG layout and node identity."""
        selected_members: frozenset[NodeKey] | None = None
        if self._view is not None and self.selected_id:
            selected_visible = self._id_to_visible.get(self.selected_id)
            if selected_visible is not None:
                selected_members = frozenset(self._view.original_nodes[selected_visible])
        try:
            view = self._make_view()
            svg = view.svg() or ""
        except Exception as exc:
            self.status = f"Unable to render Graphviz SVG: {exc}"
            return
        self._view = view
        self._id_to_visible = {node_id: visible for visible, node_id in view.node_index_map.items()}
        self._canonical_graph_svg = svg
        self.node_states = node_states(view)
        self.composite_ids = [view.node_index_map[node] for node in view.composite_nodes]
        self.revision = self.computation.revision
        if self.selected_id:
            selected_visible = self._id_to_visible.get(self.selected_id)
            refreshed_members = None if selected_visible is None else frozenset(view.original_nodes[selected_visible])
            if refreshed_members != selected_members:
                self.selected_id = ""
        self.graph_svg = svg
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        """Refresh the selected node's lazy detail payload."""
        self.detail = {} if self._view is None else build_detail(self._view, self.selected_id, editable=self.editable)

    def _on_computation_event(self, event: ComputationEvent) -> None:
        """Apply an automatic incremental or structural computation update."""
        if event.graph_changed or not self.repaint_states:
            self.refresh()
            return
        if self._view is None:
            self.refresh()
            return
        self.node_states = node_states(self._view)
        self.revision = event.revision
        visible = self._id_to_visible.get(self.selected_id)
        if visible is not None and set(self._view.original_nodes[visible]).intersection(event.changed_nodes):
            self._refresh_detail()

    @traitlets.observe("selected_id")
    def _selected_changed(self, _change: dict[str, Any]) -> None:
        """Populate details when the browser selects a rendered node."""
        if hasattr(self, "_view"):
            self._refresh_detail()

    @traitlets.observe("composite_ids", "detail", "graph_svg", "node_states", "revision")
    def _canonical_output_changed(self, change: dict[str, Any]) -> None:
        """Reject stale derived traits echoed by a reconnecting browser model."""
        if not hasattr(self, "_view") or self._view is None:
            return
        name = change["name"]
        if name == "composite_ids":
            expected = [self._view.node_index_map[node] for node in self._view.composite_nodes]
        elif name == "detail":
            expected = build_detail(self._view, self.selected_id, editable=self.editable)
        elif name == "graph_svg":
            expected = self._canonical_graph_svg
        elif name == "node_states":
            expected = node_states(self._view)
        else:
            expected = self.computation.revision
        if change["new"] != expected:
            setattr(self, name, expected)

    @traitlets.observe("edit_request")
    def _edit_requested(self, change: dict[str, Any]) -> None:
        """Validate and apply one scalar edit requested by the browser."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible"):
            return
        if not self.editable:
            self.status = "Edit failed: this widget is read-only"
            return
        try:
            visible = self._id_to_visible[request["id"]]
            assert self._view is not None  # noqa: S101
            members = self._view.original_nodes[visible]
            if len(members) != 1:
                self.status = "Edit failed: collapsed blocks cannot be edited"
                return
            current_detail = build_detail(self._view, request["id"], editable=self.editable)
            if not current_detail.get("editable"):
                self.status = "Edit failed: this node is not an editable scalar input"
                return
            self.computation.insert(members[0], from_wire(request["value"]))
            self.status = f"Updated {members[0]}"
        except Exception as exc:
            self.status = f"Edit failed: {exc}"

    @traitlets.observe("compute_request")
    def _compute_requested(self, change: dict[str, Any]) -> None:
        """Compute a selected target or the whole graph."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible"):
            return
        if not self.editable:
            self.status = "Compute failed: this widget is read-only"
            return
        try:
            if request.get("all"):
                self.computation.compute_all()
                self.status = "Computed all available nodes"
                return
            visible = self._id_to_visible[request["id"]]
            assert self._view is not None  # noqa: S101
            names = [member.name for member in self._view.original_nodes[visible]]
            self.computation.compute(names)
            self.status = f"Computed {self._full_visible_key(visible)}"
        except Exception as exc:
            self.status = f"Compute failed: {exc}"

    @traitlets.observe("toggle_request")
    def _toggle_requested(self, change: dict[str, Any]) -> None:
        """Expand a composite node or collapse all interactive expansions."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible"):
            return
        try:
            if request.get("collapse_all"):
                self.status = "Collapsing all blocks..."
                self._expanded.clear()
                success = "Collapsed all blocks"
            else:
                visible = self._id_to_visible[request["id"]]
                if self._view is None or visible not in self._view.composite_nodes:
                    self.status = "Expand/collapse failed: only collapsed blocks can be expanded"
                    return
                block = self._full_visible_key(visible)
                self.status = f"Opening {block}..."
                self._expanded.add(block)
                success = f"Opened {block}"
            self.refresh()
            if not self.status.startswith("Unable to render Graphviz SVG:"):
                self.status = success
        except Exception as exc:
            self.status = f"Expand/collapse failed: {exc}"

    def close(self) -> None:
        """Unsubscribe from the computation and close the widget comm."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        super().close()
