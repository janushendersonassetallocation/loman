"""AnyWidget implementation for live Loman computation graphs."""

from __future__ import annotations

import functools
import logging
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import anywidget
import traitlets

from loman.computeengine import ComputationEvent
from loman.consts import NodeTransformations
from loman.nodekey import Name, NodeKey, to_nodekey

from .value import apply_cell_edit, from_wire
from .viewmodel import build_detail, node_states, state_colors

if TYPE_CHECKING:
    from collections.abc import Iterator

    from loman.computeengine import Computation
    from loman.visualization import GraphView

LOG = logging.getLogger("loman.ui.widget")

_STATIC = Path(__file__).parent / "static"

#: How many recent browser request IDs to remember. Requests carry a nonce so
#: that repeating an identical action still registers as a trait change; keeping
#: the recent ones lets a widget ignore a request a reconnecting or replayed
#: front-end model pushes back at it. Small on purpose: this guards against
#: accidental replay, not against a determined caller.
_REQUEST_HISTORY = 64

#: Default ceiling on how many nodes one expand request may put on screen.
#: Graphviz output measures at roughly 0.6 KiB and 0.6 ms of ``dot`` time per
#: rendered node, both linear, so 500 nodes is about 300 KiB and a third of a
#: second per relayout --- slow but usable. Beyond that a single click can hang
#: the kernel for seconds and push a megabyte at the browser, so opening a block
#: that large is refused rather than merely sluggish. Raise it with
#: ``max_rendered_nodes=`` if you know what you are asking for.
DEFAULT_MAX_RENDERED_NODES = 500

#: Sentinel from :meth:`ComputationWidget._canonical_output`: the trait is
#: view-dependent and no view exists yet, so its echoed value is left alone.
_NO_CANONICAL = object()


def _acknowledges(method: Callable[[Any, dict[str, Any]], None]) -> Callable[[Any, dict[str, Any]], None]:
    """Acknowledge a browser request once the observer has finished with it."""

    @functools.wraps(method)
    def wrapped(self: ComputationWidget, change: dict[str, Any]) -> None:
        """Run the observer, then bump the acknowledgement counter."""
        try:
            method(self, change)
        finally:
            self._acknowledge()

    return wrapped


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
    """Interactive graph view that automatically follows a computation.

    The widget subscribes to its computation and repaints as that computation
    changes. It navigates and lightly controls; the real object stays in Python,
    so ``comp.v[widget.selected_name]`` remains the way to get at a value.

    Two costs are worth knowing about:

    * Computation happens synchronously in the kernel, inside the observer that
      handles the request. A slow graph freezes the widget, so drive long
      computations from an ordinary cell and let the widget observe the result.
    * With the default ``colors="state"`` a state change repaints existing SVG
      shapes in place. Any other colouring, such as ``colors="timing"``, depends
      on values rather than states, so every mutation re-runs Graphviz --- one
      ``dot`` subprocess per change.

    On lifetime: the computation subscribes to a bound method and so holds only
    a weak reference, meaning it never keeps a widget alive by itself. ipywidgets
    is the one that does --- it registers every open widget in a process-wide
    table until the widget is closed. Call :meth:`close` when you are finished
    with a widget; that both unsubscribes it and releases it.
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    graph_svg = traitlets.Unicode("").tag(sync=True)
    node_states = traitlets.Dict(default_value={}).tag(sync=True)
    state_colors = traitlets.Dict(default_value={}).tag(sync=True)
    composite_ids = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    selected_id = traitlets.Unicode("").tag(sync=True)
    detail = traitlets.Dict(default_value={}).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    status_severity = traitlets.Unicode("idle").tag(sync=True)
    #: Bumped after every browser request, whatever the outcome. The front end
    #: shows an optimistic busy state while it waits, and a request can
    #: legitimately change nothing else --- collapsing an already-collapsed
    #: graph re-renders identical SVG and re-reports an identical status, so
    #: neither trait fires and the browser would wait for ever.
    ack = traitlets.Int(0).tag(sync=True)
    expanded_paths = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    editable = traitlets.Bool(True).tag(sync=True)
    repaint_states = traitlets.Bool(True).tag(sync=True)
    revision = traitlets.Int(0).tag(sync=True)
    #: Graphviz layout direction. Defaults to ``LR`` because computations read
    #: left to right, from inputs to results; the toolbar toggles it to ``TB``.
    rankdir = traitlets.Unicode("LR").tag(sync=True)
    #: Breadcrumb from the widget's own root down to the block in focus. Each
    #: entry is ``{"label": str, "path": str}``; the front end renders it and
    #: sends a focus_request to climb back up.
    focus_trail = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)

    edit_request = traitlets.Dict(default_value={}).tag(sync=True)
    compute_request = traitlets.Dict(default_value={}).tag(sync=True)
    toggle_request = traitlets.Dict(default_value={}).tag(sync=True)
    layout_request = traitlets.Dict(default_value={}).tag(sync=True)
    focus_request = traitlets.Dict(default_value={}).tag(sync=True)

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
        max_rendered_nodes: int = DEFAULT_MAX_RENDERED_NODES,
        rankdir: str = "LR",
    ) -> None:
        """Create a widget and subscribe it to ``computation``.

        Arguments other than ``editable`` and ``max_rendered_nodes`` mirror
        :meth:`Computation.draw`.

        :param editable: Permit scalar input edits and computation controls.
            Expanding and collapsing blocks stays available either way, because
            navigating a graph does not mutate it.
        :param max_rendered_nodes: Refuse an expand request that would put more
            than this many nodes on screen. It does not cap the initial view:
            what you asked to draw is drawn.
        :param rankdir: Initial Graphviz layout direction, ``LR`` (default) or
            ``TB``. A ``rankdir`` given in ``graph_attr`` takes precedence. The
            toolbar toggles it live either way.
        """
        self.computation = computation
        self._root = root
        self._base_root = root
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
        self._canonical_status = ""
        self._canonical_severity = "idle"
        self._canonical_ack = 0
        # An explicit rankdir in graph_attr wins, so a caller who set the layout
        # direction the old way still gets what they asked for; otherwise the
        # left-to-right default applies.
        self._canonical_rankdir = str(graph_attr["rankdir"]) if graph_attr and "rankdir" in graph_attr else rankdir
        self._seen_requests: deque[str] = deque(maxlen=_REQUEST_HISTORY)
        self._max_rendered_nodes = max_rendered_nodes
        self._writing = 0
        self._unsubscribe: Callable[[], None] | None = None
        super().__init__(editable=editable, repaint_states=colors == "state", rankdir=self._canonical_rankdir)
        custom_colors = cmap if colors == "state" and isinstance(cmap, dict) else None
        self.state_colors = state_colors(custom_colors)
        self.refresh()
        self._unsubscribe = computation.subscribe(self._on_computation_event)

    @property
    def selected_names(self) -> list[Name]:
        """Return the real Loman names represented by the selected shape.

        A collapsed block reports every member; an ordinary node reports one
        name; nothing selected reports an empty list.
        """
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

    def _focus_trail(self) -> list[dict[str, str]]:
        """Describe the path from the widget's own root to the block in focus.

        The first entry is the widget's root, labelled ``All`` when the whole
        computation is in view; each further entry is one block descended into.
        Paths are full computation paths, so the front end can hand any of them
        straight back as a focus_request.
        """
        base = None if self._base_root is None else to_nodekey(self._base_root)
        current = None if self._root is None else to_nodekey(self._root)
        trail = [{"label": "All" if base is None else base.label, "path": "" if base is None else str(base)}]
        if current is None or current == base:
            return trail
        relative = current.drop_root(base)
        acc = base if base is not None else NodeKey.root()
        for part in relative.parts:  # type: ignore[union-attr]
            acc = acc.join_parts(part)
            trail.append({"label": str(part), "path": str(acc)})
        return trail

    def _make_view(self) -> GraphView:
        """Create the current GraphView, including interactive expansions."""
        transformations = self._base_transformations.copy()
        transformations.update(dict.fromkeys(self._expanded, NodeTransformations.EXPAND))
        options = dict(self._draw_options)
        graph_attr = dict(options["graph_attr"] or {})
        graph_attr["rankdir"] = self.rankdir
        options["graph_attr"] = graph_attr
        return self.computation.draw(
            self._root,
            node_transformations=transformations,
            **options,  # type: ignore[arg-type]
        )

    @contextmanager
    def _own_write(self) -> Iterator[None]:
        """Mark trait writes as Python's own, so echo checking can skip them."""
        self._writing += 1
        try:
            yield
        finally:
            self._writing -= 1

    def _set_status(self, text: str, severity: str = "success") -> None:
        """Set the status line and record it as Python's canonical value.

        Status must go through here rather than being assigned directly. When
        the browser sends a request, ipywidgets applies the whole incoming state
        inside one ``hold_trait_notifications`` block: our observer runs and sets
        a status, and the browser's own stale copy of ``status`` is then applied
        on top, silently reverting it. Recording the value here lets
        :meth:`_canonical_output_changed` put it back.

        :param text: Message to show.
        :param severity: ``success``, ``error`` or ``idle``. Sent explicitly so
            the front end styles the message rather than guessing from its
            wording.
        """
        self._canonical_status = text
        self._canonical_severity = "idle" if not text else severity
        with self._own_write():
            self.status = text
            self.status_severity = self._canonical_severity

    def _fail(self, text: str) -> None:
        """Report a failed request on the status line."""
        self._set_status(text, "error")

    def _acknowledge(self) -> None:
        """Tell the browser a request has been dealt with.

        Sent unconditionally, because "nothing changed" is a real outcome and
        the front end cannot distinguish it from "still working" otherwise.
        """
        self._canonical_ack += 1
        with self._own_write():
            self.ack = self._canonical_ack

    def refresh(self) -> bool:
        """Force a full graph refresh, including SVG layout and node identity.

        This is the explicit escape hatch for changes the subscription cannot
        see, such as mutating ``computation.dag`` directly.

        :return: True on success. On failure the previous picture is left alone
            and :attr:`status` explains what went wrong.
        """
        selected_members: frozenset[NodeKey] | None = None
        if self._view is not None and self.selected_id:
            selected_visible = self._id_to_visible.get(self.selected_id)
            if selected_visible is not None:
                selected_members = frozenset(self._view.original_nodes[selected_visible])
        try:
            view = self._make_view()
            svg = view.svg() or ""
        except Exception as exc:
            # Rendering shells out to ``dot``; a missing binary, an unwriteable
            # temp dir or a malformed attribute all surface here, and none of
            # them should propagate out of a traitlets observer.
            LOG.exception("Loman widget could not render the computation graph")
            self._fail(f"Unable to render graph: {type(exc).__name__}: {exc}")
            return False
        with self._own_write():
            self._view = view
            self._id_to_visible = {node_id: visible for visible, node_id in view.node_index_map.items()}
            self._canonical_graph_svg = svg
            self.node_states = node_states(view)
            self.composite_ids = [view.node_index_map[node] for node in view.composite_nodes]
            # An expanded block is drawn as a Graphviz cluster, not a node, so
            # there is no shape left to click to close it again. Naming the open
            # blocks lets the front end make their cluster labels the handle.
            self.expanded_paths = sorted(str(block) for block in self._expanded)
            self.focus_trail = self._focus_trail()
            self.revision = self.computation.revision
            if self.selected_id:
                selected_visible = self._id_to_visible.get(self.selected_id)
                refreshed = None if selected_visible is None else frozenset(view.original_nodes[selected_visible])
                if refreshed != selected_members:
                    # A relayout reuses rendered IDs, so the shape this ID now
                    # names may be a different node. Drop the selection rather
                    # than silently move it.
                    self.selected_id = ""
            self.graph_svg = svg
            self._refresh_detail()
        return True

    def _detail_for(self, node_id: str) -> dict[str, Any]:
        """Build the detail payload for one rendered node ID."""
        if self._view is None:
            return {}
        return build_detail(self._view, node_id, editable=self.editable, id_to_visible=self._id_to_visible)

    def _refresh_detail(self) -> None:
        """Refresh the selected node's lazy detail payload."""
        with self._own_write():
            self.detail = self._detail_for(self.selected_id)

    def _on_computation_event(self, event: ComputationEvent) -> None:
        """Apply an automatic incremental or structural computation update."""
        if event.graph_changed or not self.repaint_states or self._view is None:
            self.refresh()
            return
        with self._own_write():
            self.node_states = node_states(self._view)
            self.revision = event.revision
        visible = self._id_to_visible.get(self.selected_id)
        if visible is not None and not set(self._view.original_nodes[visible]).isdisjoint(event.changed_nodes):
            self._refresh_detail()

    def _claim_request(self, request: dict[str, Any]) -> bool:
        """Report whether a browser request is new rather than a replay.

        Every request carries a ``request_id`` nonce so that repeating the same
        action still reads as a trait change. Remembering the recent ones stops
        a reconnecting or recreated front-end model from re-applying an edit or
        a compute it already had in hand.
        """
        request_id = request.get("request_id")
        if request_id is None:
            return True
        if request_id in self._seen_requests:
            LOG.debug("Ignoring replayed Loman widget request %s", request_id)
            return False
        self._seen_requests.append(request_id)
        return True

    @traitlets.observe("selected_id")
    def _selected_changed(self, _change: dict[str, Any]) -> None:
        """Populate details when the browser selects a rendered node."""
        if hasattr(self, "_view"):
            self._refresh_detail()

    def _canonical_output(self, name: str) -> Any:
        """Return the value trait ``name`` should hold.

        Split from :meth:`_canonical_output_changed` to keep that observer flat:
        this owns the per-trait lookup. Returns :data:`_NO_CANONICAL` when the
        trait is view-dependent and no view has been rendered yet, so its echoed
        value is left untouched rather than reverted to nothing.

        :param name: Name of the derived trait being checked.
        :return: The canonical value, or :data:`_NO_CANONICAL`.
        """
        view_independent: dict[str, Callable[[], Any]] = {
            "ack": lambda: self._canonical_ack,
            "status": lambda: self._canonical_status,
            "status_severity": lambda: self._canonical_severity,
            "rankdir": lambda: self._canonical_rankdir,
            "focus_trail": self._focus_trail,
        }
        if name in view_independent:
            return view_independent[name]()
        if self._view is None:
            return _NO_CANONICAL
        view = self._view
        view_dependent: dict[str, Callable[[], Any]] = {
            "composite_ids": lambda: [view.node_index_map[node] for node in view.composite_nodes],
            "detail": lambda: self._detail_for(self.selected_id),
            "expanded_paths": lambda: sorted(str(block) for block in self._expanded),
            "graph_svg": lambda: self._canonical_graph_svg,
            "node_states": lambda: node_states(view),
            "revision": lambda: self.computation.revision,
        }
        return view_dependent[name]()

    @traitlets.observe(
        "ack",
        "composite_ids",
        "detail",
        "expanded_paths",
        "focus_trail",
        "graph_svg",
        "node_states",
        "rankdir",
        "revision",
        "status",
        "status_severity",
    )
    def _canonical_output_changed(self, change: dict[str, Any]) -> None:
        """Reject stale derived traits echoed by the browser model.

        These traits are Python's to own. The browser never writes them
        deliberately, but every message it sends carries its cached copy of the
        model, and ipywidgets applies that inside one
        ``hold_trait_notifications`` block --- so a value the browser captured
        before our observer ran lands *after* it, reverting the update. The same
        thing happens on reconnect. Anything that does not match is put back.
        """
        if self._writing or not hasattr(self, "_canonical_status"):
            return
        name = change["name"]
        expected = self._canonical_output(name)
        if expected is _NO_CANONICAL or change["new"] == expected:
            return
        with self._own_write():
            setattr(self, name, expected)

    @traitlets.observe("edit_request")
    @_acknowledges
    def _edit_requested(self, change: dict[str, Any]) -> None:
        """Validate and apply one edit requested by the browser.

        Two shapes arrive here: replacing a scalar node's value outright, and
        replacing a single cell of a tabular one.
        """
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible") or not self._claim_request(request):
            return
        if not self.editable:
            self._fail("Edit failed: this widget is read-only")
            return
        if self._view is None:
            self._fail("Edit failed: the graph is not rendered")
            return
        try:
            visible = self._id_to_visible[request["id"]]
            members = self._view.original_nodes[visible]
            if len(members) != 1:
                self._fail("Edit failed: collapsed blocks cannot be edited")
                return
            current_detail = self._detail_for(request["id"])
            cell = request.get("cell")
            if cell is not None:
                if not current_detail.get("cells_editable"):
                    self._fail("Edit failed: this node's cells are not editable")
                    return
                node_key = members[0]
                updated = apply_cell_edit(
                    self.computation.value(node_key),
                    int(cell["row"]),
                    int(cell["column"]),
                    from_wire(request["value"]),
                )
                self.computation.insert(node_key, updated)
                self._set_status(f"Updated {node_key} [{cell['row']}, {cell['column']}]")
                return
            if not current_detail.get("editable"):
                self._fail("Edit failed: this node is not an editable scalar input")
                return
            self.computation.insert(members[0], from_wire(request["value"]))
            self._set_status(f"Updated {members[0]}")
        except Exception as exc:
            # insert() rejects placeholder and missing nodes, and from_wire()
            # rejects malformed payloads. Raising here would surface only in the
            # kernel log and leave the UI looking like nothing happened.
            LOG.debug("Loman widget edit request failed", exc_info=True)
            self._fail(f"Edit failed: {type(exc).__name__}: {exc}")

    @traitlets.observe("compute_request")
    @_acknowledges
    def _compute_requested(self, change: dict[str, Any]) -> None:
        """Compute a selected target or the whole graph."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible") or not self._claim_request(request):
            return
        if not self.editable:
            self._fail("Compute failed: this widget is read-only")
            return
        try:
            if request.get("all"):
                self.computation.compute_all()
                self._set_status("Computed all available nodes")
                return
            if self._view is None:
                self._fail("Compute failed: the graph is not rendered")
                return
            visible = self._id_to_visible[request["id"]]
            names = [member.name for member in self._view.original_nodes[visible]]
            self.computation.compute(names)
            self._set_status(f"Computed {self._full_visible_key(visible)}")
        except Exception as exc:
            # compute() validates before running and raises for uninitialized
            # or placeholder ancestors; node failures land as ERROR states.
            LOG.debug("Loman widget compute request failed", exc_info=True)
            self._fail(f"Compute failed: {type(exc).__name__}: {exc}")

    def _collapse_block(self, path: str) -> str | None:
        """Close one open block, named by its path.

        An expanded block is drawn as a cluster rather than a node, so the front
        end identifies it by path rather than by a rendered node ID.

        :param path: Path of the block to close.
        :return: A success message, or ``None`` if that block was not open.
        """
        block = to_nodekey(path)
        if block not in self._expanded:
            return None
        self._expanded.discard(block)
        # Closing an outer block leaves anything expanded inside it unreachable,
        # so those go too rather than lingering as invisible state.
        for other in [nk for nk in self._expanded if nk.is_descendent_of(block)]:
            self._expanded.discard(other)
        return f"Closed {block}"

    @traitlets.observe("toggle_request")
    @_acknowledges
    def _toggle_requested(self, change: dict[str, Any]) -> None:
        """Open a collapsed block, close an open one, or collapse everything."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible") or not self._claim_request(request):
            return
        try:
            if request.get("collapse_all"):
                self._expanded.clear()
                success = "Collapsed all blocks"
            elif request.get("collapse"):
                closed = self._collapse_block(request["path"])
                if closed is None:
                    self._fail("Expand/collapse failed: that block is not open")
                    return
                success = closed
            elif self._view is None:
                self._fail("Expand/collapse failed: the graph is not rendered")
                return
            else:
                visible = self._id_to_visible[request["id"]]
                if visible not in self._view.composite_nodes:
                    self._fail("Expand/collapse failed: only collapsed blocks can be expanded")
                    return
                projected = len(self._view.node_index_map) - 1 + len(self._view.original_nodes[visible])
                if projected > self._max_rendered_nodes:
                    self._fail(
                        f"Expand/collapse failed: opening this block would render about {projected} nodes, "
                        f"over the limit of {self._max_rendered_nodes}. "
                        f"Pass max_rendered_nodes= to comp.widget() to raise it."
                    )
                    return
                block = self._full_visible_key(visible)
                self._expanded.add(block)
                success = f"Opened {block}"
            if self.refresh():
                self._set_status(success)
        except Exception as exc:
            LOG.debug("Loman widget expand/collapse request failed", exc_info=True)
            self._fail(f"Expand/collapse failed: {type(exc).__name__}: {exc}")

    @traitlets.observe("layout_request")
    @_acknowledges
    def _layout_requested(self, change: dict[str, Any]) -> None:
        """Change the Graphviz layout direction, then relayout."""
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible") or not self._claim_request(request):
            return
        rankdir = str(request.get("rankdir", "")).upper()
        if rankdir not in {"LR", "TB", "RL", "BT"}:
            self._fail(f"Layout failed: {rankdir or '(empty)'} is not a valid rankdir")
            return
        previous = self._canonical_rankdir
        self._canonical_rankdir = rankdir
        with self._own_write():
            self.rankdir = rankdir
        if self.refresh():
            self._set_status(f"Layout direction {rankdir}")
        else:
            # The relayout failed and left the old picture, so keep rankdir
            # agreeing with what is on screen rather than what was asked for.
            self._canonical_rankdir = previous
            with self._own_write():
                self.rankdir = previous

    def _resolve_focus(self, request: dict[str, Any]) -> NodeKey | None:
        """Resolve a focus request to the block it names, or the widget's root.

        :param request: A ``{"path": ...}`` climbing the breadcrumb, or a
            ``{"id": ...}`` descending into a rendered composite block.
        :return: The block to focus on, or the widget's own root when reset.
        :raises ValueError: If the request names somewhere outside the root.
        :raises KeyError: If the rendered ID is unknown.
        """
        if "path" in request:
            path = request["path"]
            if not path:
                return self._base_root if self._base_root is None else to_nodekey(self._base_root)
            target = to_nodekey(path)
            base = None if self._base_root is None else to_nodekey(self._base_root)
            if base is not None and target != base and not target.is_descendent_of(base):
                msg = f"{target} is not within this widget's root"
                raise ValueError(msg)
            return target
        assert self._view is not None  # noqa: S101
        visible = self._id_to_visible[request["id"]]
        if visible not in self._view.composite_nodes:
            msg = "only blocks can be focused"
            raise ValueError(msg)
        return self._full_visible_key(visible)

    @traitlets.observe("focus_request")
    @_acknowledges
    def _focus_requested(self, change: dict[str, Any]) -> None:
        """Re-root the view onto one block, or back to the widget's own root.

        Focusing drops every open expansion that is no longer under the new
        root, since those blocks are no longer on screen to close.
        """
        request = change["new"]
        if not request or not hasattr(self, "_id_to_visible") or not self._claim_request(request):
            return
        if self._view is None and "id" in request:
            self._fail("Focus failed: the graph is not rendered")
            return
        try:
            target = self._resolve_focus(request)
            self._root = target
            if target is None:
                self._expanded.clear()
            else:
                root_nk = to_nodekey(target)
                self._expanded = {nk for nk in self._expanded if nk.is_descendent_of(root_nk)}
            if self.refresh():
                self._set_status("Showing the whole graph" if target is None else f"Focused on {target}")
        except Exception as exc:
            LOG.debug("Loman widget focus request failed", exc_info=True)
            self._fail(f"Focus failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        """Unsubscribe from the computation and close the widget comm."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        super().close()
