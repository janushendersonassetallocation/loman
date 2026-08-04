"""Live Loman computation widget demonstration."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
#     "loman[ui]",
# ]
#
# [tool.uv.sources]
# loman = { path = "../..", editable = true }
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A live computation graph

    `loman[ui]` turns a `Computation` into a notebook-native dashboard. The graph
    is laid out by Loman's existing Graphviz renderer, so it is the same picture
    `comp.draw()` gives you — but selection, node details, state changes and
    lightweight controls are live.

    The widget follows its computation through `Computation.subscribe()`. It is
    never recreated, and no separate web server is running.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import loman

    return loman, mo, np, pd


@app.cell
def _(loman, np, pd):
    def build_book():
        """A small book holding every value type the widget can render."""
        computation = loman.Computation(metadata={"purpose": "widget feature tour"})

        # Scalars, editable directly in the detail panel.
        computation.add_node("market/spot", value=100.0)
        computation.add_node("position/quantity", value=12)
        computation.add_node("risk/shock", value=-0.08)

        # A DataFrame, editable cell by cell.
        computation.add_node(
            "market/holdings",
            value=pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT", "NVDA", "TSLA"],
                    "weight": [0.4, 0.3, 0.2, 0.1],
                    "shares": [1200, 800, 450, 300],
                    "active": [True, True, False, True],
                }
            ),
        )

        # A Series, also cell-editable.
        computation.add_node("market/tenors", value=pd.Series([0.5, 1.0, 2.0, 5.0, 10.0], name="years"))

        # An array, rendered as a table but deliberately read-only.
        computation.add_node("market/corr", value=np.array([[1.0, 0.35], [0.35, 1.0]]))

        # Nested configuration, rendered as a tree.
        computation.add_node(
            "market/config",
            value={
                "engine": {"seed": 42, "paths": 10_000, "antithetic": True},
                "limits": {"var": 5.0e6, "dv01": 25_000},
                "desks": ["rates", "credit", "equity"],
            },
        )

        computation.add_node(
            "position/market_value",
            lambda spot, quantity: spot * quantity,
            kwds={"spot": "market/spot", "quantity": "position/quantity"},
        )
        computation.add_node(
            "position/weighted",
            lambda holdings, mv: float((holdings["weight"] * mv).sum()),
            kwds={"holdings": "market/holdings", "mv": "position/market_value"},
        )
        computation.add_node(
            "risk/stressed_value",
            lambda mv, shock: mv * (1 + shock),
            kwds={"mv": "position/market_value", "shock": "risk/shock"},
        )
        computation.add_node(
            "risk/headroom",
            lambda config: config["limits"]["var"],
            kwds={"config": "market/config"},
        )

        # A node that fails, so the traceback panel has something to show.
        computation.add_node("risk/zero_budget", lambda: 0.0)
        computation.add_node(
            "risk/utilisation",
            lambda used, budget: used / budget,
            kwds={"used": "risk/stressed_value", "budget": "risk/zero_budget"},
        )

        computation.add_node(
            "report/stress_loss",
            lambda base, stressed: stressed - base,
            kwds={"base": "position/market_value", "stressed": "risk/stressed_value"},
        )
        computation.compute_all()
        return computation

    book = build_book()
    return (book,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The widget

    Blocks start collapsed, so a large graph opens as a readable overview. Try:

    | Action | How |
    |---|---|
    | Open a block | Click it |
    | Close one block | Click its **title** once open |
    | Close everything | **Collapse all** |
    | Inspect a node | Click it |
    | Zoom | **+** / **−**, or ctrl/⌘ with the wheel |
    | Whole graph | **Fit** — **1:1** returns to natural size |
    | Pan | Drag the background |

    `risk` is red because `risk/utilisation` divides by zero. Open the block,
    click the failing node, and the panel shows the traceback.
    """)
    return


@app.cell
def _(book, mo):
    widget = book.widget()
    widget_ui = mo.ui.anywidget(widget)
    return widget, widget_ui


@app.cell(hide_code=True)
def _(mo, widget_ui):
    # Marimo's static exporter rejects inlined `data:` JavaScript, so an exported
    # page falls back to the rendered SVG. A live kernel gets the real widget.
    _static_export = "data:text/javascript" in widget_ui.text
    mo.Html(widget_ui.widget.graph_svg) if _static_export else widget_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value types

    Open `market` and click through its nodes. The detail panel renders each
    according to what it holds:

    | Node | Value | Shown as | Editable |
    |---|---|---|---|
    | `market/spot` | `float` | A field | Yes |
    | `market/holdings` | `DataFrame` | A table | Yes, per cell |
    | `market/tenors` | `Series` | A table | Yes, per cell |
    | `market/corr` | `ndarray` | A table | No |
    | `market/config` | `dict` | A tree | No |

    Click a table cell to edit it: Enter commits, Escape cancels. A cell edit is
    an ordinary `comp.insert` of a **modified copy**, so downstream nodes go
    stale exactly as they would from Python, and column dtypes are enforced
    rather than coerced. Arrays stay read-only because NumPy coerces silently on
    assignment, so an edit could change a value without saying so.

    Large values are never sent whole: a table shows its first 50 rows and 20
    columns and says so. For a full grid, read the value out and use marimo's
    own editor — `mo.ui.data_editor(comp.v[widget.selected_name])`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The selection is real Python

    **The widget navigates and lightly controls; the real object stays in
    Python.** That is what makes it a dashboard component rather than a walled
    garden. Click any node and this panel follows.
    """)
    return


@app.cell(hide_code=True)
def _(book, mo, widget, widget_ui):
    # Reading the marimo wrapper's value makes this cell reactive to selection.
    _widget_state = widget_ui.value
    _selected = widget.selected_name
    if _selected is None:
        selection_panel = mo.md("**Nothing selected.** Click a node in the graph above.")
    else:
        selection_panel = mo.vstack(
            [
                mo.md(f"`widget.selected_name` → `{_selected!r}`"),
                mo.md(f"`comp.s[...]` → `{book.state(_selected)}`"),
                mo.md(f"`comp.i[...]` → `{book.get_inputs(_selected)}`"),
            ]
        )
    selection_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Following the computation

    The widget subscribes to the computation, so it repaints whether a change
    comes from the browser or from Python. Move a slider: the graph is mutated
    through the ordinary API and the widget follows.

    State-only changes repaint the existing picture in place — no relayout, no
    nodes jumping about. Only structural changes re-run Graphviz.
    """)
    return


@app.cell
def _(mo):
    quantity = mo.ui.slider(1, 25, value=12, step=1, label="Quantity")
    shock = mo.ui.slider(-0.30, 0.0, value=-0.08, step=0.01, label="Shock")
    auto_compute = mo.ui.checkbox(value=True, label="Compute after each change")
    return auto_compute, quantity, shock


@app.cell(hide_code=True)
def _(auto_compute, mo, quantity, shock):
    mo.hstack([quantity, shock, auto_compute], justify="start", gap=2)
    return


@app.cell
def _(auto_compute, book, quantity, shock, widget):
    # Referencing ``widget`` makes the subscription an explicit dependency of
    # this cell. The widget object is not recreated when a slider moves.
    _widget_revision_before_update = widget.revision
    book.insert_many([("position/quantity", int(quantity.value)), ("risk/shock", float(shock.value))])
    if auto_compute.value:
        book.compute_all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The subscription API

    `Computation.subscribe()` is public, and useful without the widget. Each
    outermost public mutation publishes one batched event, so `insert_many()`
    and `compute_all()` produce a single row each rather than one per node.
    """)
    return


@app.cell
def _(book, mo):
    get_event_log, set_event_log = mo.state([])

    def capture_event(event):
        """Record one batched computation event."""
        set_event_log(
            lambda events: [
                *events,
                {
                    "revision": event.revision,
                    "changed": len(event.changed_nodes),
                    "graph changed": event.graph_changed,
                    "nodes": ", ".join(sorted(str(node) for node in event.changed_nodes))[:80],
                },
            ]
        )

    unsubscribe = book.subscribe(capture_event)
    return get_event_log, unsubscribe


@app.cell(hide_code=True)
def _(get_event_log, mo):
    mo.ui.table(list(reversed(get_event_log()[-8:])), selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read-only mode

    `editable=False` removes the edit and compute controls. Navigation stays,
    because opening a block inspects the graph rather than mutating it. The
    guard is enforced in Python, not just hidden in the browser.

    This is a second widget on the same computation: both follow it, and both
    stay in step.
    """)
    return


@app.cell
def _(book, mo):
    read_only = book.widget(editable=False)
    read_only_ui = mo.ui.anywidget(read_only)
    return (read_only_ui,)


@app.cell(hide_code=True)
def _(mo, read_only_ui):
    _static_export = "data:text/javascript" in read_only_ui.text
    mo.Html(read_only_ui.widget.graph_svg) if _static_export else read_only_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Worth knowing

    **A bare `comp` stays a static picture.** Evaluating `comp` on its own
    renders an SVG through `_repr_svg_`, and always will. Interactivity is opt-in
    through `comp.widget()`.

    **Computation is synchronous.** Pressing a compute button runs in the kernel,
    on the thread handling the request, so a slow graph freezes the widget while
    it works. `Computation` is not thread-safe, so the widget deliberately does
    not compute on a background thread. For long computations, drive them from an
    ordinary cell and let the widget observe the result.

    **Large graphs.** Graphviz costs roughly 0.6 KiB and 0.6 ms per rendered
    node, both linear. State changes repaint in place and re-send only a small
    state map, about 32× smaller than the SVG. To stop one click hanging the
    kernel, the widget refuses to open a block that would put more than 500 nodes
    on screen; raise it with `comp.widget(max_rendered_nodes=2000)`.

    **State colours.** Loman's default state colours are shared with
    `comp.draw()` and are not colourblind-safe — `STALE` and `COMPUTABLE` are
    close. The legend in the footer names every state on screen so colour never
    carries the meaning alone. Pass `cmap=` for your own palette.

    **Closing.** ipywidgets keeps every open widget in a process-wide table, so
    call `widget.close()` when finished; that unsubscribes it and releases it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The API in one place

    ```python
    widget = comp.widget()          # mirrors comp.draw(), plus editable=
    widget.selected_name            # the real Loman name, not a browser ID
    widget.selected_names           # every member, for a collapsed block
    widget.refresh()                # escape hatch after mutating comp.dag directly
    widget.close()                  # unsubscribe and release

    unsubscribe = comp.subscribe(on_change)   # one batched event per mutation
    comp.revision                             # monotonic change counter
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
