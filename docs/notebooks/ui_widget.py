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

    `loman[ui]` turns a `Computation` into a notebook-native dashboard. The
    graph below is still laid out by Loman's existing Graphviz renderer, while
    node selection, details, state changes and lightweight controls remain live.

    Move either input slider. The computation is mutated through its normal
    Python API; the widget follows automatically through `Computation.subscribe()`.
    It is never recreated and no separate web server is running.
    """)
    return


@app.cell
def _():
    import marimo as mo

    import loman

    return loman, mo


@app.cell
def _(loman, mo):
    def build_portfolio():
        computation = loman.Computation(metadata={"purpose": "live UI demonstration"})
        computation.add_node("market/spot", value=100.0)
        computation.add_node("position/quantity", value=12)
        computation.add_node("risk/shock", value=-0.08)
        computation.add_node(
            "position/market_value",
            lambda spot, quantity: spot * quantity,
            kwds={"spot": "market/spot", "quantity": "position/quantity"},
        )
        computation.add_node(
            "risk/stressed_value",
            lambda spot, quantity, shock: spot * (1 + shock) * quantity,
            kwds={
                "spot": "market/spot",
                "quantity": "position/quantity",
                "shock": "risk/shock",
            },
        )
        computation.add_node(
            "report/stress_loss",
            lambda market_value, stressed_value: stressed_value - market_value,
            kwds={
                "market_value": "position/market_value",
                "stressed_value": "risk/stressed_value",
            },
        )
        computation.compute_all()
        return computation

    comp = build_portfolio()
    get_event_log, set_event_log = mo.state([])

    def capture_event(event):
        record = {
            "revision": event.revision,
            "nodes": ", ".join(sorted(str(node) for node in event.changed_nodes)),
            "graph changed": event.graph_changed,
            "states": ", ".join(f"{node}: {state.name}" for node, state in event.states.items()),
        }
        set_event_log(lambda events: [*events, record])

    comp.subscribe(capture_event)
    widget = comp.widget()
    widget_ui = mo.ui.anywidget(widget)
    return comp, get_event_log, widget, widget_ui


@app.cell
def _(mo):
    spot = mo.ui.slider(75, 125, value=100, step=1, label="Market spot")
    quantity = mo.ui.slider(1, 25, value=12, step=1, label="Quantity")
    auto_compute = mo.ui.checkbox(value=True, label="Compute after each change")
    return auto_compute, quantity, spot


@app.cell
def _(auto_compute, comp, quantity, spot, widget):
    # Referencing ``widget`` makes subscription setup an explicit dependency of
    # this cell. The object is not recreated when either input changes.
    _widget_revision_before_update = widget.revision
    comp.insert_many(
        [
            ("market/spot", float(spot.value)),
            ("position/quantity", int(quantity.value)),
        ]
    )
    if auto_compute.value:
        comp.compute_all()
    return


@app.cell(hide_code=True)
def _(auto_compute, mo, quantity, spot, widget_ui):
    _static_export = "data:text/javascript" in widget_ui.text
    _graph_output = mo.Html(widget_ui.widget.graph_svg) if _static_export else widget_ui
    mo.vstack(
        [
            mo.hstack([spot, quantity, auto_compute], justify="start", gap=2),
            mo.callout(
                mo.md(
                    "Click a node for its value, timing, source, inputs and outputs. "
                    "Click a collapsed block to open it. Scalar input nodes "
                    "can also be edited directly in the detail panel."
                ),
                kind="info",
            ),
            _graph_output,
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, widget, widget_ui):
    # Reading the Marimo wrapper value makes this cell reactive to browser-side
    # selection, while ``selected_name`` preserves the real Python node name.
    _widget_state = widget_ui.value
    _static_export = "data:text/javascript" in widget_ui.text
    selected = widget.selected_name
    if _static_export:
        selection_panel = mo.md(
            "**Static preview:** run this notebook in Marimo to select, edit, compute and expand nodes."
        )
    elif selected is None:
        selection_panel = mo.md("**Python selection:** click a node in the graph.")
    else:
        selection_panel = mo.md(
            f"**Python selection:** `{selected!r}`  \n"
            f"Use `comp.value(widget.selected_name)` to retrieve its real value."
        )
    selection_panel

    return


@app.cell(hide_code=True)
def _(get_event_log, mo):
    # The subscription callback publishes immutable snapshots through Marimo state,
    # so this panel rerenders after Python- or browser-initiated operations.
    recent_events = list(reversed(get_event_log()[-8:]))
    mo.vstack(
        [
            mo.md("## Subscription events"),
            mo.md(
                "Each public operation produces one batched event. `insert_many()` "
                "and `compute_all()` therefore add one row each, rather than one row "
                "for every internal node transition."
            ),
            mo.ui.table(recent_events, selection=None, pagination=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The underlying API

    ```python
    widget = comp.widget()
    unsubscribe = comp.subscribe(on_change)

    comp.insert("market/spot", 105.0)  # widget updates automatically
    comp.compute_all()                 # state colours and details update in place

    widget.selected_name               # the real Loman name, not a browser ID
    unsubscribe()
    ```

    A normal Marimo or Jupyter session has a live Python kernel, so editing and
    computation controls work. An exported snapshot uses the same rendered graph
    as a static SVG preview because there is no trusted widget runtime or Python
    process to mutate.
    """)
    return


if __name__ == "__main__":
    app.run()
