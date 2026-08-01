# The Interactive Widget

`comp.draw()` gives you a picture. `comp.widget()` gives you the same picture,
laid out by the same Graphviz renderer, but live: it follows the computation as
you change it, and you can click a node to inspect it.

It needs the optional `ui` extra:

```bash
pip install 'loman[ui]'
```

Nothing in a bare `import loman` touches AnyWidget or ipywidgets — the extra is
imported lazily, the first time you call `comp.widget()`.

## Basic use

In Jupyter, the widget renders when it is the last expression in a cell:

```python
import loman

comp = build_portfolio()
comp.widget()
```

In marimo, wrap it so the notebook can react to it:

```python
import marimo as mo

widget = comp.widget()
w = mo.ui.anywidget(widget)
w
```

The widget follows the computation automatically. Mutate the computation
however you normally would, from any cell, and the picture updates:

```python
comp.insert("market/spot", 105.0)
comp.compute_all()
```

## Getting values back out

**The widget navigates and lightly controls; the real object stays in Python.**
That is what makes it a dashboard component rather than a walled garden. Read the
selection back and use the ordinary API:

```python
widget.selected_name          # the real Loman name, not a browser ID
comp.v[widget.selected_name]  # the real value, of its real type
```

`selected_name` returns the actual node name, including non-string names — a
node called `1` and a node called `"1"` stay distinct. For a collapsed block it
returns the block path, and `selected_names` returns every member.

## What you can do in the widget

- **Click a node** to see its state, value, timing, source, inputs and outputs.
  For a failed node, the panel shows the traceback.
- **Click a collapsed block** to open it.
- **Edit a scalar input** directly in the detail panel. Only input nodes holding
  an `int`, `float`, `str`, `bool` or `None` are editable — everything else is
  shown read-only, as a `repr`. This maps to `comp.insert`.
- **Compute** a node, a block, or the whole graph. This maps to `comp.compute`
  and `comp.compute_all`.
- **Zoom** with the toolbar controls, and scroll to pan.

Pass `editable=False` for a read-only widget. Opening and closing blocks stays
available, because navigating a graph does not mutate it.

```python
comp.widget(editable=False)
```

Everything else mirrors `comp.draw()`, so `root`, `colors`, `shapes`,
`collapse_all` and the Graphviz attribute dictionaries all work the same way.

## Observing changes yourself

The widget is built on a public API you can use directly:

```python
def on_change(event):
    print(event.revision, event.changed_nodes, event.graph_changed)

unsubscribe = comp.subscribe(on_change)
comp.compute_all()      # one event, not one per node
unsubscribe()
```

Each outermost public mutation publishes one batched `ComputationEvent`, so
`insert_many()` and `compute_all()` produce a single event rather than one per
internal state transition. Subscribers run synchronously, in registration order,
and a subscriber that raises is logged and skipped rather than breaking the
computation.

Bound methods are held weakly, so `comp.subscribe(obj.handler)` does not keep
`obj` alive. Plain functions and lambdas are held strongly until you unsubscribe.

## Things worth knowing

### A bare `comp` stays a static picture

Evaluating `comp` on its own in a cell renders a static SVG through
`_repr_svg_`, and always will. Interactivity is opt-in through `comp.widget()`,
so that displaying a computation never quietly starts a widget you did not ask
for.

### Computation is synchronous

Pressing a compute button runs the computation in the kernel, on the thread
handling the request. A slow graph will freeze the widget until it finishes.
`Computation` is not thread-safe, so the widget deliberately does not compute on
a background thread. For long computations, drive them from an ordinary cell and
let the widget observe the result:

```python
comp.compute_all()   # widget updates when this returns
```

### Large graphs

Graphviz output costs roughly 0.6 KiB and 0.6 ms of layout per rendered node,
both linear. State changes repaint the existing picture in place and re-send only
a small state map — about 32× smaller than the SVG — so computing a large graph
is cheap. Structural changes re-run Graphviz.

To keep one click from hanging the kernel, the widget refuses to open a block
that would put more than 500 nodes on screen. Raise it if you mean to:

```python
comp.widget(max_rendered_nodes=2000)
```

This does not cap the initial view — what you asked to draw is drawn.

### Colouring by anything other than state

`colors="state"` (the default) repaints shapes in place when states change. Any
other colouring, such as `colors="timing"`, depends on values rather than states,
so every change re-runs Graphviz — one `dot` subprocess per mutation. Fine for a
small graph, noticeable on a large one.

### Static HTML export

An exported notebook has no Python kernel, so editing and computing cannot work.
The graph still exports as a static SVG snapshot. WebAssembly export
(`marimo export html-wasm`) is not supported at all, because Graphviz layout
needs the `dot` binary and there is none in Pyodide.

### Closing

ipywidgets keeps every open widget in a process-wide table. Call `close()` when
you are finished with one; that unsubscribes it and releases it.

```python
widget.close()
```

## See also

- [Visualizing Computation Graphs](visualizing_computation_graph.md) for the
  static renderer the widget shares.
- The [Live Computation Widget notebook](../../../notebooks/ui_widget.html) for
  a worked example.
