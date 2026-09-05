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
widget.selected_name  # the real Loman name, not a browser ID
comp.v[widget.selected_name]  # the real value, of its real type
```

`selected_name` returns the actual node name, including non-string names — a
node called `1` and a node called `"1"` stay distinct. For a collapsed block it
returns the block path, and `selected_names` returns every member.

## What you can do in the widget

- **Click a node** to see its state, value, timing, source, inputs and outputs.
  For a failed node, the panel shows the traceback. The panel opens on that
  click and closes again on **Escape** or its **×**, so the graph has the full
  width whenever you are not reading a node.
- **Click a block** to open it where it stands, so its insides appear beside
  its neighbours with the edges between them drawn. Clicking anywhere along an
  open block's **title bar** closes it again, and **Collapse all** closes
  everything.
- **Alt-click a block** to isolate it instead: the block becomes the root and
  you see only its top layer, the way clicking a folder shows that folder. That
  is the move for a graph too large to open in place.
- The breadcrumb climbs back out a level at a time. Its first entry, **Reset**,
  returns to the whole graph and closes everything; its last carries a
  **Compute** for the block you are standing in.
- **Edit a scalar input** directly in the detail panel. This maps to
  `comp.insert`.
- **Build the graph itself** — add, redefine, rename and delete nodes — when the
  widget was created with `buildable=True`. See
  [Building the graph](#building-the-graph).
- **Compute** a node, a block, or the whole graph. This maps to `comp.compute`
  and `comp.compute_all`.
- **Zoom** with the toolbar controls, and scroll to pan.

Pass `editable=False` for a read-only widget. Opening and closing blocks stays
available, because navigating a graph does not mutate it.

```python
comp.widget(editable=False)
```

## Building the graph

Everything above changes the *values* in a computation. With
`buildable=True` the widget also builds the computation itself — adding
nodes, redefining them, renaming and deleting them:

```python
comp.widget(buildable=True, namespace=globals())
```

**+ Node** in the toolbar opens a form, and a selected node grows a
**Definition** section carrying **Edit**, **Rename** and **Delete**. A node is
one of two things:

- An **input**: a name, and optionally a scalar to start it off. With no value
  it is created `UNINITIALIZED`, which is the ordinary way to declare an input
  you will supply later. This is `comp.add_node(name, value=...)`.
- A **calculation**: a name, a list of inputs and a Python expression. Each
  input becomes a parameter of the function and an edge in the graph, and the
  expression becomes its body. This is
  `comp.add_node(name, func, kwds={...})`.

An input is written as a node name, and the parameter is named after the node —
`market/spot` arrives as `spot`. Where that will not do, because the last part
of the path is not a valid Python name or because two inputs would collide,
write `parameter=node` instead:

```text
price                  # the node "price", as the parameter `price`
market/spot            # the node "market/spot", as the parameter `spot`
futures=market/spot    # the same node, as the parameter `futures`
```

Editing a node is `add_node` again, which is what `add_node` already means: the
node keeps its name and its dependents and gets a new definition. Deleting one
that others still depend on leaves Loman's `PLACEHOLDER` behind rather than
removing it, and the status bar says which happened.

### Names are relative to where you are

A name is read against the block in focus, so a name typed while inside
`market` lands inside `market` — inputs included. Start a name with `/` to
reach out of the block:

```text
spot                   # market/spot, when focused on market
/rates/curve           # rates/curve, wherever you are
```

That is also how a block gets built: there is no "new block" button, because a
block is a naming convention. Add `market/spot` and `market/vol` from the top
level and the block appears around them.

### The namespace

An expression is compiled against `namespace`, so `namespace=globals()` is what
lets it use the notebook's own imports:

```python
import numpy as np

comp.widget(buildable=True, namespace=globals())  # np.sqrt(x) now works
```

Without one, only builtins are in scope. The function's globals stay pointed at
the live mapping, so an import made after a node was built is still visible to
it, and defining a node does not put its name into your namespace.

### Things worth knowing

**It runs code typed in a browser.** Defining a calculation node compiles an
expression and runs it in the kernel, with whatever `namespace` gives it. That
is why it is opt-in rather than part of `editable`, and why it should stay off
wherever the front end is not yours. `editable=False` refuses it either way.

**A node built here cannot be saved with its function.** Loman stores a function
by the module path it is importable from, and one compiled from a text box has
none — the same limitation a lambda has. `comp.save()` warns with
`UnserializableFunctionWarning`, and the reloaded node keeps its value but
cannot recompute. Promote a definition you mean to keep into a real function in
a cell.

**Its source is still visible.** The expression is registered where `inspect`
looks for it, so `comp.get_source(name)` and the panel's **Source** section show
what was typed rather than reporting that the source is unavailable.

**The form only offers to edit what it could put back.** A node whose function
was written in Python, or which takes positional or constant arguments, has no
**Edit** button: the form has no field for those, so offering to edit one would
be offering to replace it with something else. **Rename** and **Delete** stay
available on every node.

Pass `fit_on_render=True` to scale the graph to fit the pane on every render
instead of opening at natural size. It only ever shrinks — blowing a small graph
up to fill the pane is not what fit means — so it is worth turning on when the
shape of a large graph matters more than its labels.

```python
comp.widget(fit_on_render=True)
```

Everything else mirrors `comp.draw()`, so `root`, `colors`, `shapes`,
`collapse_all` and the Graphviz attribute dictionaries all work the same way.

## Value types

The detail panel renders what it can:

| Node value | Shown as | Editable |
|---|---|---|
| `int`, `float`, `str`, `bool`, `None` | A single field | Yes, on input nodes |
| `DataFrame`, `Series` | A table | Yes, cell by cell, on input nodes |
| `ndarray` (1-D or 2-D) | A table | No |
| `dict`, `list`, `tuple` | An expandable tree | No |
| Anything else | `repr` text | No |

Click a table cell to edit it; Enter commits and Escape cancels. A cell edit is
an ordinary `comp.insert` of a **modified copy**, so downstream nodes go stale
exactly as they would from Python, and the previous value is never mutated in
place — which matters because `Computation.copy()` is shallow and a value may be
shared.

Column types are enforced. Putting text in a float column is refused rather than
silently changing the column's dtype, and `bool` and `int` columns are kept apart.
Arrays are deliberately read-only: NumPy coerces on assignment, so an edit could
change a value without saying so.

### Large values stay in Python

The widget never sends a value in bulk. A table shows its **last** 50 rows and
first 20 columns and tells you the true shape; a tree is bounded by depth and
breadth. The tail is shown because rows are usually appended, so the recent end
is the interesting one. Editing only reaches what is on screen.

For the whole value, press **Show full**. The widget cannot render it — it is a
host-neutral AnyWidget, and calling marimo's renderers would make the extra
depend on marimo and drop Jupyter support — so it publishes the node name for
the notebook to render:

```python
_ = widget_ui.value  # react to the button
mo.ui.table(widget.full_view_value) if widget.full_view else None
```

`full_view` is a *label*, and a label is lossy — a node called `1` and a node
called `"1"` share one. So fetch through `full_view_value`, which resolves the
node that was actually asked for, or read `full_view_name` for the name with its
original type. That is the same guarantee `selected_name` gives, and the same
principle as everywhere else here: the widget navigates, and the real object
stays in Python.

## Observing changes yourself

The widget is built on a public API you can use directly:

```python
def on_change(event):
    print(event.revision, event.changed_nodes, event.graph_changed)


unsubscribe = comp.subscribe(on_change)
comp.compute_all()  # one event, not one per node
unsubscribe()
```

Each outermost public mutation publishes one batched `ComputationEvent`, so
`insert_many()` and `compute_all()` produce a single event rather than one per
internal state transition. Subscribers run synchronously, in registration order,
and a subscriber that raises is logged and skipped rather than breaking the
computation.

Anything with an object behind it — a `__self__`, whether the method is written
in Python or in C — is held weakly, so `comp.subscribe(obj.handler)` and
`comp.subscribe(events.append)` do not keep the owner alive. Plain functions,
lambdas, callable objects and `functools.partial` are held strongly until you
unsubscribe. The exception is an owner that supports no weak reference at all,
such as `list` and `dict`, which falls back to a strong reference.

## Things worth knowing

### It wears the host's colours

On load the widget samples the background of the page it is embedded in. That
brightness picks light or dark, which is more reliable than
`prefers-color-scheme`: a notebook's own theme toggle never changes the
operating system setting.

If the host also publishes a shadcn-style palette — `--background`,
`--foreground`, `--card`, `--border`, `--primary`, `--muted-foreground`,
`--radius`, as marimo does — the widget wears that palette directly, so it is
the same colours as the rest of the app rather than an approximation of them.
Those names are only adopted once the host's declared `--background` matches the
backdrop it actually paints; otherwise another design system owns them and the
widget keeps its own.

The graph is included in that. Graphviz would paint an opaque white page and
black ink into the SVG, so it is told `bgcolor="transparent"` and its ink —
edges, arrowheads, block borders and block titles — is retinted to the widget's
own, which is the host's wherever the host publishes a palette. Pass
`graph_attr={"bgcolor": ...}` if you want a page back.

Node fills are the exception: they carry state, so they stay exactly the colours
`comp.draw()` gives them. Each node's *label* is then inked black or white
against the fill it lands on, rather than assuming a white page — which also
fixes `UNINITIALIZED`, whose blue reads at 2.87:1 under black and 7.31:1 under
white.

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
comp.compute_all()  # widget updates when this returns
```

### Large graphs

Graphviz output costs roughly 0.6 KiB and 0.6 ms of layout per rendered node,
both linear. State changes repaint the existing picture in place and re-send only
a small state map — about 32× smaller than the SVG — so computing a large graph
is cheap. Structural changes re-run Graphviz.

Navigating does not. The last dozen layouts are kept, so going into a block and
coming back out again does not lay the same picture out twice — the trip back is
the one certain to happen. On a 36-block graph a round trip measures 0.2 ms
rather than 199 ms, which is the difference between instant and noticeable.
Anything that changes the graph's shape discards the stored layouts, and so does
`refresh()`. With `colors="timing"` nothing is kept, because there the picture
depends on values rather than on shape.

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
