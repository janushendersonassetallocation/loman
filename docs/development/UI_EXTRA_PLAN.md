# Design and implementation: `loman[ui]`

Status: implemented on this branch. The public subscription API is the core
boundary between `Computation` and the optional UI; the widget is its first
consumer.

This branch also **owns the shared extras convention** — the first
`[project.optional-dependencies]` section and the optional-import helper — which
the sibling `loman[airflow]` plan will build on rather than reinvent.

## Recommendation

Build one widget class, `loman.ui.ComputationWidget`, exposed as
`Computation.widget()`. It renders the *existing* graphviz picture as an SVG
string synced over a traitlet, overlays click handling in a small hand-written
vanilla JavaScript module, and drives a detail panel showing state, value repr,
timing, source, inputs and outputs, plus two write actions: edit a scalar input,
and compute.

The most important decision is **server-side layout**: keep graphviz as the
renderer and ship an SVG string, rather than sending a node and edge view model
to be laid out in the browser. That choice is what keeps this small enough to
land in this repo. It adds no JavaScript dependencies, no build step and no CDN
exposure, it guarantees the widget is identical to `comp.draw()`, and it inherits
the hierarchical cluster rendering that `to_pydot` in
`src/loman/visualization.py` already does correctly.

Its one real cost is that marimo's WebAssembly export cannot work, because there
is no `dot` binary in Pyodide. **That cost is unconfirmed as acceptable, so
phase 0 below is a spike to settle it before any code is written.**

## What this is not

Worth stating up front, because it is the obvious wrong turn: this is **not** a
standalone web server you browse to. A separate server process does not satisfy
"embed into a dashboard or applet", it cannot participate in a marimo or Jupyter
cell, and it forces a second rendering stack that will drift from
`visualization.py`. The widget runs in the notebook kernel and renders in the
cell, and nothing in this plan adds an HTTP server, a WebSocket, or a
`Computation.serve()` method.

There is a public in-process event subscription API, but no standalone network
WebSocket endpoint. A future server integration can translate the same
`ComputationEvent` stream to WebSocket messages without coupling the core
library to a web framework.

For the same reason, no front-end library is loaded from a CDN. Corporate
environments block it, offline use breaks, and it pins unaudited JavaScript with
no supply-chain review.

## Proposed API

```python
import loman

comp = build_portfolio()
w = comp.widget()

unsubscribe = comp.subscribe(lambda event: print(event.changed_nodes))
comp.compute_all()
unsubscribe()
```

`Computation.widget()` mirrors `Computation.draw()`'s signature so the two are
learnable together, adding only `editable: bool = True`.

In marimo:

```python
import marimo as mo

w = mo.ui.anywidget(comp.widget())
w
```

```python
w.value["selected"]
comp.v[w.value["selected"]]
```

That last line is the pattern to teach: **the widget navigates and lightly
controls; the real object stays in Python.** That is what makes it a dashboard
component rather than a walled garden.

In Jupyter, `comp.widget()` as the last expression in a cell renders directly,
with traits read via `w.selected` or `w.observe(...)`.

## Subscription architecture

`Computation.subscribe(callback)` returns an idempotent unsubscribe function.
The callback receives an immutable `ComputationEvent` containing the
computation, a monotonic revision, changed node keys, their final states, and a
`graph_changed` flag. Nested implementation calls are batched at the outermost
public mutation boundary, so operations such as `insert_many()` and
`compute_all()` publish one coherent event rather than exposing intermediate
states. Subscriber failures are logged and isolated from both the mutation and
other subscribers.

The widget subscribes automatically and unsubscribes when closed. State-only
events repaint existing SVG shapes and update selected-node detail without
running Graphviz again. Structural events perform a full layout refresh. Direct
mutation of the public `dag` remains outside this contract; `widget.refresh()`
is the explicit escape hatch for that legacy path.

## Widget architecture

State is synced as traitlets, not custom messages. Python to browser:
`graph_svg`, `node_ids`, `composite_ids`, `node_states`, `state_colors`,
`detail`, `status`, `editable`. Browser to Python: `selected`, `expanded`,
`edit_request`, `compute_request`.

Keeping everything in traits rather than messages lets compatible hosts recreate
widget state without Python running. Marimo's static exporter deliberately
rejects inlined `data:` JavaScript, so the demonstration notebook detects that
form and emits `graph_svg` directly as its safe read-only fallback. Every
browser-to-Python path remains a trait assignment, which makes the round-trip
testable without a browser.

The scaling rule is to **never serialize node values in bulk**. The full-graph
payload is the SVG plus two small string maps. Python computes a structure hash
over node paths, edges, root and expanded set, and skips resending `graph_svg`
when it is unchanged, so a `compute_all()` on a large graph re-sends only
`node_states` — a few kilobytes. The browser repaints by looking up each node's
state and setting the fill on the shape inside its group. No relayout, no node
jumping, no `dot` subprocess.

`detail` is populated lazily when `selected` changes and carries exactly one
node's data.

The browser needs no change to the graphviz output at all: each node group in
graphviz SVG carries a title element holding the DOT node name, which
`create_viz_dag` already assigns. All that is missing is the map from those names
to node keys, which currently exists as a local variable and is discarded.

## Design decisions

### Rendering location

Shipping graphviz SVG adds no JavaScript dependencies, works offline and behind a
content security policy, is identical to `comp.draw()`, and gets hierarchical
clusters correct for free. Against it: a `dot` subprocess per structural
relayout, no pan or zoom unless hand-rolled, a larger payload than a node list,
and no WebAssembly.

Client-side layout with elkjs or cytoscape gives smooth interaction and works in
WebAssembly, but pulls either a CDN or a bundler and npm lockfile into a repo
with no JavaScript tooling at all, and requires reimplementing cluster rendering
and the node formatter rules in JavaScript, where they will drift from
`visualization.py`.

Recommended: server-side SVG, with pan and zoom added later as a vendored single
file or by hand. **Conditional on the phase 0 spike.**

### Source of the node identifier map

`GraphView.refresh()` discards three things the UI needs: the original node set,
the composite node set, and `create_viz_dag`'s internal index map. Options are to
expose them as fields on `GraphView` (purely additive, blast radius of one
function), to re-derive them in `loman.ui` (duplicates a private invariant that
will silently break), or to emit graphviz `id` attributes (changes `draw()`
output, which tests assert on).

Recommended: expose them, landed as a standalone no-behaviour-change change
before any UI code.

### Scope of interaction

Read-only is too little to justify an extra, since `comp.draw()` already exists.
Full editing is too much. v1 ships exactly two mutations, each mapping to one
existing API call: edit a scalar input, offered only for input nodes holding a
scalar, mapping to `comp.insert`; and compute, mapping to `comp.compute_all()` or
`comp.compute(selected)` with exceptions left to land as error states, which is
the honest presentation.

Explicitly not in v1: adding or removing nodes, editing functions, setting stale,
pinning, or DataFrame grids.

### Value serialization for the wire

`ComputationSerializer` is built for round-tripping a whole computation, with
dill, function references and a format version. Pointing it at a UI wire format
would serialize things the browser must never see and would couple the wire
format to the persistence format's version.

Recommended: a small dedicated module for the wire format, handling scalars
(`int`, `float`, `str`, `bool`, `None`) in both directions and falling back to a
read-only `repr` for everything else. It needs a type discriminator so the
browser knows what it is holding, and explicit handling for NaN and infinity,
which are not valid JSON. Note the deliberate duplication in a module docstring,
and reuse the existing DataFrame and ndarray transformers if a grid is added
later rather than growing a second implementation.

### Long-running computation

Compute happens in the kernel inside an observer, so a slow graph freezes the
widget. A background thread is not an option: `Computation` is not thread-safe,
and making the widget mutate one off-thread would be a concurrency change to the
core library rather than a UI feature. Host-specific async would leave Jupyter
broken.

Recommended: synchronous, with a documented caveat that long computations should
be driven from a normal cell with the widget observing the result.

### Front-end asset form

anywidget accepts an inline string or a file path. A few hundred lines of
JavaScript inside a Python string would be invisible to every tool in this repo
and unreadable in review. A file path also unlocks anywidget's hot reload during
development.

Recommended: `.js` and `.css` files under `src/loman/ui/static/`, hand-written
vanilla ESM, no bundler, with dark mode via a media query since marimo and
JupyterLab both have dark themes.

## Layout and packaging

New modules under `src/loman/ui/`: the widget class, a view model builder, the
value wire-format module, and the static assets. A shared `src/loman/_extras.py`
optional-import helper lives outside `ui/` because the airflow extra will use it
too.

Modified: `visualization.py` to expose the identifier map;
`computeengine.py` to add `Computation.widget()` with a deferred import inside
the method body. `src/loman/__init__.py` must **not** import `loman.ui`; this is
the load-bearing constraint and deserves both a comment and a test.

Verify explicitly that the static assets land in the wheel, since the build
config copies the package tree but this has not been confirmed for non-Python
files.

The extras convention itself is specified in full below, because both this extra
and the airflow one depend on it.

## The extras convention

This branch owns this section. `loman` has no `[project.optional-dependencies]`
today, so this is the first one, and the airflow extra adds an entry against the
same convention rather than inventing a second.

### Declaration

```toml
[project.optional-dependencies]
ui = ["anywidget>=0.9", "traitlets>=5.14"]
airflow = []          # owned by the airflow plan; pin settled by its phase 0 spike

[tool.deptry.package_module_name_map]
anywidget = "anywidget"
traitlets = "traitlets"
```

`traitlets` is declared explicitly even though anywidget provides it
transitively, because the widget module imports it directly and deptry flags a
transitive dependency (DEP003) otherwise. The deptry map entries are required
because deptry otherwise guesses the module name from the distribution name.

Deliberately **no** `all = ["loman[ui,airflow]"]` aggregate. It looks tidy but it
means `pip install loman[all]` drags in Airflow for someone who wanted a widget,
and it has to be updated every time an extra is added. Add it later if anyone
asks.

### Ordering constraint, verified

**The declaration cannot land before the code that imports it.** `make deptry`
runs `deptry src/` in CI, and an extra declared with nothing importing it fails:

```text
pyproject.toml: DEP002 'anywidget' defined as a dependency but not used in the codebase
Found 1 dependency issue.
```

This was confirmed against the real deptry, not assumed. Two consequences:

- The `[project.optional-dependencies]` stanza ships in the **same** pull request
  as the first module that imports anywidget, not before it.
- If it ever needs to land earlier, the escape hatch is
  `[tool.deptry.per_rule_ignores] DEP002 = ["anywidget"]`, which was also
  verified to silence it. Treat that as temporary and delete it when the
  importing code arrives — a permanent ignore defeats the check.

The helper module below has no such constraint: it imports nothing outside the
standard library, so it can land on its own, fully tested, ahead of everything
else.

### The optional-import helper

`src/loman/_extras.py`, outside `ui/` because the airflow extra uses it too:

```python
def require(module: str, extra: str) -> ModuleType:
    """Import a module provided by an optional extra, or explain how to get it."""
```

On failure it raises `ImportError` chained from the original, with a message
naming both the missing module and the install command:

```text
'anywidget' is required for loman's 'ui' extra.
Install it with:  pip install 'loman[ui]'
```

Call it at the top of `src/loman/ui/__init__.py`, not inside each function, so a
missing extra fails once at import with a clear message rather than at some
arbitrary later call.

### What this must not break

`import loman` must not import anywidget, traitlets, ipywidgets or airflow. The
`Computation.widget()` method uses a deferred import inside the method body, and
`src/loman/__init__.py` never imports `loman.ui`. This is the load-bearing
constraint of the whole design and gets a subprocess test asserting the module is
absent from `sys.modules` after a bare `import loman`.

Weight warning: anywidget depends on ipywidgets, so `pip install loman[ui]` pulls
the whole ipywidgets stack. Acceptable for an extra, and exactly why it must not
be a base dependency.

### CI and lockfile

No workflow changes are needed. `make install` already runs
`uv sync --all-extras --all-groups`, so every extra installs automatically across
the whole matrix, in the docs build and in the release job.

Two consequences follow, and both need handling rather than noticing later:

- The **missing-extra path is never naturally exercised**, because CI always
  installs everything. It must be covered by monkeypatching `importlib`, which
  is a real assertion on the error message rather than a coverage pragma.
- Any extra that **cannot install on every matrix platform** breaks the
  everything-installs assumption. That does not bite the ui extra, but it does
  bite airflow, which has no Windows support — so the airflow plan needs a
  marker such as `; sys_platform != "win32"`, or its dependency moved to a
  dedicated group that the matrix does not sync. Worth settling when that extra
  lands, not after a red Windows job.

`uv.lock` must be regenerated in the same commit; the `uv-lock` pre-commit hook
and `uv lock --check` in `make install` both enforce it. Expect a large lockfile
diff, and keep it in its own commit so the reviewable change stays readable.

## Delivery phases

Phases 0–5 are represented by the implementation on this branch. Phase 6
remains a possible follow-up.

- **Phase 0 — spike, blocking.** Settle the WebAssembly question: is
  `marimo export html-wasm` actually reachable for this repo's notebooks, and
  does it matter? Measure the graphviz SVG payload for the largest example graph.
  If WebAssembly is required, the rendering decision flips and the cost profile
  changes completely. No code beyond throwaway measurement.
- **Phase 1 — expose `GraphView` internals.** No new dependency, no behaviour
  change. Unblocks everything and is reviewable in minutes.
- **Phase 2 — the optional-import helper.** `src/loman/_extras.py` and its tests,
  standard library only. No dependency declared and none imported, so this lands
  cleanly on its own and the airflow work can build on it immediately. Deliberately
  does **not** include the `[project.optional-dependencies]` stanza — see the
  ordering constraint above.
- **Phase 3 — read-only widget, with the extra declared.** The bulk of the value:
  view model, value wire format, widget class, static assets, click to inspect.
  The `[project.optional-dependencies]` stanza, deptry entries and lockfile regen
  ship **here**, in the same change as the first code that imports anywidget,
  because deptry rejects a declared-but-unused dependency. Also the
  wheel-contents check and the test that `import loman` does not import
  anywidget. Docs page and a marimo notebook. Shippable and useful on its own.
- **Phase 4 — collapse and expand.** Wire expansion requests to node
  transformations; click a composite node to drill into it. Structure-hash
  gating of the SVG lands here.
- **Phase 5 — interactivity.** Scalar edit, compute button, status trait, error
  and traceback display.
- **Phase 6, optional — validation panel.** Surface `validate()` and `plan()` as
  a second tab; their `to_df()` output renders as a table and the blocker
  information maps naturally onto node highlighting.

## Testing approach

Coverage measures Python only, so the JavaScript is invisible to the gate. That
is what makes the no-JavaScript-tooling recommendation affordable.

The view model and value modules are pure functions over a `Computation` and are
straightforward to cover with the existing fixtures, one of which already gives
nested blocks. The widget class needs no browser: setting a trait directly fires
the same observer the browser would, so the entire Python round-trip — selection,
edit, compute — is reachable in an ordinary test. The missing-extra branch, the
one path CI installs away, is covered by patching the import to raise. A
subprocess test asserts `import loman` does not pull in anywidget.

Recommended: no JavaScript build step and no JavaScript test runner. This repo
has no npm anywhere, and adding one would mean a new toolchain, lockfile,
dependency-update surface and CI job for a few hundred lines of vanilla ESM.
Instead, a Python contract test greps the JavaScript for trait names and asserts
they are all declared and writable, which catches the realistic failure — a trait
renamed on one side only — for no tooling. It should also assert that the state
colour map covers every member of `States`, so a new state can never render
unstyled.

## Resolved constraints and follow-ups

Resolved for this implementation:

- The notebook targets a live Python kernel and ordinary static HTML export.
  WebAssembly export is intentionally unsupported because Graphviz requires the
  `dot` binary; the exported HTML remains a useful read-only snapshot.
- The ipywidgets stack is isolated behind `loman[ui]` and is never imported by
  bare `import loman`.
- The AnyWidget module uses `render({model, el})` and returns a cleanup callback,
  remaining compatible with both Marimo 0.23's lifecycle arguments and newer
  AFM hosts that provide an abort signal. The built wheel contains both static
  assets.

Possible follow-ups:

- The graphviz SVG payload size for very large graphs was not measured, and no
  documented payload limit was found for anywidget or marimo. The structure-hash
  gating mitigates repeated sends, but a node-count threshold above which the
  widget refuses to auto-expand is probably needed.
- Getting a node's source can raise for lambdas defined in a REPL or restored
  from dill. The detail builder already degrades gracefully, but richer source
  metadata could be added later.
- Inserting a value can raise for missing or placeholder nodes; the edit observer
  routes these failures to the status trait. More specific, actionable error
  messages are a possible refinement.
- `Computation._repr_svg_` already renders a static picture for a bare `comp` in
  a cell. Users will reasonably expect that to become interactive; it should not,
  and the docs should say so.
- `GraphView.svg()` shells out to `dot`. A missing binary produces an opaque
  error, so the widget should catch it and put a readable message in the status
  trait.
