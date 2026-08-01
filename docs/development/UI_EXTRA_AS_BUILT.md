# As built: `loman[ui]`

Companion to [UI_EXTRA_PLAN.md](UI_EXTRA_PLAN.md), which is kept as the design
that was agreed. This records what shipped, what phase 0 measured, and where the
implementation departed from the plan.

## Phase 0, settled

### WebAssembly export

Not supported, by decision rather than by accident. Server-side Graphviz layout
needs the `dot` binary, and there is none in Pyodide, so `marimo export
html-wasm` cannot produce a working widget. Ordinary static HTML export still
produces a useful read-only snapshot: the demonstration notebook detects a
static export and renders `graph_svg` directly.

The alternative — client-side layout with elkjs or cytoscape — was rejected for
the reasons in the plan: it would pull a bundler and an npm lockfile into a repo
with no JavaScript tooling, and would require reimplementing cluster rendering
and the node formatter rules in JavaScript, where they would drift from
`visualization.py`.

### Payload size, measured

The plan required this before committing to server-side layout. Measured on a
chain-of-layers graph with all nodes computed, one `dot` invocation per row:

| Nodes | SVG payload | Repaint payload | `dot` time |
|------:|------------:|----------------:|-----------:|
|    11 |     6.9 KiB |         0.2 KiB |     100 ms |
|   101 |    60.8 KiB |         1.9 KiB |     142 ms |
|   251 |   152.9 KiB |         4.8 KiB |     211 ms |
|   501 |   306.3 KiB |         9.7 KiB |     342 ms |
|  1001 |   618.7 KiB |        19.4 KiB |     611 ms |
|  2001 |  1251.3 KiB |        40.0 KiB |    1213 ms |

Everything is linear: about **0.62 KiB of SVG and 0.6 ms of `dot` time per
rendered node**. Two conclusions:

- The repaint path is worth having. The state map is roughly **32× smaller**
  than the SVG, so a `compute_all()` on a 500-node graph re-sends about 10 KiB
  instead of 300 KiB, and runs no subprocess.
- A ceiling is warranted, and 500 rendered nodes is the right place for it —
  about 300 KiB and a third of a second per relayout. `ComputationWidget` takes
  `max_rendered_nodes` (default 500) and refuses an expand request that would
  cross it, with a status message naming the parameter. It does not cap the
  initial view: what the caller asked to draw is drawn.

## What changed from the plan

### A public subscription API, which the plan did not have

The plan had the widget read state on demand. The implementation adds
`Computation.subscribe()`, `ComputationEvent` and `Computation.revision` so the
widget follows its computation automatically. This is the load-bearing addition,
and it is public API, so it carries its own guarantees:

- **No cost when unused.** With no subscribers the decorator is a straight
  pass-through. This matters: an earlier revision snapshotted the node set on
  every structural call and made graph construction quadratic — 1600 `add_node`
  calls went from 0.039 s to 1.206 s. `test_computation_graph_construction_stays_linear`
  guards against a regression.
- **Batched at the outermost public mutation.** `insert_many()` and
  `compute_all()` publish one event, not one per internal transition.
- **Ordered and idempotent.** Subscribers fire in registration order; the
  returned unsubscribe function tolerates repeat calls.
- **Weak on bound methods.** `comp.subscribe(obj.handler)` does not keep `obj`
  alive. Plain functions and lambdas are held strongly, because callers pass
  throwaway closures. Dead subscriptions are pruned on the next dispatch.
- **Re-entrancy is bounded, not forbidden.** A subscriber that mutates the
  computation produces a further event rather than recursing; after
  `_MAX_NOTIFICATION_CASCADES` rounds the loop is broken and logged.
- **Events are immutable.** `frozen=True`, a `frozenset` of changed nodes and a
  `MappingProxyType` of states. The `computation` attribute is a live handle by
  design, and says so.

`changed_nodes` is the set of nodes whose *state* changed. When `graph_changed`
is true it is not a complete description of the change — adding, deleting or
renaming a node, or altering a tag or style, need not change any state — and
consumers are told to re-read the graph instead. The alternative, sending the
before-and-after node sets, is what made construction quadratic.

### Aggregation lives in one place

The plan warned about the node formatter rules drifting if reimplemented. The
first implementation duplicated `ColorByState.format`'s ERROR → STALE →
unanimous → mixed ladder in the view model. Both now call
`loman.visualization.aggregate_states`, and a test asserts the widget's label
and the Graphviz fill colour agree for each case.

### Replay protection is symmetric

Derived traits (`graph_svg`, `node_states`, `composite_ids`, `detail`,
`revision`) are Python's to own, and an echo from a reconnecting browser model is
put back. Request traits carry a `request_id` nonce, and the widget now
remembers the recent ones, so a recreated front-end model cannot re-apply an edit
or a compute it still holds.

### `refresh()` reports failure

It returns `bool` rather than leaving callers to inspect the status string.

## Still open

- Long computations remain synchronous, as the plan recommended. `Computation`
  is not thread-safe, and making the widget mutate one off-thread would be a
  concurrency change to the core library rather than a UI feature. The widget
  docstring and the user guide both say to drive long computations from an
  ordinary cell.
- Phase 6, the validation and planning panel, is not implemented.
- Pan is handled by scrolling the container; zoom is a toolbar control. There is
  no drag-to-pan.
