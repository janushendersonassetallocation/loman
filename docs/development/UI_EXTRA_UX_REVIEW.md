# UX review: `loman[ui]` widget

Findings from driving the widget through realistic tasks against a realistic
graph, before deciding what to build. Companion to
[UI_EXTRA_PLAN.md](UI_EXTRA_PLAN.md) and
[UI_EXTRA_AS_BUILT.md](UI_EXTRA_AS_BUILT.md).

## Method

A 32-node portfolio risk book in five blocks — `market`, `curves`, `positions`,
`risk`, `report` — with mixed states, one node that fails (a division by zero
reaching `risk/tail_ratio`), and one deliberately slow node (`risk/var_95`,
1.2 s). Driven in marimo 0.23.13 in a real browser, measuring the rendered DOM
rather than reasoning about the source.

32 nodes is deliberately modest. `max_rendered_nodes` defaults to 500.

Where a number below came from browser timing it is called out, because
background-tab throttling made in-page timers unreliable; the load-bearing
numbers are measured either in the kernel or from rendered DOM geometry.

## What works

Worth stating first, because it is the thing to protect.

**The collapsed overview is excellent.** Thirty-two nodes reduce to five blocks,
and the failing block is instantly visible as the one red rectangle. Answering
"what is this computation, and is anything broken?" took one glance and zero
interactions. That is the widget earning its place over `comp.draw()`.

**Selection identity is solid.** Clicking a node gives the real Loman name in
Python, blocks report their path, and the detail panel's inputs and outputs make
the local dependency structure clear.

**The repaint path is invisible in the right way.** Editing an input repainted
downstream nodes to STALE with no relayout and no flicker.

## Findings

### S1 — Drilling in makes the graph unreadable

The core interaction of the widget destroys its own legibility.

`.loman-graph svg` is styled `width: 100%; height: 100%`, so the SVG is always
scaled to fit the container instead of overflowing it. Graphviz emits a fixed
`viewBox`, so the more you expand, the smaller everything gets. Measured node
label height in the live DOM, container 396 px wide:

| Expanded blocks | Rendered nodes | Natural width | Scale | Label size |
|---|---:|---:|---:|---:|
| none (overview) | 5 | 162 pt | 1.00 | 18.7 px |
| `risk` | 14 | 697 pt | 0.57 | 10.6 px |
| `risk`, `positions` | 21 | 1116 pt | 0.35 | 6.6 px |
| `+ curves` | 25 | 1295 pt | 0.31 | 5.7 px |
| all five | 32 | 1411 pt | 0.28 | 5.2 px |

Opening a single block already takes labels to 10.6 px. Opening three takes them
to 5.7 px, roughly a third of readable size — on a graph an order of magnitude
smaller than the widget is willing to render.

Three things compound it:

- **The container never scrolls.** `overflow: auto` on `.loman-graph` never
  engages at the default zoom, because the SVG is forced to fit rather than
  overflow. There is no way to see part of the graph at full size.
- **Recovering takes eight clicks.** Zoom steps 25 % at a time from 100 %, and
  only at 300 % — the cap — do labels reach 24.5 px and the pane become
  scrollable.
- **"Fit" does not fit.** It is `setZoom(1)`, a reset to 100 %, which is the
  unreadable state. The label promises the one thing the control does not do.

This is the finding that matters. Everything else below is friction; this one
defeats the purpose.

### S2 — No indication that a computation is running

Compute is synchronous in the kernel by design, and `compute_all()` on this
graph measured **1.21 s** in the kernel — unbounded in general.

Throughout that time the widget shows nothing. `_compute_requested` calls
`self.computation.compute_all()` and only then sets a status, so by construction
there is no "computing" state. Sampled at the moment of the click: the compute
button was still enabled, no busy cursor, no `aria-busy`, and the status line
still displayed the **previous** action's message — "Opened risk" — which reads
as though the click did nothing.

A user will click again. The replay guard will not stop them, because a second
click is a genuinely new request.

### S2 — A block can be opened but not closed

`_expanded` supports only `add()` and `clear()`. Clicking a block opens it;
nothing closes it except "Collapse all". Having opened `risk`, `positions` and
`curves`, closing just `curves` means collapsing all three and reopening two.
Opening is one click; closing one thing is three.

### S2 — Error nodes render their traceback twice, once mangled

For `risk/tail_ratio` the detail panel shows a **Value** row containing the raw
repr of the `Error` object:

```text
Error(exception=ZeroDivisionError('float division by zero'), traceback='Traceback (most recent call last...
```

— an escaped, single-line, 419-character string wrapped across a narrow column —
and then renders the same traceback properly formatted below. The useful copy is
below the useless one.

### S2 — Hover and selection are indistinguishable

`g.node:hover` and `g.node.loman-selected` both apply
`stroke: #2563eb; stroke-width: 3px`. While the pointer is anywhere over the
graph you cannot tell which node is actually selected, which matters because the
detail panel and `widget.selected_name` follow selection, not hover.

### S3 — Ergonomics

- **No way to find a node by name.** With `max_rendered_nodes` at 500 and labels
  at 5 px, locating `curves/fwd_5y5y` means expanding, zooming and panning by
  eye. This is the gap that will bite first on a real book.
- **No sense of place after drilling in.** Nothing shows which blocks are open.
- **Status has no severity.** `Updated market/spot` and
  `Edit failed: ValueWireError...` are styled identically; `.loman-error` only
  applies to the traceback block.
- **The timing view is constructor-only.** Diagnosing a slow graph means
  rebuilding the widget with `colors="timing"` in Python, which is exactly the
  context switch the widget exists to avoid.
- **`Duration 1.204832 s`** — a fixed six decimal places, wrong at both ends of
  the range.
- **"revision 0"** occupies prime toolbar space and means nothing to a user.

### S4 — Minor

Zoom caps at 300 %, which will not be enough for the graphs the node limit
permits. No scroll-wheel zoom or drag-to-pan. Tracebacks include absolute
interpreter paths.

## Recommendation

Fix S1 first and alone, then re-evaluate. It is the difference between a widget
that works on toy graphs and one that works on real ones, and several S3 items
may look different once the graph is legible — a breadcrumb matters less if you
can see where you are.

Suggested order:

1. **Legibility.** Render the SVG at natural size and let the pane scroll; make
   "Fit" genuinely fit to the container and add a "100 %" actual-size control;
   raise the zoom cap; add scroll-wheel zoom and drag-to-pan.
2. **Progress.** Set a status and disable the controls *before* the work starts,
   so the synchronous freeze is at least explained.
3. **Per-block collapse**, so navigation is reversible.
4. **Error presentation**: suppress the `Error` repr in the Value row and keep
   the formatted traceback.
5. **Distinguish selection from hover.**
6. Then reconsider search, breadcrumb, status severity and the timing toggle.

## Caveat

These findings come from the author of the widget driving their own work. They
are grounded in measurement rather than opinion where possible, but a second
person doing the same tasks would likely find things this missed.
