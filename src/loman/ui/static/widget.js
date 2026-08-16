/* Loman computation widget front end.
 *
 * Hand-written vanilla ESM: no bundler, no npm, no CDN. The graph itself is a
 * Graphviz SVG laid out in the kernel, so this module is responsible for
 * navigation, presentation and relaying requests back to Python.
 */

const ZOOM_MIN = 0.1;
const ZOOM_MAX = 8;
const ZOOM_STEP = 1.25;

const SVG_NS = "http://www.w3.org/2000/svg";

// How far the pointer must travel before a press counts as a drag rather than
// a click. A press that never moves has to stay a click, because panning makes
// the canvas inert to hit-testing and would swallow it.
const PAN_THRESHOLD = 4;

// Point-to-pixel, the ratio Graphviz's `pt` dimensions imply in CSS.
const PT_TO_PX = 96 / 72;

// Loman's roles mapped onto the shadcn-style names marimo and others publish.
// The graph is included: Graphviz is told to paint no background, and its ink
// is retinted from these, so the picture sits in the host's theme rather than
// on a white sheet laid over it.
const HOST_TOKEN_MAP = [
  ["--loman-chrome", "var(--background)"],
  ["--loman-panel", "var(--card, var(--background))"],
  ["--loman-raised", "var(--card, var(--background))"],
  ["--loman-ink", "var(--foreground, currentColor)"],
  ["--loman-ink-2", "var(--muted-foreground, var(--foreground))"],
  ["--loman-ink-3", "var(--muted-foreground, var(--foreground))"],
  ["--loman-line-strong", "var(--border, var(--input))"],
  ["--loman-field", "var(--background)"],
  ["--loman-accent", "var(--primary)"],
  ["--loman-radius", "var(--radius, 8px)"],
];

function render({ model, el }) {
  const controller = new AbortController();
  const signal = controller.signal;
  el.classList.add("loman-widget");
  el.innerHTML = `
    <div class="loman-toolbar">
      <div class="loman-group">
        <button data-action="compute-all" class="loman-primary">Compute all</button>
        <button data-action="collapse-all">Collapse all</button>
      </div>
      <div class="loman-group">
        <button data-action="zoom-out" class="loman-icon" aria-label="Zoom out" title="Zoom out">&minus;</button>
        <span class="loman-zoom" role="status" aria-label="Zoom level">100%</span>
        <button data-action="zoom-in" class="loman-icon" aria-label="Zoom in" title="Zoom in">+</button>
        <button data-action="fit" title="Scale the graph to fit the pane">Fit</button>
        <button data-action="actual" title="Show the graph at its natural size">1:1</button>
      </div>
      <div class="loman-group">
        <button data-action="layout" title="Toggle graph direction (left-to-right or top-to-bottom)">LR</button>
      </div>
      <div class="loman-group loman-build-group" hidden>
        <button data-action="add-node" title="Define a new node in this computation">+ Node</button>
      </div>
      <span class="loman-spacer"></span>
      <span class="loman-revision" title="Computation revision"></span>
    </div>
    <nav class="loman-breadcrumb" aria-label="Focus path" hidden></nav>
    <div class="loman-main">
      <div class="loman-canvas" tabindex="0">
        <div class="loman-stage"></div>
      </div>
      <aside class="loman-inspector" aria-label="Node inspector" hidden></aside>
      <aside class="loman-builder" aria-label="Node definition" hidden></aside>
    </div>
    <div class="loman-status" role="status" data-severity="idle">
      <span class="loman-status-dot" aria-hidden="true"></span>
      <span class="loman-status-text"></span>
      <span class="loman-spacer"></span>
      <ul class="loman-legend" aria-label="States in view"></ul>
    </div>
  `;

  const canvas = el.querySelector(".loman-canvas");
  const stage = el.querySelector(".loman-stage");
  const inspector = el.querySelector(".loman-inspector");
  const builder = el.querySelector(".loman-builder");
  const buildGroup = el.querySelector(".loman-build-group");
  const statusBar = el.querySelector(".loman-status");
  const statusText = el.querySelector(".loman-status-text");
  const legendList = el.querySelector(".loman-legend");
  const revision = el.querySelector(".loman-revision");
  const zoomLabel = el.querySelector(".loman-zoom");
  const breadcrumb = el.querySelector(".loman-breadcrumb");
  const buttons = (action) => el.querySelector(`[data-action="${action}"]`);

  let sequence = 0;
  let zoom = 1;
  let naturalSize = null;
  let busy = false;

  const requestId = () => {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `${Date.now()}-${Math.random()}-${++sequence}`;
  };

  /* ------------------------------------------------------- host integration */

  // The widget should look part of the page it is embedded in rather than a
  // box dropped onto it, so it samples the host's own background and wears it.
  //
  // That colour is also a better theme signal than prefers-color-scheme: a
  // notebook's own light/dark toggle does not touch the OS setting, and
  // :host-context is not supported everywhere. Luminance of the actual backdrop
  // is what the widget is really sitting on.
  // Resolved through a canvas rather than parsed. A computed background can
  // come back in any colour syntax the browser supports --- marimo reports
  // `color(srgb 0.09 0.11 0.10)`, whose components are 0-1 rather than 0-255 ---
  // and painting a pixel is the one way to normalise all of them.
  let colourProbe = null;
  const parseColour = (value) => {
    const text = String(value);
    if (!text || text === "transparent") return null;
    if (!colourProbe) {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 1;
      colourProbe = canvas.getContext("2d", { willReadFrequently: true });
    }
    if (!colourProbe) return null;
    // fillStyle silently keeps its old value if the colour will not parse, so
    // a known sentinel makes an unsupported syntax detectable.
    colourProbe.fillStyle = "#000000";
    colourProbe.fillStyle = text;
    colourProbe.clearRect(0, 0, 1, 1);
    colourProbe.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = colourProbe.getImageData(0, 0, 1, 1).data;
    return a === 0 ? null : { r, g, b };
  };

  const relativeLuminance = ({ r, g, b }) => {
    const channel = (value) => {
      const v = value / 255;
      return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  };

  // Walks out of the shadow root as well as up the DOM, since the widget is
  // mounted inside one and its host page is what supplies the backdrop.
  const hostBackground = () => {
    let node = el.parentNode;
    while (node) {
      if (node.nodeType === Node.ELEMENT_NODE) {
        const colour = parseColour(getComputedStyle(node).backgroundColor);
        if (colour) return colour;
      }
      node = node.parentNode ?? node.host ?? null;
    }
    return null;
  };

  // Resolves a host custom property to a real colour. Values arrive as
  // `light-dark(#fff, #181c1a)` or `var(...)` chains, which only the browser can
  // work out, so a throwaway probe in the host document does the resolving.
  const resolveHostToken = (name) => {
    const probe = document.createElement("span");
    probe.style.cssText = `position:absolute;visibility:hidden;color:var(${name})`;
    document.body?.appendChild(probe);
    const resolved = getComputedStyle(probe).color;
    probe.remove();
    return parseColour(resolved);
  };

  const sameColour = (a, b, tolerance = 12) =>
    a && b && Math.abs(a.r - b.r) <= tolerance && Math.abs(a.g - b.g) <= tolerance && Math.abs(a.b - b.b) <= tolerance;

  const applyHostTheme = () => {
    const backdrop = hostBackground();
    if (!backdrop) return;
    // Set the scheme first: the host's tokens are written with light-dark(), so
    // they only resolve correctly once this element has the right color-scheme.
    el.dataset.hostTheme = relativeLuminance(backdrop) < 0.4 ? "dark" : "light";

    el.style.setProperty("--loman-backdrop", `rgb(${backdrop.r}, ${backdrop.g}, ${backdrop.b})`);

    // Custom properties inherit through the shadow boundary, so the host's own
    // palette can be used directly --- but only once it is confirmed to mean
    // what the shadcn-style names suggest. If --background does not agree with
    // the backdrop actually painted, another design system owns those names and
    // the widget keeps its own palette.
    const adopt = sameColour(resolveHostToken("--background"), backdrop);
    el.dataset.hostTokens = String(adopt);
    // Assigned here rather than in a stylesheet rule: these are set on the
    // shadow host, where a rule inside the shadow root does not reliably win.
    // Each keeps a fallback, for a host that defines only some of them.
    for (const [ours, theirs] of HOST_TOKEN_MAP) {
      if (adopt) el.style.setProperty(ours, theirs);
      else el.style.removeProperty(ours);
    }
  };

  /* ------------------------------------------------------------ busy state */

  // Compute runs synchronously in the kernel, so the tab would otherwise sit
  // silent for the whole operation with the previous status still showing.
  // The browser sets its own optimistic busy state on send and clears it when
  // Python answers, which does not depend on the kernel flushing anything
  // before it blocks.
  // Both `busy` and `editable` disable controls, so they are applied together.
  // Clearing busy must not re-enable what a read-only widget never offered.
  const applyEnabledState = () => {
    const canMutate = model.get("editable") && !busy;
    buttons("compute-all").disabled = !canMutate;
    // Navigation is not mutation, so it stays available when read-only.
    buttons("collapse-all").disabled = busy;
    buttons("layout").disabled = busy;
    breadcrumb.querySelectorAll("button").forEach((node) => { node.disabled = busy; });
    // Except the one crumb that acts rather than navigates.
    breadcrumb.querySelectorAll("button.loman-crumb-action").forEach((node) => { node.disabled = !canMutate; });
    // Focusing a block navigates rather than mutates, so it stays live when
    // read-only; only the mutating controls follow `canMutate`.
    inspector.querySelectorAll("button:not(.loman-nav)").forEach((node) => { node.disabled = !canMutate; });
    inspector.querySelectorAll("button.loman-nav").forEach((node) => { node.disabled = busy; });
    inspector.querySelectorAll("input").forEach((node) => { node.disabled = busy; });
    // The node form writes Python into the kernel, so it needs the same
    // permission the mutating controls do, plus `buildable` on top.
    const canBuild = canMutate && model.get("buildable");
    buttons("add-node").disabled = !canBuild;
    builder.querySelectorAll("button, input, select, textarea").forEach((node) => { node.disabled = !canBuild; });
  };

  const setBusy = (isBusy, message) => {
    busy = isBusy;
    el.dataset.busy = String(isBusy);
    applyEnabledState();
    if (isBusy) {
      statusBar.dataset.severity = "busy";
      statusText.textContent = message;
    }
  };

  const send = (trait, payload, busyMessage) => {
    if (busyMessage) setBusy(true, busyMessage);
    model.set(trait, { ...payload, request_id: requestId() });
    model.save_changes();
  };

  /* ----------------------------------------------------------------- zoom */

  const readNaturalSize = (svg) => {
    const viewBox = svg.getAttribute("viewBox");
    if (viewBox) {
      const [, , w, h] = viewBox.split(/[\s,]+/).map(Number);
      if (w > 0 && h > 0) return { w: w * PT_TO_PX, h: h * PT_TO_PX };
    }
    const rect = svg.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  };

  // The SVG is sized in absolute pixels rather than percentages. A percentage
  // width makes the browser scale the graph down to fit the pane, which is what
  // made labels shrink to a few pixels as soon as a block was opened.
  const applyZoom = () => {
    const svg = stage.querySelector("svg");
    if (svg && naturalSize) {
      svg.style.width = `${Math.round(naturalSize.w * zoom)}px`;
      svg.style.height = `${Math.round(naturalSize.h * zoom)}px`;
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    }
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  };

  const setZoom = (next, anchor) => {
    const previous = zoom;
    zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
    if (zoom === previous) return;
    // Keep the point under the cursor (or the pane centre) where it was.
    const point = anchor ?? { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 };
    const ratio = zoom / previous;
    const left = (canvas.scrollLeft + point.x) * ratio - point.x;
    const top = (canvas.scrollTop + point.y) * ratio - point.y;
    applyZoom();
    canvas.scrollLeft = left;
    canvas.scrollTop = top;
  };

  const fitToPane = () => {
    if (!naturalSize) return;
    const padding = 24;
    const available = {
      w: Math.max(canvas.clientWidth - padding, 1),
      h: Math.max(canvas.clientHeight - padding, 1),
    };
    setZoom(Math.min(available.w / naturalSize.w, available.h / naturalSize.h, 1));
    canvas.scrollTo({ left: 0, top: 0 });
  };

  /* ------------------------------------------------------------ panning */

  let panning = null;
  canvas.addEventListener("pointerdown", (event) => {
    // Left button on empty canvas only; node clicks and text selection win.
    if (event.button !== 0 || event.target.closest("g.node, .loman-legend")) return;
    panning = { x: event.clientX, y: event.clientY, left: canvas.scrollLeft, top: canvas.scrollTop, dragging: false };
  }, { signal });

  canvas.addEventListener("pointermove", (event) => {
    if (!panning) return;
    const dx = event.clientX - panning.x;
    const dy = event.clientY - panning.y;
    if (!panning.dragging) {
      if (Math.abs(dx) < PAN_THRESHOLD && Math.abs(dy) < PAN_THRESHOLD) return;
      // Only now is this a drag rather than a click. Committing on pointerdown
      // is what broke closing an open block: panning sets pointer-events:none
      // across the canvas, so the title stopped being a target between press
      // and release and the click landed on the canvas instead.
      panning.dragging = true;
      canvas.classList.add("loman-panning");
      // Capture keeps the drag alive once the pointer leaves the pane. It is
      // an improvement on panning, not a requirement for it, and it throws
      // when the pointer is no longer active --- so it must not sit between
      // the gesture and the scroll it performs.
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        // The drag still works, it just stops at the edge of the pane.
      }
    }
    canvas.scrollLeft = panning.left - dx;
    canvas.scrollTop = panning.top - dy;
  }, { signal });

  const endPan = (event) => {
    if (!panning) return;
    const wasDragging = panning.dragging;
    panning = null;
    canvas.classList.remove("loman-panning");
    if (wasDragging && event.pointerId !== undefined && canvas.hasPointerCapture?.(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };
  canvas.addEventListener("pointerup", endPan, { signal });
  canvas.addEventListener("pointercancel", endPan, { signal });

  // Ctrl/Cmd + wheel zooms; a plain wheel keeps scrolling the host page, which
  // is the polite behaviour for a widget embedded in someone's notebook.
  canvas.addEventListener("wheel", (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const anchor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    setZoom(zoom * (event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP), anchor);
  }, { signal, passive: false });

  /* ----------------------------------------------------------- the graph */

  // Graphviz writes every label black, which was only ever safe because it also
  // painted a white page. It no longer paints one, and a node's label sits on
  // its state colour --- so each label takes whichever ink its own fill can
  // carry. Read from the shape rather than the state map, so this holds under
  // colours the widget does not choose, such as colors="timing".
  //
  // Decided by measuring both inks rather than by a luminance threshold. A
  // threshold has to be the WCAG crossover (~0.179), and guessing it wrong is
  // silent: at 0.4 this put white on the UPTODATE green, 2.89:1 where black
  // would have given 7.26:1.
  const INK_DARK = "#0b0b0b";
  const INK_LIGHT = "#ffffff";
  const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

  const inkNodeLabels = () => {
    const onDark = relativeLuminance({ r: 11, g: 11, b: 11 });
    const onLight = relativeLuminance({ r: 255, g: 255, b: 255 });
    stage.querySelectorAll("g.node").forEach((node) => {
      const fill = parseColour(node.querySelector("ellipse, polygon, path")?.getAttribute("fill"));
      const behind = fill ? relativeLuminance(fill) : onLight;
      const ink = contrast(behind, onDark) >= contrast(behind, onLight) ? INK_DARK : INK_LIGHT;
      node.querySelectorAll("text").forEach((label) => label.setAttribute("fill", ink));
    });
  };

  const repaint = () => {
    if (!model.get("repaint_states")) return;
    const states = model.get("node_states");
    const colors = model.get("state_colors");
    const selected = model.get("selected_id");
    stage.querySelectorAll("g.node").forEach((node) => {
      const id = node.dataset.lomanId;
      const shape = node.querySelector("ellipse, polygon, path");
      if (id && shape && colors[states[id]]) shape.setAttribute("fill", colors[states[id]]);
      node.classList.toggle("loman-selected", id === selected);
      if (id) node.setAttribute("aria-selected", String(id === selected));
    });
    inkNodeLabels();
    renderLegend();
  };

  // Names every state on screen. Loman's default state colours are not
  // colourblind-safe, so the graph must never be the only place the state is
  // said out loud.
  const renderLegend = () => {
    const states = model.get("node_states");
    const colors = model.get("state_colors");
    const present = [...new Set(Object.values(states))].sort();
    legendList.replaceChildren();
    for (const state of present) {
      const item = document.createElement("li");
      const swatch = document.createElement("span");
      swatch.className = "loman-swatch";
      swatch.style.background = colors[state] ?? "transparent";
      const label = document.createElement("span");
      label.textContent = state;
      item.append(swatch, label);
      legendList.append(item);
    }
  };

  // Clicking a block opens it where it stands, so its insides appear beside
  // its neighbours with the edges between them drawn. That keeps the block in
  // the context it belongs to, which is the reason to look inside one at all.
  //
  // Alt-click isolates it instead: the block becomes the root and the view
  // shows only its top layer. That is the move for a graph too big to open in
  // place, and the breadcrumb is how you come back out.
  //
  // Neither used to be a plain click --- both were reached by clicking a
  // different part of the same shape, its interior versus its border, which
  // nobody guesses and nothing on screen announces.
  const activate = (id, composite, event) => {
    if (busy) return;
    if (!composite) {
      model.set("selected_id", id);
      model.save_changes();
      return;
    }
    if (event?.altKey) {
      send("focus_request", { id }, "Isolating block…");
      return;
    }
    send("toggle_request", { id }, "Opening block…");
  };

  // An open block is a Graphviz cluster, not a node, so there is no shape to
  // click to close it. Its label is the handle instead: Graphviz titles the
  // group "cluster_<path>", and Python says which of those are open blocks
  // rather than plain `group=` clusters, which are not closeable.
  // SVG has no title attribute, so a hover tooltip is a child <title> element.
  const svgTooltip = (element, text) => {
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = text;
    element.appendChild(title);
  };

  // A block title is a thin thing to aim at --- "market" measures 58x21 --- and
  // it is the only way to close a block. A transparent rectangle behind it,
  // padded and stretched to the full width of the block, turns the whole title
  // bar into the target instead of the six characters of the word.
  const closeTarget = (cluster, label) => {
    let box;
    try {
      box = label.getBBox();
    } catch {
      return null; // Not laid out yet; the label itself still works.
    }
    const frame = cluster.querySelector("polygon");
    const span = frame?.getBBox?.();
    const pad = 7;
    const hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("x", span ? span.x : box.x - pad);
    hit.setAttribute("y", box.y - pad);
    hit.setAttribute("width", span ? span.width : box.width + pad * 2);
    hit.setAttribute("height", box.height + pad * 2);
    hit.setAttribute("fill", "transparent");
    hit.classList.add("loman-block-handle-hit");
    // Behind the label so the text still takes its own hover styling, and
    // before the member nodes so it can never sit over one of them.
    cluster.insertBefore(hit, label);
    return hit;
  };

  const wireOpenBlocks = () => {
    const open = new Set(model.get("expanded_paths") ?? []);
    stage.querySelectorAll("g.cluster").forEach((cluster) => {
      const title = cluster.querySelector("title")?.textContent ?? "";
      const path = title.startsWith("cluster_") ? title.slice("cluster_".length) : null;
      if (!path || !open.has(path)) return;
      const label = cluster.querySelector("text");
      if (!label) return;
      cluster.classList.add("loman-open-block");
      label.classList.add("loman-block-handle");
      label.setAttribute("tabindex", "0");
      label.setAttribute("role", "button");
      label.setAttribute("aria-label", `Close block ${path}`);
      svgTooltip(label, `Close ${path}`);
      const close = () => { if (!busy) send("toggle_request", { path, collapse: true }, "Closing block…"); };
      const onClick = (event) => { event.stopPropagation(); close(); };
      label.addEventListener("click", onClick, { signal });
      label.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); close(); }
      }, { signal });
      const hit = closeTarget(cluster, label);
      if (hit) {
        svgTooltip(hit, `Close ${path}`);
        hit.addEventListener("click", onClick, { signal });
      }
    });
  };

  const renderGraph = () => {
    const previousZoom = zoom;
    stage.innerHTML = model.get("graph_svg");
    const svg = stage.querySelector("svg");
    naturalSize = svg ? readNaturalSize(svg) : null;
    stage.querySelectorAll("g.node").forEach((node) => {
      const heading = node.querySelector("title");
      const id = heading?.textContent;
      if (!id) return;
      // Graphviz's rendered ID lives in <title>, which is also the only thing
      // SVG has by way of a tooltip --- so hovering any node showed "n2".
      // Moving the ID to a data attribute frees <title> to say something.
      // (SVGElement has no `title` property to assign, unlike HTMLElement.)
      node.dataset.lomanId = id;
      const composite = model.get("composite_ids").includes(id);
      node.tabIndex = 0;
      node.setAttribute("role", "option");
      node.classList.toggle("loman-composite", composite);
      node.setAttribute("aria-label", composite ? "Block, activate to open it" : "Computation node");
      heading.textContent = composite
        ? "Click to open this block · alt-click to isolate it"
        : "Click to inspect this node";
      node.addEventListener("click", (event) => activate(id, composite, event), { signal });
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(id, composite, event);
        }
      }, { signal });
    });
    wireOpenBlocks();
    zoom = previousZoom;
    applyZoom();
    repaint();
    // repaint() returns early unless colours come from state, and a fresh SVG
    // always arrives with Graphviz's black labels on it.
    inkNodeLabels();
  };

  /* -------------------------------------------------------- the inspector */

  const swatchFor = (state) => {
    const swatch = document.createElement("span");
    swatch.className = "loman-swatch";
    swatch.style.background = model.get("state_colors")[state] ?? "transparent";
    return swatch;
  };

  const section = (title, ...children) => {
    const wrapper = document.createElement("section");
    wrapper.className = "loman-section";
    if (title) {
      const heading = document.createElement("h4");
      heading.textContent = title;
      wrapper.append(heading);
    }
    wrapper.append(...children);
    return wrapper;
  };

  const refList = (names) => {
    const wrapper = document.createElement("div");
    wrapper.className = "loman-refs";
    for (const name of names) {
      const chip = document.createElement("span");
      chip.className = "loman-ref";
      chip.textContent = name;
      wrapper.append(chip);
    }
    return wrapper;
  };

  const formatDuration = (seconds) => {
    if (seconds >= 1) return `${seconds.toFixed(2)} s`;
    if (seconds >= 0.001) return `${(seconds * 1e3).toFixed(1)} ms`;
    return `${(seconds * 1e6).toFixed(0)} µs`;
  };

  const formatCell = (cell) => (cell === null || cell === undefined ? "" : String(cell));

  const isNumericKind = (kind) => kind === "int" || kind === "float";

  /* --------------------------------------------------------------- tables */

  // One editor exists at a time rather than an input per cell: a 50x20 window
  // would otherwise be a thousand live form controls.
  const openCellEditor = (td, data, windowRow, column) => {
    // The window is the tail of the frame, so the row on screen is not the row
    // in the value. Edits address absolute positions.
    const row = (data.row_offset ?? 0) + windowRow;
    if (busy || td.querySelector("input")) return;
    const kind = data.column_kinds[column];
    const original = td.textContent;
    const input = document.createElement("input");
    input.className = "loman-cell-input";
    input.type = kind === "bool" ? "checkbox" : isNumericKind(kind) ? "number" : "text";
    if (kind === "float") input.step = "any";
    if (kind === "bool") input.checked = original === "true";
    else input.value = original;
    input.setAttribute("aria-label", `${data.columns[column]}, row ${row}`);

    let settled = false;
    const cancel = () => {
      if (settled) return;
      settled = true;
      td.textContent = original;
    };
    const commit = () => {
      if (settled) return;
      settled = true;
      const value = kind === "bool" ? input.checked
        : kind === "int" ? Number.parseInt(input.value, 10)
        : kind === "float" ? Number.parseFloat(input.value)
        : input.value;
      td.textContent = formatCell(value);
      send(
        "edit_request",
        { id: data.nodeId, cell: { row, column }, value: { kind: "scalar", type: kind, value } },
        "Updating cell…",
      );
    };

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); commit(); }
      if (event.key === "Escape") { event.preventDefault(); cancel(); }
    }, { signal });
    input.addEventListener("blur", commit, { signal });
    td.replaceChildren(input);
    input.focus();
    if (input.type !== "checkbox") input.select();
  };

  const buildTable = (data) => {
    const table = document.createElement("table");
    table.className = "loman-table";
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.append(document.createElement("th"));
    data.columns.forEach((column, index) => {
      const th = document.createElement("th");
      th.textContent = column;
      th.title = `${column} (${data.column_kinds[index]})`;
      if (isNumericKind(data.column_kinds[index])) th.classList.add("loman-numeric");
      headRow.append(th);
    });
    head.append(headRow);

    const body = document.createElement("tbody");
    data.rows.forEach((row, rowIndex) => {
      const tr = document.createElement("tr");
      const indexCell = document.createElement("th");
      indexCell.scope = "row";
      indexCell.textContent = formatCell(data.index[rowIndex]);
      tr.append(indexCell);
      row.forEach((cell, columnIndex) => {
        const td = document.createElement("td");
        const kind = data.column_kinds[columnIndex];
        td.textContent = formatCell(cell);
        if (cell === null) td.classList.add("loman-null");
        if (isNumericKind(kind)) td.classList.add("loman-numeric");
        if (data.cellsEditable && kind !== "other") {
          td.classList.add("loman-editable-cell");
          td.tabIndex = 0;
          td.title = "Click to edit";
          td.addEventListener("click", () => openCellEditor(td, data, rowIndex, columnIndex), { signal });
          td.addEventListener("keydown", (event) => {
            if (event.key === "Enter") { event.preventDefault(); openCellEditor(td, data, rowIndex, columnIndex); }
          }, { signal });
        }
        tr.append(td);
      });
      body.append(tr);
    });
    table.append(head, body);

    const wrapper = document.createElement("div");
    wrapper.className = "loman-table-wrap";
    wrapper.append(table);

    const [rows, cols] = data.shape;
    const [shownRows, shownCols] = data.shown;
    const parts = [];
    if (shownRows < rows) parts.push(`last ${shownRows} of ${rows} rows`);
    if (cols !== undefined && shownCols < cols) parts.push(`first ${shownCols} of ${cols} columns`);
    if (parts.length) {
      const note = document.createElement("p");
      note.className = "loman-note";
      note.textContent = `Showing ${parts.join(", ")}.`;
      wrapper.append(note);
    }
    return wrapper;
  };

  /* ---------------------------------------------------------------- trees */

  const buildTreeNode = (node, depth) => {
    const label = node.key === undefined ? null : node.key;
    if (node.type === "leaf") {
      const line = document.createElement("div");
      line.className = "loman-tree-leaf";
      if (label !== null) {
        const key = document.createElement("span");
        key.className = "loman-tree-key";
        key.textContent = `${label}:`;
        line.append(key);
      }
      const value = document.createElement("span");
      value.className = node.value === null ? "loman-null" : "loman-tree-value";
      value.textContent = formatCell(node.value);
      line.append(value);
      return line;
    }
    const details = document.createElement("details");
    details.className = "loman-tree-branch";
    if (depth < 2) details.open = true;
    const summary = document.createElement("summary");
    const bracket = node.type === "dict" ? "{ }" : "[ ]";
    summary.textContent = label === null ? `${bracket} ${node.size}` : `${label}: ${bracket} ${node.size}`;
    details.append(summary);
    for (const child of node.children ?? []) details.append(buildTreeNode(child, depth + 1));
    if (node.truncated) {
      const more = document.createElement("div");
      more.className = "loman-note";
      more.textContent = node.children ? "More items not shown." : "Nested further than the panel shows.";
      details.append(more);
    }
    return details;
  };

  const inputTypeFor = (type) => {
    if (type === "bool") return "checkbox";
    return (type === "int" || type === "float") ? "number" : "text";
  };

  const readEditedValue = (input, type) => {
    if (type === "bool") return input.checked;
    if (type === "int") return Number.parseInt(input.value, 10);
    if (type === "float") return Number.parseFloat(input.value);
    if (type === "none") return input.value === "" ? null : input.value;
    return input.value;
  };

  const buildEditForm = (data) => {
    const type = data.value.type;
    const form = document.createElement("form");
    form.className = "loman-edit";
    const input = document.createElement("input");
    input.type = inputTypeFor(type);
    input.setAttribute("aria-label", `New value for ${data.name}`);
    if (type === "float") input.step = "any";
    if (type === "bool") input.checked = data.value.value;
    else if (data.value.value !== null) input.value = data.value.value;
    const button = document.createElement("button");
    button.type = "submit";
    button.textContent = "Update";
    form.append(input, button);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const value = readEditedValue(input, type);
      send("edit_request", { id: data.id, value: { kind: "scalar", type, value } }, "Updating…");
    }, { signal });
    return form;
  };

  const buildActions = (data) => {
    const wrapper = document.createElement("div");
    wrapper.className = "loman-actions";
    const compute = document.createElement("button");
    compute.textContent = data.composite ? "Compute block" : "Compute node";
    compute.addEventListener(
      "click", () => send("compute_request", { id: data.id }, "Computing…"), { signal },
    );
    wrapper.append(compute);
    if (data.composite) {
      // Focusing re-roots the graph on this block, so its own nested blocks
      // become the whole view. Navigation, not mutation, hence loman-nav.
      const focus = document.createElement("button");
      focus.className = "loman-nav";
      focus.textContent = "Focus";
      focus.title = "Show only this block, so its nested blocks fill the view";
      focus.addEventListener(
        "click", () => send("focus_request", { id: data.id }, "Focusing…"), { signal },
      );
      wrapper.append(focus);
    }
    return wrapper;
  };

  // Clearing the selection is what closes the inspector, so the graph gets the
  // whole width back. They are the same act, and only Python owns `detail`.
  const clearSelection = () => {
    model.set("selected_id", "");
    model.save_changes();
  };

  const buildHead = (data) => {
    const head = document.createElement("div");
    head.className = "loman-node-head";
    const name = document.createElement("h3");
    name.className = "loman-node-name";
    name.textContent = data.name;
    const badge = document.createElement("span");
    badge.className = "loman-badge";
    badge.append(swatchFor(data.state), document.createTextNode(data.state));
    const close = document.createElement("button");
    close.className = "loman-nav loman-close";
    close.textContent = "×";
    close.setAttribute("aria-label", "Close the inspector");
    close.title = "Close the inspector (Escape)";
    close.addEventListener("click", clearSelection, { signal });
    head.append(name, badge, close);
    return head;
  };

  // The panel only ever holds a window onto a big value, and this widget cannot
  // call the host's renderers. So "Show full" hands the node's name back to
  // Python and the notebook renders it with whatever it has.
  const buildShowFull = (data) => {
    const showing = model.get("full_view") === data.name;
    const button = document.createElement("button");
    button.className = "loman-nav loman-show-full";
    button.textContent = showing ? "Hide full" : "Show full";
    button.title = showing
      ? "Stop publishing this node for the notebook to render"
      : "Publish this node so the notebook can render all of it";
    button.addEventListener("click", () => send(
      "full_view_request",
      showing ? {} : { id: data.id },
      showing ? "Closing…" : "Opening full view…",
    ), { signal });
    return button;
  };

  const buildValue = (data) => {
    // An Error value reprs as the whole exception plus an escaped traceback.
    // The traceback gets its own section below, properly formatted, so showing
    // the repr here would be the same information twice and unreadable once.
    if (data.error) return null;
    const value = data.value;
    // Anything windowed or truncated is worth offering in full elsewhere.
    const partial = value.kind === "table" || value.kind === "tree" || value.kind === "repr";
    const heading = (title) => {
      const wrapper = document.createElement("div");
      wrapper.className = "loman-section-head";
      const label = document.createElement("h4");
      label.textContent = title;
      wrapper.append(label);
      if (partial && !data.composite) wrapper.append(buildShowFull(data));
      return wrapper;
    };
    if (value.kind === "table") {
      const [rows, cols] = value.shape;
      const size = cols === undefined ? `${rows}` : `${rows} × ${cols}`;
      // Carried on the payload so the cell editor knows where to send an edit.
      value.nodeId = data.id;
      value.cellsEditable = Boolean(data.cells_editable);
      return section(null, heading(`${value.type} (${size})`), buildTable(value));
    }
    if (value.kind === "tree") {
      const tree = document.createElement("div");
      tree.className = "loman-tree";
      tree.append(buildTreeNode(value.root, 0));
      return section(null, heading(`${value.type} (${value.root.size})`), tree);
    }
    const pre = document.createElement("pre");
    pre.className = "loman-value";
    pre.textContent = value.kind === "repr" ? value.repr : String(value.value);
    if (value.kind !== "repr") return section("Value", pre);
    return section(null, heading(`Value (${value.type})`), pre);
  };

  const buildMeta = (data) => {
    const rows = [];
    if (data.timing) rows.push(["Duration", formatDuration(data.timing.duration)]);
    if (data.timing) rows.push(["Computed", new Date(data.timing.end).toLocaleTimeString()]);
    if (!rows.length) return null;
    const list = document.createElement("dl");
    list.className = "loman-meta";
    for (const [term, value] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      list.append(dt, dd);
    }
    return section("Timing", list);
  };

  const buildError = (data) => {
    const pre = document.createElement("pre");
    pre.className = "loman-trace";
    pre.textContent = data.error;
    const wrapper = section("Traceback", pre);
    wrapper.classList.add("loman-is-error");
    return wrapper;
  };

  const buildSource = (source) => {
    const details = document.createElement("details");
    details.className = "loman-source";
    const summary = document.createElement("summary");
    summary.textContent = "Source";
    const pre = document.createElement("pre");
    pre.className = "loman-trace";
    pre.textContent = source;
    details.append(summary, pre);
    return section(null, details);
  };

  // The inspector is only earned by a click. Left standing empty it costs a
  // third of the width for a sentence, which is the difference between a
  // widget you embed in an app and one you tolerate in a notebook.
  const renderDetail = () => {
    const data = model.get("detail");
    inspector.replaceChildren();
    // The node form shares this column, so while it is up the inspector waits
    // its turn rather than fighting it for the space.
    const open = Boolean(data?.id) && formState === null;
    inspector.hidden = !open;
    el.dataset.inspector = open ? "open" : "closed";
    if (!open) {
      repaint();
      // The canvas just got wider, so a fitted graph is no longer fitted.
      fitIfRequested();
      return;
    }
    inspector.append(buildHead(data));
    if (data.composite) inspector.append(section(`Members (${data.members.length})`, refList(data.members)));
    const value = data.value ? buildValue(data) : null;
    if (value) inspector.append(value);
    if (data.error) inspector.append(buildError(data));
    const meta = buildMeta(data);
    if (meta) inspector.append(meta);
    if (data.inputs?.length) inspector.append(section("Inputs", refList(data.inputs)));
    if (data.outputs?.length) inspector.append(section("Outputs", refList(data.outputs)));
    if (data.editable) inspector.append(section("Edit value", buildEditForm(data)));
    inspector.append(section(null, buildActions(data)));
    // Building the graph is a separate permission, and a block is several
    // nodes rather than one, so it has no definition of its own to change.
    if (model.get("buildable") && !data.composite) {
      inspector.append(section("Definition", buildGraphActions(data)));
    }
    if (data.source) inspector.append(buildSource(data.source));
    applyEnabledState();
    repaint();
    // Likewise: opening the panel narrows the canvas.
    fitIfRequested();
  };

  /* -------------------------------------------------------- the node form */

  // Everything else here asks Python to do something to a node that exists.
  // This asks it to make one, which needs a name, a kind, and then either a
  // value or an expression --- more than a click can carry, hence a form. It
  // takes the inspector's column while it is open: two panels either side of
  // the graph would leave the graph with nothing.
  //
  // Names are read relative to the block in focus, so a name typed while
  // inside `market` lands inside it; a leading `/` names a node from the top.
  // That is Python's rule, and the form only has to say so.

  // Scalar types an input node can be seeded with, labelled for people rather
  // than for Python. "unset" is first because declaring an input you will fill
  // in later is the common case, and it is what UNINITIALIZED means.
  const SCALAR_TYPES = [
    ["unset", "Leave unset"],
    ["float", "Number"],
    ["int", "Whole number"],
    ["str", "Text"],
    ["bool", "True or false"],
    ["none", "None"],
  ];

  const NODE_NAMES_ID = "loman-node-names";

  // What the form was opened on: null when it is closed, otherwise the node it
  // is defining. Kept here rather than read back out of the DOM so that a
  // repaint of the graph underneath cannot disturb what is half typed.
  let formState = null;
  // A definition has been sent and Python has not answered yet. On success the
  // form closes; on failure it stays exactly as it was, because the message on
  // the status bar is about the text still in these fields.
  let awaitingBuild = false;

  const field = (text, control, hint) => {
    const wrapper = document.createElement("label");
    wrapper.className = "loman-field";
    const caption = document.createElement("span");
    caption.className = "loman-field-label";
    caption.textContent = text;
    wrapper.append(caption, control);
    if (hint) {
      const note = document.createElement("span");
      note.className = "loman-note";
      note.textContent = hint;
      wrapper.append(note);
    }
    return wrapper;
  };

  const selectFrom = (options, current) => {
    const select = document.createElement("select");
    select.className = "loman-select";
    for (const [value, text] of options) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      option.selected = value === current;
      select.append(option);
    }
    return select;
  };

  const nodeNameList = () => {
    const list = document.createElement("datalist");
    list.id = NODE_NAMES_ID;
    for (const name of model.get("node_names") ?? []) {
      const option = document.createElement("option");
      option.value = name;
      list.append(option);
    }
    return list;
  };

  // One row per input, rather than one comma-separated field, so each can
  // offer the graph's own node names as suggestions while it is typed.
  const inputRow = (value) => {
    const row = document.createElement("div");
    row.className = "loman-input-row";
    const input = document.createElement("input");
    input.className = "loman-input-ref";
    input.value = value ?? "";
    input.placeholder = "node, or parameter=node";
    input.setAttribute("list", NODE_NAMES_ID);
    input.setAttribute("aria-label", "Input node");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "loman-icon";
    remove.textContent = "×";
    remove.title = "Remove this input";
    remove.setAttribute("aria-label", "Remove this input");
    remove.addEventListener("click", () => row.remove(), { signal });
    row.append(input, remove);
    return row;
  };

  const focusLabel = () => {
    const trail = model.get("focus_trail") ?? [];
    return trail.length > 1 ? trail[trail.length - 1].label : "";
  };

  const buildNodeForm = (state) => {
    const form = document.createElement("form");
    form.className = "loman-form";

    const head = document.createElement("div");
    head.className = "loman-node-head";
    const title = document.createElement("h3");
    title.className = "loman-node-name";
    title.textContent = state.replace ? `Edit ${state.name}` : "New node";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "loman-close";
    close.textContent = "×";
    close.title = "Close the node form (Escape)";
    close.setAttribute("aria-label", "Close the node form");
    close.addEventListener("click", () => closeNodeForm(), { signal });
    head.append(title, close);

    const name = document.createElement("input");
    name.className = "loman-text";
    name.value = state.name ?? "";
    name.placeholder = "name, or block/name";
    name.setAttribute("aria-label", "Node name");
    const inside = focusLabel();
    const nameField = field(
      "Name",
      name,
      inside ? `Goes inside ${inside}. Start with / to name one from the top.` : "Use block/name to put it in a block.",
    );

    const kind = selectFrom([["input", "Input"], ["calc", "Calculation"]], state.kind);
    kind.setAttribute("aria-label", "Node kind");

    const valueType = selectFrom(SCALAR_TYPES, state.valueType ?? "unset");
    valueType.setAttribute("aria-label", "Value type");
    const value = document.createElement("input");
    value.className = "loman-text";
    value.setAttribute("aria-label", "Value");
    const applyValueType = () => {
      value.type = inputTypeFor(valueType.value);
      if (valueType.value === "float") value.step = "any";
      value.hidden = valueType.value === "unset" || valueType.value === "none";
    };
    if (state.valueType === "bool") value.checked = state.value === true;
    else if (state.value !== undefined && state.value !== null) value.value = state.value;
    applyValueType();
    valueType.addEventListener("change", applyValueType, { signal });

    const valueRow = document.createElement("div");
    valueRow.className = "loman-value-row";
    valueRow.append(valueType, value);
    const inputPane = document.createElement("div");
    inputPane.className = "loman-form-pane";
    inputPane.append(field("Value", valueRow, "An input node with no value starts UNINITIALIZED."));

    const inputs = document.createElement("div");
    inputs.className = "loman-input-rows";
    for (const entry of state.inputs?.length ? state.inputs : [""]) inputs.append(inputRow(entry));
    const addInput = document.createElement("button");
    addInput.type = "button";
    addInput.className = "loman-add-input";
    addInput.textContent = "+ Input";
    addInput.addEventListener("click", () => inputs.append(inputRow("")), { signal });

    const expression = document.createElement("textarea");
    expression.className = "loman-expression";
    expression.rows = 3;
    expression.value = state.expression ?? "";
    expression.placeholder = "price * quantity";
    expression.setAttribute("aria-label", "Expression");

    const calcPane = document.createElement("div");
    calcPane.className = "loman-form-pane";
    calcPane.append(
      field("Inputs", inputs, "Each becomes a parameter, named after the node unless you say otherwise."),
      addInput,
      field("Expression", expression, "Python, using the parameter names above."),
    );

    const applyKind = () => {
      inputPane.hidden = kind.value !== "input";
      calcPane.hidden = kind.value !== "calc";
    };
    applyKind();
    kind.addEventListener("change", applyKind, { signal });

    const actions = document.createElement("div");
    actions.className = "loman-actions";
    const submit = document.createElement("button");
    submit.type = "submit";
    submit.className = "loman-primary";
    submit.textContent = state.replace ? "Save" : "Add node";
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => closeNodeForm(), { signal });
    actions.append(submit, cancel);

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const payload = { action: "add", name: name.value, kind: kind.value, replace: Boolean(state.replace) };
      if (kind.value === "calc") {
        payload.inputs = [...inputs.querySelectorAll(".loman-input-ref")].map((node) => node.value);
        payload.expression = expression.value;
      } else if (valueType.value !== "unset") {
        payload.value = valueType.value === "none"
          ? { kind: "scalar", type: "none", value: null }
          : { kind: "scalar", type: valueType.value, value: readEditedValue(value, valueType.value) };
      }
      awaitingBuild = true;
      send("graph_request", payload, state.replace ? "Saving definition…" : "Adding node…");
    }, { signal });

    form.append(head, nodeNameList(), section(null, nameField, field("Kind", kind)), inputPane, calcPane);
    // Pinned to the bottom of the panel: a definition is taller than the pane
    // on any real graph, and hunting for the submit button by scrolling is not
    // a thing anyone should have to do.
    const footer = section(null, actions);
    footer.classList.add("loman-form-actions");
    form.append(footer);
    return { form, name };
  };

  const renderNodeForm = () => {
    builder.replaceChildren();
    builder.hidden = formState === null;
    el.dataset.builder = formState === null ? "closed" : "open";
    if (formState === null) {
      // The column just went back to the inspector, or to nothing at all.
      renderDetail();
      return;
    }
    const { form, name } = buildNodeForm(formState);
    builder.append(form);
    applyEnabledState();
    // The inspector shares this column, so opening the form hides it.
    inspector.hidden = true;
    fitIfRequested();
    name.focus();
    name.select();
  };

  const openNodeForm = (state) => {
    formState = { kind: "input", name: "", inputs: [], expression: "", ...state };
    awaitingBuild = false;
    renderNodeForm();
  };

  const closeNodeForm = () => {
    if (formState === null) return;
    formState = null;
    awaitingBuild = false;
    renderNodeForm();
  };

  // Called whenever Python answers. A rejected definition leaves the form
  // standing, because everything the message complains about is still in it.
  const settleNodeForm = () => {
    if (!awaitingBuild) return;
    awaitingBuild = false;
    if (model.get("status_severity") !== "error") closeNodeForm();
  };

  // Prefilling the form from the node's own definition is what makes editing
  // one editing rather than retyping. Python says whether it can be described
  // in these fields at all --- a function written in Python is not an
  // expression this form could put back, so it does not offer to.
  const editDefinition = (data) => {
    const definition = data.definition ?? {};
    openNodeForm({
      replace: true,
      name: definition.name ?? data.name,
      kind: definition.kind ?? "input",
      inputs: definition.inputs ?? [],
      expression: definition.expression ?? "",
      valueType: data.value?.kind === "scalar" ? data.value.type : "unset",
      value: data.value?.kind === "scalar" ? data.value.value : "",
    });
  };

  // Renaming is a single field, so it opens in place in the inspector rather
  // than taking over the column the way a whole definition does.
  const buildRenameForm = (data) => {
    const form = document.createElement("form");
    form.className = "loman-edit";
    const input = document.createElement("input");
    input.value = data.definition?.name ?? data.name;
    input.setAttribute("aria-label", `New name for ${data.name}`);
    const button = document.createElement("button");
    button.type = "submit";
    button.textContent = "Rename";
    form.append(input, button);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send("graph_request", { action: "rename", id: data.id, name: input.value }, "Renaming…");
    }, { signal });
    return form;
  };

  // Two presses rather than a confirm dialog: a notebook widget has no business
  // opening a modal, and an armed button says what the next click will do.
  const buildDeleteButton = (data) => {
    const button = document.createElement("button");
    button.className = "loman-danger";
    button.textContent = "Delete";
    button.title = `Delete ${data.name} from the computation`;
    let armed = false;
    button.addEventListener("click", () => {
      if (!armed) {
        armed = true;
        button.textContent = "Delete, really";
        button.dataset.armed = "true";
        return;
      }
      send("graph_request", { action: "delete", id: data.id }, "Deleting…");
    }, { signal });
    // Looking away disarms it, so a stray click later cannot land on a button
    // that is still counting the one before it.
    button.addEventListener("blur", () => {
      armed = false;
      button.textContent = "Delete";
      delete button.dataset.armed;
    }, { signal });
    return button;
  };

  const buildGraphActions = (data) => {
    const wrapper = document.createElement("div");
    wrapper.className = "loman-actions";
    if (data.definition?.editable) {
      const edit = document.createElement("button");
      edit.textContent = "Edit";
      edit.title = "Change this node's definition";
      edit.addEventListener("click", () => editDefinition(data), { signal });
      wrapper.append(edit);
    }
    const rename = document.createElement("button");
    rename.textContent = "Rename";
    let renameForm = null;
    rename.addEventListener("click", () => {
      if (renameForm) {
        renameForm.remove();
        renameForm = null;
        return;
      }
      renameForm = buildRenameForm(data);
      wrapper.after(renameForm);
      applyEnabledState();
      renameForm.querySelector("input").focus();
    }, { signal });
    wrapper.append(rename, buildDeleteButton(data));
    return wrapper;
  };

  /* ------------------------------------------------------------- wiring */

  // With the inspector closed until it is asked for, the status bar is the only
  // place left that can say what the graph responds to. It carries the hint
  // until Python has something of its own to report.
  const IDLE_HINT =
    "Click a node to inspect it · click a block to open it · drag to pan · ctrl or ⌘ with the wheel to zoom";

  const renderStatus = () => {
    setBusy(false);
    statusBar.dataset.severity = model.get("status_severity") || "idle";
    statusText.textContent =
      model.get("status") || (model.get("graph_svg") ? IDLE_HINT : "No graph to show.");
    settleNodeForm();
  };

  // The graph builder is opt-in, so its controls do not merely go inert when
  // it is off --- they are not there at all.
  const renderBuildControls = () => {
    const allowed = model.get("buildable");
    buildGroup.hidden = !allowed;
    if (!allowed) closeNodeForm();
    renderDetail();
    applyEnabledState();
  };

  const renderRevision = () => {
    const value = model.get("revision");
    revision.textContent = value ? `rev ${value}` : "";
  };

  const renderLayout = () => {
    const rankdir = model.get("rankdir") || "LR";
    const button = buttons("layout");
    button.textContent = rankdir;
    button.setAttribute("aria-label", `Graph direction ${rankdir}, activate to switch`);
  };

  // The trail runs from the widget's own root to the block in focus. Its first
  // entry is that root; anything beyond it is a block we have drilled into, so
  // the bar only earns its space once something is in focus.
  const renderBreadcrumb = () => {
    const trail = model.get("focus_trail") ?? [];
    const opened = (model.get("expanded_paths") ?? []).length > 0;
    breadcrumb.replaceChildren();
    // Shown whenever the view has been narrowed at all, not only once something
    // is in focus. Reset was previously invisible until you had already
    // focused, which is the one thing it exists to undo.
    breadcrumb.hidden = trail.length < 2 && !opened;
    if (breadcrumb.hidden) return;
    if (trail.length < 2) {
      const reset = document.createElement("button");
      reset.className = "loman-crumb";
      reset.textContent = trail[0]?.label ?? "Reset";
      reset.title = "Show the whole graph and close every open block";
      reset.addEventListener(
        "click", () => send("focus_request", { path: "" }, "Resetting…"), { signal },
      );
      breadcrumb.append(reset);
      applyEnabledState();
      return;
    }
    trail.forEach((entry, index) => {
      if (index > 0) {
        const sep = document.createElement("span");
        sep.className = "loman-crumb-sep";
        sep.textContent = "›";
        sep.setAttribute("aria-hidden", "true");
        breadcrumb.append(sep);
      }
      const last = index === trail.length - 1;
      if (last) {
        const here = document.createElement("span");
        here.className = "loman-crumb loman-crumb-current";
        here.setAttribute("aria-current", "location");
        here.textContent = entry.label;
        breadcrumb.append(here);
        // A block used to be selectable, and its panel is where "compute this
        // whole block" lived. Clicking a block now goes into it instead, so the
        // action follows you: it acts on the block you are standing in.
        const compute = document.createElement("button");
        compute.className = "loman-crumb loman-crumb-action";
        compute.textContent = "Compute";
        compute.title = `Compute every node in ${entry.label}`;
        compute.addEventListener(
          "click", () => send("compute_request", { path: entry.path }, "Computing…"), { signal },
        );
        breadcrumb.append(compute);
        return;
      }
      const crumb = document.createElement("button");
      crumb.className = "loman-crumb";
      crumb.textContent = entry.label;
      crumb.addEventListener(
        "click", () => send("focus_request", { path: entry.path }, "Focusing…"), { signal },
      );
      breadcrumb.append(crumb);
    });
    applyEnabledState();
  };

  const renderEditable = () => {
    renderDetail();
    applyEnabledState();
  };

  const fitIfRequested = () => {
    if (!model.get("fit_on_render") || !naturalSize) return;
    // Only shrink. Blowing a small graph up to fill the pane is not what
    // "fit" means here, and it would make every label enormous.
    const padding = 24;
    const scale = Math.min(
      (canvas.clientWidth - padding) / naturalSize.w,
      (canvas.clientHeight - padding) / naturalSize.h,
    );
    if (scale < 1) fitToPane();
  };

  const onGraphChanged = () => {
    setBusy(false);
    renderGraph();
    fitIfRequested();
  };

  // Escape closes the inspector, the same as its × button. Bound on the widget
  // rather than the canvas, so it still works from inside the panel itself.
  el.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    // Escape in a field cancels that edit. Closing a panel out from under
    // someone mid-keystroke would throw away what they were typing.
    if (event.target?.closest?.("input, textarea")) return;
    if (formState !== null) {
      event.preventDefault();
      closeNodeForm();
      return;
    }
    if (inspector.hidden) return;
    event.preventDefault();
    clearSelection();
  }, { signal });

  buttons("compute-all").addEventListener(
    "click", () => send("compute_request", { all: true }, "Computing…"), { signal },
  );
  buttons("collapse-all").addEventListener(
    "click", () => send("toggle_request", { collapse_all: true }, "Collapsing…"), { signal },
  );
  buttons("zoom-out").addEventListener("click", () => setZoom(zoom / ZOOM_STEP), { signal });
  buttons("zoom-in").addEventListener("click", () => setZoom(zoom * ZOOM_STEP), { signal });
  buttons("fit").addEventListener("click", fitToPane, { signal });
  buttons("actual").addEventListener("click", () => setZoom(1), { signal });
  buttons("layout").addEventListener("click", () => {
    const next = model.get("rankdir") === "LR" ? "TB" : "LR";
    send("layout_request", { rankdir: next }, "Changing layout…");
  }, { signal });
  buttons("add-node").addEventListener("click", () => openNodeForm({}), { signal });

  model.on("change:graph_svg", onGraphChanged);
  model.on("change:expanded_paths", wireOpenBlocks);
  model.on("change:expanded_paths", renderBreadcrumb);
  model.on("change:node_states", repaint);
  model.on("change:selected_id", renderDetail);
  model.on("change:detail", renderDetail);
  model.on("change:full_view", renderDetail);
  model.on("change:status", renderStatus);
  model.on("change:status_severity", renderStatus);
  // A request that changes nothing else still acknowledges, which is what
  // releases the optimistic busy state. renderStatus re-reads the model, so
  // the status shown falls back to whatever Python last set.
  model.on("change:ack", renderStatus);
  model.on("change:revision", renderRevision);
  model.on("change:editable", renderEditable);
  model.on("change:buildable", renderBuildControls);
  model.on("change:rankdir", renderLayout);
  model.on("change:focus_trail", renderBreadcrumb);
  model.on("change:fit_on_render", fitIfRequested);

  const cleanup = () => {
    controller.abort();
    themeWatcher.disconnect();
    model.off("change:graph_svg", onGraphChanged);
    model.off("change:expanded_paths", wireOpenBlocks);
    model.off("change:expanded_paths", renderBreadcrumb);
    model.off("change:node_states", repaint);
    model.off("change:selected_id", renderDetail);
    model.off("change:detail", renderDetail);
    model.off("change:full_view", renderDetail);
    model.off("change:status", renderStatus);
    model.off("change:status_severity", renderStatus);
    model.off("change:ack", renderStatus);
    model.off("change:revision", renderRevision);
    model.off("change:editable", renderEditable);
    model.off("change:buildable", renderBuildControls);
    model.off("change:rankdir", renderLayout);
    model.off("change:focus_trail", renderBreadcrumb);
    model.off("change:fit_on_render", fitIfRequested);
  };

  applyHostTheme();
  // A notebook theme toggle restyles the page rather than the widget, so the
  // backdrop is re-sampled whenever the host's own attributes change.
  const themeWatcher = new MutationObserver(applyHostTheme);
  for (const target of [document.documentElement, document.body].filter(Boolean)) {
    themeWatcher.observe(target, { attributes: true, attributeFilter: ["class", "style", "data-theme"] });
  }

  renderGraph();
  // clientWidth is 0 until the widget is laid out, so the first fit waits a frame.
  requestAnimationFrame(fitIfRequested);
  renderBuildControls();
  renderEditable();
  renderStatus();
  renderRevision();
  renderLayout();
  renderBreadcrumb();
  // Opens at natural size rather than fitted: a large graph fitted to a notebook
  // pane is unreadable, and "show me the whole shape" is one click away on Fit.
  return cleanup;
}

export default { render };
