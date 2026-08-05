/* Loman computation widget front end.
 *
 * Hand-written vanilla ESM: no bundler, no npm, no CDN. The graph itself is a
 * Graphviz SVG laid out in the kernel, so this module is responsible for
 * navigation, presentation and relaying requests back to Python.
 */

const ZOOM_MIN = 0.1;
const ZOOM_MAX = 8;
const ZOOM_STEP = 1.25;

// Point-to-pixel, the ratio Graphviz's `pt` dimensions imply in CSS.
const PT_TO_PX = 96 / 72;

// Loman's roles mapped onto the shadcn-style names marimo and others publish.
// --loman-canvas is deliberately absent: Graphviz paints a white background and
// black labels into the SVG itself, so that surface is not the host's to theme.
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
      <span class="loman-spacer"></span>
      <span class="loman-revision" title="Computation revision"></span>
    </div>
    <nav class="loman-breadcrumb" aria-label="Focus path" hidden></nav>
    <div class="loman-main">
      <div class="loman-canvas" tabindex="0">
        <div class="loman-stage"></div>
      </div>
      <aside class="loman-inspector" aria-label="Node inspector"></aside>
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
    // Focusing a block navigates rather than mutates, so it stays live when
    // read-only; only the mutating controls follow `canMutate`.
    inspector.querySelectorAll("button:not(.loman-nav)").forEach((node) => { node.disabled = !canMutate; });
    inspector.querySelectorAll("button.loman-nav").forEach((node) => { node.disabled = busy; });
    inspector.querySelectorAll("input").forEach((node) => { node.disabled = busy; });
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
    panning = { x: event.clientX, y: event.clientY, left: canvas.scrollLeft, top: canvas.scrollTop };
    canvas.classList.add("loman-panning");
    canvas.setPointerCapture(event.pointerId);
  }, { signal });

  canvas.addEventListener("pointermove", (event) => {
    if (!panning) return;
    canvas.scrollLeft = panning.left - (event.clientX - panning.x);
    canvas.scrollTop = panning.top - (event.clientY - panning.y);
  }, { signal });

  const endPan = (event) => {
    if (!panning) return;
    panning = null;
    canvas.classList.remove("loman-panning");
    if (event.pointerId !== undefined && canvas.hasPointerCapture?.(event.pointerId)) {
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

  const repaint = () => {
    if (!model.get("repaint_states")) return;
    const states = model.get("node_states");
    const colors = model.get("state_colors");
    const selected = model.get("selected_id");
    stage.querySelectorAll("g.node").forEach((node) => {
      const id = node.querySelector("title")?.textContent;
      const shape = node.querySelector("ellipse, polygon, path");
      if (id && shape && colors[states[id]]) shape.setAttribute("fill", colors[states[id]]);
      node.classList.toggle("loman-selected", id === selected);
      if (id) node.setAttribute("aria-selected", String(id === selected));
    });
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

  // A collapsed block is drawn as nested rectangles: an outer frame and an
  // inner fill. Clicking the fill opens the block; clicking the frame around it
  // isolates the block as the root instead, which is the same thing the
  // inspector's Focus button does but without hunting for the button.
  const onFrame = (node, event) => {
    const rects = [...node.querySelectorAll("polygon, ellipse, path")].map((s) => s.getBoundingClientRect());
    if (rects.length < 2) return false;
    // Graphviz emits the nested shapes inner-first, but that is an
    // implementation detail of the renderer rather than a promise, so the
    // smallest one is taken as the interior.
    const inner = rects.reduce((a, b) => (a.width * a.height <= b.width * b.height ? a : b));
    return (
      event.clientX < inner.left || event.clientX > inner.right ||
      event.clientY < inner.top || event.clientY > inner.bottom
    );
  };

  const activate = (id, composite, event) => {
    if (busy) return;
    if (!composite) {
      model.set("selected_id", id);
      model.save_changes();
      return;
    }
    if (event && onFrame(event.currentTarget, event)) {
      send("focus_request", { id }, "Focusing…");
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
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = text;
    element.appendChild(title);
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
      label.addEventListener("click", (event) => { event.stopPropagation(); close(); }, { signal });
      label.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); close(); }
      }, { signal });

      // An opened block keeps its frame, and the frame keeps its meaning:
      // clicking it isolates the block as the root, exactly as it does while
      // the block is still collapsed. Without this there is no way to focus
      // from an expanded view, and so no way to reach the breadcrumb.
      const frame = cluster.querySelector("polygon, path");
      if (!frame) return;
      frame.classList.add("loman-block-frame");
      frame.setAttribute("tabindex", "0");
      frame.setAttribute("role", "button");
      frame.setAttribute("aria-label", `Isolate block ${path}`);
      svgTooltip(frame, `Isolate ${path} — make it the root`);
      const focus = () => { if (!busy) send("focus_request", { path }, "Focusing…"); };
      frame.addEventListener("click", (event) => { event.stopPropagation(); focus(); }, { signal });
      frame.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); focus(); }
      }, { signal });
    });
  };

  const renderGraph = () => {
    const previousZoom = zoom;
    stage.innerHTML = model.get("graph_svg");
    const svg = stage.querySelector("svg");
    naturalSize = svg ? readNaturalSize(svg) : null;
    stage.querySelectorAll("g.node").forEach((node) => {
      const id = node.querySelector("title")?.textContent;
      if (!id) return;
      const composite = model.get("composite_ids").includes(id);
      node.tabIndex = 0;
      node.setAttribute("role", "option");
      node.classList.toggle("loman-composite", composite);
      node.setAttribute("aria-label", composite ? "Collapsed block, activate to open" : "Computation node");
      if (composite) node.title = "Click to open · click the frame to isolate this block";
      node.addEventListener("click", (event) => activate(id, composite, event), { signal });
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(id, composite, null);
        }
      }, { signal });
    });
    wireOpenBlocks();
    zoom = previousZoom;
    applyZoom();
    repaint();
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

  const buildHead = (data) => {
    const head = document.createElement("div");
    head.className = "loman-node-head";
    const name = document.createElement("h3");
    name.className = "loman-node-name";
    name.textContent = data.name;
    const badge = document.createElement("span");
    badge.className = "loman-badge";
    badge.append(swatchFor(data.state), document.createTextNode(data.state));
    head.append(name, badge);
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

  const renderEmpty = () => {
    const empty = document.createElement("p");
    empty.className = "loman-empty";
    empty.textContent = model.get("graph_svg")
      ? "Select a node to inspect it. Drag to pan, ctrl or ⌘ with the wheel to zoom."
      : "No graph to show.";
    inspector.append(empty);
  };

  const renderDetail = () => {
    const data = model.get("detail");
    inspector.replaceChildren();
    if (!data?.id) {
      renderEmpty();
      repaint();
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
    if (data.source) inspector.append(buildSource(data.source));
    applyEnabledState();
    repaint();
  };

  /* ------------------------------------------------------------- wiring */

  const renderStatus = () => {
    setBusy(false);
    statusBar.dataset.severity = model.get("status_severity") || "idle";
    statusText.textContent = model.get("status");
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
    breadcrumb.replaceChildren();
    breadcrumb.hidden = trail.length < 2;
    if (trail.length < 2) return;
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

  model.on("change:graph_svg", onGraphChanged);
  model.on("change:expanded_paths", wireOpenBlocks);
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
  model.on("change:rankdir", renderLayout);
  model.on("change:focus_trail", renderBreadcrumb);
  model.on("change:fit_on_render", fitIfRequested);

  const cleanup = () => {
    controller.abort();
    themeWatcher.disconnect();
    model.off("change:graph_svg", onGraphChanged);
    model.off("change:expanded_paths", wireOpenBlocks);
    model.off("change:node_states", repaint);
    model.off("change:selected_id", renderDetail);
    model.off("change:detail", renderDetail);
    model.off("change:full_view", renderDetail);
    model.off("change:status", renderStatus);
    model.off("change:status_severity", renderStatus);
    model.off("change:ack", renderStatus);
    model.off("change:revision", renderRevision);
    model.off("change:editable", renderEditable);
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
