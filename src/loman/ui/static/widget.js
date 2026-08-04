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
      <span class="loman-spacer"></span>
      <span class="loman-revision" title="Computation revision"></span>
    </div>
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
  const buttons = (action) => el.querySelector(`[data-action="${action}"]`);

  let sequence = 0;
  let zoom = 1;
  let naturalSize = null;
  let busy = false;

  const requestId = () => {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `${Date.now()}-${Math.random()}-${++sequence}`;
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
    inspector.querySelectorAll("button").forEach((node) => { node.disabled = !canMutate; });
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

  const activate = (id, composite) => {
    if (busy) return;
    if (composite) {
      send("toggle_request", { id }, "Opening block…");
    } else {
      model.set("selected_id", id);
      model.save_changes();
    }
  };

  // An open block is a Graphviz cluster, not a node, so there is no shape to
  // click to close it. Its label is the handle instead: Graphviz titles the
  // group "cluster_<path>", and Python says which of those are open blocks
  // rather than plain `group=` clusters, which are not closeable.
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
      const close = () => { if (!busy) send("toggle_request", { path, collapse: true }, "Closing block…"); };
      label.addEventListener("click", (event) => { event.stopPropagation(); close(); }, { signal });
      label.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); close(); }
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
      node.addEventListener("click", () => activate(id, composite), { signal });
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(id, composite);
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
  const openCellEditor = (td, data, row, column) => {
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
    if (shownRows < rows) parts.push(`${shownRows} of ${rows} rows`);
    if (cols !== undefined && shownCols < cols) parts.push(`${shownCols} of ${cols} columns`);
    if (parts.length) {
      const note = document.createElement("p");
      note.className = "loman-note";
      note.textContent = `Showing ${parts.join(", ")}. The whole value is in Python.`;
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

  const buildValue = (data) => {
    // An Error value reprs as the whole exception plus an escaped traceback.
    // The traceback gets its own section below, properly formatted, so showing
    // the repr here would be the same information twice and unreadable once.
    if (data.error) return null;
    const value = data.value;
    if (value.kind === "table") {
      const [rows, cols] = value.shape;
      const size = cols === undefined ? `${rows}` : `${rows} × ${cols}`;
      // Carried on the payload so the cell editor knows where to send an edit.
      value.nodeId = data.id;
      value.cellsEditable = Boolean(data.cells_editable);
      return section(`${value.type} (${size})`, buildTable(value));
    }
    if (value.kind === "tree") {
      const tree = document.createElement("div");
      tree.className = "loman-tree";
      tree.append(buildTreeNode(value.root, 0));
      return section(`${value.type} (${value.root.size})`, tree);
    }
    const pre = document.createElement("pre");
    pre.className = "loman-value";
    pre.textContent = value.kind === "repr" ? value.repr : String(value.value);
    const heading = value.kind === "repr" ? `Value (${value.type})` : "Value";
    return section(heading, pre);
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

  const renderEditable = () => {
    renderDetail();
    applyEnabledState();
  };

  const onGraphChanged = () => {
    setBusy(false);
    renderGraph();
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

  model.on("change:graph_svg", onGraphChanged);
  model.on("change:expanded_paths", wireOpenBlocks);
  model.on("change:node_states", repaint);
  model.on("change:selected_id", renderDetail);
  model.on("change:detail", renderDetail);
  model.on("change:status", renderStatus);
  model.on("change:status_severity", renderStatus);
  // A request that changes nothing else still acknowledges, which is what
  // releases the optimistic busy state. renderStatus re-reads the model, so
  // the status shown falls back to whatever Python last set.
  model.on("change:ack", renderStatus);
  model.on("change:revision", renderRevision);
  model.on("change:editable", renderEditable);

  const cleanup = () => {
    controller.abort();
    model.off("change:graph_svg", onGraphChanged);
    model.off("change:expanded_paths", wireOpenBlocks);
    model.off("change:node_states", repaint);
    model.off("change:selected_id", renderDetail);
    model.off("change:detail", renderDetail);
    model.off("change:status", renderStatus);
    model.off("change:status_severity", renderStatus);
    model.off("change:ack", renderStatus);
    model.off("change:revision", renderRevision);
    model.off("change:editable", renderEditable);
  };

  renderGraph();
  renderEditable();
  renderStatus();
  renderRevision();
  // Opens at natural size rather than fitted: a large graph fitted to a notebook
  // pane is unreadable, and "show me the whole shape" is one click away on Fit.
  return cleanup;
}

export default { render };
