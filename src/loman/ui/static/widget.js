function render({ model, el }) {
  const controller = new AbortController();
  const signal = controller.signal;
  el.classList.add("loman-widget");
  el.innerHTML = `
    <div class="loman-toolbar">
      <button data-action="compute-all">Compute all</button>
      <button data-action="collapse-all">Collapse all</button>
      <span class="loman-toolbar-divider" aria-hidden="true"></span>
      <button data-action="zoom-out" aria-label="Zoom out">−</button>
      <button data-action="fit">Fit</button>
      <button data-action="zoom-in" aria-label="Zoom in">+</button>
      <span class="loman-zoom">100%</span>
      <span class="loman-hint">Click a block to open it</span>
      <span class="loman-revision"></span>
    </div>
    <div class="loman-main">
      <div class="loman-graph"></div>
      <aside class="loman-detail"><p>Select a node to inspect it.</p></aside>
    </div>
    <div class="loman-status" role="status"></div>
  `;

  const graph = el.querySelector(".loman-graph");
  const detail = el.querySelector(".loman-detail");
  const status = el.querySelector(".loman-status");
  const revision = el.querySelector(".loman-revision");
  const zoomLabel = el.querySelector(".loman-zoom");
  const computeAll = el.querySelector('[data-action="compute-all"]');
  let sequence = 0;
  let zoom = 1;

  const requestId = () => {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `${Date.now()}-${Math.random()}-${++sequence}`;
  };

  const send = (trait, payload) => {
    model.set(trait, { ...payload, request_id: requestId() });
    model.save_changes();
  };

  const repaint = () => {
    if (!model.get("repaint_states")) return;
    const states = model.get("node_states");
    const colors = model.get("state_colors");
    graph.querySelectorAll("g.node").forEach((node) => {
      const id = node.querySelector("title")?.textContent;
      const shape = node.querySelector("ellipse, polygon, path");
      if (id && shape && colors[states[id]]) shape.setAttribute("fill", colors[states[id]]);
      node.classList.toggle("loman-selected", id === model.get("selected_id"));
    });
  };

  const applyZoom = () => {
    const svg = graph.querySelector("svg");
    if (svg) {
      svg.style.width = `${zoom * 100}%`;
      svg.style.height = `${zoom * 100}%`;
      svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    }
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  };

  const setZoom = (nextZoom) => {
    zoom = Math.min(3, Math.max(0.5, nextZoom));
    applyZoom();
  };

  const renderGraph = () => {
    graph.innerHTML = model.get("graph_svg");
    graph.querySelectorAll("g.node").forEach((node) => {
      const id = node.querySelector("title")?.textContent;
      if (!id) return;
      const composite = model.get("composite_ids").includes(id);
      node.tabIndex = 0;
      node.classList.toggle("loman-composite", composite);
      node.setAttribute("aria-label", composite ? "Open computation block" : "Select computation node");
      node.addEventListener("click", () => {
        if (composite) {
          send("toggle_request", { id });
        } else {
          model.set("selected_id", id);
          model.save_changes();
        }
      }, { signal });
      node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          if (composite) {
            send("toggle_request", { id });
          } else {
            model.set("selected_id", id);
            model.save_changes();
          }
        }
      }, { signal });
    });
    applyZoom();
    repaint();
  };

  const row = (label, value) => {
    const wrapper = document.createElement("div");
    wrapper.className = "loman-detail-row";
    const term = document.createElement("strong");
    term.textContent = label;
    const content = document.createElement("span");
    content.textContent = value ?? "—";
    wrapper.append(term, content);
    return wrapper;
  };

  const renderDetail = () => {
    const data = model.get("detail");
    detail.replaceChildren();
    if (!data?.id) {
      const empty = document.createElement("p");
      empty.textContent = "Select a node to inspect it.";
      detail.append(empty);
      repaint();
      return;
    }
    const title = document.createElement("h3");
    title.textContent = data.name;
    detail.append(title, row("State", data.state));
    if (data.composite) detail.append(row("Members", data.members.join(", ")));
    if (data.value) {
      const shown = data.value.kind === "repr" ? data.value.repr : String(data.value.value);
      detail.append(row("Value", shown));
    }
    if (data.timing) detail.append(row("Duration", `${data.timing.duration.toFixed(6)} s`));
    if (data.inputs?.length) detail.append(row("Inputs", data.inputs.join(", ")));
    if (data.outputs?.length) detail.append(row("Outputs", data.outputs.join(", ")));

    if (data.editable) {
      const form = document.createElement("form");
      form.className = "loman-edit";
      const input = document.createElement("input");
      const type = data.value.type;
      input.type = type === "bool" ? "checkbox" : (type === "int" || type === "float" ? "number" : "text");
      if (type === "float") input.step = "any";
      if (type === "bool") input.checked = data.value.value;
      else if (data.value.value !== null) input.value = data.value.value;
      const button = document.createElement("button");
      button.type = "submit";
      button.textContent = "Update input";
      form.append(input, button);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        let value = input.value;
        if (type === "bool") value = input.checked;
        else if (type === "int") value = Number.parseInt(input.value, 10);
        else if (type === "float") value = Number.parseFloat(input.value);
        else if (type === "none") value = input.value === "" ? null : input.value;
        send("edit_request", { id: data.id, value: { kind: "scalar", type, value } });
      }, { signal });
      detail.append(form);
    }

    const compute = document.createElement("button");
    compute.textContent = data.composite ? "Compute block" : "Compute node";
    compute.disabled = !model.get("editable");
    compute.addEventListener("click", () => send("compute_request", { id: data.id }), { signal });
    detail.append(compute);

    if (data.source) {
      const source = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Source";
      const pre = document.createElement("pre");
      pre.textContent = data.source;
      source.append(summary, pre);
      detail.append(source);
    }
    if (data.error) {
      const pre = document.createElement("pre");
      pre.className = "loman-error";
      pre.textContent = data.error;
      detail.append(pre);
    }
    repaint();
  };

  const renderStatus = () => { status.textContent = model.get("status"); };
  const renderRevision = () => { revision.textContent = `revision ${model.get("revision")}`; };
  const renderEditable = () => {
    computeAll.disabled = !model.get("editable");
    renderDetail();
  };

  computeAll.addEventListener(
    "click", () => send("compute_request", { all: true }), { signal }
  );
  el.querySelector('[data-action="collapse-all"]').addEventListener(
    "click", () => send("toggle_request", { collapse_all: true }), { signal }
  );
  el.querySelector('[data-action="zoom-out"]').addEventListener(
    "click", () => setZoom(zoom - 0.25), { signal }
  );
  el.querySelector('[data-action="fit"]').addEventListener(
    "click", () => setZoom(1), { signal }
  );
  el.querySelector('[data-action="zoom-in"]').addEventListener(
    "click", () => setZoom(zoom + 0.25), { signal }
  );

  model.on("change:graph_svg", renderGraph);
  model.on("change:node_states", repaint);
  model.on("change:selected_id", renderDetail);
  model.on("change:detail", renderDetail);
  model.on("change:status", renderStatus);
  model.on("change:revision", renderRevision);
  model.on("change:editable", renderEditable);
  const cleanup = () => {
    controller.abort();
    model.off("change:graph_svg", renderGraph);
    model.off("change:node_states", repaint);
    model.off("change:selected_id", renderDetail);
    model.off("change:detail", renderDetail);
    model.off("change:status", renderStatus);
    model.off("change:revision", renderRevision);
    model.off("change:editable", renderEditable);
  };

  renderGraph();
  renderEditable();
  renderStatus();
  renderRevision();
  return cleanup;
}

export default { render };
