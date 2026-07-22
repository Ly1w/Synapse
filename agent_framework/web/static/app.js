const ui = {
  connectionBadge: document.querySelector("#connectionBadge"),
  connectionLabel: document.querySelector("#connectionLabel"),
  modelChip: document.querySelector("#modelChip"),
  searchChip: document.querySelector("#searchChip"),
  newRunForm: document.querySelector("#newRunForm"),
  requestInput: document.querySelector("#requestInput"),
  launchButton: document.querySelector("#launchButton"),
  runList: document.querySelector("#runList"),
  runCount: document.querySelector("#runCount"),
  refreshRunsButton: document.querySelector("#refreshRunsButton"),
  emptyState: document.querySelector("#emptyState"),
  runView: document.querySelector("#runView"),
  runStatus: document.querySelector("#runStatus"),
  runIdLabel: document.querySelector("#runIdLabel"),
  revisionLabel: document.querySelector("#revisionLabel"),
  runTitle: document.querySelector("#runTitle"),
  runMeta: document.querySelector("#runMeta"),
  activeRequirement: document.querySelector("#activeRequirement"),
  topologyCanvas: document.querySelector("#topologyCanvas"),
  checkpointBody: document.querySelector("#checkpointBody"),
  copyResponseButton: document.querySelector("#copyResponseButton"),
  steerForm: document.querySelector("#steerForm"),
  steerInput: document.querySelector("#steerInput"),
  steerButton: document.querySelector("#steerButton"),
  cancelButton: document.querySelector("#cancelButton"),
  archiveButton: document.querySelector("#archiveButton"),
  activityList: document.querySelector("#activityList"),
  activityFilters: document.querySelector("#activityFilters"),
  drawerBackdrop: document.querySelector("#drawerBackdrop"),
  agentDrawer: document.querySelector("#agentDrawer"),
  drawerClose: document.querySelector("#drawerClose"),
  drawerKind: document.querySelector("#drawerKind"),
  drawerTitle: document.querySelector("#drawerTitle"),
  drawerId: document.querySelector("#drawerId"),
  drawerContent: document.querySelector("#drawerContent"),
  showRequirementsButton: document.querySelector("#showRequirementsButton"),
  requirementsModal: document.querySelector("#requirementsModal"),
  requirementsClose: document.querySelector("#requirementsClose"),
  requirementsList: document.querySelector("#requirementsList"),
  toastRegion: document.querySelector("#toastRegion"),
  mobileRunsButton: document.querySelector("#mobileRunsButton"),
  runsPanel: document.querySelector("#runsPanel"),
};

const state = {
  runs: [],
  selectedRunId: null,
  selectedRun: null,
  events: [],
  eventSequences: new Set(),
  eventCursor: 0,
  eventSource: null,
  eventFilter: "all",
  stateRefreshTimer: null,
  listRefreshTimer: null,
};

const STATUS_LABELS = {
  created: "CREATED",
  running: "RUNNING",
  completed: "COMPLETED",
  completed_retained: "CHECKPOINT",
  partial: "PARTIAL",
  blocked: "BLOCKED",
  failed: "FAILED",
  failed_retained: "FAILED · RETAINED",
  cancelled: "CANCELLED",
  cancelled_retained: "CANCELLED · RETAINED",
  archived: "ARCHIVED",
  retained: "RETAINED",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function shortId(value, size = 10) {
  if (!value) return "—";
  return value.length > size ? value.slice(0, size) : value;
}

function truncate(value, max = 90) {
  const text = String(value ?? "").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function timeAgo(timestamp) {
  if (!timestamp) return "—";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - Number(timestamp)));
  if (seconds < 5) return "now";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

function formatClock(timestamp) {
  if (!timestamp) return "—";
  return new Date(Number(timestamp) * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function toast(message, type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type === "error" ? "error" : ""}`;
  item.textContent = message;
  ui.toastRegion.append(item);
  window.setTimeout(() => item.remove(), 3600);
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.original = button.innerHTML;
    button.disabled = true;
    button.textContent = label;
  } else {
    button.disabled = false;
    if (button.dataset.original) button.innerHTML = button.dataset.original;
  }
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    ui.connectionBadge.classList.toggle("offline", !health.configured);
    ui.connectionLabel.textContent = health.configured ? "Runtime ready" : "Needs API config";
    ui.modelChip.textContent = `MODEL ${health.planner_model}`;
    const searchStatus = health.mcp?.status || "degraded";
    ui.searchChip.className = `model-chip search-chip ${searchStatus}`;
    ui.searchChip.textContent = searchStatus === "connected" ? "SEARCH READY" : `SEARCH ${searchStatus}`;
  } catch (error) {
    ui.connectionBadge.classList.add("offline");
    ui.connectionLabel.textContent = "Runtime offline";
    ui.modelChip.textContent = "MODEL —";
    ui.searchChip.className = "model-chip search-chip degraded";
    ui.searchChip.textContent = "SEARCH OFFLINE";
  }
}

async function loadRuns({ preserveSelection = true } = {}) {
  try {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];
    renderRunList();
    if (!preserveSelection || !state.selectedRunId) {
      const preferred = state.runs.find((run) => run.status === "running") || state.runs[0];
      if (preferred) await selectRun(preferred.run_id);
    }
  } catch (error) {
    ui.runList.innerHTML = `<div class="activity-empty"><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function renderRunList() {
  ui.runCount.textContent = String(state.runs.length);
  if (!state.runs.length) {
    ui.runList.innerHTML = `
      <div class="activity-empty">
        <p>No retained Runs yet. Launch the first mission above.</p>
      </div>`;
    return;
  }
  ui.runList.innerHTML = state.runs.map((run) => {
    const selected = run.run_id === state.selectedRunId ? "active" : "";
    const title = truncate(run.user_request, 54) || "Untitled run";
    return `
      <button class="run-item ${selected}" data-run-id="${escapeHtml(run.run_id)}">
        <span class="run-status-dot ${escapeHtml(run.status)}"></span>
        <span class="run-item-copy">
          <span class="run-item-title">${escapeHtml(title)}</span>
          <span class="run-item-meta">
            <span>${escapeHtml(STATUS_LABELS[run.status] || run.status)}</span>
            <span>${escapeHtml(timeAgo(run.updated_at))}</span>
          </span>
        </span>
      </button>`;
  }).join("");
}

async function selectRun(runId) {
  if (!runId) return;
  state.selectedRunId = runId;
  state.events = [];
  state.eventSequences.clear();
  state.eventCursor = 0;
  renderRunList();
  ui.emptyState.classList.add("hidden");
  ui.runView.classList.remove("hidden");
  ui.runsPanel.classList.remove("mobile-open");
  closeEventStream();
  renderActivity();
  try {
    await Promise.all([refreshSelectedRun(), loadInitialEvents()]);
    openEventStream();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function refreshSelectedRun() {
  if (!state.selectedRunId) return;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}`);
    if (run.run_id !== state.selectedRunId) return;
    state.selectedRun = run;
    const index = state.runs.findIndex((item) => item.run_id === run.run_id);
    if (index >= 0) state.runs[index] = run;
    else state.runs.unshift(run);
    renderRunList();
    renderSelectedRun();
  } catch (error) {
    if (error.message.includes("Unknown run")) {
      state.selectedRunId = null;
      state.selectedRun = null;
      ui.emptyState.classList.remove("hidden");
      ui.runView.classList.add("hidden");
    }
  }
}

function renderSelectedRun() {
  const run = state.selectedRun;
  if (!run) return;
  const status = run.status || "created";
  ui.runStatus.className = `status-pill ${status}`;
  ui.runStatus.textContent = STATUS_LABELS[status] || status.toUpperCase();
  ui.runIdLabel.textContent = `RUN ${shortId(run.run_id, 16).toUpperCase()}`;
  ui.revisionLabel.textContent = `REV ${run.revision || 0}`;
  ui.runTitle.textContent = run.user_request || "Untitled run";

  const topology = run.topology || { heads: [] };
  const nodeCount = (topology.heads || []).reduce((total, head) => total + (head.nodes || []).length, 0);
  const created = new Date(Number(run.created_at || 0) * 1000).toLocaleString();
  ui.runMeta.textContent = `${topology.heads?.length || 0} Heads · ${nodeCount} Nodes · started ${created}`;
  ui.activeRequirement.textContent = run.requirements?.at(-1) || run.user_request || "—";

  const isRunning = status === "running";
  const isArchived = status === "archived";
  const isResumable = run.resumable !== false;
  ui.cancelButton.disabled = !isRunning;
  ui.archiveButton.disabled = isRunning || isArchived;
  ui.steerInput.disabled = isArchived || !isResumable;
  ui.steerButton.disabled = isArchived || !isResumable;
  ui.steerInput.placeholder = isArchived
    ? "This Run is archived. Its history remains available for inspection."
    : !isResumable
      ? "Snapshot loaded after restart. Start a new Run to continue this work."
    : "Add a constraint, correct direction, or request another checkpoint…";

  renderTopology(topology);
  renderCheckpoint(run);
}

function agentCard(agent, kind, extraClass = "") {
  const status = agent.status || "retained";
  const role = kind === "MASTER" ? "Global owner" : agent.role || kind;
  const goal = kind === "MASTER"
    ? "Task graph · steering · global validation"
    : agent.goal || agent.outcome?.summary || "Awaiting contract";
  return `
    <button
      class="agent-card ${extraClass} ${escapeHtml(status)}"
      data-agent-id="${escapeHtml(agent.agent_id)}"
      data-agent-kind="${kind.toLowerCase()}"
    >
      <span class="agent-card-top">
        <span class="agent-kind">${kind}</span>
        <span class="agent-state">${escapeHtml(STATUS_LABELS[status] || status)}</span>
      </span>
      <span class="agent-role">${escapeHtml(role)}</span>
      <span class="agent-goal">${escapeHtml(goal)}</span>
    </button>`;
}

function renderTopology(topology) {
  const master = topology?.master;
  const heads = topology?.heads || [];
  if (!master) {
    ui.topologyCanvas.innerHTML = `
      <div class="topology-placeholder"><span class="mini-spinner"></span> Waiting for the task graph</div>`;
    return;
  }
  const headMarkup = heads.map((head) => {
    const nodes = head.nodes || [];
    const nodesMarkup = nodes.length
      ? nodes.map((node) => agentCard(node, "NODE", "node-card")).join("")
      : `<div class="empty-node">Head-owned work · no Node needed</div>`;
    return `
      <div class="head-column">
        ${agentCard(head, "HEAD")}
        <div class="node-stack">${nodesMarkup}</div>
      </div>`;
  }).join("");

  ui.topologyCanvas.innerHTML = `
    <div class="master-row">${agentCard(master, "MASTER", "master-card")}</div>
    ${heads.length ? `<div class="head-grid" style="--head-count:${heads.length}">${headMarkup}</div>` : `
      <div class="topology-placeholder"><span class="mini-spinner"></span> Master is forming the task graph</div>`}`;
}

function renderCheckpoint(run) {
  if (run.final_response) {
    ui.checkpointBody.textContent = run.final_response;
  } else {
    ui.checkpointBody.innerHTML = `
      <div class="checkpoint-waiting">
        <span class="typing-dots"><i></i><i></i><i></i></span>
        Master has not committed a checkpoint yet.
      </div>`;
  }
}

async function loadInitialEvents() {
  if (!state.selectedRunId) return;
  const payload = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/events?limit=1000`);
  addEvents(payload.events || []);
}

function closeEventStream() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function openEventStream() {
  if (!state.selectedRunId) return;
  closeEventStream();
  const source = new EventSource(
    `/api/runs/${encodeURIComponent(state.selectedRunId)}/stream?after=${state.eventCursor}`,
  );
  source.addEventListener("run_event", (message) => {
    try {
      addEvents([JSON.parse(message.data)]);
      scheduleStateRefresh();
    } catch {
      // Ignore one malformed event without closing the stream.
    }
  });
  source.onerror = () => {
    ui.connectionBadge.classList.add("offline");
    ui.connectionLabel.textContent = "Reconnecting";
  };
  source.onopen = () => {
    ui.connectionBadge.classList.remove("offline");
    ui.connectionLabel.textContent = "Runtime ready";
  };
  state.eventSource = source;
}

function scheduleStateRefresh() {
  window.clearTimeout(state.stateRefreshTimer);
  state.stateRefreshTimer = window.setTimeout(refreshSelectedRun, 180);
}

function addEvents(events) {
  let changed = false;
  for (const event of events) {
    const sequence = Number(event.sequence || 0);
    if (!sequence || state.eventSequences.has(sequence)) continue;
    state.eventSequences.add(sequence);
    state.eventCursor = Math.max(state.eventCursor, sequence);
    state.events.push(event);
    changed = true;
  }
  if (changed) {
    state.events.sort((a, b) => Number(a.sequence) - Number(b.sequence));
    if (state.events.length > 500) state.events = state.events.slice(-500);
    renderActivity();
  }
}

function classifyEvent(event) {
  const type = String(event.type || "");
  if (type.includes("tool")) return "tool";
  if (
    type.includes("update") || type.includes("discovery") || type.includes("guidance")
    || type.includes("message") || type.includes("budget") || type.includes("control")
  ) return "control";
  return "agent";
}

function eventPresentation(event) {
  const type = String(event.type || "event");
  const payload = event.payload || {};
  const map = {
    agent_registered: ["Agent registered", "A new owner entered the Run", "＋"],
    agent_phase_started: ["Agent phase started", payload.contract?.goal || "Execution phase opened", "▶"],
    agent_phase_finished: ["Agent settled", payload.summary || payload.status || "Outcome recorded", "✓"],
    agent_resumed: ["Agent resumed", payload.guidance || "Retained context reopened", "↻"],
    tool_call: [`Tool · ${payload.name || "call"}`, JSON.stringify(payload.arguments || {}), "T"],
    tool_result: [`Result · ${payload.name || "tool"}`, payload.result || "Result recorded", "↳"],
    message_sent: [
      `Message · ${payload.message_type || "sent"}`,
      payload.content?.text || payload.content?.description || payload.content?.question || "Routed through hierarchy",
      "→",
    ],
    user_update: ["User revision accepted", payload.text || "Requirement changed", "U"],
    user_update_applied: ["Master routed revision", payload.update || "Revision applied", "M"],
    guidance_applied: ["Head applied guidance", payload.guidance || "Contract updated", "H"],
    control_action: [`Control · ${payload.name || "action"}`, JSON.stringify(payload.result || {}), "C"],
    discovery_triaged: ["Discovery triaged", payload.discovery?.description || "Scope observation handled", "D"],
    head_discovery_triaged: ["Master triaged discovery", payload.discovery?.description || "Topology decision made", "D"],
    head_budget_exhausted: ["Head budget reached", "Remaining contracts were deferred", "!"],
    master_budget_exhausted: ["Run time budget reached", "Master committed the best available checkpoint", "!"],
  };
  return map[type] || [type.replaceAll("_", " "), summarizePayload(payload), "·"];
}

function summarizePayload(payload) {
  if (!payload || typeof payload !== "object") return String(payload || "");
  for (const key of ["summary", "text", "description", "guidance", "reason", "status"]) {
    if (payload[key]) return String(payload[key]);
  }
  const keys = Object.keys(payload);
  return keys.length ? keys.slice(0, 4).join(" · ") : "Event recorded";
}

function renderActivity() {
  const visible = state.events
    .filter((event) => state.eventFilter === "all" || classifyEvent(event) === state.eventFilter)
    .slice()
    .reverse();
  if (!visible.length) {
    ui.activityList.innerHTML = `
      <div class="activity-empty">
        <svg viewBox="0 0 48 48" aria-hidden="true"><path d="M8 24h8l4-10 7 22 5-12h8" /></svg>
        <p>Events will appear here as agents think, act, and report.</p>
      </div>`;
    return;
  }
  ui.activityList.innerHTML = visible.map((event) => {
    const category = classifyEvent(event);
    const [title, detail, icon] = eventPresentation(event);
    return `
      <article class="event-item ${category}">
        <div class="event-icon">${escapeHtml(icon)}</div>
        <div class="event-copy">
          <div class="event-title-row">
            <span class="event-title">${escapeHtml(title)}</span>
            <time class="event-time">${escapeHtml(formatClock(event.timestamp))}</time>
          </div>
          <div class="event-detail">${escapeHtml(truncate(detail, 150))}</div>
          <div class="event-source">${escapeHtml(shortId(event.source, 18))}${event.target ? ` → ${escapeHtml(shortId(event.target, 18))}` : ""}</div>
        </div>
      </article>`;
  }).join("");
}

async function openAgent(agentId, kind) {
  if (!state.selectedRunId || !agentId) return;
  ui.drawerKind.textContent = `${String(kind || "agent").toUpperCase()} SNAPSHOT`;
  ui.drawerTitle.textContent = "Loading…";
  ui.drawerId.textContent = agentId;
  ui.drawerContent.innerHTML = `<div class="topology-placeholder"><span class="mini-spinner"></span> Loading retained state</div>`;
  ui.drawerBackdrop.classList.remove("hidden");
  ui.agentDrawer.classList.add("open");
  ui.agentDrawer.setAttribute("aria-hidden", "false");
  try {
    const snapshot = await api(
      `/api/runs/${encodeURIComponent(state.selectedRunId)}/agents/${encodeURIComponent(agentId)}`,
    );
    ui.drawerTitle.textContent = snapshot.role || kind || "Agent";
    renderAgentSnapshot(snapshot);
  } catch (error) {
    ui.drawerTitle.textContent = "Unavailable";
    ui.drawerContent.innerHTML = `<div class="detail-value">${escapeHtml(error.message)}</div>`;
  }
}

function renderAgentSnapshot(snapshot) {
  const contract = snapshot.contract || {};
  const outcome = snapshot.outcome || {};
  const context = snapshot.context || {};
  const budget = snapshot.budget || {};
  const messages = Array.isArray(context.messages) ? context.messages.length : 0;
  const sent = snapshot.sent_counts || {};
  const sentTotal = Object.values(sent).reduce((sum, value) => sum + Number(value || 0), 0);
  const goal = contract.goal || snapshot.task_id || "No task contract exposed for this snapshot.";
  const summary = outcome.summary || latestOutcomeSummary(snapshot) || "No terminal outcome yet.";
  ui.drawerContent.innerHTML = `
    <section class="metric-grid">
      <div class="metric"><span>Messages</span><strong>${messages}</strong></div>
      <div class="metric"><span>Sent</span><strong>${sentTotal}</strong></div>
      <div class="metric"><span>State</span><strong>${snapshot.running ? "LIVE" : "IDLE"}</strong></div>
    </section>
    <section class="detail-section">
      <div class="detail-label">Contract goal</div>
      <div class="detail-value">${escapeHtml(goal)}</div>
    </section>
    <section class="detail-section">
      <div class="detail-label">Latest outcome</div>
      <div class="detail-value">${escapeHtml(summary)}</div>
    </section>
    <section class="detail-section">
      <div class="detail-label">Runtime budget</div>
      <pre class="json-block">${escapeHtml(JSON.stringify(budget, null, 2))}</pre>
    </section>
    <section class="detail-section">
      <div class="detail-label">State snapshot</div>
      <pre class="json-block">${escapeHtml(JSON.stringify(snapshot, null, 2))}</pre>
    </section>`;
}

function latestOutcomeSummary(snapshot) {
  const grouped = snapshot.node_outcomes || {};
  const outcomes = Object.values(grouped).flat();
  return outcomes.at(-1)?.summary || "";
}

function closeDrawer() {
  ui.drawerBackdrop.classList.add("hidden");
  ui.agentDrawer.classList.remove("open");
  ui.agentDrawer.setAttribute("aria-hidden", "true");
}

function openRequirements() {
  const requirements = state.selectedRun?.requirements || [];
  ui.requirementsList.innerHTML = requirements.map((requirement, index) => `
    <article class="requirement-item">
      <div class="requirement-number">${String(index + 1).padStart(2, "0")}</div>
      <p>${escapeHtml(requirement)}</p>
    </article>`).join("") || `<div class="detail-value">No requirements recorded.</div>`;
  ui.requirementsModal.classList.remove("hidden");
}

function closeRequirements() {
  ui.requirementsModal.classList.add("hidden");
}

async function startRun(event) {
  event.preventDefault();
  const request = ui.requestInput.value.trim();
  if (!request) return;
  setBusy(ui.launchButton, true, "Launching…");
  try {
    const payload = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ request }),
    });
    ui.requestInput.value = "";
    toast("Run launched. You can steer it while agents work.");
    await loadRuns();
    await selectRun(payload.run_id);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(ui.launchButton, false);
  }
}

async function steerRun(event) {
  event.preventDefault();
  const requirement = ui.steerInput.value.trim();
  if (!state.selectedRunId || !requirement) return;
  ui.steerButton.disabled = true;
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/steer`, {
      method: "POST",
      body: JSON.stringify({ requirement }),
    });
    ui.steerInput.value = "";
    toast(`Revision ${payload.revision} accepted by Master.`);
    await refreshSelectedRun();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    ui.steerButton.disabled = (
      state.selectedRun?.status === "archived" || state.selectedRun?.resumable === false
    );
  }
}

async function cancelRun() {
  if (!state.selectedRunId || !window.confirm("Cancel active work and retain all partial state?")) return;
  ui.cancelButton.disabled = true;
  try {
    await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/cancel`, { method: "POST" });
    toast("Run cancelled. Partial state remains retained.");
    await refreshSelectedRun();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function archiveRun() {
  if (!state.selectedRunId || !window.confirm("Release live agents? Persisted history will remain.")) return;
  ui.archiveButton.disabled = true;
  try {
    await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/archive`, { method: "POST" });
    toast("Live agents released. Run history remains on disk.");
    await refreshSelectedRun();
  } catch (error) {
    toast(error.message, "error");
    ui.archiveButton.disabled = false;
  }
}

function bindEvents() {
  ui.newRunForm.addEventListener("submit", startRun);
  ui.steerForm.addEventListener("submit", steerRun);
  ui.cancelButton.addEventListener("click", cancelRun);
  ui.archiveButton.addEventListener("click", archiveRun);
  ui.refreshRunsButton.addEventListener("click", () => loadRuns());
  ui.runList.addEventListener("click", (event) => {
    const item = event.target.closest("[data-run-id]");
    if (item) selectRun(item.dataset.runId);
  });
  ui.topologyCanvas.addEventListener("click", (event) => {
    const card = event.target.closest("[data-agent-id]");
    if (card) openAgent(card.dataset.agentId, card.dataset.agentKind);
  });
  ui.activityFilters.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-filter]");
    if (!chip) return;
    state.eventFilter = chip.dataset.filter;
    ui.activityFilters.querySelectorAll(".filter-chip").forEach((item) => {
      item.classList.toggle("active", item === chip);
    });
    renderActivity();
  });
  ui.drawerClose.addEventListener("click", closeDrawer);
  ui.drawerBackdrop.addEventListener("click", closeDrawer);
  ui.showRequirementsButton.addEventListener("click", openRequirements);
  ui.requirementsClose.addEventListener("click", closeRequirements);
  ui.requirementsModal.addEventListener("click", (event) => {
    if (event.target === ui.requirementsModal) closeRequirements();
  });
  ui.copyResponseButton.addEventListener("click", async () => {
    const response = state.selectedRun?.final_response;
    if (!response) return toast("No checkpoint to copy yet.", "error");
    await navigator.clipboard.writeText(response);
    toast("Checkpoint copied.");
  });
  ui.mobileRunsButton.addEventListener("click", () => {
    ui.runsPanel.classList.toggle("mobile-open");
  });
  ui.requestInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      ui.newRunForm.requestSubmit();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeDrawer();
      closeRequirements();
      ui.runsPanel.classList.remove("mobile-open");
    }
  });
  window.addEventListener("beforeunload", closeEventStream);
}

async function initialize() {
  bindEvents();
  await Promise.all([loadHealth(), loadRuns({ preserveSelection: false })]);
  state.listRefreshTimer = window.setInterval(async () => {
    await loadRuns();
    // SSE carries actions, while this poll also catches status-only transitions
    // such as Master committing a final checkpoint.
    if (state.selectedRunId) await refreshSelectedRun();
  }, 5000);
}

initialize();
