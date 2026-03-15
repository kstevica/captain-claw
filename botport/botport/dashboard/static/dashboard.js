/* BotPort Dashboard - vanilla JS, no build step. */

const POLL_INTERVAL = 2000;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ---- Router ---- */

let currentPage = null;
let pollTimer = null;

function navigate(page) {
  if (currentPage === page) return;

  // Teardown previous page.
  if (currentPage === "swarm" && typeof teardownSwarmPage === "function") {
    teardownSwarmPage();
  }

  currentPage = page;

  // Hide all pages.
  $$(".page").forEach((el) => el.classList.add("hidden"));

  // Stop polling when leaving activity page.
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  const target = $(`#page-${page}`);
  if (target) {
    target.classList.remove("hidden");
  }

  if (page === "home") {
    refreshStats();
  } else if (page === "activity") {
    refresh();
    pollTimer = setInterval(refresh, POLL_INTERVAL);
  } else if (page === "swarm") {
    if (typeof initSwarmPage === "function") initSwarmPage();
  }
}

function handleRoute() {
  const hash = location.hash.replace(/^#\/?/, "") || "";
  if (hash === "" || hash === "/") {
    navigate("home");
  } else if (hash === "activity") {
    navigate("activity");
  } else if (hash === "swarm") {
    navigate("swarm");
  } else {
    navigate("home");
  }
}

/* ---- API helpers ---- */

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`fetch ${url}:`, err);
    return null;
  }
}

/* ---- Rendering ---- */

function shortId(id) {
  if (!id) return "\u2014";
  return id.length > 12 ? id.slice(0, 8) + "..." : id;
}

function timeAgo(iso) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
  return Math.floor(diff / 86400) + "d ago";
}

function renderPersona(p) {
  const tags = (p.expertise_tags || []).map((t) =>
    `<span class="expertise-tag">${esc(t)}</span>`
  ).join("");
  const desc = p.description ? `<div class="persona-description">${esc(p.description)}</div>` : "";
  const bg = p.background ? `<div class="persona-background">${esc(p.background)}</div>` : "";
  return `
    <div class="persona-card">
      <div class="persona-name">${esc(p.name)}</div>
      ${desc}
      ${bg}
      ${tags ? `<div class="persona-tags">${tags}</div>` : ""}
    </div>`;
}

function renderActivity(activity) {
  if (!activity || Object.keys(activity).length === 0) return "";
  // Render activity slots in priority order: status, thinking, tool_output.
  const order = ["status", "thinking", "tool_output"];
  const lines = [];
  for (const key of order) {
    const slot = activity[key];
    if (!slot) continue;
    const stale = slot.updated_at && (Date.now() - new Date(slot.updated_at).getTime()) > 30000;
    const cls = stale ? "activity-line stale" : "activity-line";
    if (key === "status") {
      lines.push(`<div class="${cls}"><span class="activity-icon">&#x27F3;</span> ${esc(slot.text || "")}</div>`);
    } else if (key === "thinking") {
      const toolLabel = slot.tool ? `<span class="activity-tool">${esc(slot.tool)}</span> ` : "";
      lines.push(`<div class="${cls}"><span class="activity-icon">&#x1F9E0;</span> ${toolLabel}${esc(truncate(slot.text || "", 120))}</div>`);
    } else if (key === "tool_output") {
      lines.push(`<div class="${cls}"><span class="activity-icon">&#x2699;</span> <span class="activity-tool">${esc(slot.tool_name || "")}</span> ${esc(truncate(slot.output || "", 80))}</div>`);
    }
  }
  if (lines.length === 0) return "";
  return `<div class="activity-section">${lines.join("")}</div>`;
}

function renderInstances(instances) {
  const el = $("#instances-list");
  if (!instances || instances.length === 0) {
    el.innerHTML = '<p class="empty-state">No instances connected</p>';
    return;
  }
  el.innerHTML = instances.map((inst) => {
    const loadPct = inst.max_concurrent > 0
      ? Math.round((inst.active_concerns / inst.max_concurrent) * 100)
      : 0;
    const loadColor = loadPct > 80 ? "var(--red)" : loadPct > 50 ? "var(--yellow)" : "var(--green)";
    const personaCards = (inst.personas || []).map(renderPersona).join("");
    const activityHtml = renderActivity(inst.activity);
    return `
      <div class="instance-card">
        <div class="instance-header">
          <span class="instance-name">${esc(inst.name)}</span>
          <span class="instance-status status-${inst.status}">${inst.status}</span>
        </div>
        <div class="instance-meta">
          <span>${inst.active_concerns}/${inst.max_concurrent} active</span>
          <span>${(inst.tools || []).length} tools</span>
          <span>${(inst.models || []).length} models</span>
          <span>${timeAgo(inst.last_heartbeat)}</span>
        </div>
        ${activityHtml}
        ${personaCards ? `<div class="persona-list">${personaCards}</div>` : ""}
        <div class="load-bar"><div class="load-bar-fill" style="width:${loadPct}%;background:${loadColor}"></div></div>
      </div>`;
  }).join("");
}

function renderConcerns(concerns) {
  const el = $("#concerns-list");
  if (!concerns || concerns.length === 0) {
    el.innerHTML = '<p class="empty-state">No active concerns</p>';
    return;
  }
  el.innerHTML = concerns.map((c) => {
    const fromLabel = c.from_instance_name || shortId(c.from_instance);
    const toLabel = c.assigned_instance_name || shortId(c.assigned_instance);
    const personaName = (c.metadata && c.metadata.persona_name) ? c.metadata.persona_name : "";
    const personaBadge = personaName
      ? `<span class="persona-badge" title="Answered by persona">${esc(personaName)}</span>`
      : "";
    return `
    <div class="concern-card" onclick="showConcern('${esc(c.id)}')">
      <div class="concern-header">
        <span class="concern-id">${shortId(c.id)}</span>
        ${personaBadge}
        <span class="concern-status status-${c.status}">${c.status}</span>
      </div>
      <div class="concern-task" title="${esc(c.task)}">${esc(truncate(c.task, 120))}</div>
      <div class="concern-meta">
        <span>${esc(fromLabel)} &rarr; ${esc(toLabel)}</span>
        <span>${timeAgo(c.updated_at)}</span>
      </div>
      ${renderTags(c.expertise_tags)}
    </div>`;
  }).join("");
}

function renderHistory(concerns) {
  const el = $("#history-list");
  if (!concerns || concerns.length === 0) {
    el.innerHTML = '<p class="empty-state">No concerns yet</p>';
    return;
  }
  el.innerHTML = concerns.map((c) => {
    const fromLabel = c.from_instance_name || shortId(c.from_instance);
    const toLabel = c.assigned_instance_name || shortId(c.assigned_instance);
    const personaName = (c.metadata && c.metadata.persona_name) ? c.metadata.persona_name : "";
    const personaBadge = personaName
      ? `<span class="persona-badge" title="Answered by persona">${esc(personaName)}</span>`
      : "";
    return `
    <div class="concern-card" onclick="showConcern('${esc(c.id)}')">
      <div class="concern-header">
        <span class="concern-id">${shortId(c.id)}</span>
        ${personaBadge}
        <span class="concern-status status-${c.status}">${c.status}</span>
      </div>
      <div class="concern-task" title="${esc(c.task)}">${esc(truncate(c.task, 100))}</div>
      <div class="concern-meta">
        <span>${esc(fromLabel)} &rarr; ${esc(toLabel)}</span>
        <span>${timeAgo(c.created_at)}</span>
      </div>
    </div>`;
  }).join("");
}

function renderTags(tags) {
  if (!tags || tags.length === 0) return "";
  return `<div class="concern-tags">${
    tags.map((t) => `<span class="expertise-tag">${esc(t)}</span>`).join("")
  }</div>`;
}

function renderStats(stats) {
  if (!stats) return;
  const instanceCount = stats.connected_instances || 0;
  const active = (stats.by_status || {}).assigned || 0;
  const inProgress = (stats.by_status || {}).in_progress || 0;
  const pending = (stats.by_status || {}).pending || 0;
  const activeCount = pending + active + inProgress;
  const rate = stats.success_rate != null ? Math.round(stats.success_rate) : "--";

  // Header stats.
  $("#stat-instances").textContent = `${instanceCount} instances`;
  $("#stat-active").textContent = `${activeCount} active`;
  $("#stat-success").textContent = `${rate}% success`;
  if (stats.botport_version) {
    $("#stat-version").textContent = stats.botport_version;
  }

  // Home page stats (may not exist if on different page).
  const homeInstEl = $("#home-stat-instances");
  const homeActiveEl = $("#home-stat-active");
  if (homeInstEl) homeInstEl.textContent = `${instanceCount} instances`;
  if (homeActiveEl) homeActiveEl.textContent = `${activeCount} active`;
}

/* ---- Concern detail modal ---- */

async function showConcern(id) {
  const data = await fetchJSON(`/api/concerns/${id}`);
  if (!data || data.error) return;

  $("#modal-title").textContent = `Concern ${shortId(data.id)}`;
  const fromLabel = data.from_instance_name
    ? `${data.from_instance_name} (${shortId(data.from_instance)})`
    : data.from_instance;
  const toLabel = data.assigned_instance_name
    ? `${data.assigned_instance_name} (${shortId(data.assigned_instance)})`
    : (data.assigned_instance || "\u2014");
  const personaName = (data.metadata && data.metadata.persona_name) || "\u2014";
  const fields = [
    { label: "ID", value: data.id },
    { label: "Status", value: data.status },
    { label: "Task", value: data.task },
    { label: "From", value: fromLabel },
    { label: "Assigned To", value: toLabel },
    { label: "Persona", value: personaName },
    { label: "Created", value: data.created_at },
    { label: "Updated", value: data.updated_at },
    { label: "Expertise", value: (data.expertise_tags || []).join(", ") || "\u2014" },
  ];

  let html = fields.map((f) => `
    <div class="modal-field">
      <label>${f.label}</label>
      <div class="value">${esc(String(f.value))}</div>
    </div>
  `).join("");

  if (data.context && Object.keys(data.context).length > 0) {
    html += `
      <div class="modal-field">
        <label>Context</label>
        <div class="value" style="font-size:12px;font-family:monospace;white-space:pre-wrap">${esc(JSON.stringify(data.context, null, 2))}</div>
      </div>`;
  }

  if (data.messages && data.messages.length > 0) {
    html += `<div class="modal-field"><label>Messages (${data.messages.length})</label></div>`;
    html += '<div class="message-list">';
    for (const msg of data.messages) {
      html += `
        <div class="message-item direction-${msg.direction}">
          <div class="message-direction">${msg.direction} ${msg.from_instance ? "from " + esc(msg.from_instance) : ""}</div>
          <div class="message-content">${esc(truncate(msg.content, 500))}</div>
          <div class="message-meta">${timeAgo(msg.timestamp)}</div>
        </div>`;
    }
    html += "</div>";
  }

  $("#modal-body").innerHTML = html;
  $("#concern-modal").classList.remove("hidden");
}

function closeModal() {
  $("#concern-modal").classList.add("hidden");
}

/* ---- Utilities ---- */

function esc(s) {
  if (!s) return "";
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function truncate(s, max) {
  if (!s) return "";
  return s.length > max ? s.slice(0, max) + "..." : s;
}

/* ---- Data refresh ---- */

async function refreshStats() {
  const [stats, projects] = await Promise.all([
    fetchJSON("/api/stats"),
    fetchJSON("/api/swarm/projects"),
  ]);
  renderStats(stats);

  // Update swarm home card stats.
  const projEl = document.getElementById("home-stat-projects");
  const swarmEl = document.getElementById("home-stat-swarms");
  if (projEl && projects) projEl.textContent = `${projects.length} projects`;
  if (swarmEl && projects) {
    // Fetch total swarm count.
    const swarms = await fetchJSON("/api/swarm/swarms");
    if (swarms) swarmEl.textContent = `${swarms.length} swarms`;
  }
}

async function refresh() {
  const [instances, stats, activeConcerns, allConcerns] = await Promise.all([
    fetchJSON("/api/instances"),
    fetchJSON("/api/stats"),
    fetchJSON("/api/concerns?active=true"),
    fetchJSON("/api/concerns"),
  ]);

  renderInstances(instances);
  renderStats(stats);
  renderConcerns(activeConcerns);
  renderHistory(allConcerns);
}

/* ---- Init ---- */

document.addEventListener("DOMContentLoaded", () => {
  // Modal close handlers.
  $(".modal-close").addEventListener("click", closeModal);
  $(".modal-backdrop").addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Hash-based routing.
  window.addEventListener("hashchange", handleRoute);
  handleRoute();
});
