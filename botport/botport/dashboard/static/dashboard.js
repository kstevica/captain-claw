/* BotPort Dashboard - vanilla JS, no build step. */

const POLL_INTERVAL = 5000;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

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
  if (!id) return "—";
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
  el.innerHTML = concerns.map((c) => `
    <div class="concern-card" onclick="showConcern('${esc(c.id)}')">
      <div class="concern-header">
        <span class="concern-id">${shortId(c.id)}</span>
        <span class="concern-status status-${c.status}">${c.status}</span>
      </div>
      <div class="concern-task" title="${esc(c.task)}">${esc(truncate(c.task, 120))}</div>
      <div class="concern-meta">
        <span>from: ${esc(c.from_instance || "?")}</span>
        <span>to: ${esc(c.assigned_instance || "—")}</span>
        <span>${timeAgo(c.updated_at)}</span>
      </div>
      ${renderTags(c.expertise_tags)}
    </div>
  `).join("");
}

function renderHistory(concerns) {
  const el = $("#history-list");
  if (!concerns || concerns.length === 0) {
    el.innerHTML = '<p class="empty-state">No concerns yet</p>';
    return;
  }
  el.innerHTML = concerns.map((c) => `
    <div class="concern-card" onclick="showConcern('${esc(c.id)}')">
      <div class="concern-header">
        <span class="concern-id">${shortId(c.id)}</span>
        <span class="concern-status status-${c.status}">${c.status}</span>
      </div>
      <div class="concern-task" title="${esc(c.task)}">${esc(truncate(c.task, 100))}</div>
      <div class="concern-meta">
        <span>${esc(c.from_instance || "?")} &rarr; ${esc(c.assigned_instance || "—")}</span>
        <span>${timeAgo(c.created_at)}</span>
      </div>
    </div>
  `).join("");
}

function renderTags(tags) {
  if (!tags || tags.length === 0) return "";
  return `<div class="concern-tags">${
    tags.map((t) => `<span class="expertise-tag">${esc(t)}</span>`).join("")
  }</div>`;
}

function renderStats(stats) {
  if (!stats) return;
  $("#stat-instances").textContent = `${stats.connected_instances || 0} instances`;
  const active = (stats.by_status || {}).assigned || 0;
  const inProgress = (stats.by_status || {}).in_progress || 0;
  const pending = (stats.by_status || {}).pending || 0;
  $("#stat-active").textContent = `${pending + active + inProgress} active`;
  const rate = stats.success_rate != null ? Math.round(stats.success_rate) : "--";
  $("#stat-success").textContent = `${rate}% success`;
  if (stats.botport_version) {
    $("#stat-version").textContent = stats.botport_version;
  }
}

/* ---- Concern detail modal ---- */

async function showConcern(id) {
  const data = await fetchJSON(`/api/concerns/${id}`);
  if (!data || data.error) return;

  $("#modal-title").textContent = `Concern ${shortId(data.id)}`;
  const fields = [
    { label: "ID", value: data.id },
    { label: "Status", value: data.status },
    { label: "Task", value: data.task },
    { label: "From Instance", value: data.from_instance },
    { label: "Assigned Instance", value: data.assigned_instance || "—" },
    { label: "Created", value: data.created_at },
    { label: "Updated", value: data.updated_at },
    { label: "Expertise", value: (data.expertise_tags || []).join(", ") || "—" },
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

/* ---- Polling loop ---- */

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

  // Initial load + polling.
  refresh();
  setInterval(refresh, POLL_INTERVAL);
});
