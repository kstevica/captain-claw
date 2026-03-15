/* BotPort Swarm UI - project/swarm management and DAG visualization. */

/* ---- State ---- */

let currentProject = null;   // Selected project ID.
let currentSwarm = null;     // Selected swarm ID.
let swarmPollTimer = null;
let dagCanvas = null;
let dagCtx = null;
let llmDefaultModel = "";    // Default LLM model from config.
let llmModels = [];          // Allowed models from CC config.
let _lastSwarmStatus = "";   // Track to avoid re-rendering toolbar on poll.
let _lastSelectedModel = ""; // Preserve model selection across re-renders.

/* ---- Helpers ---- */

function renderMd(text) {
  /* Lightweight markdown → HTML. Handles headers, bold, italic, code,
     inline code, lists (ul/ol), blockquotes, horizontal rules, links,
     and line breaks. Input is NOT pre-escaped — we escape first. */
  if (!text) return "";
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Fenced code blocks: ```...```
  s = s.replace(/```([\s\S]*?)```/g, (_, code) =>
    `<pre class="md-code-block"><code>${code.trim()}</code></pre>`);

  // Split into lines for block-level processing.
  const lines = s.split("\n");
  const out = [];
  let inUl = false, inOl = false;

  function closeLists() {
    if (inUl) { out.push("</ul>"); inUl = false; }
    if (inOl) { out.push("</ol>"); inOl = false; }
  }

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Horizontal rule.
    if (/^(\s*[-*_]\s*){3,}$/.test(line)) {
      closeLists();
      out.push("<hr>");
      continue;
    }

    // Table: detect a pipe-delimited row followed by a separator row.
    if (line.trim().startsWith("|") && i + 1 < lines.length
        && /^\|?\s*[-:]+[-|\s:]*$/.test(lines[i + 1].trim())) {
      closeLists();
      const headerCells = _parsePipeRow(line);
      const sepLine = lines[i + 1].trim();
      const aligns = _parseAlignRow(sepLine);
      i++; // skip separator

      let tableHtml = '<table class="md-table"><thead><tr>';
      headerCells.forEach((cell, ci) => {
        const align = aligns[ci] || "";
        const style = align ? ` style="text-align:${align}"` : "";
        tableHtml += `<th${style}>${inlineMd(cell)}</th>`;
      });
      tableHtml += "</tr></thead><tbody>";

      // Consume body rows.
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) {
        i++;
        const cells = _parsePipeRow(lines[i]);
        tableHtml += "<tr>";
        cells.forEach((cell, ci) => {
          const align = aligns[ci] || "";
          const style = align ? ` style="text-align:${align}"` : "";
          tableHtml += `<td${style}>${inlineMd(cell)}</td>`;
        });
        tableHtml += "</tr>";
      }
      tableHtml += "</tbody></table>";
      out.push(tableHtml);
      continue;
    }

    // Headers (# to ####).
    const hMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      closeLists();
      const level = hMatch[1].length;
      out.push(`<h${level + 2} class="md-h">${inlineMd(hMatch[2])}</h${level + 2}>`);
      continue;
    }

    // Blockquote.
    const bqMatch = line.match(/^&gt;\s?(.*)$/);
    if (bqMatch) {
      closeLists();
      out.push(`<blockquote class="md-bq">${inlineMd(bqMatch[1])}</blockquote>`);
      continue;
    }

    // Unordered list item.
    const ulMatch = line.match(/^\s*[-*+]\s+(.+)$/);
    if (ulMatch) {
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (!inUl) { out.push("<ul>"); inUl = true; }
      out.push(`<li>${inlineMd(ulMatch[1])}</li>`);
      continue;
    }

    // Ordered list item.
    const olMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (olMatch) {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (!inOl) { out.push("<ol>"); inOl = true; }
      out.push(`<li>${inlineMd(olMatch[1])}</li>`);
      continue;
    }

    // Close any open list.
    closeLists();

    // Empty line → paragraph break.
    if (!line.trim()) {
      out.push("<br>");
      continue;
    }

    // Regular paragraph line.
    out.push(`<p class="md-p">${inlineMd(line)}</p>`);
  }

  if (inUl) out.push("</ul>");
  if (inOl) out.push("</ol>");

  return out.join("\n");
}

function inlineMd(s) {
  /* Inline markdown: bold, italic, code, links. */
  return s
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/_([^_]+)_/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
}

function _parsePipeRow(line) {
  /* Split a pipe-delimited table row into cell strings. */
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map(c => c.trim());
}

function _parseAlignRow(line) {
  /* Parse separator row like |:---|---:|:---:| into alignment array. */
  return _parsePipeRow(line).map(cell => {
    const c = cell.trim();
    if (c.startsWith(":") && c.endsWith(":")) return "center";
    if (c.endsWith(":")) return "right";
    if (c.startsWith(":")) return "left";
    return "";
  });
}

function elapsedSince(iso) {
  if (!iso) return "";
  try {
    const s = String(iso);
    const start = new Date(s.endsWith("Z") ? s : s + "Z");
    if (isNaN(start.getTime())) return "";
    const secs = Math.max(0, Math.floor((Date.now() - start.getTime()) / 1000));
    if (secs < 60) return `${secs}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return `${h}h ${m}m`;
  } catch (e) { return ""; }
}

/* ---- API ---- */

async function swarmAPI(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  try {
    const res = await fetch(`/api/swarm${path}`, opts);
    const data = await res.json();
    if (!res.ok) data._error = true;
    return data;
  } catch (err) {
    console.warn(`swarmAPI ${method} ${path}:`, err);
    return { _error: true, error: err.message };
  }
}

/* ---- Swarm page init / teardown ---- */

function initSwarmPage() {
  loadProjects();
  loadLLMConfig();
}

async function loadLLMConfig() {
  const data = await swarmAPI("GET", "/llm-config");
  if (!data._error) {
    llmDefaultModel = `${data.default_provider || ""}/${data.default_model || ""}`;
    llmModels = data.models || [];
  }
}

function getSelectedModel() {
  const el = document.getElementById("swarm-model-select");
  if (el) _lastSelectedModel = el.value;
  return _lastSelectedModel;
}

function buildModelSelector() {
  const selected = _lastSelectedModel;
  if (!llmModels.length) {
    return `<select id="swarm-model-select" class="form-input model-select-input" onchange="_lastSelectedModel=this.value">
              <option value="">Default</option>
            </select>`;
  }
  let opts = `<option value="" ${!selected ? 'selected' : ''}>Default (${esc(llmDefaultModel)})</option>`;
  for (const m of llmModels) {
    const id = m.id || `${m.provider}/${m.model}`;
    const label = m.description ? `${id} — ${m.description}` : id;
    const mtype = m.model_type && m.model_type !== "llm" ? ` [${m.model_type}]` : "";
    const sel = selected === id ? "selected" : "";
    opts += `<option value="${esc(id)}" ${sel}>${esc(label)}${esc(mtype)}</option>`;
  }
  return `<select id="swarm-model-select" class="form-input model-select-input"
            title="LLM model for rephrase/decompose" onchange="_lastSelectedModel=this.value">${opts}</select>`;
}

function teardownSwarmPage() {
  if (swarmPollTimer) {
    clearInterval(swarmPollTimer);
    swarmPollTimer = null;
  }
  currentProject = null;
  currentSwarm = null;
}

/* ---- Projects ---- */

async function loadProjects() {
  const projects = await swarmAPI("GET", "/projects");
  if (projects._error) return;
  renderProjectList(projects);
}

function renderProjectList(projects) {
  const el = $("#swarm-project-list");
  if (!el) return;

  let html = '<div class="swarm-list-header"><h3>Projects</h3>'
    + '<button class="btn btn-sm" onclick="showCreateProject()">+ New</button></div>';

  if (!projects || projects.length === 0) {
    html += '<p class="empty-state">No projects yet</p>';
  } else {
    html += projects.map(p => `
      <div class="swarm-list-item ${currentProject === p.id ? 'active' : ''}"
           onclick="selectProject('${esc(p.id)}')">
        <div class="swarm-list-item-name">${esc(p.name)}</div>
        <div class="swarm-list-item-desc">${esc(truncate(p.description, 60))}</div>
      </div>
    `).join("");
  }

  el.innerHTML = html;
}

async function selectProject(projectId) {
  currentProject = projectId;
  currentSwarm = null;
  const project = await swarmAPI("GET", `/projects/${projectId}`);
  if (project._error) return;

  // Re-render project list to highlight active.
  await loadProjects();

  // Render swarm list for this project.
  renderSwarmList(project.swarms || []);

  // Clear main area.
  renderSwarmEmpty();
}

function showCreateProject() {
  const modal = $("#swarm-modal");
  if (!modal) return;

  $("#swarm-modal-title").textContent = "New Project";
  $("#swarm-modal-body").innerHTML = `
    <div class="form-group">
      <label>Name</label>
      <input type="text" id="input-project-name" class="form-input" placeholder="Project name" />
    </div>
    <div class="form-group">
      <label>Description</label>
      <textarea id="input-project-desc" class="form-input" rows="3" placeholder="Short description"></textarea>
    </div>
    <div class="form-actions">
      <button class="btn btn-primary" onclick="createProject()">Create</button>
      <button class="btn" onclick="closeSwarmModal()">Cancel</button>
    </div>
  `;
  modal.classList.remove("hidden");
  setTimeout(() => document.getElementById("input-project-name")?.focus(), 50);
}

async function createProject() {
  const name = document.getElementById("input-project-name")?.value?.trim();
  const desc = document.getElementById("input-project-desc")?.value?.trim();
  if (!name) return;

  const result = await swarmAPI("POST", "/projects", { name, description: desc || "" });
  if (result._error) return;

  closeSwarmModal();
  await loadProjects();
  selectProject(result.id);
}

/* ---- Swarms ---- */

function renderSwarmList(swarms) {
  const el = $("#swarm-swarm-list");
  if (!el) return;

  let html = '<div class="swarm-list-header"><h3>Swarms</h3>';
  if (currentProject) {
    html += '<div class="swarm-list-btns"><button class="btn btn-sm" onclick="showTemplates()">Templates</button>';
    html += '<button class="btn btn-sm" onclick="showCreateSwarm()">+ New</button></div>';
  }
  html += '</div>';

  if (!swarms || swarms.length === 0) {
    html += '<p class="empty-state">No swarms in this project</p>';
  } else {
    html += swarms.map(s => `
      <div class="swarm-list-item ${currentSwarm === s.id ? 'active' : ''}"
           data-swarm-id="${esc(s.id)}"
           onclick="selectSwarm('${esc(s.id)}')">
        <div class="swarm-list-item-header">
          <span class="swarm-list-item-name">${esc(s.name || 'Untitled')}</span>
          <span class="swarm-status-badge status-swarm-${s.status}">${s.status}</span>
        </div>
        <div class="swarm-list-item-desc">${esc(truncate(s.original_task, 80))}</div>
      </div>
    `).join("");
  }

  el.innerHTML = html;
}

function showCreateSwarm() {
  if (!currentProject) return;

  const modal = $("#swarm-modal");
  if (!modal) return;

  $("#swarm-modal-title").textContent = "New Swarm";
  $("#swarm-modal-body").innerHTML = `
    <div class="form-group">
      <label>Name</label>
      <input type="text" id="input-swarm-name" class="form-input" placeholder="Swarm name" />
    </div>
    <div class="form-group">
      <label>Task</label>
      <textarea id="input-swarm-task" class="form-input" rows="6"
                placeholder="Describe the complex task you want the swarm to accomplish..."></textarea>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Concurrency Limit</label>
        <input type="number" id="input-swarm-concurrency" class="form-input" value="5" min="1" max="50" />
      </div>
      <div class="form-group">
        <label>Priority</label>
        <input type="number" id="input-swarm-priority" class="form-input" value="0" min="0" max="10" />
      </div>
    </div>
    <div class="form-group">
      <label>Error Policy</label>
      <select id="input-swarm-error-policy" class="form-input">
        <option value="fail_fast">Fail Fast — stop on first failure</option>
        <option value="continue_on_error">Continue on Error — skip dependents</option>
        <option value="manual_review">Manual Review — pause for human review</option>
      </select>
    </div>
    <div class="form-actions">
      <button class="btn btn-primary" onclick="createSwarm()">Create</button>
      <button class="btn" onclick="closeSwarmModal()">Cancel</button>
    </div>
  `;
  modal.classList.remove("hidden");
  setTimeout(() => document.getElementById("input-swarm-name")?.focus(), 50);
}

async function createSwarm() {
  const name = document.getElementById("input-swarm-name")?.value?.trim();
  const task = document.getElementById("input-swarm-task")?.value?.trim();
  const concurrency = parseInt(document.getElementById("input-swarm-concurrency")?.value || "5", 10);
  const priority = parseInt(document.getElementById("input-swarm-priority")?.value || "0", 10);

  if (!task) return;

  const errorPolicy = document.getElementById("input-swarm-error-policy")?.value || "fail_fast";

  const result = await swarmAPI("POST", "/swarms", {
    project_id: currentProject,
    name: name || "",
    task,
    concurrency_limit: concurrency,
    priority,
    error_policy: errorPolicy,
  });
  if (result._error) return;

  closeSwarmModal();
  selectSwarm(result.id);

  // Refresh project to update swarm list.
  const project = await swarmAPI("GET", `/projects/${currentProject}`);
  if (!project._error) renderSwarmList(project.swarms || []);
}

async function selectSwarm(swarmId) {
  currentSwarm = swarmId;
  _lastSwarmStatus = "";

  // Stop previous polling.
  if (swarmPollTimer) {
    clearInterval(swarmPollTimer);
    swarmPollTimer = null;
  }

  await refreshSwarmDetail();

  // Start polling for running swarms.
  swarmPollTimer = setInterval(refreshSwarmDetail, 2000);

  // Update swarm list highlighting.
  if (currentProject) {
    const project = await swarmAPI("GET", `/projects/${currentProject}`);
    if (!project._error) renderSwarmList(project.swarms || []);
  }
}

async function refreshSwarmDetail() {
  if (!currentSwarm) return;

  // Save model selector value before re-render.
  const modelEl = document.getElementById("swarm-model-select");
  if (modelEl) _lastSelectedModel = modelEl.value;

  const swarm = await swarmAPI("GET", `/swarms/${currentSwarm}`);
  if (swarm._error) return;

  // If status unchanged, do incremental update (no full DOM rebuild).
  if (_lastSwarmStatus === swarm.status && document.querySelector(".swarm-detail")) {
    _updateSwarmDetailInPlace(swarm);
  } else {
    renderSwarmDetail(swarm);
  }

  // Keep sidebar status badge in sync.
  _updateSidebarStatus(swarm);
}

function _updateSidebarStatus(swarm) {
  const item = document.querySelector(`.swarm-list-item[data-swarm-id="${swarm.id}"]`);
  if (item) {
    const badge = item.querySelector(".swarm-status-badge");
    if (badge) {
      badge.textContent = swarm.status;
      badge.className = `swarm-status-badge status-swarm-${swarm.status}`;
    }
  }
}

/* ---- Swarm detail view ---- */

function renderSwarmEmpty() {
  const el = $("#swarm-main");
  if (!el) return;
  el.innerHTML = '<div class="swarm-empty-main"><p class="empty-state">Select or create a swarm to get started</p></div>';
}

function _updateSwarmDetailInPlace(swarm) {
  /* Incremental update — refresh dynamic content without rebuilding toolbar/tabs. */
  const tasks = swarm.tasks || [];
  const edges = swarm.edges || [];

  // Update rephrased text (appears after rephrase while still in draft).
  const rephrasedEl = document.querySelector(".swarm-detail-rephrased");
  if (swarm.rephrased_task && !rephrasedEl) {
    // Insert rephrased div after task div.
    const taskEl = document.querySelector(".swarm-detail-task");
    if (taskEl) {
      const div = document.createElement("div");
      div.className = "swarm-detail-rephrased";
      div.innerHTML = `<div class="md-label">Rephrased:</div>${renderMd(swarm.rephrased_task)}`;
      taskEl.after(div);
    }
  } else if (swarm.rephrased_task && rephrasedEl) {
    rephrasedEl.innerHTML = `<div class="md-label">Rephrased:</div>${renderMd(swarm.rephrased_task)}`;
  }

  // Update decomposition reasoning.
  const reasoningEl = document.querySelector(".swarm-detail-reasoning");
  const reasoning = swarm.metadata?.decomposition_reasoning;
  if (reasoning && !reasoningEl) {
    const anchor = document.querySelector(".swarm-detail-rephrased") || document.querySelector(".swarm-detail-task");
    if (anchor) {
      const div = document.createElement("div");
      div.className = "swarm-detail-reasoning";
      div.innerHTML = `<strong>Decomposition strategy:</strong> ${esc(reasoning)}`;
      anchor.after(div);
    }
  } else if (reasoning && reasoningEl) {
    reasoningEl.innerHTML = `<strong>Decomposition strategy:</strong> ${esc(reasoning)}`;
  }

  // Update pipeline steps.
  const pipelineEl = document.querySelector(".pipeline-steps");
  if (pipelineEl) {
    const steps = [
      { label: "Task", done: !!swarm.original_task },
      { label: "Rephrase", done: !!swarm.rephrased_task },
      { label: "Decompose", done: tasks.length > 0 },
      { label: "Agents", done: tasks.some(t => t.assigned_persona) },
      { label: "Execute", done: swarm.status === "running" || swarm.status === "completed" },
    ];
    pipelineEl.innerHTML = steps.map((s, i) =>
      `<div class="pipeline-step ${s.done ? 'done' : ''}">
        <span class="pipeline-step-num">${i + 1}</span>
        <span class="pipeline-step-label">${s.label}</span>
      </div>`
    ).join('<div class="pipeline-arrow">&#x2192;</div>');
  }

  const taskStats = {
    total: tasks.length,
    completed: tasks.filter(t => t.status === "completed").length,
    running: tasks.filter(t => t.status === "running").length,
    failed: tasks.filter(t => t.status === "failed").length,
    queued: tasks.filter(t => t.status === "queued" || t.status === "waiting").length,
    pending_approval: tasks.filter(t => t.status === "pending_approval").length,
  };

  // Update stat pills.
  const statsEl = document.querySelector(".swarm-detail-stats");
  if (statsEl) {
    statsEl.innerHTML = `
      <span class="stat-pill">${taskStats.total} tasks</span>
      <span class="stat-pill stat-completed">${taskStats.completed} done</span>
      <span class="stat-pill stat-running">${taskStats.running} running</span>
      <span class="stat-pill stat-failed">${taskStats.failed} failed</span>
      <span class="stat-pill stat-queued">${taskStats.queued} queued</span>
      ${taskStats.pending_approval > 0 ? `<span class="stat-pill stat-pending-approval">${taskStats.pending_approval} awaiting approval</span>` : ""}
    `;
  }

  // Update progress bar.
  const progressEl = document.querySelector(".swarm-progress-bar");
  if (progressEl && taskStats.total > 0) {
    progressEl.innerHTML = `
      <div class="swarm-progress-fill swarm-progress-completed" style="width: ${(taskStats.completed / taskStats.total * 100).toFixed(1)}%"></div>
      <div class="swarm-progress-fill swarm-progress-running" style="width: ${(taskStats.running / taskStats.total * 100).toFixed(1)}%"></div>
      <div class="swarm-progress-fill swarm-progress-failed" style="width: ${(taskStats.failed / taskStats.total * 100).toFixed(1)}%"></div>
    `;
  }

  // Update task list tab.
  const canAddTasks = !swarm.status || swarm.status === "draft" || swarm.status === "ready" || swarm.status === "paused";
  const taskListEl = document.getElementById("swarm-tab-tasks");
  if (taskListEl) {
    taskListEl.innerHTML = renderTaskList(tasks, edges, canAddTasks);
  }

  // Redraw DAG and update drag references.
  if (dagCanvas && dagCtx) {
    // Preserve dragged positions: if a drag is in progress, skip redraw.
    if (!_dagDragState) {
      _dagTasks = tasks;
      _dagEdges = edges;
      drawDAG(tasks, edges);
    }
  }

  // Reload active tab content.
  const activeTab = document.querySelector(".swarm-tab.active");
  if (activeTab && activeTab.dataset.tab === "audit") {
    loadAuditLog();
  } else if (activeTab && activeTab.dataset.tab === "checkpoints") {
    loadCheckpoints();
  }
}

function renderSwarmDetail(swarm) {
  const el = $("#swarm-main");
  if (!el) return;

  _lastSwarmStatus = swarm.status;

  const tasks = swarm.tasks || [];
  const edges = swarm.edges || [];

  const taskStats = {
    total: tasks.length,
    completed: tasks.filter(t => t.status === "completed").length,
    running: tasks.filter(t => t.status === "running").length,
    failed: tasks.filter(t => t.status === "failed").length,
    queued: tasks.filter(t => t.status === "queued" || t.status === "waiting").length,
    pending_approval: tasks.filter(t => t.status === "pending_approval").length,
  };

  // Action buttons based on state.
  let actions = "";
  // Model selector for LLM operations.
  const modelSelector = buildModelSelector();

  if (swarm.status === "draft") {
    actions = `${modelSelector}
               <button class="btn btn-sm" onclick="rephraseTask()" id="btn-rephrase">Rephrase</button>
               <button class="btn btn-sm btn-primary" onclick="decomposeTask()" id="btn-decompose">Decompose</button>
               <button class="btn btn-danger btn-sm" onclick="swarmAction('cancel')">Cancel</button>`;
  } else if (swarm.status === "decomposing") {
    actions = `<span class="swarm-status-badge status-swarm-decomposing">Decomposing...</span>`;
  } else if (swarm.status === "ready") {
    actions = `${modelSelector}
               <button class="btn btn-sm" onclick="decomposeTask()">Re-decompose</button>
               <button class="btn btn-sm" onclick="selectAgents()">Select Agents</button>
               <button class="btn btn-sm" onclick="saveAsTemplate()">Save Template</button>
               <button class="btn btn-sm btn-primary" onclick="swarmAction('start')">Start</button>
               <button class="btn btn-danger btn-sm" onclick="swarmAction('cancel')">Cancel</button>`;
  } else if (swarm.status === "running") {
    actions = `<button class="btn btn-warning btn-sm" onclick="swarmAction('pause')">Pause</button>
               <button class="btn btn-danger btn-sm" onclick="swarmAction('cancel')">Cancel</button>`;
  } else if (swarm.status === "paused") {
    actions = `<button class="btn btn-primary btn-sm" onclick="swarmAction('resume')">Resume</button>
               <button class="btn btn-danger btn-sm" onclick="swarmAction('cancel')">Cancel</button>`;
  }

  const canAddTasks = !swarm.status || swarm.status === "draft" || swarm.status === "ready" || swarm.status === "paused";
  const isDraft = swarm.status === "draft";

  // Pipeline steps indicator.
  const steps = [
    { label: "Task", done: !!swarm.original_task },
    { label: "Rephrase", done: !!swarm.rephrased_task },
    { label: "Decompose", done: tasks.length > 0 },
    { label: "Agents", done: tasks.some(t => t.assigned_persona) },
    { label: "Execute", done: swarm.status === "running" || swarm.status === "completed" },
  ];
  const pipelineHtml = `<div class="pipeline-steps">${steps.map((s, i) =>
    `<div class="pipeline-step ${s.done ? 'done' : ''}">
      <span class="pipeline-step-num">${i + 1}</span>
      <span class="pipeline-step-label">${s.label}</span>
    </div>`
  ).join('<div class="pipeline-arrow">&#x2192;</div>')}</div>`;

  el.innerHTML = `
    <div class="swarm-detail">
      <div class="swarm-detail-toolbar">
        <div class="swarm-detail-info">
          <h2>${esc(swarm.name || 'Untitled Swarm')}</h2>
          <span class="swarm-status-badge status-swarm-${swarm.status}">${swarm.status}</span>
          <span class="error-policy-badge policy-${swarm.error_policy || 'fail_fast'}">${(swarm.error_policy || 'fail_fast').replace(/_/g, ' ')}</span>
        </div>
        <div class="swarm-detail-actions">${actions}</div>
      </div>

      ${pipelineHtml}

      <div class="swarm-detail-task">${renderMd(swarm.original_task)}</div>
      ${swarm.rephrased_task ? `<div class="swarm-detail-rephrased"><div class="md-label">Rephrased:</div>${renderMd(swarm.rephrased_task)}</div>` : ""}
      ${swarm.metadata?.decomposition_reasoning ? `<div class="swarm-detail-reasoning"><strong>Decomposition strategy:</strong> ${esc(swarm.metadata.decomposition_reasoning)}</div>` : ""}

      <div class="swarm-detail-stats">
        <span class="stat-pill">${taskStats.total} tasks</span>
        <span class="stat-pill stat-completed">${taskStats.completed} done</span>
        <span class="stat-pill stat-running">${taskStats.running} running</span>
        <span class="stat-pill stat-failed">${taskStats.failed} failed</span>
        <span class="stat-pill stat-queued">${taskStats.queued} queued</span>
        ${taskStats.pending_approval > 0 ? `<span class="stat-pill stat-pending-approval">${taskStats.pending_approval} awaiting approval</span>` : ""}
      </div>
      ${taskStats.total > 0 && (swarm.status === "running" || swarm.status === "completed" || swarm.status === "failed") ? `
      <div class="swarm-progress-bar">
        <div class="swarm-progress-fill swarm-progress-completed" style="width: ${(taskStats.completed / taskStats.total * 100).toFixed(1)}%"></div>
        <div class="swarm-progress-fill swarm-progress-running" style="width: ${(taskStats.running / taskStats.total * 100).toFixed(1)}%"></div>
        <div class="swarm-progress-fill swarm-progress-failed" style="width: ${(taskStats.failed / taskStats.total * 100).toFixed(1)}%"></div>
      </div>` : ""}

      <div class="swarm-tabs">
        <button class="swarm-tab active" data-tab="dag" onclick="switchSwarmTab(this, 'dag')">DAG</button>
        <button class="swarm-tab" data-tab="tasks" onclick="switchSwarmTab(this, 'tasks')">Tasks</button>
        <button class="swarm-tab" data-tab="gantt" onclick="switchSwarmTab(this, 'gantt')">Timeline</button>
        <button class="swarm-tab" data-tab="costs" onclick="switchSwarmTab(this, 'costs')">Costs</button>
        <button class="swarm-tab" data-tab="checkpoints" onclick="switchSwarmTab(this, 'checkpoints')">Checkpoints</button>
        <button class="swarm-tab" data-tab="files" onclick="switchSwarmTab(this, 'files')">Files</button>
        <button class="swarm-tab" data-tab="audit" onclick="switchSwarmTab(this, 'audit')">Audit</button>
      </div>

      <div id="swarm-tab-dag" class="swarm-tab-content active">
        <div class="dag-toolbar">
          ${canAddTasks ? '<button class="btn btn-sm" onclick="showCreateTask()">+ Add Task</button>' : ''}
          <button class="btn btn-sm" onclick="autoLayoutDAG()">Auto Layout</button>
        </div>
        <canvas id="dag-canvas" width="800" height="400"></canvas>
      </div>

      <div id="swarm-tab-tasks" class="swarm-tab-content">
        ${renderTaskList(tasks, edges, canAddTasks)}
      </div>

      <div id="swarm-tab-gantt" class="swarm-tab-content">
        <div id="gantt-content"><p class="empty-state">Loading timeline...</p></div>
      </div>

      <div id="swarm-tab-costs" class="swarm-tab-content">
        <div id="costs-content"><p class="empty-state">Loading costs...</p></div>
      </div>

      <div id="swarm-tab-checkpoints" class="swarm-tab-content">
        <div class="checkpoint-toolbar">
          <button class="btn btn-sm" onclick="createCheckpoint()">+ Create Checkpoint</button>
        </div>
        <div id="checkpoint-list-content"><p class="empty-state">Loading...</p></div>
      </div>

      <div id="swarm-tab-files" class="swarm-tab-content">
        <div id="files-content"><p class="empty-state">Loading files...</p></div>
      </div>

      <div id="swarm-tab-audit" class="swarm-tab-content">
        <div id="audit-log-content"><p class="empty-state">Loading...</p></div>
      </div>
    </div>
  `;

  // Initialize DAG canvas.
  dagCanvas = document.getElementById("dag-canvas");
  if (dagCanvas) {
    dagCtx = dagCanvas.getContext("2d");
    resizeDAGCanvas();
    drawDAG(tasks, edges);

    // Drag & click handler for DAG nodes.
    _initDAGDrag(dagCanvas, tasks, edges);
  }

  // Load audit/checkpoints if on that tab.
  const activeTab = el.querySelector(".swarm-tab.active");
  if (activeTab && activeTab.dataset.tab === "audit") {
    loadAuditLog();
  } else if (activeTab && activeTab.dataset.tab === "checkpoints") {
    loadCheckpoints();
  }
}

function renderTaskList(tasks, edges, canAddTasks) {
  if (!tasks || tasks.length === 0) {
    return `<p class="empty-state">No tasks yet. ${canAddTasks ? 'Add tasks to build the DAG.' : ''}</p>`;
  }

  // Build dependency lookup.
  const deps = {};
  for (const e of edges) {
    if (!deps[e.to_task_id]) deps[e.to_task_id] = [];
    deps[e.to_task_id].push(e.from_task_id);
  }

  const taskMap = {};
  tasks.forEach(t => taskMap[t.id] = t);

  return tasks.map(t => {
    const depNames = (deps[t.id] || []).map(did => {
      const dt = taskMap[did];
      return dt ? esc(dt.name || shortId(did)) : shortId(did);
    }).join(", ");

    const depLine = depNames ? `<div class="task-deps">Depends on: ${depNames}</div>` : "";

    let taskActions = "";
    if (t.status === "pending_approval") {
      taskActions = `<button class="btn btn-xs btn-approve" onclick="approveTask('${esc(t.id)}')">Approve</button>
                     <button class="btn btn-xs btn-danger" onclick="rejectTask('${esc(t.id)}')">Reject</button>
                     <button class="btn btn-xs" onclick="showOverrideOutput('${esc(t.id)}')">Override Output</button>`;
    } else if (t.status === "failed" || t.status === "skipped") {
      taskActions = `<button class="btn btn-xs" onclick="taskAction('${esc(t.id)}', 'retry')">Retry</button>`;
    }
    if (t.status === "failed" || t.status === "pending_approval") {
      taskActions += `<button class="btn btn-xs" onclick="showOverrideOutput('${esc(t.id)}')">Override</button>`;
    }
    if (t.status !== "completed" && t.status !== "skipped" && t.status !== "running" && t.status !== "pending_approval") {
      taskActions += `<button class="btn btn-xs" onclick="taskAction('${esc(t.id)}', 'skip')">Skip</button>`;
    }
    if (t.status === "queued" || t.status === "pending_approval") {
      taskActions += `<button class="btn btn-xs" onclick="taskAction('${esc(t.id)}', 'pause')">Pause</button>`;
    }
    if (t.status === "paused") {
      taskActions += `<button class="btn btn-xs" onclick="taskAction('${esc(t.id)}', 'unpause')">Unpause</button>`;
    }
    if (canAddTasks && t.status !== "running") {
      taskActions += `<button class="btn btn-xs btn-danger" onclick="deleteTask('${esc(t.id)}')">Delete</button>`;
    }
    // Detail button for all tasks.
    taskActions += `<button class="btn btn-xs" onclick="showTaskDetail('${esc(t.id)}')">Detail</button>`;

    // Live execution info.
    let execInfo = "";
    if (t.status === "running") {
      const elapsed = t.started_at ? elapsedSince(t.started_at) : "";
      execInfo = `<div class="task-exec-info task-exec-running">
        <span class="task-exec-pulse"></span>
        ${t.assigned_instance ? `<span class="task-exec-instance">${esc(t.assigned_instance)}</span>` : ""}
        ${t.assigned_persona ? `<span class="task-exec-persona">${esc(t.assigned_persona)}</span>` : ""}
        ${elapsed ? `<span class="task-exec-elapsed">${elapsed}</span>` : ""}
        ${t.concern_id ? `<span class="task-exec-concern" title="${esc(t.concern_id)}">concern:${shortId(t.concern_id)}</span>` : ""}
      </div>`;
    } else if (t.status === "completed") {
      const output = t.output_data;
      let outputPreview = "";
      if (output && output.response) {
        outputPreview = `<div class="task-output-preview">${esc(truncate(output.response, 300))}</div>`;
      }
      execInfo = `<div class="task-exec-info task-exec-completed">
        ${t.assigned_instance ? `<span class="task-exec-instance">${esc(t.assigned_instance)}</span>` : ""}
        ${output?.persona ? `<span class="task-exec-persona">${esc(output.persona)}</span>` : ""}
        ${t.completed_at ? `<span class="task-exec-elapsed">completed ${timeAgo(t.completed_at)}</span>` : ""}
        ${outputPreview}
      </div>`;
    } else if (t.status === "retrying") {
      const retryAt = t.metadata?.retry_at;
      execInfo = `<div class="task-exec-info task-exec-retrying">
        <span class="task-exec-retry-count">Retry ${t.retry_count}/${t.max_retries}</span>
        ${retryAt ? `<span class="task-exec-elapsed">next attempt ${timeAgo(retryAt)}</span>` : ""}
      </div>`;
    } else if (t.status === "failed") {
      execInfo = `<div class="task-exec-info task-exec-failed">
        ${t.retry_count > 0 ? `<span class="task-exec-retry-count">${t.retry_count} retries exhausted</span>` : ""}
      </div>`;
    } else if (t.status === "pending_approval") {
      execInfo = `<div class="task-exec-info task-exec-pending-approval">
        <span class="task-exec-approval-icon">&#x1F6A7;</span>
        <span>Awaiting human approval before execution</span>
        ${t.requires_approval ? '<span class="task-exec-gate-badge">Approval Gate</span>' : ""}
      </div>`;
    }

    return `
      <div class="task-list-item task-status-${t.status}">
        <div class="task-list-header">
          <span class="task-list-name">${esc(t.name || shortId(t.id))}</span>
          <span class="swarm-status-badge status-task-${t.status}">${t.status}</span>
        </div>
        <div class="task-list-desc">${esc(truncate(t.description, 200))}</div>
        ${depLine}
        ${t.assigned_persona && t.status !== "running" && t.status !== "completed" ? `<div class="task-persona">Persona: ${esc(t.assigned_persona)}</div>` : ""}
        ${execInfo}
        ${t.error_message ? `<div class="task-error">${esc(t.error_message)}</div>` : ""}
        <div class="task-list-actions">${taskActions}</div>
      </div>
    `;
  }).join("");
}

/* ---- Tabs ---- */

function switchSwarmTab(btn, tabName) {
  // Deactivate all tabs.
  const parent = btn.closest(".swarm-detail");
  parent.querySelectorAll(".swarm-tab").forEach(t => t.classList.remove("active"));
  parent.querySelectorAll(".swarm-tab-content").forEach(t => t.classList.remove("active"));

  btn.classList.add("active");
  const content = parent.querySelector(`#swarm-tab-${tabName}`);
  if (content) content.classList.add("active");

  if (tabName === "audit") loadAuditLog();
  if (tabName === "checkpoints") loadCheckpoints();
  if (tabName === "gantt") loadGanttTimeline();
  if (tabName === "costs") loadCostDashboard();
  if (tabName === "files") loadSwarmFiles();
}

async function loadAuditLog() {
  if (!currentSwarm) return;
  const entries = await swarmAPI("GET", `/swarms/${currentSwarm}/audit`);
  const el = document.getElementById("audit-log-content");
  if (!el) return;

  if (!entries || entries._error || entries.length === 0) {
    el.innerHTML = '<p class="empty-state">No audit entries</p>';
    return;
  }

  el.innerHTML = entries.map(e => `
    <div class="audit-entry audit-severity-${e.severity || 'info'}">
      <span class="audit-type">${esc(e.event_type)}</span>
      <span class="audit-actor">${esc(e.actor || 'system')}</span>
      <span class="audit-time">${timeAgo(e.created_at)}</span>
      ${e.task_id ? `<span class="audit-task">${shortId(e.task_id)}</span>` : ""}
      ${Object.keys(e.details || {}).length > 0 ? `<div class="audit-details">${esc(JSON.stringify(e.details))}</div>` : ""}
    </div>
  `).join("");
}

/* ---- Swarm actions ---- */

async function swarmAction(action) {
  if (!currentSwarm) return;
  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/${action}`);
  if (result._error) {
    alert(`Action '${action}' failed: ${result.error || "Unknown error"}`);
  }
  _lastSwarmStatus = "";  // Force full re-render after state change.
  await refreshSwarmDetail();
}

/* ---- Decompose workflow ---- */

async function rephraseTask() {
  if (!currentSwarm) return;
  const btn = document.getElementById("btn-rephrase");
  if (btn) { btn.disabled = true; btn.textContent = "Rephrasing..."; }

  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/rephrase`, { model: getSelectedModel() });
  if (result._error) {
    alert("Rephrase failed: " + (result.error || "Unknown error"));
  }
  if (btn) { btn.disabled = false; btn.textContent = "Rephrase"; }
  await refreshSwarmDetail();
}

async function decomposeTask() {
  if (!currentSwarm) return;
  const btn = document.getElementById("btn-decompose");
  if (btn) { btn.disabled = true; btn.textContent = "Decomposing..."; }

  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/decompose`, { model: getSelectedModel() });
  if (result._error) {
    alert("Decomposition failed: " + (result.error || "Unknown error"));
  }
  if (btn) { btn.disabled = false; btn.textContent = "Decompose"; }
  await refreshSwarmDetail();
}

async function selectAgents() {
  if (!currentSwarm) return;
  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/select-agents`, { model: getSelectedModel() });
  if (result._error) {
    alert("Agent selection failed: " + (result.error || "Unknown error"));
    return;
  }
  await refreshSwarmDetail();
}

/* ---- Task CRUD ---- */

function showCreateTask() {
  if (!currentSwarm) return;

  const modal = $("#swarm-modal");
  if (!modal) return;

  // Get existing tasks for dependency selection.
  const taskItems = document.querySelectorAll(".task-list-item");
  let depsCheckboxes = "";

  // Fetch tasks from current swarm data.
  swarmAPI("GET", `/swarms/${currentSwarm}/tasks`).then(data => {
    if (data._error) return;
    const tasks = data.tasks || [];
    if (tasks.length > 0) {
      depsCheckboxes = '<div class="form-group"><label>Depends On</label><div class="deps-checkboxes">'
        + tasks.map(t => `
          <label class="checkbox-label">
            <input type="checkbox" class="dep-checkbox" value="${esc(t.id)}" />
            ${esc(t.name || shortId(t.id))}
          </label>
        `).join("")
        + '</div></div>';
    }

    $("#swarm-modal-title").textContent = "Add Task";
    $("#swarm-modal-body").innerHTML = `
      <div class="form-group">
        <label>Name</label>
        <input type="text" id="input-task-name" class="form-input" placeholder="Task name" />
      </div>
      <div class="form-group">
        <label>Description</label>
        <textarea id="input-task-desc" class="form-input" rows="4"
                  placeholder="What should this task accomplish?"></textarea>
      </div>
      ${depsCheckboxes}
      <div class="form-row">
        <div class="form-group">
          <label>Persona</label>
          <input type="text" id="input-task-persona" class="form-input" placeholder="(auto)" />
        </div>
        <div class="form-group">
          <label>Priority</label>
          <input type="number" id="input-task-priority" class="form-input" value="0" min="0" max="10" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Max Retries</label>
          <input type="number" id="input-task-retries" class="form-input" value="3" min="0" max="10" />
        </div>
        <div class="form-group">
          <label>Timeout (seconds)</label>
          <input type="number" id="input-task-timeout" class="form-input" value="600" min="30" />
        </div>
      </div>
      <div class="form-group">
        <label class="checkbox-label">
          <input type="checkbox" id="input-task-approval" />
          Requires human approval before execution
        </label>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" onclick="createTask()">Add Task</button>
        <button class="btn" onclick="closeSwarmModal()">Cancel</button>
      </div>
    `;
    modal.classList.remove("hidden");
    setTimeout(() => document.getElementById("input-task-name")?.focus(), 50);
  });
}

async function createTask() {
  const name = document.getElementById("input-task-name")?.value?.trim();
  const desc = document.getElementById("input-task-desc")?.value?.trim();
  const persona = document.getElementById("input-task-persona")?.value?.trim();
  const priority = parseInt(document.getElementById("input-task-priority")?.value || "0", 10);
  const retries = parseInt(document.getElementById("input-task-retries")?.value || "3", 10);
  const timeout = parseInt(document.getElementById("input-task-timeout")?.value || "600", 10);

  if (!desc) return;

  const deps = [];
  document.querySelectorAll(".dep-checkbox:checked").forEach(cb => deps.push(cb.value));

  const requiresApproval = document.getElementById("input-task-approval")?.checked || false;

  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/tasks`, {
    name: name || "",
    description: desc,
    assigned_persona: persona || "",
    priority,
    max_retries: retries,
    timeout_seconds: timeout,
    depends_on: deps,
    requires_approval: requiresApproval,
  });

  if (result._error) return;

  closeSwarmModal();
  await refreshSwarmDetail();
}

async function deleteTask(taskId) {
  await swarmAPI("DELETE", `/tasks/${taskId}`);
  await refreshSwarmDetail();
}

async function taskAction(taskId, action) {
  await swarmAPI("POST", `/tasks/${taskId}/${action}`);
  await refreshSwarmDetail();
}

async function autoLayoutDAG() {
  if (!currentSwarm) return;
  await swarmAPI("POST", `/swarms/${currentSwarm}/auto-layout`);
  await refreshSwarmDetail();
}

/* ---- DAG Canvas Drawing ---- */

const STATUS_COLORS = {
  queued: "#e0a830",
  waiting: "#e0a830",
  pending_approval: "#e8b020",
  running: "#6c8cff",
  completed: "#4caf7c",
  failed: "#d45050",
  retrying: "#d48040",
  paused: "#8888a0",
  skipped: "#555570",
};

/* ---- DAG Drag Interaction ---- */

let _dagDragState = null;  // { taskId, startX, startY, origX, origY, moved }
let _dagTasks = [];        // Current tasks reference for hit-testing.
let _dagEdges = [];        // Current edges reference for redraw.

function _initDAGDrag(canvas, tasks, edges) {
  _dagTasks = tasks;
  _dagEdges = edges;

  // Remove old listeners (re-init on each render).
  canvas.onmousedown = null;
  canvas.onmousemove = null;
  canvas.onmouseup = null;
  canvas.onmouseleave = null;
  canvas.style.cursor = "default";

  const nodeW = 180, nodeH = 60;

  function hitTask(mx, my) {
    for (const t of _dagTasks) {
      if (mx >= t.position_x && mx <= t.position_x + nodeW
          && my >= t.position_y && my <= t.position_y + nodeH) {
        return t;
      }
    }
    return null;
  }

  canvas.onmousedown = function(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const task = hitTask(mx, my);
    if (task) {
      _dagDragState = {
        taskId: task.id,
        startX: mx, startY: my,
        origX: task.position_x, origY: task.position_y,
        moved: false,
      };
      canvas.style.cursor = "grabbing";
      e.preventDefault();
    }
  };

  canvas.onmousemove = function(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (_dagDragState) {
      const dx = mx - _dagDragState.startX;
      const dy = my - _dagDragState.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) _dagDragState.moved = true;

      // Update task position locally for real-time feedback.
      const task = _dagTasks.find(t => t.id === _dagDragState.taskId);
      if (task) {
        task.position_x = Math.max(0, _dagDragState.origX + dx);
        task.position_y = Math.max(0, _dagDragState.origY + dy);
        drawDAG(_dagTasks, _dagEdges);
      }
    } else {
      // Show grab cursor on hover over nodes.
      canvas.style.cursor = hitTask(mx, my) ? "grab" : "default";
    }
  };

  canvas.onmouseup = function(e) {
    if (!_dagDragState) return;

    const state = _dagDragState;
    _dagDragState = null;
    canvas.style.cursor = "default";

    if (state.moved) {
      // Persist new position to server.
      const task = _dagTasks.find(t => t.id === state.taskId);
      if (task) {
        swarmAPI("PUT", `/tasks/${task.id}`, {
          position_x: task.position_x,
          position_y: task.position_y,
        });
      }
    } else {
      // Short click — show task detail.
      showTaskDetail(state.taskId);
    }
  };

  canvas.onmouseleave = function() {
    if (_dagDragState) {
      // Cancel drag — revert position.
      const task = _dagTasks.find(t => t.id === _dagDragState.taskId);
      if (task) {
        task.position_x = _dagDragState.origX;
        task.position_y = _dagDragState.origY;
        drawDAG(_dagTasks, _dagEdges);
      }
      _dagDragState = null;
      canvas.style.cursor = "default";
    }
  };
}

function resizeDAGCanvas() {
  if (!dagCanvas) return;
  const container = dagCanvas.parentElement;
  if (!container) return;
  dagCanvas.width = container.clientWidth || 800;
  dagCanvas.height = Math.max(400, container.clientHeight || 400);
}

function drawDAG(tasks, edges) {
  if (!dagCtx || !dagCanvas) return;
  const ctx = dagCtx;
  const w = dagCanvas.width;
  const h = dagCanvas.height;

  ctx.clearRect(0, 0, w, h);

  if (!tasks || tasks.length === 0) {
    ctx.fillStyle = "#8888a0";
    ctx.font = "14px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Add tasks to build the DAG", w / 2, h / 2);
    return;
  }

  const nodeW = 180;
  const nodeH = 60;
  const taskMap = {};
  tasks.forEach(t => taskMap[t.id] = t);

  // Draw edges first (behind nodes).
  ctx.strokeStyle = "#2a2e3d";
  ctx.lineWidth = 2;
  for (const edge of edges) {
    const from = taskMap[edge.from_task_id];
    const to = taskMap[edge.to_task_id];
    if (!from || !to) continue;

    const x1 = from.position_x + nodeW;
    const y1 = from.position_y + nodeH / 2;
    const x2 = to.position_x;
    const y2 = to.position_y + nodeH / 2;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    // Bezier curve for smooth edge.
    const cpx = (x1 + x2) / 2;
    ctx.bezierCurveTo(cpx, y1, cpx, y2, x2, y2);
    ctx.stroke();

    // Arrow head.
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.fillStyle = "#2a2e3d";
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 10 * Math.cos(angle - 0.3), y2 - 10 * Math.sin(angle - 0.3));
    ctx.lineTo(x2 - 10 * Math.cos(angle + 0.3), y2 - 10 * Math.sin(angle + 0.3));
    ctx.closePath();
    ctx.fill();
  }

  // Draw nodes.
  for (const task of tasks) {
    const x = task.position_x;
    const y = task.position_y;
    const color = STATUS_COLORS[task.status] || "#8888a0";

    // Node background.
    ctx.fillStyle = "#1a1d27";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (task.status === "pending_approval") {
      ctx.setLineDash([6, 4]);
    } else {
      ctx.setLineDash([]);
    }
    roundRect(ctx, x, y, nodeW, nodeH, 8);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // Status dot.
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x + 14, y + 16, 5, 0, Math.PI * 2);
    ctx.fill();

    // Task name.
    ctx.fillStyle = "#e0e0e6";
    ctx.font = "bold 12px -apple-system, sans-serif";
    ctx.textAlign = "left";
    const label = task.name || task.id.slice(0, 8);
    ctx.fillText(truncate(label, 18), x + 26, y + 20);

    // Status text + elapsed time for running tasks.
    ctx.fillStyle = color;
    ctx.font = "10px -apple-system, sans-serif";
    let statusLabel = task.status;
    if (task.status === "running" && task.started_at) {
      statusLabel += " · " + elapsedSince(task.started_at);
    } else if (task.status === "retrying") {
      statusLabel += ` (${task.retry_count || 0}/${task.max_retries || 3})`;
    } else if (task.status === "pending_approval") {
      statusLabel = "needs approval";
    }
    ctx.fillText(statusLabel, x + 26, y + 36);

    // Persona / instance if assigned.
    const subLabel = task.status === "running" && task.assigned_instance
      ? task.assigned_instance
      : task.assigned_persona || "";
    if (subLabel) {
      ctx.fillStyle = "#8888a0";
      ctx.font = "10px -apple-system, sans-serif";
      ctx.fillText(truncate(subLabel, 20), x + 26, y + 50);
    }
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/* ---- Task approval & intervention ---- */

async function approveTask(taskId) {
  await swarmAPI("POST", `/tasks/${taskId}/approve`, { approved_by: "user" });
  await refreshSwarmDetail();
}

async function rejectTask(taskId) {
  const reason = prompt("Reason for rejection (optional):");
  await swarmAPI("POST", `/tasks/${taskId}/reject`, { reason: reason || "Rejected by user" });
  await refreshSwarmDetail();
}

function showOverrideOutput(taskId) {
  const modal = $("#swarm-modal");
  if (!modal) return;

  $("#swarm-modal-title").textContent = "Override Task Output";
  $("#swarm-modal-body").innerHTML = `
    <p class="form-hint">Provide output data to manually complete this task. Downstream tasks will receive this as input.</p>
    <div class="form-group">
      <label>Response Text</label>
      <textarea id="input-override-response" class="form-input" rows="8"
                placeholder="Enter the output text for this task..."></textarea>
    </div>
    <div class="form-actions">
      <button class="btn btn-primary" onclick="submitOverrideOutput('${esc(taskId)}')">Override &amp; Complete</button>
      <button class="btn" onclick="closeSwarmModal()">Cancel</button>
    </div>
  `;
  modal.classList.remove("hidden");
  setTimeout(() => document.getElementById("input-override-response")?.focus(), 50);
}

async function submitOverrideOutput(taskId) {
  const response = document.getElementById("input-override-response")?.value || "";
  await swarmAPI("POST", `/tasks/${taskId}/override-output`, {
    output_data: { response },
  });
  closeSwarmModal();
  await refreshSwarmDetail();
}

/* ---- Task detail modal ---- */

async function showTaskDetail(taskId) {
  const data = await swarmAPI("GET", `/swarms/${currentSwarm}`);
  if (data._error) return;

  const tasks = data.tasks || [];
  const edges = data.edges || [];
  const task = tasks.find(t => t.id === taskId);
  if (!task) return;

  const modal = $("#swarm-modal");
  if (!modal) return;

  // Build dependency info.
  const depNames = edges
    .filter(e => e.to_task_id === taskId)
    .map(e => {
      const dt = tasks.find(t => t.id === e.from_task_id);
      return dt ? esc(dt.name || shortId(e.from_task_id)) : shortId(e.from_task_id);
    }).join(", ");

  const successorNames = edges
    .filter(e => e.from_task_id === taskId)
    .map(e => {
      const dt = tasks.find(t => t.id === e.to_task_id);
      return dt ? esc(dt.name || shortId(e.to_task_id)) : shortId(e.to_task_id);
    }).join(", ");

  // Build action buttons.
  let actions = "";
  if (task.status === "pending_approval") {
    actions += `<button class="btn btn-sm btn-approve" onclick="approveTask('${esc(taskId)}');closeSwarmModal()">Approve</button>`;
    actions += `<button class="btn btn-sm btn-danger" onclick="rejectTask('${esc(taskId)}');closeSwarmModal()">Reject</button>`;
  }
  if (task.status === "failed" || task.status === "skipped") {
    actions += `<button class="btn btn-sm" onclick="taskAction('${esc(taskId)}','retry');closeSwarmModal()">Retry</button>`;
  }
  if (task.status === "failed" || task.status === "pending_approval" || task.status === "queued") {
    actions += `<button class="btn btn-sm" onclick="showOverrideOutput('${esc(taskId)}')">Override Output</button>`;
  }

  // Format input: show predecessor responses readably.
  let inputHtml = "<em>None</em>";
  if (task.input_data && Object.keys(task.input_data).length > 0) {
    inputHtml = Object.entries(task.input_data).map(([key, val]) => {
      if (val && typeof val === "object" && val.response) {
        return `<div class="task-detail-input-block"><div class="md-label">${esc(key)}</div>${renderMd(val.response)}</div>`;
      }
      return `<div class="task-detail-input-block"><div class="md-label">${esc(key)}</div><pre class="task-detail-json">${esc(typeof val === "string" ? val : JSON.stringify(val, null, 2))}</pre></div>`;
    }).join("");
  }

  // Format output: render response as markdown if available.
  let outputHtml = "<em>None</em>";
  if (task.output_data && Object.keys(task.output_data).length > 0) {
    if (task.output_data.response) {
      outputHtml = `<div class="task-detail-response">${renderMd(task.output_data.response)}</div>`;
    } else {
      outputHtml = `<pre class="task-detail-json">${esc(JSON.stringify(task.output_data, null, 2))}</pre>`;
    }
  }

  $("#swarm-modal-title").textContent = task.name || shortId(taskId);
  $("#swarm-modal-body").innerHTML = `
    <div class="task-detail-modal">
      <div class="task-detail-row">
        <span class="swarm-status-badge status-task-${task.status}">${task.status}</span>
        ${task.requires_approval ? '<span class="task-exec-gate-badge">Approval Gate</span>' : ""}
        ${task.assigned_persona ? `<span class="task-exec-persona">${esc(task.assigned_persona)}</span>` : ""}
        ${task.assigned_instance ? `<span class="task-exec-instance">${esc(task.assigned_instance)}</span>` : ""}
      </div>

      <div class="task-detail-section">
        <strong>Description</strong>
        <div class="task-detail-desc">${renderMd(task.description)}</div>
      </div>

      ${depNames ? `<div class="task-detail-section"><strong>Depends on:</strong> ${depNames}</div>` : ""}
      ${successorNames ? `<div class="task-detail-section"><strong>Feeds into:</strong> ${successorNames}</div>` : ""}

      <div class="task-detail-section">
        <strong>Retry Policy</strong>
        <span>${task.retry_count}/${task.max_retries} retries, ${task.retry_backoff_seconds}s backoff, ${task.timeout_seconds}s timeout</span>
        ${task.timeout_warn_seconds > 0 ? `<span> (warn at ${task.timeout_warn_seconds}s, extend ${task.timeout_extend_seconds}s)</span>` : ""}
      </div>

      ${task.started_at ? `<div class="task-detail-section"><strong>Started:</strong> ${timeAgo(task.started_at)}</div>` : ""}
      ${task.completed_at ? `<div class="task-detail-section"><strong>Completed:</strong> ${timeAgo(task.completed_at)}</div>` : ""}
      ${task.error_message ? `<div class="task-detail-section"><strong>Error:</strong> <span class="task-error">${esc(task.error_message)}</span></div>` : ""}

      <div class="task-detail-section">
        <strong>Input Data</strong>
        ${inputHtml}
      </div>

      <div class="task-detail-section">
        <strong>Output / Result</strong>
        ${outputHtml}
      </div>

      ${(task.output_data?.files && task.output_data.files.length > 0) ? `
      <div class="task-detail-section">
        <strong>Files Produced</strong>
        <div class="files-list">
          ${task.output_data.files.map(f => `
            <div class="file-item">
              <span class="file-icon">${_fileIcon(f.mime_type || "")}</span>
              <div class="file-info">
                <a class="file-name" href="/api/swarm/swarms/${esc(currentSwarm)}/files/${esc(f.path)}"
                   target="_blank" download="${esc(f.filename)}">${esc(f.filename)}</a>
                <span class="file-meta">${_formatFileSize(f.size)}</span>
              </div>
            </div>
          `).join("")}
        </div>
      </div>` : ""}

      ${actions ? `<div class="task-detail-actions">${actions}</div>` : ""}
    </div>
  `;
  modal.classList.remove("hidden");
}

/* ---- Checkpoints ---- */

async function loadCheckpoints() {
  if (!currentSwarm) return;
  const checkpoints = await swarmAPI("GET", `/swarms/${currentSwarm}/checkpoints`);
  const el = document.getElementById("checkpoint-list-content");
  if (!el) return;

  if (!checkpoints || checkpoints._error || checkpoints.length === 0) {
    el.innerHTML = '<p class="empty-state">No checkpoints yet</p>';
    return;
  }

  el.innerHTML = checkpoints.map(c => {
    const taskCount = (c.task_states || []).length;
    const completedCount = (c.task_states || []).filter(t => t.status === "completed").length;
    return `
      <div class="checkpoint-item">
        <div class="checkpoint-header">
          <span class="checkpoint-label">${esc(c.label || 'Checkpoint')}</span>
          <span class="checkpoint-time">${timeAgo(c.created_at)}</span>
        </div>
        <div class="checkpoint-info">
          ${taskCount} tasks, ${completedCount} completed
        </div>
        <div class="checkpoint-actions">
          <button class="btn btn-xs btn-primary" onclick="restoreCheckpoint('${esc(c.id)}')">Restore</button>
          <button class="btn btn-xs btn-danger" onclick="deleteCheckpoint('${esc(c.id)}')">Delete</button>
        </div>
      </div>
    `;
  }).join("");
}

async function createCheckpoint() {
  if (!currentSwarm) return;
  const label = prompt("Checkpoint label (optional):");
  await swarmAPI("POST", `/swarms/${currentSwarm}/checkpoints`, { label: label || "Manual checkpoint" });
  loadCheckpoints();
}

async function restoreCheckpoint(checkpointId) {
  if (!confirm("Restore this checkpoint? The swarm will be set to 'ready' state and non-completed tasks will be reset to 'queued'.")) return;
  const result = await swarmAPI("POST", `/checkpoints/${checkpointId}/restore`);
  if (result._error) {
    alert("Restore failed: " + (result.error || "Unknown error"));
    return;
  }
  await refreshSwarmDetail();
}

async function deleteCheckpoint(checkpointId) {
  await swarmAPI("DELETE", `/checkpoints/${checkpointId}`);
  loadCheckpoints();
}

/* ---- Gantt timeline ---- */

async function loadGanttTimeline() {
  if (!currentSwarm) return;
  const data = await swarmAPI("GET", `/swarms/${currentSwarm}/timeline`);
  const el = document.getElementById("gantt-content");
  if (!el) return;

  if (!data || data._error || !data.tasks || data.tasks.length === 0) {
    el.innerHTML = '<p class="empty-state">No timeline data yet. Start the swarm to see execution timeline.</p>';
    return;
  }

  const swarmStart = data.swarm_started_at ? new Date(data.swarm_started_at.endsWith("Z") ? data.swarm_started_at : data.swarm_started_at + "Z").getTime() : 0;
  const nowMs = new Date(data.now.endsWith("Z") ? data.now : data.now + "Z").getTime();

  if (!swarmStart) {
    el.innerHTML = '<p class="empty-state">Swarm has not started yet.</p>';
    return;
  }

  const totalDuration = nowMs - swarmStart;
  const tasks = data.tasks.sort((a, b) => a.depth - b.depth || a.name.localeCompare(b.name));

  let html = '<div class="gantt-chart">';
  html += '<div class="gantt-header">';
  html += `<div class="gantt-label-col">Task</div>`;
  html += `<div class="gantt-bar-col">`;

  // Time markers.
  const markers = 5;
  for (let i = 0; i <= markers; i++) {
    const pct = (i / markers * 100).toFixed(1);
    const secs = Math.round(totalDuration / 1000 * i / markers);
    const label = secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m${secs % 60}s`;
    html += `<span class="gantt-marker" style="left:${pct}%">${label}</span>`;
  }
  html += `</div></div>`;

  for (const task of tasks) {
    const taskStart = task.started_at ? new Date(task.started_at.endsWith("Z") ? task.started_at : task.started_at + "Z").getTime() : 0;
    const taskEnd = task.completed_at ? new Date(task.completed_at.endsWith("Z") ? task.completed_at : task.completed_at + "Z").getTime() : (task.status === "running" ? nowMs : 0);

    const color = STATUS_COLORS[task.status] || "#8888a0";

    let barStyle = "";
    if (taskStart && totalDuration > 0) {
      const startPct = Math.max(0, (taskStart - swarmStart) / totalDuration * 100);
      const widthPct = Math.max(1, (taskEnd - taskStart) / totalDuration * 100);
      barStyle = `left:${startPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%;background:${color};`;
    }

    const duration = taskStart && taskEnd ? elapsedSince(task.started_at) : "";

    html += `<div class="gantt-row">`;
    html += `<div class="gantt-label-col">
      <span class="gantt-task-name">${esc(task.name || shortId(task.id))}</span>
      <span class="swarm-status-badge status-task-${task.status}" style="font-size:9px;padding:1px 5px">${task.status}</span>
    </div>`;
    html += `<div class="gantt-bar-col">`;
    if (barStyle) {
      html += `<div class="gantt-bar ${task.status === 'running' ? 'gantt-bar-pulse' : ''}" style="${barStyle}" title="${duration}"></div>`;
    }
    html += `</div></div>`;
  }

  html += '</div>';
  el.innerHTML = html;
}

/* ---- Cost dashboard ---- */

async function loadCostDashboard() {
  if (!currentSwarm) return;
  const summary = await swarmAPI("GET", `/swarms/${currentSwarm}/costs`);
  const el = document.getElementById("costs-content");
  if (!el) return;

  if (!summary || summary._error) {
    el.innerHTML = '<p class="empty-state">Unable to load cost data</p>';
    return;
  }

  if (summary.total_tokens_in === 0 && summary.total_tokens_out === 0 && summary.total_cost_usd === 0) {
    el.innerHTML = '<p class="empty-state">No cost data recorded yet. Costs are tracked when tasks complete.</p>';
    return;
  }

  let html = '<div class="cost-dashboard">';

  // Summary cards.
  html += '<div class="cost-summary-row">';
  html += `<div class="cost-card"><div class="cost-card-value">${summary.total_tokens_in.toLocaleString()}</div><div class="cost-card-label">Tokens In</div></div>`;
  html += `<div class="cost-card"><div class="cost-card-value">${summary.total_tokens_out.toLocaleString()}</div><div class="cost-card-label">Tokens Out</div></div>`;
  html += `<div class="cost-card"><div class="cost-card-value">$${summary.total_cost_usd.toFixed(4)}</div><div class="cost-card-label">Total Cost</div></div>`;
  html += `<div class="cost-card"><div class="cost-card-value">${(summary.total_tokens_in + summary.total_tokens_out).toLocaleString()}</div><div class="cost-card-label">Total Tokens</div></div>`;
  html += '</div>';

  // Per-task breakdown.
  if (summary.by_task && summary.by_task.length > 0) {
    html += '<h4 class="cost-section-title">By Task</h4>';
    html += '<div class="cost-table">';
    html += '<div class="cost-table-header"><span>Task</span><span>Persona</span><span>Tokens In</span><span>Tokens Out</span><span>Cost</span></div>';
    for (const entry of summary.by_task) {
      html += `<div class="cost-table-row">
        <span>${esc(shortId(entry.task_id))}</span>
        <span>${esc(entry.persona_name || entry.instance_name || '-')}</span>
        <span>${entry.tokens_in.toLocaleString()}</span>
        <span>${entry.tokens_out.toLocaleString()}</span>
        <span>$${entry.cost_usd.toFixed(4)}</span>
      </div>`;
    }
    html += '</div>';
  }

  html += '</div>';
  el.innerHTML = html;
}

/* ---- Swarm files ---- */

function _formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

async function loadSwarmFiles() {
  if (!currentSwarm) return;
  const data = await swarmAPI("GET", `/swarms/${currentSwarm}/files`);
  const el = document.getElementById("files-content");
  if (!el) return;

  if (!data || data._error) {
    el.innerHTML = '<p class="empty-state">Unable to load files</p>';
    return;
  }

  const files = data.files || [];

  if (files.length === 0) {
    el.innerHTML = '<p class="empty-state">No files yet. Files created by agents during task execution will appear here.</p>';
    return;
  }

  let html = '<div class="files-dashboard">';
  html += `<div class="files-summary"><span class="stat-pill">${files.length} files</span>`;
  html += `<span class="stat-pill">${_formatFileSize(data.total_size || 0)} total</span></div>`;

  // Group by agent.
  const byAgent = {};
  for (const f of files) {
    const agent = f.agent || "unknown";
    if (!byAgent[agent]) byAgent[agent] = [];
    byAgent[agent].push(f);
  }

  for (const [agent, agentFiles] of Object.entries(byAgent)) {
    html += `<div class="files-agent-group">`;
    html += `<h4 class="files-agent-name">${esc(agent)}</h4>`;
    html += '<div class="files-list">';

    for (const f of agentFiles) {
      const icon = _fileIcon(f.mime_type || "");
      html += `
        <div class="file-item">
          <span class="file-icon">${icon}</span>
          <div class="file-info">
            <a class="file-name" href="/api/swarm/swarms/${esc(currentSwarm)}/files/${esc(f.path)}"
               target="_blank" download="${esc(f.filename)}">${esc(f.filename)}</a>
            <span class="file-meta">${_formatFileSize(f.size)} &middot; ${esc(f.mime_type || 'unknown')}</span>
          </div>
        </div>
      `;
    }
    html += '</div></div>';
  }

  html += '</div>';
  el.innerHTML = html;
}

function _fileIcon(mime) {
  if (mime.startsWith("image/")) return "🖼️";
  if (mime.includes("pdf")) return "📄";
  if (mime.includes("spreadsheet") || mime.includes("xlsx") || mime.includes("csv")) return "📊";
  if (mime.includes("presentation") || mime.includes("pptx")) return "📽️";
  if (mime.includes("word") || mime.includes("docx")) return "📝";
  if (mime.includes("zip") || mime.includes("compress") || mime.includes("archive")) return "📦";
  if (mime.includes("json") || mime.includes("xml") || mime.includes("yaml")) return "📋";
  if (mime.includes("text")) return "📃";
  return "📎";
}

/* ---- Templates ---- */

async function saveAsTemplate() {
  if (!currentSwarm) return;
  const name = prompt("Template name:");
  if (!name) return;

  const result = await swarmAPI("POST", `/swarms/${currentSwarm}/save-as-template`, {
    name,
    description: "",
  });
  if (result._error) {
    alert("Failed to save template: " + (result.error || "Unknown error"));
    return;
  }
  alert("Template saved: " + result.name);
}

async function showTemplates() {
  const templates = await swarmAPI("GET", "/templates");
  if (templates._error) return;

  const modal = $("#swarm-modal");
  if (!modal) return;

  let list = "";
  if (!templates || templates.length === 0) {
    list = '<p class="empty-state">No templates saved yet. Save a swarm as a template from the toolbar.</p>';
  } else {
    list = templates.map(t => {
      const taskCount = (t.dag_definition?.tasks || []).length;
      return `
        <div class="template-item">
          <div class="template-header">
            <span class="template-name">${esc(t.name)}</span>
            <span class="template-task-count">${taskCount} tasks</span>
          </div>
          <div class="template-desc">${esc(truncate(t.description, 120))}</div>
          <div class="template-meta">${timeAgo(t.created_at)}</div>
          <div class="template-actions">
            <button class="btn btn-xs btn-primary" onclick="instantiateTemplate('${esc(t.id)}')">Use Template</button>
            <button class="btn btn-xs btn-danger" onclick="deleteTemplate('${esc(t.id)}')">Delete</button>
          </div>
        </div>
      `;
    }).join("");
  }

  $("#swarm-modal-title").textContent = "Templates";
  $("#swarm-modal-body").innerHTML = `
    <div class="template-list">${list}</div>
  `;
  modal.classList.remove("hidden");
}

async function instantiateTemplate(templateId) {
  if (!currentProject) {
    alert("Select a project first");
    return;
  }

  const name = prompt("Swarm name for this template instance:");
  if (!name) return;

  const result = await swarmAPI("POST", `/templates/${templateId}/instantiate`, {
    project_id: currentProject,
    name,
    task: "",
  });
  if (result._error) {
    alert("Failed to create swarm from template: " + (result.error || "Unknown error"));
    return;
  }

  closeSwarmModal();
  selectSwarm(result.id);

  // Refresh project to update swarm list.
  const project = await swarmAPI("GET", `/projects/${currentProject}`);
  if (!project._error) renderSwarmList(project.swarms || []);
}

async function deleteTemplate(templateId) {
  if (!confirm("Delete this template?")) return;
  await swarmAPI("DELETE", `/templates/${templateId}`);
  showTemplates();
}

/* ---- Swarm modal ---- */

function closeSwarmModal() {
  const modal = $("#swarm-modal");
  if (modal) modal.classList.add("hidden");
}
