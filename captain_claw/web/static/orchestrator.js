/* Captain Claw — Orchestrator Dashboard */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    let ws = null;
    let isConnected = false;
    let tasks = {};              // id → task object
    let graphSummary = {};       // { total, completed, failed, running, ready, blocked }
    let eventLog = [];           // chronological event entries
    let selectedTaskId = null;
    let orchestratorState = 'idle'; // idle | running | completed | failed
    let isRephrasing = false;
    let availableSkills = [];     // [{name, skill_name, description}]
    let selectedSkills = new Set(); // skill names currently selected

    // ── DOM ──────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);

    const orchInput = $('#orchInput');
    const orchPrepareBtn = $('#orchPrepareBtn');
    const orchRunBtn = $('#orchRunBtn');
    const orchDiscardBtn = $('#orchDiscardBtn');
    const orchRephrase = $('#orchRephrase');
    const orchRephrased = $('#orchRephrased');
    const orchRephraseLoading = $('#orchRephraseLoading');
    const orchStatusDot = $('#orchStatusDot');
    const orchStatusText = $('#orchStatusText');
    const orchGraph = $('#orchGraph');
    const orchGraphEmpty = $('#orchGraphEmpty');
    const orchDetail = $('#orchDetail');
    const detailTitle = $('#detailTitle');
    const detailBody = $('#detailBody');
    const detailClose = $('#detailClose');
    const orchLogMessages = $('#orchLogMessages');
    const orchLogEmpty = $('#orchLogEmpty');
    const logClearBtn = $('#logClearBtn');

    // Summary bar counts
    const summaryTotal = $('#summaryTotal .count');
    const summaryCompleted = $('#summaryCompleted .count');
    const summaryRunning = $('#summaryRunning .count');
    const summaryFailed = $('#summaryFailed .count');
    const summaryPending = $('#summaryPending .count');

    // ── WebSocket ────────────────────────────────────────────

    function connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/ws`;
        ws = new WebSocket(url);

        ws.onopen = () => {
            isConnected = true;
            setConnectionStatus(true);
        };

        ws.onclose = () => {
            isConnected = false;
            setConnectionStatus(false);
            setTimeout(connect, 3000);
        };

        ws.onerror = () => {
            isConnected = false;
            setConnectionStatus(false);
        };

        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                handleMessage(msg);
            } catch (e) {
                // ignore parse errors
            }
        };
    }

    function setConnectionStatus(connected) {
        if (!connected) {
            orchStatusDot.className = 'orch-status-dot';
            orchStatusText.textContent = 'disconnected';
        } else if (orchestratorState === 'idle') {
            orchStatusDot.className = 'orch-status-dot idle';
            orchStatusText.textContent = 'idle';
        }
    }

    // ── Message handling ─────────────────────────────────────

    function handleMessage(msg) {
        switch (msg.type) {
            case 'orchestrator_event':
                handleOrchestratorEvent(msg);
                break;
            case 'status':
                // Show status changes during orchestration
                if (orchestratorState === 'running' && msg.status) {
                    if (msg.status === 'ready') {
                        // Server done processing
                    } else if (msg.status === 'thinking') {
                        addLogEntry('progress', 'Processing...');
                    }
                }
                break;
            case 'chat_message':
                if (msg.role === 'user') {
                    addLogEntry('progress', 'Request submitted');
                } else if (msg.role === 'assistant' && orchestratorState === 'running') {
                    addLogEntry('completed', 'Final synthesized result received');
                    setOrchestratorState('completed');
                }
                break;
            case 'error':
                addLogEntry('task_failed', `Server error: ${msg.message || 'unknown'}`);
                if (orchestratorState === 'running') {
                    setOrchestratorState('failed');
                }
                break;
            case 'monitor':
            case 'tool_output':
                // Show tool execution events during orchestration
                if (orchestratorState === 'running') {
                    const toolName = msg.tool_name || msg.tool || '';
                    if (toolName && toolName !== 'orchestrator') {
                        addLogEntry('progress', `Tool: ${toolName}`);
                    }
                }
                break;
        }
    }

    function handleOrchestratorEvent(msg) {
        const event = msg.event;

        // Update graph summary if present
        if (msg.graph) {
            graphSummary = msg.graph;
            updateSummaryBar();
        }

        switch (event) {
            case 'decomposing':
                setOrchestratorState('running');
                addLogEntry('progress', 'Decomposing request into tasks...');
                break;
            case 'decomposed':
                setOrchestratorState('running');
                handleDecomposed(msg);
                break;
            case 'building_graph':
                addLogEntry('progress', `Building task graph (${msg.task_count || '?'} tasks)...`);
                break;
            case 'assigning_sessions':
                addLogEntry('progress', `Assigning sessions to ${msg.task_count || '?'} tasks...`);
                break;
            case 'assigned':
                handleAssigned(msg);
                break;
            case 'executing':
                addLogEntry('progress', `Starting parallel execution of ${msg.task_count || '?'} tasks...`);
                break;
            case 'task_started':
                handleTaskStarted(msg);
                break;
            case 'task_completed':
                handleTaskCompleted(msg);
                break;
            case 'task_failed':
                handleTaskFailed(msg);
                break;
            case 'progress':
                updateSummaryBar();
                renderGraph();
                break;
            case 'synthesizing':
                addLogEntry('synthesizing', 'Synthesizing final result...');
                break;
            case 'completed':
                handleCompleted(msg);
                break;
            case 'error':
                addLogEntry('task_failed', msg.message || 'An error occurred');
                setOrchestratorState('failed');
                break;
            default:
                // Log unknown events for visibility
                addLogEntry('progress', `${event}: ${JSON.stringify(msg).slice(0, 200)}`);
                break;
        }
    }

    // ── Event handlers ───────────────────────────────────────

    function handleDecomposed(msg) {
        const tasksList = msg.tasks || [];
        tasks = {};

        tasksList.forEach(t => {
            tasks[t.id] = {
                id: t.id,
                title: t.title || t.id,
                description: t.description || '',
                depends_on: t.depends_on || [],
                session_id: t.session_id || null,
                session_name: t.session_name || null,
                status: 'pending',
                result: null,
                error: null,
                retries: 0,
                startTime: null,
                endTime: null,
            };
        });

        addLogEntry('decomposed', `Decomposed into ${tasksList.length} tasks: ${msg.summary || ''}`);
        renderGraph();
    }

    function handleAssigned(msg) {
        const tasksList = msg.tasks || [];
        tasksList.forEach(t => {
            if (tasks[t.id]) {
                tasks[t.id].session_id = t.session_id || null;
                tasks[t.id].status = t.status || tasks[t.id].status;
            }
        });
        addLogEntry('assigned', 'Sessions assigned to all tasks');
        renderGraph();
    }

    function handleTaskStarted(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'running';
            tasks[tid].session_id = msg.session_id || tasks[tid].session_id;
            tasks[tid].startTime = Date.now();
        }
        addLogEntry('task_started', `Started: ${msg.title || tid}`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTaskCompleted(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'completed';
            tasks[tid].result = msg.output || '';
            tasks[tid].endTime = Date.now();
            tasks[tid].usage = msg.usage || null;
            tasks[tid].context = msg.context || null;
        }

        // Build context metrics string for the log entry
        const metrics = formatContextMetrics(msg.usage, msg.context);
        addLogEntry('task_completed', `Completed: ${msg.title || tid}`, metrics);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTaskFailed(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'failed';
            tasks[tid].error = msg.error || 'Unknown error';
            tasks[tid].endTime = Date.now();
        }
        addLogEntry('task_failed', `Failed: ${msg.title || tid} — ${msg.error || '?'}`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleCompleted(msg) {
        setOrchestratorState(msg.has_failures ? 'failed' : 'completed');
        addLogEntry('completed', msg.has_failures
            ? 'Orchestration completed with failures'
            : 'Orchestration completed successfully');
    }

    // ── Orchestrator state ───────────────────────────────────

    function setOrchestratorState(state) {
        orchestratorState = state;
        orchStatusDot.className = `orch-status-dot ${state}`;
        orchStatusText.textContent = state;
        orchRunBtn.disabled = (state === 'running');
        orchPrepareBtn.disabled = (state === 'running');
    }

    // ── Summary bar ──────────────────────────────────────────

    function updateSummaryBar() {
        const s = graphSummary;
        summaryTotal.textContent = s.total || Object.keys(tasks).length || 0;
        summaryCompleted.textContent = s.completed || 0;
        summaryRunning.textContent = s.running || 0;
        summaryFailed.textContent = s.failed || 0;
        summaryPending.textContent = (s.ready || 0) + (s.blocked || 0);
    }

    // ── Graph rendering ──────────────────────────────────────

    function renderGraph() {
        const taskIds = Object.keys(tasks);
        if (taskIds.length === 0) {
            orchGraphEmpty.style.display = '';
            return;
        }
        orchGraphEmpty.style.display = 'none';

        // Compute dependency depth for layering
        const depths = computeDepths();
        const maxDepth = Math.max(...Object.values(depths), 0);

        // Group tasks by depth
        const layers = {};
        for (let d = 0; d <= maxDepth; d++) layers[d] = [];
        taskIds.forEach(tid => {
            const d = depths[tid] || 0;
            layers[d].push(tid);
        });

        // Build flow HTML
        let html = '<div class="orch-graph-flow">';
        for (let d = 0; d <= maxDepth; d++) {
            const layerTasks = layers[d];
            if (layerTasks.length === 0) continue;

            const label = d === 0 ? 'Start' : (d === maxDepth && maxDepth > 0 ? 'Final' : `Layer ${d}`);
            html += `<div class="orch-graph-layer">`;
            html += `<div class="orch-layer-label">${label}</div>`;

            layerTasks.forEach(tid => {
                const t = tasks[tid];
                const selected = tid === selectedTaskId ? ' selected' : '';
                const statusClass = t.status || 'pending';

                html += `<div class="task-card ${statusClass}${selected}" data-task-id="${tid}">`;
                html += `<div class="task-card-header">`;
                html += `<span class="task-card-id">${esc(tid)}</span>`;
                html += `<span class="task-card-status ${statusClass}">${esc(statusClass)}</span>`;
                html += `</div>`;
                html += `<div class="task-card-title">${esc(t.title)}</div>`;

                if (t.session_id) {
                    const shortSession = t.session_id.length > 12 ? t.session_id.slice(0, 12) + '...' : t.session_id;
                    html += `<div class="task-card-session">${esc(shortSession)}</div>`;
                }

                if (t.depends_on && t.depends_on.length > 0) {
                    html += `<div class="task-card-deps">depends: ${t.depends_on.map(esc).join(', ')}</div>`;
                }

                html += `</div>`;
            });

            html += `</div>`;
        }
        html += '</div>';

        orchGraph.innerHTML = html;

        // Attach click handlers to task cards
        orchGraph.querySelectorAll('.task-card').forEach(card => {
            card.addEventListener('click', () => {
                const tid = card.dataset.taskId;
                selectTask(tid);
            });
        });
    }

    function computeDepths() {
        const depths = {};
        const taskIds = Object.keys(tasks);

        // Initialize
        taskIds.forEach(tid => { depths[tid] = 0; });

        // Iterative depth calculation
        let changed = true;
        let iterations = 0;
        while (changed && iterations < 100) {
            changed = false;
            iterations++;
            taskIds.forEach(tid => {
                const t = tasks[tid];
                if (!t.depends_on || t.depends_on.length === 0) return;
                t.depends_on.forEach(depId => {
                    if (depths[depId] !== undefined) {
                        const newDepth = depths[depId] + 1;
                        if (newDepth > depths[tid]) {
                            depths[tid] = newDepth;
                            changed = true;
                        }
                    }
                });
            });
        }

        return depths;
    }

    // ── Task selection / detail ───────────────────────────────

    function selectTask(tid) {
        if (selectedTaskId === tid) {
            // Deselect
            selectedTaskId = null;
            orchDetail.classList.remove('visible');
            renderGraph();
            return;
        }
        selectedTaskId = tid;
        renderGraph();
        showTaskDetail(tid);
    }

    function showTaskDetail(tid) {
        const t = tasks[tid];
        if (!t) return;

        detailTitle.textContent = `${t.title} (${tid})`;

        let html = '<dl>';
        html += `<dt>Status</dt><dd>${esc(t.status)}</dd>`;
        html += `<dt>Session</dt><dd>${esc(t.session_id || '—')}</dd>`;
        html += `<dt>Depends on</dt><dd>${t.depends_on && t.depends_on.length ? t.depends_on.map(esc).join(', ') : 'none'}</dd>`;
        html += `<dt>Retries</dt><dd>${t.retries || 0}</dd>`;

        if (t.startTime) {
            const elapsed = ((t.endTime || Date.now()) - t.startTime) / 1000;
            html += `<dt>Elapsed</dt><dd>${elapsed.toFixed(1)}s</dd>`;
        }

        // Usage / context metrics
        if (t.usage || t.context) {
            const u = t.usage || {};
            const c = t.context || {};
            html += '<dt>Tokens</dt><dd>';
            if (u.total_tokens) {
                html += `${formatTokenCount(u.total_tokens)} total (${formatTokenCount(u.prompt_tokens || 0)} in / ${formatTokenCount(u.completion_tokens || 0)} out)`;
            } else {
                html += '—';
            }
            html += '</dd>';
            html += `<dt>Context</dt><dd>`;
            const parts = [];
            if (c.budget) parts.push(`budget: ${formatTokenCount(c.budget)}`);
            if (c.utilization) parts.push(`used: ${c.utilization}%`);
            if (c.messages) parts.push(`msgs: ${c.messages}`);
            html += parts.length > 0 ? parts.join(' · ') : '—';
            html += '</dd>';
        }

        html += '</dl>';

        // Description
        if (t.description) {
            html += `<div style="margin-top:10px;font-size:12px;color:var(--text-secondary);">${esc(t.description)}</div>`;
        }

        // Result output
        if (t.result) {
            html += `<div class="orch-detail-output">${esc(t.result)}</div>`;
        }

        // Error
        if (t.error) {
            html += `<div class="orch-detail-output orch-detail-error">${esc(t.error)}</div>`;
        }

        detailBody.innerHTML = html;
        orchDetail.classList.add('visible');
    }

    function updateSelectedDetail() {
        if (selectedTaskId && tasks[selectedTaskId]) {
            showTaskDetail(selectedTaskId);
        }
    }

    // ── Event log ────────────────────────────────────────────

    function formatContextMetrics(usage, context) {
        if (!usage && !context) return null;

        const parts = [];

        if (usage) {
            const total = usage.total_tokens || 0;
            const prompt = usage.prompt_tokens || 0;
            const completion = usage.completion_tokens || 0;
            if (total > 0) {
                parts.push({ label: 'tokens', value: formatTokenCount(total), cls: '' });
                parts.push({ label: 'in/out', value: `${formatTokenCount(prompt)}/${formatTokenCount(completion)}`, cls: '' });
            }
        }

        if (context) {
            const budget = context.budget || 0;
            const utilization = context.utilization || 0;
            const messages = context.messages || 0;
            if (budget > 0) {
                parts.push({ label: 'budget', value: formatTokenCount(budget), cls: '' });
            }
            if (utilization > 0) {
                parts.push({ label: 'used', value: `${utilization}%`, cls: utilization > 80 ? 'high-util' : '' });
            }
            if (messages > 0) {
                parts.push({ label: 'msgs', value: String(messages), cls: '' });
            }
        }

        return parts;
    }

    function formatTokenCount(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
        return String(n);
    }

    function addLogEntry(event, text, metrics) {
        const now = new Date();
        const ts = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

        eventLog.push({ time: ts, event, text });

        // Hide empty state
        if (orchLogEmpty) orchLogEmpty.style.display = 'none';

        const entry = document.createElement('div');
        entry.className = 'log-entry';

        let html = `<span class="log-entry-time">${ts}</span><span class="log-entry-event ${event}">${esc(event)}</span><span class="log-entry-text">${esc(text)}</span>`;

        // Append context metrics row if present
        if (metrics && metrics.length > 0) {
            html += '<div class="log-entry-metrics">';
            metrics.forEach(m => {
                const cls = m.cls ? ` ${m.cls}` : '';
                html += `<span class="log-metric"><span class="log-metric-label">${esc(m.label)}</span> <span class="log-metric-value${cls}">${esc(m.value)}</span></span>`;
            });
            html += '</div>';
        }

        entry.innerHTML = html;
        orchLogMessages.appendChild(entry);

        // Auto-scroll to bottom
        orchLogMessages.scrollTop = orchLogMessages.scrollHeight;
    }

    function clearLog() {
        eventLog = [];
        orchLogMessages.innerHTML = '';
        if (orchLogEmpty) {
            orchLogMessages.appendChild(orchLogEmpty);
            orchLogEmpty.style.display = '';
        }
    }

    // ── Prepare (rephrase) ───────────────────────────────────

    async function prepareOrchestrator() {
        const input = orchInput.value.trim();
        if (!input || isRephrasing) return;

        isRephrasing = true;
        orchPrepareBtn.disabled = true;
        orchRephraseLoading.classList.remove('hidden');
        orchRephrase.classList.add('hidden');

        try {
            const resp = await fetch('/api/orchestrator/rephrase', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input }),
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }

            const data = await resp.json();
            const rephrased = data.rephrased || input;

            // Show the rephrase area with the result
            orchRephrased.value = rephrased;
            orchRephrase.classList.remove('hidden');
            orchRephrased.focus();

            // Auto-resize textarea to fit content
            autoResizeRephrased();
        } catch (e) {
            // On failure, show original text for manual editing
            orchRephrased.value = input;
            orchRephrase.classList.remove('hidden');
            orchRephrased.focus();
            autoResizeRephrased();
        } finally {
            isRephrasing = false;
            orchPrepareBtn.disabled = false;
            orchRephraseLoading.classList.add('hidden');
        }
    }

    function discardRephrase() {
        orchRephrase.classList.add('hidden');
        orchRephrased.value = '';
        orchInput.focus();
    }

    function autoResizeRephrased() {
        orchRephrased.style.height = 'auto';
        orchRephrased.style.height = Math.min(orchRephrased.scrollHeight, 200) + 'px';
    }

    // ── Run orchestrator ─────────────────────────────────────

    function runOrchestrator() {
        // Use rephrased text if visible, otherwise fall back to input
        const rephraseVisible = !orchRephrase.classList.contains('hidden');
        const input = rephraseVisible
            ? orchRephrased.value.trim()
            : orchInput.value.trim();

        if (!input || !isConnected) return;

        // Hide rephrase area
        orchRephrase.classList.add('hidden');
        orchRephrased.value = '';

        // Reset state
        tasks = {};
        graphSummary = {};
        selectedTaskId = null;
        orchDetail.classList.remove('visible');
        setOrchestratorState('running');
        clearLog();
        addLogEntry('progress', `Submitting: ${input}`);
        renderGraph();
        updateSummaryBar();

        // Send as a command message with /orchestrate prefix, appending skill filter if needed
        const skillsParam = getSelectedSkillsParam();
        ws.send(JSON.stringify({
            type: 'command',
            command: `/orchestrate ${input}${skillsParam}`,
        }));

        orchInput.value = '';
    }

    // ── Helpers ──────────────────────────────────────────────

    function esc(str) {
        if (str === null || str === undefined) return '';
        const d = document.createElement('div');
        d.textContent = String(str);
        return d.innerHTML;
    }

    // ── Skills ───────────────────────────────────────────

    async function loadSkills() {
        const orchSkillsList = $('#orchSkillsList');
        if (!orchSkillsList) return;

        try {
            const resp = await fetch('/api/orchestrator/skills');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            availableSkills = data.skills || [];

            if (availableSkills.length === 0) {
                orchSkillsList.innerHTML = '<span class="orch-skills-loading">no skills available</span>';
                return;
            }

            // All skills selected by default
            selectedSkills = new Set(availableSkills.map(s => s.skill_name || s.name));
            renderSkills();
        } catch (e) {
            orchSkillsList.innerHTML = '<span class="orch-skills-loading">failed to load skills</span>';
        }
    }

    function renderSkills() {
        const orchSkillsList = $('#orchSkillsList');
        if (!orchSkillsList) return;

        orchSkillsList.innerHTML = '';

        availableSkills.forEach(skill => {
            const skillId = skill.skill_name || skill.name;
            const isActive = selectedSkills.has(skillId);

            const chip = document.createElement('label');
            chip.className = `orch-skill-chip${isActive ? ' active' : ''}`;
            chip.title = skill.description || skillId;

            chip.innerHTML = `<span class="chip-dot"></span><span>${esc(skill.name)}</span>`;

            chip.addEventListener('click', (e) => {
                e.preventDefault();
                if (selectedSkills.has(skillId)) {
                    selectedSkills.delete(skillId);
                    chip.classList.remove('active');
                } else {
                    selectedSkills.add(skillId);
                    chip.classList.add('active');
                }
            });

            orchSkillsList.appendChild(chip);
        });
    }

    function getSelectedSkillsParam() {
        // If all skills are selected, no need to pass filter
        if (selectedSkills.size === availableSkills.length) return '';
        if (selectedSkills.size === 0) return '';
        return ` [skills: ${Array.from(selectedSkills).join(', ')}]`;
    }

    // ── Init ─────────────────────────────────────────────────

    function init() {
        // Connect WebSocket
        connect();

        // Prepare button → rephrase
        orchPrepareBtn.addEventListener('click', prepareOrchestrator);

        // Enter key in input → prepare (rephrase)
        orchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                prepareOrchestrator();
            }
        });

        // Run button → execute rephrased prompt
        orchRunBtn.addEventListener('click', runOrchestrator);

        // Ctrl+Enter in rephrased textarea → run
        orchRephrased.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                runOrchestrator();
            }
        });

        // Auto-resize rephrased textarea on input
        orchRephrased.addEventListener('input', autoResizeRephrased);

        // Discard button
        orchDiscardBtn.addEventListener('click', discardRephrase);

        // Detail close
        detailClose.addEventListener('click', () => {
            selectedTaskId = null;
            orchDetail.classList.remove('visible');
            renderGraph();
        });

        // Clear log
        logClearBtn.addEventListener('click', clearLog);

        // Load skills and current status on page load
        loadSkills();
        loadCurrentStatus();
    }

    async function loadCurrentStatus() {
        try {
            const resp = await fetch('/api/orchestrator/status');
            const data = await resp.json();
            if (data.status && data.status.tasks && data.status.tasks.length > 0) {
                // Restore task state from existing orchestration
                data.status.tasks.forEach(t => {
                    tasks[t.id] = {
                        id: t.id,
                        title: t.title || t.id,
                        description: t.description || '',
                        depends_on: t.depends_on || [],
                        session_id: t.session_id || null,
                        session_name: null,
                        status: t.status || 'pending',
                        result: t.result_preview || null,
                        error: t.error || null,
                        retries: t.retries || 0,
                        startTime: null,
                        endTime: null,
                    };
                });
                graphSummary = data.status.summary || {};
                updateSummaryBar();
                renderGraph();

                // Determine orchestrator state
                const hasRunning = data.status.tasks.some(t => t.status === 'running');
                const hasFailed = data.status.tasks.some(t => t.status === 'failed');
                const allDone = data.status.tasks.every(t => t.status === 'completed' || t.status === 'failed');

                if (hasRunning) {
                    setOrchestratorState('running');
                } else if (allDone && hasFailed) {
                    setOrchestratorState('failed');
                } else if (allDone) {
                    setOrchestratorState('completed');
                }

                addLogEntry('progress', 'Loaded existing orchestration state');
            }
        } catch (e) {
            // API not available or no active orchestration
        }
    }

    // Start
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
