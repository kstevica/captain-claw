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
    let orchestratorState = 'idle'; // idle | preview | running | completed | failed
    let isRephrasing = false;
    let isDecomposing = false;
    let availableSkills = [];     // [{name, skill_name, description}]
    let selectedSkills = new Set(); // skill names currently selected
    let editingTaskId = null;     // task currently being edited (during running state)
    let aggregatedContext = { totalTokens: 0, promptTokens: 0, completionTokens: 0 };
    let timeoutCountdowns = {};   // task_id → remaining seconds
    let countdownInterval = null; // client-side 1s countdown ticker
    let workflowName = '';        // current workflow name
    let taskOverrides = {};       // tid → {title, description, session_id, model_id, skills}
    let availableSessions = [];   // [{id, name}]
    let availableModels = [];     // [{id, provider, model}]
    let workflowVariables = [];   // [{name, label, default}]
    let variableValues = {};      // name → current input value

    // ── DOM ──────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);

    const orchInput = $('#orchInput');
    const orchPrepareBtn = $('#orchPrepareBtn');
    const orchRunBtn = $('#orchRunBtn');
    const orchDiscardBtn = $('#orchDiscardBtn');
    const orchRephrase = $('#orchRephrase');
    const orchRephrased = $('#orchRephrased');
    const orchRephraseLoading = $('#orchRephraseLoading');
    const orchDecomposeLoading = $('#orchDecomposeLoading');
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
    const orchWorkflowName = $('#orchWorkflowName');
    const orchWorkflowBar = $('#orchWorkflowBar');
    const orchWorkflowNameInput = $('#orchWorkflowNameInput');
    const orchSaveWorkflowBtn = $('#orchSaveWorkflowBtn');
    const orchLoadWorkflowSelect = $('#orchLoadWorkflowSelect');
    const orchExecuteBtn = $('#orchExecuteBtn');
    const orchResetBtn = $('#orchResetBtn');

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
                addLogEntry('progress', 'Decomposing request into tasks...');
                break;
            case 'decomposed':
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
            case 'task_paused':
                handleTaskPaused(msg);
                break;
            case 'task_editing':
                handleTaskEditingEvent(msg);
                break;
            case 'task_updated':
                handleTaskUpdated(msg);
                break;
            case 'task_restarted':
                handleTaskRestarted(msg);
                break;
            case 'task_resumed':
                handleTaskResumed(msg);
                break;
            case 'timeout_warning':
                handleTimeoutWarning(msg);
                break;
            case 'timeout_countdown':
                handleTimeoutCountdown(msg);
                break;
            case 'timeout_postponed':
                handleTimeoutPostponed(msg);
                break;
            case 'workflow_saved':
                addLogEntry('completed', `Workflow saved: ${msg.name || ''}`);
                loadWorkflowList();
                break;
            case 'output_saved':
                addLogEntry('completed', `Run output saved: ${msg.filename || ''}`);
                break;
            case 'error':
                console.error('[orch-event] error event received', msg);
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
        console.log('[orch-decomposed] WS event received', { taskCount: (msg.tasks || []).length, summary: msg.summary, workflowName: msg.workflow_name, hasVariables: !!(msg.variables && msg.variables.length), keys: Object.keys(msg) });
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
                model_id: t.model_id || '',
                skills: t.skills || [],
                status: 'pending',
                result: null,
                error: null,
                retries: 0,
                startTime: null,
                endTime: null,
            };
        });

        if (msg.workflow_name) {
            workflowName = msg.workflow_name;
            updateWorkflowNameDisplay();
        }

        // Show the rephrased prompt that was used to decompose (e.g. from loaded workflow).
        if (msg.user_input && !orchRephrased.value) {
            orchRephrased.value = msg.user_input;
            syncPaneHeights();
        }

        // Apply workflow variables if present (from loaded workflow or fresh decompose).
        if (msg.variables && msg.variables.length > 0) {
            applyVariables(msg.variables);
        } else {
            // Auto-detect {{var}} from user_input and task descriptions when
            // the backend didn't supply a variables array (e.g. older server).
            var detected = detectVariables(msg);
            if (detected.length > 0) {
                applyVariables(detected);
            } else {
                applyVariables([]);
            }
        }

        addLogEntry('decomposed', `Decomposed into ${tasksList.length} tasks: ${msg.summary || ''}`);

        // If we're in decompose-only mode (preview), enter preview state.
        // If we're in running mode (from /orchestrate wrapper), keep running.
        if (orchestratorState !== 'running') {
            setOrchestratorState('preview');
        }

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

        // Aggregate token consumption across all tasks
        if (msg.usage) {
            aggregatedContext.totalTokens += (msg.usage.total_tokens || 0);
            aggregatedContext.promptTokens += (msg.usage.prompt_tokens || 0);
            aggregatedContext.completionTokens += (msg.usage.completion_tokens || 0);
        }
        updateContextSummary();

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
        // Clean up timeout countdown state.
        delete timeoutCountdowns[tid];
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

    function handleTaskPaused(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'paused';
        }
        addLogEntry('task_paused', `Paused: ${msg.title || tid}`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTaskEditingEvent(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'editing';
            tasks[tid].description = msg.description || tasks[tid].description;
        }
        addLogEntry('task_editing', `Editing: ${msg.title || tid}`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTaskUpdated(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].description = msg.description || tasks[tid].description;
        }
        addLogEntry('progress', `Instructions updated: ${msg.title || tid}`);
        updateSelectedDetail();
    }

    function handleTaskRestarted(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'pending';
            tasks[tid].result = null;
            tasks[tid].error = null;
            tasks[tid].startTime = null;
            tasks[tid].endTime = null;
            tasks[tid].usage = null;
            tasks[tid].context = null;
        }
        // Clean up timeout countdown state.
        delete timeoutCountdowns[tid];
        const reason = msg.reason === 'timeout' ? ' (timeout)' : '';
        addLogEntry('task_restarted', `Restarted${reason}: ${msg.title || tid}`);
        setOrchestratorState('running');
        renderGraph();
        updateSelectedDetail();
    }

    function handleTaskResumed(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'pending';
        }
        editingTaskId = null;
        addLogEntry('progress', `Resumed: ${msg.title || tid}`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTimeoutWarning(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'timeout_warning';
        }
        timeoutCountdowns[tid] = msg.remaining_seconds || 60;
        startCountdownTicker();
        addLogEntry('task_failed', `\u23F1 Timeout warning: ${msg.title || tid} \u2014 will restart in ${timeoutCountdowns[tid]}s`);
        renderGraph();
        updateSelectedDetail();
    }

    function handleTimeoutCountdown(msg) {
        const taskList = msg.tasks || [];
        taskList.forEach(function (entry) {
            timeoutCountdowns[entry.task_id] = entry.remaining_seconds;
        });
        // Remove countdown entries for tasks no longer in warning
        Object.keys(timeoutCountdowns).forEach(function (tid) {
            if (!taskList.find(function (e) { return e.task_id === tid; })) {
                delete timeoutCountdowns[tid];
            }
        });
        if (Object.keys(timeoutCountdowns).length > 0) {
            startCountdownTicker();
        }
        updateCountdownDisplays();
        updateSelectedDetail();
    }

    function handleTimeoutPostponed(msg) {
        const tid = msg.task_id;
        if (tasks[tid]) {
            tasks[tid].status = 'running';
        }
        delete timeoutCountdowns[tid];
        addLogEntry('progress', `\u23F1 Timeout postponed: ${msg.title || tid} \u2014 timer reset`);
        renderGraph();
        updateSelectedDetail();
    }

    function startCountdownTicker() {
        if (countdownInterval) return; // already running
        countdownInterval = setInterval(function () {
            let anyActive = false;
            Object.keys(timeoutCountdowns).forEach(function (tid) {
                if (timeoutCountdowns[tid] > 0) {
                    timeoutCountdowns[tid]--;
                    anyActive = true;
                }
            });
            if (!anyActive) {
                clearInterval(countdownInterval);
                countdownInterval = null;
            }
            updateCountdownDisplays();
        }, 1000);
    }

    function updateCountdownDisplays() {
        // Update countdown badges on task cards
        Object.keys(timeoutCountdowns).forEach(function (tid) {
            const card = orchGraph.querySelector(`.task-card[data-task-id="${tid}"]`);
            if (!card) return;
            let badge = card.querySelector('.timeout-countdown-badge');
            if (timeoutCountdowns[tid] > 0) {
                if (!badge) {
                    badge = document.createElement('div');
                    badge.className = 'timeout-countdown-badge';
                    card.appendChild(badge);
                }
                badge.textContent = `\u23F1 Restarting in ${timeoutCountdowns[tid]}s`;
            } else if (badge) {
                badge.remove();
            }
        });
        // Update detail panel countdown if visible
        if (selectedTaskId && timeoutCountdowns[selectedTaskId] !== undefined) {
            const countdownEl = detailBody.querySelector('.detail-timeout-countdown');
            if (countdownEl) {
                countdownEl.textContent = `Restarting in ${timeoutCountdowns[selectedTaskId]}s`;
            }
        }
    }

    // ── Orchestrator state ───────────────────────────────────

    function setOrchestratorState(state) {
        orchestratorState = state;
        orchStatusDot.className = `orch-status-dot ${state}`;
        orchStatusText.textContent = state;

        // Button states
        orchRunBtn.disabled = (state === 'running' || state === 'preview');
        orchPrepareBtn.disabled = (state === 'running');

        // Show/hide workflow bar & execute button
        if (state === 'preview') {
            orchWorkflowBar.classList.remove('hidden');
            orchExecuteBtn.disabled = false;
        } else if (state === 'running') {
            orchWorkflowBar.classList.remove('hidden');
            orchExecuteBtn.disabled = true;
        } else if (state === 'completed' || state === 'failed') {
            orchWorkflowBar.classList.remove('hidden');
            orchExecuteBtn.disabled = false;  // allow re-execute
        } else {
            orchWorkflowBar.classList.add('hidden');
        }
    }

    // ── Workflow name display ────────────────────────────────

    function updateWorkflowNameDisplay() {
        if (orchWorkflowName) {
            orchWorkflowName.textContent = workflowName ? `\u2014 ${workflowName}` : '';
        }
        if (orchWorkflowNameInput) {
            orchWorkflowNameInput.value = workflowName;
        }
    }

    // ── Summary bar ──────────────────────────────────────────

    function updateSummaryBar() {
        const s = graphSummary;
        summaryTotal.textContent = s.total || Object.keys(tasks).length || 0;
        summaryCompleted.textContent = s.completed || 0;
        summaryRunning.textContent = s.running || 0;
        summaryFailed.textContent = s.failed || 0;
        summaryPending.textContent = (s.ready || 0) + (s.blocked || 0);
        const pausedEl = document.querySelector('#summaryPaused .count');
        if (pausedEl) pausedEl.textContent = s.paused || 0;
    }

    function updateContextSummary() {
        var ctxIn = document.getElementById('summaryCtxIn');
        var ctxOut = document.getElementById('summaryCtxOut');
        var ctxTotal = document.getElementById('summaryCtxTotal');
        if (!ctxTotal) return;

        if (aggregatedContext.totalTokens === 0) {
            ctxIn.textContent = '\u2014';
            ctxOut.textContent = '\u2014';
            ctxTotal.textContent = '\u2014';
            return;
        }

        ctxIn.textContent = formatTokenCount(aggregatedContext.promptTokens);
        ctxOut.textContent = formatTokenCount(aggregatedContext.completionTokens);
        ctxTotal.textContent = formatTokenCount(aggregatedContext.totalTokens);
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
                const hasOverride = taskOverrides[tid] ? ' has-override' : '';

                html += `<div class="task-card ${statusClass}${selected}${hasOverride}" data-task-id="${tid}">`;
                html += `<div class="task-card-header">`;
                html += `<span class="task-card-id">${esc(tid)}</span>`;
                html += `<span class="task-card-status ${statusClass}">${esc(statusClass)}</span>`;
                html += `</div>`;
                html += `<div class="task-card-title">${esc((taskOverrides[tid] && taskOverrides[tid].title) || t.title)}</div>`;

                if (t.session_id) {
                    html += `<div class="task-card-session">${esc(t.session_id)}</div>`;
                }

                // Show token consumption and context utilization on card
                if (t.usage || t.context) {
                    var u = t.usage || {};
                    var c = t.context || {};
                    var ctxParts = [];
                    if (u.total_tokens) ctxParts.push(formatTokenCount(u.total_tokens) + ' tok');
                    if (c.budget) ctxParts.push(formatTokenCount(c.prompt_tokens || c.budget) + '/' + formatTokenCount(c.budget));
                    if (c.utilization) ctxParts.push(c.utilization + '%');
                    if (ctxParts.length > 0) {
                        var highCls = (c.utilization && c.utilization > 80) ? ' high-util' : '';
                        html += '<div class="task-card-ctx' + highCls + '">' + ctxParts.join(' \u00B7 ') + '</div>';
                    }
                }

                // Show per-task override indicators in preview
                if (orchestratorState === 'preview' && taskOverrides[tid]) {
                    const ov = taskOverrides[tid];
                    const parts = [];
                    if (ov.model_id) parts.push(ov.model_id);
                    if (ov.session_id) parts.push('session');
                    if (parts.length) {
                        html += `<div class="task-card-override">${parts.join(' \u00B7 ')}</div>`;
                    }
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
            hideDetailPanel();
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

        const displayTitle = (taskOverrides[tid] && taskOverrides[tid].title) || t.title;
        detailTitle.textContent = `${displayTitle} (${tid})`;

        let html = '';

        // ── Preview mode: editable fields ──
        if (orchestratorState === 'preview') {
            html += buildPreviewEditor(tid, t);
        } else {
            // ── Action bar (running/completed mode) ──
            const actionHtml = buildActionButtons(t);
            if (actionHtml) {
                html += `<div class="task-actions">${actionHtml}</div>`;
            }

            // ── Timeout warning banner ──
            if (t.status === 'timeout_warning' && timeoutCountdowns[tid] !== undefined) {
                html += '<div class="timeout-warning-banner">';
                html += '<span class="timeout-warning-icon">\u23F1</span>';
                html += '<span class="detail-timeout-countdown">Restarting in ' + timeoutCountdowns[tid] + 's</span>';
                html += '</div>';
            }

            // ── Session pipeline (orchestrator stages) ──
            html += buildSessionPipeline();

            // ── Task pipeline visualization ──
            html += '<div class="task-section-label">Task Pipeline</div>';
            html += buildPipeline(t);

            // ── Metadata ──
            html += '<dl>';
            html += `<dt>Status</dt><dd><span class="status-badge ${t.status}">${esc(t.status)}</span></dd>`;
            html += `<dt>Session</dt><dd><span class="detail-session-id">${esc(t.session_id || '\u2014')}</span></dd>`;
            html += `<dt>Depends on</dt><dd>${t.depends_on && t.depends_on.length ? t.depends_on.map(esc).join(', ') : 'none'}</dd>`;
            html += `<dt>Retries</dt><dd>${t.retries || 0}</dd>`;

            if (t.startTime) {
                const elapsed = ((t.endTime || Date.now()) - t.startTime) / 1000;
                html += `<dt>Elapsed</dt><dd>${elapsed.toFixed(1)}s</dd>`;
            }

            // Token consumption for this session
            if (t.usage) {
                const u = t.usage;
                html += '<dt>Tokens</dt><dd>';
                if (u.total_tokens) {
                    html += `<strong>${formatTokenCount(u.total_tokens)}</strong> total`;
                    html += ` <span class="detail-tokens-breakdown">(${formatTokenCount(u.prompt_tokens || 0)} in / ${formatTokenCount(u.completion_tokens || 0)} out)</span>`;
                } else {
                    html += '\u2014';
                }
                html += '</dd>';
            }

            // Session context status (tokens / cap / %)
            if (t.context) {
                const c = t.context;
                html += '<dt>Context</dt><dd>';
                if (c.budget) {
                    const promptTok = c.prompt_tokens || 0;
                    const pct = c.utilization || 0;
                    const pctCls = pct > 80 ? ' ctx-high' : '';
                    html += `<span class="detail-ctx-bar">`;
                    html += `${formatTokenCount(promptTok)} / ${formatTokenCount(c.budget)}`;
                    if (pct > 0) {
                        html += ` <span class="detail-ctx-pct${pctCls}">${pct}%</span>`;
                    }
                    html += `</span>`;
                    if (c.messages) {
                        html += ` \u00B7 ${c.messages} msgs`;
                    }
                } else {
                    html += '\u2014';
                }
                html += '</dd>';
            }

            html += '</dl>';

            // ── Instructions section (running/completed mode) ──
            html += '<div class="task-instructions-section">';
            html += '<div class="task-instructions-label">Instructions</div>';
            if (editingTaskId === tid) {
                html += `<textarea class="task-instructions-editor" id="taskInstructionsEditor">${esc(t.description || '')}</textarea>`;
                html += '<div class="task-instructions-actions">';
                html += '<button class="btn-save-instructions">Save &amp; Run</button>';
                html += '<button class="btn-cancel-edit">Cancel</button>';
                html += '</div>';
            } else {
                const desc = t.description || '';
                html += `<div class="task-instructions-content">${desc ? esc(desc) : '<em style="color:var(--text-muted);">No instructions</em>'}</div>`;
            }
            html += '</div>';
        }

        // ── Output ──
        if (t.result) {
            html += '<div class="task-section-label">Output</div>';
            html += `<div class="orch-detail-output">${esc(t.result)}</div>`;
        }

        // ── Error ──
        if (t.error) {
            html += '<div class="task-section-label">Error</div>';
            html += `<div class="orch-detail-output orch-detail-error">${esc(t.error)}</div>`;
        }

        detailBody.innerHTML = html;
        showDetailPanel();

        // Wire up action buttons and editor buttons
        if (orchestratorState === 'preview') {
            wirePreviewEditorButtons(tid);
        } else {
            wireActionButtons(tid);
        }

        // Auto-focus editor if in edit mode
        if (editingTaskId === tid) {
            const editor = detailBody.querySelector('#taskInstructionsEditor');
            if (editor) editor.focus();
        }
    }

    function updateSelectedDetail() {
        if (selectedTaskId && tasks[selectedTaskId]) {
            showTaskDetail(selectedTaskId);
        }
    }

    // ── Preview editor (task editing before execution) ────────

    function buildPreviewEditor(tid, t) {
        const ov = taskOverrides[tid] || {};
        const title = ov.title !== undefined ? ov.title : t.title;
        const desc = ov.description !== undefined ? ov.description : t.description;
        const sessionId = ov.session_id || t.session_id || '';
        const modelId = ov.model_id || t.model_id || '';

        let html = '<div class="preview-editor">';

        // Title
        html += '<div class="preview-field">';
        html += '<label class="preview-field-label">Title</label>';
        html += `<input type="text" class="preview-field-input" id="previewTitle" value="${esc(title)}">`;
        html += '</div>';

        // Description
        html += '<div class="preview-field">';
        html += '<label class="preview-field-label">Instructions</label>';
        html += `<textarea class="preview-field-textarea" id="previewDescription">${esc(desc)}</textarea>`;
        html += '</div>';

        // Session selector
        html += '<div class="preview-field">';
        html += '<label class="preview-field-label">Session</label>';
        html += '<select class="preview-field-select" id="previewSession">';
        html += '<option value="">New session (auto)</option>';
        availableSessions.forEach(s => {
            const sel = sessionId === s.id ? ' selected' : '';
            html += `<option value="${esc(s.id)}"${sel}>${esc(s.name)} (${esc(s.id.slice(0, 8))})</option>`;
        });
        html += '</select>';
        html += '</div>';

        // Model selector
        html += '<div class="preview-field">';
        html += '<label class="preview-field-label">Model</label>';
        html += '<select class="preview-field-select" id="previewModel">';
        html += '<option value="">Default</option>';
        availableModels.forEach(m => {
            const mId = m.id || `${m.provider}:${m.model}`;
            const sel = modelId === mId ? ' selected' : '';
            html += `<option value="${esc(mId)}"${sel}>${esc(mId)}</option>`;
        });
        html += '</select>';
        html += '</div>';

        // Dependencies (read-only)
        html += '<div class="preview-field">';
        html += '<label class="preview-field-label">Depends on</label>';
        html += `<div class="preview-field-value">${t.depends_on && t.depends_on.length ? t.depends_on.map(esc).join(', ') : 'none'}</div>`;
        html += '</div>';

        // Save button
        html += '<div class="preview-editor-actions">';
        html += '<button class="btn-save-preview" id="btnSavePreview">Save Changes</button>';
        if (taskOverrides[tid]) {
            html += '<button class="btn-reset-preview" id="btnResetPreview">Reset</button>';
        }
        html += '</div>';

        html += '</div>';
        return html;
    }

    function wirePreviewEditorButtons(tid) {
        const saveBtn = detailBody.querySelector('#btnSavePreview');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                const titleEl = detailBody.querySelector('#previewTitle');
                const descEl = detailBody.querySelector('#previewDescription');
                const sessionEl = detailBody.querySelector('#previewSession');
                const modelEl = detailBody.querySelector('#previewModel');

                const ov = {};
                if (titleEl && titleEl.value.trim() !== tasks[tid].title) {
                    ov.title = titleEl.value.trim();
                }
                if (descEl && descEl.value !== tasks[tid].description) {
                    ov.description = descEl.value;
                }
                if (sessionEl && sessionEl.value) {
                    ov.session_id = sessionEl.value;
                }
                if (modelEl && modelEl.value) {
                    ov.model_id = modelEl.value;
                }

                if (Object.keys(ov).length > 0) {
                    taskOverrides[tid] = { ...(taskOverrides[tid] || {}), ...ov };
                    // Also update the local tasks for display
                    if (ov.title) tasks[tid].title = ov.title;
                    if (ov.description !== undefined) tasks[tid].description = ov.description;
                    addLogEntry('progress', `Task ${tid} configured`);
                    renderGraph();
                    showTaskDetail(tid);
                }
            });
        }

        const resetBtn = detailBody.querySelector('#btnResetPreview');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                delete taskOverrides[tid];
                addLogEntry('progress', `Task ${tid} reset to defaults`);
                renderGraph();
                showTaskDetail(tid);
            });
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
        orchRephrased.value = '';

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

            orchRephrased.value = rephrased;
            orchRephrased.focus();
            syncPaneHeights();
        } catch (e) {
            // On failure, copy original text for manual editing
            orchRephrased.value = input;
            orchRephrased.focus();
            syncPaneHeights();
        } finally {
            isRephrasing = false;
            orchPrepareBtn.disabled = false;
            orchRephraseLoading.classList.add('hidden');
        }
    }

    function discardRephrase() {
        orchRephrased.value = '';
        orchInput.focus();
        syncPaneHeights();
    }

    function autoResizeRephrased() {
        syncPaneHeights();
    }

    function syncPaneHeights() {
        // Reset both to auto so scrollHeight is accurate
        orchInput.style.height = 'auto';
        orchRephrased.style.height = 'auto';
        // Use the taller of the two, clamped to a reasonable max
        var h = Math.max(orchInput.scrollHeight, orchRephrased.scrollHeight, 80);
        h = Math.min(h, 300);
        orchInput.style.height = h + 'px';
        orchRephrased.style.height = h + 'px';
    }

    // ── Reset ────────────────────────────────────────────────

    async function resetOrchestrator() {
        if (!confirm('Reset the current orchestration? This will cancel any running tasks and clear the graph.')) {
            return;
        }

        try {
            orchResetBtn.disabled = true;
            orchResetBtn.textContent = 'Resetting...';

            await fetch('/api/orchestrator/reset', { method: 'POST' });

            // Reset client-side state
            tasks = {};
            graphSummary = {};
            eventLog = [];
            selectedTaskId = null;
            workflowName = '';
            taskOverrides = {};
            workflowVariables = [];
            variableValues = {};
            aggregatedContext = { totalTokens: 0, promptTokens: 0, completionTokens: 0 };
            timeoutCountdowns = {};
            if (countdownInterval) {
                clearInterval(countdownInterval);
                countdownInterval = null;
            }

            // Reset UI
            orchInput.value = '';
            orchRephrased.value = '';
            syncPaneHeights();
            hideVariablesPanel();
            updateWorkflowNameDisplay();
            clearLog();
            updateSummaryBar();
            updateContextSummary();

            // Reset graph area
            orchGraph.innerHTML = '';
            if (orchGraphEmpty) {
                orchGraph.appendChild(orchGraphEmpty);
                orchGraphEmpty.style.display = '';
            }

            // Hide detail panel
            hideDetailPanel();

            // Return to idle state
            setOrchestratorState('idle');

        } catch (e) {
            alert('Reset failed: ' + e.message);
        } finally {
            orchResetBtn.disabled = false;
            orchResetBtn.textContent = 'Reset';
        }
    }

    // ── Decompose (was "Run") — now does prepare() only ──────

    async function decomposeOrchestrator() {
        // Use rephrased text if available, otherwise fall back to input
        const rephrasedText = orchRephrased.value.trim();
        const input = rephrasedText || orchInput.value.trim();

        console.log('[orch-decompose] start', { inputLen: input.length, inputPreview: input.slice(0, 200), isDecomposing: isDecomposing });
        if (!input || isDecomposing) {
            console.warn('[orch-decompose] aborted: empty input or already decomposing', { hasInput: !!input, isDecomposing: isDecomposing });
            return;
        }

        isDecomposing = true;
        orchRunBtn.disabled = true;
        orchDecomposeLoading.classList.remove('hidden');

        // Reset state
        tasks = {};
        graphSummary = {};
        selectedTaskId = null;
        taskOverrides = {};
        workflowVariables = [];
        variableValues = {};
        hideVariablesPanel();
        aggregatedContext = { totalTokens: 0, promptTokens: 0, completionTokens: 0 };
        timeoutCountdowns = {};
        if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
        hideDetailPanel();
        clearLog();
        updateContextSummary();
        addLogEntry('progress', `Decomposing: ${input.slice(0, 200)}`);
        renderGraph();
        updateSummaryBar();

        try {
            const skillsParam = getSelectedSkillsParam();
            const fullInput = input + skillsParam;

            console.log('[orch-decompose] calling /api/orchestrator/prepare', { fullInputLen: fullInput.length, skills: skillsParam });
            const resp = await fetch('/api/orchestrator/prepare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: fullInput }),
            });

            console.log('[orch-decompose] response status', resp.status, resp.statusText);
            const data = await resp.json();
            console.log('[orch-decompose] response data', { ok: data.ok, error: data.error, taskCount: (data.tasks || []).length, keys: Object.keys(data) });
            if (!data.ok) {
                console.error('[orch-decompose] FAILED', data.error);
                addLogEntry('task_failed', `Decompose failed: ${data.error || 'unknown'}`);
                setOrchestratorState('failed');
            }
            // The prepare endpoint broadcasts 'decomposed' via WebSocket,
            // which is handled by handleDecomposed() above.
            // The state transitions to 'preview' there.
        } catch (e) {
            console.error('[orch-decompose] EXCEPTION', e);
            addLogEntry('task_failed', `Decompose error: ${e.message}`);
            setOrchestratorState('failed');
        } finally {
            isDecomposing = false;
            orchRunBtn.disabled = false;
            orchDecomposeLoading.classList.add('hidden');
        }
    }

    // ── Execute (after preview) ──────────────────────────────

    function executeOrchestrator() {
        if (!isConnected || orchestratorState === 'running') return;

        // Collect overrides and variable values
        const overrides = Object.keys(taskOverrides).length > 0 ? taskOverrides : null;
        const varVals = Object.keys(variableValues).length > 0 ? variableValues : null;

        setOrchestratorState('running');
        addLogEntry('progress', 'Starting execution...');

        // Build payload: new format includes both overrides and variable values
        var payload = '';
        if (overrides || varVals) {
            payload = JSON.stringify({
                task_overrides: overrides,
                variable_values: varVals,
            });
        }
        ws.send(JSON.stringify({
            type: 'command',
            command: `/orchestrate-execute ${payload}`.trim(),
        }));
    }

    // ── Legacy run (directly via /orchestrate for backward compat) ──

    function runOrchestrator() {
        // This is now "Decompose" — call decomposeOrchestrator
        decomposeOrchestrator();
    }

    // ── Detail panel visibility helper ──────────────────────

    function showDetailPanel() {
        orchDetail.classList.add('visible');
        if (orchDetail.parentNode) orchDetail.parentNode.classList.add('detail-open');
    }

    function hideDetailPanel() {
        orchDetail.classList.remove('visible');
        if (orchDetail.parentNode) orchDetail.parentNode.classList.remove('detail-open');
    }

    // ── Helpers ──────────────────────────────────────────────

    function esc(str) {
        if (str === null || str === undefined) return '';
        const d = document.createElement('div');
        d.textContent = String(str);
        return d.innerHTML;
    }

    // ── Workflow Variables Panel ─────────────────────────

    function renderVariablesPanel() {
        var panel = document.getElementById('orchVariablesPanel');
        if (!panel) {
            panel = document.createElement('div');
            panel.id = 'orchVariablesPanel';
            panel.className = 'orch-variables-panel';
            // Insert after orchWorkflowBar in the input section
            var bar = document.getElementById('orchWorkflowBar');
            if (bar && bar.parentNode) {
                bar.parentNode.insertBefore(panel, bar.nextSibling);
            }
        }

        if (!workflowVariables || workflowVariables.length === 0) {
            panel.style.display = 'none';
            return;
        }

        panel.style.display = '';
        var html = '<div class="orch-variables-header">';
        html += '<span class="orch-variables-label">Variables</span>';
        html += '<span class="orch-variables-count">' + workflowVariables.length + ' var' + (workflowVariables.length !== 1 ? 's' : '') + '</span>';
        html += '</div>';
        html += '<div class="orch-variables-grid">';

        workflowVariables.forEach(function (v) {
            var value = variableValues[v.name] || '';
            var label = v.label || v.name;
            html += '<div class="orch-variable-field">';
            html += '<label class="orch-variable-label" for="orchVar_' + esc(v.name) + '">' + esc(label) + '</label>';
            html += '<input type="text" class="orch-variable-input" id="orchVar_' + esc(v.name) + '" data-var-name="' + esc(v.name) + '" value="' + esc(value) + '" placeholder="' + esc(v.name) + '" spellcheck="false">';
            html += '</div>';
        });

        html += '</div>';
        panel.innerHTML = html;

        // Wire up live value capture
        panel.querySelectorAll('.orch-variable-input').forEach(function (input) {
            input.addEventListener('input', function () {
                variableValues[input.dataset.varName] = input.value;
            });
        });
    }

    function hideVariablesPanel() {
        var panel = document.getElementById('orchVariablesPanel');
        if (panel) panel.style.display = 'none';
    }

    function detectVariables(msg) {
        // Client-side fallback: scan texts for {{variable_name}} patterns.
        var texts = [];
        if (msg.user_input) texts.push(msg.user_input);
        var tasksList = msg.tasks || [];
        tasksList.forEach(function (t) {
            if (t.title) texts.push(t.title);
            if (t.description) texts.push(t.description);
        });
        // Also scan the rephrased textarea (covers fresh decompose case).
        if (orchRephrased && orchRephrased.value) texts.push(orchRephrased.value);

        var seen = {};
        var result = [];
        var re = /\{\{(\w+)\}\}/g;
        texts.forEach(function (text) {
            var m;
            while ((m = re.exec(text)) !== null) {
                var name = m[1];
                if (!seen[name]) {
                    seen[name] = true;
                    result.push({
                        name: name,
                        label: name.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); }),
                        'default': ''
                    });
                }
            }
            re.lastIndex = 0;  // reset for next string
        });
        return result;
    }

    function applyVariables(vars) {
        workflowVariables = vars || [];
        variableValues = {};
        workflowVariables.forEach(function (v) {
            variableValues[v.name] = v['default'] || '';
        });
        if (workflowVariables.length > 0) {
            renderVariablesPanel();
        } else {
            hideVariablesPanel();
        }
    }

    // ── Task API actions ────────────────────────────────

    async function apiTaskAction(action, taskId, extraBody) {
        try {
            const body = { task_id: taskId, ...(extraBody || {}) };
            const resp = await fetch(`/api/orchestrator/task/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (!data.ok) {
                addLogEntry('task_failed', `Action "${action}" failed: ${data.error || 'unknown'}`);
            }
            return data;
        } catch (e) {
            addLogEntry('task_failed', `Action "${action}" error: ${e.message}`);
            return { ok: false, error: e.message };
        }
    }

    function buildActionButtons(t) {
        const status = t.status;
        let html = '';

        if (status === 'pending' || status === 'queued') {
            html += '<button class="task-btn task-btn-edit" data-action="edit">Edit Instructions</button>';
        } else if (status === 'running') {
            html += '<button class="task-btn task-btn-pause" data-action="pause">Pause</button>';
        } else if (status === 'timeout_warning') {
            html += '<button class="task-btn task-btn-postpone" data-action="postpone">Postpone (5 min)</button>';
        } else if (status === 'paused') {
            html += '<button class="task-btn task-btn-edit" data-action="edit">Edit Instructions</button>';
            html += '<button class="task-btn task-btn-resume" data-action="resume">Resume</button>';
            html += '<button class="task-btn task-btn-restart" data-action="restart">Restart</button>';
        } else if (status === 'editing') {
            // Buttons are in the editor area instead
        } else if (status === 'completed') {
            html += '<button class="task-btn task-btn-edit" data-action="edit">Edit &amp; Re-run</button>';
            html += '<button class="task-btn task-btn-restart" data-action="restart">Re-run</button>';
        } else if (status === 'failed') {
            html += '<button class="task-btn task-btn-edit" data-action="edit">Edit &amp; Retry</button>';
            html += '<button class="task-btn task-btn-restart" data-action="restart">Restart</button>';
        }

        return html;
    }

    function buildPipeline(t) {
        const stages = ['pending', 'running', 'completed'];
        const statusMap = {
            'pending': 0, 'queued': 0,
            'running': 1, 'paused': 1, 'editing': 1, 'timeout_warning': 1,
            'completed': 2, 'failed': 2,
        };
        const currentIdx = (t.status in statusMap) ? statusMap[t.status] : 0;

        let html = '<div class="task-pipeline">';
        stages.forEach((stage, i) => {
            let cls = '';
            if (i < currentIdx) cls = 'done';
            else if (i === currentIdx) {
                cls = 'current';
                if (t.status === 'failed') cls += ' failed';
                else if (t.status === 'paused') cls += ' paused';
                else if (t.status === 'editing') cls += ' editing';
                else if (t.status === 'timeout_warning') cls += ' timeout-warning';
            }
            html += `<span class="pipeline-stage ${cls}">${stage}</span>`;
            if (i < stages.length - 1) {
                html += '<span class="pipeline-arrow">&rarr;</span>';
            }
        });
        html += '</div>';
        return html;
    }

    function buildSessionPipeline() {
        // Shows the 5 orchestrator stages with current progress
        const stages = [
            { key: 'decompose', label: 'Decompose' },
            { key: 'build',     label: 'Build Graph' },
            { key: 'assign',    label: 'Assign Sessions' },
            { key: 'execute',   label: 'Execute' },
            { key: 'synthesize', label: 'Synthesize' },
        ];

        // Determine current stage index from orchestratorState and graph state
        let currentStageIdx = -1;
        if (orchestratorState === 'idle') {
            currentStageIdx = -1;
        } else if (orchestratorState === 'preview') {
            currentStageIdx = 1; // decompose done, graph built, waiting for user
        } else if (orchestratorState === 'completed' || orchestratorState === 'failed') {
            currentStageIdx = 5; // all done
        } else {
            // Check graph state to infer which stage we're in
            const taskIds = Object.keys(tasks);
            if (taskIds.length === 0) {
                currentStageIdx = 0; // still decomposing
            } else {
                const hasSession = taskIds.some(function(tid) { return tasks[tid].session_id; });
                const hasRunningOrDone = taskIds.some(function(tid) {
                    var s = tasks[tid].status;
                    return s === 'running' || s === 'completed' || s === 'failed' || s === 'paused' || s === 'editing' || s === 'timeout_warning';
                });
                if (hasRunningOrDone) {
                    currentStageIdx = 3; // executing
                } else if (hasSession) {
                    currentStageIdx = 3; // assigned, about to execute
                } else {
                    currentStageIdx = 1; // building graph
                }
            }
        }

        let html = '<div class="session-pipeline">';
        html += '<div class="session-pipeline-label">Orchestration Pipeline</div>';
        html += '<div class="session-pipeline-stages">';
        stages.forEach(function(stage, i) {
            let cls = 'session-pipeline-stage';
            if (i < currentStageIdx) cls += ' done';
            else if (i === currentStageIdx) {
                cls += ' current';
                if (orchestratorState === 'failed') cls += ' failed';
            }
            html += '<span class="' + cls + '">' + stage.label + '</span>';
            if (i < stages.length - 1) {
                html += '<span class="pipeline-arrow">&rarr;</span>';
            }
        });
        html += '</div></div>';
        return html;
    }

    function wireActionButtons(tid) {
        detailBody.querySelectorAll('.task-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const action = btn.dataset.action;
                btn.disabled = true;
                if (action === 'edit') {
                    const result = await apiTaskAction('edit', tid);
                    if (result.ok) {
                        editingTaskId = tid;
                        showTaskDetail(tid);
                    }
                } else if (action === 'pause') {
                    await apiTaskAction('pause', tid);
                } else if (action === 'resume') {
                    editingTaskId = null;
                    await apiTaskAction('resume', tid);
                } else if (action === 'restart') {
                    editingTaskId = null;
                    await apiTaskAction('restart', tid);
                } else if (action === 'postpone') {
                    await apiTaskAction('postpone', tid);
                }
                btn.disabled = false;
            });
        });

        // Save & Run button in editor
        const saveBtn = detailBody.querySelector('.btn-save-instructions');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                const editor = detailBody.querySelector('#taskInstructionsEditor');
                if (editor) {
                    saveBtn.disabled = true;
                    await apiTaskAction('update', tid, { description: editor.value });
                    editingTaskId = null;
                    await apiTaskAction('resume', tid);
                    saveBtn.disabled = false;
                }
            });
        }

        // Cancel button in editor
        const cancelBtn = detailBody.querySelector('.btn-cancel-edit');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', async () => {
                cancelBtn.disabled = true;
                editingTaskId = null;
                await apiTaskAction('resume', tid);
                cancelBtn.disabled = false;
            });
        }
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

    // ── Workflow save/load ──────────────────────────────────

    async function saveWorkflow() {
        // Use the editable input value as the name
        var saveName = (orchWorkflowNameInput && orchWorkflowNameInput.value.trim()) || workflowName;
        if (saveName) {
            workflowName = saveName;
            updateWorkflowNameDisplay();
        }
        try {
            // Send task overrides so per-task session/model config is persisted.
            var overrides = Object.keys(taskOverrides).length > 0 ? taskOverrides : null;
            const resp = await fetch('/api/orchestrator/workflows/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: saveName, task_overrides: overrides }),
            });
            const data = await resp.json();
            if (data.ok) {
                addLogEntry('completed', `Workflow saved: ${data.name || saveName}`);
                loadWorkflowList();
            } else {
                addLogEntry('task_failed', `Save failed: ${data.error || 'unknown'}`);
            }
        } catch (e) {
            addLogEntry('task_failed', `Save error: ${e.message}`);
        }
    }

    async function loadWorkflowList() {
        try {
            const resp = await fetch('/api/orchestrator/workflows');
            const data = await resp.json();
            const workflows = data.workflows || [];

            orchLoadWorkflowSelect.innerHTML = '<option value="">Load workflow...</option>';
            workflows.forEach(wf => {
                const opt = document.createElement('option');
                opt.value = wf.filename || wf.name;
                opt.textContent = `${wf.name} (${wf.task_count} tasks)`;
                orchLoadWorkflowSelect.appendChild(opt);
            });
        } catch (e) {
            // Silently fail
        }
    }

    async function loadWorkflow(name) {
        if (!name) return;
        try {
            const resp = await fetch('/api/orchestrator/workflows/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            });
            const data = await resp.json();
            if (data.ok) {
                addLogEntry('progress', `Loaded workflow: ${data.workflow_name || name}`);
                // The load endpoint broadcasts 'decomposed' via WebSocket
                // which populates tasks and enters preview state.

                // Show the rephrased prompt that was used to decompose.
                var userInput = data.user_input || '';
                if (userInput) {
                    orchRephrased.value = userInput;
                    syncPaneHeights();
                }

                // Apply workflow variables from response (also arrives via WS but
                // the HTTP response is more immediate).
                if (data.variables && data.variables.length > 0) {
                    applyVariables(data.variables);
                } else {
                    applyVariables([]);
                }
            } else {
                addLogEntry('task_failed', `Load failed: ${data.error || 'unknown'}`);
            }
        } catch (e) {
            addLogEntry('task_failed', `Load error: ${e.message}`);
        }
    }

    // ── Load available sessions and models for per-task config ──

    async function loadSessionsAndModels() {
        try {
            const [sessResp, modelResp] = await Promise.all([
                fetch('/api/orchestrator/sessions'),
                fetch('/api/orchestrator/models'),
            ]);
            if (sessResp.ok) {
                const sessData = await sessResp.json();
                availableSessions = sessData.sessions || [];
            }
            if (modelResp.ok) {
                const modelData = await modelResp.json();
                availableModels = modelData.models || [];
            }
        } catch (e) {
            // Silently fail
        }
    }

    // ── Init ─────────────────────────────────────────────────

    function init() {
        // Connect WebSocket
        connect();

        // Prepare button → rephrase
        orchPrepareBtn.addEventListener('click', prepareOrchestrator);

        // Ctrl+Enter in prompt textarea → prepare (rephrase)
        orchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                prepareOrchestrator();
            }
        });

        // Auto-sync pane heights when user types in prompt
        orchInput.addEventListener('input', syncPaneHeights);

        // Run button (now "Decompose") → decompose into tasks for preview
        orchRunBtn.addEventListener('click', decomposeOrchestrator);

        // Ctrl+Enter in rephrased textarea → decompose
        orchRephrased.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                decomposeOrchestrator();
            }
        });

        // Auto-resize rephrased textarea on input
        orchRephrased.addEventListener('input', autoResizeRephrased);

        // Discard button
        orchDiscardBtn.addEventListener('click', discardRephrase);

        // Execute button → run the prepared graph
        orchExecuteBtn.addEventListener('click', executeOrchestrator);

        // Reset button → clear everything
        if (orchResetBtn) orchResetBtn.addEventListener('click', resetOrchestrator);

        // Detail close
        detailClose.addEventListener('click', () => {
            selectedTaskId = null;
            hideDetailPanel();
            renderGraph();
        });

        // Clear log
        logClearBtn.addEventListener('click', clearLog);

        // Save workflow button
        orchSaveWorkflowBtn.addEventListener('click', saveWorkflow);

        // Sync workflow name from input to header on edit
        orchWorkflowNameInput.addEventListener('input', function () {
            workflowName = orchWorkflowNameInput.value.trim();
            if (orchWorkflowName) {
                orchWorkflowName.textContent = workflowName ? '\u2014 ' + workflowName : '';
            }
        });

        // Load workflow select
        orchLoadWorkflowSelect.addEventListener('change', (e) => {
            const name = e.target.value;
            if (name) {
                loadWorkflow(name);
                e.target.value = '';  // Reset select
            }
        });

        // Load skills, sessions, models, workflows, and current status on page load
        loadSkills();
        loadSessionsAndModels();
        loadWorkflowList();
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
                        model_id: '',
                        skills: [],
                        status: t.status || 'pending',
                        result: t.result_preview || null,
                        error: t.error || null,
                        retries: t.retries || 0,
                        startTime: null,
                        endTime: null,
                    };
                });
                graphSummary = data.status.summary || {};

                if (data.status.workflow_name) {
                    workflowName = data.status.workflow_name;
                    updateWorkflowNameDisplay();
                }

                // Restore the rephrased prompt text if available.
                if (data.status.user_input) {
                    orchRephrased.value = data.status.user_input;
                    syncPaneHeights();
                }

                updateSummaryBar();
                renderGraph();

                // Determine orchestrator state
                const hasRunning = data.status.tasks.some(t => t.status === 'running' || t.status === 'timeout_warning');
                const hasPaused = data.status.tasks.some(t => t.status === 'paused' || t.status === 'editing');
                const hasFailed = data.status.tasks.some(t => t.status === 'failed');
                const allTerminal = data.status.tasks.every(t => t.status === 'completed' || t.status === 'failed');
                const allPending = data.status.tasks.every(t => t.status === 'pending');

                if (allPending) {
                    setOrchestratorState('preview');
                } else if (hasRunning || hasPaused) {
                    setOrchestratorState('running');
                } else if (allTerminal && hasFailed) {
                    setOrchestratorState('failed');
                } else if (allTerminal) {
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
