(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    var ws = null;
    var isConnected = false;
    var workflowsData = [];          // [{name, filename, task_count, has_variables, variables}]
    var selectedWorkflow = null;      // current workflow object from workflowsData
    var iterations = [];              // [{id, variable_values: {name: val}, status, duration, result, error}]
    var loopId = null;                // active loop ID from server
    var loopState = 'idle';           // idle | running | completed | stopped | failed

    // ── DOM References ───────────────────────────────────────
    var loopWorkflowSelect = document.getElementById('loopWorkflowSelect');
    var loopIterationCount = document.getElementById('loopIterationCount');
    var loopApplyCount = document.getElementById('loopApplyCount');
    var loopVarGridWrapper = document.getElementById('loopVarGridWrapper');
    var loopVarThead = document.getElementById('loopVarThead');
    var loopVarTbody = document.getElementById('loopVarTbody');
    var loopNoVars = document.getElementById('loopNoVars');
    var loopAddRow = document.getElementById('loopAddRow');
    var loopCopyDown = document.getElementById('loopCopyDown');
    var loopStartBtn = document.getElementById('loopStartBtn');
    var loopStopBtn = document.getElementById('loopStopBtn');
    var loopProgress = document.getElementById('loopProgress');
    var loopSetup = document.getElementById('loopSetup');
    var loopProgressBar = document.getElementById('loopProgressBar');
    var loopProgressText = document.getElementById('loopProgressText');
    var loopIterationsEl = document.getElementById('loopIterations');
    var loopStatusDot = document.getElementById('loopStatusDot');
    var loopStatusText = document.getElementById('loopStatusText');
    var loopBadge = document.getElementById('loopBadge');
    var loopDoneActions = document.getElementById('loopDoneActions');
    var loopNewBtn = document.getElementById('loopNewBtn');
    var loopLogMessages = document.getElementById('loopLogMessages');
    var loopLogEmpty = document.getElementById('loopLogEmpty');
    var loopLogClear = document.getElementById('loopLogClear');

    // ── Init ─────────────────────────────────────────────────

    init();

    async function init() {
        loopWorkflowSelect.addEventListener('change', onWorkflowSelected);
        loopApplyCount.addEventListener('click', applyIterationCount);
        loopAddRow.addEventListener('click', addIteration);
        loopCopyDown.addEventListener('click', copyFirstRowDown);
        loopStartBtn.addEventListener('click', startLoop);
        loopStopBtn.addEventListener('click', stopLoop);
        loopNewBtn.addEventListener('click', newLoop);
        loopLogClear.addEventListener('click', clearLog);

        // Enter key on iteration count triggers Apply
        loopIterationCount.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') applyIterationCount();
        });

        await loadWorkflows();
        connectWebSocket();
        await checkStatus();
    }

    // ── WebSocket ────────────────────────────────────────────

    function connectWebSocket() {
        var protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        var url = protocol + '//' + location.host + '/ws';
        ws = new WebSocket(url);

        ws.onopen = function () { isConnected = true; };
        ws.onclose = function () {
            isConnected = false;
            setTimeout(connectWebSocket, 3000);
        };
        ws.onerror = function () { isConnected = false; };
        ws.onmessage = function (ev) {
            try {
                var msg = JSON.parse(ev.data);
                handleWsMessage(msg);
            } catch (e) { /* ignore */ }
        };
    }

    function handleWsMessage(msg) {
        if (msg.type === 'loop_orchestrator_event') {
            handleOrchestratorEvent(msg);
            return;
        }
        if (msg.type !== 'loop_event') return;

        switch (msg.event) {
            case 'iteration_started':
                onIterationStarted(msg);
                break;
            case 'iteration_completed':
                onIterationCompleted(msg);
                break;
            case 'iteration_failed':
                onIterationFailed(msg);
                break;
            case 'loop_completed':
                onLoopCompleted(msg);
                break;
            case 'loop_stopped':
                onLoopStopped(msg);
                break;
        }
    }

    // ── API Helpers ──────────────────────────────────────────

    async function apiFetch(url, options) {
        var res = await fetch(url, options || {});
        if (!res.ok) {
            var text = '';
            try { text = await res.text(); } catch (e) {}
            throw new Error('HTTP ' + res.status + (text ? ': ' + text : ''));
        }
        return await res.json();
    }

    async function apiPost(url, body) {
        return apiFetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    }

    // ── Load Workflows ───────────────────────────────────────

    async function loadWorkflows() {
        try {
            var data = await apiFetch('/api/orchestrator/workflows');
            workflowsData = (data && data.workflows) ? data.workflows : [];
        } catch (e) {
            workflowsData = [];
        }
        renderWorkflowDropdown();
    }

    function renderWorkflowDropdown() {
        loopWorkflowSelect.innerHTML = '<option value="">Select workflow...</option>';
        for (var i = 0; i < workflowsData.length; i++) {
            var wf = workflowsData[i];
            var opt = document.createElement('option');
            opt.value = wf.name;
            var label = wf.name + ' (' + wf.task_count + ' tasks';
            if (wf.has_variables) label += ', ' + wf.variables.length + ' vars';
            label += ')';
            opt.textContent = label;
            loopWorkflowSelect.appendChild(opt);
        }
    }

    // ── Workflow Selection ────────────────────────────────────

    function onWorkflowSelected() {
        var name = loopWorkflowSelect.value;
        selectedWorkflow = null;
        for (var i = 0; i < workflowsData.length; i++) {
            if (workflowsData[i].name === name) {
                selectedWorkflow = workflowsData[i];
                break;
            }
        }
        applyIterationCount();
    }

    // ── Iteration Management ─────────────────────────────────

    function applyIterationCount() {
        if (!selectedWorkflow) {
            loopVarGridWrapper.style.display = 'none';
            loopNoVars.style.display = 'none';
            loopStartBtn.disabled = true;
            return;
        }

        var count = Math.max(1, Math.min(200, parseInt(loopIterationCount.value, 10) || 1));
        loopIterationCount.value = String(count);
        var variables = selectedWorkflow.variables || [];

        // Preserve existing iteration values where possible
        var oldIterations = iterations;
        iterations = [];
        for (var i = 0; i < count; i++) {
            var varVals = {};
            for (var j = 0; j < variables.length; j++) {
                var vname = variables[j].name;
                // Carry over value from old row if it exists
                if (oldIterations[i] && oldIterations[i].variable_values &&
                    oldIterations[i].variable_values[vname] !== undefined) {
                    varVals[vname] = oldIterations[i].variable_values[vname];
                } else {
                    varVals[vname] = variables[j]['default'] || '';
                }
            }
            iterations.push({
                id: i + 1,
                variable_values: varVals,
                status: 'pending',
                duration: null,
                result: null,
                error: null,
            });
        }

        if (variables.length > 0) {
            loopVarGridWrapper.style.display = '';
            loopNoVars.style.display = 'none';
            renderVarTable();
        } else {
            loopVarGridWrapper.style.display = 'none';
            loopNoVars.style.display = '';
        }
        loopStartBtn.disabled = false;
    }

    function renderVarTable() {
        var variables = selectedWorkflow.variables || [];

        // Build header: # | var1 | var2 | ... | (remove)
        loopVarThead.innerHTML = '';
        var headerRow = document.createElement('tr');
        var th0 = document.createElement('th');
        th0.textContent = '#';
        th0.style.width = '40px';
        headerRow.appendChild(th0);
        for (var v = 0; v < variables.length; v++) {
            var th = document.createElement('th');
            th.textContent = variables[v].label || variables[v].name;
            headerRow.appendChild(th);
        }
        var thRemove = document.createElement('th');
        thRemove.style.width = '36px';
        headerRow.appendChild(thRemove);
        loopVarThead.appendChild(headerRow);

        // Build body rows
        loopVarTbody.innerHTML = '';
        for (var i = 0; i < iterations.length; i++) {
            loopVarTbody.appendChild(createIterationRow(i));
        }
    }

    function createIterationRow(index) {
        var variables = selectedWorkflow.variables || [];
        var iter = iterations[index];
        var tr = document.createElement('tr');

        // Row number
        var tdNum = document.createElement('td');
        tdNum.className = 'loop-row-num';
        tdNum.textContent = String(index + 1);
        tr.appendChild(tdNum);

        // Variable inputs
        for (var v = 0; v < variables.length; v++) {
            var td = document.createElement('td');
            var input = document.createElement('input');
            input.type = 'text';
            input.className = 'loop-var-input';
            input.value = iter.variable_values[variables[v].name] || '';
            input.dataset.iterIndex = String(index);
            input.dataset.varName = variables[v].name;
            input.addEventListener('input', onVarInputChange);
            td.appendChild(input);
            tr.appendChild(td);
        }

        // Remove button
        var tdRemove = document.createElement('td');
        var removeBtn = document.createElement('button');
        removeBtn.className = 'loop-remove-btn';
        removeBtn.textContent = '\u00d7';
        removeBtn.title = 'Remove iteration';
        removeBtn.dataset.iterIndex = String(index);
        removeBtn.addEventListener('click', function () {
            var idx = parseInt(this.dataset.iterIndex, 10);
            removeIteration(idx);
        });
        tdRemove.appendChild(removeBtn);
        tr.appendChild(tdRemove);

        return tr;
    }

    function onVarInputChange(e) {
        var idx = parseInt(e.target.dataset.iterIndex, 10);
        var varName = e.target.dataset.varName;
        if (iterations[idx]) {
            iterations[idx].variable_values[varName] = e.target.value;
        }
    }

    function addIteration() {
        if (!selectedWorkflow) return;
        var variables = selectedWorkflow.variables || [];
        var varVals = {};
        for (var j = 0; j < variables.length; j++) {
            varVals[variables[j].name] = variables[j]['default'] || '';
        }
        iterations.push({
            id: iterations.length + 1,
            variable_values: varVals,
            status: 'pending',
            duration: null,
            result: null,
            error: null,
        });
        loopIterationCount.value = String(iterations.length);
        if (variables.length > 0) {
            renderVarTable();
        }
    }

    function removeIteration(index) {
        if (iterations.length <= 1) return;
        iterations.splice(index, 1);
        for (var i = 0; i < iterations.length; i++) {
            iterations[i].id = i + 1;
        }
        loopIterationCount.value = String(iterations.length);
        renderVarTable();
    }

    function copyFirstRowDown() {
        if (iterations.length < 2) return;
        var first = iterations[0].variable_values;
        for (var i = 1; i < iterations.length; i++) {
            for (var key in first) {
                if (first.hasOwnProperty(key)) {
                    iterations[i].variable_values[key] = first[key];
                }
            }
        }
        renderVarTable();
    }

    // ── Start / Stop / New ───────────────────────────────────

    async function startLoop() {
        if (!selectedWorkflow || iterations.length === 0) return;
        loopStartBtn.disabled = true;

        // Collect iteration data from current state
        var payload = {
            workflow: selectedWorkflow.name,
            iterations: [],
        };
        for (var i = 0; i < iterations.length; i++) {
            payload.iterations.push({
                variable_values: iterations[i].variable_values,
            });
        }

        try {
            var data = await apiPost('/api/loops/start', payload);
            loopId = data.loop_id;
            loopState = 'running';
            // Reset status for display
            for (var j = 0; j < iterations.length; j++) {
                iterations[j].status = 'pending';
                iterations[j].duration = null;
                iterations[j].result = null;
                iterations[j].error = null;
            }
            showProgress();
        } catch (e) {
            loopStartBtn.disabled = false;
            console.error('[loop-runner] start failed', e);
            alert('Failed to start loop: ' + e.message);
        }
    }

    async function stopLoop() {
        if (!loopId) return;
        loopStopBtn.disabled = true;
        try {
            await apiPost('/api/loops/stop', {});
        } catch (e) {
            console.error('[loop-runner] stop failed', e);
        }
    }

    function newLoop() {
        loopState = 'idle';
        loopId = null;
        loopProgress.style.display = 'none';
        loopDoneActions.style.display = 'none';
        loopSetup.style.display = '';
        loopStartBtn.disabled = false;
        loopStopBtn.disabled = false;
        updateStatusIndicator('idle');
        loopBadge.textContent = '';
    }

    async function checkStatus() {
        try {
            var data = await apiFetch('/api/loops/status');
            if (data && data.loop_id && data.state === 'running') {
                loopId = data.loop_id;
                loopState = 'running';
                iterations = data.iterations || [];
                // Try to restore workflow name in dropdown
                if (data.workflow) {
                    loopWorkflowSelect.value = data.workflow;
                }
                showProgress();
            }
        } catch (e) {
            // No active loop — stay in setup mode
        }
    }

    // ── Progress UI ──────────────────────────────────────────

    function showProgress() {
        loopSetup.style.display = 'none';
        loopProgress.style.display = '';
        loopDoneActions.style.display = 'none';
        loopStopBtn.disabled = false;
        renderIterationCards();
        updateProgressUI();
        updateStatusIndicator('running');
    }

    function renderIterationCards() {
        loopIterationsEl.innerHTML = '';
        for (var i = 0; i < iterations.length; i++) {
            loopIterationsEl.appendChild(createIterCard(iterations[i], i));
        }
    }

    function createIterCard(iter, index) {
        var card = document.createElement('div');
        var status = iter.status || 'pending';
        card.className = 'loop-iter-card loop-iter-' + status;
        card.id = 'loopIter' + index;

        // Header row: iteration # | badge | duration
        var header = document.createElement('div');
        header.className = 'loop-iter-header';

        var num = document.createElement('span');
        num.className = 'loop-iter-num';
        num.textContent = 'Iteration ' + (index + 1);

        var badge = document.createElement('span');
        badge.className = 'loop-iter-badge loop-iter-badge-' + status;
        badge.textContent = status;

        var dur = document.createElement('span');
        dur.className = 'loop-iter-duration';
        if (iter.duration != null) {
            dur.textContent = formatDuration(iter.duration);
        }

        header.appendChild(num);
        header.appendChild(badge);
        header.appendChild(dur);
        card.appendChild(header);

        // Variable values summary
        var varVals = iter.variable_values;
        if (varVals && Object.keys(varVals).length > 0) {
            var vars = document.createElement('div');
            vars.className = 'loop-iter-vars';
            var parts = [];
            for (var k in varVals) {
                if (varVals.hasOwnProperty(k)) {
                    var val = varVals[k];
                    if (val.length > 60) val = val.substring(0, 60) + '...';
                    parts.push(k + '=' + val);
                }
            }
            vars.textContent = parts.join('  |  ');
            card.appendChild(vars);
        }

        // Result or error
        if (status === 'failed' && iter.error) {
            var errorEl = document.createElement('div');
            errorEl.className = 'loop-iter-error';
            errorEl.textContent = iter.error;
            card.appendChild(errorEl);
        } else if (iter.result_preview || iter.result) {
            var resultText = iter.result_preview || iter.result || '';
            if (resultText) {
                var result = document.createElement('div');
                result.className = 'loop-iter-result';
                result.textContent = resultText.length > 500
                    ? resultText.substring(0, 500) + '...'
                    : resultText;
                card.appendChild(result);
            }
        }

        return card;
    }

    function updateProgressUI() {
        var total = iterations.length;
        var completed = 0;
        var failed = 0;
        for (var i = 0; i < iterations.length; i++) {
            if (iterations[i].status === 'completed') completed++;
            if (iterations[i].status === 'failed') failed++;
        }
        var done = completed + failed;
        var pct = total > 0 ? Math.round((done / total) * 100) : 0;
        loopProgressBar.style.width = pct + '%';

        // Color the bar green when all done
        if (loopState === 'completed') {
            loopProgressBar.style.background = failed > 0 ? 'var(--yellow)' : 'var(--green)';
        } else if (loopState === 'stopped') {
            loopProgressBar.style.background = 'var(--yellow)';
        } else {
            loopProgressBar.style.background = '';
        }

        loopProgressText.textContent = done + ' / ' + total;
        loopBadge.textContent = done + '/' + total;
    }

    function updateIterCard(index) {
        var existing = document.getElementById('loopIter' + index);
        if (!existing) return;
        var newCard = createIterCard(iterations[index], index);
        existing.replaceWith(newCard);
    }

    function updateStatusIndicator(state) {
        loopStatusDot.className = 'loop-status-dot ' + state;
        loopStatusText.textContent = state;
    }

    function formatDuration(seconds) {
        if (seconds < 60) return seconds.toFixed(1) + 's';
        var m = Math.floor(seconds / 60);
        var s = Math.round(seconds % 60);
        return m + 'm ' + s + 's';
    }

    // ── WebSocket Event Handlers ─────────────────────────────

    function onIterationStarted(msg) {
        var idx = msg.iteration_index;
        if (idx == null || !iterations[idx]) return;
        iterations[idx].status = 'running';
        updateIterCard(idx);
        updateProgressUI();
        addLogEntry('iteration', 'Iteration ' + (idx + 1) + ' started');
    }

    function onIterationCompleted(msg) {
        var idx = msg.iteration_index;
        if (idx == null || !iterations[idx]) return;
        iterations[idx].status = 'completed';
        iterations[idx].duration = msg.duration || null;
        iterations[idx].result_preview = msg.result_preview || '';
        updateIterCard(idx);
        updateProgressUI();
        addLogEntry('completed', 'Iteration ' + (idx + 1) + ' completed' + (msg.duration ? ' (' + formatDuration(msg.duration) + ')' : ''));
    }

    function onIterationFailed(msg) {
        var idx = msg.iteration_index;
        if (idx == null || !iterations[idx]) return;
        iterations[idx].status = 'failed';
        iterations[idx].duration = msg.duration || null;
        iterations[idx].error = msg.error || 'Unknown error';
        updateIterCard(idx);
        updateProgressUI();
        addLogEntry('error', 'Iteration ' + (idx + 1) + ' failed: ' + (msg.error || 'unknown'));
    }

    function onLoopCompleted(msg) {
        loopState = 'completed';
        loopStopBtn.disabled = true;
        loopDoneActions.style.display = '';
        updateStatusIndicator('completed');
        addLogEntry('completed', 'Loop completed');
        updateProgressUI();
    }

    function onLoopStopped(msg) {
        loopState = 'stopped';
        loopStopBtn.disabled = true;
        loopDoneActions.style.display = '';
        // Mark remaining pending as cancelled
        for (var i = 0; i < iterations.length; i++) {
            if (iterations[i].status === 'pending') {
                iterations[i].status = 'cancelled';
                updateIterCard(i);
            }
        }
        updateStatusIndicator('stopped');
        updateProgressUI();
        addLogEntry('error', 'Loop stopped');
    }

    // ── Orchestrator Event → Log ─────────────────────────────

    function handleOrchestratorEvent(msg) {
        // msg.type === 'loop_orchestrator_event'
        // msg.event === orchestrator event name (decomposed, task_started, etc.)
        // msg.iteration_index === which iteration this belongs to
        var iter = (msg.iteration_index != null) ? (msg.iteration_index + 1) : '?';
        var prefix = '[' + iter + '] ';
        var ev = msg.event || '';

        switch (ev) {
            case 'decomposing':
                addLogEntry('progress', prefix + 'Decomposing...');
                break;
            case 'decomposed':
                addLogEntry('progress', prefix + 'Decomposed into ' + (msg.task_count || (msg.tasks || []).length || '?') + ' tasks');
                break;
            case 'building_graph':
                addLogEntry('progress', prefix + 'Building graph (' + (msg.task_count || '?') + ' tasks)');
                break;
            case 'assigning_sessions':
                addLogEntry('progress', prefix + 'Assigning sessions');
                break;
            case 'executing':
                addLogEntry('progress', prefix + 'Executing ' + (msg.task_count || '?') + ' tasks');
                break;
            case 'task_started':
                addLogEntry('task_started', prefix + 'Task started: ' + (msg.title || msg.task_id || ''));
                break;
            case 'task_completed':
                addLogEntry('task_completed', prefix + 'Task done: ' + (msg.title || msg.task_id || ''));
                break;
            case 'task_failed':
                addLogEntry('task_failed', prefix + 'Task failed: ' + (msg.title || msg.task_id || '') + (msg.error ? ' — ' + msg.error : ''));
                break;
            case 'synthesizing':
                addLogEntry('synthesizing', prefix + 'Synthesizing results...');
                break;
            case 'completed':
                addLogEntry('completed', prefix + 'Orchestration complete');
                break;
            case 'error':
                addLogEntry('error', prefix + (msg.message || 'Error'));
                break;
            default:
                addLogEntry('progress', prefix + ev);
                break;
        }
    }

    // ── Event Log ────────────────────────────────────────────

    function addLogEntry(tag, text) {
        if (loopLogEmpty) loopLogEmpty.style.display = 'none';

        var now = new Date();
        var ts = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

        var entry = document.createElement('div');
        entry.className = 'loop-log-entry';
        entry.innerHTML =
            '<span class="loop-log-time">' + ts + '</span>' +
            '<span class="loop-log-tag ' + esc(tag) + '">' + esc(tag) + '</span>' +
            '<span class="loop-log-text">' + esc(text) + '</span>';
        loopLogMessages.appendChild(entry);
        loopLogMessages.scrollTop = loopLogMessages.scrollHeight;
    }

    function clearLog() {
        loopLogMessages.innerHTML = '';
        if (loopLogEmpty) {
            loopLogMessages.appendChild(loopLogEmpty);
            loopLogEmpty.style.display = '';
        }
    }

    function esc(s) {
        var d = document.createElement('div');
        d.textContent = s || '';
        return d.innerHTML;
    }

})();
