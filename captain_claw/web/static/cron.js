/* Captain Claw Cron Management - Client-Side Logic */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    var refreshTimer = null;
    var currentJobs = [];
    var selectedJobId = null;
    var activeTab = 'chat';

    // ── DOM References ───────────────────────────────────────
    var cronBadge = document.getElementById('cronBadge');
    var cronToggleAdd = document.getElementById('cronToggleAdd');
    var cronAddPanel = document.getElementById('cronAddPanel');
    var cronSchedType = document.getElementById('cronSchedType');
    var cronInterval = document.getElementById('cronInterval');
    var cronIntervalUnit = document.getElementById('cronIntervalUnit');
    var cronDailyTime = document.getElementById('cronDailyTime');
    var cronWeeklyDay = document.getElementById('cronWeeklyDay');
    var cronWeeklyTime = document.getElementById('cronWeeklyTime');
    var cronKind = document.getElementById('cronKind');
    var cronSession = document.getElementById('cronSession');
    var cronPayloadPrompt = document.getElementById('cronPayloadPrompt');
    var cronPayloadPath = document.getElementById('cronPayloadPath');
    var cronPayloadWorkflow = document.getElementById('cronPayloadWorkflow');
    var cronAddBtn = document.getElementById('cronAddBtn');
    var cronTableBody = document.getElementById('cronTableBody');
    var cronTable = document.getElementById('cronTable');
    var cronEmpty = document.getElementById('cronEmpty');
    var cronShowDisabled = document.getElementById('cronShowDisabled');
    var cronDetail = document.getElementById('cronDetail');
    var cronDetailTitle = document.getElementById('cronDetailTitle');
    var cronDetailInfo = document.getElementById('cronDetailInfo');
    var cronDetailHistory = document.getElementById('cronDetailHistory');
    var cronDetailClose = document.getElementById('cronDetailClose');

    // Field groups
    var cronIntervalFields = document.getElementById('cronIntervalFields');
    var cronDailyFields = document.getElementById('cronDailyFields');
    var cronWeeklyFields = document.getElementById('cronWeeklyFields');

    // Tab buttons
    var tabButtons = document.querySelectorAll('.cron-detail-tab');

    // ── Initialization ───────────────────────────────────────

    init();

    async function init() {
        // Setup event listeners
        if (cronToggleAdd) {
            cronToggleAdd.addEventListener('click', toggleAddPanel);
        }
        if (cronSchedType) {
            cronSchedType.addEventListener('change', updateScheduleFields);
        }
        if (cronKind) {
            cronKind.addEventListener('change', updatePayloadFields);
        }
        if (cronAddBtn) {
            cronAddBtn.addEventListener('click', addJob);
        }
        if (cronShowDisabled) {
            cronShowDisabled.addEventListener('change', renderTable);
        }
        if (cronDetailClose) {
            cronDetailClose.addEventListener('click', closeDetail);
        }
        tabButtons.forEach(function (btn) {
            btn.addEventListener('click', function () {
                switchTab(btn.getAttribute('data-tab'));
            });
        });

        // Load initial data
        await Promise.all([loadJobs(), loadSessions(), loadWorkflows()]);

        // Start auto-refresh
        refreshTimer = setInterval(loadJobs, 30000);
    }

    // ── API Helpers ──────────────────────────────────────────

    async function apiFetch(url, options) {
        try {
            var res = await fetch(url, options || {});
            if (!res.ok) {
                var text = '';
                try { text = await res.text(); } catch (e) { /* ignore */ }
                throw new Error('HTTP ' + res.status + (text ? ': ' + text : ''));
            }
            return await res.json();
        } catch (e) {
            console.error('API error:', url, e);
            throw e;
        }
    }

    // ── Load Jobs ────────────────────────────────────────────

    async function loadJobs() {
        try {
            var data = await apiFetch('/api/cron/jobs');
            currentJobs = Array.isArray(data) ? data : [];
            renderTable();
            updateBadge();
        } catch (e) {
            // Silently fail on auto-refresh; jobs array stays as-is
        }
    }

    function updateBadge() {
        if (!cronBadge) return;
        var activeCount = 0;
        for (var i = 0; i < currentJobs.length; i++) {
            if (currentJobs[i].enabled) activeCount++;
        }
        cronBadge.textContent = activeCount + ' active';
    }

    // ── Load Sessions ────────────────────────────────────────

    async function loadSessions() {
        if (!cronSession) return;
        try {
            var data = await apiFetch('/api/sessions');
            var sessions = Array.isArray(data) ? data : (data && data.sessions ? data.sessions : []);
            cronSession.innerHTML = '';
            for (var i = 0; i < sessions.length; i++) {
                var s = sessions[i];
                var opt = document.createElement('option');
                opt.value = s.id;
                opt.textContent = s.name || s.id;
                cronSession.appendChild(opt);
            }
        } catch (e) {
            // Leave dropdown empty on error
        }
    }

    // ── Load Workflows ───────────────────────────────────────

    async function loadWorkflows() {
        if (!cronPayloadWorkflow) return;
        try {
            var data = await apiFetch('/api/orchestrator/workflows');
            var workflows = (data && data.workflows) ? data.workflows : [];
            cronPayloadWorkflow.innerHTML = '';
            if (workflows.length === 0) {
                var emptyOpt = document.createElement('option');
                emptyOpt.value = '';
                emptyOpt.textContent = '(no workflows)';
                cronPayloadWorkflow.appendChild(emptyOpt);
            }
            for (var i = 0; i < workflows.length; i++) {
                var w = workflows[i];
                var opt = document.createElement('option');
                var name = typeof w === 'string' ? w : (w.name || '');
                opt.value = name;
                opt.textContent = name;
                cronPayloadWorkflow.appendChild(opt);
            }
        } catch (e) {
            // Leave dropdown empty on error
        }
    }

    // ── Render Table ─────────────────────────────────────────

    function renderTable() {
        if (!cronTableBody) return;
        cronTableBody.innerHTML = '';

        var showDisabled = cronShowDisabled ? cronShowDisabled.checked : true;
        var visibleJobs = [];
        for (var i = 0; i < currentJobs.length; i++) {
            var job = currentJobs[i];
            if (!showDisabled && !job.enabled) continue;
            visibleJobs.push(job);
        }

        if (visibleJobs.length === 0) {
            if (cronTable) cronTable.style.display = 'none';
            if (cronEmpty) cronEmpty.style.display = '';
            return;
        }

        if (cronTable) cronTable.style.display = '';
        if (cronEmpty) cronEmpty.style.display = 'none';

        for (var j = 0; j < visibleJobs.length; j++) {
            var row = createJobRow(visibleJobs[j]);
            cronTableBody.appendChild(row);
        }
    }

    function createJobRow(job) {
        var tr = document.createElement('tr');
        if (!job.enabled) tr.classList.add('disabled');
        if (selectedJobId && job.id === selectedJobId) tr.classList.add('selected');

        // Row click -> show detail
        tr.addEventListener('click', function () {
            showDetail(job);
        });

        // Status dot
        var tdStatus = document.createElement('td');
        tdStatus.className = 'col-status';
        var dot = document.createElement('span');
        dot.className = 'cron-status-dot ' + getStatusClass(job);
        tdStatus.appendChild(dot);
        tr.appendChild(tdStatus);

        // Kind
        var tdKind = document.createElement('td');
        var kindBadge = document.createElement('span');
        kindBadge.className = 'cron-kind-badge ' + escapeAttr(job.kind || 'prompt');
        kindBadge.textContent = job.kind || 'prompt';
        tdKind.appendChild(kindBadge);
        tr.appendChild(tdKind);

        // Description
        var tdDesc = document.createElement('td');
        tdDesc.textContent = getJobDescription(job);
        tr.appendChild(tdDesc);

        // Schedule
        var tdSchedule = document.createElement('td');
        tdSchedule.className = 'job-schedule';
        tdSchedule.textContent = getScheduleText(job.schedule);
        tr.appendChild(tdSchedule);

        // Last Run
        var tdLastRun = document.createElement('td');
        tdLastRun.className = 'job-last-run';
        tdLastRun.textContent = relativeTime(job.last_run_at);
        tr.appendChild(tdLastRun);

        // Next Run
        var tdNextRun = document.createElement('td');
        tdNextRun.className = 'job-next-run';
        tdNextRun.textContent = relativeTime(job.next_run_at);
        tr.appendChild(tdNextRun);

        // Status text
        var tdStatusText = document.createElement('td');
        var statusLabel = document.createElement('span');
        statusLabel.className = 'cron-status-label ' + getStatusClass(job);
        statusLabel.textContent = job.last_status || (job.enabled ? 'idle' : 'disabled');
        tdStatusText.appendChild(statusLabel);
        tr.appendChild(tdStatusText);

        // Actions
        var tdActions = document.createElement('td');
        tdActions.className = 'col-actions';
        var actionsDiv = document.createElement('div');
        actionsDiv.className = 'cron-actions';

        // Run Now button
        var runBtn = document.createElement('button');
        runBtn.className = 'cron-action-btn run';
        runBtn.title = 'Run Now';
        runBtn.innerHTML = '&#9654;';
        runBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            runJob(job.id);
        });
        actionsDiv.appendChild(runBtn);

        // Pause/Resume button
        if (job.enabled) {
            var pauseBtn = document.createElement('button');
            pauseBtn.className = 'cron-action-btn pause';
            pauseBtn.title = 'Pause';
            pauseBtn.innerHTML = '&#9208;';
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                pauseJob(job.id);
            });
            actionsDiv.appendChild(pauseBtn);
        } else {
            var resumeBtn = document.createElement('button');
            resumeBtn.className = 'cron-action-btn resume';
            resumeBtn.title = 'Resume';
            resumeBtn.innerHTML = '&#9654;';
            resumeBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                resumeJob(job.id);
            });
            actionsDiv.appendChild(resumeBtn);
        }

        // Delete button
        var deleteBtn = document.createElement('button');
        deleteBtn.className = 'cron-action-btn delete';
        deleteBtn.title = 'Delete';
        deleteBtn.innerHTML = '&#10005;';
        deleteBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            deleteJob(job.id);
        });
        actionsDiv.appendChild(deleteBtn);

        tdActions.appendChild(actionsDiv);
        tr.appendChild(tdActions);

        return tr;
    }

    function getStatusClass(job) {
        if (!job.enabled) return 'disabled';
        if (job.last_status === 'running') return 'running';
        if (job.last_status === 'failed') return 'failed';
        if (job.enabled) return 'active';
        return 'paused';
    }

    function getJobDescription(job) {
        var payload = job.payload || {};
        var kind = job.kind || 'prompt';
        if (kind === 'prompt') {
            var text = payload.text || '';
            return text.length > 60 ? text.substring(0, 60) + '...' : text;
        }
        if (kind === 'script' || kind === 'tool') {
            return payload.path || '';
        }
        if (kind === 'orchestrate') {
            return 'wf: ' + (payload.workflow || '');
        }
        return '';
    }

    // ── Schedule Helpers ─────────────────────────────────────

    function getScheduleText(schedule) {
        if (!schedule) return '';
        if (schedule._text) return schedule._text;
        return scheduleToText(schedule);
    }

    function scheduleToText(schedule) {
        if (!schedule) return '';
        var type = schedule.type;
        if (type === 'every' || type === 'interval') {
            var interval = schedule.interval || '';
            var unit = schedule.unit || 'minutes';
            var suffix = (unit === 'minutes' || unit === 'minute' || unit.charAt(0) === 'm') ? 'm' : 'h';
            return 'every ' + interval + suffix;
        }
        if (type === 'daily') {
            var h = String(schedule.hour || 0).padStart(2, '0');
            var m = String(schedule.minute || 0).padStart(2, '0');
            return 'daily ' + h + ':' + m;
        }
        if (type === 'weekly') {
            var day = schedule.day || 'mon';
            var wh = String(schedule.hour || 0).padStart(2, '0');
            var wm = String(schedule.minute || 0).padStart(2, '0');
            return 'weekly ' + day + ' ' + wh + ':' + wm;
        }
        return JSON.stringify(schedule);
    }

    // ── Relative Time ────────────────────────────────────────

    function relativeTime(isoString) {
        if (!isoString) return '\u2014';
        var date;
        try {
            date = new Date(isoString);
            if (isNaN(date.getTime())) return '\u2014';
        } catch (e) {
            return '\u2014';
        }

        var now = new Date();
        var diffMs = date.getTime() - now.getTime();
        var absDiffMs = Math.abs(diffMs);
        var isPast = diffMs < 0;

        var seconds = Math.floor(absDiffMs / 1000);
        var minutes = Math.floor(seconds / 60);
        var hours = Math.floor(minutes / 60);
        var days = Math.floor(hours / 24);

        if (seconds < 60) {
            return isPast ? 'just now' : 'in <1m';
        }
        if (minutes < 60) {
            return isPast ? minutes + 'm ago' : 'in ' + minutes + 'm';
        }
        if (hours < 24) {
            return isPast ? hours + 'h ago' : 'in ' + hours + 'h';
        }
        if (days === 1) {
            return isPast ? 'yesterday' : 'tomorrow';
        }
        return isPast ? days + 'd ago' : 'in ' + days + 'd';
    }

    // ── Toggle Add Panel ─────────────────────────────────────

    function toggleAddPanel() {
        if (!cronAddPanel || !cronToggleAdd) return;
        var isHidden = cronAddPanel.style.display === 'none';
        cronAddPanel.style.display = isHidden ? '' : 'none';
        if (isHidden) {
            cronToggleAdd.classList.add('open');
        } else {
            cronToggleAdd.classList.remove('open');
        }
    }

    // ── Update Schedule Fields ───────────────────────────────

    function updateScheduleFields() {
        var type = cronSchedType ? cronSchedType.value : 'every';
        if (cronIntervalFields) {
            cronIntervalFields.style.display = (type === 'every') ? '' : 'none';
        }
        if (cronDailyFields) {
            cronDailyFields.style.display = (type === 'daily') ? '' : 'none';
        }
        if (cronWeeklyFields) {
            cronWeeklyFields.style.display = (type === 'weekly') ? '' : 'none';
        }
    }

    // ── Update Payload Fields ────────────────────────────────

    function updatePayloadFields() {
        var kind = cronKind ? cronKind.value : 'prompt';
        if (cronPayloadPrompt) {
            cronPayloadPrompt.style.display = (kind === 'prompt') ? '' : 'none';
        }
        if (cronPayloadPath) {
            cronPayloadPath.style.display = (kind === 'script' || kind === 'tool') ? '' : 'none';
        }
        if (cronPayloadWorkflow) {
            cronPayloadWorkflow.style.display = (kind === 'orchestrate') ? '' : 'none';
        }
    }

    // ── Add Job ──────────────────────────────────────────────

    async function addJob() {
        var kind = cronKind ? cronKind.value : 'prompt';
        var sessionId = cronSession ? cronSession.value : '';

        // Build schedule
        var schedule = {};
        var schedType = cronSchedType ? cronSchedType.value : 'every';
        schedule.type = schedType;

        if (schedType === 'every') {
            schedule.type = 'interval';
            schedule.interval = cronInterval ? parseInt(cronInterval.value, 10) || 15 : 15;
            schedule.unit = cronIntervalUnit ? cronIntervalUnit.value : 'minutes';
        } else if (schedType === 'daily') {
            var dailyVal = cronDailyTime ? cronDailyTime.value : '09:00';
            var dailyParts = dailyVal.split(':');
            schedule.hour = parseInt(dailyParts[0], 10) || 0;
            schedule.minute = parseInt(dailyParts[1], 10) || 0;
        } else if (schedType === 'weekly') {
            var weeklyDay = cronWeeklyDay ? cronWeeklyDay.value : 'mon';
            var weeklyVal = cronWeeklyTime ? cronWeeklyTime.value : '09:00';
            var weeklyParts = weeklyVal.split(':');
            schedule.day = weeklyDay;
            // Map day name to weekday number for compute_next_run
            var dayMap = {mon:0,tue:1,wed:2,thu:3,fri:4,sat:5,sun:6};
            schedule.weekday = dayMap[weeklyDay] !== undefined ? dayMap[weeklyDay] : 0;
            schedule.hour = parseInt(weeklyParts[0], 10) || 0;
            schedule.minute = parseInt(weeklyParts[1], 10) || 0;
        }

        // Build payload
        var payload = {};
        if (kind === 'prompt') {
            var promptText = cronPayloadPrompt ? cronPayloadPrompt.value.trim() : '';
            if (!promptText) {
                alert('Please enter a prompt.');
                return;
            }
            payload.text = promptText;
        } else if (kind === 'script' || kind === 'tool') {
            var pathText = cronPayloadPath ? cronPayloadPath.value.trim() : '';
            if (!pathText) {
                alert('Please enter a file path.');
                return;
            }
            payload.path = pathText;
        } else if (kind === 'orchestrate') {
            var workflowName = cronPayloadWorkflow ? cronPayloadWorkflow.value : '';
            if (!workflowName) {
                alert('Please select a workflow.');
                return;
            }
            payload.workflow = workflowName;
        }

        if (!sessionId) {
            alert('Please select a session.');
            return;
        }

        try {
            cronAddBtn.disabled = true;
            cronAddBtn.textContent = 'Adding...';
            await apiFetch('/api/cron/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    kind: kind,
                    schedule: schedule,
                    payload: payload,
                    session_id: sessionId,
                }),
            });
            // Reset form fields
            if (cronPayloadPrompt) cronPayloadPrompt.value = '';
            if (cronPayloadPath) cronPayloadPath.value = '';
            // Collapse the add panel
            if (cronAddPanel) cronAddPanel.style.display = 'none';
            if (cronToggleAdd) cronToggleAdd.classList.remove('open');
            // Reload jobs
            await loadJobs();
        } catch (e) {
            alert('Failed to add job: ' + e.message);
        } finally {
            if (cronAddBtn) {
                cronAddBtn.disabled = false;
                cronAddBtn.textContent = 'Add Job';
            }
        }
    }

    // ── Job Actions ──────────────────────────────────────────

    async function runJob(id) {
        try {
            await apiFetch('/api/cron/jobs/' + encodeURIComponent(id) + '/run', {
                method: 'POST',
            });
            await loadJobs();
        } catch (e) {
            alert('Failed to run job: ' + e.message);
        }
    }

    async function pauseJob(id) {
        try {
            await apiFetch('/api/cron/jobs/' + encodeURIComponent(id) + '/pause', {
                method: 'POST',
            });
            await loadJobs();
        } catch (e) {
            alert('Failed to pause job: ' + e.message);
        }
    }

    async function resumeJob(id) {
        try {
            await apiFetch('/api/cron/jobs/' + encodeURIComponent(id) + '/resume', {
                method: 'POST',
            });
            await loadJobs();
        } catch (e) {
            alert('Failed to resume job: ' + e.message);
        }
    }

    async function deleteJob(id) {
        if (!confirm('Delete this cron job?')) return;
        try {
            await apiFetch('/api/cron/jobs/' + encodeURIComponent(id), {
                method: 'DELETE',
            });
            // If detail panel shows this job, close it
            if (selectedJobId === id) {
                closeDetail();
            }
            await loadJobs();
        } catch (e) {
            alert('Failed to delete job: ' + e.message);
        }
    }

    // ── Detail Panel ─────────────────────────────────────────

    async function showDetail(job) {
        selectedJobId = job.id;
        if (!cronDetail) return;

        cronDetail.style.display = 'block';

        // Highlight selected row
        renderTable();

        // Fill title
        if (cronDetailTitle) {
            cronDetailTitle.textContent = (job.kind || 'prompt') + ' job: ' + getJobDescription(job);
        }

        // Fill info
        if (cronDetailInfo) {
            var infoHtml = '';
            infoHtml += '<dt>ID</dt><dd class="mono">' + escapeHtml(job.id || '') + '</dd>';
            infoHtml += '<dt>Kind</dt><dd>' + escapeHtml(job.kind || '') + '</dd>';
            infoHtml += '<dt>Schedule</dt><dd class="mono">' + escapeHtml(getScheduleText(job.schedule)) + '</dd>';
            infoHtml += '<dt>Session</dt><dd class="mono">' + escapeHtml(job.session_id || '') + '</dd>';
            infoHtml += '<dt>Enabled</dt><dd>' + (job.enabled ? 'Yes' : 'No') + '</dd>';
            infoHtml += '<dt>Created</dt><dd class="timestamp">' + escapeHtml(job.created_at || '') + '</dd>';
            infoHtml += '<dt>Updated</dt><dd class="timestamp">' + escapeHtml(job.updated_at || '') + '</dd>';
            infoHtml += '<dt>Last Run</dt><dd class="timestamp">' + escapeHtml(job.last_run_at || '\u2014') + '</dd>';
            infoHtml += '<dt>Next Run</dt><dd class="timestamp">' + escapeHtml(job.next_run_at || '\u2014') + '</dd>';
            infoHtml += '<dt>Last Status</dt><dd>' + escapeHtml(job.last_status || '\u2014') + '</dd>';
            if (job.last_error) {
                infoHtml += '<dt>Last Error</dt><dd style="color:var(--red)">' + escapeHtml(job.last_error) + '</dd>';
            }

            // Payload detail
            var payload = job.payload || {};
            if (job.kind === 'prompt' && payload.text) {
                infoHtml += '<dt>Payload</dt><dd>' + escapeHtml(payload.text) + '</dd>';
            } else if ((job.kind === 'script' || job.kind === 'tool') && payload.path) {
                infoHtml += '<dt>Payload</dt><dd class="mono">' + escapeHtml(payload.path) + '</dd>';
            } else if (job.kind === 'orchestrate' && payload.workflow) {
                infoHtml += '<dt>Workflow</dt><dd>' + escapeHtml(payload.workflow) + '</dd>';
            }

            cronDetailInfo.innerHTML = infoHtml;
        }

        // Default to chat tab
        activeTab = 'chat';
        updateTabUI();

        // Load history
        await loadHistory(job.id);
    }

    function closeDetail() {
        selectedJobId = null;
        if (cronDetail) cronDetail.style.display = 'none';
        renderTable();
    }

    // ── History ──────────────────────────────────────────────

    var cachedHistory = { chat: [], monitor: [] };

    async function loadHistory(jobId) {
        cachedHistory = { chat: [], monitor: [] };
        if (cronDetailHistory) {
            cronDetailHistory.innerHTML = '<div class="cron-history-empty">Loading...</div>';
        }
        try {
            var data = await apiFetch('/api/cron/jobs/' + encodeURIComponent(jobId) + '/history');
            cachedHistory.chat = data.chat_history || [];
            cachedHistory.monitor = data.monitor_history || [];
            renderCurrentHistory();
        } catch (e) {
            if (cronDetailHistory) {
                cronDetailHistory.innerHTML = '<div class="cron-history-empty">Failed to load history.</div>';
            }
        }
    }

    function renderCurrentHistory() {
        var historyArray = activeTab === 'chat' ? cachedHistory.chat : cachedHistory.monitor;
        renderHistory(historyArray);
    }

    function renderHistory(historyArray) {
        if (!cronDetailHistory) return;

        if (!historyArray || historyArray.length === 0) {
            cronDetailHistory.innerHTML = '<div class="cron-history-empty">No history yet.</div>';
            return;
        }

        var html = '';
        for (var i = 0; i < historyArray.length; i++) {
            var entry = historyArray[i];
            html += '<div class="cron-history-entry">';

            // Timestamp
            var ts = entry.timestamp || '';
            if (ts) {
                var shortTs = formatTimestamp(ts);
                html += '<span class="cron-history-time">' + escapeHtml(shortTs) + '</span>';
            }

            if (activeTab === 'chat') {
                // Chat: role + content
                var role = entry.role || '';
                var statusClass = '';
                if (role === 'assistant') statusClass = 'success';
                else if (role === 'system') statusClass = 'running';
                else if (role === 'error') statusClass = 'failed';
                html += '<span class="cron-history-status ' + statusClass + '">' + escapeHtml(role) + '</span>';
                html += '<span class="cron-history-message">' + escapeHtml(entry.content || '') + '</span>';
            } else {
                // Monitor: step + data
                var step = entry.step || '';
                var stepClass = '';
                if (step.indexOf('done') !== -1) stepClass = 'success';
                else if (step.indexOf('fail') !== -1) stepClass = 'failed';
                else if (step.indexOf('start') !== -1) stepClass = 'running';
                html += '<span class="cron-history-status ' + stepClass + '">' + escapeHtml(step) + '</span>';

                // Collect remaining data fields (exclude timestamp and step)
                var dataFields = [];
                for (var key in entry) {
                    if (entry.hasOwnProperty(key) && key !== 'timestamp' && key !== 'step') {
                        dataFields.push(key + '=' + String(entry[key]));
                    }
                }
                html += '<span class="cron-history-message">' + escapeHtml(dataFields.join(' ')) + '</span>';
            }

            html += '</div>';
        }

        cronDetailHistory.innerHTML = html;
    }

    function formatTimestamp(isoString) {
        if (!isoString) return '';
        try {
            var d = new Date(isoString);
            if (isNaN(d.getTime())) return isoString;
            var h = String(d.getHours()).padStart(2, '0');
            var m = String(d.getMinutes()).padStart(2, '0');
            var s = String(d.getSeconds()).padStart(2, '0');
            var mo = String(d.getMonth() + 1).padStart(2, '0');
            var day = String(d.getDate()).padStart(2, '0');
            return mo + '/' + day + ' ' + h + ':' + m + ':' + s;
        } catch (e) {
            return isoString;
        }
    }

    // ── Tab Switching ────────────────────────────────────────

    function switchTab(tab) {
        activeTab = tab || 'chat';
        updateTabUI();
        renderCurrentHistory();
    }

    function updateTabUI() {
        tabButtons.forEach(function (btn) {
            if (btn.getAttribute('data-tab') === activeTab) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    // ── Utilities ────────────────────────────────────────────

    function escapeHtml(str) {
        if (str == null) return '';
        var div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    function escapeAttr(str) {
        if (str == null) return '';
        return String(str).replace(/[&"'<>]/g, function (c) {
            return { '&': '&amp;', '"': '&quot;', "'": '&#39;', '<': '&lt;', '>': '&gt;' }[c];
        });
    }

})();
