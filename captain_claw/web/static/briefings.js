/* Captain Claw — Sister Session / Briefings Browser */

(function () {
    'use strict';

    // ── DOM refs ────────────────────────────────────────────
    var listEl = document.getElementById('contentList');
    var toastEl = document.getElementById('toast');
    var headerUnread = document.getElementById('headerUnread');

    // Stats
    var statUnread = document.getElementById('statUnread');
    var statBriefings = document.getElementById('statBriefings');
    var statQueued = document.getElementById('statQueued');
    var statTokens = document.getElementById('statTokens');

    // Tabs
    var tabs = document.querySelectorAll('.brf-tab');
    var tabBriefingCount = document.getElementById('tabBriefingCount');
    var tabTaskCount = document.getElementById('tabTaskCount');
    var tabWatchCount = document.getElementById('tabWatchCount');

    // Toolbars
    var briefingsToolbar = document.getElementById('briefingsToolbar');
    var tasksToolbar = document.getElementById('tasksToolbar');
    var watchesToolbar = document.getElementById('watchesToolbar');

    // Briefing controls
    var briefingStatusFilter = document.getElementById('briefingStatusFilter');
    var dismissAllBtn = document.getElementById('dismissAllBtn');
    var briefingCount = document.getElementById('briefingCount');

    // Task controls
    var taskStatusFilter = document.getElementById('taskStatusFilter');
    var createTaskBtn = document.getElementById('createTaskBtn');
    var taskCount = document.getElementById('taskCount');

    // Watch controls
    var createWatchBtn = document.getElementById('createWatchBtn');
    var watchCount = document.getElementById('watchCount');

    // Detail modal
    var detailOverlay = document.getElementById('detailOverlay');
    var detailTitle = document.getElementById('detailTitle');
    var detailStatus = document.getElementById('detailStatus');
    var detailSource = document.getElementById('detailSource');
    var detailSummary = document.getElementById('detailSummary');
    var detailFindings = document.getElementById('detailFindings');
    var detailActionable = document.getElementById('detailActionable');
    var detailConfidence = document.getElementById('detailConfidence');
    var detailTags = document.getElementById('detailTags');
    var detailCreated = document.getElementById('detailCreated');
    var detailDeleteBtn = document.getElementById('detailDeleteBtn');
    var detailDismissBtn = document.getElementById('detailDismissBtn');
    var detailActedBtn = document.getElementById('detailActedBtn');
    var detailCloseBtn = document.getElementById('detailCloseBtn');

    // Task modal
    var taskOverlay = document.getElementById('taskOverlay');
    var taskQuery = document.getElementById('taskQuery');
    var taskPriority = document.getElementById('taskPriority');
    var taskSubmitBtn = document.getElementById('taskSubmitBtn');
    var taskCancelBtn = document.getElementById('taskCancelBtn');

    // Watch modal
    var watchOverlay = document.getElementById('watchOverlay');
    var watchQuery = document.getElementById('watchQuery');
    var watchInterval = document.getElementById('watchInterval');
    var watchSubmitBtn = document.getElementById('watchSubmitBtn');
    var watchCancelBtn = document.getElementById('watchCancelBtn');

    var activeTab = 'briefings';
    var currentBriefingId = null;

    // ── API helper ──────────────────────────────────────────

    function apiFetch(url, options) {
        options = options || {};
        var fetchOpts = {
            method: options.method || 'GET',
            headers: {}
        };
        if (options.body) {
            fetchOpts.headers['Content-Type'] = 'application/json';
            fetchOpts.body = JSON.stringify(options.body);
        }
        return fetch(url, fetchOpts).then(function (res) {
            if (res.status === 204) return { ok: true };
            return res.json().then(function (data) {
                if (!res.ok && data.error) throw new Error(data.error);
                return data;
            });
        });
    }

    // ── Toast ───────────────────────────────────────────────

    function toast(msg) {
        toastEl.textContent = msg;
        toastEl.classList.add('visible');
        setTimeout(function () { toastEl.classList.remove('visible'); }, 2500);
    }

    // ── Stats ───────────────────────────────────────────────

    function loadStats() {
        apiFetch('/api/sister/stats').then(function (data) {
            statUnread.textContent = data.unread_briefings || 0;
            statBriefings.textContent = data.total_briefings || 0;
            statQueued.textContent = data.queued_tasks || 0;
            statTokens.textContent = data.daily_tokens_used || 0;

            var unread = data.unread_briefings || 0;
            if (unread > 0) {
                headerUnread.textContent = unread;
                headerUnread.style.display = '';
            } else {
                headerUnread.style.display = 'none';
            }

            tabBriefingCount.textContent = '(' + (data.total_briefings || 0) + ')';
            tabTaskCount.textContent = '(' + (data.total_tasks || 0) + ')';
        }).catch(function () {
            statUnread.textContent = '—';
        });
    }

    // ── Briefings ───────────────────────────────────────────

    function loadBriefings() {
        var status = briefingStatusFilter.value;
        var params = ['limit=100'];
        if (status) params.push('status=' + encodeURIComponent(status));
        var url = '/api/briefings?' + params.join('&');

        apiFetch(url).then(function (data) {
            renderBriefings(data.items || []);
            briefingCount.textContent = (data.items || []).length + ' of ' + (data.total || 0);
        }).catch(function (err) {
            listEl.innerHTML = '<div class="brf-empty">Error loading briefings: ' + escHtml(err.message) + '</div>';
        });
    }

    function renderBriefings(items) {
        if (!items.length) {
            listEl.innerHTML = '<div class="brf-empty">No briefings yet. The sister session will automatically investigate insights and intuitions and report findings here.</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var source = it.source_type || 'manual';
            var status = it.status || 'unread';
            var unreadClass = status === 'unread' ? ' unread' : '';
            var summary = it.summary || '(no summary)';
            var content = it.content || '';
            var preview = content.length > 150 ? content.substring(0, 150) + '...' : content;
            var created = it.created_at ? formatDate(it.created_at) : '';
            var actionable = it.actionable ? '<span class="brf-actionable">actionable</span>' : '<span class="brf-not-actionable">info only</span>';
            var conf = it.confidence != null ? parseFloat(it.confidence).toFixed(2) : '';

            var meta = [];
            if (created) meta.push(created);
            if (conf) meta.push('conf: ' + conf);
            if (it.tags) meta.push(escHtml(it.tags));

            html += '<div class="brf-card' + unreadClass + '" data-id="' + escHtml(it.id) + '">'
                + '<div class="brf-card-top">'
                + '<span class="brf-badge brf-badge-' + source + '">' + source + '</span>'
                + '<span class="brf-status-badge brf-status-' + status + '">' + status + '</span>'
                + ' ' + actionable
                + '</div>'
                + '<div class="brf-card-summary">' + escHtml(summary) + '</div>'
                + (preview ? '<div class="brf-card-content">' + escHtml(preview) + '</div>' : '')
                + '<div class="brf-card-meta">' + meta.join(' &middot; ') + '</div>'
                + '</div>';
        }
        listEl.innerHTML = html;

        var cards = listEl.querySelectorAll('.brf-card');
        for (var j = 0; j < cards.length; j++) {
            cards[j].addEventListener('click', onBriefingClick);
        }
    }

    function onBriefingClick(e) {
        var card = e.currentTarget;
        var id = card.getAttribute('data-id');
        openBriefingDetail(id);
    }

    // ── Briefing Detail ─────────────────────────────────────

    function openBriefingDetail(id) {
        currentBriefingId = id;
        apiFetch('/api/briefings/' + id).then(function (item) {
            detailTitle.textContent = 'Briefing';
            detailStatus.innerHTML = '<span class="brf-status-badge brf-status-' + (item.status || 'unread') + '">' + (item.status || 'unread') + '</span>';
            detailSource.innerHTML = '<span class="brf-badge brf-badge-' + (item.source_type || 'manual') + '">' + (item.source_type || 'manual') + '</span>';
            detailSummary.textContent = item.summary || '(no summary)';
            detailFindings.textContent = item.content || '(no content)';
            detailActionable.innerHTML = item.actionable ? '<span class="brf-actionable">Yes — actionable</span>' : '<span class="brf-not-actionable">No — informational</span>';
            detailConfidence.textContent = item.confidence != null ? parseFloat(item.confidence).toFixed(2) : '—';
            detailTags.textContent = item.tags || '—';
            detailCreated.textContent = item.created_at || '';

            // Show/hide dismiss button based on status
            detailDismissBtn.style.display = (item.status === 'dismissed') ? 'none' : '';
            detailActedBtn.style.display = (item.status === 'acted') ? 'none' : '';

            detailOverlay.classList.add('visible');
            loadStats();
            loadBriefings();
        }).catch(function (err) {
            toast('Error: ' + err.message);
        });
    }

    function closeBriefingDetail() {
        detailOverlay.classList.remove('visible');
        currentBriefingId = null;
    }

    function dismissBriefing() {
        if (!currentBriefingId) return;
        apiFetch('/api/briefings/' + currentBriefingId, { method: 'PATCH', body: { status: 'dismissed' } }).then(function () {
            toast('Briefing dismissed');
            closeBriefingDetail();
            loadBriefings();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    function markActed() {
        if (!currentBriefingId) return;
        apiFetch('/api/briefings/' + currentBriefingId, { method: 'PATCH', body: { status: 'acted' } }).then(function () {
            toast('Briefing marked as acted');
            closeBriefingDetail();
            loadBriefings();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    function deleteBriefing() {
        if (!currentBriefingId) return;
        if (!confirm('Delete this briefing?')) return;
        apiFetch('/api/briefings/' + currentBriefingId, { method: 'DELETE' }).then(function () {
            toast('Briefing deleted');
            closeBriefingDetail();
            loadBriefings();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    function dismissAllUnread() {
        apiFetch('/api/briefings?status=unread&limit=200').then(function (data) {
            var items = data.items || [];
            if (!items.length) {
                toast('No unread briefings');
                return;
            }
            var promises = items.map(function (it) {
                return apiFetch('/api/briefings/' + it.id, { method: 'PATCH', body: { status: 'dismissed' } });
            });
            return Promise.all(promises);
        }).then(function () {
            toast('All unread briefings dismissed');
            loadBriefings();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    // ── Tasks ───────────────────────────────────────────────

    function loadTasks() {
        var status = taskStatusFilter.value;
        var params = ['limit=100'];
        if (status) params.push('status=' + encodeURIComponent(status));
        var url = '/api/sister/tasks?' + params.join('&');

        apiFetch(url).then(function (data) {
            renderTasks(data.items || []);
            taskCount.textContent = (data.items || []).length + ' of ' + (data.total || 0);
        }).catch(function (err) {
            listEl.innerHTML = '<div class="brf-empty">Error loading tasks: ' + escHtml(err.message) + '</div>';
        });
    }

    function renderTasks(items) {
        if (!items.length) {
            listEl.innerHTML = '<div class="brf-empty">No tasks. Tasks are automatically created from insights and intuitions, or create one manually.</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var status = it.status || 'queued';
            var source = it.source_type || 'manual';
            var priority = it.priority || 5;
            var created = it.created_at ? formatDate(it.created_at) : '';
            var budget = it.token_budget || 0;
            var used = it.tokens_used || 0;

            var meta = [];
            meta.push('<span class="brf-badge brf-badge-' + source + '">' + source + '</span>');
            if (created) meta.push(created);
            if (budget) meta.push('budget: ' + used + '/' + budget);

            html += '<div class="brf-task-card" data-id="' + escHtml(it.id) + '">'
                + '<div class="brf-task-top">'
                + '<span class="brf-task-status brf-task-' + status + '">' + status + '</span>'
                + '<span class="brf-task-priority">priority: ' + priority + '/10</span>'
                + '</div>'
                + '<div class="brf-task-reason">' + escHtml(it.trigger_reason || '') + '</div>'
                + '<div class="brf-task-meta">' + meta.join(' &middot; ')
                + (status === 'queued' ? '<span class="brf-card-actions" style="margin-left:auto"><button class="task-delete-btn" data-id="' + escHtml(it.id) + '">Cancel</button></span>' : '')
                + '</div>'
                + '</div>';
        }
        listEl.innerHTML = html;

        var deleteBtns = listEl.querySelectorAll('.task-delete-btn');
        for (var j = 0; j < deleteBtns.length; j++) {
            deleteBtns[j].addEventListener('click', function (e) {
                e.stopPropagation();
                var id = this.getAttribute('data-id');
                apiFetch('/api/sister/tasks/' + id, { method: 'DELETE' }).then(function () {
                    toast('Task cancelled');
                    loadTasks();
                    loadStats();
                }).catch(function (err) { toast('Error: ' + err.message); });
            });
        }
    }

    // ── Create Task Modal ───────────────────────────────────

    function openTaskModal() {
        taskQuery.value = '';
        taskPriority.value = '8';
        taskOverlay.classList.add('visible');
        taskQuery.focus();
    }

    function closeTaskModal() {
        taskOverlay.classList.remove('visible');
    }

    function submitTask() {
        var query = taskQuery.value.trim();
        if (!query) {
            toast('Please enter a query');
            return;
        }
        var priority = parseInt(taskPriority.value, 10) || 8;
        apiFetch('/api/sister/tasks', { method: 'POST', body: { query: query, priority: priority } }).then(function () {
            toast('Investigation task created');
            closeTaskModal();
            loadTasks();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    // ── Watches ─────────────────────────────────────────────

    function loadWatches() {
        apiFetch('/api/sister/watches').then(function (data) {
            renderWatches(data.items || []);
            watchCount.textContent = (data.items || []).length + ' watches';
            tabWatchCount.textContent = '(' + (data.items || []).length + ')';
        }).catch(function (err) {
            listEl.innerHTML = '<div class="brf-empty">Error loading watches: ' + escHtml(err.message) + '</div>';
        });
    }

    function renderWatches(items) {
        if (!items.length) {
            listEl.innerHTML = '<div class="brf-empty">No active watches. Create a watch to periodically investigate a topic.</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var interval = formatInterval(it.interval_seconds || 3600);
            var nextCheck = it.next_check_at ? formatDate(it.next_check_at) : 'soon';
            var created = it.created_at ? formatDate(it.created_at) : '';

            html += '<div class="brf-watch-card" data-id="' + escHtml(it.id) + '">'
                + '<div class="brf-watch-top">'
                + '<span class="brf-badge brf-badge-watch">watch</span>'
                + '<span style="font-size:11px;color:var(--text-muted);margin-left:auto">' + interval + '</span>'
                + '</div>'
                + '<div class="brf-watch-query">' + escHtml(it.query || '') + '</div>'
                + '<div class="brf-watch-meta">'
                + (created ? 'created ' + created : '')
                + ' &middot; next check: ' + nextCheck
                + '<span class="brf-card-actions" style="margin-left:auto"><button class="watch-delete-btn" data-id="' + escHtml(it.id) + '">Delete</button></span>'
                + '</div>'
                + '</div>';
        }
        listEl.innerHTML = html;

        var deleteBtns = listEl.querySelectorAll('.watch-delete-btn');
        for (var j = 0; j < deleteBtns.length; j++) {
            deleteBtns[j].addEventListener('click', function (e) {
                e.stopPropagation();
                var id = this.getAttribute('data-id');
                apiFetch('/api/sister/watches/' + id, { method: 'DELETE' }).then(function () {
                    toast('Watch deleted');
                    loadWatches();
                }).catch(function (err) { toast('Error: ' + err.message); });
            });
        }
    }

    // ── Create Watch Modal ──────────────────────────────────

    function openWatchModal() {
        watchQuery.value = '';
        watchInterval.value = '3600';
        watchOverlay.classList.add('visible');
        watchQuery.focus();
    }

    function closeWatchModal() {
        watchOverlay.classList.remove('visible');
    }

    function submitWatch() {
        var query = watchQuery.value.trim();
        if (!query) {
            toast('Please enter a query');
            return;
        }
        var interval = parseInt(watchInterval.value, 10) || 3600;
        apiFetch('/api/sister/watches', { method: 'POST', body: { query: query, interval_seconds: interval } }).then(function () {
            toast('Watch created');
            closeWatchModal();
            loadWatches();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    // ── Tab switching ───────────────────────────────────────

    function switchTab(tabName) {
        activeTab = tabName;
        for (var i = 0; i < tabs.length; i++) {
            if (tabs[i].getAttribute('data-tab') === tabName) {
                tabs[i].classList.add('active');
            } else {
                tabs[i].classList.remove('active');
            }
        }

        briefingsToolbar.style.display = tabName === 'briefings' ? '' : 'none';
        tasksToolbar.style.display = tabName === 'tasks' ? '' : 'none';
        watchesToolbar.style.display = tabName === 'watches' ? '' : 'none';

        if (tabName === 'briefings') loadBriefings();
        else if (tabName === 'tasks') loadTasks();
        else if (tabName === 'watches') loadWatches();
    }

    // ── Helpers ──────────────────────────────────────────────

    function escHtml(s) {
        if (!s) return '';
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatDate(iso) {
        if (!iso) return '';
        try {
            var d = new Date(iso);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) {
            return iso;
        }
    }

    function formatInterval(seconds) {
        if (seconds < 60) return seconds + 's';
        if (seconds < 3600) return Math.round(seconds / 60) + 'm';
        if (seconds < 86400) return Math.round(seconds / 3600) + 'h';
        return Math.round(seconds / 86400) + 'd';
    }

    // ── Events ──────────────────────────────────────────────

    for (var i = 0; i < tabs.length; i++) {
        tabs[i].addEventListener('click', function () {
            switchTab(this.getAttribute('data-tab'));
        });
    }

    briefingStatusFilter.addEventListener('change', loadBriefings);
    dismissAllBtn.addEventListener('click', dismissAllUnread);

    taskStatusFilter.addEventListener('change', loadTasks);
    createTaskBtn.addEventListener('click', openTaskModal);
    taskSubmitBtn.addEventListener('click', submitTask);
    taskCancelBtn.addEventListener('click', closeTaskModal);

    createWatchBtn.addEventListener('click', openWatchModal);
    watchSubmitBtn.addEventListener('click', submitWatch);
    watchCancelBtn.addEventListener('click', closeWatchModal);

    detailDismissBtn.addEventListener('click', dismissBriefing);
    detailActedBtn.addEventListener('click', markActed);
    detailDeleteBtn.addEventListener('click', deleteBriefing);
    detailCloseBtn.addEventListener('click', closeBriefingDetail);

    detailOverlay.addEventListener('click', function (e) {
        if (e.target === detailOverlay) closeBriefingDetail();
    });
    taskOverlay.addEventListener('click', function (e) {
        if (e.target === taskOverlay) closeTaskModal();
    });
    watchOverlay.addEventListener('click', function (e) {
        if (e.target === watchOverlay) closeWatchModal();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            closeBriefingDetail();
            closeTaskModal();
            closeWatchModal();
        }
    });

    // ── Init ────────────────────────────────────────────────

    loadStats();
    loadBriefings();
})();
