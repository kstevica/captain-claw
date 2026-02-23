/* Captain Claw — Sessions Browser */

(function () {
    'use strict';

    // ── State ───────────────────────────────────────────
    let sessions = [];
    let selectedId = null;
    let selectedSession = null;
    let selectedIds = new Set();  // multiselect for bulk actions

    // ── DOM ─────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const sessList = $('#sessList');
    const sessCount = $('#sessCount');
    const searchInput = $('#searchInput');
    const sessDetail = $('#sessDetail');
    const sessDetailEmpty = $('#sessDetailEmpty');
    const sessActions = $('#sessActions');
    const sessPanels = $('#sessPanels');
    const sessActiveName = $('#sessActiveName');
    const sessActiveMeta = $('#sessActiveMeta');
    const sessChatMessages = $('#sessChatMessages');
    const sessMonitorMessages = $('#sessMonitorMessages');
    const chatMsgCount = $('#chatMsgCount');
    const monitorMsgCount = $('#monitorMsgCount');

    // Multiselect
    const selectAllCb = $('#selectAllCb');
    const bulkDeleteBtn = $('#bulkDeleteBtn');

    // Modals
    const renameModal = $('#renameModal');
    const renameInput = $('#renameInput');
    const descModal = $('#descModal');
    const descInput = $('#descInput');

    // ── Init ────────────────────────────────────────────
    loadSessions();

    searchInput.addEventListener('input', () => {
        const q = searchInput.value.trim().toLowerCase();
        renderSessionList(q);
    });

    // Action buttons
    $('#renameBtn').addEventListener('click', showRenameModal);
    $('#descBtn').addEventListener('click', showDescModal);
    $('#autoDescBtn').addEventListener('click', autoDescribe);
    $('#deleteBtn').addEventListener('click', deleteSession);

    // Modal buttons
    $('#renameSaveBtn').addEventListener('click', saveRename);
    $('#renameCancelBtn').addEventListener('click', () => renameModal.classList.add('hidden'));
    $('#descSaveBtn').addEventListener('click', saveDescription);
    $('#descCancelBtn').addEventListener('click', () => descModal.classList.add('hidden'));

    // Export buttons
    $('#exportChatBtn').addEventListener('click', () => exportSession('chat'));
    $('#exportMonitorBtn').addEventListener('click', () => exportSession('monitor'));
    $('#exportAllBtn').addEventListener('click', () => exportSession('all'));

    // Multiselect
    selectAllCb.addEventListener('change', toggleSelectAll);
    bulkDeleteBtn.addEventListener('click', bulkDeleteSessions);

    // Close modals on overlay click
    renameModal.addEventListener('click', (e) => { if (e.target === renameModal) renameModal.classList.add('hidden'); });
    descModal.addEventListener('click', (e) => { if (e.target === descModal) descModal.classList.add('hidden'); });

    // Keyboard shortcuts in modals
    renameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') saveRename(); if (e.key === 'Escape') renameModal.classList.add('hidden'); });
    descInput.addEventListener('keydown', (e) => { if (e.key === 'Escape') descModal.classList.add('hidden'); });

    // Resize handle
    initResize();

    // ── API ─────────────────────────────────────────────

    async function loadSessions() {
        try {
            const res = await fetch('/api/sessions');
            sessions = await res.json();
            sessCount.textContent = sessions.length + ' session' + (sessions.length !== 1 ? 's' : '');
            // Prune selectedIds to only keep IDs that still exist
            const existingIds = new Set(sessions.map(s => s.id));
            for (const id of selectedIds) {
                if (!existingIds.has(id)) selectedIds.delete(id);
            }
            renderSessionList(searchInput.value.trim().toLowerCase());
        } catch (e) {
            sessList.innerHTML = '<div class="sess-loading">Failed to load sessions</div>';
        }
    }

    function renderSessionList(filter) {
        const filtered = filter
            ? sessions.filter(s => s.name.toLowerCase().includes(filter) || (s.description || '').toLowerCase().includes(filter))
            : sessions;

        if (!filtered.length) {
            sessList.innerHTML = '<div class="sess-loading">' + (filter ? 'No matching sessions' : 'No sessions yet') + '</div>';
            updateBulkUI();
            return;
        }

        sessList.innerHTML = filtered.map(s => {
            const active = s.id === selectedId ? ' active' : '';
            const checked = selectedIds.has(s.id) ? ' checked' : '';
            const desc = s.description ? escapeHtml(s.description) : '';
            const ago = timeAgo(s.updated_at);
            return '<div class="sess-item' + active + '" data-id="' + escapeHtml(s.id) + '">'
                + '<label class="sess-item-checkbox" onclick="event.stopPropagation()">'
                + '<input type="checkbox" class="sess-cb" data-id="' + escapeHtml(s.id) + '"' + checked + '>'
                + '</label>'
                + '<div class="sess-item-content">'
                + '<div class="sess-item-name">' + escapeHtml(s.name) + '</div>'
                + (desc ? '<div class="sess-item-desc">' + desc + '</div>' : '')
                + '<div class="sess-item-meta">' + s.message_count + ' msgs &middot; ' + ago + '</div>'
                + '</div>'
                + '</div>';
        }).join('');

        // Bind click on content area to select session
        sessList.querySelectorAll('.sess-item').forEach(el => {
            el.querySelector('.sess-item-content').addEventListener('click', () => selectSession(el.dataset.id));
        });

        // Bind checkboxes
        sessList.querySelectorAll('.sess-cb').forEach(cb => {
            cb.addEventListener('change', () => {
                if (cb.checked) {
                    selectedIds.add(cb.dataset.id);
                } else {
                    selectedIds.delete(cb.dataset.id);
                }
                updateBulkUI();
            });
        });

        updateBulkUI();
    }

    async function selectSession(id) {
        selectedId = id;
        renderSessionList(searchInput.value.trim().toLowerCase());

        // Show loading
        sessDetailEmpty.classList.add('hidden');
        sessActions.classList.remove('hidden');
        sessPanels.classList.remove('hidden');
        sessChatMessages.innerHTML = '<div class="sess-loading">Loading...</div>';
        sessMonitorMessages.innerHTML = '<div class="sess-loading">Loading...</div>';

        try {
            const res = await fetch('/api/sessions/' + encodeURIComponent(id));
            if (!res.ok) throw new Error('Not found');
            selectedSession = await res.json();
        } catch (e) {
            sessChatMessages.innerHTML = '<div class="sess-loading">Failed to load session</div>';
            sessMonitorMessages.innerHTML = '';
            return;
        }

        // Update actions bar
        sessActiveName.textContent = selectedSession.name;
        const desc = selectedSession.description || '';
        sessActiveMeta.textContent = selectedSession.message_count + ' messages' + (desc ? ' — ' + desc : '');

        // Split messages
        const chatMsgs = [];
        const monitorMsgs = [];
        for (const msg of selectedSession.messages) {
            if (msg.role === 'tool' && msg.tool_name) {
                monitorMsgs.push(msg);
            } else if (msg.role === 'user' || msg.role === 'assistant' || msg.role === 'system') {
                chatMsgs.push(msg);
            }
        }

        renderChatHistory(chatMsgs);
        renderMonitorHistory(monitorMsgs);
        chatMsgCount.textContent = chatMsgs.length;
        monitorMsgCount.textContent = monitorMsgs.length;
    }

    // ── Render Chat ─────────────────────────────────────

    function renderChatHistory(messages) {
        if (!messages.length) {
            sessChatMessages.innerHTML = '<div class="sess-loading">No chat messages</div>';
            return;
        }

        sessChatMessages.innerHTML = '';
        for (const msg of messages) {
            const div = document.createElement('div');
            div.className = 'sess-msg ' + msg.role;

            const label = document.createElement('div');
            label.className = 'sess-msg-label';
            label.textContent = msg.role === 'user' ? 'You' : msg.role === 'system' ? 'System' : 'Captain Claw';

            const bubble = document.createElement('div');
            bubble.className = 'sess-msg-bubble';
            bubble.innerHTML = renderMarkdown(msg.content || '');

            div.appendChild(label);
            div.appendChild(bubble);

            if (msg.timestamp) {
                const time = document.createElement('div');
                time.className = 'sess-msg-time';
                time.textContent = formatTime(msg.timestamp);
                div.appendChild(time);
            }

            sessChatMessages.appendChild(div);
        }
    }

    // ── Render Monitor ──────────────────────────────────

    function renderMonitorHistory(messages) {
        if (!messages.length) {
            sessMonitorMessages.innerHTML = '<div class="sess-loading">No tool activity</div>';
            return;
        }

        sessMonitorMessages.innerHTML = '';
        for (const msg of messages) {
            const entry = document.createElement('div');
            entry.className = 'sess-monitor-entry';

            // Header
            const header = document.createElement('div');
            header.className = 'sess-monitor-header';

            const toggle = document.createElement('span');
            toggle.className = 'sess-monitor-toggle';
            toggle.textContent = '\u25B6';

            const tool = document.createElement('span');
            tool.className = 'sess-monitor-tool';
            tool.textContent = msg.tool_name || 'tool';

            const args = document.createElement('span');
            args.className = 'sess-monitor-args';
            args.textContent = summarizeArgs(msg.tool_arguments);

            header.appendChild(toggle);
            header.appendChild(tool);
            header.appendChild(args);

            if (msg.timestamp) {
                const time = document.createElement('span');
                time.className = 'sess-monitor-time';
                time.textContent = formatTime(msg.timestamp);
                header.appendChild(time);
            }

            // Body
            const body = document.createElement('div');
            body.className = 'sess-monitor-body';
            body.textContent = msg.content || '[no output]';

            // Toggle
            header.addEventListener('click', () => {
                const open = body.classList.toggle('open');
                toggle.textContent = open ? '\u25BC' : '\u25B6';
            });

            entry.appendChild(header);
            entry.appendChild(body);
            sessMonitorMessages.appendChild(entry);
        }
    }

    function summarizeArgs(args) {
        if (!args || typeof args !== 'object') return '';
        const keys = Object.keys(args);
        if (!keys.length) return '';
        // Show first key=value, truncated
        const first = keys[0];
        let val = String(args[first] || '');
        if (val.length > 60) val = val.substring(0, 57) + '...';
        return first + '=' + val;
    }

    // ── Actions ─────────────────────────────────────────

    function showRenameModal() {
        if (!selectedSession) return;
        renameInput.value = selectedSession.name;
        renameModal.classList.remove('hidden');
        renameInput.focus();
        renameInput.select();
    }

    async function saveRename() {
        const name = renameInput.value.trim();
        if (!name || !selectedSession) return;
        try {
            const res = await fetch('/api/sessions/' + encodeURIComponent(selectedSession.id), {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            });
            if (!res.ok) throw new Error('Failed');
            selectedSession.name = name;
            sessActiveName.textContent = name;
            renameModal.classList.add('hidden');
            toast('Session renamed', 'success');
            loadSessions();
        } catch (e) {
            toast('Failed to rename session', 'error');
        }
    }

    function showDescModal() {
        if (!selectedSession) return;
        descInput.value = selectedSession.description || '';
        descModal.classList.remove('hidden');
        descInput.focus();
    }

    async function saveDescription() {
        if (!selectedSession) return;
        const description = descInput.value.trim();
        try {
            const res = await fetch('/api/sessions/' + encodeURIComponent(selectedSession.id), {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description }),
            });
            if (!res.ok) throw new Error('Failed');
            selectedSession.description = description;
            sessActiveMeta.textContent = selectedSession.message_count + ' messages'
                + (description ? ' \u2014 ' + description : '');
            descModal.classList.add('hidden');
            toast('Description saved', 'success');
            loadSessions();
        } catch (e) {
            toast('Failed to save description', 'error');
        }
    }

    async function autoDescribe() {
        if (!selectedSession) return;
        const btn = $('#autoDescBtn');
        btn.disabled = true;
        btn.textContent = '\u23F3 Generating...';
        try {
            const res = await fetch('/api/sessions/' + encodeURIComponent(selectedSession.id) + '/auto-describe', {
                method: 'POST',
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed');
            selectedSession.description = data.description;
            sessActiveMeta.textContent = selectedSession.message_count + ' messages \u2014 ' + data.description;
            toast('Description generated', 'success');
            loadSessions();
        } catch (e) {
            toast('Auto-describe failed: ' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = '\uD83E\uDD16 Auto Describe';
        }
    }

    async function deleteSession() {
        if (!selectedSession) return;
        if (!confirm('Delete session "' + selectedSession.name + '"? This cannot be undone.')) return;
        try {
            const res = await fetch('/api/sessions/' + encodeURIComponent(selectedSession.id), { method: 'DELETE' });
            if (!res.ok) throw new Error('Failed');
            selectedId = null;
            selectedSession = null;
            sessActions.classList.add('hidden');
            sessPanels.classList.add('hidden');
            sessDetailEmpty.classList.remove('hidden');
            toast('Session deleted', 'success');
            loadSessions();
        } catch (e) {
            toast('Failed to delete session', 'error');
        }
    }

    // ── Multiselect / Bulk Actions ─────────────────────

    function toggleSelectAll() {
        const checkAll = selectAllCb.checked;
        const visible = sessList.querySelectorAll('.sess-cb');
        visible.forEach(cb => {
            cb.checked = checkAll;
            if (checkAll) {
                selectedIds.add(cb.dataset.id);
            } else {
                selectedIds.delete(cb.dataset.id);
            }
        });
        updateBulkUI();
    }

    function updateBulkUI() {
        const count = selectedIds.size;
        if (count > 0) {
            bulkDeleteBtn.classList.remove('hidden');
            bulkDeleteBtn.textContent = '\uD83D\uDDD1\uFE0F Delete Selected (' + count + ')';
        } else {
            bulkDeleteBtn.classList.add('hidden');
        }

        // Sync select-all checkbox state
        const visible = sessList.querySelectorAll('.sess-cb');
        if (visible.length > 0 && count >= visible.length) {
            selectAllCb.checked = true;
            selectAllCb.indeterminate = false;
        } else if (count > 0) {
            selectAllCb.checked = false;
            selectAllCb.indeterminate = true;
        } else {
            selectAllCb.checked = false;
            selectAllCb.indeterminate = false;
        }
    }

    async function bulkDeleteSessions() {
        const count = selectedIds.size;
        if (!count) return;
        if (!confirm('Delete ' + count + ' session' + (count > 1 ? 's' : '') + '? This cannot be undone.')) return;

        const ids = Array.from(selectedIds);
        try {
            const res = await fetch('/api/sessions/bulk-delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids }),
            });
            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            const deletedCount = (data.deleted || []).length;
            const failedCount = (data.failed || []).length;

            // Clear selection for deleted ones
            for (const id of (data.deleted || [])) {
                selectedIds.delete(id);
            }

            // If the currently viewed session was deleted, reset detail view
            if (selectedId && (data.deleted || []).includes(selectedId)) {
                selectedId = null;
                selectedSession = null;
                sessActions.classList.add('hidden');
                sessPanels.classList.add('hidden');
                sessDetailEmpty.classList.remove('hidden');
            }

            if (failedCount) {
                toast('Deleted ' + deletedCount + ', failed ' + failedCount, 'error');
            } else {
                toast(deletedCount + ' session' + (deletedCount > 1 ? 's' : '') + ' deleted', 'success');
            }
            loadSessions();
        } catch (e) {
            toast('Bulk delete failed', 'error');
        }
    }

    // ── Export ───────────────────────────────────────────

    function exportSession(mode) {
        if (!selectedSession) return;
        // Trigger a browser download via a hidden link
        const url = '/api/sessions/' + encodeURIComponent(selectedSession.id)
            + '/export?mode=' + encodeURIComponent(mode);
        const a = document.createElement('a');
        a.href = url;
        a.download = '';  // browser will use Content-Disposition filename
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast('Downloading ' + mode + ' export...', 'success');
    }

    // ── Resize Handle ───────────────────────────────────

    function initResize() {
        const handle = $('#sessResizeHandle');
        const chatPanel = $('#sessChatPanel');
        const monitorPanel = $('#sessMonitorPanel');
        let startX, startChatWidth;

        handle.addEventListener('mousedown', (e) => {
            startX = e.clientX;
            startChatWidth = chatPanel.offsetWidth;
            handle.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            const onMove = (e) => {
                const dx = e.clientX - startX;
                const totalWidth = chatPanel.parentElement.offsetWidth - handle.offsetWidth;
                const newChatWidth = Math.max(200, Math.min(totalWidth - 200, startChatWidth + dx));
                const pct = (newChatWidth / totalWidth) * 100;
                chatPanel.style.flex = '0 0 ' + pct + '%';
                monitorPanel.style.flex = '1 1 0';
            };

            const onUp = () => {
                handle.classList.remove('dragging');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }

    // ── Utilities ────────────────────────────────────────

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function renderMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
            '<pre><code>' + code + '</code></pre>'
        );

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // Unordered lists
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Ordered lists
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // Paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        if (!html.startsWith('<')) html = '<p>' + html + '</p>';

        return html;
    }

    function timeAgo(isoStr) {
        if (!isoStr) return '';
        try {
            const d = new Date(isoStr);
            const now = new Date();
            const diffMs = now - d;
            const diffSec = Math.floor(diffMs / 1000);
            if (diffSec < 60) return 'just now';
            const diffMin = Math.floor(diffSec / 60);
            if (diffMin < 60) return diffMin + 'm ago';
            const diffHr = Math.floor(diffMin / 60);
            if (diffHr < 24) return diffHr + 'h ago';
            const diffDay = Math.floor(diffHr / 24);
            if (diffDay < 30) return diffDay + 'd ago';
            return d.toLocaleDateString();
        } catch (e) {
            return '';
        }
    }

    function formatTime(isoStr) {
        if (!isoStr) return '';
        try {
            const d = new Date(isoStr);
            return d.toLocaleString(undefined, {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit',
            });
        } catch (e) {
            return '';
        }
    }

    function toast(message, type) {
        const el = document.createElement('div');
        el.className = 'sess-toast' + (type ? ' ' + type : '');
        el.textContent = message;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 3000);
    }

})();
