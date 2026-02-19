/* Captain Claw Web UI - Frontend Logic */

(function () {
    'use strict';

    // ── State ───────────────────────────────────────────────
    let ws = null;
    let commands = [];
    let sessionInfo = {};
    let isConnected = false;
    let suggestionIndex = -1;
    let paletteIndex = -1;
    let currentApprovalId = null;

    // ── DOM references ──────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const chatMessages = $('#chatMessages');
    const chatEmpty = $('#chatEmpty');
    const chatCount = $('#chatCount');
    const monitorMessages = $('#monitorMessages');
    const monitorEmpty = $('#monitorEmpty');
    const messageInput = $('#messageInput');
    const sendBtn = $('#sendBtn');
    const sessionName = $('#sessionName');
    const modelName = $('#modelName');
    const statusDot = $('#statusDot');
    const statusText = $('#statusText');
    const sidebar = $('#sidebar');
    const commandSuggestions = $('#commandSuggestions');
    const commandPalette = $('#commandPalette');
    const paletteInput = $('#paletteInput');
    const paletteResults = $('#paletteResults');
    const approvalModal = $('#approvalModal');
    const approvalMessage = $('#approvalMessage');
    const sessionsList = $('#sessionsList');
    const instructionsList = $('#instructionsList');
    const instructionEditor = $('#instructionEditor');
    const instructionContent = $('#instructionContent');
    const editorFileName = $('#editorFileName');

    // ── WebSocket ───────────────────────────────────────────

    function connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/ws`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            isConnected = true;
            setStatus('ready', 'ready');
        };

        ws.onclose = () => {
            isConnected = false;
            setStatus('error', 'disconnected');
            // Auto-reconnect after 3 seconds
            setTimeout(connect, 3000);
        };

        ws.onerror = () => {
            setStatus('error', 'error');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    }

    function send(data) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(data));
        }
    }

    // ── Message Handler ─────────────────────────────────────

    function handleMessage(data) {
        switch (data.type) {
            case 'welcome':
                handleWelcome(data);
                break;
            case 'chat_message':
                addChatMessage(data.role, data.content, data.replay);
                break;
            case 'command_result':
                addCommandResult(data.command, data.content);
                break;
            case 'monitor':
                addMonitorEntry(data.tool_name, data.arguments, data.output);
                break;
            case 'status':
                setStatus(data.status, data.status);
                break;
            case 'session_info':
                updateSessionInfo(data);
                break;
            case 'session_switched':
                // Clear chat for the new session and reload messages via reconnect
                clearChat();
                break;
            case 'replay_done':
                // Session history replay finished
                break;
            case 'usage':
                // Could display token usage somewhere
                break;
            case 'approval_request':
                showApprovalModal(data.id, data.message);
                break;
            case 'approval_notice':
                addMonitorEntry('approval', {action: 'auto-approved'}, data.message);
                break;
            case 'error':
                addChatMessage('error', data.message);
                break;
        }
    }

    function handleWelcome(data) {
        commands = data.commands || [];
        if (data.session) {
            updateSessionInfo(data.session);
        }
        if (data.models && data.models.length) {
            const active = data.models.find(m => m.id === 'default') || data.models[0];
            modelName.textContent = `${active.provider}:${active.model}`;
        }
        buildCommandReference();
    }

    // ── Chat ────────────────────────────────────────────────

    let msgCount = 0;

    function addChatMessage(role, content, isReplay) {
        chatEmpty.style.display = 'none';
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        if (isReplay) div.style.animation = 'none';

        const label = document.createElement('div');
        label.className = 'msg-label';
        label.textContent = role === 'user' ? 'You' : role === 'error' ? 'Error' : 'Captain Claw';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = renderMarkdown(content);

        div.appendChild(label);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        msgCount++;
        chatCount.textContent = `${msgCount} messages`;
    }

    function addCommandResult(command, content) {
        chatEmpty.style.display = 'none';
        const div = document.createElement('div');
        div.className = 'msg command-result';

        const label = document.createElement('div');
        label.className = 'msg-label';
        label.textContent = command;

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = renderMarkdown(content);

        div.appendChild(label);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function clearChat() {
        chatMessages.innerHTML = '';
        chatMessages.appendChild(chatEmpty);
        chatEmpty.style.display = '';
        msgCount = 0;
        chatCount.textContent = '0 messages';
    }

    // ── Monitor ─────────────────────────────────────────────

    function addMonitorEntry(toolName, args, output) {
        monitorEmpty.style.display = 'none';
        const div = document.createElement('div');
        div.className = 'monitor-entry';

        const name = document.createElement('div');
        name.className = 'monitor-tool-name';
        name.textContent = toolName;

        const argsDiv = document.createElement('div');
        argsDiv.className = 'monitor-args';
        argsDiv.textContent = typeof args === 'object' ? JSON.stringify(args, null, 0).slice(0, 200) : String(args).slice(0, 200);

        const outDiv = document.createElement('div');
        outDiv.className = 'monitor-output';
        outDiv.textContent = String(output).slice(0, 5000);

        div.appendChild(name);
        div.appendChild(argsDiv);
        div.appendChild(outDiv);
        monitorMessages.appendChild(div);
        monitorMessages.scrollTop = monitorMessages.scrollHeight;
    }

    // ── Status ──────────────────────────────────────────────

    function setStatus(dotClass, text) {
        statusDot.className = `status-dot ${dotClass}`;
        statusText.textContent = text;
    }

    function updateSessionInfo(info) {
        sessionInfo = info;
        if (info.name) sessionName.textContent = info.name;
        if (info.model) {
            const provider = info.provider || '';
            modelName.textContent = provider ? `${provider}:${info.model}` : info.model;
        }
    }

    // ── Input Handling ──────────────────────────────────────

    function handleSend() {
        const text = messageInput.value.trim();
        if (!text) return;
        if (!isConnected) return;

        hideSuggestions();

        if (text.startsWith('/')) {
            send({ type: 'command', command: text });
        } else {
            send({ type: 'chat', content: text });
        }

        messageInput.value = '';
        autoResizeInput();
    }

    function autoResizeInput() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    }

    // ── Command Suggestions ─────────────────────────────────

    function showSuggestions(filter) {
        const query = filter.toLowerCase().slice(1); // remove leading /
        const matches = commands.filter(c => {
            const cmd = c.command.toLowerCase();
            return cmd.includes(query) || c.description.toLowerCase().includes(query);
        }).slice(0, 12);

        if (!matches.length) {
            hideSuggestions();
            return;
        }

        suggestionIndex = -1;
        commandSuggestions.innerHTML = matches.map((c, i) =>
            `<div class="suggestion-item" data-index="${i}" data-command="${escapeHtml(c.command.split(' ')[0])}">
                <div>
                    <span class="suggestion-cmd">${escapeHtml(c.command)}</span>
                    <span class="suggestion-cat">${escapeHtml(c.category)}</span>
                </div>
                <span class="suggestion-desc">${escapeHtml(c.description)}</span>
            </div>`
        ).join('');
        commandSuggestions.classList.remove('hidden');

        // Click handler for suggestions
        commandSuggestions.querySelectorAll('.suggestion-item').forEach(el => {
            el.addEventListener('click', () => {
                const cmd = el.dataset.command;
                messageInput.value = cmd + ' ';
                messageInput.focus();
                hideSuggestions();
            });
        });
    }

    function hideSuggestions() {
        commandSuggestions.classList.add('hidden');
        suggestionIndex = -1;
    }

    function navigateSuggestions(direction) {
        const items = commandSuggestions.querySelectorAll('.suggestion-item');
        if (!items.length) return;

        items.forEach(el => el.classList.remove('selected'));
        suggestionIndex += direction;
        if (suggestionIndex < 0) suggestionIndex = items.length - 1;
        if (suggestionIndex >= items.length) suggestionIndex = 0;

        items[suggestionIndex].classList.add('selected');
        items[suggestionIndex].scrollIntoView({ block: 'nearest' });
    }

    function selectSuggestion() {
        const items = commandSuggestions.querySelectorAll('.suggestion-item');
        if (suggestionIndex >= 0 && suggestionIndex < items.length) {
            const cmd = items[suggestionIndex].dataset.command;
            messageInput.value = cmd + ' ';
            messageInput.focus();
            hideSuggestions();
            return true;
        }
        return false;
    }

    // ── Command Palette ─────────────────────────────────────

    function showPalette() {
        commandPalette.classList.remove('hidden');
        paletteInput.value = '';
        paletteIndex = -1;
        renderPaletteResults('');
        paletteInput.focus();
    }

    function hidePalette() {
        commandPalette.classList.add('hidden');
    }

    function renderPaletteResults(filter) {
        const query = filter.toLowerCase();
        const matches = commands.filter(c =>
            c.command.toLowerCase().includes(query) ||
            c.description.toLowerCase().includes(query) ||
            c.category.toLowerCase().includes(query)
        );

        paletteIndex = -1;
        paletteResults.innerHTML = matches.map((c, i) =>
            `<div class="palette-item" data-index="${i}" data-command="${escapeHtml(c.command.split(' ')[0])}">
                <span class="palette-item-cmd">${escapeHtml(c.command)}</span>
                <span class="palette-item-desc">${escapeHtml(c.description)}</span>
            </div>`
        ).join('');

        paletteResults.querySelectorAll('.palette-item').forEach(el => {
            el.addEventListener('click', () => {
                executePaletteCommand(el.dataset.command);
            });
        });
    }

    function navigatePalette(direction) {
        const items = paletteResults.querySelectorAll('.palette-item');
        if (!items.length) return;

        items.forEach(el => el.classList.remove('selected'));
        paletteIndex += direction;
        if (paletteIndex < 0) paletteIndex = items.length - 1;
        if (paletteIndex >= items.length) paletteIndex = 0;

        items[paletteIndex].classList.add('selected');
        items[paletteIndex].scrollIntoView({ block: 'nearest' });
    }

    function selectPaletteItem() {
        const items = paletteResults.querySelectorAll('.palette-item');
        if (paletteIndex >= 0 && paletteIndex < items.length) {
            executePaletteCommand(items[paletteIndex].dataset.command);
            return true;
        }
        return false;
    }

    function executePaletteCommand(cmd) {
        hidePalette();
        // For commands that need args, put in input
        if (cmd.includes('<') || cmd.includes('[')) {
            messageInput.value = cmd.split(' ')[0] + ' ';
            messageInput.focus();
        } else {
            send({ type: 'command', command: cmd });
        }
    }

    // ── Approval Modal ──────────────────────────────────────

    function showApprovalModal(id, message) {
        currentApprovalId = id;
        approvalMessage.textContent = message;
        approvalModal.classList.remove('hidden');
    }

    function respondApproval(approved) {
        if (currentApprovalId) {
            send({
                type: 'approval_response',
                id: currentApprovalId,
                approved: approved,
            });
        }
        currentApprovalId = null;
        approvalModal.classList.add('hidden');
    }

    // ── Sidebar ─────────────────────────────────────────────

    function toggleSidebar() {
        sidebar.classList.toggle('hidden');
        if (!sidebar.classList.contains('hidden')) {
            loadSessions();
            loadInstructions(); // always refresh file list (sizes may have changed)
        }
    }

    function switchTab(tabName) {
        // Warn if leaving instructions tab with unsaved changes
        if (tabName !== 'instructions' && instructionDirty) {
            if (!confirm('You have unsaved changes in the instruction editor. Discard and switch tabs?')) return;
            markDirty(false);
        }
        $$('.sidebar-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
        $$('.sidebar-content').forEach(c => c.classList.remove('active'));
        $(`#tab${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`).classList.add('active');
    }

    // Sessions
    async function loadSessions() {
        try {
            const res = await fetch('/api/sessions');
            const sessions = await res.json();
            sessionsList.innerHTML = sessions.map(s => {
                const active = sessionInfo.id === s.id ? ' active' : '';
                const desc = s.description ? ` - ${s.description}` : '';
                return `<div class="session-item${active}" data-id="${escapeHtml(s.id)}" title="${escapeHtml(s.name)}${desc}">
                    <div>
                        <div class="session-item-name">${escapeHtml(s.name)}</div>
                        <div class="session-item-meta">${s.message_count} msgs${desc}</div>
                    </div>
                </div>`;
            }).join('');

            sessionsList.querySelectorAll('.session-item').forEach(el => {
                el.addEventListener('click', () => {
                    const id = el.dataset.id;
                    send({ type: 'command', command: `/session switch ${id}` });
                });
            });
        } catch (e) {
            sessionsList.innerHTML = '<div class="loading">Failed to load sessions</div>';
        }
    }

    // Instructions
    let currentInstructionName = null;
    let instructionDirty = false;
    const editorDirtyMark = $('#editorDirtyMark');
    const editorSaveStatus = $('#editorSaveStatus');

    function markDirty(dirty) {
        instructionDirty = dirty;
        if (dirty) {
            editorDirtyMark.classList.remove('hidden');
            editorSaveStatus.textContent = '';
        } else {
            editorDirtyMark.classList.add('hidden');
        }
    }

    function highlightActiveFile(name) {
        instructionsList.querySelectorAll('.instruction-item').forEach(el => {
            el.classList.toggle('active-file', el.dataset.name === name);
        });
    }

    async function loadInstructions() {
        try {
            const res = await fetch('/api/instructions');
            const files = await res.json();
            if (!files.length) {
                instructionsList.innerHTML = '<div class="loading">No instruction files found.</div>';
                return;
            }
            instructionsList.innerHTML = files.map(f => {
                const sizeKb = (f.size / 1024).toFixed(1);
                return `<div class="instruction-item" data-name="${escapeHtml(f.name)}" title="${escapeHtml(f.name)} — ${sizeKb} KB — click to edit">
                    <span class="instruction-item-name">${escapeHtml(f.name)}</span>
                    <span class="instruction-item-size">${sizeKb} KB</span>
                </div>`;
            }).join('');

            instructionsList.querySelectorAll('.instruction-item').forEach(el => {
                el.addEventListener('click', () => {
                    if (instructionDirty && currentInstructionName && currentInstructionName !== el.dataset.name) {
                        if (!confirm(`You have unsaved changes in "${currentInstructionName}". Discard and open "${el.dataset.name}"?`)) return;
                    }
                    openInstruction(el.dataset.name);
                });
            });

            // Re-highlight active file if one is open
            if (currentInstructionName) highlightActiveFile(currentInstructionName);
        } catch (e) {
            instructionsList.innerHTML = '<div class="loading">Failed to load instructions</div>';
        }
    }

    async function openInstruction(name) {
        try {
            const res = await fetch(`/api/instructions/${encodeURIComponent(name)}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            currentInstructionName = name;
            editorFileName.textContent = name;
            instructionContent.value = data.content;
            instructionEditor.dataset.name = name;
            instructionEditor.classList.remove('hidden');
            markDirty(false);
            highlightActiveFile(name);
            instructionContent.focus();
        } catch (e) {
            alert(`Failed to load "${name}": ${e.message}`);
        }
    }

    async function saveInstruction() {
        const name = instructionEditor.dataset.name;
        if (!name) return;
        const content = instructionContent.value;
        const btn = $('#saveInstructionBtn');
        btn.disabled = true;
        btn.textContent = 'Saving…';
        try {
            const res = await fetch(`/api/instructions/${encodeURIComponent(name)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            markDirty(false);
            editorSaveStatus.textContent = 'Saved ✓';
            setTimeout(() => { editorSaveStatus.textContent = ''; }, 2000);
            // Refresh file list to show updated size
            loadInstructions();
        } catch (e) {
            alert(`Failed to save "${name}": ${e.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Save';
        }
    }

    async function createNewInstruction() {
        const name = prompt('New instruction file name (will add .md if needed):');
        if (!name) return;
        const filename = name.endsWith('.md') ? name : name + '.md';
        // Validate: no slashes, no dots except the .md suffix
        if (/[/\\]/.test(filename) || filename.split('.').length > 2) {
            alert('Invalid file name. Use simple names like "my-instructions.md".');
            return;
        }
        try {
            const res = await fetch(`/api/instructions/${encodeURIComponent(filename)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: `# ${filename}\n\n` }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            await loadInstructions();
            openInstruction(filename);
        } catch (e) {
            alert(`Failed to create "${filename}": ${e.message}`);
        }
    }

    // ── Command Reference ───────────────────────────────────

    function buildCommandReference() {
        const ref = $('#commandReference');
        if (!ref) return;

        const categories = {};
        commands.forEach(c => {
            if (!categories[c.category]) categories[c.category] = [];
            categories[c.category].push(c);
        });

        let html = '<h3>Command Reference</h3>';
        Object.entries(categories).forEach(([cat, cmds]) => {
            html += `<h3>${escapeHtml(cat)}</h3><table class="help-table">`;
            cmds.forEach(c => {
                html += `<tr>
                    <td><code>${escapeHtml(c.command)}</code></td>
                    <td>${escapeHtml(c.description)}</td>
                </tr>`;
            });
            html += '</table>';
        });
        ref.innerHTML = html;
    }

    // ── Markdown Rendering (simple) ─────────────────────────

    function renderMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);

        // Code blocks (``` ... ```)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
            `<pre><code>${code}</code></pre>`
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

        // Paragraphs (double newlines)
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        // Wrap in paragraph if not already
        if (!html.startsWith('<')) html = '<p>' + html + '</p>';

        return html;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Resize Handle ───────────────────────────────────────

    function initResize() {
        const handle = $('#resizeHandle');
        const chatPane = $('#chatPane');
        const monitorPane = $('#monitorPane');
        let startX, startChatWidth;

        handle.addEventListener('mousedown', (e) => {
            startX = e.clientX;
            startChatWidth = chatPane.offsetWidth;
            handle.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            const onMove = (e) => {
                const diff = e.clientX - startX;
                const totalWidth = chatPane.parentElement.offsetWidth - handle.offsetWidth;
                const newChatWidth = Math.max(300, Math.min(totalWidth - 250, startChatWidth + diff));
                const chatFlex = newChatWidth / totalWidth * 10;
                const monitorFlex = (totalWidth - newChatWidth) / totalWidth * 10;
                chatPane.style.flex = chatFlex;
                monitorPane.style.flex = monitorFlex;
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

    // ── Event Listeners ─────────────────────────────────────

    function initEvents() {
        // Send button
        sendBtn.addEventListener('click', handleSend);

        // Input events
        messageInput.addEventListener('keydown', (e) => {
            // Enter to send (without shift)
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                // If suggestions are visible and an item is selected
                if (!commandSuggestions.classList.contains('hidden')) {
                    if (selectSuggestion()) return;
                }
                handleSend();
                return;
            }

            // Arrow keys for suggestions
            if (!commandSuggestions.classList.contains('hidden')) {
                if (e.key === 'ArrowDown') { e.preventDefault(); navigateSuggestions(1); return; }
                if (e.key === 'ArrowUp') { e.preventDefault(); navigateSuggestions(-1); return; }
                if (e.key === 'Tab') { e.preventDefault(); selectSuggestion(); return; }
            }

            // Escape to hide suggestions/palette
            if (e.key === 'Escape') {
                if (!commandSuggestions.classList.contains('hidden')) {
                    hideSuggestions();
                    return;
                }
            }
        });

        messageInput.addEventListener('input', () => {
            autoResizeInput();
            const val = messageInput.value;
            if (val.startsWith('/') && !val.includes('\n')) {
                showSuggestions(val);
            } else {
                hideSuggestions();
            }
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+K - command palette
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                if (commandPalette.classList.contains('hidden')) {
                    showPalette();
                } else {
                    hidePalette();
                }
                return;
            }

            // Ctrl+N - new session
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                send({ type: 'command', command: '/new' });
                return;
            }

            // Ctrl+B - toggle sidebar
            if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
                e.preventDefault();
                toggleSidebar();
                return;
            }

            // Escape - close modals/palette
            if (e.key === 'Escape') {
                if (!approvalModal.classList.contains('hidden')) return; // Don't close approval modal with Esc
                if (!commandPalette.classList.contains('hidden')) { hidePalette(); return; }
                if (!sidebar.classList.contains('hidden')) { sidebar.classList.add('hidden'); return; }
            }
        });

        // Palette events
        paletteInput.addEventListener('input', () => {
            renderPaletteResults(paletteInput.value);
        });
        paletteInput.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown') { e.preventDefault(); navigatePalette(1); }
            if (e.key === 'ArrowUp') { e.preventDefault(); navigatePalette(-1); }
            if (e.key === 'Enter') { e.preventDefault(); selectPaletteItem(); }
            if (e.key === 'Escape') { hidePalette(); }
        });
        commandPalette.addEventListener('click', (e) => {
            if (e.target === commandPalette) hidePalette();
        });

        // Sidebar
        $('#sidebarToggle').addEventListener('click', toggleSidebar);
        $('#sidebarClose').addEventListener('click', () => sidebar.classList.add('hidden'));
        $$('.sidebar-tab').forEach(tab => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        // Session badge opens sidebar to sessions tab
        $('#sessionBadge').addEventListener('click', () => {
            sidebar.classList.remove('hidden');
            switchTab('sessions');
            loadSessions();
        });

        // New session button
        $('#newSessionBtn').addEventListener('click', () => {
            const name = prompt('Session name (optional):');
            if (name !== null) {
                send({ type: 'command', command: name ? `/new ${name}` : '/new' });
                setTimeout(loadSessions, 500);
            }
        });

        // Palette button
        $('#paletteBtn').addEventListener('click', showPalette);

        // Help button
        $('#helpBtn').addEventListener('click', () => {
            sidebar.classList.remove('hidden');
            switchTab('help');
        });

        // Quick action buttons
        $$('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                send({ type: 'command', command: btn.dataset.command });
            });
        });

        // Monitor clear
        $('#clearMonitor').addEventListener('click', () => {
            monitorMessages.innerHTML = '';
            const empty = document.createElement('div');
            empty.className = 'empty-state';
            empty.id = 'monitorEmpty';
            empty.innerHTML = '<div class="empty-icon">&#x1F4CA;</div><div class="empty-text">Tool outputs and traces will appear here</div>';
            monitorMessages.appendChild(empty);
        });

        // Approval buttons
        $('#approveBtn').addEventListener('click', () => respondApproval(true));
        $('#denyBtn').addEventListener('click', () => respondApproval(false));

        // Instruction editor
        $('#saveInstructionBtn').addEventListener('click', saveInstruction);
        $('#closeEditorBtn').addEventListener('click', () => {
            if (instructionDirty) {
                if (!confirm('You have unsaved changes. Discard and close?')) return;
            }
            instructionEditor.classList.add('hidden');
            currentInstructionName = null;
            markDirty(false);
            highlightActiveFile(null);
        });
        $('#newInstructionBtn').addEventListener('click', createNewInstruction);

        // Mark dirty on content change
        instructionContent.addEventListener('input', () => {
            if (!instructionDirty) markDirty(true);
        });

        // Tab support + Ctrl+S in instruction editor textarea
        instructionContent.addEventListener('keydown', (e) => {
            // Ctrl+S / Cmd+S to save
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                saveInstruction();
                return;
            }
            // Tab inserts 4 spaces
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = instructionContent.selectionStart;
                const end = instructionContent.selectionEnd;
                instructionContent.value =
                    instructionContent.value.substring(0, start) +
                    '    ' +
                    instructionContent.value.substring(end);
                instructionContent.selectionStart = instructionContent.selectionEnd = start + 4;
            }
        });

        // Resize handle
        initResize();
    }

    // ── Init ────────────────────────────────────────────────

    function init() {
        initEvents();
        connect();
        messageInput.focus();
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
