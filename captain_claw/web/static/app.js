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
    let allowedModels = [];
    let activeModelLabel = '';

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
    const stopBtn = $('#stopBtn');
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
                addChatMessage(data.role, data.content, data.replay, data.timestamp, data.model);
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
            case 'thinking':
                updateThinkingIndicator(data.text, data.tool, data.phase);
                break;
            case 'error':
                addChatMessage('error', data.message);
                break;
        }
    }

    function handleWelcome(data) {
        commands = data.commands || [];
        allowedModels = data.models || [];
        // Set a fallback active model label from the models list.
        if (allowedModels.length && !activeModelLabel) {
            const active = allowedModels.find(m => m.id === 'default') || allowedModels[0];
            activeModelLabel = `${active.provider}:${active.model}`;
            modelName.textContent = activeModelLabel;
        }
        // Session info may override the active model (e.g. per-session selection).
        if (data.session) {
            updateSessionInfo(data.session);
        }
        buildModelDropdown();
        buildCommandReference();
    }

    // ── Chat ────────────────────────────────────────────────

    let msgCount = 0;

    function formatTimestamp(isoStr) {
        if (!isoStr) return '';
        try {
            const d = new Date(isoStr);
            if (isNaN(d.getTime())) return '';
            const now = new Date();
            const isToday = d.toDateString() === now.toDateString();
            const yesterday = new Date(now);
            yesterday.setDate(yesterday.getDate() - 1);
            const isYesterday = d.toDateString() === yesterday.toDateString();

            const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            if (isToday) return time;
            if (isYesterday) return `Yesterday ${time}`;
            const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
            return `${date} ${time}`;
        } catch {
            return '';
        }
    }

    function addChatMessage(role, content, isReplay, timestamp, model) {
        // Remove thinking indicator(s) when assistant or error message arrives.
        if (role === 'assistant' || role === 'error') {
            removeThinkingIndicator();
            // Also remove all frozen micro-loop steps.
            chatMessages.querySelectorAll('.thinking-frozen').forEach(el => el.remove());
            _thinkingTool = '';
        }
        chatEmpty.style.display = 'none';

        // Rephrase messages render as a persistent collapsible panel.
        if (role === 'rephrase') {
            addRephrasePanel(content, isReplay);
            return;
        }

        // Image messages render as inline images.
        if (role === 'image') {
            addImagePanel(content, isReplay);
            return;
        }

        const div = document.createElement('div');
        div.className = `msg ${role}`;
        if (isReplay) div.style.animation = 'none';

        // Header row: label + meta (timestamp, model)
        const header = document.createElement('div');
        header.className = 'msg-header';

        const label = document.createElement('span');
        label.className = 'msg-label';
        if (role === 'user') {
            label.textContent = 'You';
        } else if (role === 'error') {
            label.textContent = 'Error';
        } else {
            label.innerHTML = '<span class="msg-avatar">&#x1F980;</span> Captain Claw';
        }

        const meta = document.createElement('span');
        meta.className = 'msg-meta';
        const timeStr = formatTimestamp(timestamp);
        const parts = [];
        if (timeStr) parts.push(timeStr);
        if (model && role === 'assistant') parts.push(model);
        meta.textContent = parts.join(' \u00b7 ');

        header.appendChild(label);
        header.appendChild(meta);

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = renderMarkdown(content);

        div.appendChild(header);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        msgCount++;
        chatCount.textContent = `${msgCount} messages`;
    }

    function addRephrasePanel(content, isReplay) {
        const panel = document.createElement('div');
        panel.className = 'rephrase-panel';
        if (isReplay) panel.style.animation = 'none';

        const header = document.createElement('div');
        header.className = 'rephrase-header';
        header.innerHTML = '<span class="rephrase-icon">\u2728</span>'
            + '<span class="rephrase-title">Task Rephrased</span>'
            + '<span class="rephrase-toggle">\u25BC</span>';

        const body = document.createElement('div');
        body.className = 'rephrase-body';
        body.innerHTML = renderMarkdown(content);

        // Start collapsed — user can click to expand.
        body.style.display = 'none';
        header.querySelector('.rephrase-toggle').textContent = '\u25B6';

        header.addEventListener('click', function() {
            const toggle = header.querySelector('.rephrase-toggle');
            if (body.style.display === 'none') {
                body.style.display = '';
                toggle.textContent = '\u25BC';
            } else {
                body.style.display = 'none';
                toggle.textContent = '\u25B6';
            }
        });

        panel.appendChild(header);
        panel.appendChild(body);
        chatMessages.appendChild(panel);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addImagePanel(imagePath, isReplay) {
        chatEmpty.style.display = 'none';
        const div = document.createElement('div');
        div.className = 'msg assistant';
        if (isReplay) div.style.animation = 'none';

        const header = document.createElement('div');
        header.className = 'msg-header';
        const label = document.createElement('span');
        label.className = 'msg-label';
        label.innerHTML = '<span class="msg-avatar">&#x1F980;</span> Captain Claw';
        header.appendChild(label);

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        const img = document.createElement('img');
        img.src = '/api/media?path=' + encodeURIComponent(imagePath);
        img.alt = 'Generated image';
        img.style.cssText = 'max-width:100%;border-radius:8px;cursor:pointer;';
        img.addEventListener('click', function() {
            window.open(img.src, '_blank');
        });
        img.onerror = function() {
            const fallback = document.createElement('div');
            fallback.style.cssText = 'padding:12px;color:#aaa;font-size:13px;';
            fallback.textContent = 'Image: ' + imagePath;
            bubble.replaceChild(fallback, img);
        };
        bubble.appendChild(img);

        div.appendChild(header);
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

    // ── Thinking Indicator ─────────────────────────────────

    function removeThinkingIndicator() {
        const el = document.getElementById('thinkingIndicator');
        if (el) el.remove();
    }

    // Track whether the current live indicator belongs to a scale_micro_loop
    // so we can freeze (persist) it when the next step arrives.
    let _thinkingTool = '';

    function _freezeThinkingIndicator() {
        // Convert the live indicator into a static (frozen) element so it
        // stays visible while the next step takes over.
        const el = document.getElementById('thinkingIndicator');
        if (!el) return;
        el.removeAttribute('id');
        el.classList.add('thinking-frozen');
        // Stop the pulsing animation on the icon.
        const icon = el.querySelector('.thinking-icon');
        if (icon) icon.style.animation = 'none';
    }

    function _buildThinkingHtml(text, tool) {
        // Multi-line support: first line = tool/action (purple),
        // subsequent lines = detail (normal text, no truncation).
        const lines = text.split('\n');
        const firstLine = lines[0] || text;
        const detailLines = lines.slice(1).filter(l => l.length > 0);

        // Parse the first line: prefix before ':' is the tool label.
        const prefix = firstLine.split(':')[0] || firstLine;
        const rest = firstLine.includes(':')
            ? firstLine.slice(firstLine.indexOf(':') + 1).trim()
            : '';

        let html = '<span class="thinking-tool">' + escapeHtml(prefix) + '</span>';
        if (rest) html += ' ' + escapeHtml(rest);
        if (detailLines.length > 0) {
            html += '<span class="thinking-detail">'
                + detailLines.map(l => escapeHtml(l)).join('<br>')
                + '</span>';
        }
        return html;
    }

    function updateThinkingIndicator(text, tool, phase) {
        // Clear on "done" phase or empty text.
        if (phase === 'done' || !text) {
            removeThinkingIndicator();
            _thinkingTool = '';
            return;
        }

        // Hide empty state so indicator is visible.
        chatEmpty.style.display = 'none';

        // For scale_micro_loop: freeze the previous step so it stays
        // visible, then create a fresh indicator for the new step.
        if (tool === 'scale_micro_loop' && _thinkingTool === 'scale_micro_loop') {
            _freezeThinkingIndicator();
        }
        // If switching away from scale_micro_loop to another tool,
        // just remove the live indicator (frozen ones stay).
        if (tool !== 'scale_micro_loop' && _thinkingTool === 'scale_micro_loop') {
            _freezeThinkingIndicator();
        }
        _thinkingTool = tool || '';

        let el = document.getElementById('thinkingIndicator');
        if (!el) {
            el = document.createElement('div');
            el.id = 'thinkingIndicator';
            el.className = 'thinking-indicator';
            const inner = document.createElement('div');
            inner.className = 'thinking-inner';
            const icon = document.createElement('span');
            icon.className = 'thinking-icon';
            icon.textContent = '\u2728';  // sparkles emoji
            const textSpan = document.createElement('span');
            textSpan.className = 'thinking-text';
            inner.appendChild(icon);
            inner.appendChild(textSpan);
            el.appendChild(inner);
            chatMessages.appendChild(el);
        }

        // Update the text content.
        const textSpan = el.querySelector('.thinking-text');
        if (phase === 'reasoning') {
            textSpan.innerHTML = text;
        } else if (tool) {
            textSpan.innerHTML = _buildThinkingHtml(text, tool);
        } else {
            textSpan.textContent = text;
        }

        // Keep it at the bottom and scroll.
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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

    let _agentBusy = false;

    function setStatus(dotClass, text) {
        statusDot.className = `status-dot ${dotClass}`;
        statusText.textContent = text;

        // Toggle send/stop button visibility based on agent state.
        const busy = (text === 'thinking' || text === 'streaming');
        if (busy !== _agentBusy) {
            _agentBusy = busy;
            sendBtn.style.display = busy ? 'none' : '';
            stopBtn.style.display = busy ? '' : 'none';
        }
    }

    function updateSessionInfo(info) {
        sessionInfo = info;
        if (info.name) sessionName.textContent = info.name;
        if (info.model) {
            const provider = info.provider || '';
            activeModelLabel = provider ? `${provider}:${info.model}` : info.model;
            modelName.textContent = activeModelLabel;
            highlightActiveModelInDropdown();
        }
    }

    // ── Model Selector ────────────────────────────────────────

    const modelDropdown = $('#modelDropdown');
    const modelBadge = $('#modelBadge');

    function buildModelDropdown() {
        if (!allowedModels.length) {
            modelDropdown.innerHTML = '<div class="model-dropdown-empty">No models configured</div>';
            return;
        }
        modelDropdown.innerHTML = allowedModels.map((m, i) => {
            const label = `${m.provider}:${m.model}`;
            const desc = m.description || '';
            const isActive = label === activeModelLabel;
            return `<div class="model-option${isActive ? ' active' : ''}" data-selector="${escapeHtml(m.id || String(i + 1))}" data-label="${escapeHtml(label)}" title="${escapeHtml(desc || label)}">
                <span class="model-option-name">${escapeHtml(label)}</span>
                ${desc ? `<span class="model-option-desc">${escapeHtml(desc)}</span>` : ''}
                ${isActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
            </div>`;
        }).join('');

        modelDropdown.querySelectorAll('.model-option').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                const selector = el.dataset.selector;
                send({ type: 'set_model', selector: selector });
                activeModelLabel = el.dataset.label;
                modelName.textContent = activeModelLabel;
                highlightActiveModelInDropdown();
                hideModelDropdown();
            });
        });
    }

    function highlightActiveModelInDropdown() {
        modelDropdown.querySelectorAll('.model-option').forEach(el => {
            const isActive = el.dataset.label === activeModelLabel;
            el.classList.toggle('active', isActive);
            const check = el.querySelector('.model-option-check');
            if (isActive && !check) {
                const span = document.createElement('span');
                span.className = 'model-option-check';
                span.innerHTML = '&#x2713;';
                el.appendChild(span);
            } else if (!isActive && check) {
                check.remove();
            }
        });
    }

    function toggleModelDropdown() {
        modelDropdown.classList.toggle('hidden');
    }

    function hideModelDropdown() {
        modelDropdown.classList.add('hidden');
    }

    // ── File Upload ──────────────────────────────────────────

    const fileInput = $('#fileInput');
    const uploadBtn = $('#uploadBtn');
    const dropOverlay = $('#dropOverlay');
    const chatPane = $('#chatPane');
    const attachmentPreview = $('#attachmentPreview');

    var pendingImagePath = null;
    var _IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'];

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function showAttachmentChip(filename, path, objectUrl) {
        pendingImagePath = path;
        attachmentPreview.innerHTML = '';
        var chip = document.createElement('div');
        chip.className = 'attachment-chip';
        if (objectUrl) {
            var thumb = document.createElement('img');
            thumb.src = objectUrl;
            thumb.className = 'attachment-thumb';
            chip.appendChild(thumb);
        }
        var name = document.createElement('span');
        name.className = 'attachment-name';
        name.textContent = filename;
        chip.appendChild(name);
        var removeBtn = document.createElement('button');
        removeBtn.className = 'attachment-remove';
        removeBtn.textContent = '\u2715';
        removeBtn.title = 'Remove attachment';
        removeBtn.addEventListener('click', function() {
            clearAttachment();
        });
        chip.appendChild(removeBtn);
        attachmentPreview.appendChild(chip);
        attachmentPreview.classList.remove('hidden');
    }

    function clearAttachment() {
        pendingImagePath = null;
        attachmentPreview.innerHTML = '';
        attachmentPreview.classList.add('hidden');
        fileInput.value = '';
    }

    async function uploadFile(file) {
        if (!file) return;
        var ext = file.name.split('.').pop().toLowerCase();
        var isImage = _IMAGE_EXTS.indexOf(ext) !== -1;
        var isData = (ext === 'csv' || ext === 'xlsx');

        if (!isImage && !isData) {
            addChatMessage('assistant', '**Error:** Supported files: images (.png, .jpg, .jpeg, .webp, .gif, .bmp) and data (.csv, .xlsx).');
            return;
        }

        if (isImage) {
            // Upload image and stage as attachment (don't auto-submit).
            uploadBtn.disabled = true;
            uploadBtn.textContent = '\u23F3';

            var formData = new FormData();
            formData.append('file', file);

            try {
                var res = await fetch('/api/image/upload', {
                    method: 'POST',
                    body: formData,
                });
                var data = await res.json();
                if (!res.ok) {
                    addChatMessage('assistant', '**Upload failed:** ' + (data.error || 'Unknown error'));
                } else {
                    var objectUrl = URL.createObjectURL(file);
                    showAttachmentChip(file.name, data.path, objectUrl);
                    messageInput.focus();
                }
            } catch (err) {
                addChatMessage('assistant', '**Upload failed:** ' + err.message);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = '\uD83D\uDCCE';
                fileInput.value = '';
            }
            return;
        }

        // Data file (csv/xlsx) — existing datastore import flow.
        addChatMessage('user', 'Uploading file: **' + file.name + '** (' + formatFileSize(file.size) + ')');
        uploadBtn.disabled = true;
        uploadBtn.textContent = '\u23F3';

        var formData = new FormData();
        formData.append('file', file);

        try {
            var res = await fetch('/api/datastore/upload', {
                method: 'POST',
                body: formData,
            });
            var data = await res.json();
            if (!res.ok) {
                addChatMessage('assistant', '**Import failed:** ' + (data.error || 'Unknown error'));
            }
        } catch (err) {
            addChatMessage('assistant', '**Upload failed:** ' + err.message);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '\uD83D\uDCCE';
            fileInput.value = '';
        }
    }

    // ── Input Handling ──────────────────────────────────────

    function handleSend() {
        const text = messageInput.value.trim();
        if (!text && !pendingImagePath) return;
        if (!isConnected) return;

        hideSuggestions();

        if (text.startsWith('/')) {
            send({ type: 'command', command: text });
        } else {
            var msg = { type: 'chat', content: text || '' };
            if (pendingImagePath) {
                msg.image_path = pendingImagePath;
            }
            send(msg);
        }

        messageInput.value = '';
        clearAttachment();
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
                const customized = f.overridden ? ' <span class="instruction-item-customized">customized</span>' : '';
                return `<div class="instruction-item${f.overridden ? ' overridden' : ''}" data-name="${escapeHtml(f.name)}" title="${escapeHtml(f.name)} — ${sizeKb} KB${f.overridden ? ' — customized' : ''} — click to edit">
                    <span class="instruction-item-name">${escapeHtml(f.name)}${customized}</span>
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

        // Images: ![alt](path) — route saved/ and output/ paths through /api/media
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_, alt, src) {
            var url = src;
            if (/^saved\/|^output\//.test(src)) {
                url = '/api/media?path=' + encodeURIComponent(src);
            }
            return '<img src="' + url + '" alt="' + alt +
                '" style="max-width:100%;border-radius:8px;cursor:pointer;" ' +
                'onclick="window.open(this.src,\'_blank\')">';
        });

        // Links: [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

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

    function handleStop() {
        if (!isConnected) return;
        send({ type: 'cancel' });
        // Provide immediate visual feedback.
        stopBtn.classList.add('stop-btn-active');
        setTimeout(() => stopBtn.classList.remove('stop-btn-active'), 600);
    }

    function initEvents() {
        // Send button
        sendBtn.addEventListener('click', handleSend);
        // Stop button — cancels current processing
        stopBtn.addEventListener('click', handleStop);

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

            // Escape to hide suggestions/palette, or stop processing
            if (e.key === 'Escape') {
                if (!commandSuggestions.classList.contains('hidden')) {
                    hideSuggestions();
                    return;
                }
                if (_agentBusy) {
                    handleStop();
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
                if (!modelDropdown.classList.contains('hidden')) { hideModelDropdown(); return; }
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

        // Model badge toggles model selector dropdown
        modelBadge.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleModelDropdown();
        });
        // Close model dropdown on outside click
        document.addEventListener('click', (e) => {
            if (!modelDropdown.contains(e.target) && e.target !== modelBadge) {
                hideModelDropdown();
            }
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

        // File upload button
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                uploadFile(fileInput.files[0]);
            }
        });

        // Drag and drop on chat pane
        var dragCounter = 0;
        chatPane.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            dropOverlay.classList.remove('hidden');
        });
        chatPane.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        chatPane.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                dropOverlay.classList.add('hidden');
            }
        });
        chatPane.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            dropOverlay.classList.add('hidden');
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

        // Paste image from clipboard into chat input
        messageInput.addEventListener('paste', (e) => {
            var items = e.clipboardData && e.clipboardData.items;
            if (!items) return;
            for (var i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image/') === 0) {
                    e.preventDefault();
                    var file = items[i].getAsFile();
                    if (file) {
                        // Give pasted images a sensible name if missing.
                        if (!file.name || file.name === 'image.png') {
                            var ext = file.type.split('/')[1] || 'png';
                            if (ext === 'jpeg') ext = 'jpg';
                            var ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                            file = new File([file], 'pasted-' + ts + '.' + ext, { type: file.type });
                        }
                        uploadFile(file);
                    }
                    return;
                }
            }
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
