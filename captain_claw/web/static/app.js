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
    let personalities = [];
    let activePersonalityId = '';
    let activePersonalityName = 'No profile';
    let playbooks = [];
    let activePlaybookId = '';   // '' = auto, '__none__' = disabled, or a playbook ID
    let activePlaybookName = 'Auto';
    let forceScriptMode = false;

    // ── Voice recording state ─────────────────────────────
    let micState = 'idle'; // 'idle' | 'recording' | 'transcribing'
    let sttWs = null;
    let sttAudioCtx = null;
    let sttSource = null;
    let sttProcessor = null;
    let sttStream = null;
    let sttIsRealtime = false;
    let sttPreText = '';
    const MAX_RECORDING_SECONDS = 30;

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
    const micBtn = $('#micBtn');
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
                addChatMessage(data.role, data.content, data.replay, data.timestamp, data.model, data.feedback);
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
                updateContextInfo(data);
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
            case 'tool_output_inline':
                updateThinkingWithOutput(data.tool, data.summary, data.output);
                break;
            case 'tool_stream':
                appendToolStreamChunk(data.chunk);
                break;
            case 'next_steps':
                renderNextSteps(data.options || []);
                break;
            case 'error':
                addChatMessage('error', data.message);
                break;
        }
    }

    function handleWelcome(data) {
        commands = data.commands || [];
        allowedModels = data.models || [];
        personalities = data.personalities || [];
        playbooks = data.playbooks || [];
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
        buildPersonaDropdown();
        buildPlaybookDropdown();
        buildCommandReference();
        // Sync force-script mode from sessionStorage to backend on connect.
        if (forceScriptMode) {
            send({ type: 'set_force_script', enabled: true });
        }
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

    function addChatMessage(role, content, isReplay, timestamp, model, feedback) {
        // Remove stale next-steps buttons when new messages arrive.
        if (role === 'user' || role === 'assistant' || role === 'error') {
            removeNextSteps();
        }
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

        // Audio messages render as inline audio players.
        if (role === 'audio') {
            addAudioPanel(content, isReplay);
            return;
        }

        // HTML file messages render as inline view cards.
        if (role === 'html_file') {
            addHtmlFilePanel(content, isReplay);
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

        // Copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'msg-copy-btn';
        copyBtn.title = 'Copy to clipboard';
        copyBtn.innerHTML = '&#x1F4CB;';
        copyBtn.addEventListener('click', function () {
            var text = bubble.innerText || bubble.textContent || '';
            navigator.clipboard.writeText(text).then(function () {
                copyBtn.innerHTML = '&#x2705;';
                setTimeout(function () { copyBtn.innerHTML = '&#x1F4CB;'; }, 1500);
            });
        });

        div.appendChild(header);
        div.appendChild(bubble);
        div.appendChild(copyBtn);

        // Feedback buttons (like/dislike) for assistant messages
        if (role === 'assistant' && timestamp) {
            var fbRow = document.createElement('div');
            fbRow.className = 'msg-feedback';

            var likeBtn = document.createElement('button');
            likeBtn.className = 'msg-fb-btn' + (feedback === 'good' ? ' active' : '');
            likeBtn.title = 'Good response';
            likeBtn.innerHTML = '&#x1F44D;';

            var dislikeBtn = document.createElement('button');
            dislikeBtn.className = 'msg-fb-btn' + (feedback === 'bad' ? ' active' : '');
            dislikeBtn.title = 'Bad response';
            dislikeBtn.innerHTML = '&#x1F44E;';

            var fbSaved = document.createElement('span');
            fbSaved.className = 'msg-fb-saved';
            fbSaved.textContent = 'Saved';

            function sendFeedback(value) {
                // Toggle: clicking the same button clears feedback
                var current = fbRow.dataset.feedback || '';
                var newValue = current === value ? '' : value;
                fbRow.dataset.feedback = newValue;
                likeBtn.className = 'msg-fb-btn' + (newValue === 'good' ? ' active' : '');
                dislikeBtn.className = 'msg-fb-btn' + (newValue === 'bad' ? ' active' : '');
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'message_feedback',
                        timestamp: timestamp,
                        feedback: newValue || null,
                    }));
                }
                // Flash "Saved" acknowledgment
                fbSaved.classList.remove('show');
                void fbSaved.offsetWidth; // reflow to restart animation
                fbSaved.classList.add('show');
            }

            likeBtn.addEventListener('click', function () { sendFeedback('good'); });
            dislikeBtn.addEventListener('click', function () { sendFeedback('bad'); });

            // Brain Graph button
            var graphBtn = document.createElement('button');
            graphBtn.className = 'msg-fb-btn';
            graphBtn.title = 'View in Brain Graph';
            graphBtn.innerHTML = '&#x1F9E0;';
            graphBtn.addEventListener('click', function () {
                var params = '?focus_ts=' + encodeURIComponent(timestamp || '');
                window.open('/brain-graph' + params, '_blank');
            });

            if (feedback) fbRow.dataset.feedback = feedback;
            fbRow.appendChild(likeBtn);
            fbRow.appendChild(dislikeBtn);
            fbRow.appendChild(graphBtn);
            fbRow.appendChild(fbSaved);
            div.appendChild(fbRow);
        }

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

    function addAudioPanel(audioPath, isReplay) {
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

        const audio = document.createElement('audio');
        audio.controls = true;
        audio.style.cssText = 'display:block;width:100%;border-radius:8px;';
        audio.src = '/api/media?path=' + encodeURIComponent(audioPath);
        audio.onerror = function() {
            const fallback = document.createElement('div');
            fallback.style.cssText = 'padding:12px;color:#aaa;font-size:13px;';
            fallback.textContent = 'Audio: ' + audioPath;
            bubble.replaceChild(fallback, audio);
        };
        bubble.appendChild(audio);

        div.appendChild(header);
        div.appendChild(bubble);
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        msgCount++;
        chatCount.textContent = `${msgCount} messages`;
    }

    function addHtmlFilePanel(filePath, isReplay) {
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

        const card = document.createElement('div');
        card.style.cssText = 'display:flex;align-items:center;gap:10px;padding:8px 0;';

        const icon = document.createElement('span');
        icon.style.cssText = 'font-size:24px;';
        icon.textContent = '\u{1F310}';
        card.appendChild(icon);

        const info = document.createElement('div');
        info.style.cssText = 'flex:1;min-width:0;';
        const fname = document.createElement('div');
        fname.style.cssText = 'font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
        fname.textContent = filePath.split('/').pop();
        info.appendChild(fname);
        const fpath = document.createElement('div');
        fpath.style.cssText = 'font-size:11px;opacity:0.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
        fpath.textContent = filePath;
        info.appendChild(fpath);
        card.appendChild(info);

        const viewBtn = document.createElement('a');
        viewBtn.href = '/api/files/view?path=' + encodeURIComponent(filePath);
        viewBtn.target = '_blank';
        viewBtn.textContent = '\u{1F441} View';
        viewBtn.style.cssText = 'padding:5px 14px;background:var(--accent,#3b82f6);color:#fff;border-radius:6px;font-size:12px;font-weight:500;text-decoration:none;white-space:nowrap;';
        card.appendChild(viewBtn);

        bubble.appendChild(card);
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
        const ctxEl = document.getElementById('contextInfo');
        if (ctxEl) ctxEl.innerHTML = '';
    }

    // ── Context Token Info ────────────────────────────────────

    function updateContextInfo(data) {
        const el = document.getElementById('contextInfo');
        if (!el) return;
        const ctx = data.context_window || {};
        const last = data.last || {};
        const total = data.total || {};
        const budget = ctx.context_budget_tokens || 0;
        const prompt = ctx.prompt_tokens || 0;
        const util = ctx.utilization || 0;
        const completion = last.completion_tokens || 0;
        const cacheRead = last.cache_read_input_tokens || 0;
        const totalTokens = total.total_tokens || 0;
        if (!budget && !totalTokens) {
            el.innerHTML = '';
            return;
        }
        const fmtK = (n) => {
            if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
            return String(n);
        };
        const pct = Math.round(util * 100);
        const barClass = pct >= 90 ? 'critical' : pct >= 70 ? 'warn' : '';
        let parts = [];
        if (budget) {
            parts.push(
                `<span class="ctx-item">`
                + `<span class="ctx-label">ctx</span>`
                + `<span class="ctx-bar"><span class="ctx-bar-fill ${barClass}" style="width:${Math.min(pct, 100)}%"></span></span>`
                + `<span class="ctx-value">${fmtK(prompt)}/${fmtK(budget)} (${pct}%)</span>`
                + `</span>`
            );
        }
        if (completion) {
            parts.push(
                `<span class="ctx-item">`
                + `<span class="ctx-label">out</span>`
                + `<span class="ctx-value">${fmtK(completion)}</span>`
                + `</span>`
            );
        }
        if (cacheRead) {
            parts.push(
                `<span class="ctx-item">`
                + `<span class="ctx-label">cache</span>`
                + `<span class="ctx-value">${fmtK(cacheRead)}</span>`
                + `</span>`
            );
        }
        if (totalTokens) {
            parts.push(
                `<span class="ctx-item">`
                + `<span class="ctx-label">session</span>`
                + `<span class="ctx-value">${fmtK(totalTokens)}</span>`
                + `</span>`
            );
        }
        el.innerHTML = parts.join('');
    }

    // ── Next Steps ──────────────────────────────────────────

    function removeNextSteps() {
        chatMessages.querySelectorAll('.next-steps-container').forEach(el => el.remove());
    }

    function renderNextSteps(options) {
        if (!options || !options.length) return;
        removeNextSteps();

        const container = document.createElement('div');
        container.className = 'next-steps-container';

        const label = document.createElement('div');
        label.className = 'next-steps-label';
        label.textContent = 'Suggested next steps';
        container.appendChild(label);

        const row = document.createElement('div');
        row.className = 'next-steps-row';

        options.forEach(function(opt) {
            const btn = document.createElement('button');
            btn.className = 'next-step-btn';
            btn.title = opt.description || opt.action;
            btn.textContent = opt.label;
            btn.addEventListener('click', function() {
                removeNextSteps();
                send({ type: 'chat', content: opt.action });
            });
            row.appendChild(btn);
        });

        container.appendChild(row);
        chatMessages.appendChild(container);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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
        // Auto-collapse console output in frozen state.
        const consoleEl = el.querySelector('.thinking-console');
        if (consoleEl) consoleEl.classList.add('collapsed');
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

        // Freeze any previous indicator that has a console (completed tool output).
        if (phase === 'tool') {
            const existing = document.getElementById('thinkingIndicator');
            if (existing && existing.querySelector('.thinking-console')) {
                _freezeThinkingIndicator();
            }
        }

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

    function updateThinkingWithOutput(tool, summary, output) {
        // Tool has completed — update the thinking indicator to show its
        // output in an inline console block beneath the summary line.
        chatEmpty.style.display = 'none';

        // Freeze any previous indicator that already has console output
        // (a previous tool in the same turn).
        const existing = document.getElementById('thinkingIndicator');
        if (existing && existing.querySelector('.thinking-console')) {
            _freezeThinkingIndicator();
        }

        // Create or reuse the live indicator.
        let el = document.getElementById('thinkingIndicator');
        if (!el) {
            el = document.createElement('div');
            el.id = 'thinkingIndicator';
            el.className = 'thinking-indicator';
            const inner = document.createElement('div');
            inner.className = 'thinking-inner';
            const icon = document.createElement('span');
            icon.className = 'thinking-icon';
            icon.textContent = '\u2705';  // checkmark — tool is done
            const textSpan = document.createElement('span');
            textSpan.className = 'thinking-text';
            inner.appendChild(icon);
            inner.appendChild(textSpan);
            el.appendChild(inner);
            chatMessages.appendChild(el);
        }

        // Update the summary line.
        const textSpan = el.querySelector('.thinking-text');
        textSpan.innerHTML = _buildThinkingHtml(summary, tool);

        // Mark icon as done (stop pulsing, show checkmark).
        const icon = el.querySelector('.thinking-icon');
        if (icon) {
            icon.textContent = '\u2705';
            icon.style.animation = 'none';
        }

        // Add or update the console output block.
        // If a console already exists (from live tool_stream chunks), keep
        // its streamed content — don't overwrite with the final output.
        let consoleEl = el.querySelector('.thinking-console');
        if (!consoleEl) {
            // No live stream was received — create the console with final output.
            consoleEl = document.createElement('div');
            consoleEl.className = 'thinking-console';

            // Clickable header to toggle expand/collapse.
            const header = document.createElement('div');
            header.className = 'thinking-console-header';
            header.innerHTML = '<span class="thinking-console-chevron">\u25BE</span> Output';
            header.addEventListener('click', function() {
                consoleEl.classList.toggle('collapsed');
            });
            consoleEl.appendChild(header);

            // Output pre block.
            const pre = document.createElement('pre');
            pre.className = 'thinking-console-output';
            pre.textContent = output || '(no output)';
            consoleEl.appendChild(pre);

            el.appendChild(consoleEl);
        }
        // If console already exists from streaming, leave it as-is.

        // Keep it at the bottom and scroll.
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        _thinkingTool = tool || '';
    }

    function appendToolStreamChunk(chunk) {
        // Append a live output chunk to the thinking console.
        // If no console block exists yet, create one on the current
        // thinking indicator (the tool is still running).
        if (!chunk) return;

        let el = document.getElementById('thinkingIndicator');
        if (!el) return;  // No active indicator — ignore.

        let consoleEl = el.querySelector('.thinking-console');
        if (!consoleEl) {
            // First chunk — create the console block.
            consoleEl = document.createElement('div');
            consoleEl.className = 'thinking-console';

            const header = document.createElement('div');
            header.className = 'thinking-console-header';
            header.innerHTML = '<span class="thinking-console-chevron">\u25BE</span> Output';
            header.addEventListener('click', function() {
                consoleEl.classList.toggle('collapsed');
            });
            consoleEl.appendChild(header);

            const pre = document.createElement('pre');
            pre.className = 'thinking-console-output';
            consoleEl.appendChild(pre);

            el.appendChild(consoleEl);
        }

        const pre = consoleEl.querySelector('.thinking-console-output');
        pre.textContent += chunk;

        // Auto-scroll the console to the bottom.
        pre.scrollTop = pre.scrollHeight;
        // Keep the chat scrolled to the bottom too.
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
        if (info.tools) updateToolsBar(info.tools);
        if (info.skills) updateSkillsBar(info.skills);
        // User profile updates (who the agent is talking to).
        if ('personality_id' in info) {
            activePersonalityId = info.personality_id || '';
            activePersonalityName = info.personality_name || 'No profile';
            personaName.textContent = activePersonalityName;
            highlightActivePersonaInDropdown();
        }
        // Playbook override updates.
        if ('playbook_id' in info) {
            activePlaybookId = info.playbook_id || '';
            activePlaybookName = info.playbook_name || 'Auto';
            $('#playbookName').textContent = activePlaybookName;
            highlightActivePlaybookInDropdown();
        }
        // Force script mode sync.
        if ('force_script' in info) {
            forceScriptMode = !!info.force_script;
            var toggle = $('#forceScriptToggle');
            if (toggle) toggle.checked = forceScriptMode;
            sessionStorage.setItem('forceScriptMode', forceScriptMode ? 'true' : 'false');
        }
    }

    function updateToolsBar(tools) {
        var list = $('#toolsList');
        if (!list) return;
        if (!tools || !tools.length) {
            list.innerHTML = '<span class="tools-empty">No tools enabled</span>';
            return;
        }
        list.innerHTML = tools.slice().sort().map(function(t) {
            return '<span class="tool-chip" title="' + escapeHtml(t) + '">' + escapeHtml(t) + '</span>';
        }).join('');
    }

    function updateSkillsBar(skills) {
        var bar = $('#skillsBar');
        var list = $('#skillsList');
        if (!list || !bar) return;
        if (!skills || !skills.length) {
            bar.style.display = 'none';
            return;
        }
        bar.style.display = '';
        list.innerHTML = skills.slice().sort(function(a, b) { return a.name.localeCompare(b.name); }).map(function(s) {
            var tip = '/' + escapeHtml(s.name);
            if (s.description) tip += ' \u2014 ' + escapeHtml(s.description);
            return '<span class="tool-chip skill-chip" title="' + tip + '">/' + escapeHtml(s.name) + '</span>';
        }).join('');
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
            const mtype = m.model_type || 'llm';
            const isActive = label === activeModelLabel;
            return `<div class="model-option${isActive ? ' active' : ''}" data-selector="${escapeHtml(m.id || String(i + 1))}" data-label="${escapeHtml(label)}" title="${escapeHtml(desc || label)}">
                <span class="model-option-name">${escapeHtml(label)}</span>
                <span class="model-option-type">${escapeHtml(mtype)}</span>
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

    // ── Persona Selector ───────────────────────────────────────

    const personaDropdown = $('#personaDropdown');
    const personaBadge = $('#personaBadge');
    const personaName = $('#personaName');

    function buildPersonaDropdown() {
        let html = '';

        // "No profile" option — clears user context.
        const isNoneActive = !activePersonalityId;
        html += `<div class="model-option${isNoneActive ? ' active' : ''}" data-pid="" title="No user profile — generic responses">
            <span class="model-option-name">No profile</span>
            <span class="model-option-desc">Generic context</span>
            ${isNoneActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
        </div>`;

        // User profiles — these describe who the agent is talking to.
        const userProfiles = personalities.filter(p => p.id);
        if (userProfiles.length) {
            html += '<div class="pd-section-label">User Profiles</div>';
            userProfiles.forEach(p => {
                const isActive = activePersonalityId === p.id;
                const desc = p.description ? p.description.slice(0, 60) : '';
                const sourceTag = p.is_telegram
                    ? '<span class="persona-source-tag">Telegram</span>'
                    : '';
                html += `<div class="model-option${isActive ? ' active' : ''}" data-pid="${escapeHtml(p.id)}" title="${escapeHtml(p.description || p.name)}">
                    <span class="model-option-name"><span class="persona-name-wrap"><span class="persona-name-text">${escapeHtml(p.name)}</span>${sourceTag}</span></span>
                    ${desc ? `<span class="model-option-desc">${escapeHtml(desc)}</span>` : ''}
                    ${isActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
                </div>`;
            });
        }

        // "Create new" button.
        html += '<div class="pd-create-btn" id="pdCreateBtn">&#x2795; Create new user profile</div>';

        personaDropdown.innerHTML = html;

        // Bind click handlers for personality options.
        personaDropdown.querySelectorAll('.model-option').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                const pid = el.dataset.pid;
                selectPersonality(pid);
            });
        });

        // Bind "Create new" button.
        const createBtn = personaDropdown.querySelector('#pdCreateBtn');
        if (createBtn) {
            createBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                showPersonaCreateForm();
            });
        }
    }

    function selectPersonality(pid) {
        activePersonalityId = pid;
        const p = personalities.find(pp => pp.id === pid);
        activePersonalityName = p ? p.name : 'No profile';
        if (!pid) activePersonalityName = 'No profile';
        personaName.textContent = activePersonalityName;
        send({ type: 'set_personality', personality_id: pid });
        highlightActivePersonaInDropdown();
        hidePersonaDropdown();
    }

    function highlightActivePersonaInDropdown() {
        personaDropdown.querySelectorAll('.model-option').forEach(el => {
            const isActive = (el.dataset.pid || '') === (activePersonalityId || '');
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

    function togglePersonaDropdown() {
        personaDropdown.classList.toggle('hidden');
    }

    function hidePersonaDropdown() {
        personaDropdown.classList.add('hidden');
        // Remove any open create form.
        const form = personaDropdown.querySelector('.pd-create-form');
        if (form) form.remove();
    }

    function showPersonaCreateForm() {
        // Don't add a second form.
        if (personaDropdown.querySelector('.pd-create-form')) return;

        const form = document.createElement('div');
        form.className = 'pd-create-form';
        form.innerHTML = `
            <input type="text" placeholder="Your name (e.g. Mile Kekin)" class="pd-name-input" />
            <textarea placeholder="Short description (role, title)" rows="2" class="pd-desc-input"></textarea>
            <textarea placeholder="Background &amp; experience" rows="2" class="pd-bg-input"></textarea>
            <input type="text" placeholder="Expertise areas (comma-separated)" class="pd-exp-input" />
            <div class="pd-form-actions">
                <button class="pd-cancel-btn">Cancel</button>
                <button class="primary pd-save-btn">Create</button>
            </div>
        `;

        personaDropdown.appendChild(form);

        // Stop clicks inside form from propagating.
        form.addEventListener('click', (e) => e.stopPropagation());

        form.querySelector('.pd-cancel-btn').addEventListener('click', () => {
            form.remove();
        });

        form.querySelector('.pd-save-btn').addEventListener('click', async () => {
            const nameVal = form.querySelector('.pd-name-input').value.trim();
            if (!nameVal) { form.querySelector('.pd-name-input').focus(); return; }

            const descVal = form.querySelector('.pd-desc-input').value.trim();
            const bgVal = form.querySelector('.pd-bg-input').value.trim();
            const expVal = form.querySelector('.pd-exp-input').value.trim();

            // Create a web-prefixed personality via REST API.
            const pid = 'web_' + Date.now();
            const body = { name: nameVal };
            if (descVal) body.description = descVal;
            if (bgVal) body.background = bgVal;
            if (expVal) body.expertise = expVal;

            try {
                const resp = await fetch('/api/user-personalities/' + encodeURIComponent(pid), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                if (!resp.ok) throw new Error('Failed to create user profile');
                const created = await resp.json();
                // Add to our local list.
                created.id = pid;
                personalities.push(created);
                // Rebuild and select.
                buildPersonaDropdown();
                selectPersonality(pid);
            } catch (err) {
                console.error('Failed to create user profile:', err);
                addChatMessage('error', 'Failed to create user profile: ' + err.message);
            }
        });

        // Focus on name input.
        setTimeout(() => form.querySelector('.pd-name-input').focus(), 50);
    }

    // ── Playbook Selector ──────────────────────────────────────

    const playbookDropdown = $('#playbookDropdown');
    const playbookBadge = $('#playbookBadge');
    const playbookNameEl = $('#playbookName');

    function buildPlaybookDropdown() {
        let html = '';

        // "Auto" option — system auto-retrieves relevant playbooks.
        const isAutoActive = !activePlaybookId;
        html += `<div class="model-option${isAutoActive ? ' active' : ''}" data-pbid="" title="Automatically select relevant playbooks based on task type">
            <span class="model-option-name">Auto</span>
            <span class="model-option-desc">System picks best match</span>
            ${isAutoActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
        </div>`;

        // "None" option — disable playbook injection entirely.
        const isNoneActive = activePlaybookId === '__none__';
        html += `<div class="model-option${isNoneActive ? ' active' : ''}" data-pbid="__none__" title="Disable playbook injection for this session">
            <span class="model-option-name">None</span>
            <span class="model-option-desc">No playbook guidance</span>
            ${isNoneActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
        </div>`;

        // Existing playbooks.
        if (playbooks.length) {
            html += '<div class="pd-section-label">Saved Playbooks</div>';
            playbooks.forEach(pb => {
                const isActive = activePlaybookId === pb.id;
                const desc = pb.trigger_description ? pb.trigger_description.slice(0, 60) : (pb.task_type || '');
                html += `<div class="model-option${isActive ? ' active' : ''}" data-pbid="${escapeHtml(pb.id)}" title="${escapeHtml(pb.name + (pb.trigger_description ? ' — ' + pb.trigger_description : ''))}">
                    <span class="model-option-name">${escapeHtml(pb.name)}</span>
                    ${desc ? `<span class="model-option-desc">${escapeHtml(desc)}</span>` : ''}
                    ${isActive ? '<span class="model-option-check">&#x2713;</span>' : ''}
                </div>`;
            });
        }

        playbookDropdown.innerHTML = html;

        // Bind click handlers for playbook options.
        playbookDropdown.querySelectorAll('.model-option').forEach(el => {
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                const pbid = el.dataset.pbid;
                selectPlaybook(pbid);
            });
        });
    }

    function selectPlaybook(pbid) {
        activePlaybookId = pbid;
        if (!pbid) {
            activePlaybookName = 'Auto';
        } else if (pbid === '__none__') {
            activePlaybookName = 'None';
        } else {
            const pb = playbooks.find(p => p.id === pbid);
            activePlaybookName = pb ? pb.name : pbid;
        }
        playbookNameEl.textContent = activePlaybookName;
        send({ type: 'set_playbook', playbook_id: pbid });
        highlightActivePlaybookInDropdown();
        hidePlaybookDropdown();
    }

    function highlightActivePlaybookInDropdown() {
        playbookDropdown.querySelectorAll('.model-option').forEach(el => {
            const isActive = (el.dataset.pbid || '') === (activePlaybookId || '');
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

    function togglePlaybookDropdown() {
        playbookDropdown.classList.toggle('hidden');
    }

    function hidePlaybookDropdown() {
        playbookDropdown.classList.add('hidden');
    }

    // ── File Upload ──────────────────────────────────────────

    const fileInput = $('#fileInput');
    const uploadBtn = $('#uploadBtn');
    const dropOverlay = $('#dropOverlay');
    const chatPane = $('#chatPane');
    const attachmentPreview = $('#attachmentPreview');

    var pendingImagePath = null;
    var pendingFilePath = null;
    var _IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'];

    function resizeImageFile(file, maxSide) {
        return new Promise(function(resolve, reject) {
            var img = new Image();
            var url = URL.createObjectURL(file);
            img.onload = function() {
                URL.revokeObjectURL(url);
                var w = img.width, h = img.height;
                var needsResize = (w > maxSide || h > maxSide);
                var scale = needsResize ? maxSide / Math.max(w, h) : 1;
                var nw = Math.round(w * scale), nh = Math.round(h * scale);
                var canvas = document.createElement('canvas');
                canvas.width = nw; canvas.height = nh;
                canvas.getContext('2d').drawImage(img, 0, 0, nw, nh);
                var outName = file.name.replace(/\.[^.]+$/, '.jpg');
                canvas.toBlob(function(blob) {
                    if (!blob) { resolve(file); return; }
                    resolve(new File([blob], outName, { type: 'image/jpeg' }));
                }, 'image/jpeg', 0.85);
            };
            img.onerror = function() { URL.revokeObjectURL(url); reject(new Error('load')); };
            img.src = url;
        });
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function showAttachmentChip(filename, path, objectUrl, isFile) {
        if (isFile) {
            pendingFilePath = path;
        } else {
            pendingImagePath = path;
        }
        attachmentPreview.innerHTML = '';
        var chip = document.createElement('div');
        chip.className = 'attachment-chip';
        if (objectUrl) {
            var thumb = document.createElement('img');
            thumb.src = objectUrl;
            thumb.className = 'attachment-thumb';
            chip.appendChild(thumb);
        } else if (isFile) {
            var icon = document.createElement('span');
            icon.className = 'attachment-file-icon';
            icon.textContent = '\uD83D\uDCC4';
            chip.appendChild(icon);
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
        pendingFilePath = null;
        attachmentPreview.innerHTML = '';
        attachmentPreview.classList.add('hidden');
        fileInput.value = '';
    }

    async function uploadFile(file) {
        if (!file) return;
        var ext = file.name.split('.').pop().toLowerCase();
        var isImage = _IMAGE_EXTS.indexOf(ext) !== -1;
        var isData = (ext === 'csv' || ext === 'xlsx' || ext === 'zip');

        if (!isImage && !isData) {
            addChatMessage('assistant', '**Error:** Supported files: images (.png, .jpg, .jpeg, .webp, .gif, .bmp) and data (.csv, .xlsx, .zip).');
            return;
        }

        if (isImage) {
            // Upload image and stage as attachment (don't auto-submit).
            uploadBtn.disabled = true;
            uploadBtn.textContent = '\u23F3';

            // Resize so longest side is max 1024px to save LLM tokens.
            try {
                file = await resizeImageFile(file, 1024);
            } catch (_) { /* use original on error */ }

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

        // Data file (csv/xlsx) — upload and stage as attachment, let the user
        // decide what to do (datastore import, deep memory indexing, extraction, etc.).
        uploadBtn.disabled = true;
        uploadBtn.textContent = '\u23F3';

        var formData = new FormData();
        formData.append('file', file);

        try {
            var res = await fetch('/api/file/upload', {
                method: 'POST',
                body: formData,
            });
            var data = await res.json();
            if (!res.ok) {
                addChatMessage('assistant', '**Upload failed:** ' + (data.error || 'Unknown error'));
            } else {
                showAttachmentChip(file.name, data.path, null, true);
                messageInput.focus();
            }
        } catch (err) {
            addChatMessage('assistant', '**Upload failed:** ' + err.message);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '\uD83D\uDCCE';
            fileInput.value = '';
        }
    }

    // ── Voice Recording (live STT via WebSocket) ─────────────

    function setMicState(state) {
        micState = state;
        if (!micBtn) return;

        micBtn.classList.remove('recording', 'transcribing');
        micBtn.disabled = false;

        switch (state) {
            case 'idle':
                micBtn.textContent = '\uD83C\uDF99\uFE0F';
                micBtn.title = 'Voice input (click to record)';
                break;
            case 'recording':
                micBtn.classList.add('recording');
                micBtn.textContent = '\u23F9';
                micBtn.title = 'Click to stop recording';
                break;
            case 'transcribing':
                micBtn.classList.add('transcribing');
                micBtn.textContent = '\u23F3';
                micBtn.title = 'Transcribing...';
                micBtn.disabled = true;
                break;
        }
    }

    function sttCleanup() {
        if (sttProcessor) { sttProcessor.disconnect(); sttProcessor = null; }
        if (sttSource) { sttSource.disconnect(); sttSource = null; }
        if (sttAudioCtx) { sttAudioCtx.close().catch(() => {}); sttAudioCtx = null; }
        if (sttStream) { sttStream.getTracks().forEach(t => t.stop()); sttStream = null; }
    }

    async function startRecording() {
        if (micState !== 'idle') return;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const msg = window.isSecureContext === false
                ? 'Microphone requires HTTPS or localhost. Current page is not in a secure context.'
                : 'Voice input is not supported in this browser.';
            addChatMessage('error', msg);
            return;
        }

        try {
            sttStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                addChatMessage('error', 'Microphone permission denied. Please allow microphone access and try again.');
            } else if (err.name === 'NotFoundError') {
                addChatMessage('error', 'No microphone found. Please connect a microphone and try again.');
            } else {
                addChatMessage('error', 'Failed to start recording: ' + err.message);
            }
            return;
        }

        // Save text already in the input — transcription appends after it.
        sttPreText = messageInput.value;

        // Open STT WebSocket.
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        sttWs = new WebSocket(proto + '//' + location.host + '/ws/stt');
        sttWs.binaryType = 'arraybuffer';

        sttWs.onopen = () => {
            // Wait for stt_ready before starting audio — handled in onmessage.
        };

        sttWs.onmessage = (e) => {
            const data = JSON.parse(e.data);

            if (data.type === 'stt_ready') {
                sttIsRealtime = data.realtime;
                console.log('[mic] STT ready, realtime:', sttIsRealtime);
                _startAudioPipeline();
                return;
            }

            if (data.type === 'stt_partial' || data.type === 'stt_final') {
                const sep = sttPreText && !sttPreText.endsWith(' ') && !sttPreText.endsWith('\n') ? ' ' : '';
                messageInput.value = sttPreText + (data.text ? sep + data.text : '');
                autoResizeInput();
            }

            if (data.type === 'stt_final') {
                _closeSttWs();
                sttCleanup();
                setMicState('idle');
                messageInput.focus();
            }

            if (data.type === 'stt_error') {
                addChatMessage('error', 'STT error: ' + data.error);
                _closeSttWs();
                sttCleanup();
                setMicState('idle');
            }
        };

        sttWs.onerror = () => {
            sttCleanup();
            setMicState('idle');
            addChatMessage('error', 'STT WebSocket connection failed.');
        };

        sttWs.onclose = () => {
            if (micState === 'recording') {
                sttCleanup();
                setMicState('idle');
            }
        };

        setMicState('recording');
    }

    function _startAudioPipeline() {
        // Create AudioContext at 16 kHz — browser resamples from native rate.
        try {
            sttAudioCtx = new AudioContext({ sampleRate: 16000 });
        } catch (_) {
            sttAudioCtx = new AudioContext();
            console.warn('[mic] Could not set 16 kHz sample rate, using', sttAudioCtx.sampleRate);
        }
        sttSource = sttAudioCtx.createMediaStreamSource(sttStream);

        // ScriptProcessorNode: capture raw PCM float32 → convert to int16 → send.
        sttProcessor = sttAudioCtx.createScriptProcessor(4096, 1, 1);
        sttProcessor.onaudioprocess = (e) => {
            if (!sttWs || sttWs.readyState !== WebSocket.OPEN) return;
            const f32 = e.inputBuffer.getChannelData(0);
            const i16 = new Int16Array(f32.length);
            for (let i = 0; i < f32.length; i++) {
                const s = Math.max(-1, Math.min(1, f32[i]));
                i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            sttWs.send(i16.buffer);
        };

        sttSource.connect(sttProcessor);
        sttProcessor.connect(sttAudioCtx.destination);

        // Auto-stop safety.
        setTimeout(() => {
            if (micState === 'recording') stopRecording();
        }, MAX_RECORDING_SECONDS * 1000);
    }

    function _closeSttWs() {
        if (sttWs) {
            try { sttWs.close(); } catch (_) {}
            sttWs = null;
        }
    }

    function stopRecording() {
        if (micState !== 'recording') return;

        // Stop audio capture first.
        sttCleanup();

        // Tell backend we're done.
        if (sttWs && sttWs.readyState === WebSocket.OPEN) {
            sttWs.send(JSON.stringify({ type: 'stop' }));
        }

        // In realtime mode the final text is already visible; in batch mode
        // we wait for stt_final, so show transcribing state.
        if (!sttIsRealtime) {
            setMicState('transcribing');
        } else {
            // Small grace period for last stt_final to arrive.
            setMicState('transcribing');
        }
    }

    function toggleMic() {
        if (micState === 'idle') startRecording();
        else if (micState === 'recording') stopRecording();
    }

    // ── Input Handling ──────────────────────────────────────

    function handleSend() {
        const text = messageInput.value.trim();
        if (!text && !pendingImagePath && !pendingFilePath) return;
        if (!isConnected) return;

        hideSuggestions();

        // ── /btw command: inject additional instructions while task is running ──
        var btwMatch = text.match(/^\/btw\s+([\s\S]+)/i) || text.match(/^btw\s+([\s\S]+)/i);
        if (btwMatch) {
            var btwText = btwMatch[1].trim();
            if (btwText) {
                send({ type: 'btw', content: btwText });
                appendMessage('user', 'btw: ' + btwText);
            }
            messageInput.value = '';
            autoResizeInput();
            return;
        }

        if (text.startsWith('/')) {
            send({ type: 'command', command: text });
        } else {
            var msg = { type: 'chat', content: text || '' };
            if (pendingImagePath) {
                msg.image_path = pendingImagePath;
            }
            if (pendingFilePath) {
                msg.file_path = pendingFilePath;
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
            loadSessionFiles();
        }
    }

    function switchTab(tabName) {
        $$('.sidebar-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
        $$('.sidebar-content').forEach(c => c.classList.remove('active'));
        $(`#tab${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`).classList.add('active');
        if (tabName === 'files') loadSessionFiles();
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

    // ── File Explorer ────────────────────────────────────────

    let sessionFiles = [];
    const filesList = $('#filesList');
    const filesCount = $('#filesCount');
    const filesSearch = $('#filesSearch');
    const filePreview = $('#filePreview');
    const filePreviewName = $('#filePreviewName');
    const filePreviewMeta = $('#filePreviewMeta');
    const filePreviewBody = $('#filePreviewBody');
    const fileDownloadBtn = $('#fileDownloadBtn');
    const fileViewBtn = $('#fileViewBtn');
    const fileMdViewBtn = $('#fileMdViewBtn');
    var currentMdContent = null;
    var currentMdFilename = null;

    function _isMarkdownFile(filename) {
        return /\.(md|markdown|mdown|mkd|mkdn|mdx)$/i.test(filename || '');
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function fileIcon(ext) {
        var icons = {
            '.py': '\u{1F40D}', '.js': '\u{1F4DC}', '.ts': '\u{1F4DC}',
            '.html': '\u{1F310}', '.css': '\u{1F3A8}', '.json': '\u{1F4CB}',
            '.md': '\u{1F4DD}', '.txt': '\u{1F4C4}', '.csv': '\u{1F4CA}',
            '.png': '\u{1F5BC}', '.jpg': '\u{1F5BC}', '.jpeg': '\u{1F5BC}',
            '.gif': '\u{1F5BC}', '.webp': '\u{1F5BC}', '.svg': '\u{1F5BC}',
            '.mp3': '\u{1F3B5}', '.wav': '\u{1F3B5}', '.mp4': '\u{1F3AC}',
            '.sh': '\u{1F4DF}', '.sql': '\u{1F5C3}', '.log': '\u{1F4C3}',
        };
        return icons[ext] || '\u{1F4C1}';
    }

    async function loadSessionFiles() {
        if (!sessionInfo.id) {
            filesList.innerHTML = '<div class="loading">No active session</div>';
            filesCount.textContent = '0 files';
            return;
        }
        try {
            var res = await fetch('/api/files/session/' + encodeURIComponent(sessionInfo.id));
            sessionFiles = await res.json();
            renderFilesList(sessionFiles);
        } catch (e) {
            filesList.innerHTML = '<div class="loading">Failed to load files</div>';
        }
    }

    function getFileCategory(logical) {
        var first = (logical || '').split('/')[0];
        var cats = ['downloads','media','output','scripts','showcase','skills','summaries','tmp','tools'];
        return cats.indexOf(first) >= 0 ? first : 'other';
    }

    var categoryLabels = {
        scripts: '\u{1F4DC} Scripts',
        output: '\u{1F4E4} Output',
        media: '\u{1F5BC}\uFE0F Media',
        downloads: '\u{2B07}\uFE0F Downloads',
        tools: '\u{1F527} Tools',
        showcase: '\u{2B50} Showcase',
        summaries: '\u{1F4DD} Summaries',
        skills: '\u{26A1} Skills',
        tmp: '\u{1F4C2} Temp',
        other: '\u{1F4C1} Other',
    };

    function renderFilesList(files) {
        var query = (filesSearch.value || '').toLowerCase().trim();
        var filtered = files;
        if (query) {
            filtered = files.filter(function(f) {
                return f.filename.toLowerCase().includes(query) ||
                       f.logical.toLowerCase().includes(query) ||
                       f.extension.toLowerCase().includes(query);
            });
        }
        filesCount.textContent = filtered.length + ' file' + (filtered.length !== 1 ? 's' : '');

        if (!filtered.length) {
            filesList.innerHTML = query
                ? '<div class="loading">No matching files</div>'
                : '<div class="loading">No files generated in this session</div>';
            return;
        }

        // Group by category folder.
        var groups = {};
        filtered.forEach(function(f) {
            var cat = getFileCategory(f.logical);
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(f);
        });

        var order = ['scripts','output','media','downloads','tools','showcase','summaries','skills','tmp','other'];
        var html = '';
        order.forEach(function(cat) {
            var items = groups[cat];
            if (!items || !items.length) return;
            html += '<div class="files-group-label">' + (categoryLabels[cat] || cat) + ' <span class="files-group-count">(' + items.length + ')</span></div>';
            items.forEach(function(f) {
                var icon = fileIcon(f.extension);
                var size = formatFileSize(f.size);
                var missing = !f.exists ? ' <span class="file-missing">missing</span>' : '';
                // Format modification date/time instead of repeating the filename.
                var dateStr = '';
                if (f.modified) {
                    var d = new Date(f.modified * 1000);
                    dateStr = d.toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'}) +
                              ' ' + d.toLocaleTimeString(undefined, {hour:'2-digit',minute:'2-digit'});
                }
                html += '<div class="file-item" data-physical="' + escapeHtml(f.physical) + '" data-logical="' + escapeHtml(f.logical) + '" title="' + escapeHtml(f.logical) + '">' +
                    '<span class="file-item-icon">' + icon + '</span>' +
                    '<div class="file-item-info">' +
                    '<div class="file-item-name">' + escapeHtml(f.filename) + missing + '</div>' +
                    '<div class="file-item-path">' + dateStr + ' &middot; ' + size + '</div>' +
                    '</div></div>';
            });
        });

        filesList.innerHTML = html;

        filesList.querySelectorAll('.file-item').forEach(function(el) {
            el.addEventListener('click', function() {
                var physical = el.dataset.physical;
                var f = sessionFiles.find(function(x) { return x.physical === physical; });
                if (f) openFilePreview(f);
            });
        });
    }

    async function openFilePreview(f) {
        filePreviewName.textContent = f.filename;
        filePreviewMeta.textContent = f.logical + ' \u2022 ' + formatFileSize(f.size) + ' \u2022 ' + f.mime_type;
        fileDownloadBtn.href = '/api/files/download?path=' + encodeURIComponent(f.physical);
        filePreview.classList.remove('hidden');
        currentMdContent = null;
        currentMdFilename = null;
        fileMdViewBtn.style.display = 'none';

        var ext = (f.extension || '').toLowerCase();
        if (ext === '.html' || ext === '.htm' || ext === '.svg') {
            fileViewBtn.href = '/api/files/view?path=' + encodeURIComponent(f.physical);
            fileViewBtn.style.display = '';
        } else {
            fileViewBtn.style.display = 'none';
        }

        if (!f.exists) {
            filePreviewBody.innerHTML = '<div class="file-preview-empty">File no longer exists on disk</div>';
            return;
        }

        if (f.is_text) {
            try {
                var res = await fetch('/api/files/content?path=' + encodeURIComponent(f.physical));
                var data = await res.json();
                if (data.error) {
                    filePreviewBody.innerHTML = '<div class="file-preview-empty">' + escapeHtml(data.error) + '</div>';
                } else {
                    filePreviewBody.innerHTML = '<pre class="file-preview-code"><code>' + escapeHtml(data.content) + '</code></pre>';
                    if (_isMarkdownFile(f.filename)) {
                        currentMdContent = data.content;
                        currentMdFilename = f.filename;
                        fileMdViewBtn.style.display = '';
                    }
                }
            } catch (e) {
                filePreviewBody.innerHTML = '<div class="file-preview-empty">Failed to load content</div>';
            }
        } else if (/^image\//.test(f.mime_type)) {
            filePreviewBody.innerHTML = '<img src="/api/files/download?path=' + encodeURIComponent(f.physical) +
                '" class="file-preview-image" alt="' + escapeHtml(f.filename) + '">';
        } else {
            filePreviewBody.innerHTML = '<div class="file-preview-empty">Binary file \u2014 use Download to view</div>';
        }
    }

    function closeFilePreview() {
        filePreview.classList.add('hidden');
        filePreviewBody.innerHTML = '';
        currentMdContent = null;
        currentMdFilename = null;
        fileMdViewBtn.style.display = 'none';
        fileViewBtn.style.display = 'none';
    }

    function openMdViewer() {
        if (!currentMdContent) return;
        $('#mdViewerTitle').textContent = currentMdFilename || 'Markdown';
        $('#mdViewerBody').innerHTML = renderMarkdown(currentMdContent);
        $('#mdViewerModal').classList.remove('hidden');
    }

    function closeMdViewer() {
        $('#mdViewerModal').classList.add('hidden');
        $('#mdViewerBody').innerHTML = '';
    }

    function _exportMd(format) {
        if (!currentMdContent) return;
        var btn = format === 'pdf' ? $('#mdExportPdf') : $('#mdExportDocx');
        var origText = btn.textContent;
        btn.textContent = '...';
        btn.disabled = true;
        fetch('/api/files/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                markdown: currentMdContent,
                format: format,
                filename: currentMdFilename || 'document'
            })
        }).then(function(resp) {
            if (!resp.ok) throw new Error('Export failed: ' + resp.status);
            return resp.blob();
        }).then(function(blob) {
            var base = (currentMdFilename || 'document').replace(/\.md$/i, '');
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = base + '.' + format;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }).catch(function(err) {
            console.error('MD export error:', err);
            alert('Export failed: ' + err.message);
        }).finally(function() {
            btn.textContent = origText;
            btn.disabled = false;
        });
    }

    function exportMdAsPdf() { _exportMd('pdf'); }
    function exportMdAsDocx() { _exportMd('docx'); }

    // ── Folder Browser ─────────────────────────────────────

    var folderCurrentPath = null;
    var _gwsAvailable = false;

    function openFolderModal() {
        $('#folderModal').classList.remove('hidden');
        loadExtraDirs();
        browseDir('');
        _initFolderTabs();
    }

    function closeFolderModal() {
        $('#folderModal').classList.add('hidden');
    }

    async function _initFolderTabs() {
        // Check for Windows drives.
        try {
            var dRes = await fetch('/api/drives');
            var dData = await dRes.json();
            var driveBar = $('#folderDriveBar');
            if (dData.drives && dData.drives.length > 0) {
                driveBar.classList.remove('hidden');
                driveBar.innerHTML = dData.drives.map(function(d) {
                    return '<button class="folder-drive-btn" data-drive="' +
                        escapeHtml(d) + '">' + escapeHtml(d) + '</button>';
                }).join('');
                driveBar.querySelectorAll('.folder-drive-btn').forEach(function(btn) {
                    btn.addEventListener('click', function() {
                        browseDir(btn.dataset.drive + '\\');
                    });
                });
            } else {
                driveBar.classList.add('hidden');
            }
        } catch (e) {
            $('#folderDriveBar').classList.add('hidden');
        }

        // Check for gws availability → show/hide GDrive tab.
        try {
            var gRes = await fetch('/api/gws-status');
            var gData = await gRes.json();
            _gwsAvailable = !!gData.available;
            $('#folderTabGdriveBtn').style.display = _gwsAvailable ? '' : 'none';
        } catch (e) {
            _gwsAvailable = false;
            $('#folderTabGdriveBtn').style.display = 'none';
        }

        // Tab switching.
        document.querySelectorAll('.folder-tab').forEach(function(tab) {
            tab.onclick = function() {
                document.querySelectorAll('.folder-tab').forEach(function(t) { t.classList.remove('active'); });
                tab.classList.add('active');
                var target = tab.dataset.tab;
                $('#folderTabLocal').classList.toggle('hidden', target !== 'local');
                $('#folderTabGdrive').classList.toggle('hidden', target !== 'gdrive');
                if (target === 'gdrive') {
                    _loadGdriveTab();
                }
            };
        });
    }

    async function loadExtraDirs() {
        var list = $('#folderList');
        try {
            var res = await fetch('/api/read-folders');
            var data = await res.json();
            var dirs = data.dirs || [];
            if (!dirs.length) {
                list.innerHTML = '<div class="folder-empty">No extra folders configured</div>';
                return;
            }
            list.innerHTML = dirs.map(function(d) {
                var cls = d.exists ? '' : ' missing';
                return '<div class="folder-entry' + cls + '">' +
                    '<span class="folder-entry-path" title="' + escapeHtml(d.resolved) + '">' + escapeHtml(d.resolved) + '</span>' +
                    (!d.exists ? '<span class="folder-entry-warn" title="Directory does not exist">&#x26A0;</span>' : '') +
                    '<button class="folder-entry-remove" data-path="' + escapeHtml(d.resolved) + '" title="Remove">&times;</button>' +
                    '</div>';
            }).join('');
            list.querySelectorAll('.folder-entry-remove').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    removeExtraDir(btn.dataset.path);
                });
            });
        } catch (e) {
            list.innerHTML = '<div class="folder-empty">Failed to load</div>';
        }
    }

    async function browseDir(path) {
        var dirsEl = $('#folderDirs');
        dirsEl.innerHTML = '<div class="folder-empty">Loading...</div>';

        try {
            var url = '/api/browse' + (path ? '?path=' + encodeURIComponent(path) : '');
            var res = await fetch(url);
            var data = await res.json();
            if (data.error) {
                dirsEl.innerHTML = '<div class="folder-empty">' + escapeHtml(data.error) + '</div>';
                return;
            }
            folderCurrentPath = data.path;
            $('#folderSelectedPath').textContent = data.path;
            var addBtn = $('#folderAddBtn');
            addBtn.disabled = !!data.blocked;
            addBtn.title = data.blocked ? 'Cannot add root or system directories' : 'Add this folder as a read directory';

            renderBreadcrumb(data.path);

            var html = '';
            if (data.parent) {
                html += '<div class="folder-dir-item parent" data-path="' + escapeHtml(data.parent) + '">' +
                    '<span class="folder-dir-icon">\u2B06</span> ..</div>';
            }
            if (!data.dirs.length && !data.parent) {
                html += '<div class="folder-empty">No subdirectories</div>';
            }
            data.dirs.forEach(function(name) {
                var sep = data.path.includes('\\') ? '\\' : '/';
                var full = data.path + (data.path.endsWith(sep) ? '' : sep) + name;
                html += '<div class="folder-dir-item" data-path="' + escapeHtml(full) + '">' +
                    '<span class="folder-dir-icon">\uD83D\uDCC1</span> ' + escapeHtml(name) + '</div>';
            });
            dirsEl.innerHTML = html;

            dirsEl.querySelectorAll('.folder-dir-item').forEach(function(el) {
                el.addEventListener('click', function() {
                    browseDir(el.dataset.path);
                });
            });
        } catch (e) {
            dirsEl.innerHTML = '<div class="folder-empty">Error: ' + escapeHtml(e.message) + '</div>';
        }
    }

    function renderBreadcrumb(fullPath) {
        var bc = $('#folderBreadcrumb');
        var sep = fullPath.includes('\\') ? '\\' : '/';
        var parts = fullPath.split(sep).filter(Boolean);
        var html = '';
        var built = '';

        if (fullPath.startsWith('/')) {
            html += '<span class="folder-bc-item" data-path="/">/</span>';
            built = '/';
        }

        parts.forEach(function(p, i) {
            if (i === 0 && !fullPath.startsWith('/')) {
                // Windows drive letter (e.g. "C:").
                built = p + sep;
            } else {
                built += (built.endsWith(sep) ? '' : sep) + p;
            }
            if (i > 0 || fullPath.startsWith('/')) {
                html += '<span class="folder-bc-sep">' + sep + '</span>';
            }
            html += '<span class="folder-bc-item" data-path="' + escapeHtml(built) + '">' + escapeHtml(p) + '</span>';
        });

        bc.innerHTML = html;
        bc.querySelectorAll('.folder-bc-item').forEach(function(el) {
            el.addEventListener('click', function() {
                browseDir(el.dataset.path);
            });
        });
    }

    async function addCurrentFolder() {
        if (!folderCurrentPath) return;
        var btn = $('#folderAddBtn');
        btn.disabled = true;
        btn.textContent = 'Adding...';
        try {
            var res = await fetch('/api/read-folders', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: folderCurrentPath }),
            });
            var data = await res.json();
            if (!res.ok || !data.ok) {
                addChatMessage('error', 'Add folder failed: ' + (data.error || 'Unknown error'));
            } else {
                addChatMessage('assistant', 'Read folder added: `' + data.path + '`');
                loadExtraDirs();
            }
        } catch (e) {
            addChatMessage('error', 'Add folder failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Add this folder';
        }
    }

    async function removeExtraDir(path) {
        try {
            var res = await fetch('/api/read-folders', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path }),
            });
            var data = await res.json();
            if (!res.ok || !data.ok) {
                addChatMessage('error', 'Remove folder failed: ' + (data.error || 'Unknown error'));
            } else {
                addChatMessage('assistant', 'Read folder removed: `' + data.path + '`');
                loadExtraDirs();
            }
        } catch (e) {
            addChatMessage('error', 'Remove folder failed: ' + e.message);
        }
    }

    // ── Google Drive Folder Browser ─────────────────────────

    var _gdriveCurrent = { id: 'root', name: 'My Drive' };
    var _gdriveBcStack = [{ id: 'root', name: 'My Drive' }];

    async function _loadGdriveTab() {
        _loadGdriveFolderList();
        _browseGdrive('root', 'My Drive');
    }

    async function _loadGdriveFolderList() {
        var list = $('#gdriveList');
        try {
            var res = await fetch('/api/read-folders/gdrive');
            var data = await res.json();
            var folders = data.folders || [];
            if (!folders.length) {
                list.innerHTML = '<div class="folder-empty">No Google Drive folders configured</div>';
                return;
            }
            list.innerHTML = folders.map(function(f) {
                return '<div class="folder-entry">' +
                    '<span class="folder-entry-path" title="ID: ' + escapeHtml(f.id) + '">' +
                    escapeHtml(f.name) + '</span>' +
                    '<button class="folder-entry-remove" data-id="' + escapeHtml(f.id) + '" title="Remove">&times;</button>' +
                    '</div>';
            }).join('');
            list.querySelectorAll('.folder-entry-remove').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    _removeGdriveFolder(btn.dataset.id);
                });
            });
        } catch (e) {
            list.innerHTML = '<div class="folder-empty">Failed to load</div>';
        }
    }

    async function _browseGdrive(folderId, folderName) {
        var dirsEl = $('#gdriveDirs');
        dirsEl.innerHTML = '<div class="folder-empty">Loading...</div>';

        _gdriveCurrent = { id: folderId, name: folderName };
        $('#gdriveSelectedName').textContent = folderName;
        $('#gdriveAddBtn').disabled = (folderId === 'root');

        // Update breadcrumb stack.
        if (folderId === 'root') {
            _gdriveBcStack = [{ id: 'root', name: 'My Drive' }];
        } else {
            var idx = _gdriveBcStack.findIndex(function(b) { return b.id === folderId; });
            if (idx >= 0) {
                _gdriveBcStack = _gdriveBcStack.slice(0, idx + 1);
            } else {
                _gdriveBcStack.push({ id: folderId, name: folderName });
            }
        }
        _renderGdriveBreadcrumb();

        try {
            var res = await fetch('/api/read-folders/gdrive/browse?folder_id=' + encodeURIComponent(folderId));
            var data = await res.json();
            if (data.error) {
                dirsEl.innerHTML = '<div class="folder-empty">' + escapeHtml(data.error) + '</div>';
                return;
            }
            var folders = data.folders || [];
            var sharedDrives = data.shared_drives || [];
            var html = '';
            if (_gdriveBcStack.length > 1) {
                var parent = _gdriveBcStack[_gdriveBcStack.length - 2];
                html += '<div class="folder-dir-item parent" data-gid="' + escapeHtml(parent.id) +
                    '" data-gname="' + escapeHtml(parent.name) + '">' +
                    '<span class="folder-dir-icon">\u2B06</span> ..</div>';
            }
            // Shared drives section (only at root).
            if (sharedDrives.length) {
                html += '<div class="folder-section-label" style="margin-top:4px">Shared Drives</div>';
                sharedDrives.forEach(function(d) {
                    html += '<div class="folder-dir-item" data-gid="' + escapeHtml(d.id) +
                        '" data-gname="' + escapeHtml(d.name) + '">' +
                        '<span class="folder-dir-icon">\uD83D\uDCE4</span> ' + escapeHtml(d.name) + '</div>';
                });
                if (folders.length) {
                    html += '<div class="folder-section-label" style="margin-top:4px">My Drive</div>';
                }
            }
            if (!folders.length && !sharedDrives.length && _gdriveBcStack.length <= 1) {
                html += '<div class="folder-empty">No subfolders</div>';
            }
            folders.forEach(function(f) {
                html += '<div class="folder-dir-item" data-gid="' + escapeHtml(f.id) +
                    '" data-gname="' + escapeHtml(f.name) + '">' +
                    '<span class="folder-dir-icon">\uD83D\uDCC1</span> ' + escapeHtml(f.name) + '</div>';
            });
            dirsEl.innerHTML = html;
            dirsEl.querySelectorAll('.folder-dir-item').forEach(function(el) {
                el.addEventListener('click', function() {
                    _browseGdrive(el.dataset.gid, el.dataset.gname);
                });
            });
        } catch (e) {
            dirsEl.innerHTML = '<div class="folder-empty">Error: ' + escapeHtml(e.message) + '</div>';
        }
    }

    function _renderGdriveBreadcrumb() {
        var bc = $('#gdriveBreadcrumb');
        var html = '';
        _gdriveBcStack.forEach(function(item, i) {
            if (i > 0) html += '<span class="folder-bc-sep">/</span>';
            html += '<span class="folder-bc-item" data-gid="' + escapeHtml(item.id) +
                '" data-gname="' + escapeHtml(item.name) + '">' + escapeHtml(item.name) + '</span>';
        });
        bc.innerHTML = html;
        bc.querySelectorAll('.folder-bc-item').forEach(function(el) {
            el.addEventListener('click', function() {
                _browseGdrive(el.dataset.gid, el.dataset.gname);
            });
        });
    }

    async function _addGdriveFolder() {
        if (_gdriveCurrent.id === 'root') return;
        var btn = $('#gdriveAddBtn');
        btn.disabled = true;
        btn.textContent = 'Adding...';
        try {
            var res = await fetch('/api/read-folders/gdrive', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: _gdriveCurrent.id, name: _gdriveCurrent.name }),
            });
            var data = await res.json();
            if (!res.ok || !data.ok) {
                addChatMessage('error', 'Add GDrive folder failed: ' + (data.error || 'Unknown error'));
            } else {
                addChatMessage('assistant', 'Google Drive folder added: `' + data.name + '`');
                _loadGdriveFolderList();
            }
        } catch (e) {
            addChatMessage('error', 'Add GDrive folder failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Add this folder';
        }
    }

    async function _removeGdriveFolder(id) {
        try {
            var res = await fetch('/api/read-folders/gdrive', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: id }),
            });
            var data = await res.json();
            if (!res.ok || !data.ok) {
                addChatMessage('error', 'Remove GDrive folder failed: ' + (data.error || 'Unknown error'));
            } else {
                addChatMessage('assistant', 'Google Drive folder removed');
                _loadGdriveFolderList();
            }
        } catch (e) {
            addChatMessage('error', 'Remove GDrive folder failed: ' + e.message);
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

        // Images: ![alt](path) — route local paths through /api/media
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_, alt, src) {
            var url = src;
            if (/^file:\/\/\//i.test(src)) {
                url = '/api/media?path=' + encodeURIComponent(src.replace(/^file:\/\/\//i, '/'));
            } else if (/^saved\/|^output\/|^\//.test(src)) {
                url = '/api/media?path=' + encodeURIComponent(src);
            }
            return '<img src="' + url + '" alt="' + alt +
                '" style="max-width:100%;border-radius:8px;cursor:pointer;" ' +
                'onclick="window.open(this.src,\'_blank\')">';
        });

        // Attached image: [Attached image: /path/to/file.jpg]
        html = html.replace(/\[Attached image: ([^\]]+)\]/g, function(_, path) {
            var url = '/api/media?path=' + encodeURIComponent(path.trim());
            return '<img src="' + url + '" alt="Attached image" ' +
                'style="max-width:100%;max-height:300px;border-radius:8px;cursor:pointer;display:block;margin:6px 0;" ' +
                'onclick="window.open(this.src,\'_blank\')">';
        });

        // Audio: [Audio: /path/to/file.mp3]
        html = html.replace(/\[Audio: ([^\]]+)\]/gi, function(_, path) {
            var p = path.trim();
            var url = /^https?:\/\//.test(p) ? p : '/api/media?path=' + encodeURIComponent(p);
            return '<audio controls style="display:block;width:100%;margin:6px 0;border-radius:8px;">' +
                '<source src="' + url + '">Your browser does not support audio.</audio>';
        });

        // Links to audio files: [text](path.mp3) → inline <audio> player
        html = html.replace(/\[([^\]]+)\]\(([^)]+\.(?:mp3|wav|ogg|flac|m4a|aac))\)/gi, function(_, label, src) {
            var url = /^https?:\/\//.test(src) ? src : '/api/media?path=' + encodeURIComponent(src);
            return '<div style="margin:6px 0;"><strong>' + label + '</strong>' +
                '<audio controls style="display:block;width:100%;margin:4px 0;border-radius:8px;">' +
                '<source src="' + url + '">Your browser does not support audio.</audio></div>';
        });

        // Auto-detect bare audio file paths → inline <audio> player.
        // Optional surrounding backticks are consumed so the inline-code
        // regex that runs later won't wrap the <audio> tag in <code>.
        html = html.replace(/`?((?:\/|saved\/|output\/)[^\s<>&"'`]+\.(?:mp3|wav|ogg|flac|m4a|aac))`?/gi, function(_, path) {
            var url = '/api/media?path=' + encodeURIComponent(path);
            return '<audio controls style="display:block;width:100%;margin:6px 0;border-radius:8px;">' +
                '<source src="' + url + '">Your browser does not support audio.</audio>';
        });

        // Auto-detect bare image file paths → inline <img>.
        html = html.replace(/(?:\*\*|`)?(?:file:\/\/\/)?((?:\/|saved\/|output\/)[^\s<>&"'`*]+\.(?:png|jpe?g|gif|webp|bmp|svg))(?:\*\*|`)?/gi, function(_, path) {
            var url = '/api/media?path=' + encodeURIComponent(path);
            return '<img src="' + url + '" alt="Image" ' +
                'style="max-width:100%;border-radius:8px;cursor:pointer;display:block;margin:8px 0;" ' +
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

        // Tables
        html = html.replace(/((?:^\|.+\|$\n?){2,})/gm, function(block) {
            var rows = block.trim().split('\n');
            if (rows.length < 2) return block;
            var parseRow = function(row) {
                return row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
            };
            // Check if second row is a separator (e.g. | :--- | :--- |)
            var sepCells = parseRow(rows[1]);
            var isSep = sepCells.every(function(c) { return /^:?-{1,}:?$/.test(c); });
            if (!isSep) return block;
            var headCells = parseRow(rows[0]);
            var thead = '<thead><tr>' + headCells.map(function(c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead>';
            var bodyRows = rows.slice(2);
            var tbody = '<tbody>' + bodyRows.map(function(row) {
                var cells = parseRow(row);
                return '<tr>' + cells.map(function(c) { return '<td>' + c + '</td>'; }).join('') + '</tr>';
            }).join('') + '</tbody>';
            return '<table>' + thead + tbody + '</table>';
        });

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

        // Force script mode toggle
        var forceScriptToggle = $('#forceScriptToggle');
        if (forceScriptToggle) {
            // Restore from sessionStorage
            var stored = sessionStorage.getItem('forceScriptMode');
            if (stored === 'true') {
                forceScriptMode = true;
                forceScriptToggle.checked = true;
            }
            forceScriptToggle.addEventListener('change', function () {
                forceScriptMode = forceScriptToggle.checked;
                sessionStorage.setItem('forceScriptMode', forceScriptMode ? 'true' : 'false');
                send({ type: 'set_force_script', enabled: forceScriptMode });
            });
        }

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

            // Ctrl+Shift+M - toggle voice recording
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
                e.preventDefault();
                toggleMic();
                return;
            }

            // Escape - close modals/palette
            if (e.key === 'Escape') {
                if (!approvalModal.classList.contains('hidden')) return; // Don't close approval modal with Esc
                if (!personaDropdown.classList.contains('hidden')) { hidePersonaDropdown(); return; }
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
            hidePersonaDropdown();
            hidePlaybookDropdown();
            toggleModelDropdown();
        });
        // Close model dropdown on outside click
        document.addEventListener('click', (e) => {
            if (!modelDropdown.contains(e.target) && e.target !== modelBadge) {
                hideModelDropdown();
            }
        });

        // Persona badge toggles persona selector dropdown
        personaBadge.addEventListener('click', (e) => {
            e.stopPropagation();
            hideModelDropdown();
            hidePlaybookDropdown();
            togglePersonaDropdown();
        });
        // Close persona dropdown on outside click
        document.addEventListener('click', (e) => {
            if (!personaDropdown.contains(e.target) && e.target !== personaBadge) {
                hidePersonaDropdown();
            }
        });

        // Playbook badge toggles playbook selector dropdown
        playbookBadge.addEventListener('click', (e) => {
            e.stopPropagation();
            hideModelDropdown();
            hidePersonaDropdown();
            togglePlaybookDropdown();
        });
        // Close playbook dropdown on outside click
        document.addEventListener('click', (e) => {
            if (!playbookDropdown.contains(e.target) && e.target !== playbookBadge) {
                hidePlaybookDropdown();
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

        // Mic button
        if (micBtn) micBtn.addEventListener('click', toggleMic);

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

        // File explorer
        filesSearch.addEventListener('input', () => renderFilesList(sessionFiles));
        $('#filePreviewClose').addEventListener('click', closeFilePreview);
        fileMdViewBtn.addEventListener('click', openMdViewer);
        $('#mdViewerClose').addEventListener('click', closeMdViewer);
        $('#mdExportPdf').addEventListener('click', exportMdAsPdf);
        $('#mdExportDocx').addEventListener('click', exportMdAsDocx);
        $('#mdViewerModal').addEventListener('click', function(e) {
            if (e.target === e.currentTarget) closeMdViewer();
        });

        // Folder browser
        $('#skillsFolderBtn').addEventListener('click', openFolderModal);
        $('#folderModalClose').addEventListener('click', closeFolderModal);
        $('#folderAddBtn').addEventListener('click', addCurrentFolder);
        $('#gdriveAddBtn').addEventListener('click', _addGdriveFolder);
        $('#folderModal').addEventListener('click', function(e) {
            if (e.target === e.currentTarget) closeFolderModal();
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
