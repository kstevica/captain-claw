(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────────────
    var messages = [];      // {role: 'assistant'|'user', content: string}
    var currentPhase = 'outcome';
    var isBusy = false;
    var playbookData = null; // populated when LLM returns type=playbook

    // ── DOM refs ─────────────────────────────────────────────────────
    var elConversation = document.getElementById('wizConversation');
    var elPreviewCard = document.getElementById('wizPreviewCard');
    var elInputArea = document.getElementById('wizInputArea');
    var elUserInput = document.getElementById('wizUserInput');
    var elSendBtn = document.getElementById('wizSendBtn');
    var elSaveBtn = document.getElementById('wizSaveBtn');
    var elStartOver = document.getElementById('wizStartOver');
    var elPhases = document.getElementById('wizPhases');
    var elToast = document.getElementById('wizToast');

    // Preview fields
    var prevFields = {
        name: document.getElementById('prevName'),
        task_type: document.getElementById('prevTaskType'),
        trigger_description: document.getElementById('prevTrigger'),
        do_pattern: document.getElementById('prevDoPattern'),
        dont_pattern: document.getElementById('prevDontPattern'),
        examples: document.getElementById('prevExamples'),
        reasoning: document.getElementById('prevReasoning'),
        tags: document.getElementById('prevTags'),
    };

    // ── Init ─────────────────────────────────────────────────────────
    function init() {
        elSendBtn.addEventListener('click', onSend);
        elSaveBtn.addEventListener('click', onSave);
        elStartOver.addEventListener('click', onStartOver);
        elUserInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                onSend();
            }
        });
        elUserInput.addEventListener('input', autoResize);

        // Start the wizard with the first assistant message.
        appendAssistantMessage('What outcome do you want this playbook to achieve?\n\nDescribe the task or workflow you want to capture as a reusable pattern.');
        messages.push({
            role: 'assistant',
            content: 'What outcome do you want this playbook to achieve?\n\nDescribe the task or workflow you want to capture as a reusable pattern.',
        });
        elUserInput.focus();
    }

    // ── Send user message ────────────────────────────────────────────
    function onSend() {
        if (isBusy) return;
        var text = elUserInput.value.trim();
        if (!text) return;

        // Add user message.
        messages.push({ role: 'user', content: text });
        appendUserMessage(text);
        elUserInput.value = '';
        autoResize();
        setBusy(true);

        // Show typing indicator.
        var typing = showTyping();

        // Call backend.
        apiFetch('/api/playbook-wizard/step', {
            method: 'POST',
            body: { messages: messages },
        })
            .then(function (data) {
                removeTyping(typing);
                handleResponse(data);
            })
            .catch(function (err) {
                removeTyping(typing);
                appendAssistantMessage('Sorry, something went wrong. Please try again.');
                showToast(err.message || 'Request failed', 'error');
            })
            .finally(function () {
                setBusy(false);
            });
    }

    // ── Handle LLM response ─────────────────────────────────────────
    function handleResponse(data) {
        if (data.error) {
            appendAssistantMessage('Error: ' + data.error);
            return;
        }

        if (data.type === 'question') {
            // Update phase.
            if (data.phase) setPhase(data.phase);

            var content = data.content || 'Could you tell me more?';
            messages.push({ role: 'assistant', content: content });
            appendAssistantMessage(content);
            elUserInput.focus();
        } else if (data.type === 'playbook') {
            playbookData = data.playbook || {};
            setPhase('review');
            showPreview(playbookData);
        } else {
            // Unknown type — treat as question.
            var fallback = data.content || JSON.stringify(data);
            messages.push({ role: 'assistant', content: fallback });
            appendAssistantMessage(fallback);
        }
    }

    // ── Preview card ─────────────────────────────────────────────────
    function showPreview(pb) {
        prevFields.name.value = pb.name || '';
        prevFields.task_type.value = pb.task_type || 'other';
        prevFields.trigger_description.value = pb.trigger_description || '';
        prevFields.do_pattern.value = pb.do_pattern || '';
        prevFields.dont_pattern.value = pb.dont_pattern || '';
        prevFields.examples.value = pb.examples || '';
        prevFields.reasoning.value = pb.reasoning || '';
        prevFields.tags.value = pb.tags || '';

        elPreviewCard.style.display = '';
        elInputArea.classList.add('hidden');
        elPreviewCard.scrollIntoView({ behavior: 'smooth' });
    }

    // ── Save playbook ────────────────────────────────────────────────
    function onSave() {
        var payload = {
            name: prevFields.name.value.trim(),
            task_type: prevFields.task_type.value,
            rating: 'good',
            trigger_description: prevFields.trigger_description.value.trim(),
            do_pattern: prevFields.do_pattern.value.trim(),
            dont_pattern: prevFields.dont_pattern.value.trim(),
            examples: prevFields.examples.value.trim(),
            reasoning: prevFields.reasoning.value.trim(),
            tags: prevFields.tags.value.trim(),
        };

        if (!payload.name) {
            showToast('Name is required', 'error');
            return;
        }
        if (!payload.do_pattern && !payload.dont_pattern) {
            showToast('At least one pattern (DO or DON\'T) is required', 'error');
            return;
        }

        elSaveBtn.disabled = true;
        elSaveBtn.textContent = 'Saving...';

        apiFetch('/api/playbooks', { method: 'POST', body: payload })
            .then(function () {
                showToast('Playbook saved successfully!', 'success');
                setTimeout(function () {
                    // Offer to create another or go to playbooks.
                    elSaveBtn.textContent = 'Saved!';
                    elSaveBtn.disabled = true;
                    elStartOver.textContent = 'Create Another';
                }, 500);
            })
            .catch(function (err) {
                showToast('Save failed: ' + (err.message || 'Unknown error'), 'error');
                elSaveBtn.disabled = false;
                elSaveBtn.textContent = 'Save Playbook';
            });
    }

    // ── Start over ───────────────────────────────────────────────────
    function onStartOver() {
        messages = [];
        playbookData = null;
        currentPhase = 'outcome';
        elConversation.innerHTML = '';
        elPreviewCard.style.display = 'none';
        elInputArea.classList.remove('hidden');
        elSaveBtn.disabled = false;
        elSaveBtn.textContent = 'Save Playbook';
        elStartOver.textContent = 'Start Over';
        updatePhaseUI();

        appendAssistantMessage('What outcome do you want this playbook to achieve?\n\nDescribe the task or workflow you want to capture as a reusable pattern.');
        messages.push({
            role: 'assistant',
            content: 'What outcome do you want this playbook to achieve?\n\nDescribe the task or workflow you want to capture as a reusable pattern.',
        });
        elUserInput.focus();
    }

    // ── DOM helpers ──────────────────────────────────────────────────
    function appendAssistantMessage(text) {
        var el = document.createElement('div');
        el.className = 'wiz-msg assistant';
        el.innerHTML = renderMarkdown(text);
        elConversation.appendChild(el);
        scrollToBottom();
    }

    function appendUserMessage(text) {
        var el = document.createElement('div');
        el.className = 'wiz-msg user';
        el.textContent = text;
        elConversation.appendChild(el);
        scrollToBottom();
    }

    function showTyping() {
        var el = document.createElement('div');
        el.className = 'wiz-typing';
        el.innerHTML = '<span class="wiz-typing-dot"></span><span class="wiz-typing-dot"></span><span class="wiz-typing-dot"></span>';
        elConversation.appendChild(el);
        scrollToBottom();
        return el;
    }

    function removeTyping(el) {
        if (el && el.parentNode) el.parentNode.removeChild(el);
    }

    function scrollToBottom() {
        var main = document.querySelector('.wiz-main');
        requestAnimationFrame(function () {
            main.scrollTop = main.scrollHeight;
        });
    }

    function autoResize() {
        elUserInput.style.height = 'auto';
        elUserInput.style.height = Math.min(elUserInput.scrollHeight, 120) + 'px';
    }

    function setBusy(busy) {
        isBusy = busy;
        elSendBtn.disabled = busy;
        elUserInput.disabled = busy;
        if (!busy) elUserInput.focus();
    }

    // ── Phase indicator ──────────────────────────────────────────────
    function setPhase(phase) {
        var order = ['outcome', 'details', 'review'];
        var idx = order.indexOf(phase);
        if (idx === -1) return;
        currentPhase = phase;
        updatePhaseUI();
    }

    function updatePhaseUI() {
        var order = ['outcome', 'details', 'review'];
        var idx = order.indexOf(currentPhase);
        var spans = elPhases.querySelectorAll('.wiz-phase');
        spans.forEach(function (span, i) {
            span.classList.remove('active', 'done');
            if (i < idx) span.classList.add('done');
            else if (i === idx) span.classList.add('active');
        });
    }

    // ── Simple markdown rendering ────────────────────────────────────
    function renderMarkdown(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    // ── Toast ────────────────────────────────────────────────────────
    function showToast(msg, type) {
        elToast.textContent = msg;
        elToast.className = 'wiz-toast show ' + (type || '');
        clearTimeout(showToast._timer);
        showToast._timer = setTimeout(function () {
            elToast.className = 'wiz-toast';
        }, 3500);
    }

    // ── API helper ───────────────────────────────────────────────────
    function apiFetch(url, options) {
        var opts = {
            method: (options && options.method) || 'GET',
            headers: { 'Content-Type': 'application/json' },
        };
        if (options && options.body) {
            opts.body = JSON.stringify(options.body);
        }
        return fetch(url, opts).then(function (res) {
            return res.json().then(function (data) {
                if (!res.ok) throw new Error(data.error || 'HTTP ' + res.status);
                return data;
            });
        });
    }

    // ── Boot ─────────────────────────────────────────────────────────
    init();
})();
