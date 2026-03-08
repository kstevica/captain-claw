/* Captain Claw – Onboarding Wizard */

(function () {
    'use strict';

    var PROVIDERS = [
        { id: 'openai',    name: 'OpenAI / ChatGPT',           model: 'gpt-4.1-mini',                  env: 'OPENAI_API_KEY' },
        { id: 'anthropic', name: 'Anthropic / Claude',          model: 'claude-sonnet-4-20250514',       env: 'ANTHROPIC_API_KEY' },
        { id: 'gemini',    name: 'Google / Gemini',             model: 'gemini-3-flash-preview',         env: 'GOOGLE_API_KEY' },
        { id: 'xai',       name: 'xAI / Grok',                 model: 'grok-3-mini',                    env: 'XAI_API_KEY' },
        { id: 'ollama',    name: 'Ollama (local/self-hosted)',   model: 'llama3.2',                       env: '' }
    ];

    // Provider accent colours for key cards
    var PROVIDER_COLORS = {
        openai:    '#10a37f',
        anthropic: '#d4a574',
        gemini:    '#4285f4',
        xai:       '#e44d26',
        brave:     '#fb542b'
    };

    var TOTAL_STEPS = 10; // 0-9
    var currentStep = 0;
    var state = {
        provider: 'openai',
        model: '',
        api_key: '',
        base_url: '',
        use_env: false,
        validated: false,
        enable_guards: false,
        extra_models: [],  // {provider, model, api_key}
        telegram_enabled: false,
        telegram_token: '',
        telegram_use_env: false,
        // New: per-provider keys
        provider_keys: { openai: '', anthropic: '', gemini: '', xai: '', brave: '' },
        openai_headers: [],     // ['Authorization: Bearer ...', 'chatgpt-account-id: ...']
        codex_imported: false
    };

    // ── Initialise ────────────────────────────────────────

    function init() {
        buildProgressDots();
        buildProviderGrid();
        buildApiKeysStep();
        showStep(0);

        document.getElementById('use-env-var').addEventListener('change', function () {
            state.use_env = this.checked;
            document.getElementById('api-key').disabled = this.checked;
            if (this.checked) document.getElementById('api-key').value = '';
        });

        document.getElementById('enable-telegram').addEventListener('change', function () {
            state.telegram_enabled = this.checked;
            document.getElementById('telegram-token-section').style.display = this.checked ? '' : 'none';
        });

        document.getElementById('telegram-use-env').addEventListener('change', function () {
            state.telegram_use_env = this.checked;
            document.getElementById('telegram-token').disabled = this.checked;
            if (this.checked) document.getElementById('telegram-token').value = '';
        });
    }

    function buildProgressDots() {
        var el = document.getElementById('progress');
        el.innerHTML = '';
        for (var i = 0; i < TOTAL_STEPS; i++) {
            var dot = document.createElement('div');
            dot.className = 'ob-dot';
            dot.dataset.idx = i;
            el.appendChild(dot);
        }
    }

    function updateDots() {
        var dots = document.querySelectorAll('.ob-dot');
        dots.forEach(function (d) {
            var idx = parseInt(d.dataset.idx, 10);
            d.className = 'ob-dot';
            if (idx < currentStep) d.classList.add('done');
            if (idx === currentStep) d.classList.add('active');
        });
    }

    function showStep(n) {
        currentStep = n;
        var steps = document.querySelectorAll('.ob-step');
        steps.forEach(function (s) { s.classList.remove('visible'); });
        var target = document.querySelector('.ob-step[data-step="' + n + '"]');
        if (target) target.classList.add('visible');
        updateDots();

        // Hook: populate fields when entering certain steps
        if (n === 1) refreshApiKeysStatus();
        if (n === 3) populateModelStep();
        if (n === 8) populateSummary();
    }

    // ── API Keys step (step 1) ────────────────────────────

    var API_KEY_PROVIDERS = [
        {
            id: 'openai',
            name: 'OpenAI',
            placeholder: 'sk-...',
            hint: 'Env var: OPENAI_API_KEY',
            extra: 'codex'  // special: show Import from Codex button
        },
        {
            id: 'anthropic',
            name: 'Anthropic',
            placeholder: 'sk-ant-oat-... or sk-ant-api-...',
            hint: 'Env var: ANTHROPIC_API_KEY',
            extra: 'claude-tip'  // special: show setup-token callout
        },
        {
            id: 'gemini',
            name: 'Google Gemini',
            placeholder: 'AI...',
            hint: 'Env var: GOOGLE_API_KEY',
            extra: null
        },
        {
            id: 'xai',
            name: 'xAI (Grok)',
            placeholder: 'xai-...',
            hint: 'Env var: XAI_API_KEY',
            extra: null
        },
        {
            id: 'brave',
            name: 'Brave Search',
            placeholder: 'BSA...',
            hint: 'Optional \u2014 enables the web search tool',
            extra: null
        }
    ];

    function buildApiKeysStep() {
        var container = document.getElementById('api-keys-list');
        container.innerHTML = '';

        API_KEY_PROVIDERS.forEach(function (prov) {
            var color = PROVIDER_COLORS[prov.id] || 'var(--ob-accent)';
            var card = document.createElement('div');
            card.className = 'ob-key-card';
            card.dataset.provider = prov.id;
            card.style.borderLeftColor = color;

            var header = '<div class="ob-key-card-header">' +
                '<span class="ob-key-card-name">' + esc(prov.name) + '</span>' +
                '<span class="ob-key-card-badge" id="key-badge-' + prov.id + '"></span>' +
                '</div>';

            var input = '<div class="ob-field" style="margin-bottom:8px;">' +
                '<input type="password" id="pk-' + prov.id + '" placeholder="' + esc(prov.placeholder) + '" autocomplete="off">' +
                '<div class="ob-hint">' + esc(prov.hint) + '</div>' +
                '</div>';

            var extra = '';
            if (prov.extra === 'codex') {
                extra = '<div class="ob-key-codex-row">' +
                    '<button class="ob-key-import-btn" id="codex-import-btn" onclick="OB.importCodex()">' +
                    '<span class="ob-key-import-icon">&#8631;</span> Import from Codex CLI' +
                    '</button>' +
                    '<span class="ob-key-import-status" id="codex-import-status"></span>' +
                    '</div>';
            } else if (prov.extra === 'claude-tip') {
                extra = '<div class="ob-key-callout">' +
                    '<div class="ob-key-callout-icon">&#128161;</div>' +
                    '<div class="ob-key-callout-text">' +
                    'Run <code>claude setup-token</code> to get an auth key ' +
                    '(<code>sk-ant-oat...</code>) instead of an API key (<code>sk-ant-api...</code>).<br>' +
                    'Auth keys use your Claude Pro/Max subscription &mdash; no separate API billing.' +
                    '</div>' +
                    '</div>';
            }

            card.innerHTML = header + input + extra;
            container.appendChild(card);

            // Wire up input change → update badge
            var inp = card.querySelector('input');
            inp.addEventListener('input', function () {
                state.provider_keys[prov.id] = this.value.trim();
                updateKeyBadge(prov.id);
            });
        });
    }

    function updateKeyBadge(providerId) {
        var badge = document.getElementById('key-badge-' + providerId);
        if (!badge) return;
        var hasKey = !!state.provider_keys[providerId];
        var hasHeaders = providerId === 'openai' && state.codex_imported;
        if (hasKey || hasHeaders) {
            badge.textContent = '\u2713';
            badge.className = 'ob-key-card-badge configured';
        } else {
            badge.textContent = '';
            badge.className = 'ob-key-card-badge';
        }
        // Update card border state
        var card = badge.closest('.ob-key-card');
        if (card) {
            card.classList.toggle('configured', hasKey || hasHeaders);
        }
    }

    function refreshApiKeysStatus() {
        // Re-populate inputs from state (in case user goes back)
        API_KEY_PROVIDERS.forEach(function (prov) {
            var inp = document.getElementById('pk-' + prov.id);
            if (inp && state.provider_keys[prov.id]) {
                inp.value = state.provider_keys[prov.id];
            }
            updateKeyBadge(prov.id);
        });
    }

    function readApiKeysStep() {
        API_KEY_PROVIDERS.forEach(function (prov) {
            var inp = document.getElementById('pk-' + prov.id);
            if (inp) {
                state.provider_keys[prov.id] = inp.value.trim();
            }
        });
    }

    function importCodexAuth() {
        var btn = document.getElementById('codex-import-btn');
        var statusEl = document.getElementById('codex-import-status');
        btn.disabled = true;
        statusEl.textContent = 'Reading...';
        statusEl.className = 'ob-key-import-status';

        fetch('/api/onboarding/codex-auth')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                btn.disabled = false;
                if (data.ok) {
                    // Build headers
                    var headers = [];
                    headers.push('Authorization: Bearer ' + data.access_token);
                    if (data.account_id) {
                        headers.push('chatgpt-account-id: ' + data.account_id);
                    }
                    state.openai_headers = headers;
                    state.codex_imported = true;
                    statusEl.textContent = 'Codex auth imported \u2713';
                    statusEl.className = 'ob-key-import-status ok';
                    updateKeyBadge('openai');
                } else {
                    statusEl.textContent = data.error || 'Import failed';
                    statusEl.className = 'ob-key-import-status fail';
                    state.codex_imported = false;
                }
            })
            .catch(function (err) {
                btn.disabled = false;
                statusEl.textContent = 'Network error: ' + err.message;
                statusEl.className = 'ob-key-import-status fail';
                state.codex_imported = false;
            });
    }

    // ── Provider grid (step 2) ────────────────────────────

    function buildProviderGrid() {
        var grid = document.getElementById('provider-grid');
        grid.innerHTML = '';
        PROVIDERS.forEach(function (p) {
            var card = document.createElement('div');
            card.className = 'ob-provider-card' + (p.id === state.provider ? ' selected' : '');
            card.dataset.provider = p.id;
            card.innerHTML =
                '<div class="ob-prov-name">' + p.name + '</div>' +
                '<div class="ob-prov-model">' + p.model + '</div>' +
                (p.env ? '<div class="ob-prov-env">' + p.env + '</div>' : '<div class="ob-prov-env">(no key needed)</div>');
            card.onclick = function () { selectProvider(p.id); };
            grid.appendChild(card);
        });
    }

    function selectProvider(id) {
        state.provider = id;
        document.querySelectorAll('.ob-provider-card').forEach(function (c) {
            c.classList.toggle('selected', c.dataset.provider === id);
        });
    }

    // ── Model + Key step (step 3) ─────────────────────────

    function populateModelStep() {
        var prov = findProvider(state.provider);
        var modelInput = document.getElementById('model-name');
        var keyInput = document.getElementById('api-key');

        if (!state.model || state.model === findDefaultModel(state.provider)) {
            // Reset to provider default
        }
        modelInput.value = state.model || prov.model;
        document.getElementById('model-hint').textContent = 'Default: ' + prov.model;

        var isOllama = state.provider === 'ollama';
        document.getElementById('api-key-section').style.display = isOllama ? 'none' : '';
        document.getElementById('base-url-section').style.display = isOllama ? '' : 'none';

        if (!isOllama) {
            // Pre-fill from provider_keys if user entered one in step 1
            var providerKey = state.provider_keys[state.provider] || '';
            keyInput.value = state.api_key || providerKey;
            document.getElementById('key-hint').textContent = prov.env ? 'Env var: ' + prov.env : '';

            // If provider has a key from step 1 or codex headers, suggest env var toggle
            var hasProviderKey = !!providerKey || (state.provider === 'openai' && state.codex_imported);
            if (hasProviderKey && !state.api_key) {
                document.getElementById('use-env-var').checked = false;
                state.use_env = false;
                keyInput.disabled = false;
                if (providerKey) keyInput.value = providerKey;
            } else {
                document.getElementById('use-env-var').checked = state.use_env;
                keyInput.disabled = state.use_env;
            }
        } else {
            document.getElementById('base-url').value = state.base_url || 'http://127.0.0.1:11434';
        }
    }

    function readModelStep() {
        state.model = document.getElementById('model-name').value.trim();
        if (state.provider === 'ollama') {
            state.base_url = document.getElementById('base-url').value.trim();
            state.api_key = '';
        } else {
            state.use_env = document.getElementById('use-env-var').checked;
            state.api_key = state.use_env ? '' : document.getElementById('api-key').value.trim();
            state.base_url = '';
        }
    }

    // ── Validation step (step 4) ──────────────────────────

    function runValidation() {
        var spinner = document.getElementById('val-spinner');
        var statusEl = document.getElementById('val-status');
        var actionsEl = document.getElementById('val-actions');
        var nextBtn = document.getElementById('val-next-btn');

        spinner.style.display = '';
        statusEl.className = 'ob-status';
        statusEl.textContent = 'Connecting to provider...';
        actionsEl.style.display = 'none';

        fetch('/api/onboarding/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: state.provider,
                model: state.model,
                api_key: state.api_key,
                base_url: state.base_url
            })
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            spinner.style.display = 'none';
            actionsEl.style.display = '';
            if (data.ok) {
                statusEl.className = 'ob-status ok';
                statusEl.textContent = 'Connection successful!';
                state.validated = true;
                nextBtn.textContent = 'Next';
            } else {
                statusEl.className = 'ob-status fail';
                statusEl.textContent = 'Failed: ' + (data.error || 'Unknown error');
                state.validated = false;
                nextBtn.textContent = 'Continue Anyway';
            }
        })
        .catch(function (err) {
            spinner.style.display = 'none';
            actionsEl.style.display = '';
            statusEl.className = 'ob-status fail';
            statusEl.textContent = 'Network error: ' + err.message;
            state.validated = false;
            nextBtn.textContent = 'Continue Anyway';
        });
    }

    // ── Extra models (step 5) ─────────────────────────────

    var MAX_EXTRA = 3;

    function addExtraModel() {
        if (state.extra_models.length >= MAX_EXTRA) return;
        state.extra_models.push({ provider: 'openai', model: '', api_key: '', status: '' });
        renderExtraModels();
    }

    function removeExtraModel(idx) {
        state.extra_models.splice(idx, 1);
        renderExtraModels();
    }

    function renderExtraModels() {
        var list = document.getElementById('extra-models-list');
        var btn = document.getElementById('add-model-btn');
        list.innerHTML = '';

        state.extra_models.forEach(function (m, idx) {
            var prov = findProvider(m.provider);
            var isOllama = m.provider === 'ollama';
            var entry = document.createElement('div');
            entry.className = 'ob-extra-model-entry';

            var provBtns = PROVIDERS.map(function (p) {
                return '<button class="ob-mini-prov' + (p.id === m.provider ? ' selected' : '') +
                       '" data-idx="' + idx + '" data-prov="' + p.id + '">' + p.id + '</button>';
            }).join('');

            entry.innerHTML =
                '<button class="ob-remove-model" data-idx="' + idx + '">&times;</button>' +
                '<div class="ob-mini-providers">' + provBtns + '</div>' +
                '<div class="ob-extra-inline">' +
                    '<input type="text" placeholder="Model name" value="' + (m.model || prov.model) + '" data-idx="' + idx + '" data-field="model">' +
                    (isOllama ? '' : '<input type="password" placeholder="API key" value="' + (m.api_key || '') + '" data-idx="' + idx + '" data-field="api_key">') +
                '</div>' +
                '<div class="ob-extra-status ' + (m.status || '') + '" data-idx="' + idx + '">' +
                    (m.status === 'ok' ? 'Connected' : m.status === 'fail' ? 'Validation failed' : '') +
                '</div>';

            list.appendChild(entry);
        });

        // Wire up events
        list.querySelectorAll('.ob-remove-model').forEach(function (b) {
            b.onclick = function () { removeExtraModel(parseInt(b.dataset.idx, 10)); };
        });
        list.querySelectorAll('.ob-mini-prov').forEach(function (b) {
            b.onclick = function () {
                var i = parseInt(b.dataset.idx, 10);
                state.extra_models[i].provider = b.dataset.prov;
                state.extra_models[i].model = '';
                state.extra_models[i].api_key = '';
                state.extra_models[i].status = '';
                renderExtraModels();
            };
        });
        list.querySelectorAll('input[data-field]').forEach(function (inp) {
            inp.onchange = function () {
                var i = parseInt(inp.dataset.idx, 10);
                state.extra_models[i][inp.dataset.field] = inp.value.trim();
            };
        });

        btn.style.display = state.extra_models.length >= MAX_EXTRA ? 'none' : '';
    }

    function readExtraModels() {
        // Ensure we read the latest input values
        document.querySelectorAll('#extra-models-list input[data-field]').forEach(function (inp) {
            var i = parseInt(inp.dataset.idx, 10);
            if (state.extra_models[i]) {
                state.extra_models[i][inp.dataset.field] = inp.value.trim();
            }
        });
        // Fill missing model names with defaults
        state.extra_models.forEach(function (m) {
            if (!m.model) m.model = findProvider(m.provider).model;
        });
    }

    // ── Summary (step 8) ─────────────────────────────────

    function readTelegramStep() {
        state.telegram_enabled = document.getElementById('enable-telegram').checked;
        state.telegram_use_env = document.getElementById('telegram-use-env').checked;
        state.telegram_token = state.telegram_use_env ? '' : document.getElementById('telegram-token').value.trim();
    }

    function populateSummary() {
        readApiKeysStep();
        readExtraModels();
        readTelegramStep();
        state.enable_guards = document.getElementById('enable-guards').checked;

        var rows = [
            ['Provider', findProvider(state.provider).name],
            ['Model', state.model || findProvider(state.provider).model],
            ['API key', state.api_key ? 'stored in config' : 'via environment variable'],
        ];

        // Show which provider keys are configured
        var configuredKeys = [];
        Object.keys(state.provider_keys).forEach(function (k) {
            if (state.provider_keys[k]) configuredKeys.push(k);
        });
        if (state.codex_imported) {
            if (configuredKeys.indexOf('openai') === -1) configuredKeys.push('openai (Codex OAuth)');
        }
        if (configuredKeys.length > 0) {
            rows.push(['Provider keys', configuredKeys.join(', ')]);
        }

        if (state.base_url) rows.push(['Base URL', state.base_url]);
        rows.push(['Safety guards', state.enable_guards ? 'enabled (ask_for_approval)' : 'disabled']);
        rows.push(['Pre-configured models', '12 (OpenAI, Anthropic, Gemini + image/OCR/vision)']);

        state.extra_models.forEach(function (m, i) {
            rows.push(['Extra model #' + (i + 1), m.provider + ' / ' + (m.model || findProvider(m.provider).model)]);
        });

        if (state.telegram_enabled) {
            rows.push(['Telegram', state.telegram_token ? 'enabled (token stored)' : 'enabled (env var)']);
        } else {
            rows.push(['Telegram', 'disabled']);
        }

        var table = document.getElementById('summary-table');
        table.innerHTML = rows.map(function (r) {
            return '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>';
        }).join('');
    }

    // ── Save (step 8) ────────────────────────────────────

    function save() {
        var btn = document.getElementById('save-btn');
        btn.disabled = true;
        btn.textContent = 'Saving...';

        readApiKeysStep();
        readExtraModels();
        readTelegramStep();
        state.enable_guards = document.getElementById('enable-guards').checked;

        var allowed = state.extra_models.map(function (m) {
            return { provider: m.provider, model: m.model || findProvider(m.provider).model, api_key: m.api_key || '' };
        });

        fetch('/api/onboarding/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: state.provider,
                model: state.model || findProvider(state.provider).model,
                api_key: state.api_key,
                base_url: state.base_url,
                enable_guards: state.enable_guards,
                allowed_models: allowed,
                telegram_enabled: state.telegram_enabled,
                telegram_token: state.telegram_token,
                // New: provider keys
                provider_keys: state.provider_keys,
                openai_headers: state.openai_headers
            })
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.ok) {
                showStep(9);
            } else {
                btn.disabled = false;
                btn.textContent = 'Save & Launch';
                alert('Save failed: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(function (err) {
            btn.disabled = false;
            btn.textContent = 'Save & Launch';
            alert('Network error: ' + err.message);
        });
    }

    // ── Helpers ────────────────────────────────────────────

    function findProvider(id) {
        for (var i = 0; i < PROVIDERS.length; i++) {
            if (PROVIDERS[i].id === id) return PROVIDERS[i];
        }
        return PROVIDERS[0];
    }

    function findDefaultModel(id) {
        return findProvider(id).model;
    }

    function esc(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Navigation ────────────────────────────────────────

    function next() {
        // Read API keys before leaving step 1
        if (currentStep === 1) {
            readApiKeysStep();
        }
        // From model step → validation
        if (currentStep === 3) {
            readModelStep();
            showStep(4);
            runValidation();
            return;
        }
        if (currentStep < TOTAL_STEPS - 1) {
            showStep(currentStep + 1);
        }
    }

    function prev() {
        // From validation → back to model step
        if (currentStep === 4) {
            showStep(3);
            return;
        }
        if (currentStep > 0) {
            showStep(currentStep - 1);
        }
    }

    function validate() {
        readModelStep();
        showStep(4);
        runValidation();
    }

    // ── Public API ────────────────────────────────────────

    window.OB = {
        next: next,
        prev: prev,
        validate: validate,
        save: save,
        addExtraModel: addExtraModel,
        importCodex: importCodexAuth
    };

    // Boot
    init();
})();
