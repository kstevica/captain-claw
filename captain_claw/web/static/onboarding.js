/* Captain Claw – Onboarding Wizard */

(function () {
    'use strict';

    var PROVIDERS = [
        { id: 'openai',    name: 'OpenAI / ChatGPT',           model: 'gpt-4.1-mini',                  env: 'OPENAI_API_KEY' },
        { id: 'anthropic', name: 'Anthropic / Claude',          model: 'claude-sonnet-4-20250514',       env: 'ANTHROPIC_API_KEY' },
        { id: 'gemini',    name: 'Google / Gemini',             model: 'gemini-3-flash-preview',         env: 'GOOGLE_API_KEY' },
        { id: 'ollama',    name: 'Ollama (local/self-hosted)',   model: 'llama3.2',                       env: '' }
    ];

    var TOTAL_STEPS = 9; // 0-8
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
        telegram_use_env: false
    };

    // ── Initialise ────────────────────────────────────────

    function init() {
        buildProgressDots();
        buildProviderGrid();
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
        if (n === 2) populateModelStep();
        if (n === 7) populateSummary();
    }

    // ── Provider grid ─────────────────────────────────────

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

    // ── Model + Key step ──────────────────────────────────

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
            keyInput.value = state.api_key;
            document.getElementById('key-hint').textContent = prov.env ? 'Env var: ' + prov.env : '';
            document.getElementById('use-env-var').checked = state.use_env;
            keyInput.disabled = state.use_env;
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

    // ── Validation step ───────────────────────────────────

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

    // ── Extra models ──────────────────────────────────────

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

    // ── Summary ───────────────────────────────────────────

    function readTelegramStep() {
        state.telegram_enabled = document.getElementById('enable-telegram').checked;
        state.telegram_use_env = document.getElementById('telegram-use-env').checked;
        state.telegram_token = state.telegram_use_env ? '' : document.getElementById('telegram-token').value.trim();
    }

    function populateSummary() {
        readExtraModels();
        readTelegramStep();
        state.enable_guards = document.getElementById('enable-guards').checked;

        var rows = [
            ['Provider', findProvider(state.provider).name],
            ['Model', state.model || findProvider(state.provider).model],
            ['API key', state.api_key ? 'stored in config' : 'via environment variable'],
        ];

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

    // ── Save ──────────────────────────────────────────────

    function save() {
        var btn = document.getElementById('save-btn');
        btn.disabled = true;
        btn.textContent = 'Saving...';

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
                telegram_token: state.telegram_token
            })
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.ok) {
                showStep(8);
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

    // ── Navigation ────────────────────────────────────────

    function next() {
        if (currentStep === 2) {
            // Read form and go to validation
            readModelStep();
            showStep(3);
            runValidation();
            return;
        }
        if (currentStep < TOTAL_STEPS - 1) {
            showStep(currentStep + 1);
        }
    }

    function prev() {
        if (currentStep === 3) {
            // From validation, go back to model+key
            showStep(2);
            return;
        }
        if (currentStep > 0) {
            showStep(currentStep - 1);
        }
    }

    function validate() {
        readModelStep();
        showStep(3);
        runValidation();
    }

    // ── Public API ────────────────────────────────────────

    window.OB = {
        next: next,
        prev: prev,
        validate: validate,
        save: save,
        addExtraModel: addExtraModel
    };

    // Boot
    init();
})();
