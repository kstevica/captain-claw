/* Captain Claw — Settings Page */

(function () {
    'use strict';

    // ── DOM refs ────────────────────────────────────────
    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    const nav      = $('#nav');
    const main     = $('#main');
    const saveBtn  = $('#saveBtn');
    const dirtyDot = $('#dirtyDot');
    const saveStatus = $('#saveStatus');
    const toastContainer = $('#toastContainer');

    // ── State ───────────────────────────────────────────
    let schema = [];        // groups array from /api/settings/schema
    let values = {};        // flat { "model.provider": "openai", ... }
    let dirty  = {};        // keys that changed since last save
    let activeGroup = null; // current sidebar selection id

    // ── Init ────────────────────────────────────────────
    async function init() {
        const [schemaRes, valuesRes] = await Promise.all([
            fetch('/api/settings/schema').then(r => r.json()),
            fetch('/api/settings').then(r => r.json()),
        ]);
        schema = schemaRes.groups || [];
        values = valuesRes.values || {};
        renderNav();
        if (schema.length) selectGroup(schema[0].id);
        saveBtn.addEventListener('click', save);
        document.addEventListener('keydown', e => {
            if ((e.metaKey || e.ctrlKey) && e.key === 's') {
                e.preventDefault();
                save();
            }
        });
        window.addEventListener('beforeunload', e => {
            if (Object.keys(dirty).length) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    // ── Nav ─────────────────────────────────────────────
    function renderNav() {
        nav.innerHTML = '';
        for (const g of schema) {
            const el = document.createElement('div');
            el.className = 'st-nav-item';
            el.dataset.id = g.id;
            el.innerHTML =
                '<span class="st-nav-icon">' + g.icon + '</span>' +
                '<span class="st-nav-label">' + esc(g.title) + '</span>';
            el.addEventListener('click', () => selectGroup(g.id));
            nav.appendChild(el);
        }
    }

    function selectGroup(id) {
        activeGroup = id;
        $$('.st-nav-item').forEach(el => el.classList.toggle('active', el.dataset.id === id));
        const group = schema.find(g => g.id === id);
        if (group) renderGroup(group);
    }

    // ── Group / Section rendering ───────────────────────
    function renderGroup(group) {
        main.innerHTML = '';

        // Group title
        const h = document.createElement('div');
        h.className = 'st-group-title';
        h.innerHTML = '<span class="icon">' + group.icon + '</span> ' + esc(group.title);
        main.appendChild(h);

        for (const section of group.sections || []) {
            const card = document.createElement('div');
            card.className = 'st-section';
            card.dataset.sectionId = section.id;

            // Handle conditional visibility (show_when)
            if (section.show_when) {
                const depKey = section.show_when.key;
                const depVal = section.show_when.value;
                const curVal = getVal(depKey);
                if (curVal !== depVal) card.classList.add('conditional-hidden');
            }

            // Section header
            const hdr = document.createElement('div');
            hdr.className = 'st-section-header';

            const titleWrap = document.createElement('div');
            let titleHtml = '<div class="st-section-title">' + esc(section.title) + '</div>';
            if (section.description) {
                titleHtml += '<div class="st-section-desc">' + esc(section.description) + '</div>';
            }
            titleWrap.innerHTML = titleHtml;
            hdr.appendChild(titleWrap);

            if (section.wizard) {
                const wb = document.createElement('button');
                wb.className = 'st-btn wizard-trigger sm';
                wb.textContent = 'Setup wizard';
                wb.addEventListener('click', () => openWizard(section.wizard, section));
                hdr.appendChild(wb);
            }
            card.appendChild(hdr);

            // Fields or array
            if (section.type === 'array') {
                renderArraySection(card, section);
            } else {
                for (const field of section.fields || []) {
                    card.appendChild(renderField(field));
                }
            }

            main.appendChild(card);
        }
    }

    // ── Field rendering ─────────────────────────────────
    function renderField(field) {
        const row = document.createElement('div');
        row.className = 'st-field';

        // Label
        const lbl = document.createElement('div');
        lbl.className = 'st-field-label';
        let labelHtml = esc(field.label);
        if (field.readonly) labelHtml += '<span class="st-readonly-badge">restart required</span>';
        if (field.hint) labelHtml += '<div class="st-field-hint">' + esc(field.hint) + '</div>';
        lbl.innerHTML = labelHtml;
        row.appendChild(lbl);

        // Control
        const ctrl = document.createElement('div');
        ctrl.className = 'st-field-control';

        const val = getVal(field.key);
        const type = field.type || 'text';

        switch (type) {
            case 'toggle':
                ctrl.appendChild(makeToggle(field.key, val));
                break;
            case 'select':
                ctrl.appendChild(makeSelect(field.key, field.options || [], val));
                break;
            case 'range':
                ctrl.appendChild(makeRange(field.key, field, val));
                break;
            case 'tags':
                ctrl.appendChild(makeTags(field.key, val));
                break;
            case 'secret':
                ctrl.appendChild(makeInput(field.key, val, 'password', field.readonly));
                break;
            case 'number':
                ctrl.appendChild(makeNumberInput(field.key, field, val));
                break;
            default:
                ctrl.appendChild(makeInput(field.key, val, 'text', field.readonly));
        }

        row.appendChild(ctrl);
        return row;
    }

    // ── Input widgets ───────────────────────────────────

    function makeInput(key, val, type, readonly) {
        const inp = document.createElement('input');
        inp.className = 'st-input';
        inp.type = type || 'text';
        inp.value = val != null ? String(val) : '';
        if (readonly) inp.disabled = true;
        if (!readonly) {
            inp.addEventListener('input', () => markDirty(key, inp.value));
        }
        return inp;
    }

    function makeNumberInput(key, field, val) {
        const inp = document.createElement('input');
        inp.className = 'st-input';
        inp.type = 'number';
        inp.value = val != null ? val : '';
        if (field.min != null) inp.min = field.min;
        if (field.max != null) inp.max = field.max;
        if (field.step != null) inp.step = field.step;
        if (field.readonly) inp.disabled = true;
        else inp.addEventListener('input', () => markDirty(key, inp.value === '' ? null : Number(inp.value)));
        return inp;
    }

    function makeSelect(key, options, val) {
        const sel = document.createElement('select');
        sel.className = 'st-select';
        for (const opt of options) {
            const o = document.createElement('option');
            o.value = opt;
            o.textContent = opt;
            if (String(val) === String(opt)) o.selected = true;
            sel.appendChild(o);
        }
        sel.addEventListener('change', () => {
            markDirty(key, sel.value);
            updateConditionalSections(key, sel.value);
        });
        return sel;
    }

    function makeToggle(key, val) {
        const wrap = document.createElement('label');
        wrap.className = 'st-toggle';
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.checked = !!val;
        const track = document.createElement('span');
        track.className = 'st-toggle-track';
        wrap.appendChild(inp);
        wrap.appendChild(track);
        inp.addEventListener('change', () => markDirty(key, inp.checked));
        return wrap;
    }

    function makeRange(key, field, val) {
        const wrap = document.createElement('div');
        wrap.className = 'st-range-wrap';
        const inp = document.createElement('input');
        inp.className = 'st-range';
        inp.type = 'range';
        inp.min = field.min != null ? field.min : 0;
        inp.max = field.max != null ? field.max : 1;
        inp.step = field.step != null ? field.step : 0.1;
        inp.value = val != null ? val : inp.min;
        const vSpan = document.createElement('span');
        vSpan.className = 'st-range-value';
        vSpan.textContent = inp.value;
        inp.addEventListener('input', () => {
            vSpan.textContent = inp.value;
            markDirty(key, Number(inp.value));
        });
        wrap.appendChild(inp);
        wrap.appendChild(vSpan);
        return wrap;
    }

    function makeTags(key, val) {
        const arr = Array.isArray(val) ? [...val] : [];
        const wrap = document.createElement('div');
        wrap.className = 'st-tags';

        function rebuild() {
            wrap.innerHTML = '';
            for (let i = 0; i < arr.length; i++) {
                const tag = document.createElement('span');
                tag.className = 'st-tag';
                tag.innerHTML = esc(arr[i]) +
                    ' <span class="st-tag-remove" data-idx="' + i + '">&times;</span>';
                wrap.appendChild(tag);
            }
            const inp = document.createElement('input');
            inp.className = 'st-tags-input';
            inp.placeholder = 'Add...';
            inp.addEventListener('keydown', e => {
                if (e.key === 'Enter' || e.key === ',') {
                    e.preventDefault();
                    const v = inp.value.trim().replace(/,/g, '');
                    if (v && !arr.includes(v)) {
                        arr.push(v);
                        markDirty(key, [...arr]);
                        rebuild();
                    }
                } else if (e.key === 'Backspace' && !inp.value && arr.length) {
                    arr.pop();
                    markDirty(key, [...arr]);
                    rebuild();
                }
            });
            wrap.appendChild(inp);
            // Focus input when clicking the container
            wrap.addEventListener('click', () => inp.focus());
        }

        wrap.addEventListener('click', e => {
            const rm = e.target.closest('.st-tag-remove');
            if (rm) {
                arr.splice(Number(rm.dataset.idx), 1);
                markDirty(key, [...arr]);
                rebuild();
            }
        });

        rebuild();
        return wrap;
    }

    // ── Array section (model.allowed) ───────────────────
    function renderArraySection(card, section) {
        if (section.layout === 'cards') {
            renderCardArraySection(card, section);
        } else {
            renderTableArraySection(card, section);
        }
    }

    function renderTableArraySection(card, section) {
        const arrKey = section.array_key;
        const items = Array.isArray(values[arrKey]) ? [...values[arrKey]] : [];
        const fields = section.item_fields || [];

        const tableWrap = document.createElement('div');
        tableWrap.style.overflowX = 'auto';

        function rebuild() {
            tableWrap.innerHTML = '';
            if (!items.length) {
                tableWrap.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:8px 0;">No items configured.</div>';
            } else {
                const table = document.createElement('table');
                table.className = 'st-array-table';
                let thead = '<tr>';
                for (const f of fields) thead += '<th>' + esc(f.label) + '</th>';
                thead += '<th></th></tr>';
                table.innerHTML = thead;
                for (let i = 0; i < items.length; i++) {
                    const tr = document.createElement('tr');
                    for (const f of fields) {
                        const td = document.createElement('td');
                        if (f.type === 'select') {
                            const sel = document.createElement('select');
                            for (const opt of f.options || []) {
                                const o = document.createElement('option');
                                o.value = opt;
                                o.textContent = opt;
                                if (items[i][f.key] === opt) o.selected = true;
                                sel.appendChild(o);
                            }
                            sel.addEventListener('change', () => {
                                items[i][f.key] = sel.value;
                                markDirty(arrKey, [...items]);
                            });
                            td.appendChild(sel);
                        } else {
                            const inp = document.createElement('input');
                            inp.type = f.type === 'number' ? 'number' : 'text';
                            inp.value = items[i][f.key] != null ? items[i][f.key] : '';
                            inp.addEventListener('input', () => {
                                items[i][f.key] = f.type === 'number' ? Number(inp.value) : inp.value;
                                markDirty(arrKey, [...items]);
                            });
                            td.appendChild(inp);
                        }
                        tr.appendChild(td);
                    }
                    const delTd = document.createElement('td');
                    const delBtn = document.createElement('button');
                    delBtn.className = 'st-btn danger sm';
                    delBtn.textContent = 'Remove';
                    delBtn.addEventListener('click', () => {
                        items.splice(i, 1);
                        markDirty(arrKey, [...items]);
                        rebuild();
                    });
                    delTd.appendChild(delBtn);
                    tr.appendChild(delTd);
                    table.appendChild(tr);
                }
                tableWrap.appendChild(table);
            }

            const actions = document.createElement('div');
            actions.className = 'st-array-actions';
            const addBtn = document.createElement('button');
            addBtn.className = 'st-btn sm';
            addBtn.textContent = '+ Add';
            addBtn.addEventListener('click', () => {
                const newItem = {};
                for (const f of fields) newItem[f.key] = '';
                items.push(newItem);
                markDirty(arrKey, [...items]);
                rebuild();
            });
            actions.appendChild(addBtn);
            tableWrap.appendChild(actions);
        }

        rebuild();
        card.appendChild(tableWrap);
    }

    // ── Card-based array (rich model editor) ─────────────
    function renderCardArraySection(card, section) {
        const arrKey = section.array_key;
        const items = Array.isArray(values[arrKey]) ? [...values[arrKey]] : [];
        const fields = section.item_fields || [];

        const wrap = document.createElement('div');
        wrap.className = 'st-model-cards';

        function rebuild() {
            wrap.innerHTML = '';

            if (!items.length) {
                wrap.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:8px 0;">No models configured.</div>';
            }

            for (let i = 0; i < items.length; i++) {
                const mcard = document.createElement('div');
                mcard.className = 'st-model-card';

                // ── Card header: ID badge + provider:model + collapse/remove ──
                const header = document.createElement('div');
                header.className = 'st-model-card-header';

                const titleArea = document.createElement('div');
                titleArea.className = 'st-model-card-title';
                const idVal = items[i].id || 'model-' + (i + 1);
                const provVal = items[i].provider || '?';
                const modVal = items[i].model || '?';
                titleArea.innerHTML =
                    '<span class="st-model-id-badge">' + esc(idVal) + '</span> ' +
                    '<span class="st-model-name">' + esc(provVal) + ' / ' + esc(modVal) + '</span>';
                header.appendChild(titleArea);

                const headerActions = document.createElement('div');
                headerActions.className = 'st-model-card-actions';

                const collapseBtn = document.createElement('button');
                collapseBtn.className = 'st-btn sm';
                collapseBtn.textContent = '▾';
                collapseBtn.title = 'Expand / Collapse';
                headerActions.appendChild(collapseBtn);

                const delBtn = document.createElement('button');
                delBtn.className = 'st-btn danger sm';
                delBtn.textContent = 'Remove';
                delBtn.addEventListener('click', () => {
                    items.splice(i, 1);
                    markDirty(arrKey, [...items]);
                    rebuild();
                });
                headerActions.appendChild(delBtn);
                header.appendChild(headerActions);
                mcard.appendChild(header);

                // ── Description preview (if set) ──
                const descVal = items[i].description || '';
                if (descVal) {
                    const descPreview = document.createElement('div');
                    descPreview.className = 'st-model-card-desc';
                    descPreview.textContent = descVal;
                    mcard.appendChild(descPreview);
                }

                // ── Card body (collapsible) ──
                const body = document.createElement('div');
                body.className = 'st-model-card-body collapsed';

                // Render fields in a two-column grid
                const grid = document.createElement('div');
                grid.className = 'st-model-card-grid';

                for (const f of fields) {
                    const fieldWrap = document.createElement('div');
                    fieldWrap.className = 'st-model-card-field';

                    const label = document.createElement('label');
                    label.className = 'st-model-card-label';
                    label.textContent = f.label;
                    fieldWrap.appendChild(label);

                    if (f.hint) {
                        const hint = document.createElement('div');
                        hint.className = 'st-model-card-hint';
                        hint.textContent = f.hint;
                        fieldWrap.appendChild(hint);
                    }

                    const control = _makeCardControl(f, items, i, arrKey);
                    fieldWrap.appendChild(control);
                    grid.appendChild(fieldWrap);
                }

                body.appendChild(grid);
                mcard.appendChild(body);

                // Collapse toggle
                collapseBtn.addEventListener('click', () => {
                    const isCollapsed = body.classList.toggle('collapsed');
                    collapseBtn.textContent = isCollapsed ? '▾' : '▴';
                });

                wrap.appendChild(mcard);
            }

            // ── Add model button ──
            const actions = document.createElement('div');
            actions.className = 'st-array-actions';
            const addBtn = document.createElement('button');
            addBtn.className = 'st-btn sm';
            addBtn.textContent = '+ Add model';
            addBtn.addEventListener('click', () => {
                const newItem = {};
                for (const f of fields) newItem[f.key] = f.type === 'number' ? 0 : '';
                items.push(newItem);
                markDirty(arrKey, [...items]);
                rebuild();
                // Auto-expand the new card
                const cards = wrap.querySelectorAll('.st-model-card');
                const last = cards[cards.length - 1];
                if (last) {
                    const bodyEl = last.querySelector('.st-model-card-body');
                    const btn = last.querySelector('.st-model-card-actions .st-btn');
                    if (bodyEl) bodyEl.classList.remove('collapsed');
                    if (btn) btn.textContent = '▴';
                }
            });
            actions.appendChild(addBtn);
            wrap.appendChild(actions);
        }

        function _refreshCardHeader(el, items, idx) {
            const card = el.closest('.st-model-card');
            if (!card) return;
            const titleEl = card.querySelector('.st-model-card-title');
            if (titleEl) {
                const idVal = items[idx].id || 'model-' + (idx + 1);
                const provVal = items[idx].provider || '?';
                const modVal = items[idx].model || '?';
                titleEl.innerHTML =
                    '<span class="st-model-id-badge">' + esc(idVal) + '</span> ' +
                    '<span class="st-model-name">' + esc(provVal) + ' / ' + esc(modVal) + '</span>';
            }
            let descEl = card.querySelector('.st-model-card-desc');
            const descVal = items[idx].description || '';
            if (descVal) {
                if (!descEl) {
                    descEl = document.createElement('div');
                    descEl.className = 'st-model-card-desc';
                    const header = card.querySelector('.st-model-card-header');
                    if (header) header.after(descEl);
                }
                descEl.textContent = descVal;
            } else if (descEl) {
                descEl.remove();
            }
        }

        function _makeCardControl(f, items, idx, arrKey) {
            if (f.type === 'select') {
                const sel = document.createElement('select');
                sel.className = 'st-input';
                for (const opt of f.options || []) {
                    const o = document.createElement('option');
                    o.value = opt;
                    o.textContent = opt || '—';
                    if (String(items[idx][f.key] || '') === String(opt)) o.selected = true;
                    sel.appendChild(o);
                }
                sel.addEventListener('change', () => {
                    items[idx][f.key] = sel.value;
                    markDirty(arrKey, [...items]);
                    // Refresh header title when provider/model/id change
                    if (f.key === 'provider' || f.key === 'model' || f.key === 'id') {
                        _refreshCardHeader(sel, items, idx);
                    }
                });
                return sel;
            }
            const inp = document.createElement('input');
            inp.className = 'st-input';
            inp.type = f.type === 'number' ? 'number' : 'text';
            if (f.placeholder) inp.placeholder = f.placeholder;
            if (f.min != null) inp.min = f.min;
            if (f.max != null) inp.max = f.max;
            if (f.step != null) inp.step = f.step;
            const raw = items[idx][f.key];
            inp.value = raw != null && raw !== 0 && raw !== '' ? raw : '';
            // For number fields show empty instead of 0 (means "use default")
            if (f.type === 'number' && (raw === 0 || raw === '0' || raw === '')) {
                inp.value = '';
                inp.placeholder = f.placeholder || '0';
            }
            inp.addEventListener('input', () => {
                if (f.type === 'number') {
                    items[idx][f.key] = inp.value === '' ? 0 : Number(inp.value);
                } else {
                    items[idx][f.key] = inp.value;
                }
                markDirty(arrKey, [...items]);
            });
            // Refresh header when id/model/description change
            inp.addEventListener('change', () => {
                if (f.key === 'id' || f.key === 'model' || f.key === 'description') {
                    _refreshCardHeader(inp, items, idx);
                }
            });
            return inp;
        }

        rebuild();
        card.appendChild(wrap);
    }

    // ── Conditional sections ────────────────────────────
    function updateConditionalSections(changedKey, newValue) {
        $$('.st-section').forEach(card => {
            const sid = card.dataset.sectionId;
            if (!sid) return;
            const group = schema.find(g => (g.sections || []).some(s => s.id === sid));
            if (!group) return;
            const section = group.sections.find(s => s.id === sid);
            if (!section || !section.show_when) return;
            if (section.show_when.key === changedKey) {
                card.classList.toggle('conditional-hidden', newValue !== section.show_when.value);
            }
        });
    }

    // ── Dirty tracking ──────────────────────────────────
    function markDirty(key, val) {
        dirty[key] = val;
        values[key] = val;
        dirtyDot.classList.remove('hidden');
        saveBtn.disabled = false;
    }

    function clearDirty() {
        dirty = {};
        dirtyDot.classList.add('hidden');
    }

    // ── Save ────────────────────────────────────────────
    async function save() {
        if (!Object.keys(dirty).length) return;
        saveBtn.disabled = true;
        saveStatus.textContent = 'Saving...';

        try {
            const res = await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ changes: dirty }),
            });
            const data = await res.json();

            if (data.ok) {
                clearDirty();
                saveStatus.textContent = 'Saved';
                toast('Settings saved successfully.', 'success');
                setTimeout(() => { saveStatus.textContent = ''; }, 3000);
                // Refresh values from server to pick up any normalization
                const fresh = await fetch('/api/settings').then(r => r.json());
                values = fresh.values || {};
            } else {
                toast('Error: ' + (data.error || 'Unknown error'), 'error');
                saveBtn.disabled = false;
                saveStatus.textContent = '';
            }
        } catch (err) {
            toast('Network error: ' + err.message, 'error');
            saveBtn.disabled = false;
            saveStatus.textContent = '';
        }
    }

    // ── Toast ────────────────────────────────────────────
    function toast(msg, type) {
        const el = document.createElement('div');
        el.className = 'st-toast ' + (type || '');
        el.textContent = msg;
        toastContainer.appendChild(el);
        setTimeout(() => { el.remove(); }, 4000);
    }

    // ── Wizards ─────────────────────────────────────────
    const wizardOverlay = $('#wizardOverlay');
    const wizardTitle   = $('#wizardTitle');
    const wizardBody    = $('#wizardBody');
    const wizardPrev    = $('#wizardPrev');
    const wizardNext    = $('#wizardNext');
    const wizardClose   = $('#wizardClose');

    let wizardState = null;

    function openWizard(type, section) {
        const steps = buildWizardSteps(type, section);
        if (!steps.length) return;
        wizardState = { type, section, steps, current: 0, values: {} };
        renderWizardStep();
        wizardOverlay.classList.remove('hidden');
    }

    function closeWizard() {
        wizardOverlay.classList.add('hidden');
        wizardState = null;
    }

    wizardClose.addEventListener('click', closeWizard);
    wizardOverlay.addEventListener('click', e => {
        if (e.target === wizardOverlay) closeWizard();
    });
    wizardPrev.addEventListener('click', () => {
        if (wizardState && wizardState.current > 0) {
            wizardState.current--;
            renderWizardStep();
        }
    });
    wizardNext.addEventListener('click', () => {
        if (!wizardState) return;
        collectWizardValues();
        if (wizardState.current < wizardState.steps.length - 1) {
            wizardState.current++;
            renderWizardStep();
        } else {
            // Apply wizard values
            for (const [k, v] of Object.entries(wizardState.values)) {
                markDirty(k, v);
            }
            closeWizard();
            // Re-render to show updated values
            const group = schema.find(g => g.id === activeGroup);
            if (group) renderGroup(group);
            toast('Wizard values applied. Click Save to persist.', 'success');
        }
    });

    function buildWizardSteps(type, section) {
        if (type === 'email') {
            return [
                {
                    title: 'Choose Email Provider',
                    fields: [
                        { key: 'tools.send_mail.provider', label: 'Provider', type: 'select',
                          options: ['smtp', 'mailgun', 'sendgrid'] },
                    ],
                },
                {
                    title: 'Credentials',
                    dynamic: true, // fields depend on step 0 selection
                },
                {
                    title: 'Sender Identity',
                    fields: [
                        { key: 'tools.send_mail.from_address', label: 'From address', type: 'text' },
                        { key: 'tools.send_mail.from_name', label: 'From name', type: 'text' },
                    ],
                },
            ];
        }
        if (type === 'messaging') {
            // Determine which messaging platform from section id
            const platform = section.id; // telegram, slack, discord
            const tokenFields = [];
            if (platform === 'telegram') {
                tokenFields.push(
                    { key: 'telegram.bot_token', label: 'Bot Token', type: 'secret',
                      hint: 'Get it from @BotFather on Telegram.' }
                );
            } else if (platform === 'slack') {
                tokenFields.push(
                    { key: 'slack.bot_token', label: 'Bot Token', type: 'secret' },
                    { key: 'slack.app_token', label: 'App Token', type: 'secret' }
                );
            } else if (platform === 'discord') {
                tokenFields.push(
                    { key: 'discord.bot_token', label: 'Bot Token', type: 'secret' },
                    { key: 'discord.application_id', label: 'Application ID', type: 'number' }
                );
            }
            return [
                {
                    title: 'Enable ' + platform.charAt(0).toUpperCase() + platform.slice(1),
                    fields: [
                        { key: platform + '.enabled', label: 'Enable', type: 'toggle' },
                    ],
                },
                {
                    title: 'Credentials',
                    fields: tokenFields,
                },
            ];
        }
        return [];
    }

    function getWizardDynamicFields() {
        if (!wizardState) return [];
        const provider = wizardState.values['tools.send_mail.provider'] ||
                         getVal('tools.send_mail.provider') || 'smtp';
        if (provider === 'mailgun') {
            return [
                { key: 'tools.send_mail.mailgun_api_key', label: 'Mailgun API Key', type: 'secret' },
                { key: 'tools.send_mail.mailgun_domain', label: 'Mailgun Domain', type: 'text' },
            ];
        }
        if (provider === 'sendgrid') {
            return [
                { key: 'tools.send_mail.sendgrid_api_key', label: 'SendGrid API Key', type: 'secret' },
            ];
        }
        return [
            { key: 'tools.send_mail.smtp_host', label: 'SMTP Host', type: 'text' },
            { key: 'tools.send_mail.smtp_port', label: 'SMTP Port', type: 'number' },
            { key: 'tools.send_mail.smtp_username', label: 'Username', type: 'text' },
            { key: 'tools.send_mail.smtp_password', label: 'Password', type: 'secret' },
            { key: 'tools.send_mail.smtp_use_tls', label: 'Use TLS', type: 'toggle' },
        ];
    }

    function renderWizardStep() {
        if (!wizardState) return;
        const step = wizardState.steps[wizardState.current];
        const total = wizardState.steps.length;
        const idx = wizardState.current;

        wizardTitle.textContent = step.title;

        // Step indicator
        let html = '<div class="st-wizard-steps">';
        for (let i = 0; i < total; i++) {
            const cls = i < idx ? 'done' : i === idx ? 'active' : '';
            html += '<div class="st-wizard-step ' + cls + '"></div>';
        }
        html += '</div>';

        wizardBody.innerHTML = html;

        // Fields
        const fields = step.dynamic ? getWizardDynamicFields() : (step.fields || []);
        for (const f of fields) {
            const fieldEl = renderField(f);
            // Override the onchange to write to wizard values
            const inputs = fieldEl.querySelectorAll('input, select');
            inputs.forEach(inp => {
                // Remove existing listeners by cloning
                const clone = inp.cloneNode(true);
                inp.parentNode.replaceChild(clone, inp);

                const handler = () => {
                    let v;
                    if (clone.type === 'checkbox') v = clone.checked;
                    else if (clone.type === 'number' || clone.type === 'range') v = Number(clone.value);
                    else v = clone.value;
                    wizardState.values[f.key] = v;
                };
                clone.addEventListener('input', handler);
                clone.addEventListener('change', handler);

                // Set value from wizard state if already set
                if (wizardState.values[f.key] != null) {
                    if (clone.type === 'checkbox') clone.checked = !!wizardState.values[f.key];
                    else clone.value = wizardState.values[f.key];
                }
            });
            wizardBody.appendChild(fieldEl);
        }

        // Button labels
        wizardPrev.style.display = idx > 0 ? '' : 'none';
        wizardNext.textContent = idx === total - 1 ? 'Apply' : 'Next';
    }

    function collectWizardValues() {
        // Values are collected via input handlers above; nothing extra needed.
    }

    // ── Helpers ──────────────────────────────────────────
    function getVal(key) {
        return key in dirty ? dirty[key] : values[key];
    }

    function esc(str) {
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    // ── Boot ────────────────────────────────────────────
    init().catch(err => {
        main.innerHTML = '<div class="st-loading" style="color:var(--red)">Failed to load settings: ' + esc(err.message) + '</div>';
    });

})();
