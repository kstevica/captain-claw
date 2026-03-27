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

        // Visualization style group gets a special layout.
        if (group.id === 'visualization_style') {
            renderVisualizationStylePane(main);
            return;
        }

        // Personality group gets a special split-pane layout.
        if (group.id === 'personality') {
            renderPersonalitySplitPane(main);
            return;
        }

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

            // Fields, array, or custom component
            if (section.type === 'custom' && section.custom_id === 'personality') {
                // Personality uses a unified split-pane rendered at group level; skip here.
            } else if (section.type === 'custom' && section.custom_id === 'user_personalities') {
                // Handled by the unified personality pane; skip here.
            } else if (section.type === 'custom' && section.custom_id === 'web_connection_info') {
                renderWebConnectionInfo(card);
            } else if (section.type === 'array') {
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

    // ── Visualization Style editor ─────────────────────────

    var _vsData = {};

    function renderVisualizationStylePane(container) {
        var pane = document.createElement('div');
        pane.className = 'st-vs-pane';
        pane.innerHTML = '<div class="st-personality-loading">Loading...</div>';
        container.appendChild(pane);

        fetch('/api/visualization-style').then(function (r) { return r.json(); })
        .then(function (data) {
            _vsData = data;
            _vsRender(pane);
        })
        .catch(function (err) {
            pane.innerHTML = '<div class="st-personality-loading" style="color:var(--red)">'
                + 'Failed to load: ' + esc(err.message) + '</div>';
        });
    }

    function _vsRender(pane) {
        pane.innerHTML = '';

        // Upload zone
        var uploadZone = document.createElement('div');
        uploadZone.className = 'st-vs-upload-zone';
        uploadZone.innerHTML =
            '<div class="st-vs-upload-icon">\uD83D\uDCC4</div>'
            + '<div class="st-vs-upload-label">Drop a brand guide, styled report, or design screenshot</div>'
            + '<div class="st-vs-upload-hint">PNG, JPG, WEBP, PDF, DOCX — AI extracts colors, fonts, and layout rules</div>'
            + '<input type="file" class="st-vs-file-input" accept=".png,.jpg,.jpeg,.webp,.pdf,.docx" style="display:none">';
        pane.appendChild(uploadZone);

        var fileInput = uploadZone.querySelector('.st-vs-file-input');
        uploadZone.addEventListener('click', function () { fileInput.click(); });
        uploadZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', function () {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', function (e) {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) _vsAnalyzeFile(e.dataTransfer.files[0], pane);
        });
        fileInput.addEventListener('change', function () {
            if (fileInput.files.length) _vsAnalyzeFile(fileInput.files[0], pane);
        });

        // Form
        var form = document.createElement('div');
        form.className = 'st-pp-editor-form st-vs-form';
        pane.appendChild(form);

        // Color palette editor
        var palRow = document.createElement('div');
        palRow.className = 'st-pp-field';
        var palLabel = document.createElement('label');
        palLabel.className = 'st-pp-field-label';
        palLabel.textContent = 'Color Palette';
        palRow.appendChild(palLabel);

        var palCtrl = document.createElement('div');
        palCtrl.className = 'st-pp-field-ctrl st-vs-palette-ctrl';

        var palSwatches = document.createElement('div');
        palSwatches.className = 'st-vs-palette';
        palCtrl.appendChild(palSwatches);

        var addColorBtn = document.createElement('button');
        addColorBtn.className = 'st-btn sm';
        addColorBtn.textContent = '+ Add Color';
        addColorBtn.addEventListener('click', function () {
            _vsData.color_palette = _vsData.color_palette || [];
            _vsData.color_palette.push('#666666');
            _vsRenderPalette(palSwatches);
        });
        palCtrl.appendChild(addColorBtn);

        palRow.appendChild(palCtrl);
        form.appendChild(palRow);
        _vsRenderPalette(palSwatches);

        // Text fields
        var nameEl = _vsFormField(form, 'Name', 'text', _vsData.name || '', 'Style profile name');
        var fontPEl = _vsFormField(form, 'Font Primary', 'text', _vsData.font_primary || '', 'e.g. Inter, system-ui, sans-serif');
        var fontHEl = _vsFormField(form, 'Font Headings', 'text', _vsData.font_headings || '', 'e.g. Poppins, sans-serif');
        var fontMEl = _vsFormField(form, 'Font Mono', 'text', _vsData.font_mono || '', 'e.g. JetBrains Mono, monospace');
        var bgEl = _vsFormField(form, 'Background Style', 'text', _vsData.background_style || '', 'e.g. dark, light, gradient');
        var chartEl = _vsFormField(form, 'Chart Style', 'text', _vsData.chart_style || '', 'e.g. minimal, corporate, modern');
        var layoutEl = _vsFormField(form, 'Layout Notes', 'textarea', _vsData.layout_notes || '', 'Layout and spacing observations', 4);
        var rulesEl = _vsFormField(form, 'Additional Rules', 'textarea', _vsData.additional_rules || '', 'Extra CSS/design conventions', 4);

        // Source description (readonly)
        if (_vsData.source_description) {
            var srcRow = document.createElement('div');
            srcRow.className = 'st-pp-field';
            var srcLbl = document.createElement('label');
            srcLbl.className = 'st-pp-field-label';
            srcLbl.textContent = 'Source';
            srcRow.appendChild(srcLbl);
            var srcVal = document.createElement('div');
            srcVal.className = 'st-vs-source-desc';
            srcVal.textContent = _vsData.source_description;
            srcRow.appendChild(srcVal);
            form.appendChild(srcRow);
        }

        // Buttons
        var btnRow = document.createElement('div');
        btnRow.className = 'st-vs-btn-row';

        var saveBtn = document.createElement('button');
        saveBtn.className = 'st-btn primary';
        saveBtn.textContent = 'Save Style';
        saveBtn.addEventListener('click', function () {
            _vsData.name = nameEl.value.trim();
            _vsData.font_primary = fontPEl.value.trim();
            _vsData.font_headings = fontHEl.value.trim();
            _vsData.font_mono = fontMEl.value.trim();
            _vsData.background_style = bgEl.value.trim();
            _vsData.chart_style = chartEl.value.trim();
            _vsData.layout_notes = layoutEl.value.trim();
            _vsData.additional_rules = rulesEl.value.trim();
            // color_palette already updated via _vsRenderPalette handlers

            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            fetch('/api/visualization-style', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(_vsData)
            })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { toast(data.error, 'error'); }
                else { _vsData = data; toast('Visualization style saved', 'success'); }
            })
            .catch(function (err) { toast('Save failed: ' + err.message, 'error'); })
            .finally(function () { saveBtn.disabled = false; saveBtn.textContent = 'Save Style'; });
        });
        btnRow.appendChild(saveBtn);

        var clearBtn = document.createElement('button');
        clearBtn.className = 'st-btn';
        clearBtn.textContent = 'Clear All';
        clearBtn.addEventListener('click', function () {
            if (!confirm('Clear all visualization style settings?')) return;
            _vsData = { name: 'Default', color_palette: [], font_primary: '', font_headings: '', font_mono: '', background_style: '', chart_style: '', layout_notes: '', additional_rules: '', source_description: '' };
            fetch('/api/visualization-style', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(_vsData)
            })
            .then(function () { toast('Style cleared', 'success'); _vsRender(pane); })
            .catch(function (err) { toast('Clear failed: ' + err.message, 'error'); });
        });
        btnRow.appendChild(clearBtn);

        form.appendChild(btnRow);
    }

    function _vsRenderPalette(container) {
        container.innerHTML = '';
        var palette = _vsData.color_palette || [];
        palette.forEach(function (color, idx) {
            var swatch = document.createElement('div');
            swatch.className = 'st-vs-swatch-wrap';

            var colorDot = document.createElement('div');
            colorDot.className = 'st-vs-swatch';
            colorDot.style.backgroundColor = color;
            swatch.appendChild(colorDot);

            var inp = document.createElement('input');
            inp.className = 'st-input st-vs-color-input';
            inp.type = 'text';
            inp.value = color;
            inp.addEventListener('change', function () {
                _vsData.color_palette[idx] = inp.value.trim();
                colorDot.style.backgroundColor = inp.value.trim();
            });
            swatch.appendChild(inp);

            var rmBtn = document.createElement('button');
            rmBtn.className = 'st-btn sm st-vs-color-rm';
            rmBtn.textContent = '\u00D7';
            rmBtn.title = 'Remove color';
            rmBtn.addEventListener('click', function () {
                _vsData.color_palette.splice(idx, 1);
                _vsRenderPalette(container);
            });
            swatch.appendChild(rmBtn);

            container.appendChild(swatch);
        });
    }

    function _vsFormField(container, label, type, value, placeholder, rows) {
        var row = document.createElement('div');
        row.className = 'st-pp-field';

        var lbl = document.createElement('label');
        lbl.className = 'st-pp-field-label';
        lbl.textContent = label;
        row.appendChild(lbl);

        var ctrlWrap = document.createElement('div');
        ctrlWrap.className = 'st-pp-field-ctrl';

        var el;
        if (type === 'textarea') {
            el = document.createElement('textarea');
            el.className = 'st-input st-textarea';
            el.rows = rows || 3;
        } else {
            el = document.createElement('input');
            el.className = 'st-input';
            el.type = type || 'text';
        }
        el.value = value;
        if (placeholder) el.placeholder = placeholder;
        ctrlWrap.appendChild(el);

        // Rephrase button for textareas
        if (type === 'textarea') {
            var fieldKey = label.toLowerCase().replace(/ /g, '_');
            var rephraseBtn = document.createElement('button');
            rephraseBtn.className = 'st-btn sm st-pp-rephrase-btn';
            rephraseBtn.innerHTML = '\u2728 Rephrase & Enrich';
            rephraseBtn.addEventListener('click', function () {
                var text = el.value.trim();
                if (!text) { toast('Nothing to rephrase', 'error'); return; }
                rephraseBtn.disabled = true;
                rephraseBtn.innerHTML = '\u23F3 Rephrasing...';
                el.classList.add('st-pp-rephrasing');

                fetch('/api/visualization-style/rephrase', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ field: fieldKey, text: text })
                })
                .then(function (r) { return r.json(); })
                .then(function (result) {
                    if (result.error) toast(result.error, 'error');
                    else if (result.text) el.value = result.text;
                })
                .catch(function (err) { toast('Rephrase failed: ' + err.message, 'error'); })
                .finally(function () {
                    rephraseBtn.disabled = false;
                    rephraseBtn.innerHTML = '\u2728 Rephrase & Enrich';
                    el.classList.remove('st-pp-rephrasing');
                });
            });
            ctrlWrap.appendChild(rephraseBtn);
        }

        row.appendChild(ctrlWrap);
        container.appendChild(row);
        return el;
    }

    function _vsAnalyzeFile(file, pane) {
        var allowed = ['.png', '.jpg', '.jpeg', '.webp', '.pdf', '.docx'];
        var ext = '.' + file.name.split('.').pop().toLowerCase();
        if (allowed.indexOf(ext) === -1) {
            toast('Unsupported file type: ' + ext, 'error');
            return;
        }

        var zone = pane.querySelector('.st-vs-upload-zone');
        var origHTML = zone.innerHTML;
        zone.innerHTML =
            '<div class="st-vs-upload-icon">\u23F3</div>'
            + '<div class="st-vs-upload-label">Analyzing ' + esc(file.name) + '...</div>'
            + '<div class="st-vs-upload-hint">Extracting colors, fonts, and layout rules via AI</div>';
        zone.classList.add('analyzing');

        var fd = new FormData();
        fd.append('file', file);

        fetch('/api/visualization-style/analyze', { method: 'POST', body: fd })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                toast('Analysis failed: ' + data.error, 'error');
                zone.innerHTML = origHTML;
                zone.classList.remove('analyzing');
                return;
            }
            // Merge extracted data into current state.
            if (data.color_palette && data.color_palette.length) _vsData.color_palette = data.color_palette;
            if (data.font_primary) _vsData.font_primary = data.font_primary;
            if (data.font_headings) _vsData.font_headings = data.font_headings;
            if (data.font_mono) _vsData.font_mono = data.font_mono;
            if (data.background_style) _vsData.background_style = data.background_style;
            if (data.chart_style) _vsData.chart_style = data.chart_style;
            if (data.layout_notes) _vsData.layout_notes = data.layout_notes;
            if (data.additional_rules) _vsData.additional_rules = data.additional_rules;
            if (data.source_description) _vsData.source_description = data.source_description;
            toast('Style extracted from ' + file.name, 'success');
            _vsRender(pane);
        })
        .catch(function (err) {
            toast('Analysis failed: ' + err.message, 'error');
            zone.innerHTML = origHTML;
            zone.classList.remove('analyzing');
        });
    }

    // ── Personality split-pane layout ───────────────────────
    // State for the personality pane.
    var _ppState = {
        agentData: null,       // global personality
        userPersonalities: [], // array of user personality objects
        telegramUsers: [],     // approved telegram users
        selectedId: '__agent__', // '__agent__' or a user_id
    };

    function renderPersonalitySplitPane(container) {
        var pane = document.createElement('div');
        pane.className = 'st-pp-split';
        pane.innerHTML = '<div class="st-pp-left"><div class="st-personality-loading">Loading...</div></div>'
            + '<div class="st-pp-right"><div class="st-personality-loading">Select a persona</div></div>';
        container.appendChild(pane);

        Promise.all([
            fetch('/api/personality').then(function (r) { return r.json(); }),
            fetch('/api/user-personalities').then(function (r) { return r.json(); }),
            fetch('/api/telegram-users').then(function (r) { return r.json(); })
        ])
        .then(function (results) {
            _ppState.agentData = results[0];
            _ppState.userPersonalities = results[1];
            _ppState.telegramUsers = results[2];
            _ppState.selectedId = '__agent__';
            _ppRenderLeft(pane);
            _ppRenderRight(pane);
        })
        .catch(function (err) {
            pane.innerHTML = '<div class="st-personality-loading" style="color:var(--red)">'
                + 'Failed to load: ' + esc(err.message) + '</div>';
        });
    }

    function _ppRenderLeft(pane) {
        var left = pane.querySelector('.st-pp-left');
        left.innerHTML = '';

        // Section label
        var lbl = document.createElement('div');
        lbl.className = 'st-pp-section-label';
        lbl.textContent = 'Agent';
        left.appendChild(lbl);

        // Agent personality item
        var agentItem = document.createElement('div');
        agentItem.className = 'st-pp-item' + (_ppState.selectedId === '__agent__' ? ' active' : '');
        agentItem.dataset.id = '__agent__';
        agentItem.innerHTML = '<span class="st-pp-item-icon">\uD83E\uDD16</span>'
            + '<span class="st-pp-item-name">' + esc(_ppState.agentData.name || 'Agent') + '</span>';
        agentItem.addEventListener('click', function () {
            _ppState.selectedId = '__agent__';
            _ppRenderLeft(pane);
            _ppRenderRight(pane);
        });
        left.appendChild(agentItem);

        // User personas section
        var ulbl = document.createElement('div');
        ulbl.className = 'st-pp-section-label';
        ulbl.textContent = 'User Personas';
        left.appendChild(ulbl);

        // Build lookup
        var personalityMap = {};
        _ppState.userPersonalities.forEach(function (p) { personalityMap[p.user_id] = p; });
        var telegramIds = {};
        _ppState.telegramUsers.forEach(function (u) { telegramIds[u.user_id] = u; });

        // Configured telegram users
        _ppState.telegramUsers.forEach(function (u) {
            if (personalityMap[u.user_id]) {
                var p = personalityMap[u.user_id];
                p.username = p.username || u.username;
                p.first_name = p.first_name || u.first_name;
                left.appendChild(_ppPersonaItem(pane, p));
            }
        });

        // Non-telegram user personalities
        _ppState.userPersonalities.forEach(function (p) {
            if (!telegramIds[p.user_id]) {
                left.appendChild(_ppPersonaItem(pane, p));
            }
        });

        // Unconfigured telegram users
        _ppState.telegramUsers.forEach(function (u) {
            if (!personalityMap[u.user_id]) {
                var item = document.createElement('div');
                item.className = 'st-pp-item unconfigured';
                item.innerHTML = '<span class="st-pp-item-icon">\uD83D\uDC64</span>'
                    + '<span class="st-pp-item-name">' + esc(_userLabel(u)) + '</span>'
                    + '<span class="st-up-badge-none">No profile</span>';
                item.addEventListener('click', function () {
                    _ppSetupUser(pane, u);
                });
                left.appendChild(item);
            }
        });

        // Add button
        var addBtn = document.createElement('button');
        addBtn.className = 'st-btn secondary sm st-pp-add-btn';
        addBtn.textContent = '+ Add by User ID';
        addBtn.addEventListener('click', function () {
            var userId = prompt('Enter user ID:');
            if (!userId || !userId.trim()) return;
            userId = userId.trim();
            if (personalityMap[userId]) { toast('Personality already exists', 'error'); return; }
            var payload = { name: 'New User', description: '', background: '', expertise: [], instructions: '' };
            fetch('/api/user-personalities/' + userId, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(function (r) { return r.json(); })
            .then(function (result) {
                if (result.error) { toast(result.error, 'error'); return; }
                toast('User personality created', 'success');
                _ppRefresh(pane, userId);
            })
            .catch(function (err) { toast('Failed: ' + err.message, 'error'); });
        });
        left.appendChild(addBtn);
    }

    function _ppPersonaItem(pane, p) {
        var item = document.createElement('div');
        item.className = 'st-pp-item' + (_ppState.selectedId === p.user_id ? ' active' : '');
        item.dataset.id = p.user_id;
        item.innerHTML = '<span class="st-pp-item-icon">\uD83D\uDC64</span>'
            + '<span class="st-pp-item-name">' + esc(p.name || _userLabel(p)) + '</span>';
        item.addEventListener('click', function () {
            _ppState.selectedId = p.user_id;
            _ppRenderLeft(pane);
            _ppRenderRight(pane);
        });
        return item;
    }

    function _ppSetupUser(pane, u) {
        var payload = {
            name: u.first_name || u.username || 'User ' + u.user_id,
            description: '', background: '', expertise: [], instructions: ''
        };
        fetch('/api/user-personalities/' + u.user_id, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(function (r) { return r.json(); })
        .then(function (result) {
            if (result.error) { toast(result.error, 'error'); return; }
            toast('Profile created for ' + _userLabel(u), 'success');
            _ppRefresh(pane, u.user_id);
        })
        .catch(function (err) { toast('Failed: ' + err.message, 'error'); });
    }

    function _ppRefresh(pane, selectId) {
        Promise.all([
            fetch('/api/personality').then(function (r) { return r.json(); }),
            fetch('/api/user-personalities').then(function (r) { return r.json(); }),
            fetch('/api/telegram-users').then(function (r) { return r.json(); })
        ])
        .then(function (results) {
            _ppState.agentData = results[0];
            _ppState.userPersonalities = results[1];
            _ppState.telegramUsers = results[2];
            if (selectId !== undefined) _ppState.selectedId = selectId;
            _ppRenderLeft(pane);
            _ppRenderRight(pane);
        });
    }

    function _ppRenderRight(pane) {
        var right = pane.querySelector('.st-pp-right');
        right.innerHTML = '';

        if (_ppState.selectedId === '__agent__') {
            _ppRenderAgentEditor(right, pane);
        } else {
            var p = null;
            _ppState.userPersonalities.forEach(function (up) {
                if (up.user_id === _ppState.selectedId) p = up;
            });
            if (p) {
                _ppRenderUserEditor(right, pane, p);
            } else {
                right.innerHTML = '<div class="st-personality-loading">Persona not found</div>';
            }
        }
    }

    function _ppRenderAgentEditor(container, pane) {
        var data = _ppState.agentData;

        var header = document.createElement('div');
        header.className = 'st-pp-editor-header';
        header.innerHTML = '<span class="st-pp-editor-title">Agent Personality</span>'
            + '<span class="st-pp-editor-subtitle">Define the agent\'s default identity, background, and expertise areas.</span>';
        container.appendChild(header);

        var form = document.createElement('div');
        form.className = 'st-pp-editor-form';

        // Name
        var nameInp = _ppFormField(form, 'Name', 'text', data.name || '', 'e.g. Captain Claw');

        // Description
        var descInp = _ppFormField(form, 'Description', 'textarea', data.description || '', '', 3);

        // Background
        var bgInp = _ppFormField(form, 'Background', 'textarea', data.background || '', '', 5);

        // Expertise
        var expInp = _ppFormField(form, 'Expertise', 'textarea', (data.expertise || []).join('\n'), 'One expertise per line', 8);

        // Instructions
        var instrInp = _ppFormField(form, 'Instructions', 'textarea', data.instructions || '', 'Additional freeform instructions injected into the system prompt', 5);

        container.appendChild(form);

        // Actions
        var actions = document.createElement('div');
        actions.className = 'st-personality-actions';
        var saveBtn = document.createElement('button');
        saveBtn.className = 'st-btn primary';
        saveBtn.textContent = 'Save Personality';
        saveBtn.addEventListener('click', function () {
            var name = nameInp.value.trim();
            if (!name) { toast('Name is required', 'error'); return; }
            var body = {
                name: name,
                description: descInp.value.trim(),
                background: bgInp.value.trim(),
                expertise: expInp.value.split('\n').map(function (e) { return e.trim(); }).filter(function (e) { return e.length > 0; }),
                instructions: instrInp.value.trim()
            };
            saveBtn.disabled = true; saveBtn.textContent = 'Saving...';
            fetch('/api/personality', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })
            .then(function (r) { return r.json(); })
            .then(function (result) {
                if (result.error) { toast(result.error, 'error'); }
                else {
                    toast('Personality saved', 'success');
                    _ppState.agentData = result;
                    _ppRenderLeft(pane);
                }
            })
            .catch(function (err) { toast('Failed: ' + err.message, 'error'); })
            .finally(function () { saveBtn.disabled = false; saveBtn.textContent = 'Save Personality'; });
        });
        actions.appendChild(saveBtn);
        container.appendChild(actions);
    }

    function _ppRenderUserEditor(container, pane, p) {
        var header = document.createElement('div');
        header.className = 'st-pp-editor-header';
        header.innerHTML = '<span class="st-pp-editor-title">' + esc(_userLabel(p)) + '</span>'
            + '<span class="st-pp-editor-subtitle">User profile — tells the agent who it is talking to.</span>';

        var delBtn = document.createElement('button');
        delBtn.className = 'st-btn sm danger';
        delBtn.textContent = 'Delete';
        delBtn.addEventListener('click', function () {
            if (!confirm('Remove personality for ' + _userLabel(p) + '?')) return;
            fetch('/api/user-personalities/' + p.user_id, { method: 'DELETE' })
                .then(function (r) { return r.json(); })
                .then(function () {
                    toast('User personality removed', 'success');
                    _ppRefresh(pane, '__agent__');
                })
                .catch(function (err) { toast('Failed: ' + err.message, 'error'); });
        });
        header.appendChild(delBtn);
        container.appendChild(header);

        var form = document.createElement('div');
        form.className = 'st-pp-editor-form';

        var nameInp = _ppFormField(form, 'Name', 'text', p.name || '', 'e.g. Toby McDev');
        var descInp = _ppFormField(form, 'Description', 'textarea', p.description || '', '', 3);
        var bgInp = _ppFormField(form, 'Background', 'textarea', p.background || '', '', 5);
        var expInp = _ppFormField(form, 'Expertise', 'textarea', (p.expertise || []).join('\n'), 'One expertise per line', 8);
        var instrInp = _ppFormField(form, 'Instructions', 'textarea', p.instructions || '', 'Additional freeform instructions for the agent when talking to this user', 5);

        container.appendChild(form);

        var actions = document.createElement('div');
        actions.className = 'st-personality-actions';
        var saveBtn = document.createElement('button');
        saveBtn.className = 'st-btn primary';
        saveBtn.textContent = 'Save';
        saveBtn.addEventListener('click', function () {
            var payload = {
                name: nameInp.value.trim(),
                description: descInp.value.trim(),
                background: bgInp.value.trim(),
                expertise: expInp.value.split('\n').map(function (e) { return e.trim(); }).filter(function (e) { return e.length > 0; }),
                instructions: instrInp.value.trim()
            };
            if (!payload.name) { toast('Name is required', 'error'); return; }
            saveBtn.disabled = true; saveBtn.textContent = 'Saving...';
            fetch('/api/user-personalities/' + p.user_id, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(function (r) { return r.json(); })
            .then(function (result) {
                if (result.error) { toast(result.error, 'error'); }
                else {
                    toast('Saved personality for ' + _userLabel(p), 'success');
                    _ppRefresh(pane);
                }
            })
            .catch(function (err) { toast('Failed: ' + err.message, 'error'); })
            .finally(function () { saveBtn.disabled = false; saveBtn.textContent = 'Save'; });
        });
        actions.appendChild(saveBtn);
        container.appendChild(actions);
    }

    // Build a form field row with optional rephrase button for textareas.
    // Returns the input/textarea element.
    function _ppFormField(container, label, type, value, placeholder, rows) {
        var row = document.createElement('div');
        row.className = 'st-pp-field';

        var lbl = document.createElement('label');
        lbl.className = 'st-pp-field-label';
        lbl.textContent = label;
        row.appendChild(lbl);

        var ctrlWrap = document.createElement('div');
        ctrlWrap.className = 'st-pp-field-ctrl';

        var el;
        if (type === 'textarea') {
            el = document.createElement('textarea');
            el.className = 'st-input st-textarea';
            el.rows = rows || 3;
        } else {
            el = document.createElement('input');
            el.className = 'st-input';
            el.type = type || 'text';
        }
        el.value = value;
        if (placeholder) el.placeholder = placeholder;
        ctrlWrap.appendChild(el);

        // Rephrase button for textareas
        if (type === 'textarea') {
            var fieldName = label.toLowerCase(); // description, background, expertise
            var rephraseBtn = document.createElement('button');
            rephraseBtn.className = 'st-btn sm st-pp-rephrase-btn';
            rephraseBtn.innerHTML = '\u2728 Rephrase & Enrich';
            rephraseBtn.title = 'Use AI to rephrase and enrich this content';
            rephraseBtn.addEventListener('click', function () {
                var text = el.value.trim();
                if (!text) { toast('Nothing to rephrase', 'error'); return; }
                // Find the name field in the same form
                var formEl = row.closest('.st-pp-editor-form');
                var nameInput = formEl ? formEl.querySelector('input.st-input') : null;
                var personaName = nameInput ? nameInput.value.trim() : '';

                rephraseBtn.disabled = true;
                rephraseBtn.innerHTML = '\u23F3 Rephrasing...';
                el.classList.add('st-pp-rephrasing');

                fetch('/api/personality/rephrase', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ field: fieldName, text: text, name: personaName })
                })
                .then(function (r) { return r.json(); })
                .then(function (result) {
                    if (result.error) { toast(result.error, 'error'); }
                    else if (result.text) { el.value = result.text; }
                })
                .catch(function (err) { toast('Rephrase failed: ' + err.message, 'error'); })
                .finally(function () {
                    rephraseBtn.disabled = false;
                    rephraseBtn.innerHTML = '\u2728 Rephrase & Enrich';
                    el.classList.remove('st-pp-rephrasing');
                });
            });
            ctrlWrap.appendChild(rephraseBtn);
        }

        row.appendChild(ctrlWrap);
        container.appendChild(row);
        return el;
    }

    function _userLabel(p) {
        var parts = [];
        if (p.username) parts.push('@' + p.username);
        if (p.first_name) parts.push(p.first_name);
        if (parts.length === 0) parts.push(p.user_id);
        return parts.join(' \u2014 ');
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

    // ── Web Connection Info ─────────────────────────────
    function renderWebConnectionInfo(card) {
        const host = getVal('web.host') || '127.0.0.1';
        const port = getVal('web.port') || 23080;
        const hasAuth = !!(getVal('web.auth_token') && getVal('web.auth_token') !== '••••••••');

        const wrap = document.createElement('div');
        wrap.className = 'st-web-connection-info';
        wrap.innerHTML = `
            <div class="st-conn-grid">
                <div class="st-conn-item">
                    <div class="st-conn-label">WebSocket URL</div>
                    <div class="st-conn-value">
                        <code id="connWsUrl">ws://${host}:${port}/ws</code>
                        <button class="st-conn-copy" data-copy="connWsUrl" title="Copy">&#x2398;</button>
                    </div>
                </div>
                <div class="st-conn-item">
                    <div class="st-conn-label">Web UI</div>
                    <div class="st-conn-value">
                        <code id="connHttpUrl">http://${host}:${port}</code>
                        <button class="st-conn-copy" data-copy="connHttpUrl" title="Copy">&#x2398;</button>
                    </div>
                </div>
                <div class="st-conn-item">
                    <div class="st-conn-label">Auth</div>
                    <div class="st-conn-value">
                        <code>${hasAuth ? 'Token set — required for connections' : 'No auth — open access'}</code>
                    </div>
                </div>
            </div>
            <p class="st-conn-hint">
                Use these details in <strong>Flight Deck → Local Agents → Add Agent</strong> to connect to this instance.
                For Docker containers, Flight Deck reads the web port automatically from container labels.
            </p>
        `;

        // Copy buttons
        wrap.querySelectorAll('.st-conn-copy').forEach(btn => {
            btn.addEventListener('click', () => {
                const el = document.getElementById(btn.dataset.copy);
                if (el) {
                    navigator.clipboard.writeText(el.textContent).then(() => {
                        btn.textContent = '✓';
                        setTimeout(() => { btn.innerHTML = '&#x2398;'; }, 1500);
                    });
                }
            });
        });

        card.appendChild(wrap);
    }

    // ── Boot ────────────────────────────────────────────
    init().catch(err => {
        main.innerHTML = '<div class="st-loading" style="color:var(--red)">Failed to load settings: ' + esc(err.message) + '</div>';
    });

})();
