/* Captain Claw — Browser Workflows Viewer */

(function () {
    'use strict';

    var API = '/api/browser-workflows';

    var STEP_ACTIONS = [
        'click', 'type', 'navigate', 'select', 'submit',
        'press_key', 'wait', 'scroll', 'hover', 'clear',
        'check', 'uncheck', 'screenshot', 'wait_for'
    ];

    // ── State ────────────────────────────────────────────────────────
    var items = [];
    var selectedItem = null;
    var searchQuery = '';
    var searchTimeout = null;
    var deleteTarget = null;
    var editMode = false;
    var editSteps = [];

    // ── DOM ──────────────────────────────────────────────────────────
    var bwSearch       = document.getElementById('bwSearch');
    var bwCount        = document.getElementById('bwCount');
    var bwItemList     = document.getElementById('bwItemList');
    var bwLoading      = document.getElementById('bwLoading');
    var bwEmpty        = document.getElementById('bwEmpty');
    var bwDetailView   = document.getElementById('bwDetailView');
    var bwDetailTitle  = document.getElementById('bwDetailTitle');
    var bwDetailMeta   = document.getElementById('bwDetailMeta');
    var bwDetailContent = document.getElementById('bwDetailContent');
    var bwEditBtn      = document.getElementById('bwEditBtn');
    var bwSaveBtn      = document.getElementById('bwSaveBtn');
    var bwCancelEditBtn = document.getElementById('bwCancelEditBtn');
    var bwDeleteBtn    = document.getElementById('bwDeleteBtn');
    var bwModalOverlay = document.getElementById('bwModalOverlay');
    var bwModalBody    = document.getElementById('bwModalBody');
    var bwModalCancel  = document.getElementById('bwModalCancel');
    var bwModalConfirm = document.getElementById('bwModalConfirm');
    var bwToast        = document.getElementById('bwToast');

    // ── Init ─────────────────────────────────────────────────────────
    init();

    function init() {
        bwSearch.addEventListener('input', onSearchInput);
        bwDeleteBtn.addEventListener('click', onDeleteClick);
        bwEditBtn.addEventListener('click', onEditClick);
        bwSaveBtn.addEventListener('click', onSaveSteps);
        bwCancelEditBtn.addEventListener('click', onCancelEdit);
        bwModalCancel.addEventListener('click', hideDeleteModal);
        bwModalConfirm.addEventListener('click', onConfirmDelete);

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                if (bwModalOverlay.style.display !== 'none') {
                    hideDeleteModal();
                } else if (editMode) {
                    onCancelEdit();
                }
            }
        });

        loadItems();
    }

    // ── Data Loading ─────────────────────────────────────────────────

    function loadItems() {
        bwLoading.style.display = '';
        clearList();

        var url = API;
        if (searchQuery) {
            // client-side filter since we don't have a search endpoint
            // still load all and filter
        }

        apiFetch(url).then(function (data) {
            bwLoading.style.display = 'none';
            if (!Array.isArray(data)) data = [];

            if (searchQuery) {
                var q = searchQuery.toLowerCase();
                data = data.filter(function (w) {
                    return (w.name || '').toLowerCase().indexOf(q) >= 0 ||
                           (w.description || '').toLowerCase().indexOf(q) >= 0 ||
                           (w.app_name || '').toLowerCase().indexOf(q) >= 0;
                });
            }

            items = data;
            bwCount.textContent = items.length;
            renderList();
        }).catch(function () {
            bwLoading.style.display = 'none';
            items = [];
            bwCount.textContent = '0';
            renderList();
        });
    }

    // ── List Rendering ───────────────────────────────────────────────

    function clearList() {
        bwItemList.querySelectorAll('.bw-list-item, .bw-list-empty').forEach(function (el) {
            el.remove();
        });
    }

    function renderList() {
        clearList();

        if (items.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'bw-list-empty';
            empty.textContent = searchQuery ? 'No matches found' : 'No workflows yet';
            if (!searchQuery) {
                var sub = document.createElement('div');
                sub.className = 'bw-list-empty-sub';
                sub.textContent = 'Record one in chat with browser(action=\'workflow_record_start\')';
                empty.appendChild(sub);
            }
            bwItemList.appendChild(empty);
            return;
        }

        items.forEach(function (item) {
            var el = document.createElement('div');
            el.className = 'bw-list-item';
            if (selectedItem && selectedItem.id === item.id) {
                el.classList.add('selected');
            }

            var nameEl = document.createElement('div');
            nameEl.className = 'bw-list-item-name';
            nameEl.textContent = item.name || '(untitled)';
            el.appendChild(nameEl);

            if (item.description) {
                var descEl = document.createElement('div');
                descEl.className = 'bw-list-item-desc';
                descEl.textContent = item.description;
                el.appendChild(descEl);
            }

            var badges = document.createElement('div');
            badges.className = 'bw-list-item-badges';

            if (item.app_name) {
                addBadge(badges, item.app_name, 'app');
            }

            var stepCount = Array.isArray(item.steps) ? item.steps.length : 0;
            addBadge(badges, stepCount + ' steps', 'steps');

            var varCount = Array.isArray(item.variables) ? item.variables.length : 0;
            if (varCount > 0) {
                addBadge(badges, varCount + ' vars', 'vars');
            }

            if (item.use_count > 0) {
                addBadge(badges, item.use_count + ' runs', 'uses');
            }

            el.appendChild(badges);
            el.addEventListener('click', function () { onItemClick(item); });
            bwItemList.appendChild(el);
        });
    }

    // ── Detail Rendering ─────────────────────────────────────────────

    function onItemClick(item) {
        if (editMode) {
            editMode = false;
            editSteps = [];
            updateEditButtons();
        }
        selectedItem = item;

        bwItemList.querySelectorAll('.bw-list-item').forEach(function (el, idx) {
            el.classList.toggle('selected', items[idx] && items[idx].id === item.id);
        });

        showDetailView();
        renderDetail();
    }

    function renderDetail() {
        if (!selectedItem) return;
        var item = selectedItem;
        var steps = Array.isArray(item.steps) ? item.steps : [];
        var variables = Array.isArray(item.variables) ? item.variables : [];

        bwDetailTitle.textContent = item.name || '(untitled)';

        // Meta badges
        bwDetailMeta.innerHTML = '';
        var displayStepCount = editMode ? editSteps.length : steps.length;
        if (item.app_name) addBadge(bwDetailMeta, item.app_name, 'app');
        addBadge(bwDetailMeta, displayStepCount + ' steps', 'steps');
        if (variables.length > 0) addBadge(bwDetailMeta, variables.length + ' vars', 'vars');
        if (item.use_count > 0) addBadge(bwDetailMeta, item.use_count + ' runs', 'uses');

        // Content
        bwDetailContent.innerHTML = '';

        if (item.description) {
            addSection(bwDetailContent, 'Description', item.description);
        }

        if (item.start_url) {
            addUrlField(bwDetailContent, 'Start URL', item.start_url);
        }

        // Variables
        if (variables.length > 0) {
            var varSec = document.createElement('div');
            varSec.className = 'bw-section';
            var varLabel = document.createElement('div');
            varLabel.className = 'bw-section-label';
            varLabel.textContent = 'Variables (' + variables.length + ')';
            varSec.appendChild(varLabel);

            var varTable = document.createElement('table');
            varTable.className = 'bw-vars-table';
            var vHead = document.createElement('thead');
            vHead.innerHTML = '<tr><th>Name</th><th>Step</th><th>Field</th><th>Description</th></tr>';
            varTable.appendChild(vHead);
            var vBody = document.createElement('tbody');
            variables.forEach(function (v) {
                var tr = document.createElement('tr');
                tr.innerHTML =
                    '<td><span class="bw-var-name">{{' + esc(v.name || '') + '}}</span></td>' +
                    '<td>' + (v.step_index != null ? v.step_index : '-') + '</td>' +
                    '<td>' + esc(v.field || 'value') + '</td>' +
                    '<td>' + esc(v.description || '') + '</td>';
                vBody.appendChild(tr);
            });
            varTable.appendChild(vBody);
            varSec.appendChild(varTable);
            bwDetailContent.appendChild(varSec);
        }

        // Steps
        if (editMode) {
            renderEditableSteps(bwDetailContent);
        } else if (steps.length > 0) {
            var stepSec = document.createElement('div');
            stepSec.className = 'bw-section';
            var stepLabel = document.createElement('div');
            stepLabel.className = 'bw-section-label';
            stepLabel.textContent = 'Steps (' + steps.length + ')';
            stepSec.appendChild(stepLabel);

            var table = document.createElement('table');
            table.className = 'bw-steps-table';
            var thead = document.createElement('thead');
            thead.innerHTML = '<tr><th>#</th><th>Action</th><th>Target</th><th>Value</th></tr>';
            table.appendChild(thead);
            var tbody = document.createElement('tbody');

            steps.forEach(function (s, i) {
                var tr = document.createElement('tr');
                var sels = s.selectors || {};
                var target = sels.role_name
                    ? sels.role + '("' + sels.role_name + '")'
                    : sels.text
                        ? 'text("' + sels.text + '")'
                        : sels.css || '';

                var val = s.value || '';
                var isVar = val.indexOf('{{') >= 0;

                tr.innerHTML =
                    '<td>' + (s.seq != null ? s.seq : i) + '</td>' +
                    '<td><span class="bw-step-action">' + esc(s.action) + '</span></td>' +
                    '<td><span class="bw-step-selector" title="' + esc(target) + '">' + esc(truncate(target, 50)) + '</span></td>' +
                    '<td><span class="bw-step-value' + (isVar ? ' is-var' : '') + '" title="' + esc(val) + '">' + esc(truncate(val, 40)) + '</span></td>';
                tbody.appendChild(tr);
            });

            table.appendChild(tbody);
            stepSec.appendChild(table);
            bwDetailContent.appendChild(stepSec);
        }

        // Metadata
        addFieldRow(bwDetailContent, 'Created', formatDate(item.created_at));
        if (item.last_used_at) addFieldRow(bwDetailContent, 'Last Run', formatDate(item.last_used_at));
        addFieldRow(bwDetailContent, 'ID', item.id);
    }

    // ── Step Editor ──────────────────────────────────────────────────

    function onEditClick() {
        if (!selectedItem) return;
        editMode = true;
        editSteps = JSON.parse(JSON.stringify(
            Array.isArray(selectedItem.steps) ? selectedItem.steps : []
        ));
        updateEditButtons();
        renderDetail();
    }

    function onCancelEdit() {
        editMode = false;
        editSteps = [];
        updateEditButtons();
        renderDetail();
    }

    function onSaveSteps() {
        if (!selectedItem) return;
        bwSaveBtn.disabled = true;
        bwSaveBtn.textContent = 'Saving...';

        // Clean up: assign seq numbers, remove empty selectors
        editSteps.forEach(function (s, i) {
            s.seq = i;
            if (s.selectors) {
                Object.keys(s.selectors).forEach(function (k) {
                    if (!s.selectors[k]) delete s.selectors[k];
                });
                if (Object.keys(s.selectors).length === 0) delete s.selectors;
            }
        });

        apiFetch(API + '/' + selectedItem.id, {
            method: 'PATCH',
            body: { steps: editSteps }
        }).then(function (data) {
            bwSaveBtn.disabled = false;
            bwSaveBtn.textContent = 'Save';
            selectedItem.steps = data.steps || editSteps;
            editMode = false;
            editSteps = [];
            updateEditButtons();
            renderDetail();
            showToast('Steps saved', 'success');
            loadItems();
        }).catch(function (err) {
            bwSaveBtn.disabled = false;
            bwSaveBtn.textContent = 'Save';
            showToast('Save failed: ' + (err.message || err), 'error');
        });
    }

    function updateEditButtons() {
        bwEditBtn.style.display = editMode ? 'none' : '';
        bwDeleteBtn.style.display = editMode ? 'none' : '';
        bwSaveBtn.style.display = editMode ? '' : 'none';
        bwCancelEditBtn.style.display = editMode ? '' : 'none';
    }

    function renderEditableSteps(parent) {
        var sec = document.createElement('div');
        sec.className = 'bw-section';

        var label = document.createElement('div');
        label.className = 'bw-section-label';
        label.textContent = 'Steps (' + editSteps.length + ') — Editing';
        sec.appendChild(label);

        var container = document.createElement('div');

        editSteps.forEach(function (step, i) {
            container.appendChild(createStepCard(step, i));
        });

        sec.appendChild(container);

        // Add step button
        var addBtn = document.createElement('button');
        addBtn.className = 'bw-btn secondary bw-add-step';
        addBtn.textContent = '+ Add Step';
        addBtn.addEventListener('click', function () {
            editSteps.push({
                action: 'click',
                selectors: { css: '' },
                value: ''
            });
            renderDetail();
        });
        sec.appendChild(addBtn);

        parent.appendChild(sec);
    }

    function createStepCard(step, index) {
        var card = document.createElement('div');
        card.className = 'bw-edit-card';

        // Header
        var header = document.createElement('div');
        header.className = 'bw-edit-card-header';

        var num = document.createElement('span');
        num.className = 'bw-edit-card-num';
        num.textContent = 'Step ' + index;
        header.appendChild(num);

        var actions = document.createElement('div');
        actions.className = 'bw-edit-card-actions';

        if (index > 0) {
            actions.appendChild(makeActionBtn('\u2191', 'Move up', function () {
                moveStep(index, -1);
            }));
        }
        if (index < editSteps.length - 1) {
            actions.appendChild(makeActionBtn('\u2193', 'Move down', function () {
                moveStep(index, 1);
            }));
        }
        actions.appendChild(makeActionBtn('\u2398', 'Duplicate', function () {
            duplicateStep(index);
        }));
        var rmBtn = makeActionBtn('\u00D7', 'Remove step', function () {
            removeStep(index);
        });
        rmBtn.classList.add('remove');
        actions.appendChild(rmBtn);

        header.appendChild(actions);
        card.appendChild(header);

        // Body
        var body = document.createElement('div');
        body.className = 'bw-edit-card-body';

        var sels = step.selectors || {};

        addEditField(body, 'Action', 'select', step.action || 'click', function (val) {
            editSteps[index].action = val;
        }, STEP_ACTIONS);

        addEditField(body, 'Value', 'text', step.value || '', function (val) {
            editSteps[index].value = val;
        }, null, false, 'Text to type, key name, URL...');

        addEditField(body, 'CSS Selector', 'text', sels.css || '', function (val) {
            ensureSelectors(index);
            editSteps[index].selectors.css = val;
        }, null, true, 'button.submit, #login, [data-id="x"]');

        addEditField(body, 'Role', 'text', sels.role || '', function (val) {
            ensureSelectors(index);
            editSteps[index].selectors.role = val;
        }, null, false, 'button, link, textbox...');

        addEditField(body, 'Role Name', 'text', sels.role_name || '', function (val) {
            ensureSelectors(index);
            editSteps[index].selectors.role_name = val;
        }, null, false, 'Submit, Login...');

        addEditField(body, 'Text', 'text', sels.text || '', function (val) {
            ensureSelectors(index);
            editSteps[index].selectors.text = val;
        }, null, false, 'Visible text content');

        addEditField(body, 'URL', 'text', step.url || '', function (val) {
            editSteps[index].url = val;
        }, null, true, 'Page URL when step was recorded');

        card.appendChild(body);
        return card;
    }

    function addEditField(parent, label, type, value, onChange, options, fullWidth, placeholder) {
        var field = document.createElement('div');
        field.className = 'bw-edit-field' + (fullWidth ? ' full' : '');

        var lbl = document.createElement('label');
        lbl.className = 'bw-edit-label';
        lbl.textContent = label;
        field.appendChild(lbl);

        var input;
        if (type === 'select' && options) {
            input = document.createElement('select');
            input.className = 'bw-edit-select';
            options.forEach(function (opt) {
                var o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                if (opt === value) o.selected = true;
                input.appendChild(o);
            });
            input.addEventListener('change', function () { onChange(input.value); });
        } else {
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'bw-edit-input';
            input.value = value;
            if (placeholder) input.placeholder = placeholder;
            input.addEventListener('input', function () { onChange(input.value); });
        }

        field.appendChild(input);
        parent.appendChild(field);
    }

    function makeActionBtn(text, title, onClick) {
        var btn = document.createElement('button');
        btn.className = 'bw-edit-action-btn';
        btn.textContent = text;
        btn.title = title;
        btn.addEventListener('click', onClick);
        return btn;
    }

    function ensureSelectors(index) {
        if (!editSteps[index].selectors) editSteps[index].selectors = {};
    }

    function moveStep(index, direction) {
        var target = index + direction;
        if (target < 0 || target >= editSteps.length) return;
        var tmp = editSteps[index];
        editSteps[index] = editSteps[target];
        editSteps[target] = tmp;
        renderDetail();
    }

    function removeStep(index) {
        editSteps.splice(index, 1);
        renderDetail();
    }

    function duplicateStep(index) {
        var copy = JSON.parse(JSON.stringify(editSteps[index]));
        editSteps.splice(index + 1, 0, copy);
        renderDetail();
    }

    // ── Shared Helpers ───────────────────────────────────────────────

    function addBadge(parent, text, cls) {
        var span = document.createElement('span');
        span.className = 'bw-badge ' + (cls || '');
        span.textContent = text;
        parent.appendChild(span);
    }

    function addSection(parent, label, value) {
        var sec = document.createElement('div');
        sec.className = 'bw-section';
        var lbl = document.createElement('div');
        lbl.className = 'bw-section-label';
        lbl.textContent = label;
        sec.appendChild(lbl);
        var val = document.createElement('div');
        val.className = 'bw-section-value';
        val.textContent = value || '';
        if (!value) val.classList.add('empty');
        sec.appendChild(val);
        parent.appendChild(sec);
    }

    function addUrlField(parent, label, url) {
        var row = document.createElement('div');
        row.className = 'bw-field-row';
        var lbl = document.createElement('div');
        lbl.className = 'bw-field-label';
        lbl.textContent = label;
        row.appendChild(lbl);
        var val = document.createElement('div');
        val.className = 'bw-field-value';
        var link = document.createElement('a');
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = url;
        val.appendChild(link);
        row.appendChild(val);
        parent.appendChild(row);
    }

    function addFieldRow(parent, label, value) {
        var row = document.createElement('div');
        row.className = 'bw-field-row';
        var lbl = document.createElement('div');
        lbl.className = 'bw-field-label';
        lbl.textContent = label;
        row.appendChild(lbl);
        var val = document.createElement('div');
        val.className = 'bw-field-value';
        val.textContent = value || '';
        if (!value) val.classList.add('empty');
        row.appendChild(val);
        parent.appendChild(row);
    }

    // ── Search ───────────────────────────────────────────────────────

    function onSearchInput() {
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            searchQuery = bwSearch.value.trim();
            selectedItem = null;
            editMode = false;
            editSteps = [];
            updateEditButtons();
            showEmptyState();
            loadItems();
        }, 300);
    }

    // ── Delete ───────────────────────────────────────────────────────

    function onDeleteClick() {
        if (!selectedItem) return;
        deleteTarget = selectedItem;
        bwModalBody.textContent = 'Delete "' + (selectedItem.name || '(untitled)') + '"? This cannot be undone.';
        bwModalOverlay.style.display = '';
    }

    function hideDeleteModal() {
        bwModalOverlay.style.display = 'none';
        deleteTarget = null;
    }

    function onConfirmDelete() {
        if (!deleteTarget) return;
        bwModalConfirm.disabled = true;
        bwModalConfirm.textContent = 'Deleting...';

        apiFetch(API + '/' + deleteTarget.id, { method: 'DELETE' }).then(function () {
            bwModalConfirm.disabled = false;
            bwModalConfirm.textContent = 'Delete';
            hideDeleteModal();

            selectedItem = null;
            editMode = false;
            editSteps = [];
            updateEditButtons();
            showEmptyState();
            showToast('Deleted', 'success');
            loadItems();
        }).catch(function (err) {
            bwModalConfirm.disabled = false;
            bwModalConfirm.textContent = 'Delete';
            showToast('Delete failed: ' + (err.message || err), 'error');
        });
    }

    // ── Panel Visibility ─────────────────────────────────────────────

    function showEmptyState() {
        bwEmpty.style.display = '';
        bwDetailView.style.display = 'none';
    }

    function showDetailView() {
        bwEmpty.style.display = 'none';
        bwDetailView.style.display = '';
    }

    // ── Toast ────────────────────────────────────────────────────────

    var toastTimer = null;

    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        bwToast.textContent = message;
        bwToast.className = 'bw-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () {
            bwToast.classList.remove('show');
        }, 2500);
    }

    // ── API ──────────────────────────────────────────────────────────

    function apiFetch(url, options) {
        options = options || {};
        var fetchOpts = { method: options.method || 'GET', headers: {} };
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

    // ── Helpers ──────────────────────────────────────────────────────

    function formatDate(isoStr) {
        if (!isoStr) return '';
        try {
            var d = new Date(isoStr);
            if (isNaN(d.getTime())) return isoStr;
            var pad = function (n) { return n < 10 ? '0' + n : String(n); };
            return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
                   ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
        } catch (e) { return isoStr; }
    }

    function esc(str) {
        if (!str) return '';
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function truncate(str, max) {
        if (!str) return '';
        return str.length > max ? str.substring(0, max) + '...' : str;
    }

})();
