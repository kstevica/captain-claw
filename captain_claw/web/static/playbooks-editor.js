/* Captain Claw — Playbooks Editor */

(function () {
    'use strict';

    var API = '/api/playbooks';

    // ── State ────────────────────────────────────────────────────────
    var items = [];
    var selectedItem = null;
    var isEditing = false;
    var isCreating = false;
    var searchQuery = '';
    var filterType = '';
    var searchTimeout = null;
    var deleteTarget = null;

    // ── DOM ──────────────────────────────────────────────────────────
    var pbSearch       = document.getElementById('pbSearch');
    var pbAddBtn       = document.getElementById('pbAddBtn');
    var pbFilterType   = document.getElementById('pbFilterType');
    var pbCount        = document.getElementById('pbCount');
    var pbItemList     = document.getElementById('pbItemList');
    var pbLoading      = document.getElementById('pbLoading');
    var pbEmpty        = document.getElementById('pbEmpty');
    var pbDetailView   = document.getElementById('pbDetailView');
    var pbDetailTitle  = document.getElementById('pbDetailTitle');
    var pbDetailMeta   = document.getElementById('pbDetailMeta');
    var pbDetailContent = document.getElementById('pbDetailContent');
    var pbEditBtn      = document.getElementById('pbEditBtn');
    var pbDeleteBtn    = document.getElementById('pbDeleteBtn');
    var pbForm         = document.getElementById('pbForm');
    var pbFormTitle    = document.getElementById('pbFormTitle');
    var pbCancelBtn    = document.getElementById('pbCancelBtn');
    var pbSaveBtn      = document.getElementById('pbSaveBtn');
    var pbModalOverlay = document.getElementById('pbModalOverlay');
    var pbModalBody    = document.getElementById('pbModalBody');
    var pbModalCancel  = document.getElementById('pbModalCancel');
    var pbModalConfirm = document.getElementById('pbModalConfirm');
    var pbToast        = document.getElementById('pbToast');

    // Form fields
    var fName       = document.getElementById('fName');
    var fTaskType   = document.getElementById('fTaskType');
    var fRating     = document.getElementById('fRating');
    var fTags       = document.getElementById('fTags');
    var fTrigger    = document.getElementById('fTrigger');
    var fDoPattern  = document.getElementById('fDoPattern');
    var fDontPattern = document.getElementById('fDontPattern');
    var fExamples   = document.getElementById('fExamples');
    var fScriptIds  = document.getElementById('fScriptIds');
    var fReasoning  = document.getElementById('fReasoning');

    // ── Init ─────────────────────────────────────────────────────────
    init();

    function init() {
        pbSearch.addEventListener('input', onSearchInput);
        pbFilterType.addEventListener('change', onFilterChange);
        pbAddBtn.addEventListener('click', onAddClick);
        pbEditBtn.addEventListener('click', onEditClick);
        pbDeleteBtn.addEventListener('click', onDeleteClick);
        pbCancelBtn.addEventListener('click', onCancelForm);
        pbSaveBtn.addEventListener('click', onSaveForm);
        pbModalCancel.addEventListener('click', hideDeleteModal);
        pbModalConfirm.addEventListener('click', onConfirmDelete);

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                if (pbModalOverlay.style.display !== 'none') {
                    hideDeleteModal();
                } else if (isEditing) {
                    onCancelForm();
                }
            }
        });

        loadItems();
    }

    // ── Data Loading ─────────────────────────────────────────────────

    function loadItems() {
        pbLoading.style.display = '';
        clearList();

        var url = API;
        var params = [];

        if (searchQuery) {
            url = API + '/search';
            params.push('q=' + encodeURIComponent(searchQuery));
        }
        if (filterType) {
            params.push('task_type=' + encodeURIComponent(filterType));
        }
        if (params.length) {
            url += (url.indexOf('?') >= 0 ? '&' : '?') + params.join('&');
        }

        apiFetch(url).then(function (data) {
            pbLoading.style.display = 'none';
            if (!Array.isArray(data)) data = [];
            items = data;
            pbCount.textContent = items.length;
            renderList();
        }).catch(function () {
            pbLoading.style.display = 'none';
            items = [];
            pbCount.textContent = '0';
            renderList();
        });
    }

    // ── List Rendering ───────────────────────────────────────────────

    function clearList() {
        pbItemList.querySelectorAll('.pb-list-item, .pb-list-empty').forEach(function (el) {
            el.remove();
        });
    }

    function renderList() {
        clearList();

        if (items.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'pb-list-empty';
            empty.textContent = searchQuery ? 'No matches found' : 'No playbooks yet';
            if (!searchQuery) {
                var sub = document.createElement('div');
                sub.className = 'pb-list-empty-sub';
                sub.textContent = 'Click + to add one, or rate a session in chat';
                empty.appendChild(sub);
            }
            pbItemList.appendChild(empty);
            return;
        }

        items.forEach(function (item) {
            var el = document.createElement('div');
            el.className = 'pb-list-item';
            if (selectedItem && selectedItem.id === item.id) {
                el.classList.add('selected');
            }

            var nameEl = document.createElement('div');
            nameEl.className = 'pb-list-item-name';
            nameEl.textContent = item.name || '(untitled)';
            el.appendChild(nameEl);

            if (item.trigger_description) {
                var trigEl = document.createElement('div');
                trigEl.className = 'pb-list-item-trigger';
                trigEl.textContent = item.trigger_description;
                el.appendChild(trigEl);
            }

            var badges = document.createElement('div');
            badges.className = 'pb-list-item-badges';

            var typeBadge = document.createElement('span');
            typeBadge.className = 'pb-badge type';
            typeBadge.textContent = item.task_type || 'other';
            badges.appendChild(typeBadge);

            var ratingBadge = document.createElement('span');
            ratingBadge.className = 'pb-badge ' + (item.rating || 'good');
            ratingBadge.textContent = item.rating || 'good';
            badges.appendChild(ratingBadge);

            if (item.use_count > 0) {
                var usesBadge = document.createElement('span');
                usesBadge.className = 'pb-badge uses';
                usesBadge.textContent = item.use_count + ' uses';
                badges.appendChild(usesBadge);
            }

            el.appendChild(badges);
            el.addEventListener('click', function () { onItemClick(item); });
            pbItemList.appendChild(el);
        });
    }

    // ── Detail Rendering ─────────────────────────────────────────────

    function onItemClick(item) {
        selectedItem = item;
        isEditing = false;
        isCreating = false;

        pbItemList.querySelectorAll('.pb-list-item').forEach(function (el, idx) {
            el.classList.toggle('selected', items[idx] && items[idx].id === item.id);
        });

        showDetailView();
        renderDetail();
    }

    function renderDetail() {
        if (!selectedItem) return;
        var item = selectedItem;

        pbDetailTitle.textContent = item.name || '(untitled)';

        // Meta badges
        pbDetailMeta.innerHTML = '';
        addBadge(pbDetailMeta, item.task_type || 'other', 'type');
        addBadge(pbDetailMeta, item.rating || 'good', item.rating || 'good');
        if (item.use_count > 0) addBadge(pbDetailMeta, item.use_count + ' uses', 'uses');

        // Content sections
        pbDetailContent.innerHTML = '';

        if (item.trigger_description) {
            addSection(pbDetailContent, 'Trigger', item.trigger_description);
        }
        if (item.do_pattern) {
            addCodeSection(pbDetailContent, 'DO (recommended)', item.do_pattern, 'do-pattern');
        }
        if (item.dont_pattern) {
            addCodeSection(pbDetailContent, "DON'T (avoid)", item.dont_pattern, 'dont-pattern');
        }
        if (item.examples) {
            addCodeSection(pbDetailContent, 'Examples', item.examples, 'examples');
        }
        if (item.script_ids) {
            addSection(pbDetailContent, 'Linked Scripts', item.script_ids);
        }
        if (item.reasoning) {
            addSection(pbDetailContent, 'Reasoning', item.reasoning);
        }
        if (item.tags) {
            addTagsSection(pbDetailContent, 'Tags', item.tags);
        }

        // Metadata fields
        addFieldRow(pbDetailContent, 'Created', formatDate(item.created_at));
        if (item.last_used_at) addFieldRow(pbDetailContent, 'Last Used', formatDate(item.last_used_at));
        if (item.source_session) addFieldRow(pbDetailContent, 'Source Session', item.source_session);
        addFieldRow(pbDetailContent, 'ID', item.id);
    }

    function addBadge(parent, text, cls) {
        var span = document.createElement('span');
        span.className = 'pb-badge ' + (cls || '');
        span.textContent = text;
        parent.appendChild(span);
    }

    function addSection(parent, label, value) {
        var sec = document.createElement('div');
        sec.className = 'pb-section';
        var lbl = document.createElement('div');
        lbl.className = 'pb-section-label';
        lbl.textContent = label;
        sec.appendChild(lbl);
        var val = document.createElement('div');
        val.className = 'pb-section-value';
        val.textContent = value || '';
        if (!value) val.classList.add('empty');
        sec.appendChild(val);
        parent.appendChild(sec);
    }

    function addCodeSection(parent, label, code, extraCls) {
        var sec = document.createElement('div');
        sec.className = 'pb-section';
        var lbl = document.createElement('div');
        lbl.className = 'pb-section-label';
        lbl.textContent = label;
        sec.appendChild(lbl);
        var pre = document.createElement('div');
        pre.className = 'pb-code-block ' + (extraCls || '');
        pre.textContent = code;
        sec.appendChild(pre);
        parent.appendChild(sec);
    }

    function addTagsSection(parent, label, tags) {
        var sec = document.createElement('div');
        sec.className = 'pb-section';
        var lbl = document.createElement('div');
        lbl.className = 'pb-section-label';
        lbl.textContent = label;
        sec.appendChild(lbl);
        var container = document.createElement('div');
        container.className = 'pb-tags';
        String(tags).split(',').forEach(function (t) {
            t = t.trim();
            if (!t) return;
            var span = document.createElement('span');
            span.className = 'pb-tag';
            span.textContent = t;
            container.appendChild(span);
        });
        sec.appendChild(container);
        parent.appendChild(sec);
    }

    function addFieldRow(parent, label, value) {
        var row = document.createElement('div');
        row.className = 'pb-field-row';
        var lbl = document.createElement('div');
        lbl.className = 'pb-field-label';
        lbl.textContent = label;
        row.appendChild(lbl);
        var val = document.createElement('div');
        val.className = 'pb-field-value';
        val.textContent = value || '—';
        if (!value) val.classList.add('empty');
        row.appendChild(val);
        parent.appendChild(row);
    }

    // ── Search / Filter ──────────────────────────────────────────────

    function onSearchInput() {
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            searchQuery = pbSearch.value.trim();
            selectedItem = null;
            showEmptyState();
            loadItems();
        }, 300);
    }

    function onFilterChange() {
        filterType = pbFilterType.value;
        selectedItem = null;
        showEmptyState();
        loadItems();
    }

    // ── Add ──────────────────────────────────────────────────────────

    function onAddClick() {
        isCreating = true;
        isEditing = true;
        selectedItem = null;

        pbItemList.querySelectorAll('.pb-list-item').forEach(function (el) {
            el.classList.remove('selected');
        });

        showFormView();
        pbFormTitle.textContent = 'New Playbook';
        populateForm(null);
    }

    // ── Edit ─────────────────────────────────────────────────────────

    function onEditClick() {
        if (!selectedItem) return;
        isEditing = true;
        isCreating = false;

        showFormView();
        pbFormTitle.textContent = 'Edit Playbook';
        populateForm(selectedItem);
    }

    function populateForm(item) {
        fName.value = item ? item.name || '' : '';
        fTaskType.value = item ? item.task_type || 'other' : 'batch-processing';
        fRating.value = item ? item.rating || 'good' : 'good';
        fTags.value = item ? item.tags || '' : '';
        fTrigger.value = item ? item.trigger_description || '' : '';
        fDoPattern.value = item ? item.do_pattern || '' : '';
        fDontPattern.value = item ? item.dont_pattern || '' : '';
        fExamples.value = item ? item.examples || '' : '';
        fScriptIds.value = item ? item.script_ids || '' : '';
        fReasoning.value = item ? item.reasoning || '' : '';

        setTimeout(function () { fName.focus(); }, 50);
    }

    // ── Save ─────────────────────────────────────────────────────────

    function onSaveForm() {
        var name = fName.value.trim();
        var taskType = fTaskType.value;
        var doPattern = fDoPattern.value.trim();
        var dontPattern = fDontPattern.value.trim();

        if (!name) { showToast('Name is required', 'error'); fName.focus(); return; }
        if (!doPattern && !dontPattern) {
            showToast('At least one DO or DON\'T pattern is required', 'error');
            fDoPattern.focus();
            return;
        }

        var body = {
            name: name,
            task_type: taskType,
            rating: fRating.value,
            do_pattern: doPattern,
            dont_pattern: dontPattern,
            trigger_description: fTrigger.value.trim(),
            reasoning: fReasoning.value.trim(),
            tags: fTags.value.trim(),
            examples: fExamples.value.trim(),
            script_ids: fScriptIds.value.trim()
        };

        pbSaveBtn.disabled = true;
        pbSaveBtn.textContent = 'Saving...';

        var url, method;
        if (isCreating) {
            url = API;
            method = 'POST';
        } else {
            url = API + '/' + selectedItem.id;
            method = 'PATCH';
        }

        apiFetch(url, { method: method, body: body }).then(function (saved) {
            pbSaveBtn.disabled = false;
            pbSaveBtn.textContent = 'Save';

            if (saved && saved.error) {
                showToast(saved.error, 'error');
                return;
            }

            selectedItem = saved;
            isEditing = false;
            isCreating = false;

            showToast('Saved', 'success');
            loadItems();

            setTimeout(function () {
                showDetailView();
                renderDetail();
            }, 100);
        }).catch(function (err) {
            pbSaveBtn.disabled = false;
            pbSaveBtn.textContent = 'Save';
            showToast('Save failed: ' + (err.message || err), 'error');
        });
    }

    // ── Delete ───────────────────────────────────────────────────────

    function onDeleteClick() {
        if (!selectedItem) return;
        deleteTarget = selectedItem;
        pbModalBody.textContent = 'Delete "' + (selectedItem.name || '(untitled)') + '"? This cannot be undone.';
        pbModalOverlay.style.display = '';
    }

    function hideDeleteModal() {
        pbModalOverlay.style.display = 'none';
        deleteTarget = null;
    }

    function onConfirmDelete() {
        if (!deleteTarget) return;
        pbModalConfirm.disabled = true;
        pbModalConfirm.textContent = 'Deleting...';

        apiFetch(API + '/' + deleteTarget.id, { method: 'DELETE' }).then(function () {
            pbModalConfirm.disabled = false;
            pbModalConfirm.textContent = 'Delete';
            hideDeleteModal();

            selectedItem = null;
            showEmptyState();
            showToast('Deleted', 'success');
            loadItems();
        }).catch(function (err) {
            pbModalConfirm.disabled = false;
            pbModalConfirm.textContent = 'Delete';
            showToast('Delete failed: ' + (err.message || err), 'error');
        });
    }

    // ── Cancel ───────────────────────────────────────────────────────

    function onCancelForm() {
        isEditing = false;
        isCreating = false;
        if (selectedItem) {
            showDetailView();
            renderDetail();
        } else {
            showEmptyState();
        }
    }

    // ── Panel Visibility ─────────────────────────────────────────────

    function showEmptyState() {
        pbEmpty.style.display = '';
        pbDetailView.style.display = 'none';
        pbForm.style.display = 'none';
    }

    function showDetailView() {
        pbEmpty.style.display = 'none';
        pbDetailView.style.display = '';
        pbForm.style.display = 'none';
    }

    function showFormView() {
        pbEmpty.style.display = 'none';
        pbDetailView.style.display = 'none';
        pbForm.style.display = '';
    }

    // ── Toast ────────────────────────────────────────────────────────

    var toastTimer = null;

    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        pbToast.textContent = message;
        pbToast.className = 'pb-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () {
            pbToast.classList.remove('show');
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
        if (!isoStr) return '—';
        try {
            var d = new Date(isoStr);
            if (isNaN(d.getTime())) return isoStr;
            var pad = function (n) { return n < 10 ? '0' + n : String(n); };
            return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
                   ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
        } catch (e) { return isoStr; }
    }

})();
