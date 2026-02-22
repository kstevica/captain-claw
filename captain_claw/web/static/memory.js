/* Captain Claw — Memory Browser */

(function () {
    'use strict';

    // ── Type Definitions ─────────────────────────────────────────────────────
    // Each memory type maps to its REST endpoints, display fields, and form schema.

    var MEMORY_TYPES = {
        todos: {
            label: 'To-dos',
            endpoint: '/api/todos',
            nameField: 'content',
            searchable: false,
            listSubtitle: function (item) {
                var parts = [];
                if (item.priority && item.priority !== 'normal') parts.push(item.priority);
                if (item.responsible) parts.push(item.responsible);
                return parts.join(' / ');
            },
            listBadges: function (item) {
                var b = [];
                if (item.status) b.push({ text: item.status.replace(/_/g, ' '), cls: item.status });
                if (item.priority && item.priority !== 'normal')
                    b.push({ text: item.priority, cls: item.priority });
                return b;
            },
            detailFields: [
                { key: 'content', label: 'Content', type: 'textarea' },
                { key: 'status', label: 'Status', type: 'select',
                  options: ['pending', 'in_progress', 'done', 'cancelled'] },
                { key: 'responsible', label: 'Responsible', type: 'select',
                  options: ['bot', 'human'] },
                { key: 'priority', label: 'Priority', type: 'select',
                  options: ['low', 'normal', 'high', 'urgent'] },
                { key: 'tags', label: 'Tags', type: 'text' },
                { key: 'source_session', label: 'Source Session', type: 'text', readonly: true },
                { key: 'target_session', label: 'Target Session', type: 'text' },
                { key: 'created_at', label: 'Created', type: 'text', readonly: true },
                { key: 'updated_at', label: 'Updated', type: 'text', readonly: true },
                { key: 'completed_at', label: 'Completed', type: 'text', readonly: true }
            ],
            createFields: ['content', 'status', 'responsible', 'priority', 'tags'],
            requiredFields: ['content']
        },
        contacts: {
            label: 'Contacts',
            endpoint: '/api/contacts',
            nameField: 'name',
            searchable: true,
            listSubtitle: function (item) {
                var parts = [];
                if (item.position) parts.push(item.position);
                if (item.organization) parts.push(item.organization);
                return parts.join(' @ ');
            },
            listBadges: function (item) {
                var b = [];
                if (item.relation) b.push({ text: item.relation, cls: '' });
                return b;
            },
            detailFields: [
                { key: 'name', label: 'Name', type: 'text' },
                { key: 'description', label: 'Description', type: 'textarea' },
                { key: 'position', label: 'Position', type: 'text' },
                { key: 'organization', label: 'Organization', type: 'text' },
                { key: 'relation', label: 'Relation', type: 'text' },
                { key: 'email', label: 'Email', type: 'text' },
                { key: 'phone', label: 'Phone', type: 'text' },
                { key: 'importance', label: 'Importance', type: 'number', min: 1, max: 10 },
                { key: 'tags', label: 'Tags', type: 'text' },
                { key: 'notes', label: 'Notes', type: 'textarea' },
                { key: 'privacy_tier', label: 'Privacy', type: 'select',
                  options: ['normal', 'private'] },
                { key: 'mention_count', label: 'Mentions', type: 'text', readonly: true },
                { key: 'created_at', label: 'Created', type: 'text', readonly: true },
                { key: 'updated_at', label: 'Updated', type: 'text', readonly: true }
            ],
            createFields: ['name', 'description', 'position', 'organization', 'relation',
                           'email', 'phone', 'importance', 'tags', 'notes', 'privacy_tier'],
            requiredFields: ['name']
        },
        scripts: {
            label: 'Scripts',
            endpoint: '/api/scripts',
            nameField: 'name',
            searchable: true,
            listSubtitle: function (item) {
                var parts = [];
                if (item.language) parts.push(item.language);
                if (item.file_path) parts.push(item.file_path);
                return parts.join(' - ');
            },
            listBadges: function (item) {
                var b = [];
                if (item.use_count > 0) b.push({ text: item.use_count + ' uses', cls: '' });
                return b;
            },
            detailFields: [
                { key: 'name', label: 'Name', type: 'text' },
                { key: 'file_path', label: 'File Path', type: 'text', mono: true },
                { key: 'language', label: 'Language', type: 'text' },
                { key: 'description', label: 'Description', type: 'textarea' },
                { key: 'purpose', label: 'Purpose', type: 'textarea' },
                { key: 'created_reason', label: 'Created Reason', type: 'textarea' },
                { key: 'tags', label: 'Tags', type: 'text' },
                { key: 'use_count', label: 'Use Count', type: 'text', readonly: true },
                { key: 'last_used_at', label: 'Last Used', type: 'text', readonly: true },
                { key: 'source_session', label: 'Source Session', type: 'text', readonly: true },
                { key: 'created_at', label: 'Created', type: 'text', readonly: true },
                { key: 'updated_at', label: 'Updated', type: 'text', readonly: true }
            ],
            createFields: ['name', 'file_path', 'language', 'description', 'purpose',
                           'created_reason', 'tags'],
            requiredFields: ['name', 'file_path']
        },
        apis: {
            label: 'APIs',
            endpoint: '/api/apis',
            nameField: 'name',
            searchable: true,
            listSubtitle: function (item) {
                return item.base_url || '';
            },
            listBadges: function (item) {
                var b = [];
                if (item.auth_type && item.auth_type !== 'none')
                    b.push({ text: item.auth_type, cls: '' });
                if (item.use_count > 0) b.push({ text: item.use_count + ' uses', cls: '' });
                return b;
            },
            detailFields: [
                { key: 'name', label: 'Name', type: 'text' },
                { key: 'base_url', label: 'Base URL', type: 'text', mono: true },
                { key: 'endpoints', label: 'Endpoints', type: 'textarea', mono: true },
                { key: 'auth_type', label: 'Auth Type', type: 'select',
                  options: ['none', 'bearer', 'api_key', 'basic'] },
                { key: 'credentials', label: 'Credentials', type: 'text', sensitive: true },
                { key: 'description', label: 'Description', type: 'textarea' },
                { key: 'purpose', label: 'Purpose', type: 'textarea' },
                { key: 'tags', label: 'Tags', type: 'text' },
                { key: 'use_count', label: 'Use Count', type: 'text', readonly: true },
                { key: 'last_used_at', label: 'Last Used', type: 'text', readonly: true },
                { key: 'source_session', label: 'Source Session', type: 'text', readonly: true },
                { key: 'created_at', label: 'Created', type: 'text', readonly: true },
                { key: 'updated_at', label: 'Updated', type: 'text', readonly: true }
            ],
            createFields: ['name', 'base_url', 'endpoints', 'auth_type', 'credentials',
                           'description', 'purpose', 'tags'],
            requiredFields: ['name', 'base_url']
        }
    };

    // ── State ────────────────────────────────────────────────────────────────

    var activeType = 'todos';
    var items = [];
    var selectedItem = null;
    var isEditing = false;
    var isCreating = false;
    var searchQuery = '';
    var searchTimeout = null;
    var deleteTarget = null;
    var credentialVisible = {};

    // ── DOM References ───────────────────────────────────────────────────────

    var catItems       = document.querySelectorAll('.mem-cat-item');
    var memSearch       = document.getElementById('memSearch');
    var memAddBtn       = document.getElementById('memAddBtn');
    var memItemList     = document.getElementById('memItemList');
    var memLoading      = document.getElementById('memLoading');
    var memEmpty        = document.getElementById('memEmpty');
    var memDetailView   = document.getElementById('memDetailView');
    var memDetailTitle  = document.getElementById('memDetailTitle');
    var memDetailFields = document.getElementById('memDetailFields');
    var memEditBtn      = document.getElementById('memEditBtn');
    var memDeleteBtn    = document.getElementById('memDeleteBtn');
    var memForm         = document.getElementById('memForm');
    var memFormTitle    = document.getElementById('memFormTitle');
    var memFormFields   = document.getElementById('memFormFields');
    var memCancelBtn    = document.getElementById('memCancelBtn');
    var memSaveBtn      = document.getElementById('memSaveBtn');
    var memModalOverlay = document.getElementById('memModalOverlay');
    var memModalBody    = document.getElementById('memModalBody');
    var memModalCancel  = document.getElementById('memModalCancel');
    var memModalConfirm = document.getElementById('memModalConfirm');
    var memToast        = document.getElementById('memToast');

    var countEls = {
        todos:    document.getElementById('countTodos'),
        contacts: document.getElementById('countContacts'),
        scripts:  document.getElementById('countScripts'),
        apis:     document.getElementById('countApis')
    };

    // ── Init ─────────────────────────────────────────────────────────────────

    init();

    function init() {
        // Category clicks
        for (var i = 0; i < catItems.length; i++) {
            catItems[i].addEventListener('click', onCategoryClick);
        }

        // Search
        memSearch.addEventListener('input', onSearchInput);

        // CRUD buttons
        memAddBtn.addEventListener('click', onAddClick);
        memEditBtn.addEventListener('click', onEditClick);
        memDeleteBtn.addEventListener('click', onDeleteClick);
        memCancelBtn.addEventListener('click', onCancelForm);
        memSaveBtn.addEventListener('click', onSaveForm);
        memModalCancel.addEventListener('click', hideDeleteModal);
        memModalConfirm.addEventListener('click', onConfirmDelete);

        // Keyboard shortcut: Escape to close modal/form
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                if (memModalOverlay.style.display !== 'none') {
                    hideDeleteModal();
                } else if (isEditing) {
                    onCancelForm();
                }
            }
        });

        // Load
        loadAllCounts();
        loadItems();
    }

    // ── Category Switching ───────────────────────────────────────────────────

    function onCategoryClick(e) {
        var el = e.currentTarget;
        var type = el.getAttribute('data-type');
        if (type === activeType) return;

        activeType = type;
        selectedItem = null;
        searchQuery = '';
        memSearch.value = '';
        isEditing = false;
        isCreating = false;
        credentialVisible = {};

        // Update selected class
        for (var i = 0; i < catItems.length; i++) {
            catItems[i].classList.toggle('selected', catItems[i] === el);
        }

        // Reset right panel
        showEmptyState();
        loadItems();
    }

    // ── Loading Data ─────────────────────────────────────────────────────────

    function loadAllCounts() {
        var types = ['todos', 'contacts', 'scripts', 'apis'];
        types.forEach(function (type) {
            var cfg = MEMORY_TYPES[type];
            apiFetch(cfg.endpoint).then(function (data) {
                if (Array.isArray(data) && countEls[type]) {
                    countEls[type].textContent = data.length;
                }
            }).catch(function () {});
        });
    }

    function loadItems() {
        memLoading.style.display = '';
        memItemList.querySelectorAll('.mem-list-item, .mem-list-empty').forEach(function (el) {
            el.remove();
        });

        var cfg = MEMORY_TYPES[activeType];
        var url = cfg.endpoint;

        if (searchQuery && cfg.searchable) {
            url = cfg.endpoint + '/search?q=' + encodeURIComponent(searchQuery);
        }

        apiFetch(url).then(function (data) {
            memLoading.style.display = 'none';
            if (!Array.isArray(data)) { data = []; }

            // Client-side filter for non-searchable types
            if (searchQuery && !cfg.searchable) {
                var q = searchQuery.toLowerCase();
                data = data.filter(function (item) {
                    var name = String(item[cfg.nameField] || '').toLowerCase();
                    var tags = String(item.tags || '').toLowerCase();
                    return name.indexOf(q) >= 0 || tags.indexOf(q) >= 0;
                });
            }

            items = data;

            // Update count for this type
            if (!searchQuery && countEls[activeType]) {
                countEls[activeType].textContent = items.length;
            }

            renderList();
        }).catch(function () {
            memLoading.style.display = 'none';
            items = [];
            renderList();
        });
    }

    // ── Rendering: List ──────────────────────────────────────────────────────

    function renderList() {
        // Clear existing
        memItemList.querySelectorAll('.mem-list-item, .mem-list-empty').forEach(function (el) {
            el.remove();
        });

        if (items.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'mem-list-empty';
            empty.textContent = searchQuery ? 'No matches found' : 'No items yet';
            if (!searchQuery) {
                var sub = document.createElement('div');
                sub.className = 'mem-list-empty-sub';
                sub.textContent = 'Click + to add one';
                empty.appendChild(sub);
            }
            memItemList.appendChild(empty);
            return;
        }

        var cfg = MEMORY_TYPES[activeType];

        items.forEach(function (item) {
            var el = document.createElement('div');
            el.className = 'mem-list-item';
            if (selectedItem && selectedItem.id === item.id) {
                el.classList.add('selected');
            }

            // Name
            var nameEl = document.createElement('div');
            nameEl.className = 'mem-list-item-name';
            nameEl.textContent = item[cfg.nameField] || '(untitled)';
            el.appendChild(nameEl);

            // Subtitle
            var subtitle = cfg.listSubtitle(item);
            if (subtitle) {
                var subEl = document.createElement('div');
                subEl.className = 'mem-list-item-sub';
                subEl.textContent = subtitle;
                el.appendChild(subEl);
            }

            // Badges
            var badges = cfg.listBadges(item);
            if (badges.length > 0) {
                var badgeContainer = document.createElement('div');
                badgeContainer.className = 'mem-list-item-badges';
                badges.forEach(function (badge) {
                    var span = document.createElement('span');
                    span.className = 'mem-badge';
                    if (badge.cls) span.classList.add(badge.cls);
                    span.textContent = badge.text;
                    badgeContainer.appendChild(span);
                });
                el.appendChild(badgeContainer);
            }

            el.addEventListener('click', function () { onItemClick(item); });
            memItemList.appendChild(el);
        });
    }

    // ── Rendering: Detail ────────────────────────────────────────────────────

    function onItemClick(item) {
        selectedItem = item;
        isEditing = false;
        isCreating = false;

        // Highlight in list
        memItemList.querySelectorAll('.mem-list-item').forEach(function (el, idx) {
            el.classList.toggle('selected', items[idx] && items[idx].id === item.id);
        });

        showDetailView();
        renderDetail();
    }

    function renderDetail() {
        if (!selectedItem) return;
        var cfg = MEMORY_TYPES[activeType];

        memDetailTitle.textContent = selectedItem[cfg.nameField] || '(untitled)';
        memDetailFields.innerHTML = '';

        cfg.detailFields.forEach(function (field) {
            var val = selectedItem[field.key];

            var row = document.createElement('div');
            row.className = 'mem-field-row';

            var label = document.createElement('div');
            label.className = 'mem-field-label';
            label.textContent = field.label;
            row.appendChild(label);

            var value = document.createElement('div');
            value.className = 'mem-field-value';

            // Format value
            if (val === null || val === undefined || val === '') {
                value.textContent = '—';
                value.classList.add('empty');
            } else if (field.key === 'tags' && val) {
                value.innerHTML = '';
                var tagsDiv = document.createElement('div');
                tagsDiv.className = 'mem-tags';
                String(val).split(',').forEach(function (t) {
                    t = t.trim();
                    if (!t) return;
                    var tagSpan = document.createElement('span');
                    tagSpan.className = 'mem-tag';
                    tagSpan.textContent = t;
                    tagsDiv.appendChild(tagSpan);
                });
                value.appendChild(tagsDiv);
            } else if (field.key.endsWith('_at') && val) {
                value.textContent = formatDate(val);
            } else if (field.sensitive && val) {
                var masked = credentialVisible[field.key] ? val : '••••••••';
                value.textContent = masked;
                value.style.cursor = 'pointer';
                value.title = 'Click to toggle visibility';
                (function (k, v) {
                    value.addEventListener('click', function () {
                        credentialVisible[k] = !credentialVisible[k];
                        renderDetail();
                    });
                })(field.key, val);
            } else if (field.type === 'select') {
                var badge = document.createElement('span');
                badge.className = 'mem-badge ' + String(val).replace(/_/g, '-');
                badge.textContent = String(val).replace(/_/g, ' ');
                value.appendChild(badge);
            } else {
                value.textContent = String(val);
                if (field.mono) value.classList.add('mono');
            }

            row.appendChild(value);
            memDetailFields.appendChild(row);
        });
    }

    // ── Search ───────────────────────────────────────────────────────────────

    function onSearchInput() {
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            searchQuery = memSearch.value.trim();
            selectedItem = null;
            showEmptyState();
            loadItems();
        }, 300);
    }

    // ── Add (Create) ─────────────────────────────────────────────────────────

    function onAddClick() {
        isCreating = true;
        isEditing = true;
        selectedItem = null;

        // Deselect list items
        memItemList.querySelectorAll('.mem-list-item').forEach(function (el) {
            el.classList.remove('selected');
        });

        showFormView();
        memFormTitle.textContent = 'New ' + MEMORY_TYPES[activeType].label.replace(/s$/, '');
        renderForm(null);
    }

    // ── Edit ─────────────────────────────────────────────────────────────────

    function onEditClick() {
        if (!selectedItem) return;
        isEditing = true;
        isCreating = false;

        showFormView();
        memFormTitle.textContent = 'Edit ' + MEMORY_TYPES[activeType].label.replace(/s$/, '');
        renderForm(selectedItem);
    }

    // ── Form Rendering ───────────────────────────────────────────────────────

    function renderForm(item) {
        memFormFields.innerHTML = '';
        var cfg = MEMORY_TYPES[activeType];

        var fieldsToShow = isCreating ? cfg.createFields : cfg.detailFields.map(function (f) {
            return f.key;
        });

        cfg.detailFields.forEach(function (field) {
            // Skip fields not in our set
            if (fieldsToShow.indexOf(field.key) < 0) return;
            // Skip readonly fields in create mode
            if (isCreating && field.readonly) return;

            var group = document.createElement('div');
            group.className = 'mem-form-group';

            var lbl = document.createElement('label');
            lbl.className = 'mem-form-label';
            lbl.textContent = field.label;
            if (cfg.requiredFields.indexOf(field.key) >= 0) {
                var req = document.createElement('span');
                req.className = 'required';
                req.textContent = '*';
                lbl.appendChild(req);
            }
            group.appendChild(lbl);

            var currentVal = item ? (item[field.key] !== null && item[field.key] !== undefined ? String(item[field.key]) : '') : '';
            var input;

            if (field.type === 'select') {
                input = document.createElement('select');
                input.className = 'mem-select';
                (field.options || []).forEach(function (opt) {
                    var option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt.replace(/_/g, ' ');
                    if (currentVal === opt) option.selected = true;
                    input.appendChild(option);
                });
            } else if (field.type === 'textarea') {
                input = document.createElement('textarea');
                input.className = 'mem-textarea';
                input.value = currentVal;
                input.placeholder = field.label + '...';
            } else if (field.type === 'number') {
                input = document.createElement('input');
                input.className = 'mem-input';
                input.type = 'number';
                input.value = currentVal;
                if (field.min !== undefined) input.min = field.min;
                if (field.max !== undefined) input.max = field.max;
            } else {
                input = document.createElement('input');
                input.className = 'mem-input';
                input.type = field.sensitive ? 'text' : 'text';
                input.value = currentVal;
                input.placeholder = field.label + '...';
            }

            input.setAttribute('data-field', field.key);
            if (field.readonly && !isCreating) {
                input.disabled = true;
            }

            group.appendChild(input);
            memFormFields.appendChild(group);
        });

        // Focus first editable input
        var firstInput = memFormFields.querySelector('input:not([disabled]), textarea:not([disabled]), select:not([disabled])');
        if (firstInput) setTimeout(function () { firstInput.focus(); }, 50);
    }

    // ── Save ─────────────────────────────────────────────────────────────────

    function onSaveForm() {
        var cfg = MEMORY_TYPES[activeType];
        var body = {};

        // Collect form values
        var inputs = memFormFields.querySelectorAll('[data-field]');
        inputs.forEach(function (inp) {
            if (inp.disabled) return;
            var key = inp.getAttribute('data-field');
            var val = inp.value.trim();
            if (val !== '') {
                // Convert number fields
                var fieldDef = cfg.detailFields.find(function (f) { return f.key === key; });
                if (fieldDef && fieldDef.type === 'number' && val) {
                    body[key] = parseInt(val, 10);
                } else {
                    body[key] = val;
                }
            }
        });

        // Validate required
        for (var i = 0; i < cfg.requiredFields.length; i++) {
            var reqKey = cfg.requiredFields[i];
            if (!body[reqKey]) {
                showToast('Required: ' + reqKey, 'error');
                var el = memFormFields.querySelector('[data-field="' + reqKey + '"]');
                if (el) el.focus();
                return;
            }
        }

        memSaveBtn.disabled = true;
        memSaveBtn.textContent = 'Saving...';

        var url, method;
        if (isCreating) {
            url = cfg.endpoint;
            method = 'POST';
        } else {
            url = cfg.endpoint + '/' + selectedItem.id;
            method = 'PATCH';
        }

        apiFetch(url, { method: method, body: body }).then(function (saved) {
            memSaveBtn.disabled = false;
            memSaveBtn.textContent = 'Save';

            if (saved && saved.error) {
                showToast(saved.error, 'error');
                return;
            }

            selectedItem = saved;
            isEditing = false;
            isCreating = false;

            showToast(isCreating ? 'Created' : 'Saved', 'success');
            loadAllCounts();
            loadItems();

            // Show detail for saved item
            setTimeout(function () {
                showDetailView();
                renderDetail();
            }, 100);
        }).catch(function (err) {
            memSaveBtn.disabled = false;
            memSaveBtn.textContent = 'Save';
            showToast('Save failed: ' + (err.message || err), 'error');
        });
    }

    // ── Delete ───────────────────────────────────────────────────────────────

    function onDeleteClick() {
        if (!selectedItem) return;
        deleteTarget = selectedItem;
        var cfg = MEMORY_TYPES[activeType];
        var name = selectedItem[cfg.nameField] || '(untitled)';
        memModalBody.textContent = 'Delete "' + name + '"? This cannot be undone.';
        memModalOverlay.style.display = '';
    }

    function hideDeleteModal() {
        memModalOverlay.style.display = 'none';
        deleteTarget = null;
    }

    function onConfirmDelete() {
        if (!deleteTarget) return;
        var cfg = MEMORY_TYPES[activeType];
        var url = cfg.endpoint + '/' + deleteTarget.id;

        memModalConfirm.disabled = true;
        memModalConfirm.textContent = 'Deleting...';

        apiFetch(url, { method: 'DELETE' }).then(function () {
            memModalConfirm.disabled = false;
            memModalConfirm.textContent = 'Delete';
            hideDeleteModal();

            selectedItem = null;
            showEmptyState();
            showToast('Deleted', 'success');
            loadAllCounts();
            loadItems();
        }).catch(function (err) {
            memModalConfirm.disabled = false;
            memModalConfirm.textContent = 'Delete';
            showToast('Delete failed: ' + (err.message || err), 'error');
        });
    }

    // ── Cancel Form ──────────────────────────────────────────────────────────

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

    // ── Panel Visibility Helpers ─────────────────────────────────────────────

    function showEmptyState() {
        memEmpty.style.display = '';
        memDetailView.style.display = 'none';
        memForm.style.display = 'none';
    }

    function showDetailView() {
        memEmpty.style.display = 'none';
        memDetailView.style.display = '';
        memForm.style.display = 'none';
    }

    function showFormView() {
        memEmpty.style.display = 'none';
        memDetailView.style.display = 'none';
        memForm.style.display = '';
    }

    // ── Toast Notification ───────────────────────────────────────────────────

    var toastTimer = null;

    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        memToast.textContent = message;
        memToast.className = 'mem-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () {
            memToast.classList.remove('show');
        }, 2500);
    }

    // ── API Utility ──────────────────────────────────────────────────────────

    function apiFetch(url, options) {
        options = options || {};
        var fetchOpts = {
            method: options.method || 'GET',
            headers: {}
        };

        if (options.body) {
            fetchOpts.headers['Content-Type'] = 'application/json';
            fetchOpts.body = JSON.stringify(options.body);
        }

        return fetch(url, fetchOpts).then(function (res) {
            if (res.status === 204) return { ok: true };
            return res.json().then(function (data) {
                if (!res.ok && data.error) {
                    throw new Error(data.error);
                }
                return data;
            });
        });
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    function formatDate(isoStr) {
        if (!isoStr) return '—';
        try {
            var d = new Date(isoStr);
            if (isNaN(d.getTime())) return isoStr;
            var pad = function (n) { return n < 10 ? '0' + n : String(n); };
            return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
                   ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
        } catch (e) {
            return isoStr;
        }
    }

})();
