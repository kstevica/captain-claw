/* Captain Claw — Insights Browser */

(function () {
    'use strict';

    // ── DOM refs ────────────────────────────────────────────
    var searchInput = document.getElementById('searchInput');
    var categoryFilter = document.getElementById('categoryFilter');
    var addBtn = document.getElementById('addBtn');
    var countLabel = document.getElementById('countLabel');
    var listEl = document.getElementById('insightsList');

    var overlay = document.getElementById('detailOverlay');
    var detailTitle = document.getElementById('detailTitle');
    var detailContent = document.getElementById('detailContent');
    var detailCategory = document.getElementById('detailCategory');
    var detailImportance = document.getElementById('detailImportance');
    var detailEntityKey = document.getElementById('detailEntityKey');
    var detailTags = document.getElementById('detailTags');
    var detailSource = document.getElementById('detailSource');
    var detailCreated = document.getElementById('detailCreated');
    var detailSaveBtn = document.getElementById('detailSaveBtn');
    var detailDeleteBtn = document.getElementById('detailDeleteBtn');
    var detailCancelBtn = document.getElementById('detailCancelBtn');

    var toastEl = document.getElementById('toast');

    var currentId = null; // null = creating, string = editing
    var debounceTimer = null;

    // ── API helper ──────────────────────────────────────────

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
                if (!res.ok && data.error) throw new Error(data.error);
                return data;
            });
        });
    }

    // ── Toast ───────────────────────────────────────────────

    function toast(msg) {
        toastEl.textContent = msg;
        toastEl.classList.add('visible');
        setTimeout(function () { toastEl.classList.remove('visible'); }, 2500);
    }

    // ── Load list ───────────────────────────────────────────

    function loadInsights() {
        var q = searchInput.value.trim();
        var cat = categoryFilter.value;
        var params = [];
        if (q) params.push('q=' + encodeURIComponent(q));
        if (cat) params.push('category=' + encodeURIComponent(cat));
        params.push('limit=100');
        var url = '/api/insights' + (params.length ? '?' + params.join('&') : '');

        apiFetch(url).then(function (data) {
            renderList(data.items || []);
            countLabel.textContent = (data.items || []).length + ' of ' + (data.total || 0);
        }).catch(function (err) {
            listEl.innerHTML = '<div class="ins-empty">Error loading insights: ' + err.message + '</div>';
        });
    }

    function renderList(items) {
        if (!items.length) {
            listEl.innerHTML = '<div class="ins-empty">No insights found. Insights are automatically extracted from conversations, or you can add them manually.</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var cat = it.category || 'fact';
            var imp = it.importance || 5;
            var tags = it.tags ? ' <span class="ins-card-tags">' + escHtml(it.tags) + '</span>' : '';
            var source = it.source_tool ? it.source_tool : '';
            var created = it.created_at ? formatDate(it.created_at) : '';
            var meta = [];
            if (source) meta.push(source);
            if (created) meta.push(created);
            if (it.entity_key) meta.push(escHtml(it.entity_key));

            html += '<div class="ins-card" data-id="' + escHtml(it.id) + '">'
                + '<div class="ins-card-top">'
                + '<span class="ins-badge ins-badge-' + cat + '">' + cat + '</span>'
                + '<span class="ins-importance">imp: ' + imp + '/10</span>'
                + '</div>'
                + '<div class="ins-card-content">' + escHtml(it.content) + '</div>'
                + '<div class="ins-card-meta">' + meta.join(' &middot; ') + tags + '</div>'
                + '</div>';
        }
        listEl.innerHTML = html;

        // Bind click handlers
        var cards = listEl.querySelectorAll('.ins-card');
        for (var j = 0; j < cards.length; j++) {
            cards[j].addEventListener('click', onCardClick);
        }
    }

    function onCardClick(e) {
        var card = e.currentTarget;
        var id = card.getAttribute('data-id');
        openDetail(id);
    }

    // ── Detail modal ────────────────────────────────────────

    function openDetail(id) {
        if (id) {
            // Edit mode
            currentId = id;
            detailTitle.textContent = 'Edit Insight';
            detailDeleteBtn.style.display = '';
            apiFetch('/api/insights/' + id).then(function (item) {
                detailContent.value = item.content || '';
                detailCategory.value = item.category || 'fact';
                detailImportance.value = item.importance || 5;
                detailEntityKey.value = item.entity_key || '';
                detailTags.value = item.tags || '';
                detailSource.textContent = (item.source_tool || 'auto') + (item.source_session ? ' / session ' + item.source_session.substring(0, 8) : '');
                detailCreated.textContent = item.created_at || '';
                overlay.classList.add('visible');
            });
        } else {
            // Create mode
            currentId = null;
            detailTitle.textContent = 'Add Insight';
            detailDeleteBtn.style.display = 'none';
            detailContent.value = '';
            detailCategory.value = 'fact';
            detailImportance.value = 5;
            detailEntityKey.value = '';
            detailTags.value = '';
            detailSource.textContent = 'manual';
            detailCreated.textContent = 'now';
            overlay.classList.add('visible');
        }
    }

    function closeDetail() {
        overlay.classList.remove('visible');
        currentId = null;
    }

    function saveDetail() {
        var body = {
            content: detailContent.value.trim(),
            category: detailCategory.value,
            importance: parseInt(detailImportance.value, 10) || 5,
            entity_key: detailEntityKey.value.trim() || null,
            tags: detailTags.value.trim() || null
        };
        if (!body.content) {
            toast('Content is required');
            return;
        }

        if (currentId) {
            // Update
            apiFetch('/api/insights/' + currentId, { method: 'PATCH', body: body }).then(function () {
                toast('Insight updated');
                closeDetail();
                loadInsights();
            }).catch(function (err) { toast('Error: ' + err.message); });
        } else {
            // Create
            apiFetch('/api/insights', { method: 'POST', body: body }).then(function (data) {
                toast(data.message || 'Insight created');
                closeDetail();
                loadInsights();
            }).catch(function (err) { toast('Error: ' + err.message); });
        }
    }

    function deleteDetail() {
        if (!currentId) return;
        if (!confirm('Delete this insight?')) return;
        apiFetch('/api/insights/' + currentId, { method: 'DELETE' }).then(function () {
            toast('Insight deleted');
            closeDetail();
            loadInsights();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    // ── Helpers ──────────────────────────────────────────────

    function escHtml(s) {
        if (!s) return '';
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatDate(iso) {
        if (!iso) return '';
        try {
            var d = new Date(iso);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) {
            return iso;
        }
    }

    // ── Events ──────────────────────────────────────────────

    searchInput.addEventListener('input', function () {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(loadInsights, 300);
    });

    searchInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { clearTimeout(debounceTimer); loadInsights(); }
    });

    categoryFilter.addEventListener('change', loadInsights);
    addBtn.addEventListener('click', function () { openDetail(null); });
    detailSaveBtn.addEventListener('click', saveDetail);
    detailDeleteBtn.addEventListener('click', deleteDetail);
    detailCancelBtn.addEventListener('click', closeDetail);

    overlay.addEventListener('click', function (e) {
        if (e.target === overlay) closeDetail();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeDetail();
    });

    // ── Init ────────────────────────────────────────────────

    loadInsights();
})();
