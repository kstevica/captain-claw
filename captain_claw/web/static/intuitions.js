/* Captain Claw — Nervous System / Intuitions Browser */

(function () {
    'use strict';

    // ── DOM refs ────────────────────────────────────────────
    var searchInput = document.getElementById('searchInput');
    var typeFilter = document.getElementById('typeFilter');
    var addBtn = document.getElementById('addBtn');
    var dreamBtn = document.getElementById('dreamBtn');
    var countLabel = document.getElementById('countLabel');
    var listEl = document.getElementById('intuitionsList');

    var overlay = document.getElementById('detailOverlay');
    var detailTitle = document.getElementById('detailTitle');
    var detailContent = document.getElementById('detailContent');
    var detailType = document.getElementById('detailType');
    var detailConfidence = document.getElementById('detailConfidence');
    var detailConfidenceLabel = document.getElementById('detailConfidenceLabel');
    var detailImportance = document.getElementById('detailImportance');
    var detailTags = document.getElementById('detailTags');
    var detailLayers = document.getElementById('detailLayers');
    var detailValidated = document.getElementById('detailValidated');
    var detailAccess = document.getElementById('detailAccess');
    var detailCreated = document.getElementById('detailCreated');
    var detailSaveBtn = document.getElementById('detailSaveBtn');
    var detailDeleteBtn = document.getElementById('detailDeleteBtn');
    var detailValidateBtn = document.getElementById('detailValidateBtn');
    var detailCancelBtn = document.getElementById('detailCancelBtn');

    var statTotal = document.getElementById('statTotal');
    var statValidated = document.getElementById('statValidated');
    var statConfidence = document.getElementById('statConfidence');
    var statImportance = document.getElementById('statImportance');

    var toastEl = document.getElementById('toast');

    var currentId = null;
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

    // ── Stats ───────────────────────────────────────────────

    function loadStats() {
        apiFetch('/api/nervous-system/stats').then(function (data) {
            statTotal.textContent = data.total || 0;
            statValidated.textContent = data.validated || 0;
            statConfidence.textContent = data.avg_confidence || '—';
            statImportance.textContent = data.avg_importance || '—';
        }).catch(function () {
            statTotal.textContent = '—';
        });
    }

    // ── Load list ───────────────────────────────────────────

    function loadIntuitions() {
        var q = searchInput.value.trim();
        var tt = typeFilter.value;
        var params = [];
        if (q) params.push('q=' + encodeURIComponent(q));
        if (tt) params.push('thread_type=' + encodeURIComponent(tt));
        params.push('limit=100');
        var url = '/api/nervous-system' + (params.length ? '?' + params.join('&') : '');

        apiFetch(url).then(function (data) {
            renderList(data.items || []);
            countLabel.textContent = (data.items || []).length + ' of ' + (data.total || 0);
        }).catch(function (err) {
            listEl.innerHTML = '<div class="nts-empty">Error loading intuitions: ' + err.message + '</div>';
        });
    }

    function renderList(items) {
        if (!items.length) {
            listEl.innerHTML = '<div class="nts-empty">No intuitions yet. Intuitions are autonomously generated during dream cycles, or use the Dream button to trigger one manually.</div>';
            return;
        }
        var html = '';
        for (var i = 0; i < items.length; i++) {
            var it = items[i];
            var tt = it.thread_type || 'association';
            var conf = parseFloat(it.confidence) || 0;
            var imp = it.importance || 5;
            var validated = it.validated ? ' <span class="nts-validated">\u2713 validated</span>' : '';
            var tags = it.tags ? ' <span class="nts-card-tags">' + escHtml(it.tags) + '</span>' : '';
            var layers = '';
            if (it.source_layers && it.source_layers.length) {
                layers = it.source_layers.join(', ');
            }
            var created = it.created_at ? formatDate(it.created_at) : '';
            var meta = [];
            if (layers) meta.push(layers);
            if (created) meta.push(created);
            if (it.access_count) meta.push('accessed ' + it.access_count + 'x');

            var confClass = conf >= 0.7 ? 'nts-conf-high' : (conf >= 0.4 ? 'nts-conf-med' : 'nts-conf-low');
            var confBar = '<span class="nts-conf-bar"><span class="nts-conf-fill ' + confClass + '" style="width:' + (conf * 100) + '%"></span></span>';

            html += '<div class="nts-card" data-id="' + escHtml(it.id) + '">'
                + '<div class="nts-card-top">'
                + '<span class="nts-badge nts-badge-' + tt + '">' + tt + '</span>'
                + '<span class="nts-confidence">conf: ' + conf.toFixed(2) + ' ' + confBar + '</span>'
                + '<span class="nts-importance">imp: ' + imp + '/10</span>'
                + validated
                + '</div>'
                + '<div class="nts-card-content">' + escHtml(it.content) + '</div>'
                + '<div class="nts-card-meta">' + meta.join(' &middot; ') + tags + '</div>'
                + '</div>';
        }
        listEl.innerHTML = html;

        var cards = listEl.querySelectorAll('.nts-card');
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
            currentId = id;
            detailTitle.textContent = 'Edit Intuition';
            detailDeleteBtn.style.display = '';
            detailValidateBtn.style.display = '';
            apiFetch('/api/nervous-system/' + id).then(function (item) {
                detailContent.value = item.content || '';
                detailType.value = item.thread_type || 'association';
                detailConfidence.value = item.confidence || 0.5;
                detailConfidenceLabel.textContent = parseFloat(item.confidence || 0.5).toFixed(2);
                detailImportance.value = item.importance || 5;
                detailTags.value = item.tags || '';
                detailLayers.textContent = (item.source_layers || []).join(', ') || 'unknown';
                detailValidated.textContent = item.validated ? '\u2713 Validated' : 'Not validated';
                detailValidated.style.color = item.validated ? 'var(--green)' : 'var(--text-muted)';
                detailValidateBtn.textContent = item.validated ? 'Validated' : 'Validate';
                detailValidateBtn.disabled = !!item.validated;
                detailAccess.textContent = (item.access_count || 0) + ' times' + (item.last_accessed ? ' (last: ' + formatDate(item.last_accessed) + ')' : '');
                detailCreated.textContent = item.created_at || '';
                overlay.classList.add('visible');
            });
        } else {
            currentId = null;
            detailTitle.textContent = 'Add Intuition';
            detailDeleteBtn.style.display = 'none';
            detailValidateBtn.style.display = 'none';
            detailContent.value = '';
            detailType.value = 'association';
            detailConfidence.value = 0.5;
            detailConfidenceLabel.textContent = '0.50';
            detailImportance.value = 5;
            detailTags.value = '';
            detailLayers.textContent = 'manual';
            detailValidated.textContent = 'New';
            detailValidated.style.color = 'var(--text-muted)';
            detailAccess.textContent = '0 times';
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
            thread_type: detailType.value,
            confidence: parseFloat(detailConfidence.value) || 0.5,
            importance: parseInt(detailImportance.value, 10) || 5,
            tags: detailTags.value.trim() || null
        };
        if (!body.content) {
            toast('Content is required');
            return;
        }

        if (currentId) {
            apiFetch('/api/nervous-system/' + currentId, { method: 'PATCH', body: body }).then(function () {
                toast('Intuition updated');
                closeDetail();
                loadIntuitions();
                loadStats();
            }).catch(function (err) { toast('Error: ' + err.message); });
        } else {
            apiFetch('/api/nervous-system', { method: 'POST', body: body }).then(function (data) {
                toast(data.message || 'Intuition created');
                closeDetail();
                loadIntuitions();
                loadStats();
            }).catch(function (err) { toast('Error: ' + err.message); });
        }
    }

    function deleteDetail() {
        if (!currentId) return;
        if (!confirm('Delete this intuition?')) return;
        apiFetch('/api/nervous-system/' + currentId, { method: 'DELETE' }).then(function () {
            toast('Intuition deleted');
            closeDetail();
            loadIntuitions();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    function validateDetail() {
        if (!currentId) return;
        apiFetch('/api/nervous-system/' + currentId, { method: 'PATCH', body: { validated: 1 } }).then(function () {
            toast('Intuition validated — protected from decay');
            closeDetail();
            loadIntuitions();
            loadStats();
        }).catch(function (err) { toast('Error: ' + err.message); });
    }

    // ── Dream trigger ───────────────────────────────────────

    function triggerDream() {
        dreamBtn.disabled = true;
        dreamBtn.textContent = '\u231B Dreaming...';
        apiFetch('/api/nervous-system/dream', { method: 'POST' }).then(function (data) {
            var count = data.stored || 0;
            toast(count ? count + ' new intuition(s) formed' : 'Dream completed — no new intuitions');
            loadIntuitions();
            loadStats();
        }).catch(function (err) {
            toast('Dream failed: ' + err.message);
        }).finally(function () {
            dreamBtn.disabled = false;
            dreamBtn.textContent = '\uD83D\uDCAD Dream';
        });
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
        debounceTimer = setTimeout(loadIntuitions, 300);
    });

    searchInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { clearTimeout(debounceTimer); loadIntuitions(); }
    });

    typeFilter.addEventListener('change', loadIntuitions);
    addBtn.addEventListener('click', function () { openDetail(null); });
    dreamBtn.addEventListener('click', triggerDream);
    detailSaveBtn.addEventListener('click', saveDetail);
    detailDeleteBtn.addEventListener('click', deleteDetail);
    detailValidateBtn.addEventListener('click', validateDetail);
    detailCancelBtn.addEventListener('click', closeDetail);

    detailConfidence.addEventListener('input', function () {
        detailConfidenceLabel.textContent = parseFloat(this.value).toFixed(2);
    });

    overlay.addEventListener('click', function (e) {
        if (e.target === overlay) closeDetail();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeDetail();
    });

    // ── Init ────────────────────────────────────────────────

    loadIntuitions();
    loadStats();
})();
