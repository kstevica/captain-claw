(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────
    var state = {
        layer: 'l2',
        tab: 'documents',
        documents: [],
        sources: [],
        searchResults: [],
        selectedDocId: null,
        sourceFilter: '',
    };

    // ── DOM refs ───────────────────────────────────────────
    var $statDocs     = document.getElementById('statDocs');
    var $statChunks   = document.getElementById('statChunks');
    var $statL1       = document.getElementById('statL1');
    var $statL2       = document.getElementById('statL2');
    var $sourceFilter = document.getElementById('sourceFilter');
    var $searchInput  = document.getElementById('searchInput');
    var $searchBtn    = document.getElementById('searchBtn');
    var $docList      = document.getElementById('documentList');
    var $searchList   = document.getElementById('searchResults');
    var $placeholder  = document.getElementById('detailPlaceholder');
    var $detail       = document.getElementById('detailContent');
    var $detailTitle  = document.getElementById('detailTitle');
    var $detailMeta   = document.getElementById('detailMeta');
    var $chunkList    = document.getElementById('chunkList');
    var $layerHint    = document.getElementById('layerHint');
    var $tabs         = document.querySelectorAll('.sm-tab');
    var $layerBtns    = document.querySelectorAll('.sm-layer-btn');

    // ── API helper ─────────────────────────────────────────
    function api(url) {
        return fetch(url).then(function (r) { return r.json(); });
    }

    // ── Init ───────────────────────────────────────────────
    function init() {
        // Layer buttons
        $layerBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                state.layer = btn.dataset.layer;
                $layerBtns.forEach(function (b) { b.classList.toggle('active', b.dataset.layer === state.layer); });
                updateLayerHint();
                // Re-render current detail if open
                if (state.selectedDocId && state.tab === 'documents') {
                    loadDocumentDetail(state.selectedDocId);
                }
                if (state.tab === 'search' && state.searchResults.length) {
                    renderSearchResults();
                }
            });
        });

        // Tabs
        $tabs.forEach(function (tab) {
            tab.addEventListener('click', function () {
                state.tab = tab.dataset.tab;
                $tabs.forEach(function (t) { t.classList.toggle('active', t.dataset.tab === state.tab); });
                $docList.style.display = state.tab === 'documents' ? '' : 'none';
                $searchList.style.display = state.tab === 'search' ? '' : 'none';
            });
        });

        // Source filter
        $sourceFilter.addEventListener('change', function () {
            state.sourceFilter = $sourceFilter.value;
            loadDocuments();
        });

        // Search
        $searchBtn.addEventListener('click', doSearch);
        $searchInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') doSearch();
        });

        updateLayerHint();
        loadStatus();
        loadDocuments();
    }

    function updateLayerHint() {
        var hints = { l1: 'One-liner headlines', l2: 'Summary view', l3: 'Full text' };
        $layerHint.textContent = hints[state.layer] || '';
    }

    // ── Load status ────────────────────────────────────────
    function loadStatus() {
        api('/api/semantic-memory/status').then(function (d) {
            if (!d.enabled) {
                $statDocs.textContent = '—';
                $statChunks.textContent = '—';
                $statL1.textContent = '—';
                $statL2.textContent = '—';
                return;
            }
            $statDocs.textContent = d.documents || 0;
            $statChunks.textContent = d.chunks || 0;
            var pctL1 = d.chunks ? Math.round((d.chunks_with_l1 / d.chunks) * 100) : 0;
            var pctL2 = d.chunks ? Math.round((d.chunks_with_l2 / d.chunks) * 100) : 0;
            $statL1.textContent = pctL1 + '%';
            $statL2.textContent = pctL2 + '%';
        });
    }

    // ── Load documents ─────────────────────────────────────
    function loadDocuments() {
        var url = '/api/semantic-memory/documents';
        if (state.sourceFilter) url += '?source=' + encodeURIComponent(state.sourceFilter);
        api(url).then(function (d) {
            state.documents = d.documents || [];
            state.sources = d.sources || [];
            renderSourceFilter();
            renderDocuments();
        });
    }

    function renderSourceFilter() {
        var current = $sourceFilter.value;
        $sourceFilter.innerHTML = '<option value="">All sources</option>';
        state.sources.forEach(function (s) {
            var opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s;
            if (s === current) opt.selected = true;
            $sourceFilter.appendChild(opt);
        });
    }

    function renderDocuments() {
        if (!state.documents.length) {
            $docList.innerHTML = '<div class="sm-empty">No documents indexed yet.</div>';
            return;
        }
        var html = '';
        state.documents.forEach(function (doc) {
            var active = doc.doc_id === state.selectedDocId ? ' active' : '';
            html += '<div class="sm-item' + active + '" data-docid="' + esc(doc.doc_id) + '">'
                + '<div class="sm-item-title">' + esc(doc.path || doc.reference) + '</div>'
                + '<div class="sm-item-meta">' + esc(doc.source) + ' &middot; ' + doc.chunk_count + ' chunks &middot; ' + esc(doc.updated_at || '') + '</div>'
                + '</div>';
        });
        $docList.innerHTML = html;
        $docList.querySelectorAll('.sm-item').forEach(function (el) {
            el.addEventListener('click', function () {
                loadDocumentDetail(el.dataset.docid);
            });
        });
    }

    // ── Load document detail ───────────────────────────────
    function loadDocumentDetail(docId) {
        state.selectedDocId = docId;
        renderDocuments(); // highlight active
        api('/api/semantic-memory/documents/' + encodeURIComponent(docId)).then(function (d) {
            if (d.error) {
                showPlaceholder(d.error);
                return;
            }
            var doc = d.document;
            $detailTitle.textContent = doc.path || doc.reference;
            $detailMeta.textContent = doc.source + ' \u00B7 ' + doc.doc_id + ' \u00B7 ' + (doc.updated_at || '');
            renderChunks(d.chunks || []);
            $placeholder.style.display = 'none';
            $detail.style.display = '';
        });
    }

    function renderChunks(chunks) {
        if (!chunks.length) {
            $chunkList.innerHTML = '<div class="sm-empty">No chunks.</div>';
            return;
        }
        var html = '';
        chunks.forEach(function (c) {
            html += '<div class="sm-chunk">'
                + '<div class="sm-chunk-header">'
                + '<span class="sm-chunk-idx">#' + c.chunk_index + '</span>'
                + '<span class="sm-chunk-lines">lines ' + c.start_line + '–' + c.end_line + '</span>'
                + '<span>' + esc(c.chunk_id) + '</span>'
                + '</div>'
                + '<div class="sm-chunk-layers">'
                + renderLayerBlock('l1', 'One-liner', c.text_l1)
                + renderLayerBlock('l2', 'Summary', c.text_l2)
                + renderLayerBlock('l3', 'Full text', c.text)
                + '</div>'
                + '</div>';
        });
        $chunkList.innerHTML = html;
    }

    function renderLayerBlock(layer, label, text) {
        var content = text
            ? '<div class="sm-chunk-layer-text">' + esc(text) + '</div>'
            : '<div class="sm-chunk-layer-empty">Not generated</div>';
        return '<div class="sm-chunk-layer">'
            + '<div class="sm-chunk-layer-label ' + layer + '">' + label + '</div>'
            + content
            + '</div>';
    }

    // ── Search ─────────────────────────────────────────────
    function doSearch() {
        var q = $searchInput.value.trim();
        if (!q) return;
        // Switch to search tab
        state.tab = 'search';
        $tabs.forEach(function (t) { t.classList.toggle('active', t.dataset.tab === 'search'); });
        $docList.style.display = 'none';
        $searchList.style.display = '';

        var url = '/api/semantic-memory/search?q=' + encodeURIComponent(q)
            + '&layer=' + state.layer + '&max=20';
        api(url).then(function (d) {
            state.searchResults = d.results || [];
            renderSearchResults();
        });
    }

    function renderSearchResults() {
        if (!state.searchResults.length) {
            $searchList.innerHTML = '<div class="sm-empty">No results.</div>';
            return;
        }
        var html = '';
        state.searchResults.forEach(function (r, idx) {
            html += '<div class="sm-item" data-idx="' + idx + '">'
                + '<div class="sm-item-title">' + esc(r.path) + ':' + r.start_line
                + '<span class="sm-item-score">' + r.score + '</span>'
                + '</div>'
                + '<div class="sm-item-meta">' + esc(r.source) + ' &middot; ' + esc(r.reference) + '</div>'
                + '<div class="sm-item-snippet">' + esc(r.snippet) + '</div>'
                + '</div>';
        });
        $searchList.innerHTML = html;
        $searchList.querySelectorAll('.sm-item').forEach(function (el) {
            el.addEventListener('click', function () {
                var r = state.searchResults[parseInt(el.dataset.idx)];
                showSearchResultDetail(r);
            });
        });
    }

    function showSearchResultDetail(r) {
        $detailTitle.textContent = r.path + ':' + r.start_line;
        $detailMeta.textContent = r.source + ' \u00B7 score=' + r.score
            + ' (text=' + r.text_score + ', vector=' + r.vector_score + ')';
        // Show all three layers for this result
        var chunks = [{
            chunk_id: r.chunk_id,
            chunk_index: 0,
            start_line: r.start_line,
            end_line: r.end_line,
            text: r.text_l3 || '',
            text_l1: r.text_l1 || '',
            text_l2: r.text_l2 || '',
        }];
        renderChunks(chunks);
        $placeholder.style.display = 'none';
        $detail.style.display = '';
    }

    // ── Helpers ─────────────────────────────────────────────
    function showPlaceholder(msg) {
        $placeholder.textContent = msg || 'Select an item to view details.';
        $placeholder.style.display = '';
        $detail.style.display = 'none';
    }

    function esc(s) {
        if (!s) return '';
        var d = document.createElement('div');
        d.textContent = String(s);
        return d.innerHTML;
    }

    init();
})();
