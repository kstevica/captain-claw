/* Captain Claw — Deep Memory Dashboard */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────────────────────

    var state = {
        connected: false,
        documents: [],
        selectedDocId: null,
        selectedDoc: null,
        searchQuery: '',
        sourceFilter: '',
        tagFilter: '',
        page: 1,
        perPage: 50,
        totalFound: 0,
        facets: { sources: [], tags: [] },
        stats: { collection: '', num_documents: 0 },
        showIndexForm: false,
        isIndexing: false,
        deleteTarget: null,
    };

    var searchTimeout = null;
    var toastTimer = null;

    // ── DOM References ───────────────────────────────────────────────────────

    var $statusDot       = document.getElementById('dmStatusDot');
    var $statusText      = document.getElementById('dmStatusText');
    var $collectionName  = document.getElementById('dmCollectionName');
    var $docCount        = document.getElementById('dmDocCount');
    var $sourceFilter    = document.getElementById('dmSourceFilter');
    var $sourceAllCount  = document.getElementById('dmSourceAllCount');
    var $tagFilter       = document.getElementById('dmTagFilter');
    var $search          = document.getElementById('dmSearch');
    var $addBtn          = document.getElementById('dmAddBtn');
    var $itemList        = document.getElementById('dmItemList');
    var $loading         = document.getElementById('dmLoading');
    var $listFooter      = document.getElementById('dmListFooter');
    var $resultInfo      = document.getElementById('dmResultInfo');
    var $empty           = document.getElementById('dmEmpty');
    var $notConnected    = document.getElementById('dmNotConnected');
    var $notConnectedMsg = document.getElementById('dmNotConnectedMsg');
    var $detailView      = document.getElementById('dmDetailView');
    var $detailTitle     = document.getElementById('dmDetailTitle');
    var $detailMeta      = document.getElementById('dmDetailMeta');
    var $detailInfo      = document.getElementById('dmDetailInfo');
    var $deleteBtn       = document.getElementById('dmDeleteBtn');
    var $chunksCount     = document.getElementById('dmChunksCount');
    var $chunksList      = document.getElementById('dmChunksList');
    var $indexForm       = document.getElementById('dmIndexForm');
    var $indexCancel     = document.getElementById('dmIndexCancel');
    var $indexSubmit     = document.getElementById('dmIndexSubmit');
    var $indexText       = document.getElementById('dmIndexText');
    var $indexSource     = document.getElementById('dmIndexSource');
    var $indexRef        = document.getElementById('dmIndexRef');
    var $indexTags       = document.getElementById('dmIndexTags');
    var $modalOverlay    = document.getElementById('dmModalOverlay');
    var $modalBody       = document.getElementById('dmModalBody');
    var $modalCancel     = document.getElementById('dmModalCancel');
    var $modalConfirm    = document.getElementById('dmModalConfirm');
    var $toast           = document.getElementById('dmToast');

    // ── Init ─────────────────────────────────────────────────────────────────

    init();

    function init() {
        // Event listeners
        $search.addEventListener('input', onSearchInput);
        $addBtn.addEventListener('click', onAddClick);
        $deleteBtn.addEventListener('click', onDeleteClick);
        $indexCancel.addEventListener('click', onIndexCancel);
        $indexSubmit.addEventListener('click', onIndexSubmit);
        $modalCancel.addEventListener('click', hideModal);
        $modalConfirm.addEventListener('click', onConfirmDelete);

        // Keyboard shortcuts
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                if ($modalOverlay.style.display !== 'none') {
                    hideModal();
                } else if (state.showIndexForm) {
                    onIndexCancel();
                }
            }
        });

        // Scroll pagination in document list
        $itemList.addEventListener('scroll', function () {
            var el = $itemList;
            if (el.scrollTop + el.clientHeight >= el.scrollHeight - 40) {
                loadNextPage();
            }
        });

        // Load
        fetchStatus();
    }

    // ── API Utility ──────────────────────────────────────────────────────────

    function apiFetch(url, options) {
        options = options || {};
        var fetchOpts = {
            method: options.method || 'GET',
            headers: {},
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

    // ── Data Fetchers ────────────────────────────────────────────────────────

    function fetchStatus() {
        apiFetch('/api/deep-memory/status')
            .then(function (data) {
                state.connected = data.connected;
                state.stats.collection = data.collection || '';
                state.stats.num_documents = data.num_documents || 0;

                renderStatus();

                if (data.connected) {
                    fetchFacets();
                    fetchDocuments();
                } else {
                    showNotConnected(data.error || 'Not connected');
                }
            })
            .catch(function (err) {
                state.connected = false;
                renderStatus();
                showNotConnected(err.message || 'Failed to connect');
            });
    }

    function fetchFacets() {
        apiFetch('/api/deep-memory/facets')
            .then(function (data) {
                state.facets.sources = data.sources || [];
                state.facets.tags = data.tags || [];
                renderFacets();
            })
            .catch(function () {});
    }

    function fetchDocuments() {
        $loading.style.display = '';
        clearDocumentItems();

        var url = '/api/deep-memory/documents?page=' + state.page +
                  '&per_page=' + state.perPage;

        if (state.searchQuery) {
            url += '&q=' + encodeURIComponent(state.searchQuery);
        }
        if (state.sourceFilter) {
            url += '&source=' + encodeURIComponent(state.sourceFilter);
        }
        if (state.tagFilter) {
            url += '&tag=' + encodeURIComponent(state.tagFilter);
        }

        apiFetch(url)
            .then(function (data) {
                $loading.style.display = 'none';

                if (state.page === 1) {
                    state.documents = data.documents || [];
                } else {
                    state.documents = state.documents.concat(data.documents || []);
                }
                state.totalFound = data.found || 0;

                renderDocumentList();
                renderResultInfo();
            })
            .catch(function () {
                $loading.style.display = 'none';
                state.documents = [];
                state.totalFound = 0;
                renderDocumentList();
                renderResultInfo();
            });
    }

    function fetchDocumentDetail(docId) {
        state.selectedDocId = docId;
        state.selectedDoc = null;

        showDetailView();
        $chunksList.innerHTML = '<div class="dm-loading">Loading chunks...</div>';

        apiFetch('/api/deep-memory/documents/' + encodeURIComponent(docId))
            .then(function (data) {
                state.selectedDoc = data;
                renderDetail();
            })
            .catch(function (err) {
                $chunksList.innerHTML = '';
                showToast('Failed to load document: ' + (err.message || err), 'error');
            });
    }

    // ── Search ───────────────────────────────────────────────────────────────

    function onSearchInput() {
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            state.searchQuery = $search.value.trim();
            state.page = 1;
            state.selectedDocId = null;
            state.selectedDoc = null;
            showEmptyState();
            fetchDocuments();
        }, 300);
    }

    // ── Source & Tag Filters ─────────────────────────────────────────────────

    function onSourceClick(source) {
        if (state.sourceFilter === source) return;
        state.sourceFilter = source;
        state.page = 1;
        state.selectedDocId = null;
        state.selectedDoc = null;
        showEmptyState();

        // Update selection UI
        var items = $sourceFilter.querySelectorAll('.dm-filter-item');
        for (var i = 0; i < items.length; i++) {
            items[i].classList.toggle('selected',
                (items[i].getAttribute('data-source') || '') === source);
        }

        fetchDocuments();
    }

    function onTagClick(tag) {
        // Toggle: click same tag = deselect
        if (state.tagFilter === tag) {
            state.tagFilter = '';
        } else {
            state.tagFilter = tag;
        }
        state.page = 1;
        state.selectedDocId = null;
        state.selectedDoc = null;
        showEmptyState();

        // Update selection UI
        var chips = $tagFilter.querySelectorAll('.dm-tag-chip');
        for (var i = 0; i < chips.length; i++) {
            chips[i].classList.toggle('selected',
                chips[i].getAttribute('data-tag') === state.tagFilter);
        }

        fetchDocuments();
    }

    // ── Pagination ───────────────────────────────────────────────────────────

    function loadNextPage() {
        var loaded = state.documents.length;
        if (loaded >= state.totalFound) return;
        if ($loading.style.display !== 'none') return;

        state.page += 1;
        $loading.style.display = '';

        var url = '/api/deep-memory/documents?page=' + state.page +
                  '&per_page=' + state.perPage;
        if (state.searchQuery) url += '&q=' + encodeURIComponent(state.searchQuery);
        if (state.sourceFilter) url += '&source=' + encodeURIComponent(state.sourceFilter);
        if (state.tagFilter) url += '&tag=' + encodeURIComponent(state.tagFilter);

        apiFetch(url)
            .then(function (data) {
                $loading.style.display = 'none';
                var newDocs = data.documents || [];
                state.documents = state.documents.concat(newDocs);
                state.totalFound = data.found || state.totalFound;
                // Append new items only
                for (var i = 0; i < newDocs.length; i++) {
                    appendDocumentItem(newDocs[i]);
                }
                renderResultInfo();
            })
            .catch(function () {
                $loading.style.display = 'none';
            });
    }

    // ── Add (Index Form) ─────────────────────────────────────────────────────

    function onAddClick() {
        state.showIndexForm = true;
        state.selectedDocId = null;
        state.selectedDoc = null;

        // Deselect list items
        var items = $itemList.querySelectorAll('.dm-list-item');
        for (var i = 0; i < items.length; i++) items[i].classList.remove('selected');

        // Reset form
        $indexText.value = '';
        $indexSource.value = 'manual';
        $indexRef.value = '';
        $indexTags.value = '';

        showIndexForm();
        setTimeout(function () { $indexText.focus(); }, 50);
    }

    function onIndexCancel() {
        state.showIndexForm = false;
        if (state.selectedDoc) {
            showDetailView();
            renderDetail();
        } else {
            showEmptyState();
        }
    }

    function onIndexSubmit() {
        var text = $indexText.value.trim();
        if (!text) {
            showToast('Text is required', 'error');
            $indexText.focus();
            return;
        }

        state.isIndexing = true;
        $indexSubmit.disabled = true;
        $indexSubmit.textContent = 'Indexing...';

        var body = {
            text: text,
            source: $indexSource.value.trim() || 'manual',
            reference: $indexRef.value.trim(),
            tags: $indexTags.value.trim(),
        };

        apiFetch('/api/deep-memory/index', { method: 'POST', body: body })
            .then(function (data) {
                state.isIndexing = false;
                $indexSubmit.disabled = false;
                $indexSubmit.textContent = 'Index';

                if (data.error) {
                    showToast(data.error, 'error');
                    return;
                }

                showToast('Indexed ' + (data.chunks_indexed || 0) + ' chunks', 'success');
                state.showIndexForm = false;
                state.page = 1;

                // Refresh everything
                fetchStatus();
            })
            .catch(function (err) {
                state.isIndexing = false;
                $indexSubmit.disabled = false;
                $indexSubmit.textContent = 'Index';
                showToast('Index failed: ' + (err.message || err), 'error');
            });
    }

    // ── Delete ───────────────────────────────────────────────────────────────

    function onDeleteClick() {
        if (!state.selectedDocId) return;
        state.deleteTarget = state.selectedDocId;
        var doc = state.selectedDoc;
        var name = (doc && (doc.reference || doc.path || doc.doc_id)) || state.selectedDocId;
        $modalBody.textContent = 'Delete "' + name + '" and all its chunks? This cannot be undone.';
        $modalOverlay.style.display = '';
    }

    function hideModal() {
        $modalOverlay.style.display = 'none';
        state.deleteTarget = null;
    }

    function onConfirmDelete() {
        if (!state.deleteTarget) return;

        $modalConfirm.disabled = true;
        $modalConfirm.textContent = 'Deleting...';

        apiFetch('/api/deep-memory/documents/' + encodeURIComponent(state.deleteTarget), {
            method: 'DELETE',
        })
            .then(function (data) {
                $modalConfirm.disabled = false;
                $modalConfirm.textContent = 'Delete';
                hideModal();

                var numDeleted = data.num_deleted || 0;
                showToast('Deleted ' + numDeleted + ' chunks', 'success');

                state.selectedDocId = null;
                state.selectedDoc = null;
                state.page = 1;
                showEmptyState();

                // Refresh
                fetchStatus();
            })
            .catch(function (err) {
                $modalConfirm.disabled = false;
                $modalConfirm.textContent = 'Delete';
                showToast('Delete failed: ' + (err.message || err), 'error');
            });
    }

    // ── Document Item Click ──────────────────────────────────────────────────

    function onDocumentClick(docId) {
        state.showIndexForm = false;

        // Highlight in list
        var items = $itemList.querySelectorAll('.dm-list-item');
        for (var i = 0; i < items.length; i++) {
            items[i].classList.toggle('selected',
                items[i].getAttribute('data-doc-id') === docId);
        }

        fetchDocumentDetail(docId);
    }

    // ── Rendering: Status ────────────────────────────────────────────────────

    function renderStatus() {
        if (state.connected) {
            $statusDot.className = 'dm-status-dot connected';
            $statusText.textContent = 'connected';
        } else {
            $statusDot.className = 'dm-status-dot error';
            $statusText.textContent = 'disconnected';
        }

        $collectionName.textContent = state.stats.collection || '-';
        $docCount.textContent = state.stats.num_documents;
    }

    // ── Rendering: Facets ────────────────────────────────────────────────────

    function renderFacets() {
        // Sources
        var sourceHtml = '';
        var totalDocs = 0;
        for (var i = 0; i < state.facets.sources.length; i++) {
            totalDocs += state.facets.sources[i].count;
        }

        sourceHtml += '<div class="dm-filter-item' +
            (state.sourceFilter === '' ? ' selected' : '') +
            '" data-source="">' +
            '<span class="dm-filter-label">All</span>' +
            '<span class="dm-filter-count">' + totalDocs + '</span></div>';

        for (var j = 0; j < state.facets.sources.length; j++) {
            var src = state.facets.sources[j];
            sourceHtml += '<div class="dm-filter-item' +
                (state.sourceFilter === src.value ? ' selected' : '') +
                '" data-source="' + escapeAttr(src.value) + '">' +
                '<span class="dm-filter-label">' + escapeHtml(src.value) + '</span>' +
                '<span class="dm-filter-count">' + src.count + '</span></div>';
        }

        $sourceFilter.innerHTML = sourceHtml;

        // Bind clicks
        var sourceItems = $sourceFilter.querySelectorAll('.dm-filter-item');
        for (var k = 0; k < sourceItems.length; k++) {
            (function (el) {
                el.addEventListener('click', function () {
                    onSourceClick(el.getAttribute('data-source') || '');
                });
            })(sourceItems[k]);
        }

        // Tags
        var tagHtml = '';
        if (state.facets.tags.length === 0) {
            tagHtml = '<span style="font-size:11px;color:var(--text-muted);padding:0 2px;">No tags</span>';
        }
        for (var m = 0; m < state.facets.tags.length; m++) {
            var tag = state.facets.tags[m];
            tagHtml += '<span class="dm-tag-chip' +
                (state.tagFilter === tag.value ? ' selected' : '') +
                '" data-tag="' + escapeAttr(tag.value) + '">' +
                escapeHtml(tag.value) + ' <span style="opacity:0.6;">' + tag.count + '</span></span>';
        }

        $tagFilter.innerHTML = tagHtml;

        // Bind clicks
        var tagChips = $tagFilter.querySelectorAll('.dm-tag-chip');
        for (var n = 0; n < tagChips.length; n++) {
            (function (el) {
                el.addEventListener('click', function () {
                    onTagClick(el.getAttribute('data-tag') || '');
                });
            })(tagChips[n]);
        }
    }

    // ── Rendering: Document List ─────────────────────────────────────────────

    function clearDocumentItems() {
        var items = $itemList.querySelectorAll('.dm-list-item, .dm-list-empty');
        for (var i = 0; i < items.length; i++) items[i].remove();
    }

    function renderDocumentList() {
        clearDocumentItems();

        if (state.documents.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'dm-list-empty';
            empty.textContent = state.searchQuery ? 'No matches found' : 'No documents indexed';
            if (!state.searchQuery) {
                var sub = document.createElement('div');
                sub.className = 'dm-list-empty-sub';
                sub.textContent = 'Click + to index one';
                empty.appendChild(sub);
            }
            $itemList.appendChild(empty);
            return;
        }

        for (var i = 0; i < state.documents.length; i++) {
            appendDocumentItem(state.documents[i]);
        }
    }

    function appendDocumentItem(doc) {
        var el = document.createElement('div');
        el.className = 'dm-list-item';
        el.setAttribute('data-doc-id', doc.doc_id);

        if (state.selectedDocId === doc.doc_id) {
            el.classList.add('selected');
        }

        // Name: reference or path or doc_id
        var nameEl = document.createElement('div');
        nameEl.className = 'dm-list-item-name';
        nameEl.textContent = doc.reference || doc.path || doc.doc_id || '(untitled)';
        el.appendChild(nameEl);

        // Subtitle: snippet
        if (doc.snippet) {
            var subEl = document.createElement('div');
            subEl.className = 'dm-list-item-sub';
            subEl.textContent = doc.snippet;
            el.appendChild(subEl);
        }

        // Badges row
        var badgeRow = document.createElement('div');
        badgeRow.className = 'dm-list-item-badges';

        // Source badge
        if (doc.source) {
            var srcBadge = document.createElement('span');
            srcBadge.className = 'dm-badge ' + getSourceBadgeClass(doc.source);
            srcBadge.textContent = doc.source;
            badgeRow.appendChild(srcBadge);
        }

        // Chunk count badge
        if (doc.chunk_count) {
            var chunkBadge = document.createElement('span');
            chunkBadge.className = 'dm-badge chunks';
            chunkBadge.textContent = doc.chunk_count + ' chunk' + (doc.chunk_count !== 1 ? 's' : '');
            badgeRow.appendChild(chunkBadge);
        }

        // Relative time
        if (doc.updated_at) {
            var timeEl = document.createElement('span');
            timeEl.className = 'dm-badge chunks';
            timeEl.textContent = formatRelativeTime(doc.updated_at);
            badgeRow.appendChild(timeEl);
        }

        el.appendChild(badgeRow);

        // Tags
        if (doc.tags && doc.tags.length > 0) {
            var tagsRow = document.createElement('div');
            tagsRow.className = 'dm-list-item-badges';
            tagsRow.style.marginTop = '2px';
            for (var t = 0; t < doc.tags.length; t++) {
                var tagEl = document.createElement('span');
                tagEl.className = 'dm-tag-sm';
                tagEl.textContent = doc.tags[t];
                tagsRow.appendChild(tagEl);
            }
            el.appendChild(tagsRow);
        }

        (function (docId) {
            el.addEventListener('click', function () { onDocumentClick(docId); });
        })(doc.doc_id);

        $itemList.appendChild(el);
    }

    function renderResultInfo() {
        if (state.documents.length > 0 || state.totalFound > 0) {
            $listFooter.style.display = '';
            $resultInfo.textContent = state.documents.length + ' of ' + state.totalFound + ' documents';
        } else {
            $listFooter.style.display = 'none';
        }
    }

    // ── Rendering: Detail View ───────────────────────────────────────────────

    function renderDetail() {
        var doc = state.selectedDoc;
        if (!doc) return;

        // Title
        $detailTitle.textContent = doc.reference || doc.path || doc.doc_id || '(untitled)';

        // Meta line
        var metaParts = [];
        if (doc.source) metaParts.push('source: ' + doc.source);
        if (doc.updated_at) metaParts.push('updated: ' + formatRelativeTime(doc.updated_at));
        $detailMeta.textContent = metaParts.join('  |  ');

        // Info bar
        var infoHtml = '';
        if (doc.doc_id) {
            infoHtml += '<span class="dm-info-item"><span class="dm-info-label">ID</span>' +
                escapeHtml(doc.doc_id) + '</span>';
        }
        if (doc.path && doc.path !== doc.reference) {
            infoHtml += '<span class="dm-info-item"><span class="dm-info-label">Path</span>' +
                escapeHtml(doc.path) + '</span>';
        }
        if (doc.tags && doc.tags.length > 0) {
            infoHtml += '<span class="dm-info-item"><span class="dm-info-label">Tags</span>';
            for (var t = 0; t < doc.tags.length; t++) {
                infoHtml += '<span class="dm-tag-sm" style="margin-right:3px;">' +
                    escapeHtml(doc.tags[t]) + '</span>';
            }
            infoHtml += '</span>';
        }
        $detailInfo.innerHTML = infoHtml;

        // Chunks
        var chunks = doc.chunks || [];
        $chunksCount.textContent = chunks.length;

        var chunksHtml = '';
        for (var i = 0; i < chunks.length; i++) {
            var chunk = chunks[i];
            chunksHtml += '<div class="dm-chunk">' +
                '<div class="dm-chunk-header">' +
                '<span class="dm-chunk-idx">#' + chunk.chunk_index + '</span>';
            if (chunk.start_line || chunk.end_line) {
                chunksHtml += '<span class="dm-chunk-lines">lines ' +
                    chunk.start_line + '-' + chunk.end_line + '</span>';
            }
            chunksHtml += '</div>' +
                '<div class="dm-chunk-text">' + escapeHtml(chunk.text || '') + '</div>' +
                '</div>';
        }

        if (chunks.length === 0) {
            chunksHtml = '<div class="dm-loading" style="color:var(--text-muted);">No chunks found</div>';
        }

        $chunksList.innerHTML = chunksHtml;
    }

    // ── Panel Visibility Helpers ─────────────────────────────────────────────

    function showEmptyState() {
        $empty.style.display = '';
        $notConnected.style.display = 'none';
        $detailView.style.display = 'none';
        $indexForm.style.display = 'none';
    }

    function showNotConnected(msg) {
        $empty.style.display = 'none';
        $notConnected.style.display = '';
        $detailView.style.display = 'none';
        $indexForm.style.display = 'none';
        $notConnectedMsg.textContent = msg || 'Check Typesense configuration in Settings.';
    }

    function showDetailView() {
        $empty.style.display = 'none';
        $notConnected.style.display = 'none';
        $detailView.style.display = '';
        $indexForm.style.display = 'none';
    }

    function showIndexForm() {
        $empty.style.display = 'none';
        $notConnected.style.display = 'none';
        $detailView.style.display = 'none';
        $indexForm.style.display = '';
    }

    // ── Toast ────────────────────────────────────────────────────────────────

    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        $toast.textContent = message;
        $toast.className = 'dm-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () {
            $toast.classList.remove('show');
        }, 2500);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    function formatRelativeTime(unixTs) {
        if (!unixTs) return '';
        var now = Math.floor(Date.now() / 1000);
        var diff = now - unixTs;

        if (diff < 0) return 'just now';
        if (diff < 60) return diff + 's ago';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        if (diff < 2592000) return Math.floor(diff / 86400) + 'd ago';
        if (diff < 31536000) return Math.floor(diff / 2592000) + 'mo ago';
        return Math.floor(diff / 31536000) + 'y ago';
    }

    function getSourceBadgeClass(source) {
        var known = ['manual', 'web_fetch', 'pdf', 'scale'];
        for (var i = 0; i < known.length; i++) {
            if (source === known[i]) return 'source-' + source;
        }
        return 'source-default';
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function escapeAttr(str) {
        return escapeHtml(str);
    }

})();
