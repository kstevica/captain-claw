/* Captain Claw — File Browser */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    var allFiles = [];
    var filteredFiles = [];
    var selectedFile = null;
    var searchQuery = '';
    var searchTimeout = null;

    // ── DOM references ───────────────────────────────────────
    var fbSearch = document.getElementById('fbSearch');
    var fbCount = document.getElementById('fbCount');
    var fbFileList = document.getElementById('fbFileList');
    var fbLoading = document.getElementById('fbLoading');
    var fbEmpty = document.getElementById('fbEmpty');
    var fbInfoBar = document.getElementById('fbInfoBar');
    var fbInfoName = document.getElementById('fbInfoName');
    var fbInfoPath = document.getElementById('fbInfoPath');
    var fbInfoMeta = document.getElementById('fbInfoMeta');
    var fbDownloadBtn = document.getElementById('fbDownloadBtn');
    var fbContent = document.getElementById('fbContent');
    var fbCode = document.getElementById('fbCode');
    var fbBinary = document.getElementById('fbBinary');
    var fbBinaryText = document.getElementById('fbBinaryText');
    var fbBinaryDownload = document.getElementById('fbBinaryDownload');
    var fbMissing = document.getElementById('fbMissing');
    var fbMissingPath = document.getElementById('fbMissingPath');
    var fbContentLoading = document.getElementById('fbContentLoading');
    var fbToast = document.getElementById('fbToast');

    // ── Init ─────────────────────────────────────────────────
    fbSearch.addEventListener('input', onSearchInput);
    loadFiles();

    // ── API helper ───────────────────────────────────────────
    function apiFetch(url) {
        return fetch(url).then(function (res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return res.json();
        });
    }

    // ── Load file list ───────────────────────────────────────
    function loadFiles() {
        apiFetch('/api/files').then(function (data) {
            allFiles = Array.isArray(data) ? data : [];
            applyFilter();
            if (fbLoading) fbLoading.style.display = 'none';
        }).catch(function (err) {
            allFiles = [];
            applyFilter();
            if (fbLoading) fbLoading.style.display = 'none';
            showToast('Failed to load files: ' + err.message, 'error');
        });
    }

    // ── Search / filter ──────────────────────────────────────
    function onSearchInput() {
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            searchQuery = fbSearch.value.trim().toLowerCase();
            applyFilter();
        }, 200);
    }

    function applyFilter() {
        if (!searchQuery) {
            filteredFiles = allFiles;
        } else {
            filteredFiles = allFiles.filter(function (f) {
                return f.filename.toLowerCase().indexOf(searchQuery) >= 0 ||
                       f.logical.toLowerCase().indexOf(searchQuery) >= 0 ||
                       (f.extension || '').toLowerCase().indexOf(searchQuery) >= 0;
            });
        }
        fbCount.textContent = String(filteredFiles.length);
        renderFileList();
    }

    // ── Render file list ─────────────────────────────────────
    function renderFileList() {
        var items = fbFileList.querySelectorAll('.fb-file-item, .fb-list-empty');
        for (var i = 0; i < items.length; i++) items[i].remove();

        if (filteredFiles.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'fb-list-empty';
            empty.textContent = searchQuery ? 'No files match filter' : 'No files registered';
            fbFileList.appendChild(empty);
            return;
        }

        for (var j = 0; j < filteredFiles.length; j++) {
            fbFileList.appendChild(createFileItem(filteredFiles[j]));
        }
    }

    function createFileItem(file) {
        var div = document.createElement('div');
        div.className = 'fb-file-item';
        if (selectedFile && selectedFile.physical === file.physical) {
            div.classList.add('selected');
        }
        if (!file.exists) div.classList.add('missing');

        var icon = document.createElement('span');
        icon.className = 'fb-file-icon';
        icon.textContent = getFileIcon(file);
        div.appendChild(icon);

        var info = document.createElement('div');
        info.className = 'fb-file-info';

        var name = document.createElement('div');
        name.className = 'fb-file-name';
        name.textContent = file.filename;
        info.appendChild(name);

        var path = document.createElement('div');
        path.className = 'fb-file-path';
        path.textContent = file.logical;
        info.appendChild(path);

        div.appendChild(info);

        if (file.exists && file.size > 0) {
            var size = document.createElement('span');
            size.className = 'fb-file-size';
            size.textContent = formatSize(file.size);
            div.appendChild(size);
        }

        div.addEventListener('click', function () { selectFile(file); });
        return div;
    }

    // ── Select file ──────────────────────────────────────────
    function selectFile(file) {
        selectedFile = file;
        renderFileList();

        // Hide all main area states.
        fbEmpty.style.display = 'none';
        fbContent.style.display = 'none';
        fbBinary.style.display = 'none';
        fbMissing.style.display = 'none';
        fbContentLoading.style.display = 'none';

        // Show info bar.
        fbInfoBar.style.display = 'flex';
        fbInfoName.textContent = file.filename;
        fbInfoPath.textContent = file.logical;
        fbInfoMeta.textContent = formatSize(file.size) + '  \u00B7  ' + (file.mime_type || 'unknown');

        var downloadUrl = '/api/files/download?path=' + encodeURIComponent(file.physical);

        if (!file.exists) {
            fbMissing.style.display = 'flex';
            fbMissingPath.textContent = file.physical;
            fbDownloadBtn.style.display = 'none';
            return;
        }

        fbDownloadBtn.style.display = '';
        fbDownloadBtn.href = downloadUrl;

        if (file.is_text) {
            fbContentLoading.style.display = 'flex';
            apiFetch('/api/files/content?path=' + encodeURIComponent(file.physical))
                .then(function (data) {
                    fbContentLoading.style.display = 'none';
                    fbCode.textContent = data.content || '';
                    fbContent.style.display = 'block';
                })
                .catch(function (err) {
                    fbContentLoading.style.display = 'none';
                    showToast('Failed to read file: ' + err.message, 'error');
                    showBinaryView(file, downloadUrl);
                });
        } else {
            showBinaryView(file, downloadUrl);
        }
    }

    function showBinaryView(file, downloadUrl) {
        fbBinaryText.textContent = (file.mime_type || 'binary') + '  \u00B7  ' + formatSize(file.size);
        fbBinaryDownload.href = downloadUrl;
        fbBinary.style.display = 'flex';
    }

    // ── Helpers ──────────────────────────────────────────────
    function getFileIcon(file) {
        if (!file.exists) return '\u26A0';
        var ext = (file.extension || '').toLowerCase();
        if (ext === '.py') return '\uD83D\uDC0D';
        if ('.js.ts.jsx.tsx'.indexOf(ext) >= 0) return '\uD83D\uDFE8';
        if ('.json.yaml.yml.toml'.indexOf(ext) >= 0) return '\u2699';
        if ('.md.markdown.txt.rst.log'.indexOf(ext) >= 0) return '\uD83D\uDCC4';
        if ('.html.htm.css.scss'.indexOf(ext) >= 0) return '\uD83C\uDF10';
        if ('.png.jpg.jpeg.gif.svg.webp'.indexOf(ext) >= 0) return '\uD83D\uDDBC';
        if (ext === '.pdf') return '\uD83D\uDCD5';
        if ('.zip.tar.gz.bz2.7z'.indexOf(ext) >= 0) return '\uD83D\uDCE6';
        if ('.csv.tsv.xlsx'.indexOf(ext) >= 0) return '\uD83D\uDCCA';
        if (ext === '.sql') return '\uD83D\uDDC4';
        if ('.sh.bash.zsh'.indexOf(ext) >= 0) return '\uD83D\uDCBB';
        if (file.is_text) return '\uD83D\uDCC4';
        return '\uD83D\uDCC1';
    }

    function formatSize(bytes) {
        if (!bytes || bytes <= 0) return '0 B';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // ── Toast ────────────────────────────────────────────────
    var toastTimer = null;
    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        fbToast.textContent = message;
        fbToast.className = 'fb-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () {
            fbToast.classList.remove('show');
        }, 3000);
    }

})();
