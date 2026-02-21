/* Captain Claw Workflow Browser - Client-Side Logic */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    var workflows = [];
    var selectedWorkflow = null;
    var currentOutputContent = '';

    // ── DOM References ───────────────────────────────────────
    var wfList = document.getElementById('wfList');
    var wfLoading = document.getElementById('wfLoading');
    var wfCount = document.getElementById('wfCount');
    var wfEmpty = document.getElementById('wfEmpty');
    var wfOutputBar = document.getElementById('wfOutputBar');
    var wfSelectedName = document.getElementById('wfSelectedName');
    var wfOutputSelect = document.getElementById('wfOutputSelect');
    var wfOutputMeta = document.getElementById('wfOutputMeta');
    var wfOutputContent = document.getElementById('wfOutputContent');
    var wfOutputLoading = document.getElementById('wfOutputLoading');
    var wfNoOutputs = document.getElementById('wfNoOutputs');

    // ── Init ─────────────────────────────────────────────────

    init();

    async function init() {
        if (wfOutputSelect) {
            wfOutputSelect.addEventListener('change', onOutputSelected);
        }
        await loadWorkflows();
    }

    // ── API ──────────────────────────────────────────────────

    async function apiFetch(url) {
        var res = await fetch(url);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return await res.json();
    }

    // ── Load Workflows ───────────────────────────────────────

    async function loadWorkflows() {
        try {
            var data = await apiFetch('/api/workflow-browser');
            workflows = Array.isArray(data) ? data : [];
        } catch (e) {
            workflows = [];
            console.error('Failed to load workflows:', e);
        }

        if (wfLoading) wfLoading.style.display = 'none';
        if (wfCount) wfCount.textContent = String(workflows.length);
        renderSidebar();
    }

    // ── Render Sidebar ───────────────────────────────────────

    function renderSidebar() {
        if (!wfList) return;

        // Remove all children except loading
        var children = Array.from(wfList.children);
        for (var i = 0; i < children.length; i++) {
            if (children[i] !== wfLoading) {
                wfList.removeChild(children[i]);
            }
        }

        if (workflows.length === 0) {
            var emptyDiv = document.createElement('div');
            emptyDiv.className = 'wf-loading';
            emptyDiv.textContent = 'No workflows found.';
            wfList.appendChild(emptyDiv);
            return;
        }

        for (var j = 0; j < workflows.length; j++) {
            var wf = workflows[j];
            var item = createWorkflowItem(wf);
            wfList.appendChild(item);
        }
    }

    function createWorkflowItem(wf) {
        var div = document.createElement('div');
        div.className = 'wf-item';
        if (selectedWorkflow && selectedWorkflow.filename === wf.filename) {
            div.classList.add('selected');
        }

        // Name
        var nameEl = document.createElement('div');
        nameEl.className = 'wf-item-name';
        nameEl.textContent = wf.name || wf.filename;
        div.appendChild(nameEl);

        // Meta row
        var metaEl = document.createElement('div');
        metaEl.className = 'wf-item-meta';

        // Task count
        if (wf.task_count > 0) {
            var tasksSpan = document.createElement('span');
            tasksSpan.textContent = wf.task_count + ' tasks';
            metaEl.appendChild(tasksSpan);
        }

        // Output count badge
        var outputCount = (wf.outputs || []).length;
        var badge = document.createElement('span');
        badge.className = 'wf-item-badge ' + (outputCount > 0 ? 'has-outputs' : 'no-outputs');
        badge.textContent = outputCount + ' output' + (outputCount !== 1 ? 's' : '');
        metaEl.appendChild(badge);

        div.appendChild(metaEl);

        // User input preview (truncated)
        if (wf.user_input) {
            var previewEl = document.createElement('div');
            previewEl.className = 'wf-item-meta';
            var previewText = wf.user_input;
            if (previewText.length > 80) previewText = previewText.substring(0, 80) + '...';
            previewEl.textContent = previewText;
            previewEl.style.marginTop = '2px';
            div.appendChild(previewEl);
        }

        div.addEventListener('click', function () {
            selectWorkflow(wf);
        });

        return div;
    }

    // ── Select Workflow ──────────────────────────────────────

    function selectWorkflow(wf) {
        selectedWorkflow = wf;
        renderSidebar(); // update selection highlight

        var outputs = wf.outputs || [];

        if (outputs.length === 0) {
            // No outputs
            showNoOutputs(wf);
            return;
        }

        // Show output bar and populate dropdown
        showOutputBar(wf);

        // Auto-load the first (newest) output
        if (wfOutputSelect && wfOutputSelect.options.length > 0) {
            loadOutput(wfOutputSelect.value);
        }
    }

    function showNoOutputs(wf) {
        if (wfEmpty) wfEmpty.style.display = 'none';
        if (wfOutputBar) wfOutputBar.style.display = 'none';
        if (wfOutputContent) wfOutputContent.style.display = 'none';
        if (wfOutputLoading) wfOutputLoading.style.display = 'none';
        if (wfNoOutputs) wfNoOutputs.style.display = 'flex';
        if (wfSelectedName) wfSelectedName.textContent = wf.name || wf.filename;
    }

    function showOutputBar(wf) {
        if (wfEmpty) wfEmpty.style.display = 'none';
        if (wfNoOutputs) wfNoOutputs.style.display = 'none';
        if (wfOutputBar) wfOutputBar.style.display = 'flex';

        if (wfSelectedName) {
            wfSelectedName.textContent = wf.name || wf.filename;
        }

        // Populate output selector
        if (wfOutputSelect) {
            wfOutputSelect.innerHTML = '';
            var outputs = wf.outputs || [];
            for (var i = 0; i < outputs.length; i++) {
                var o = outputs[i];
                var opt = document.createElement('option');
                opt.value = o.filename;
                opt.textContent = formatOutputLabel(o);
                wfOutputSelect.appendChild(opt);
            }
        }
    }

    function formatOutputLabel(output) {
        // Convert timestamp "20260221-105231" to "2026-02-21 10:52:31"
        var ts = output.timestamp || '';
        if (ts.length >= 15) {
            var formatted = ts.substring(0, 4) + '-' + ts.substring(4, 6) + '-' + ts.substring(6, 8)
                + ' ' + ts.substring(9, 11) + ':' + ts.substring(11, 13) + ':' + ts.substring(13, 15);
            return formatted;
        }
        return output.filename;
    }

    // ── Output Select Change ─────────────────────────────────

    function onOutputSelected() {
        if (!wfOutputSelect) return;
        var filename = wfOutputSelect.value;
        if (filename) {
            loadOutput(filename);
        }
    }

    // ── Load Output ──────────────────────────────────────────

    async function loadOutput(filename) {
        // Show loading
        if (wfOutputContent) wfOutputContent.style.display = 'none';
        if (wfNoOutputs) wfNoOutputs.style.display = 'none';
        if (wfOutputLoading) wfOutputLoading.style.display = 'flex';

        try {
            var data = await apiFetch('/api/workflow-browser/output/' + encodeURIComponent(filename));
            currentOutputContent = data.content || '';

            // Update meta
            if (wfOutputMeta) {
                var sizeKb = Math.round((currentOutputContent.length) / 1024);
                wfOutputMeta.textContent = sizeKb + ' KB';
            }

            // Render markdown
            renderMarkdown(currentOutputContent);

        } catch (e) {
            console.error('Failed to load output:', e);
            if (wfOutputContent) {
                wfOutputContent.innerHTML = '<p style="color:var(--red)">Failed to load output: ' + escapeHtml(e.message) + '</p>';
                wfOutputContent.style.display = 'block';
            }
        } finally {
            if (wfOutputLoading) wfOutputLoading.style.display = 'none';
        }
    }

    // ── Markdown Renderer ────────────────────────────────────
    // Simple markdown-to-HTML converter. Handles the most common
    // patterns found in workflow output files.

    function renderMarkdown(md) {
        if (!wfOutputContent) return;

        var html = markdownToHtml(md);
        wfOutputContent.innerHTML = html;
        wfOutputContent.style.display = 'block';
    }

    function markdownToHtml(md) {
        if (!md) return '';

        var lines = md.split('\n');
        var html = '';
        var inCodeBlock = false;
        var codeBlockLang = '';
        var codeLines = [];
        var inList = false;
        var listType = ''; // 'ul' or 'ol'

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i];

            // Code blocks (fenced)
            if (line.match(/^```/)) {
                if (inCodeBlock) {
                    // End code block
                    html += '<pre><code>' + escapeHtml(codeLines.join('\n')) + '</code></pre>\n';
                    codeLines = [];
                    inCodeBlock = false;
                    codeBlockLang = '';
                } else {
                    // Close any open list
                    if (inList) { html += '</' + listType + '>\n'; inList = false; }
                    // Start code block
                    inCodeBlock = true;
                    codeBlockLang = line.replace(/^```\s*/, '');
                }
                continue;
            }

            if (inCodeBlock) {
                codeLines.push(line);
                continue;
            }

            // Blank line — close lists, add spacing
            if (line.trim() === '') {
                if (inList) { html += '</' + listType + '>\n'; inList = false; }
                continue;
            }

            // Headings
            var headingMatch = line.match(/^(#{1,6})\s+(.*)/);
            if (headingMatch) {
                if (inList) { html += '</' + listType + '>\n'; inList = false; }
                var level = headingMatch[1].length;
                html += '<h' + level + '>' + inlineMarkdown(headingMatch[2]) + '</h' + level + '>\n';
                continue;
            }

            // Horizontal rule
            if (line.match(/^(-{3,}|_{3,}|\*{3,})\s*$/)) {
                if (inList) { html += '</' + listType + '>\n'; inList = false; }
                html += '<hr>\n';
                continue;
            }

            // Unordered list
            var ulMatch = line.match(/^(\s*)[*\-+]\s+(.*)/);
            if (ulMatch) {
                if (!inList || listType !== 'ul') {
                    if (inList) html += '</' + listType + '>\n';
                    html += '<ul>\n';
                    inList = true;
                    listType = 'ul';
                }
                html += '<li>' + inlineMarkdown(ulMatch[2]) + '</li>\n';
                continue;
            }

            // Ordered list
            var olMatch = line.match(/^(\s*)\d+\.\s+(.*)/);
            if (olMatch) {
                if (!inList || listType !== 'ol') {
                    if (inList) html += '</' + listType + '>\n';
                    html += '<ol>\n';
                    inList = true;
                    listType = 'ol';
                }
                html += '<li>' + inlineMarkdown(olMatch[2]) + '</li>\n';
                continue;
            }

            // Blockquote
            var bqMatch = line.match(/^>\s?(.*)/);
            if (bqMatch) {
                if (inList) { html += '</' + listType + '>\n'; inList = false; }
                html += '<blockquote>' + inlineMarkdown(bqMatch[1]) + '</blockquote>\n';
                continue;
            }

            // Table detection (simple)
            if (line.indexOf('|') !== -1 && line.trim().startsWith('|')) {
                if (inList) { html += '</' + listType + '>\n'; inList = false; }
                var tableResult = parseTable(lines, i);
                if (tableResult.html) {
                    html += tableResult.html;
                    i = tableResult.endIndex;
                    continue;
                }
            }

            // Regular paragraph
            if (inList) { html += '</' + listType + '>\n'; inList = false; }
            html += '<p>' + inlineMarkdown(line) + '</p>\n';
        }

        // Close any remaining open elements
        if (inCodeBlock) {
            html += '<pre><code>' + escapeHtml(codeLines.join('\n')) + '</code></pre>\n';
        }
        if (inList) {
            html += '</' + listType + '>\n';
        }

        return html;
    }

    function inlineMarkdown(text) {
        var s = escapeHtml(text);

        // Bold + italic: ***text*** or ___text___
        s = s.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
        s = s.replace(/___(.*?)___/g, '<strong><em>$1</em></strong>');

        // Bold: **text** or __text__
        s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/__(.*?)__/g, '<strong>$1</strong>');

        // Italic: *text* or _text_
        s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
        s = s.replace(/\b_(.*?)_\b/g, '<em>$1</em>');

        // Strikethrough: ~~text~~
        s = s.replace(/~~(.*?)~~/g, '<del>$1</del>');

        // Inline code: `code`
        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Links: [text](url)
        s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Images: ![alt](url)
        s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2">');

        return s;
    }

    function parseTable(lines, startIndex) {
        // Check if we have at least header + separator
        if (startIndex + 1 >= lines.length) return { html: '', endIndex: startIndex };

        var headerLine = lines[startIndex].trim();
        var sepLine = lines[startIndex + 1] ? lines[startIndex + 1].trim() : '';

        // Separator must be like |---|---|
        if (!sepLine.match(/^\|[\s\-:|]+\|$/)) {
            return { html: '', endIndex: startIndex };
        }

        var headers = splitTableRow(headerLine);
        if (headers.length === 0) return { html: '', endIndex: startIndex };

        var html = '<table>\n<thead><tr>';
        for (var h = 0; h < headers.length; h++) {
            html += '<th>' + inlineMarkdown(headers[h]) + '</th>';
        }
        html += '</tr></thead>\n<tbody>\n';

        var endIndex = startIndex + 1; // skip separator
        for (var r = startIndex + 2; r < lines.length; r++) {
            var rowLine = lines[r].trim();
            if (!rowLine.startsWith('|')) break;
            var cells = splitTableRow(rowLine);
            html += '<tr>';
            for (var c = 0; c < cells.length; c++) {
                html += '<td>' + inlineMarkdown(cells[c]) + '</td>';
            }
            html += '</tr>\n';
            endIndex = r;
        }

        html += '</tbody></table>\n';
        return { html: html, endIndex: endIndex };
    }

    function splitTableRow(line) {
        // Remove leading/trailing pipes and split
        var trimmed = line.replace(/^\|/, '').replace(/\|$/, '');
        var parts = trimmed.split('|');
        var result = [];
        for (var i = 0; i < parts.length; i++) {
            result.push(parts[i].trim());
        }
        return result;
    }

    // ── Utilities ────────────────────────────────────────────

    function escapeHtml(str) {
        if (str == null) return '';
        var div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

})();
