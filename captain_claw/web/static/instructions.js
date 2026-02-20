(function () {
    'use strict';

    // ── File Metadata Catalog ─────────────────────────────────────────
    // Maps each instruction filename to its group, description, and badge.
    // Files not listed here are placed in the "Other" group.

    var FILE_CATALOG = {
        'system_prompt.md': {
            group: 'Core System',
            desc: "Main system prompt — defines the agent's behavior, available tools, and workspace policies",
            badge: 'system',
        },
        'planning_mode_instructions.md': {
            group: 'Core System',
            desc: 'Extension injected when planning mode is active for step-by-step task execution',
            badge: 'system',
        },
        'planning_pipeline_header.md': {
            group: 'Core System',
            desc: 'Header template for planning context notes injected into conversation',
            badge: 'template',
        },
        'planning_pipeline_footer.md': {
            group: 'Core System',
            desc: 'Footer template for planning context notes injected into conversation',
            badge: 'template',
        },

        'task_contract_planner_system_prompt.md': {
            group: 'Task Planning & Verification',
            desc: 'Plans task contracts with nested subtasks, requirements, and prefetch URLs',
            badge: 'system',
        },
        'task_contract_planner_user_prompt.md': {
            group: 'Task Planning & Verification',
            desc: 'User prompt template for task contract planning — receives the user request',
            badge: 'user',
        },
        'task_contract_critic_system_prompt.md': {
            group: 'Task Planning & Verification',
            desc: 'Validates agent responses against task contract requirements',
            badge: 'system',
        },
        'task_contract_critic_user_prompt.md': {
            group: 'Task Planning & Verification',
            desc: 'User prompt template for requirement verification — receives request + response',
            badge: 'user',
        },

        'orchestrator_decompose_system_prompt.md': {
            group: 'Orchestrator',
            desc: 'Decomposes complex requests into parallel tasks with dependency tracking',
            badge: 'system',
        },
        'orchestrator_decompose_user_prompt.md': {
            group: 'Orchestrator',
            desc: 'User prompt template for task decomposition — receives request + session list',
            badge: 'user',
        },
        'orchestrator_rephrase_prompt.md': {
            group: 'Orchestrator',
            desc: 'Rephrases casual user requests into precise orchestrator prompts',
            badge: 'system',
        },
        'orchestrator_synthesize_user_prompt.md': {
            group: 'Orchestrator',
            desc: 'Synthesizes results from all parallel worker sessions into a final answer',
            badge: 'user',
        },
        'orchestrator_worker_prompt.md': {
            group: 'Orchestrator',
            desc: 'Prompt template for individual worker agents executing decomposed tasks',
            badge: 'system',
        },

        'guard_input_system_prompt.md': {
            group: 'Safety Guards',
            desc: 'Evaluates user input for safety before sending to the LLM',
            badge: 'system',
        },
        'guard_input_user_prompt.md': {
            group: 'Safety Guards',
            desc: 'User payload template for input guard — wraps the content to evaluate',
            badge: 'user',
        },
        'guard_output_system_prompt.md': {
            group: 'Safety Guards',
            desc: 'Evaluates LLM output for safety before showing to the user',
            badge: 'system',
        },
        'guard_output_user_prompt.md': {
            group: 'Safety Guards',
            desc: 'User payload template for output guard — wraps the response to evaluate',
            badge: 'user',
        },
        'guard_script_tool_system_prompt.md': {
            group: 'Safety Guards',
            desc: 'Evaluates scripts and commands before execution to prevent system damage',
            badge: 'system',
        },
        'guard_script_tool_user_prompt.md': {
            group: 'Safety Guards',
            desc: 'User payload template for script/tool guard — wraps the command to evaluate',
            badge: 'user',
        },

        'session_description_system_prompt.md': {
            group: 'Session & Context',
            desc: 'Generates concise session descriptions for future session selection',
            badge: 'system',
        },
        'session_description_user_prompt.md': {
            group: 'Session & Context',
            desc: 'User prompt template for session description — receives conversation excerpt',
            badge: 'user',
        },
        'compaction_summary_system_prompt.md': {
            group: 'Session & Context',
            desc: 'Summarizes earlier conversation when context window compaction is triggered',
            badge: 'system',
        },
        'compaction_summary_user_prompt.md': {
            group: 'Session & Context',
            desc: 'User prompt template for conversation summarization during compaction',
            badge: 'user',
        },
        'memory_continuity_header.md': {
            group: 'Session & Context',
            desc: 'Header for memory continuity context notes pulled from historical data',
            badge: 'template',
        },

        'script_synthesis_system_prompt.md': {
            group: 'Output Processing',
            desc: 'Generates standalone runnable Python scripts from user requests',
            badge: 'system',
        },
        'tool_output_rewrite_system_prompt.md': {
            group: 'Output Processing',
            desc: 'Rewrites raw tool output into friendly, concise final answers',
            badge: 'system',
        },
        'tool_output_rewrite_user_prompt.md': {
            group: 'Output Processing',
            desc: 'User payload template for tool output rewriting — receives raw output',
            badge: 'user',
        },

        'list_task_extractor_system_prompt.md': {
            group: 'List Processing',
            desc: 'Extracts per-member list tasks from requests for batch processing',
            badge: 'system',
        },
        'list_task_extractor_user_prompt.md': {
            group: 'List Processing',
            desc: 'User payload template for list member extraction from request context',
            badge: 'user',
        },

        'README.md': {
            group: 'Documentation',
            desc: 'Documents all instruction files and their purposes',
            badge: 'docs',
        },
    };

    // Display order for groups
    var GROUP_ORDER = [
        'Core System',
        'Task Planning & Verification',
        'Orchestrator',
        'Safety Guards',
        'Session & Context',
        'Output Processing',
        'List Processing',
        'Documentation',
    ];

    // ── State ─────────────────────────────────────────────────────────

    var currentFile = null;
    var currentOverridden = false;
    var isDirty = false;
    var allFiles = [];

    // ── DOM References ────────────────────────────────────────────────

    var $ = function (sel) { return document.querySelector(sel); };
    var instrGroups = $('#instrGroups');
    var instrSearch = $('#instrSearch');
    var editorEmpty = $('#instrEditorEmpty');
    var editorActive = $('#instrEditorActive');
    var editorBadge = $('#editorBadge');
    var editorFilename = $('#editorFilename');
    var editorDesc = $('#editorDesc');
    var editorDirty = $('#editorDirty');
    var editorOverride = $('#editorOverride');
    var editorSaveStatus = $('#editorSaveStatus');
    var textarea = $('#instrTextarea');
    var saveBtn = $('#saveBtn');
    var revertBtn = $('#revertBtn');
    var closeBtn = $('#closeBtn');

    // ── Utilities ─────────────────────────────────────────────────────

    function escapeHtml(str) {
        var el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }

    function getMeta(name) {
        return FILE_CATALOG[name] || {
            group: 'Other',
            desc: 'Custom instruction file',
            badge: 'system',
        };
    }

    function setDirty(dirty) {
        isDirty = dirty;
        editorDirty.classList.toggle('hidden', !dirty);
        if (dirty) editorSaveStatus.textContent = '';
    }

    function updateOverrideUI(overridden) {
        currentOverridden = overridden;
        editorOverride.classList.toggle('hidden', !overridden);
        revertBtn.classList.toggle('hidden', !overridden);
    }

    // ── Render Groups ─────────────────────────────────────────────────

    function renderGroups(files, filter) {
        var grouped = {};
        var i, g;
        for (i = 0; i < GROUP_ORDER.length; i++) grouped[GROUP_ORDER[i]] = [];
        grouped['Other'] = [];

        for (i = 0; i < files.length; i++) {
            var f = files[i];
            var meta = getMeta(f.name);
            var group = meta.group;
            if (!grouped[group]) grouped[group] = [];

            if (filter) {
                var q = filter.toLowerCase();
                var matches = f.name.toLowerCase().indexOf(q) !== -1
                    || meta.desc.toLowerCase().indexOf(q) !== -1
                    || group.toLowerCase().indexOf(q) !== -1;
                if (!matches) continue;
            }
            grouped[group].push({
                name: f.name,
                size: f.size,
                desc: meta.desc,
                badge: meta.badge,
                overridden: !!f.overridden,
            });
        }

        var allGroupNames = GROUP_ORDER.concat(['Other']);
        var html = '';
        var totalVisible = 0;

        for (g = 0; g < allGroupNames.length; g++) {
            var groupName = allGroupNames[g];
            var items = grouped[groupName];
            if (!items || items.length === 0) continue;
            totalVisible += items.length;

            html += '<div class="instr-group" data-group="' + escapeHtml(groupName) + '">';
            html += '<div class="instr-group-header">';
            html += '<span class="instr-group-chevron">&#x25BC;</span>';
            html += '<span class="instr-group-name">' + escapeHtml(groupName) + '</span>';
            html += '<span class="instr-group-count">' + items.length + '</span>';
            html += '</div>';
            html += '<div class="instr-group-files">';

            for (i = 0; i < items.length; i++) {
                var item = items[i];
                var sizeKb = (item.size / 1024).toFixed(1);
                var isActive = currentFile === item.name;
                var cardClass = 'instr-file-card';
                if (isActive) cardClass += ' active';
                if (item.overridden) cardClass += ' overridden';
                html += '<div class="' + cardClass + '" data-name="' + escapeHtml(item.name) + '">';
                html += '<span class="instr-file-card-badge ' + item.badge + '">' + item.badge + '</span>';
                html += '<div class="instr-file-card-info">';
                html += '<div class="instr-file-card-name">' + escapeHtml(item.name);
                if (item.overridden) {
                    html += ' <span class="instr-file-card-customized" title="Customized — saved to ~/.captain-claw/instructions/">customized</span>';
                }
                html += '</div>';
                html += '<div class="instr-file-card-desc">' + escapeHtml(item.desc) + '</div>';
                html += '</div>';
                html += '<span class="instr-file-card-size">' + sizeKb + ' KB</span>';
                html += '</div>';
            }
            html += '</div></div>';
        }

        if (totalVisible === 0 && filter) {
            html = '<div class="instr-no-results">No files matching "' + escapeHtml(filter) + '"</div>';
        }

        instrGroups.innerHTML = html;

        // Bind group collapse
        var headers = instrGroups.querySelectorAll('.instr-group-header');
        for (i = 0; i < headers.length; i++) {
            (function (header) {
                header.addEventListener('click', function () {
                    header.parentElement.classList.toggle('collapsed');
                });
            })(headers[i]);
        }

        // Bind file card clicks
        var cards = instrGroups.querySelectorAll('.instr-file-card');
        for (i = 0; i < cards.length; i++) {
            (function (card) {
                card.addEventListener('click', function () {
                    var name = card.dataset.name;
                    if (isDirty && currentFile && currentFile !== name) {
                        if (!confirm('Unsaved changes in "' + currentFile + '". Discard and open "' + name + '"?')) return;
                    }
                    openFile(name);
                });
            })(cards[i]);
        }
    }

    // ── File Operations ───────────────────────────────────────────────

    function openFile(name) {
        fetch('/api/instructions/' + encodeURIComponent(name))
            .then(function (res) {
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
            })
            .then(function (data) {
                var meta = getMeta(name);
                currentFile = name;

                editorFilename.textContent = name;
                editorDesc.textContent = meta.desc;
                editorBadge.textContent = meta.badge;
                editorBadge.className = 'instr-editor-badge ' + meta.badge;

                textarea.value = data.content;
                setDirty(false);
                updateOverrideUI(!!data.overridden);

                editorEmpty.style.display = 'none';
                editorActive.classList.remove('hidden');

                // Highlight in sidebar
                var allCards = instrGroups.querySelectorAll('.instr-file-card');
                for (var i = 0; i < allCards.length; i++) {
                    allCards[i].classList.toggle('active', allCards[i].dataset.name === name);
                }

                textarea.focus();
                textarea.scrollTop = 0;
            })
            .catch(function (e) {
                alert('Failed to load "' + name + '": ' + e.message);
            });
    }

    function saveFile() {
        if (!currentFile) return;
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving\u2026';
        fetch('/api/instructions/' + encodeURIComponent(currentFile), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: textarea.value }),
        })
            .then(function (res) {
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
            })
            .then(function (data) {
                setDirty(false);
                updateOverrideUI(!!data.overridden);
                editorSaveStatus.textContent = 'Saved \u2713';
                setTimeout(function () { editorSaveStatus.textContent = ''; }, 2000);
                loadFiles();
            })
            .catch(function (e) {
                alert('Failed to save "' + currentFile + '": ' + e.message);
            })
            .finally(function () {
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save';
            });
    }

    function revertFile() {
        if (!currentFile || !currentOverridden) return;
        if (!confirm(
            'Revert "' + currentFile + '" to the system default?\n\n' +
            'Your customized version in ~/.captain-claw/instructions/ will be deleted.'
        )) return;

        revertBtn.disabled = true;
        fetch('/api/instructions/' + encodeURIComponent(currentFile), {
            method: 'DELETE',
        })
            .then(function (res) {
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
            })
            .then(function (data) {
                textarea.value = data.content;
                setDirty(false);
                updateOverrideUI(false);
                editorSaveStatus.textContent = 'Reverted \u2713';
                setTimeout(function () { editorSaveStatus.textContent = ''; }, 2000);
                loadFiles();
            })
            .catch(function (e) {
                alert('Failed to revert "' + currentFile + '": ' + e.message);
            })
            .finally(function () {
                revertBtn.disabled = false;
            });
    }

    function closeFile() {
        if (isDirty) {
            if (!confirm('Unsaved changes in "' + currentFile + '". Discard?')) return;
        }
        currentFile = null;
        currentOverridden = false;
        setDirty(false);
        editorActive.classList.add('hidden');
        editorEmpty.style.display = '';

        var allCards = instrGroups.querySelectorAll('.instr-file-card');
        for (var i = 0; i < allCards.length; i++) {
            allCards[i].classList.remove('active');
        }
    }

    // ── Load Files ────────────────────────────────────────────────────

    function loadFiles() {
        fetch('/api/instructions')
            .then(function (res) { return res.json(); })
            .then(function (files) {
                allFiles = files;
                renderGroups(allFiles, instrSearch.value);
            })
            .catch(function () {
                instrGroups.innerHTML = '<div class="instr-loading">Failed to load instruction files.</div>';
            });
    }

    // ── Event Bindings ────────────────────────────────────────────────

    saveBtn.addEventListener('click', saveFile);
    revertBtn.addEventListener('click', revertFile);
    closeBtn.addEventListener('click', closeFile);

    textarea.addEventListener('input', function () { setDirty(true); });

    // Tab key inserts spaces
    textarea.addEventListener('keydown', function (e) {
        if (e.key === 'Tab') {
            e.preventDefault();
            var start = textarea.selectionStart;
            var end = textarea.selectionEnd;
            var value = textarea.value;
            textarea.value = value.substring(0, start) + '    ' + value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 4;
            setDirty(true);
        }
    });

    // Ctrl+S / Cmd+S
    document.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            if (currentFile) saveFile();
        }
    });

    // Search filter with debounce
    var searchTimeout;
    instrSearch.addEventListener('input', function () {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
            renderGroups(allFiles, instrSearch.value);
        }, 150);
    });

    // Warn about unsaved changes on page leave
    window.addEventListener('beforeunload', function (e) {
        if (isDirty) {
            e.preventDefault();
            e.returnValue = '';
        }
    });

    // ── Init ──────────────────────────────────────────────────────────

    loadFiles();

})();
