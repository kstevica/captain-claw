/* Captain Claw — Datastore Browser */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────────────
    var tables = [];
    var selectedTable = null;   // TableInfo object
    var rows = [];
    var columns = [];           // [{name, type, position}]
    var totalRows = 0;
    var pageSize = 50;
    var pageOffset = 0;
    var activeTab = 'data';     // 'data' | 'schema'
    var editingRowId = null;    // null = add mode, number = edit mode
    var confirmCallback = null;

    // Sort & filter state
    var sortCol = '_id';
    var sortDir = 'ASC';        // 'ASC' | 'DESC'
    var filters = {};           // { colName: "filterText", ... }
    var filterDebounceTimer = null;

    // Protection state
    var protections = [];       // array of {level, row_id, col_name, reason, ...}
    var allTableProtections = {};  // { tableName: true } for sidebar lock icons

    // ── DOM refs ─────────────────────────────────────────────────────
    var $ = function (id) { return document.getElementById(id); };

    var dsTableList     = $('dsTableList');
    var dsTableLoading  = $('dsTableLoading');
    var dsEmpty         = $('dsEmpty');
    var dsTableView     = $('dsTableView');
    var dsTableName     = $('dsTableName');
    var dsTableMeta     = $('dsTableMeta');
    var dsTabData       = $('dsTabData');
    var dsTabSchema     = $('dsTabSchema');
    var dsDataTab       = $('dsDataTab');
    var dsSchemaTab     = $('dsSchemaTab');
    var dsGridHead      = $('dsGridHead');
    var dsGridBody      = $('dsGridBody');
    var dsPrevBtn       = $('dsPrevBtn');
    var dsNextBtn       = $('dsNextBtn');
    var dsPageInfo      = $('dsPageInfo');
    var dsSchemaList    = $('dsSchemaList');
    var dsToast         = $('dsToast');

    // ── Init ─────────────────────────────────────────────────────────
    init();

    function init() {
        $('dsNewTableBtn').addEventListener('click', showCreateTableModal);
        $('dsAddRowBtn').addEventListener('click', function () { showRowModal(null); });
        $('dsDropTableBtn').addEventListener('click', onDropTable);
        $('dsAddColBtn').addEventListener('click', showAddColumnModal);
        $('dsProtectionsBtn').addEventListener('click', showProtectionsModal);
        $('dsProtectionsClose').addEventListener('click', function () { hideModal('dsProtectionsModal'); });
        $('dsProtTableCheck').addEventListener('change', onToggleTableProtection);

        // Tabs
        dsTabData.addEventListener('click', function () { switchTab('data'); });
        dsTabSchema.addEventListener('click', function () { switchTab('schema'); });

        // Pagination
        dsPrevBtn.addEventListener('click', function () {
            if (pageOffset >= pageSize) { pageOffset -= pageSize; loadRows(); }
        });
        dsNextBtn.addEventListener('click', function () {
            if (pageOffset + pageSize < totalRows) { pageOffset += pageSize; loadRows(); }
        });

        // Create table modal
        $('dsCreateTableCancel').addEventListener('click', function () { hideModal('dsCreateTableModal'); });
        $('dsCreateTableSave').addEventListener('click', onCreateTable);
        $('dsAddNewCol').addEventListener('click', addColumnRow);

        // Row modal
        $('dsRowModalCancel').addEventListener('click', function () { hideModal('dsRowModal'); });
        $('dsRowModalSave').addEventListener('click', onSaveRow);

        // Add column modal
        $('dsAddColCancel').addEventListener('click', function () { hideModal('dsAddColModal'); });
        $('dsAddColSave').addEventListener('click', onAddColumn);

        // Confirm modal
        $('dsConfirmCancel').addEventListener('click', function () { hideModal('dsConfirmModal'); confirmCallback = null; });
        $('dsConfirmOk').addEventListener('click', function () {
            hideModal('dsConfirmModal');
            if (confirmCallback) { confirmCallback(); confirmCallback = null; }
        });

        // Escape key
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                ['dsCreateTableModal', 'dsRowModal', 'dsAddColModal', 'dsConfirmModal', 'dsProtectionsModal'].forEach(function (id) {
                    $(id).style.display = 'none';
                });
                confirmCallback = null;
            }
        });

        loadTables();
    }

    // ── Table list ───────────────────────────────────────────────────

    function loadTables() {
        dsTableLoading.style.display = '';
        apiFetch('/api/datastore/tables').then(function (data) {
            dsTableLoading.style.display = 'none';
            tables = Array.isArray(data) ? data : [];
            renderTableList();
            loadAllTableProtections();
        }).catch(function () {
            dsTableLoading.style.display = 'none';
            tables = [];
            renderTableList();
        });
    }

    function renderTableList() {
        // Clear existing items
        dsTableList.querySelectorAll('.ds-table-item, .ds-list-empty').forEach(function (el) {
            el.remove();
        });

        if (tables.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'ds-list-empty';
            empty.textContent = 'No tables yet';
            dsTableList.appendChild(empty);
            return;
        }

        tables.forEach(function (t) {
            var el = document.createElement('div');
            el.className = 'ds-table-item';
            if (selectedTable && selectedTable.name === t.name) {
                el.classList.add('selected');
            }
            if (allTableProtections[t.name]) {
                el.classList.add('protected');
            }

            var nameEl = document.createElement('div');
            nameEl.className = 'ds-table-item-name';
            nameEl.textContent = t.name;
            el.appendChild(nameEl);

            var metaEl = document.createElement('div');
            metaEl.className = 'ds-table-item-meta';
            metaEl.textContent = t.columns.length + ' cols, ' + t.row_count + ' rows';
            el.appendChild(metaEl);

            el.addEventListener('click', function () { selectTable(t); });
            dsTableList.appendChild(el);
        });
    }

    function selectTable(t) {
        selectedTable = t;
        columns = t.columns || [];
        pageOffset = 0;
        activeTab = 'data';
        sortCol = '_id';
        sortDir = 'ASC';
        filters = {};
        protections = [];

        renderTableList();
        showTableView();
        loadProtections(function () {
            loadRows();
        });
    }

    // ── Show / hide panels ───────────────────────────────────────────

    function showEmpty() {
        dsEmpty.style.display = '';
        dsTableView.style.display = 'none';
    }

    function showTableView() {
        dsEmpty.style.display = 'none';
        dsTableView.style.display = '';
        dsTableName.textContent = selectedTable.name;
        dsTableMeta.textContent = columns.length + ' columns, ' + (selectedTable.row_count || 0) + ' rows';
        // Apply table-level protection state to buttons
        var tableProt = isTableProtected();
        $('dsAddRowBtn').disabled = tableProt;
        $('dsDropTableBtn').disabled = tableProt;
        $('dsAddColBtn').disabled = tableProt;
        if (tableProt) {
            $('dsAddRowBtn').classList.add('ds-btn-disabled');
            $('dsDropTableBtn').classList.add('ds-btn-disabled');
            $('dsAddColBtn').classList.add('ds-btn-disabled');
        } else {
            $('dsAddRowBtn').classList.remove('ds-btn-disabled');
            $('dsDropTableBtn').classList.remove('ds-btn-disabled');
            $('dsAddColBtn').classList.remove('ds-btn-disabled');
        }
        switchTab(activeTab);
    }

    function switchTab(tab) {
        activeTab = tab;
        dsTabData.classList.toggle('active', tab === 'data');
        dsTabSchema.classList.toggle('active', tab === 'schema');
        dsDataTab.style.display = tab === 'data' ? '' : 'none';
        dsSchemaTab.style.display = tab === 'schema' ? '' : 'none';
        if (tab === 'schema') renderSchema();
    }

    // ── Data grid ────────────────────────────────────────────────────

    function loadRows() {
        if (!selectedTable) return;

        var url = '/api/datastore/tables/' + encodeURIComponent(selectedTable.name) +
            '/rows?limit=' + pageSize + '&offset=' + pageOffset +
            '&order_by=' + encodeURIComponent(sortCol) +
            '&order_dir=' + encodeURIComponent(sortDir);

        // Build where clause from active filters
        var where = buildWhereFromFilters();
        if (where) {
            url += '&where=' + encodeURIComponent(JSON.stringify(where));
        }

        apiFetch(url).then(function (result) {
            rows = result.rows || [];
            totalRows = result.total || 0;
            var resultCols = result.columns || [];
            // Update column list from query result (includes _id)
            if (resultCols.length > 0) {
                // Keep schema columns but ensure _id is present
                var allCols = resultCols.map(function (name, idx) {
                    var found = columns.find(function (c) { return c.name === name; });
                    return { name: name, type: found ? found.type : 'text', position: idx };
                });
                columns = allCols;
            }
            renderGrid();
            renderPagination();
            // Update meta
            dsTableMeta.textContent = (selectedTable.columns || []).length + ' columns, ' + totalRows + ' rows';
            selectedTable.row_count = totalRows;
        }).catch(function (err) {
            rows = [];
            totalRows = 0;
            renderGrid();
            renderPagination();
            showToast('Load failed: ' + (err.message || err), 'error');
        });
    }

    function buildWhereFromFilters() {
        var where = {};
        var hasAny = false;
        for (var colName in filters) {
            var val = (filters[colName] || '').trim();
            if (!val) continue;
            // Find column type
            var colInfo = columns.find(function (c) { return c.name === colName; });
            var colType = colInfo ? colInfo.type : 'text';

            if (colType === 'boolean') {
                var lower = val.toLowerCase();
                if (lower === 'true' || lower === '1' || lower === 'yes') {
                    where[colName] = 1;
                } else if (lower === 'false' || lower === '0' || lower === 'no') {
                    where[colName] = 0;
                } else {
                    continue; // skip invalid boolean filter
                }
            } else if ((colType === 'integer' || colType === 'real') && !isNaN(Number(val))) {
                where[colName] = colType === 'integer' ? parseInt(val, 10) : parseFloat(val);
            } else {
                where[colName] = { op: 'LIKE', value: '%' + val + '%' };
            }
            hasAny = true;
        }
        return hasAny ? where : null;
    }

    function renderGrid() {
        dsGridHead.innerHTML = '';
        dsGridBody.innerHTML = '';

        if (columns.length === 0) return;

        // ── Header row (sortable) ──
        var headRow = document.createElement('tr');
        columns.forEach(function (col) {
            var th = document.createElement('th');
            th.classList.add('sortable');
            if (col.name === '_id') th.classList.add('col-id');
            if (col.name !== '_id' && isColumnProtected(col.name)) {
                th.classList.add('ds-protected-col');
            }

            // Label
            var label = document.createElement('span');
            label.className = 'ds-th-label';
            label.textContent = col.name;
            th.appendChild(label);

            // Sort indicator
            if (sortCol === col.name) {
                th.classList.add(sortDir === 'ASC' ? 'sort-asc' : 'sort-desc');
            }

            th.addEventListener('click', function () { onSortClick(col.name); });
            headRow.appendChild(th);
        });
        // Actions column header
        var thAct = document.createElement('th');
        thAct.className = 'col-actions';
        thAct.textContent = '';
        headRow.appendChild(thAct);
        dsGridHead.appendChild(headRow);

        // ── Filter row ──
        var filterRow = document.createElement('tr');
        filterRow.className = 'ds-filter-row';
        columns.forEach(function (col) {
            var td = document.createElement('td');
            if (col.name === '_id') { td.classList.add('col-id'); }
            var input = document.createElement('input');
            input.type = 'text';
            input.className = 'ds-filter-input';
            input.placeholder = 'Filter...';
            input.setAttribute('data-col', col.name);
            // Restore current filter value
            if (filters[col.name]) { input.value = filters[col.name]; }

            input.addEventListener('input', function () {
                var colName = this.getAttribute('data-col');
                var val = this.value;
                if (val.trim()) {
                    filters[colName] = val;
                } else {
                    delete filters[colName];
                }
                // Debounced reload
                if (filterDebounceTimer) clearTimeout(filterDebounceTimer);
                filterDebounceTimer = setTimeout(function () {
                    pageOffset = 0;
                    loadRows();
                }, 400);
            });
            input.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') {
                    // Immediate filter on Enter
                    if (filterDebounceTimer) clearTimeout(filterDebounceTimer);
                    pageOffset = 0;
                    loadRows();
                }
            });
            td.appendChild(input);
            filterRow.appendChild(td);
        });
        // Empty cell for actions column
        var tdFilterAct = document.createElement('td');
        tdFilterAct.className = 'col-actions';
        filterRow.appendChild(tdFilterAct);
        dsGridHead.appendChild(filterRow);

        // ── Body rows ──
        if (rows.length === 0) {
            var emptyRow = document.createElement('tr');
            var emptyCell = document.createElement('td');
            emptyCell.colSpan = columns.length + 1;
            emptyCell.className = 'ds-grid-empty';
            emptyCell.textContent = 'No rows';
            emptyRow.appendChild(emptyCell);
            dsGridBody.appendChild(emptyRow);
            return;
        }

        var tableProt = isTableProtected();
        rows.forEach(function (row) {
            var tr = document.createElement('tr');
            var rowProt = isRowProtected(row._id);
            if (rowProt) tr.classList.add('ds-protected-row');

            columns.forEach(function (col) {
                var td = document.createElement('td');
                var val = row[col.name];
                if (val === null || val === undefined) {
                    td.textContent = '';
                    td.classList.add('null-val');
                } else if (typeof val === 'object') {
                    td.textContent = JSON.stringify(val);
                    td.classList.add('json-val');
                } else if (typeof val === 'boolean') {
                    td.textContent = val ? 'true' : 'false';
                } else {
                    td.textContent = String(val);
                }
                if (col.name === '_id') td.classList.add('col-id');
                if (col.name !== '_id' && isCellProtected(row._id, col.name)) {
                    td.classList.add('ds-protected-cell');
                }
                // Right-click to toggle cell protection (skip _id column)
                if (col.name !== '_id') {
                    var cellProt = isCellProtected(row._id, col.name);
                    td.title = cellProt ? 'Right-click to unlock cell' : 'Right-click to lock cell';
                    (function (rowId, colName) {
                        td.addEventListener('contextmenu', function (e) {
                            e.preventDefault();
                            toggleCellProtection(rowId, colName);
                        });
                    })(row._id, col.name);
                }
                tr.appendChild(td);
            });

            // Action buttons
            var tdAct = document.createElement('td');
            tdAct.className = 'col-actions';

            var editBtn = document.createElement('button');
            editBtn.className = 'ds-row-action';
            editBtn.textContent = 'Edit';
            editBtn.title = 'Edit row';
            if (rowProt || tableProt) {
                editBtn.disabled = true;
                editBtn.classList.add('ds-btn-disabled');
            }
            editBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                showRowModal(row);
            });
            tdAct.appendChild(editBtn);

            var delBtn = document.createElement('button');
            delBtn.className = 'ds-row-action danger';
            delBtn.textContent = 'Del';
            delBtn.title = 'Delete row';
            if (rowProt || tableProt) {
                delBtn.disabled = true;
                delBtn.classList.add('ds-btn-disabled');
            }
            delBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                onDeleteRow(row);
            });
            tdAct.appendChild(delBtn);

            // Row protection toggle (lock icon)
            var lockBtn = document.createElement('button');
            lockBtn.className = 'ds-row-action ds-lock-btn' + (rowProt ? ' active' : '');
            lockBtn.textContent = rowProt ? '\u{1F512}' : '\u{1F513}';
            lockBtn.title = rowProt ? 'Unlock row' : 'Lock row';
            lockBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                toggleRowProtection(row._id);
            });
            tdAct.appendChild(lockBtn);

            tr.appendChild(tdAct);
            dsGridBody.appendChild(tr);
        });
    }

    function onSortClick(colName) {
        if (sortCol === colName) {
            // Toggle direction, or reset on third click
            if (sortDir === 'ASC') {
                sortDir = 'DESC';
            } else {
                // Reset to default
                sortCol = '_id';
                sortDir = 'ASC';
            }
        } else {
            sortCol = colName;
            sortDir = 'ASC';
        }
        pageOffset = 0;
        loadRows();
    }

    function renderPagination() {
        var start = totalRows > 0 ? pageOffset + 1 : 0;
        var end = Math.min(pageOffset + pageSize, totalRows);
        dsPageInfo.textContent = start + '-' + end + ' of ' + totalRows;
        dsPrevBtn.disabled = pageOffset === 0;
        dsNextBtn.disabled = pageOffset + pageSize >= totalRows;
    }

    // ── Schema view ──────────────────────────────────────────────────

    function renderSchema() {
        dsSchemaList.innerHTML = '';
        if (!selectedTable) return;

        // Show schema columns (not _id from query, but from the table descriptor)
        var schemaCols = selectedTable.columns || [];
        if (schemaCols.length === 0) {
            dsSchemaList.innerHTML = '<div class="ds-list-empty">No columns defined</div>';
            return;
        }

        var tableProt = isTableProtected();
        schemaCols.forEach(function (col) {
            var row = document.createElement('div');
            row.className = 'ds-schema-row';
            var colProt = isColumnProtected(col.name);
            if (colProt) row.classList.add('ds-protected-col');

            var nameEl = document.createElement('span');
            nameEl.className = 'ds-schema-col-name';
            nameEl.textContent = (colProt ? '\u{1F512} ' : '') + col.name;
            row.appendChild(nameEl);

            var typeEl = document.createElement('span');
            typeEl.className = 'ds-schema-col-type';
            typeEl.textContent = col.type;
            row.appendChild(typeEl);

            // Column protection toggle
            var colLockBtn = document.createElement('button');
            colLockBtn.className = 'ds-row-action ds-lock-btn' + (colProt ? ' active' : '');
            colLockBtn.textContent = colProt ? '\u{1F512}' : '\u{1F513}';
            colLockBtn.title = colProt ? 'Unlock column' : 'Lock column';
            colLockBtn.addEventListener('click', function () {
                toggleColumnProtection(col.name);
            });
            row.appendChild(colLockBtn);

            var dropBtn = document.createElement('button');
            dropBtn.className = 'ds-row-action danger';
            dropBtn.textContent = 'Drop';
            dropBtn.title = 'Drop column';
            if (colProt || tableProt) {
                dropBtn.disabled = true;
                dropBtn.classList.add('ds-btn-disabled');
            }
            dropBtn.addEventListener('click', function () {
                showConfirm('Drop column "' + col.name + '"?', function () {
                    onDropColumn(col.name);
                });
            });
            row.appendChild(dropBtn);

            dsSchemaList.appendChild(row);
        });
    }

    // ── Create table ─────────────────────────────────────────────────

    function showCreateTableModal() {
        $('dsNewTableName').value = '';
        var colsDiv = $('dsNewTableCols');
        colsDiv.innerHTML = '';
        addColumnRow();
        showModal('dsCreateTableModal');
        setTimeout(function () { $('dsNewTableName').focus(); }, 50);
    }

    function addColumnRow() {
        var colsDiv = $('dsNewTableCols');
        var row = document.createElement('div');
        row.className = 'ds-col-row';
        row.innerHTML =
            '<input type="text" class="ds-input col-name" placeholder="Column name">' +
            '<select class="ds-select col-type">' +
            '<option value="text">text</option>' +
            '<option value="integer">integer</option>' +
            '<option value="real">real</option>' +
            '<option value="boolean">boolean</option>' +
            '<option value="date">date</option>' +
            '<option value="datetime">datetime</option>' +
            '<option value="json">json</option>' +
            '</select>' +
            '<button class="ds-col-remove" title="Remove">&times;</button>';
        row.querySelector('.ds-col-remove').addEventListener('click', function () {
            row.remove();
        });
        colsDiv.appendChild(row);
    }

    function onCreateTable() {
        var name = $('dsNewTableName').value.trim();
        if (!name) { showToast('Table name is required', 'error'); return; }

        var colRows = $('dsNewTableCols').querySelectorAll('.ds-col-row');
        var cols = [];
        for (var i = 0; i < colRows.length; i++) {
            var cname = colRows[i].querySelector('.col-name').value.trim();
            var ctype = colRows[i].querySelector('.col-type').value;
            if (cname) cols.push({ name: cname, col_type: ctype });
        }
        if (cols.length === 0) { showToast('At least one column is required', 'error'); return; }

        apiFetch('/api/datastore/tables', { method: 'POST', body: { name: name, columns: cols } })
            .then(function (result) {
                if (result.error) { showToast(result.error, 'error'); return; }
                hideModal('dsCreateTableModal');
                showToast('Table created', 'success');
                loadTables();
                // Select the new table
                setTimeout(function () {
                    var found = tables.find(function (t) { return t.name === name; });
                    if (found) selectTable(found);
                }, 300);
            })
            .catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
    }

    // ── Drop table ───────────────────────────────────────────────────

    function onDropTable() {
        if (!selectedTable) return;
        showConfirm('Drop table "' + selectedTable.name + '"? All data will be lost.', function () {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name), { method: 'DELETE' })
                .then(function () {
                    showToast('Table dropped', 'success');
                    selectedTable = null;
                    showEmpty();
                    loadTables();
                })
                .catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        });
    }

    // ── Row modal (add / edit) ───────────────────────────────────────

    function showRowModal(row) {
        editingRowId = row ? row._id : null;
        var rowProt = row && isRowProtected(row._id);
        var tableProt = isTableProtected();

        if (row && (rowProt || tableProt)) {
            showToast('This row is protected and cannot be edited', 'error');
            return;
        }

        $('dsRowModalTitle').textContent = row ? 'Edit Row #' + row._id : 'Add Row';

        var fields = $('dsRowModalFields');
        fields.innerHTML = '';

        // Use schema columns (not _id)
        var schemaCols = selectedTable.columns || [];
        schemaCols.forEach(function (col) {
            var group = document.createElement('div');
            group.className = 'ds-form-group';

            var cellProt = row && isCellProtected(row._id, col.name);

            var label = document.createElement('label');
            label.className = 'ds-form-label';
            label.textContent = col.name + ' (' + col.type + ')' + (cellProt ? ' \u{1F512}' : '');
            group.appendChild(label);

            var input = document.createElement('input');
            input.className = 'ds-input';
            input.type = 'text';
            input.setAttribute('data-col', col.name);
            input.setAttribute('data-type', col.type);
            input.placeholder = col.type;

            if (cellProt) {
                input.disabled = true;
                input.classList.add('ds-input-protected');
            }

            if (row) {
                var val = row[col.name];
                if (val !== null && val !== undefined) {
                    if (typeof val === 'object') {
                        input.value = JSON.stringify(val);
                    } else {
                        input.value = String(val);
                    }
                }
            }

            group.appendChild(input);
            fields.appendChild(group);
        });

        showModal('dsRowModal');
        var firstInput = fields.querySelector('.ds-input:not(:disabled)');
        if (firstInput) setTimeout(function () { firstInput.focus(); }, 50);
    }

    function onSaveRow() {
        var inputs = $('dsRowModalFields').querySelectorAll('[data-col]');
        var rowData = {};

        for (var i = 0; i < inputs.length; i++) {
            var col = inputs[i].getAttribute('data-col');
            var type = inputs[i].getAttribute('data-type');
            var val = inputs[i].value;

            if (val === '') {
                rowData[col] = null;
            } else if (type === 'integer') {
                rowData[col] = parseInt(val, 10);
                if (isNaN(rowData[col])) { showToast(col + ' must be an integer', 'error'); return; }
            } else if (type === 'real') {
                rowData[col] = parseFloat(val);
                if (isNaN(rowData[col])) { showToast(col + ' must be a number', 'error'); return; }
            } else if (type === 'boolean') {
                var lower = val.toLowerCase();
                rowData[col] = (lower === 'true' || lower === '1' || lower === 'yes');
            } else if (type === 'json') {
                try { rowData[col] = JSON.parse(val); }
                catch (e) { showToast(col + ' must be valid JSON', 'error'); return; }
            } else {
                rowData[col] = val;
            }
        }

        var tableName = selectedTable.name;

        if (editingRowId !== null) {
            // Update
            apiFetch('/api/datastore/tables/' + encodeURIComponent(tableName) + '/rows', {
                method: 'PATCH',
                body: { set_values: rowData, where: { _id: editingRowId } }
            }).then(function (result) {
                if (result.error) { showToast(result.error, 'error'); return; }
                hideModal('dsRowModal');
                showToast('Row updated', 'success');
                loadRows();
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        } else {
            // Insert
            apiFetch('/api/datastore/tables/' + encodeURIComponent(tableName) + '/rows', {
                method: 'POST',
                body: { rows: [rowData] }
            }).then(function (result) {
                if (result.error) { showToast(result.error, 'error'); return; }
                hideModal('dsRowModal');
                showToast('Row added', 'success');
                loadRows();
                // Update table list counts
                loadTables();
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        }
    }

    // ── Delete row ───────────────────────────────────────────────────

    function onDeleteRow(row) {
        showConfirm('Delete row #' + row._id + '?', function () {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/rows', {
                method: 'DELETE',
                body: { where: { _id: row._id } }
            }).then(function () {
                showToast('Row deleted', 'success');
                loadRows();
                loadTables();
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        });
    }

    // ── Add column ───────────────────────────────────────────────────

    function showAddColumnModal() {
        $('dsAddColName').value = '';
        $('dsAddColType').value = 'text';
        $('dsAddColDefault').value = '';
        showModal('dsAddColModal');
        setTimeout(function () { $('dsAddColName').focus(); }, 50);
    }

    function onAddColumn() {
        var colName = $('dsAddColName').value.trim();
        var colType = $('dsAddColType').value;
        var def = $('dsAddColDefault').value.trim() || null;

        if (!colName) { showToast('Column name is required', 'error'); return; }

        apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/columns', {
            method: 'POST',
            body: { col_name: colName, col_type: colType, default: def }
        }).then(function (result) {
            if (result.error) { showToast(result.error, 'error'); return; }
            hideModal('dsAddColModal');
            showToast('Column added', 'success');
            // Refresh table info
            refreshSelectedTable();
        }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
    }

    // ── Drop column ──────────────────────────────────────────────────

    function onDropColumn(colName) {
        apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) +
            '/columns/' + encodeURIComponent(colName), { method: 'DELETE' }
        ).then(function () {
            showToast('Column dropped', 'success');
            refreshSelectedTable();
        }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
    }

    // ── Refresh selected table info ──────────────────────────────────

    function refreshSelectedTable() {
        if (!selectedTable) return;
        apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name))
            .then(function (info) {
                selectedTable = info;
                columns = info.columns || [];
                loadProtections(function () {
                    showTableView();
                    loadRows();
                    loadTables();
                    if (activeTab === 'schema') renderSchema();
                });
            })
            .catch(function () {});
    }

    // ── Modal helpers ────────────────────────────────────────────────

    function showModal(id) { $(id).style.display = ''; }
    function hideModal(id) { $(id).style.display = 'none'; }

    function showConfirm(message, callback) {
        $('dsConfirmBody').textContent = message;
        confirmCallback = callback;
        showModal('dsConfirmModal');
    }

    // ── Toast ────────────────────────────────────────────────────────

    var toastTimer = null;
    function showToast(message, type) {
        if (toastTimer) clearTimeout(toastTimer);
        dsToast.textContent = message;
        dsToast.className = 'ds-toast ' + (type || '') + ' show';
        toastTimer = setTimeout(function () { dsToast.classList.remove('show'); }, 2500);
    }

    // ── Protection helpers ─────────────────────────────────────────────

    function isTableProtected() {
        return protections.some(function (p) { return p.level === 'table'; });
    }

    function isRowProtected(rowId) {
        return protections.some(function (p) {
            return p.level === 'row' && p.row_id === rowId;
        });
    }

    function isCellProtected(rowId, colName) {
        return protections.some(function (p) {
            return p.level === 'cell' && p.row_id === rowId && p.col_name === colName;
        });
    }

    function isColumnProtected(colName) {
        return protections.some(function (p) {
            return p.level === 'column' && p.col_name === colName;
        });
    }

    function loadProtections(callback) {
        if (!selectedTable) {
            protections = [];
            if (callback) callback();
            return;
        }
        apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections')
            .then(function (data) {
                protections = Array.isArray(data) ? data : [];
                // Update sidebar lock state
                allTableProtections[selectedTable.name] = isTableProtected();
                showTableView();
                renderTableList();
                if (callback) callback();
            })
            .catch(function () {
                protections = [];
                if (callback) callback();
            });
    }

    function loadAllTableProtections() {
        // For each table, load protections to check table-level locks
        // We do this in bulk by loading for each table
        allTableProtections = {};
        var remaining = tables.length;
        if (remaining === 0) { renderTableList(); return; }
        tables.forEach(function (t) {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(t.name) + '/protections')
                .then(function (data) {
                    var prots = Array.isArray(data) ? data : [];
                    allTableProtections[t.name] = prots.some(function (p) { return p.level === 'table'; });
                    remaining--;
                    if (remaining <= 0) renderTableList();
                })
                .catch(function () {
                    remaining--;
                    if (remaining <= 0) renderTableList();
                });
        });
    }

    function toggleRowProtection(rowId) {
        if (!selectedTable) return;
        var prot = isRowProtected(rowId);
        if (prot) {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'DELETE',
                body: { level: 'row', row_id: rowId }
            }).then(function () {
                showToast('Row unlocked', 'success');
                loadProtections(function () { renderGrid(); });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        } else {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'POST',
                body: { level: 'row', row_id: rowId }
            }).then(function () {
                showToast('Row locked', 'success');
                loadProtections(function () { renderGrid(); });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        }
    }

    function toggleColumnProtection(colName) {
        if (!selectedTable) return;
        var prot = isColumnProtected(colName);
        if (prot) {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'DELETE',
                body: { level: 'column', col_name: colName }
            }).then(function () {
                showToast('Column unlocked', 'success');
                loadProtections(function () {
                    renderGrid();
                    if (activeTab === 'schema') renderSchema();
                });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        } else {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'POST',
                body: { level: 'column', col_name: colName }
            }).then(function () {
                showToast('Column locked', 'success');
                loadProtections(function () {
                    renderGrid();
                    if (activeTab === 'schema') renderSchema();
                });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        }
    }

    function toggleCellProtection(rowId, colName) {
        if (!selectedTable) return;
        var prot = isCellProtected(rowId, colName);
        if (prot) {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'DELETE',
                body: { level: 'cell', row_id: rowId, col_name: colName }
            }).then(function () {
                showToast('Cell unlocked', 'success');
                loadProtections(function () { renderGrid(); });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        } else {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'POST',
                body: { level: 'cell', row_id: rowId, col_name: colName }
            }).then(function () {
                showToast('Cell locked', 'success');
                loadProtections(function () { renderGrid(); });
            }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
        }
    }

    function onToggleTableProtection() {
        if (!selectedTable) return;
        var checked = $('dsProtTableCheck').checked;
        if (checked) {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'POST',
                body: { level: 'table' }
            }).then(function () {
                showToast('Table protected', 'success');
                loadProtections(function () { renderProtectionsModal(); });
            }).catch(function (err) {
                showToast('Failed: ' + (err.message || err), 'error');
                $('dsProtTableCheck').checked = false;
            });
        } else {
            apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                method: 'DELETE',
                body: { level: 'table' }
            }).then(function () {
                showToast('Table unprotected', 'success');
                loadProtections(function () { renderProtectionsModal(); });
            }).catch(function (err) {
                showToast('Failed: ' + (err.message || err), 'error');
                $('dsProtTableCheck').checked = true;
            });
        }
    }

    function showProtectionsModal() {
        if (!selectedTable) return;
        loadProtections(function () {
            renderProtectionsModal();
            showModal('dsProtectionsModal');
        });
    }

    function renderProtectionsModal() {
        $('dsProtTableCheck').checked = isTableProtected();

        var listEl = $('dsProtectionsList');
        listEl.innerHTML = '';

        // Show non-table protections
        var nonTable = protections.filter(function (p) { return p.level !== 'table'; });
        if (nonTable.length === 0) {
            listEl.innerHTML = '<div class="ds-list-empty">No row, column, or cell protections</div>';
            return;
        }

        nonTable.forEach(function (p) {
            var row = document.createElement('div');
            row.className = 'ds-protection-item';

            var info = document.createElement('span');
            info.className = 'ds-protection-info';
            var text = p.level;
            if (p.row_id !== null && p.row_id !== undefined) text += ' | row #' + p.row_id;
            if (p.col_name) text += ' | ' + p.col_name;
            if (p.reason) text += ' (' + p.reason + ')';
            info.textContent = text;
            row.appendChild(info);

            var removeBtn = document.createElement('button');
            removeBtn.className = 'ds-row-action danger';
            removeBtn.textContent = 'Remove';
            removeBtn.addEventListener('click', function () {
                apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) + '/protections', {
                    method: 'DELETE',
                    body: { level: p.level, row_id: p.row_id, col_name: p.col_name }
                }).then(function () {
                    showToast('Protection removed', 'success');
                    loadProtections(function () {
                        renderProtectionsModal();
                        renderGrid();
                        if (activeTab === 'schema') renderSchema();
                    });
                }).catch(function (err) { showToast('Failed: ' + (err.message || err), 'error'); });
            });
            row.appendChild(removeBtn);

            listEl.appendChild(row);
        });
    }

    // ── API utility ──────────────────────────────────────────────────

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

})();
