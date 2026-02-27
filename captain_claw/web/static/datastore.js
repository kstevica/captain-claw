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
                ['dsCreateTableModal', 'dsRowModal', 'dsAddColModal', 'dsConfirmModal'].forEach(function (id) {
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

        renderTableList();
        showTableView();
        loadRows();
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

        apiFetch('/api/datastore/tables/' + encodeURIComponent(selectedTable.name) +
            '/rows?limit=' + pageSize + '&offset=' + pageOffset + '&order_by=_id&order_dir=ASC'
        ).then(function (result) {
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

    function renderGrid() {
        dsGridHead.innerHTML = '';
        dsGridBody.innerHTML = '';

        if (columns.length === 0) return;

        // Header
        var headRow = document.createElement('tr');
        columns.forEach(function (col) {
            var th = document.createElement('th');
            th.textContent = col.name;
            if (col.name === '_id') th.classList.add('col-id');
            headRow.appendChild(th);
        });
        // Actions column
        var thAct = document.createElement('th');
        thAct.className = 'col-actions';
        thAct.textContent = '';
        headRow.appendChild(thAct);
        dsGridHead.appendChild(headRow);

        // Body
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

        rows.forEach(function (row) {
            var tr = document.createElement('tr');
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
                tr.appendChild(td);
            });

            // Action buttons
            var tdAct = document.createElement('td');
            tdAct.className = 'col-actions';

            var editBtn = document.createElement('button');
            editBtn.className = 'ds-row-action';
            editBtn.textContent = 'Edit';
            editBtn.title = 'Edit row';
            editBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                showRowModal(row);
            });
            tdAct.appendChild(editBtn);

            var delBtn = document.createElement('button');
            delBtn.className = 'ds-row-action danger';
            delBtn.textContent = 'Del';
            delBtn.title = 'Delete row';
            delBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                onDeleteRow(row);
            });
            tdAct.appendChild(delBtn);

            tr.appendChild(tdAct);
            dsGridBody.appendChild(tr);
        });
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

        schemaCols.forEach(function (col) {
            var row = document.createElement('div');
            row.className = 'ds-schema-row';

            var nameEl = document.createElement('span');
            nameEl.className = 'ds-schema-col-name';
            nameEl.textContent = col.name;
            row.appendChild(nameEl);

            var typeEl = document.createElement('span');
            typeEl.className = 'ds-schema-col-type';
            typeEl.textContent = col.type;
            row.appendChild(typeEl);

            var dropBtn = document.createElement('button');
            dropBtn.className = 'ds-row-action danger';
            dropBtn.textContent = 'Drop';
            dropBtn.title = 'Drop column';
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
        $('dsRowModalTitle').textContent = row ? 'Edit Row #' + row._id : 'Add Row';

        var fields = $('dsRowModalFields');
        fields.innerHTML = '';

        // Use schema columns (not _id)
        var schemaCols = selectedTable.columns || [];
        schemaCols.forEach(function (col) {
            var group = document.createElement('div');
            group.className = 'ds-form-group';

            var label = document.createElement('label');
            label.className = 'ds-form-label';
            label.textContent = col.name + ' (' + col.type + ')';
            group.appendChild(label);

            var input = document.createElement('input');
            input.className = 'ds-input';
            input.type = 'text';
            input.setAttribute('data-col', col.name);
            input.setAttribute('data-type', col.type);
            input.placeholder = col.type;

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
        var firstInput = fields.querySelector('.ds-input');
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
                showTableView();
                loadRows();
                loadTables();
                if (activeTab === 'schema') renderSchema();
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
