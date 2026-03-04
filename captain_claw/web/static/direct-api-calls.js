/* Direct API Calls — vanilla JS IIFE */
(function () {
    "use strict";

    const API = "/api/direct-api-calls";

    // ── State ────────────────────────────────────────────────
    let allCalls = [];
    let selectedId = null;
    let editMode = false;

    // ── DOM refs ─────────────────────────────────────────────
    const $search       = document.getElementById("dacSearch");
    const $count        = document.getElementById("dacCount");
    const $appFilter    = document.getElementById("dacAppFilter");
    const $itemList     = document.getElementById("dacItemList");
    const $loading      = document.getElementById("dacLoading");
    const $empty        = document.getElementById("dacEmpty");
    const $detailView   = document.getElementById("dacDetailView");
    const $detailTitle  = document.getElementById("dacDetailTitle");
    const $detailMeta   = document.getElementById("dacDetailMeta");
    const $detailContent = document.getElementById("dacDetailContent");
    const $editBtn      = document.getElementById("dacEditBtn");
    const $saveBtn      = document.getElementById("dacSaveBtn");
    const $cancelEditBtn = document.getElementById("dacCancelEditBtn");
    const $deleteBtn    = document.getElementById("dacDeleteBtn");
    const $execPanel    = document.getElementById("dacExecutePanel");
    const $execPayload  = document.getElementById("dacExecPayload");
    const $execQP       = document.getElementById("dacExecQueryParams");
    const $execBtn      = document.getElementById("dacExecuteBtn");
    const $execResult   = document.getElementById("dacExecuteResult");
    const $execStatus   = document.getElementById("dacExecStatus");
    const $execElapsed  = document.getElementById("dacExecElapsed");
    const $execResponse = document.getElementById("dacExecResponse");
    const $addBtn       = document.getElementById("dacAddBtn");
    const $createView   = document.getElementById("dacCreateView");
    const $createSave   = document.getElementById("dacCreateSaveBtn");
    const $createCancel = document.getElementById("dacCreateCancelBtn");
    const $modalOverlay = document.getElementById("dacModalOverlay");
    const $modalBody    = document.getElementById("dacModalBody");
    const $modalCancel  = document.getElementById("dacModalCancel");
    const $modalConfirm = document.getElementById("dacModalConfirm");
    const $toast        = document.getElementById("dacToast");

    // ── Init ─────────────────────────────────────────────────
    loadCalls();
    $search.addEventListener("input", renderList);
    $appFilter.addEventListener("change", renderList);
    $editBtn.addEventListener("click", enterEditMode);
    $saveBtn.addEventListener("click", saveEdit);
    $cancelEditBtn.addEventListener("click", exitEditMode);
    $deleteBtn.addEventListener("click", confirmDelete);
    $execBtn.addEventListener("click", executeCall);
    $addBtn.addEventListener("click", showCreateForm);
    $createSave.addEventListener("click", createCall);
    $createCancel.addEventListener("click", hideCreateForm);
    $modalCancel.addEventListener("click", () => { $modalOverlay.style.display = "none"; });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            if ($modalOverlay.style.display !== "none") {
                $modalOverlay.style.display = "none";
            } else if ($createView.style.display !== "none") {
                hideCreateForm();
            } else if (editMode) {
                exitEditMode();
            }
        }
    });

    // ── API ──────────────────────────────────────────────────

    async function loadCalls() {
        try {
            const res = await fetch(API);
            allCalls = await res.json();
        } catch (e) {
            allCalls = [];
        }
        $loading.style.display = "none";
        buildAppFilter();
        renderList();
        if (selectedId) showDetail(selectedId);
    }

    function buildAppFilter() {
        const apps = new Set();
        allCalls.forEach(c => { if (c.app_name) apps.add(c.app_name); });
        $appFilter.innerHTML = '<option value="">All apps</option>';
        [...apps].sort().forEach(a => {
            const opt = document.createElement("option");
            opt.value = a;
            opt.textContent = a;
            $appFilter.appendChild(opt);
        });
    }

    function renderList() {
        const q = $search.value.toLowerCase().trim();
        const app = $appFilter.value;
        const filtered = allCalls.filter(c => {
            if (app && c.app_name !== app) return false;
            if (q) {
                const hay = `${c.name} ${c.url} ${c.method} ${c.description} ${c.tags || ""}`.toLowerCase();
                return hay.includes(q);
            }
            return true;
        });
        $count.textContent = filtered.length;
        $itemList.innerHTML = "";
        if (!filtered.length) {
            $itemList.innerHTML = '<div class="dac-loading">No API calls found</div>';
            return;
        }
        filtered.forEach(c => {
            const el = document.createElement("div");
            el.className = "dac-item" + (c.id === selectedId ? " active" : "");
            el.dataset.id = c.id;
            const ml = c.method.toLowerCase();
            el.innerHTML = `
                <span class="dac-item-method ${ml}">${c.method}</span>
                <div class="dac-item-info">
                    <div class="dac-item-name">${esc(c.name)}</div>
                    <div class="dac-item-url">${esc(truncUrl(c.url))}</div>
                </div>
                ${c.app_name ? `<span class="dac-item-app">${esc(c.app_name)}</span>` : ""}
            `;
            el.addEventListener("click", () => showDetail(c.id));
            $itemList.appendChild(el);
        });
    }

    function showDetail(id) {
        selectedId = id;
        editMode = false;
        const c = allCalls.find(x => x.id === id);
        if (!c) return;

        hideCreateForm();
        $empty.style.display = "none";
        $detailView.style.display = "flex";
        updateEditButtons(false);

        $detailTitle.textContent = c.name;
        const ml = c.method.toLowerCase();
        $detailMeta.innerHTML = `
            <span class="dac-method-badge ${ml}">${c.method}</span>
            <span>Used ${c.use_count} time${c.use_count !== 1 ? "s" : ""}</span>
            ${c.last_status_code ? `<span>Last: ${c.last_status_code}</span>` : ""}
            ${c.app_name ? `<span>${esc(c.app_name)}</span>` : ""}
        `;

        $detailContent.innerHTML = renderDetailFields(c);

        // Reset execute panel
        $execPayload.value = "";
        $execQP.value = "";
        $execResult.style.display = "none";

        // Highlight list item
        document.querySelectorAll(".dac-item").forEach(el => {
            el.classList.toggle("active", el.dataset.id === id);
        });
    }

    function renderDetailFields(c) {
        let html = "";

        html += field("URL", `<div class="dac-field-value mono">${esc(c.url)}</div>`);

        if (c.description) {
            html += field("Description", `<div class="dac-field-value">${esc(c.description)}</div>`);
        }

        // Auth row
        if (c.auth_type) {
            const masked = c.auth_token ? maskToken(c.auth_token) : "(not set)";
            html += `<div class="dac-field-row">`;
            html += `<div class="dac-field">${fieldInner("Auth Type", c.auth_type + (c.auth_source ? ` (${c.auth_source})` : ""))}</div>`;
            html += `<div class="dac-field">${fieldInner("Auth Token", `<span class="dac-auth-masked">${esc(masked)}</span>`)}</div>`;
            html += `</div>`;
        }

        if (c.input_payload) {
            html += field("Input Payload Schema", `<div class="dac-field-value code-block">${esc(c.input_payload)}</div>`);
        }

        if (c.result_payload) {
            html += field("Result Payload Schema", `<div class="dac-field-value code-block">${esc(c.result_payload)}</div>`);
        }

        if (c.headers) {
            html += field("Extra Headers", `<div class="dac-field-value code-block">${esc(c.headers)}</div>`);
        }

        if (c.tags) {
            const tags = c.tags.split(",").map(t => t.trim()).filter(Boolean);
            html += field("Tags", `<div class="dac-tags">${tags.map(t => `<span class="dac-tag">${esc(t)}</span>`).join("")}</div>`);
        }

        if (c.last_response_preview) {
            html += field("Last Response Preview", `<div class="dac-field-value code-block">${esc(c.last_response_preview)}</div>`);
        }

        return html;
    }

    // ── Edit mode ────────────────────────────────────────────

    function enterEditMode() {
        editMode = true;
        updateEditButtons(true);
        const c = allCalls.find(x => x.id === selectedId);
        if (!c) return;

        $detailContent.innerHTML = `
            <div class="dac-edit-field">
                <label>Name</label>
                <input class="dac-edit-input" id="dacEditName" value="${attr(c.name)}">
            </div>
            <div class="dac-edit-field">
                <label>URL</label>
                <input class="dac-edit-input" id="dacEditUrl" value="${attr(c.url)}">
            </div>
            <div class="dac-edit-row">
                <div class="dac-edit-field">
                    <label>Method</label>
                    <select class="dac-edit-select" id="dacEditMethod">
                        ${["GET","POST","PUT","PATCH"].map(m => `<option value="${m}"${m===c.method ? " selected" : ""}>${m}</option>`).join("")}
                    </select>
                </div>
                <div class="dac-edit-field">
                    <label>App Name</label>
                    <input class="dac-edit-input" id="dacEditAppName" value="${attr(c.app_name || "")}">
                </div>
            </div>
            <div class="dac-edit-field">
                <label>Description</label>
                <textarea class="dac-edit-textarea" id="dacEditDescription" style="min-height:60px;">${esc(c.description || "")}</textarea>
            </div>
            <div class="dac-edit-field">
                <label>Input Payload Schema</label>
                <textarea class="dac-edit-textarea" id="dacEditInputPayload">${esc(c.input_payload || "")}</textarea>
            </div>
            <div class="dac-edit-field">
                <label>Result Payload Schema</label>
                <textarea class="dac-edit-textarea" id="dacEditResultPayload">${esc(c.result_payload || "")}</textarea>
            </div>
            <div class="dac-edit-row">
                <div class="dac-edit-field">
                    <label>Auth Type</label>
                    <select class="dac-edit-select" id="dacEditAuthType">
                        ${["","bearer","api_key","basic","cookie","custom"].map(v => `<option value="${v}"${v===(c.auth_type||"") ? " selected" : ""}>${v || "None"}</option>`).join("")}
                    </select>
                </div>
                <div class="dac-edit-field">
                    <label>Auth Token</label>
                    <input class="dac-edit-input" type="password" id="dacEditAuthToken" value="${attr(c.auth_token || "")}" placeholder="Leave empty to keep current">
                </div>
            </div>
            <div class="dac-edit-field">
                <label>Extra Headers (JSON)</label>
                <textarea class="dac-edit-textarea" id="dacEditHeaders">${esc(c.headers || "")}</textarea>
            </div>
            <div class="dac-edit-field">
                <label>Tags</label>
                <input class="dac-edit-input" id="dacEditTags" value="${attr(c.tags || "")}">
            </div>
        `;
    }

    async function saveEdit() {
        const c = allCalls.find(x => x.id === selectedId);
        if (!c) return;

        const body = {
            name: val("dacEditName"),
            url: val("dacEditUrl"),
            method: val("dacEditMethod"),
            description: val("dacEditDescription"),
            input_payload: val("dacEditInputPayload"),
            result_payload: val("dacEditResultPayload"),
            auth_type: val("dacEditAuthType") || null,
            auth_token: val("dacEditAuthToken") || null,
            headers: val("dacEditHeaders") || null,
            app_name: val("dacEditAppName") || null,
            tags: val("dacEditTags") || null,
        };

        // Don't send empty auth_token if user didn't change it — keep existing
        if (!body.auth_token && c.auth_token) {
            delete body.auth_token;
        }

        try {
            const res = await fetch(`${API}/${c.id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json();
                toast(err.error || "Save failed", true);
                return;
            }
            toast("Saved");
            await loadCalls();
            showDetail(c.id);
        } catch (e) {
            toast("Network error", true);
        }
    }

    function exitEditMode() {
        editMode = false;
        updateEditButtons(false);
        if (selectedId) showDetail(selectedId);
    }

    function updateEditButtons(editing) {
        $editBtn.style.display = editing ? "none" : "";
        $deleteBtn.style.display = editing ? "none" : "";
        $saveBtn.style.display = editing ? "" : "none";
        $cancelEditBtn.style.display = editing ? "" : "none";
        $execPanel.style.display = editing ? "none" : "";
    }

    // ── Delete ───────────────────────────────────────────────

    function confirmDelete() {
        const c = allCalls.find(x => x.id === selectedId);
        if (!c) return;
        $modalBody.textContent = `Delete "${c.name}"? This cannot be undone.`;
        $modalOverlay.style.display = "flex";
        $modalConfirm.onclick = async () => {
            $modalOverlay.style.display = "none";
            try {
                await fetch(`${API}/${c.id}`, { method: "DELETE" });
                toast("Deleted");
                selectedId = null;
                $detailView.style.display = "none";
                $empty.style.display = "flex";
                await loadCalls();
            } catch (e) {
                toast("Delete failed", true);
            }
        };
    }

    // ── Execute ──────────────────────────────────────────────

    async function executeCall() {
        const c = allCalls.find(x => x.id === selectedId);
        if (!c) return;

        $execBtn.disabled = true;
        $execBtn.innerHTML = '<span class="dac-spinner"></span>Executing...';
        $execResult.style.display = "none";

        const body = {};
        const pl = $execPayload.value.trim();
        const qp = $execQP.value.trim();
        if (pl) {
            try { body.payload = JSON.parse(pl); } catch (e) { body.payload = pl; }
        }
        if (qp) {
            try { body.query_params = JSON.parse(qp); } catch (e) { body.query_params = qp; }
        }

        try {
            const res = await fetch(`${API}/${c.id}/execute`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            displayResult(data);
            // Reload to update use_count
            await loadCalls();
            // Re-highlight
            document.querySelectorAll(".dac-item").forEach(el => {
                el.classList.toggle("active", el.dataset.id === selectedId);
            });
        } catch (e) {
            displayResult({ success: false, error: e.message, status_code: 0, elapsed_ms: 0, response_body: "" });
        }

        $execBtn.disabled = false;
        $execBtn.innerHTML = "&#x25B6; Execute";
    }

    function displayResult(data) {
        $execResult.style.display = "block";

        const sc = data.status_code || 0;
        let cls = "error";
        if (sc >= 200 && sc < 300) cls = "success";
        else if (sc >= 300 && sc < 400) cls = "redirect";

        if (data.error && !data.success) {
            $execStatus.className = "dac-exec-status error";
            $execStatus.textContent = data.error;
        } else {
            $execStatus.className = `dac-exec-status ${cls}`;
            $execStatus.textContent = `${sc}`;
        }

        $execElapsed.textContent = `${data.elapsed_ms || 0}ms`;

        // Try to pretty-print JSON
        let bodyText = data.response_body || "";
        try {
            const parsed = JSON.parse(bodyText);
            bodyText = JSON.stringify(parsed, null, 2);
        } catch (e) { /* keep as-is */ }
        $execResponse.textContent = bodyText;
    }

    // ── Create form ──────────────────────────────────────────

    function showCreateForm() {
        selectedId = null;
        editMode = false;
        $empty.style.display = "none";
        $detailView.style.display = "none";
        $createView.style.display = "flex";
        // Clear form
        ["dacCreateName","dacCreateUrl","dacCreateAppName","dacCreateDescription",
         "dacCreateInputPayload","dacCreateResultPayload","dacCreateHeaders","dacCreateTags","dacCreateAuthToken"]
            .forEach(id => { document.getElementById(id).value = ""; });
        document.getElementById("dacCreateMethod").value = "GET";
        document.getElementById("dacCreateAuthType").value = "";
        document.querySelectorAll(".dac-item").forEach(el => el.classList.remove("active"));
    }

    function hideCreateForm() {
        $createView.style.display = "none";
        if (!selectedId) {
            $empty.style.display = "flex";
        }
    }

    async function createCall() {
        const name = val("dacCreateName");
        const url = val("dacCreateUrl");
        if (!name || !url) {
            toast("Name and URL are required", true);
            return;
        }

        const body = {
            name,
            url,
            method: val("dacCreateMethod") || "GET",
            description: val("dacCreateDescription"),
            input_payload: val("dacCreateInputPayload"),
            result_payload: val("dacCreateResultPayload"),
            headers: val("dacCreateHeaders") || null,
            auth_type: val("dacCreateAuthType") || null,
            auth_token: val("dacCreateAuthToken") || null,
            app_name: val("dacCreateAppName") || null,
            tags: val("dacCreateTags") || null,
        };

        try {
            const res = await fetch(API, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json();
                toast(err.error || "Create failed", true);
                return;
            }
            const created = await res.json();
            toast("Created");
            hideCreateForm();
            await loadCalls();
            showDetail(created.id);
        } catch (e) {
            toast("Network error", true);
        }
    }

    // ── Helpers ──────────────────────────────────────────────

    function field(label, valueHtml) {
        return `<div class="dac-field"><div class="dac-field-label">${label}</div>${valueHtml}</div>`;
    }

    function fieldInner(label, value) {
        return `<div class="dac-field-label">${label}</div><div class="dac-field-value">${value}</div>`;
    }

    function esc(s) {
        if (!s) return "";
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function attr(s) {
        return esc(s).replace(/'/g, "&#39;");
    }

    function val(id) {
        const el = document.getElementById(id);
        return el ? el.value.trim() : "";
    }

    function truncUrl(url) {
        if (!url) return "";
        try {
            const u = new URL(url);
            const path = u.pathname + u.search;
            if (path.length > 50) return u.host + path.slice(0, 47) + "...";
            return u.host + path;
        } catch (e) {
            return url.length > 60 ? url.slice(0, 57) + "..." : url;
        }
    }

    function maskToken(tok) {
        if (!tok) return "(not set)";
        if (tok.length <= 8) return "****";
        return tok.slice(0, 4) + "..." + tok.slice(-4);
    }

    function toast(msg, isError) {
        $toast.textContent = msg;
        $toast.style.borderColor = isError ? "var(--dac-danger)" : "var(--dac-success)";
        $toast.classList.add("show");
        setTimeout(() => $toast.classList.remove("show"), 2500);
    }
})();
