/**
 * Brain Graph — 3D force-directed visualization of Captain Claw's cognitive topology.
 *
 * Uses 3d-force-graph (Three.js + d3-force-3d) for rendering.
 * Connects via WebSocket for live node/edge streaming.
 */

/* global ForceGraph3D, THREE */

(function () {
    "use strict";

    // ── State ────────────────────────────────────────────────────

    let graph = null;
    let graphData = { nodes: [], links: [] };
    let ws = null;
    let selectedNode = null;
    const sessionMeshes = new Map();  // sessionNodeId → THREE.Mesh

    // Node type visibility — all on by default.
    const visibility = {
        insight: true, intuition: true, tension: true, task: true,
        briefing: true, todo: true, contact: true, session: true, event: true,
        message: true,
    };

    const COLORS = {
        insight:   "#FFD700", intuition: "#9B59B6", tension:   "#E74C3C",
        task:      "#3498DB", briefing:  "#2ECC71", todo:      "#1ABC9C",
        contact:   "#F39C12", session:   "#95A5A6", event:     "#00BCD4",
        message:   "#7CB9E8",
    };

    const LINK_STYLE = {
        spawned:    { color: "#FFD700", width: 1,   particles: 0 },
        supersedes: { color: "#E67E22", width: 1,   particles: 0 },
        resolves:   { color: "#2ECC71", width: 2,   particles: 2 },
        triggers:   { color: "#3498DB", width: 1.5, particles: 1 },
        parent:     { color: "#1ABC9C", width: 1,   particles: 0 },
        mentions:   { color: "#F39C12", width: 0.5, particles: 0 },
        contains:   { color: "#95A5A6", width: 0.3, particles: 0 },
        source:     { color: "#9B59B6", width: 1,   particles: 1 },
        sequence:   { color: "#555555", width: 0.5, particles: 0 },
    };

    // ── DOM refs ─────────────────────────────────────────────────

    const container = document.getElementById("graphContainer");
    const statsLabel = document.getElementById("statsLabel");
    const detailPanel = document.getElementById("detailPanel");
    const detailClose = document.getElementById("detailClose");
    const detailTitle = document.getElementById("detailTitle");
    const detailType = document.getElementById("detailType");
    const detailBody = document.getElementById("detailBody");
    const detailMeta = document.getElementById("detailMeta");
    const searchInput = document.getElementById("searchInput");
    const limitSlider = document.getElementById("limitSlider");
    const limitLabel = document.getElementById("limitLabel");
    const liveIndicator = document.getElementById("liveIndicator");
    const btnRefresh = document.getElementById("btnRefresh");
    const btnReset = document.getElementById("btnReset");

    // ── Init graph ───────────────────────────────────────────────

    function initGraph() {
        graph = ForceGraph3D()(container)
            .backgroundColor("#0a0a0f")
            .showNavInfo(false)
            // Nodes
            .nodeVal(n => (n.size || 3) * (n.size || 3))
            .nodeColor(n => {
                if (!visibility[n.type]) return "rgba(0,0,0,0)";
                if (searchQuery && !matchesSearch(n)) return "rgba(60,60,60,0.15)";
                return n.color || COLORS[n.type] || "#888";
            })
            .nodeOpacity(0.88)
            .nodeLabel(n => {
                if (!visibility[n.type]) return "";
                return `<div class="graph-tooltip">
                    <strong>${escHtml(n.label || "")}</strong><br>
                    <span style="color:${n.color}">${n.type}</span>
                    ${n.status ? ` &middot; ${n.status}` : ""}
                    ${n.importance ? ` &middot; imp:${n.importance}` : ""}
                </div>`;
            })
            .nodeThreeObject(n => {
                if (!visibility[n.type]) return new THREE.Object3D();  // invisible

                const size = n.size || 3;
                let geometry, material;

                switch (n.type) {
                    case "tension":
                        geometry = new THREE.IcosahedronGeometry(size * 0.7, 0);
                        break;
                    case "task":
                        geometry = new THREE.BoxGeometry(size, size, size);
                        break;
                    case "briefing":
                        geometry = new THREE.ConeGeometry(size * 0.5, size * 1.2, 6);
                        break;
                    case "todo":
                        geometry = new THREE.OctahedronGeometry(size * 0.6, 0);
                        break;
                    case "contact":
                        geometry = new THREE.DodecahedronGeometry(size * 0.6, 0);
                        break;
                    case "session":
                        // Transparent sphere — starts at unit radius, scaled dynamically
                        // to enclose child nodes (see onEngineTick).
                        geometry = new THREE.SphereGeometry(1, 24, 16);
                        material = new THREE.MeshPhongMaterial({
                            color: n.color,
                            transparent: true,
                            opacity: 0.06,
                            wireframe: true,
                        });
                        const mesh = new THREE.Mesh(geometry, material);
                        mesh.userData.sessionId = n.id;
                        sessionMeshes.set(n.id, mesh);
                        return mesh;
                    case "event":
                        geometry = new THREE.SphereGeometry(size * 0.4, 8, 6);
                        break;
                    case "message":
                        geometry = new THREE.TetrahedronGeometry(size * 0.5, 0);
                        break;
                    default:
                        // insight, intuition → sphere
                        geometry = new THREE.SphereGeometry(size * 0.5, 12, 8);
                }

                const dimmed = searchQuery && !matchesSearch(n);
                const col = dimmed ? 0x333333 : new THREE.Color(n.color || "#888");

                material = new THREE.MeshPhongMaterial({
                    color: col,
                    transparent: true,
                    opacity: dimmed ? 0.1 : (n.type === "event" ? 0.6 : 0.85),
                    emissive: col,
                    emissiveIntensity: n.type === "tension" ? 0.4 : 0.15,
                });

                return new THREE.Mesh(geometry, material);
            })
            .nodeThreeObjectExtend(false)
            // Links
            .linkColor(l => {
                const style = LINK_STYLE[l.type] || {};
                return style.color || "#444";
            })
            .linkWidth(l => {
                const style = LINK_STYLE[l.type] || {};
                return style.width || 0.5;
            })
            .linkOpacity(0.35)
            .linkDirectionalArrowLength(3)
            .linkDirectionalArrowRelPos(0.9)
            .linkDirectionalParticles(l => {
                const style = LINK_STYLE[l.type] || {};
                return style.particles || 0;
            })
            .linkDirectionalParticleWidth(1.5)
            .linkDirectionalParticleColor(l => {
                const style = LINK_STYLE[l.type] || {};
                return style.color || "#444";
            })
            // Interactions
            .onNodeClick(onNodeClick)
            .onBackgroundClick(() => closeDetail())
            // Force config
            .d3AlphaDecay(0.02)
            .d3VelocityDecay(0.3)
            .warmupTicks(80)
            .cooldownTicks(200);

        // Add ambient light for better visibility.
        const scene = graph.scene();
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
        dirLight.position.set(100, 200, 100);
        scene.add(dirLight);

        // Dynamically resize session spheres to enclose child nodes.
        graph.onEngineTick(() => {
            for (const [sessionId, mesh] of sessionMeshes) {
                const sessionNode = graphData.nodes.find(n => n.id === sessionId);
                if (!sessionNode) continue;
                const sx = sessionNode.x || 0;
                const sy = sessionNode.y || 0;
                const sz = sessionNode.z || 0;

                // Find max distance from session center to any child node.
                let maxDist = 0;
                for (const n of graphData.nodes) {
                    if (n.group !== sessionId || n.id === sessionId) continue;
                    const dx = (n.x || 0) - sx;
                    const dy = (n.y || 0) - sy;
                    const dz = (n.z || 0) - sz;
                    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist > maxDist) maxDist = dist;
                }

                // Radius = furthest child + padding (20% extra).
                const radius = Math.max(maxDist * 1.2, 15);
                mesh.scale.set(radius, radius, radius);
            }
        });
    }

    // ── Data fetching ────────────────────────────────────────────

    async function loadGraphData() {
        const limit = limitSlider ? limitSlider.value : 100;
        statsLabel.textContent = "Loading...";

        try {
            const resp = await fetch(`/api/brain-graph?limit=${limit}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            graphData = { nodes: data.nodes || [], links: data.links || [] };

            if (graph) {
                graph.graphData(graphData);
            }

            const s = data.stats || {};
            statsLabel.textContent =
                `${s.total_nodes || 0} nodes \u00B7 ${s.total_links || 0} edges`;

            // Auto-focus on a node if ?focus_ts= or ?focus_id= is in the URL.
            autoFocusFromURL();
        } catch (err) {
            console.error("Brain graph load failed:", err);
            statsLabel.textContent = "Load failed";
        }
    }

    function autoFocusFromURL() {
        const params = new URLSearchParams(window.location.search);

        // Focus by node ID.
        const focusId = params.get("focus_id");
        if (focusId) {
            const node = graphData.nodes.find(n => n.id === focusId);
            if (node) {
                setTimeout(() => showNodeDetail(node, true), 800);
                return;
            }
        }

        // Focus by timestamp (from chat/computer brain button).
        const focusTs = params.get("focus_ts");
        if (focusTs) {
            // Find message node whose created_at most closely matches.
            let bestNode = null;
            let bestDist = Infinity;
            for (const n of graphData.nodes) {
                if (!n.created_at) continue;
                // Compare by string proximity (ISO timestamps sort lexicographically).
                const dist = Math.abs(new Date(n.created_at) - new Date(focusTs));
                if (dist < bestDist) {
                    bestDist = dist;
                    bestNode = n;
                }
            }
            // Only auto-focus if we found a close match (within 2 seconds).
            if (bestNode && bestDist < 2000) {
                setTimeout(() => showNodeDetail(bestNode, true), 800);
                return;
            }
            // Fallback: if no close timestamp match, try matching message nodes.
            const msgNodes = graphData.nodes.filter(n => n.type === "message");
            if (msgNodes.length) {
                let best = null;
                let bd = Infinity;
                for (const n of msgNodes) {
                    if (!n.created_at) continue;
                    const d = Math.abs(new Date(n.created_at) - new Date(focusTs));
                    if (d < bd) { bd = d; best = n; }
                }
                if (best && bd < 10000) {
                    setTimeout(() => showNodeDetail(best, true), 800);
                }
            }
        }
    }

    // ── Search ───────────────────────────────────────────────────

    let searchQuery = "";

    function matchesSearch(node) {
        if (!searchQuery) return true;
        const q = searchQuery.toLowerCase();
        return (node.label || "").toLowerCase().includes(q)
            || (node.type || "").toLowerCase().includes(q)
            || (node.status || "").toLowerCase().includes(q);
    }

    // ── Node connections + navigation ──────────────────────────────

    /**
     * For the selected node, find all connected nodes via edges.
     * Returns { incoming: [{node, link}], outgoing: [{node, link}] }
     */
    function getConnections(node) {
        const incoming = [];  // edges pointing TO this node
        const outgoing = [];  // edges pointing FROM this node

        for (const link of graphData.links) {
            // 3d-force-graph may replace source/target strings with node objects
            const srcId = typeof link.source === "object" ? link.source.id : link.source;
            const tgtId = typeof link.target === "object" ? link.target.id : link.target;

            if (tgtId === node.id) {
                const srcNode = graphData.nodes.find(n => n.id === srcId);
                if (srcNode) incoming.push({ node: srcNode, link });
            }
            if (srcId === node.id) {
                const tgtNode = graphData.nodes.find(n => n.id === tgtId);
                if (tgtNode) outgoing.push({ node: tgtNode, link });
            }
        }

        return { incoming, outgoing };
    }

    /**
     * Build a flat navigation chain: incoming → current → outgoing.
     * For walking through the "process of thoughts".
     */
    let navChain = [];
    let navIndex = -1;

    function buildNavChain(node) {
        const { incoming, outgoing } = getConnections(node);

        // Sort by created_at to get chronological order.
        const sortByTime = (a, b) => (a.node.created_at || "").localeCompare(b.node.created_at || "");
        incoming.sort(sortByTime);
        outgoing.sort(sortByTime);

        // Chain: all incoming nodes → current → all outgoing nodes
        navChain = [
            ...incoming.map(c => c.node),
            node,
            ...outgoing.map(c => c.node),
        ];

        // Deduplicate (a node could appear in both incoming and outgoing).
        const seen = new Set();
        navChain = navChain.filter(n => {
            if (seen.has(n.id)) return false;
            seen.add(n.id);
            return true;
        });

        navIndex = navChain.findIndex(n => n.id === node.id);
    }

    function updateNavButtons() {
        const btnPrev = document.getElementById("btnPrev");
        const btnNext = document.getElementById("btnNext");
        const navPos = document.getElementById("navPos");

        btnPrev.disabled = navIndex <= 0;
        btnNext.disabled = navIndex >= navChain.length - 1;
        navPos.textContent = navChain.length > 1
            ? `${navIndex + 1} / ${navChain.length}`
            : "no connections";
    }

    // ── Node click → detail panel ────────────────────────────────

    function onNodeClick(node) {
        if (!node || !visibility[node.type]) return;
        showNodeDetail(node, true);
    }

    function showNodeDetail(node, rebuildNav) {
        selectedNode = node;

        if (rebuildNav) {
            buildNavChain(node);
        }
        updateNavButtons();

        const meta = node.meta || {};
        const previewContent = meta.content || meta.summary || meta.result_summary || node.label || "";

        detailTitle.textContent = node.type + ": " + (node.label || node.id).substring(0, 60);
        detailType.textContent = node.type;
        detailType.style.background = node.color || "#888";
        detailType.style.color = "#000";

        // Body: show preview + "Show full content" button for messages
        detailBody.textContent = previewContent.substring(0, 200);
        if (previewContent.length > 200) {
            detailBody.textContent += "...";
        }

        // Add "Show full content" button — loads from API for messages, or shows inline for others
        const showBtn = document.createElement("button");
        showBtn.className = "bg-show-full-btn";
        showBtn.textContent = "Show full content";
        showBtn.addEventListener("click", () => openContentModal(node));
        detailBody.appendChild(document.createElement("br"));
        detailBody.appendChild(showBtn);

        // Connections list
        const { incoming, outgoing } = getConnections(node);
        const detailConnections = document.getElementById("detailConnections");
        let connHtml = "";

        if (incoming.length || outgoing.length) {
            connHtml += "<h4>Connections</h4>";
            for (const c of incoming) {
                connHtml += `<div class="bg-conn-item" data-node-id="${c.node.id}">
                    <span class="bg-conn-dot" style="background:${c.node.color || '#888'}"></span>
                    <span class="bg-conn-arrow">&rarr;</span>
                    <span class="bg-conn-label">${escHtml((c.node.label || c.node.id).substring(0, 50))}</span>
                    <span style="color:#555;font-size:10px;margin-left:auto">${c.link.type || ""}</span>
                </div>`;
            }
            for (const c of outgoing) {
                connHtml += `<div class="bg-conn-item" data-node-id="${c.node.id}">
                    <span class="bg-conn-arrow">&larr;</span>
                    <span class="bg-conn-dot" style="background:${c.node.color || '#888'}"></span>
                    <span class="bg-conn-label">${escHtml((c.node.label || c.node.id).substring(0, 50))}</span>
                    <span style="color:#555;font-size:10px;margin-left:auto">${c.link.type || ""}</span>
                </div>`;
            }
        }
        detailConnections.innerHTML = connHtml;

        // Make connection items clickable.
        detailConnections.querySelectorAll(".bg-conn-item").forEach(el => {
            el.addEventListener("click", () => {
                const targetId = el.dataset.nodeId;
                const targetNode = graphData.nodes.find(n => n.id === targetId);
                if (targetNode) showNodeDetail(targetNode, true);
            });
        });

        // Meta fields
        const metaHtml = [];
        metaHtml.push(`<span><span class="meta-label">ID</span> ${node.id}</span>`);
        if (node.importance) metaHtml.push(`<span><span class="meta-label">Importance</span> ${node.importance}</span>`);
        if (node.confidence != null) metaHtml.push(`<span><span class="meta-label">Confidence</span> ${node.confidence}</span>`);
        if (node.status) metaHtml.push(`<span><span class="meta-label">Status</span> ${node.status}</span>`);
        if (node.created_at) metaHtml.push(`<span><span class="meta-label">Created</span> ${node.created_at}</span>`);
        if (node.group) metaHtml.push(`<span><span class="meta-label">Session</span> ${node.group}</span>`);

        // Extra meta fields
        for (const [k, v] of Object.entries(meta)) {
            if (v != null && k !== "content" && k !== "summary" && k !== "result_summary") {
                metaHtml.push(`<span><span class="meta-label">${k}</span> ${v}</span>`);
            }
        }

        detailMeta.innerHTML = metaHtml.join("");
        detailPanel.classList.add("open");

        // Focus camera on node.
        const dist = 120;
        graph.cameraPosition(
            { x: node.x + dist, y: node.y + dist * 0.5, z: node.z + dist },
            node,
            1000
        );
    }

    function navigatePrev() {
        if (navIndex > 0) {
            navIndex--;
            showNodeDetail(navChain[navIndex], false);
        }
    }

    function navigateNext() {
        if (navIndex < navChain.length - 1) {
            navIndex++;
            showNodeDetail(navChain[navIndex], false);
        }
    }

    function closeDetail() {
        detailPanel.classList.remove("open");
        selectedNode = null;
        navChain = [];
        navIndex = -1;
    }

    // ── WebSocket (live updates) ─────────────────────────────────

    function connectWS() {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${proto}//${location.host}/ws`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            liveIndicator.classList.add("connected");
        };

        ws.onclose = () => {
            liveIndicator.classList.remove("connected");
            // Reconnect after 3s.
            setTimeout(connectWS, 3000);
        };

        ws.onerror = () => {
            liveIndicator.classList.remove("connected");
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "brain_graph_update") {
                    handleGraphUpdate(msg);
                }
            } catch (e) {
                // Not JSON or not for us — ignore.
            }
        };
    }

    function handleGraphUpdate(msg) {
        const action = msg.action;

        if (action === "add_node" && msg.node) {
            // Check if node already exists.
            const existing = graphData.nodes.find(n => n.id === msg.node.id);
            if (!existing) {
                graphData.nodes.push(msg.node);
            }
        }

        if (action === "add_link" || (msg.links && msg.links.length)) {
            const newLinks = msg.links || [];
            for (const lnk of newLinks) {
                // Avoid duplicate links.
                const exists = graphData.links.find(
                    l => l.source === lnk.source && l.target === lnk.target && l.type === lnk.type
                );
                if (!exists) {
                    graphData.links.push(lnk);
                }
            }
        }

        if (action === "update_node" && msg.node) {
            const idx = graphData.nodes.findIndex(n => n.id === msg.node.id);
            if (idx >= 0) {
                Object.assign(graphData.nodes[idx], msg.node);
            }
        }

        // Re-render.
        if (graph) {
            graph.graphData(graphData);
        }

        // Update stats.
        statsLabel.textContent =
            `${graphData.nodes.length} nodes \u00B7 ${graphData.links.length} edges`;
    }

    // ── Event handlers ───────────────────────────────────────────

    function bindEvents() {
        // Filter checkboxes.
        document.querySelectorAll("[data-type]").forEach(cb => {
            cb.addEventListener("change", () => {
                visibility[cb.dataset.type] = cb.checked;
                if (graph) graph.nodeColor(graph.nodeColor());  // force re-render
                if (graph) graph.refresh();
            });
        });

        // Search.
        searchInput.addEventListener("input", () => {
            searchQuery = searchInput.value.trim();
            if (graph) graph.nodeColor(graph.nodeColor());
            if (graph) graph.refresh();
        });

        // Limit slider.
        limitSlider.addEventListener("input", () => {
            limitLabel.textContent = limitSlider.value;
        });
        limitSlider.addEventListener("change", () => {
            loadGraphData();
        });

        // Refresh button.
        btnRefresh.addEventListener("click", loadGraphData);

        // Reset view button.
        btnReset.addEventListener("click", () => {
            if (graph) {
                graph.cameraPosition({ x: 0, y: 0, z: 500 }, { x: 0, y: 0, z: 0 }, 1000);
            }
        });

        // Detail close.
        detailClose.addEventListener("click", closeDetail);

        // Prev / Next navigation.
        document.getElementById("btnPrev").addEventListener("click", navigatePrev);
        document.getElementById("btnNext").addEventListener("click", navigateNext);

        // Keyboard navigation.
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                if (modalOverlay.classList.contains("open")) {
                    closeModal();
                } else {
                    closeDetail();
                }
            }
            if (detailPanel.classList.contains("open") && !modalOverlay.classList.contains("open")) {
                if (e.key === "ArrowLeft" || e.key === "[") navigatePrev();
                if (e.key === "ArrowRight" || e.key === "]") navigateNext();
            }
        });

        // Modal close.
        document.getElementById("modalClose").addEventListener("click", closeModal);
        modalOverlay.addEventListener("click", (e) => {
            if (e.target === modalOverlay) closeModal();
        });
    }

    // ── Content Modal ────────────────────────────────────────────

    const modalOverlay = document.getElementById("modalOverlay");
    const modalTitle = document.getElementById("modalTitle");
    const modalBody = document.getElementById("modalBody");

    function renderMd(text) {
        if (typeof marked !== "undefined" && marked.parse) {
            return marked.parse(text);
        }
        // Fallback: basic escaping + newlines
        return escHtml(text).replace(/\n/g, "<br>");
    }

    async function openContentModal(node) {
        const meta = node.meta || {};

        // For message nodes — fetch full content from API.
        if (node.type === "message") {
            const rawId = node.id.replace(/^msg_/, "");
            modalTitle.textContent = "Loading...";
            modalBody.innerHTML = "<em>Fetching full message...</em>";
            modalOverlay.classList.add("open");

            try {
                const resp = await fetch(`/api/brain-graph/message/${encodeURIComponent(rawId)}`);
                if (resp.ok) {
                    const data = await resp.json();
                    modalTitle.textContent = `${data.role || "message"} — ${data.session_name || ""}`;
                    modalBody.innerHTML = renderMd(data.content || "(empty)");
                } else {
                    modalBody.innerHTML = "<em>Message not found. It may belong to a different session.</em>";
                    modalTitle.textContent = "Not found";
                }
            } catch (err) {
                modalBody.innerHTML = `<em>Failed to load: ${escHtml(err.message)}</em>`;
                modalTitle.textContent = "Error";
            }
            return;
        }

        // For other node types — show meta.content inline (already available).
        const content = meta.content || meta.summary || meta.result_summary || node.label || "";
        modalTitle.textContent = `${node.type}: ${(node.label || "").substring(0, 50)}`;
        modalBody.innerHTML = renderMd(content);
        modalOverlay.classList.add("open");
    }

    function closeModal() {
        modalOverlay.classList.remove("open");
    }

    // ── Helpers ───────────────────────────────────────────────────

    function escHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Boot ─────────────────────────────────────────────────────

    initGraph();
    bindEvents();
    loadGraphData();
    connectWS();

})();
