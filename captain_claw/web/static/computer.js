/* ═══════════════════════════════════════════════════════════════
   Computer — Retro-themed workspace chat UI
   Connects to Captain Claw's existing WebSocket protocol.
   ═══════════════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ── State ───────────────────────────────────────────────────── */

let ws = null;
let connected = false;
let currentTheme = "amiga";
let isProcessing = false;
let ledTimer = null;

// Store last prompt/result for Visual tab re-generation.
let lastPrompt = "";
let lastResult = "";
let visualGenerating = false;

// Track HTML files created by agent during the current turn.
let agentCreatedHtmlFiles = [];

// Visual generation settings (persisted in localStorage).
let selectedTokenTier = localStorage.getItem("computer-token-tier") || "standard";
let selectedModel = localStorage.getItem("computer-model") || "";
let availableModels = [];

// Attachment state.
let pendingImagePath = null;
let pendingFilePath = null;
const _IMAGE_EXTS = ["png", "jpg", "jpeg", "webp", "gif", "bmp"];

// Folder browser state.
let folderCurrentPath = null;
let _gwsAvailable = false;
let _gdriveCurrent = { id: "root", name: "My Drive" };
let _gdriveBcStack = [{ id: "root", name: "My Drive" }];

/* ── Exploration state ───────────────────────────────────────── */

const MAX_EXPLORATION_NODES = 200;

let explorationNodes = new Map();  // id -> ExplorationNode
let currentNodeId = null;
let pendingExploration = null;     // { parentId, edgeLabel, source }
let sessionId = null;              // set on welcome
let exploreDebounceTimer = null;
let historyBranchNodeId = null;    // set when user navigates to an old node

// postMessage bridge script injected into visual iframe HTML.
const EXPLORE_BRIDGE = `<script>
document.addEventListener('DOMContentLoaded', function() {
  // Wire up explore-link elements.
  document.querySelectorAll('.explore-link').forEach(function(el) {
    el.style.cursor = 'pointer';
    el.addEventListener('click', function(e) {
      e.preventDefault();
      window.parent.postMessage({
        type: 'explore-click',
        topic: el.dataset.topic || el.textContent.trim(),
        context: el.dataset.context || ''
      }, '*');
    });
  });

  // Selection-to-explore: show a floating button when user selects text.
  var exploreFab = document.createElement('div');
  exploreFab.id = 'explore-fab';
  exploreFab.innerHTML = '\\u{1F50D} Explore this';
  exploreFab.style.cssText = 'display:none;position:fixed;z-index:99999;' +
    'padding:5px 12px;border-radius:14px;font-size:12px;font-weight:700;' +
    'cursor:pointer;pointer-events:auto;user-select:none;' +
    'box-shadow:0 2px 8px rgba(0,0,0,0.3);transition:opacity 0.15s,transform 0.15s;' +
    'opacity:0;transform:translateY(4px);';
  document.body.appendChild(exploreFab);

  // High-contrast styling that works across all themes.
  // Use a dark background with white text — guaranteed readable.
  exploreFab.style.background = '#1a1a1a';
  exploreFab.style.color = '#f0f0f0';
  exploreFab.style.border = '1px solid #555';

  // Try to pick up the page accent for a subtle highlight.
  var cs = getComputedStyle(document.body);
  var accent = (cs.getPropertyValue('--accent') || '').trim();
  if (accent) {
    exploreFab.style.borderColor = accent;
    exploreFab.style.boxShadow = '0 2px 10px rgba(0,0,0,0.4), 0 0 0 1px ' + accent;
  }

  var fabTimeout = null;
  var selectedText = '';

  function showFab(x, y, text) {
    selectedText = text;
    exploreFab.style.left = Math.min(x, window.innerWidth - 140) + 'px';
    exploreFab.style.top = Math.max(y - 36, 4) + 'px';
    exploreFab.style.display = 'block';
    requestAnimationFrame(function() {
      exploreFab.style.opacity = '1';
      exploreFab.style.transform = 'translateY(0)';
    });
    clearTimeout(fabTimeout);
    fabTimeout = setTimeout(hideFab, 6000);
  }

  function hideFab() {
    exploreFab.style.opacity = '0';
    exploreFab.style.transform = 'translateY(4px)';
    setTimeout(function() { exploreFab.style.display = 'none'; }, 150);
    clearTimeout(fabTimeout);
  }

  document.addEventListener('mouseup', function(e) {
    if (e.target === exploreFab || exploreFab.contains(e.target)) return;
    var sel = window.getSelection();
    var text = (sel ? sel.toString() : '').trim();
    if (text.length >= 3 && text.length <= 300) {
      var range = sel.getRangeAt(0);
      var rect = range.getBoundingClientRect();
      showFab(rect.left + rect.width / 2 - 50, rect.top, text);
    } else {
      setTimeout(hideFab, 100);
    }
  });

  exploreFab.addEventListener('click', function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (selectedText) {
      window.parent.postMessage({
        type: 'explore-click',
        topic: selectedText,
        context: ''
      }, '*');
      hideFab();
      window.getSelection().removeAllRanges();
    }
  });

  // Hide on scroll.
  document.addEventListener('scroll', hideFab, true);
});
<\/script>`;

function uuid4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function createExplorationNode(prompt, answer) {
  const node = {
    id: uuid4(),
    session_id: sessionId || '',
    parent_id: pendingExploration ? pendingExploration.parentId : null,
    edge_label: pendingExploration ? pendingExploration.edgeLabel : null,
    prompt: prompt,
    answer: answer,
    visual_html: null,
    theme: currentTheme,
    source: pendingExploration ? 'click' : 'manual',
    created_at: new Date().toISOString(),
    metadata: JSON.stringify(pendingExploration || {}),
  };

  explorationNodes.set(node.id, node);

  // Cap exploration tree.
  if (explorationNodes.size > MAX_EXPLORATION_NODES) {
    const oldest = explorationNodes.keys().next().value;
    explorationNodes.delete(oldest);
    logEntry("system", `Exploration tree capped at ${MAX_EXPLORATION_NODES} nodes`);
  }

  return node;
}

function injectExploreBridge(html) {
  // Inject the postMessage bridge before </body> or at end.
  if (html.includes('</body>')) {
    return html.replace('</body>', EXPLORE_BRIDGE + '</body>');
  }
  return html + EXPLORE_BRIDGE;
}

/* ── Exploration: save/load from backend ─────────────────────── */

async function saveExplorationNode(node) {
  try {
    await fetch('/api/computer/exploration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(node),
    });
  } catch (e) {
    console.warn('Failed to save exploration node:', e);
  }
}

async function updateExplorationNodeVisual(nodeId, visualHtml) {
  try {
    await fetch(`/api/computer/exploration/${nodeId}/visual`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ visual_html: visualHtml }),
    });
  } catch (e) {
    console.warn('Failed to update exploration node visual:', e);
  }
}

async function loadExplorationHistory(sid) {
  try {
    const res = await fetch(`/api/computer/exploration?session_id=${encodeURIComponent(sid)}`);
    if (!res.ok) return;
    const data = await res.json();
    const nodes = data.nodes || [];
    explorationNodes.clear();
    for (const n of nodes) {
      explorationNodes.set(n.id, n);
    }
    // Set current to most recent.
    if (nodes.length > 0) {
      currentNodeId = nodes[nodes.length - 1].id;
    }
    console.log(`[Computer] Loaded ${nodes.length} exploration nodes`);
    if (nodes.length > 0) {
      logEntry("system", `Loaded ${nodes.length} exploration nodes`);
    }
    renderHistoryDrawer();
  } catch (e) {
    console.warn('Failed to load exploration history:', e);
  }
}

/* ── History drawer ──────────────────────────────────────────── */

function renderHistoryDrawer() {
  const nodes = Array.from(explorationNodes.values())
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

  const isBranching = !!historyBranchNodeId;

  for (const tab of ['answer', 'blueprint']) {
    const drawer = $(`#${tab}-history`);
    if (!drawer) continue;
    const countEl = drawer.querySelector('.history-bar-count');
    const listEl = drawer.querySelector('.history-list');
    const labelEl = drawer.querySelector('.history-bar-label');
    countEl.textContent = nodes.length;

    // Show branch indicator in the bar.
    if (isBranching) {
      drawer.classList.add('branching');
      labelEl.innerHTML = `<span class="history-branch-badge">&#9095; fork</span> History (<span class="history-bar-count">${nodes.length}</span>)`;
    } else {
      drawer.classList.remove('branching');
      labelEl.innerHTML = `History (<span class="history-bar-count">${nodes.length}</span>)`;
    }

    if (nodes.length === 0) {
      drawer.style.display = 'none';
      continue;
    }
    drawer.style.display = '';

    listEl.innerHTML = nodes.map(n => {
      const t = n.created_at ? new Date(n.created_at) : null;
      const time = t ? t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
      const prompt = (n.prompt || '').slice(0, 55) + ((n.prompt || '').length > 55 ? '\u2026' : '');
      const active = n.id === currentNodeId ? ' active' : '';
      // Dim entries that will be pruned on next send (newer than branch point).
      let pruned = '';
      if (isBranching && historyBranchNodeId) {
        const branchNode = explorationNodes.get(historyBranchNodeId);
        if (branchNode && (n.created_at || '') > (branchNode.created_at || '')) {
          pruned = ' pruned';
        }
      }
      return `<div class="history-entry${active}${pruned}" data-node-id="${n.id}" onclick="historySelect('${n.id}')">
        <span class="history-entry-bullet">${n.id === currentNodeId ? '\u25CF' : '\u25CB'}</span>
        <span class="history-entry-time">${esc(time)}</span>
        <span class="history-entry-prompt">${esc(prompt)}</span>
      </div>`;
    }).join('');
  }
}

function toggleHistoryDrawer(tab) {
  const drawer = $(`#${tab}-history`);
  if (!drawer) return;
  drawer.classList.toggle('open');
}

function historySelect(nodeId) {
  navigateToNode(nodeId);
  renderHistoryDrawer();
}

function historyPrev(tab) {
  const nodes = Array.from(explorationNodes.values())
    .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
  if (nodes.length === 0) return;
  const idx = nodes.findIndex(n => n.id === currentNodeId);
  if (idx > 0) {
    historySelect(nodes[idx - 1].id);
  }
}

function historyNext(tab) {
  const nodes = Array.from(explorationNodes.values())
    .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
  if (nodes.length === 0) return;
  const idx = nodes.findIndex(n => n.id === currentNodeId);
  if (idx < nodes.length - 1) {
    historySelect(nodes[idx + 1].id);
  }
}

/* ── Exploration: click-to-prompt ────────────────────────────── */

function handleIframeMessage(event) {
  if (!event.data || event.data.type !== 'explore-click') return;
  const { topic, context } = event.data;
  if (!topic) return;

  // Debounce rapid clicks.
  if (exploreDebounceTimer) return;
  exploreDebounceTimer = setTimeout(() => { exploreDebounceTimer = null; }, 500);

  console.log('%c[Computer] Explore click', 'color:#ff8800;font-weight:bold', { topic, context });
  showExploreConfirm(topic, context);
}

function showExploreConfirm(topic, context) {
  if (!connected) {
    logEntry("error", "Not connected — cannot explore");
    return;
  }

  const basePrompt = context
    ? `Explore: ${topic} — ${context}`
    : `Tell me more about: ${topic}`;

  // Put the prepared prompt in the input box for the user to see/edit.
  const input = $("#input-box");
  input.value = basePrompt;
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";

  // Show the explore confirmation bar.
  let bar = $("#explore-confirm-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "explore-confirm-bar";
    bar.innerHTML = `
      <div class="explore-confirm-topic"></div>
      <div class="explore-confirm-hint">Add context or edit the prompt above, then send — or cancel.</div>
      <div class="explore-confirm-actions">
        <button id="explore-send-btn" class="tab-btn" title="Send exploration prompt">▶ Explore</button>
        <button id="explore-cancel-btn" class="tab-btn" title="Cancel exploration">✕ Cancel</button>
      </div>
    `;
    // Insert after the input-actions div.
    const inputActions = $("#input-actions");
    inputActions.parentNode.insertBefore(bar, inputActions.nextSibling);

    // Wire buttons.
    bar.querySelector("#explore-send-btn").addEventListener("click", () => {
      confirmExploration();
    });
    bar.querySelector("#explore-cancel-btn").addEventListener("click", () => {
      cancelExploration();
    });
  }

  // Store exploration metadata on the bar for later use.
  bar._exploreTopic = topic;
  bar._exploreContext = context;

  bar.querySelector(".explore-confirm-topic").textContent = `🔗 ${topic}`;
  bar.classList.add("active");
  bar.style.display = "flex";

  // Hide the Send button — Explore button replaces it.
  $("#send-btn").style.display = "none";

  // Focus the input so the user can immediately type additions.
  input.focus();
  // Place cursor at end.
  input.setSelectionRange(input.value.length, input.value.length);

  logEntry("system", `Explore: "${topic}" — edit prompt or press Explore to send`);
}

function confirmExploration() {
  const bar = $("#explore-confirm-bar");
  if (!bar) return;

  const topic = bar._exploreTopic || "";

  // Store the parent linkage before sending.
  pendingExploration = {
    parentId: currentNodeId,
    edgeLabel: topic,
    source: 'click',
  };

  // Hide the bar and restore Send button.
  bar.classList.remove("active");
  bar.style.display = "none";
  $("#send-btn").style.display = "";

  // Send whatever is in the input box (user may have edited it).
  handleSend();
}

function cancelExploration() {
  const bar = $("#explore-confirm-bar");
  if (bar) {
    bar.classList.remove("active");
    bar.style.display = "none";
  }
  $("#send-btn").style.display = "";
  $("#input-box").value = "";
  logEntry("system", "Exploration cancelled");
}

/* ── Exploration: navigate to historical node ────────────────── */

function navigateToNode(nodeId) {
  const node = explorationNodes.get(nodeId);
  if (!node) return;

  currentNodeId = nodeId;
  lastPrompt = node.prompt;
  lastResult = node.answer;

  // Detect if this is a branch point (navigating to a non-latest node).
  const sorted = Array.from(explorationNodes.values())
    .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
  const latestId = sorted.length > 0 ? sorted[sorted.length - 1].id : null;
  historyBranchNodeId = (nodeId !== latestId) ? nodeId : null;

  // Render answer without creating a new node.
  const el = $("#answer-content");
  el.innerHTML = markdownToHtml(node.answer);
  el.classList.remove("output-empty");

  // Restore visual if available.
  const frame = $("#visual-frame");
  const visualEl = $("#visual-content");
  if (node.visual_html) {
    visualEl.innerHTML = "";
    visualEl.classList.remove("output-empty");
    frame.srcdoc = injectExploreBridge(node.visual_html);
    frame.style.display = "block";
  } else {
    clearVisual();
  }

  // Rebuild blueprint.
  generateBlueprint(node.answer);

  // Update map highlighting & history drawer.
  renderMap();
  renderHistoryDrawer();

  if (historyBranchNodeId) {
    logEntry("system", `Branched to: ${node.prompt.slice(0, 50)}${node.prompt.length > 50 ? '…' : ''} — next message will fork from here`);
  } else {
    logEntry("system", `Navigated to: ${node.prompt.slice(0, 60)}${node.prompt.length > 60 ? '…' : ''}`);
  }
}

/* ── Boot sequences per theme ────────────────────────────────── */

const BOOT_SEQUENCES = {
  amiga: [
    { html: '<div class="boot-amiga-hand">💾</div><div class="boot-text">Insert Workbench disk</div>', delay: 1500 },
    { html: '<div class="boot-amiga-hand">💾</div><div class="boot-text">Reading disk...</div>', delay: 1200 },
    { html: '<div class="boot-amiga-hand">💾</div><div class="boot-text">Loading Kickstart...</div>', delay: 1400 },
    { html: '<div class="boot-amiga-check">✓</div><div class="boot-text">Kickstart 1.3 loaded</div>', delay: 1000 },
    { html: '<div class="boot-amiga-check">✓</div><div class="boot-text">Amiga Workbench 1.3</div>', delay: 800 },
  ],
  atarist: [
    { html: '<div style="font-size:32px">⚡</div><div class="boot-text">Atari ST — TOS 1.04</div>', delay: 1200 },
    { html: '<div style="font-size:32px">⚡</div><div class="boot-text">Memory test... 512KB OK</div>', delay: 1000 },
    { html: '<div style="font-size:32px">⚡</div><div class="boot-text">GEM Desktop loading...</div>', delay: 1200 },
  ],
  c64: [
    { html: '<div style="font-size:13px;text-align:left;font-family:Silkscreen,monospace;line-height:1.8">**** COMMODORE 64 GEOS V2.0 ****</div>', delay: 1200 },
    { html: '<div style="font-size:13px;text-align:left;font-family:Silkscreen,monospace;line-height:1.8">**** COMMODORE 64 GEOS V2.0 ****<br><br>64K RAM SYSTEM  38911 BASIC BYTES FREE</div>', delay: 1000 },
    { html: '<div style="font-size:13px;text-align:left;font-family:Silkscreen,monospace;line-height:1.8">**** COMMODORE 64 GEOS V2.0 ****<br><br>64K RAM SYSTEM  38911 BASIC BYTES FREE<br><br>LOADING DESKTOP...</div>', delay: 1400 },
  ],
  mac: [
    { html: '<div style="font-size:48px">🙂</div><div class="boot-text">Welcome to Macintosh</div>', delay: 1500 },
    { html: '<div style="font-size:48px">🙂</div><div class="boot-text">Loading System 7...</div>', delay: 1200 },
    { html: '<div style="font-size:48px">🙂</div><div class="boot-text">Loading extensions...</div>', delay: 1000 },
  ],
  win31: [
    { html: '<div style="font-size:13px;text-align:left;font-family:monospace;line-height:1.8">Microsoft(R) Windows(TM)<br>Version 3.1<br><br>Copyright (C) Microsoft Corp. 1985-1992</div>', delay: 1400 },
    { html: '<div style="font-size:13px;text-align:left;font-family:monospace;line-height:1.8">Microsoft(R) Windows(TM)<br>Version 3.1<br><br>Copyright (C) Microsoft Corp. 1985-1992<br><br>Loading drivers...</div>', delay: 1200 },
    { html: '<div style="font-size:13px;text-align:left;font-family:monospace;line-height:1.8">Microsoft(R) Windows(TM)<br>Version 3.1<br><br>Copyright (C) Microsoft Corp. 1985-1992<br><br>Loading Program Manager...</div>', delay: 1000 },
  ],
  hacker: [
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...</div>', delay: 1800 },
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...<br><br>The Matrix has you...</div>', delay: 1600 },
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...<br><br>The Matrix has you...<br><br>Follow the white rabbit.</div>', delay: 1400 },
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...<br><br>The Matrix has you...<br><br>Follow the white rabbit.<br><br>Knock, knock, Neo.</div>', delay: 1200 },
    { html: '<div style="font-size:14px;font-family:monospace;text-shadow:0 0 10px #00ff41">root@computer:~# <span style="animation:boot-blink 0.7s step-end infinite">_</span></div>', delay: 800 },
  ],
  modern: [
    { html: '<div style="font-size:28px;font-weight:300;letter-spacing:2px;opacity:0.3">Computer</div>', delay: 800 },
    { html: '<div style="font-size:28px;font-weight:300;letter-spacing:2px;opacity:0.7">Computer</div>', delay: 600 },
    { html: '<div style="font-size:28px;font-weight:300;letter-spacing:2px">Computer</div><div style="margin-top:16px;width:120px;height:3px;border-radius:2px;background:#2a2e3d;overflow:hidden"><div style="width:0%;height:100%;background:#6c8cff;border-radius:2px;animation:modern-load 1.2s ease-out forwards"></div></div>', delay: 1400 },
  ],
  win11: [
    { html: '<div style="font-size:32px">⊞</div>', delay: 800 },
    { html: '<div style="font-size:32px">⊞</div><div style="margin-top:12px;font-family:Segoe UI,sans-serif;font-size:14px;color:#1a1a1a">Windows 11</div>', delay: 1000 },
    { html: '<div style="font-size:32px">⊞</div><div style="margin-top:12px;font-family:Segoe UI,sans-serif;font-size:14px;color:#1a1a1a">Windows 11</div><div style="margin-top:16px;width:100px;height:3px;border-radius:2px;background:#e5e5e5;overflow:hidden"><div style="width:0%;height:100%;background:#0078d4;border-radius:2px;animation:modern-load 1.2s ease-out forwards"></div></div>', delay: 1400 },
  ],
  macos: [
    { html: '<div style="font-size:48px"></div>', delay: 800 },
    { html: '<div style="font-size:48px"></div><div style="margin-top:12px;font-family:-apple-system,sans-serif;font-size:14px;color:#e0e0e0">macOS</div>', delay: 1000 },
    { html: '<div style="font-size:48px"></div><div style="margin-top:12px;font-family:-apple-system,sans-serif;font-size:14px;color:#e0e0e0">macOS</div><div style="margin-top:16px;width:120px;height:4px;border-radius:4px;background:#3d3d3d;overflow:hidden"><div style="width:0%;height:100%;background:#fff;border-radius:4px;animation:modern-load 1.2s ease-out forwards"></div></div>', delay: 1400 },
  ],
  iphone: [
    { html: '<div style="font-size:48px;opacity:0.4"></div>', delay: 800 },
    { html: '<div style="font-size:48px"></div>', delay: 700 },
    { html: '<div style="font-size:48px"></div><div style="margin-top:16px;font-family:-apple-system,sans-serif;font-size:24px;font-weight:200;color:#fff;letter-spacing:2px">iPhone</div>', delay: 1000 },
  ],
  android: [
    { html: '<div style="font-size:48px;color:#3ddc84">▲</div>', delay: 800 },
    { html: '<div style="font-size:48px;color:#3ddc84">▲</div><div style="margin-top:12px;font-family:Roboto,sans-serif;font-size:16px;color:#e8eaed;letter-spacing:1px">android</div>', delay: 1000 },
    { html: '<div style="font-size:48px;color:#3ddc84">▲</div><div style="margin-top:12px;font-family:Roboto,sans-serif;font-size:16px;color:#e8eaed;letter-spacing:1px">android</div><div style="margin-top:12px;width:80px;height:3px;border-radius:2px;background:#333;overflow:hidden"><div style="width:0%;height:100%;background:#8ab4f8;border-radius:2px;animation:modern-load 1.2s ease-out forwards"></div></div>', delay: 1200 },
  ],
  nokia7110: [
    { html: '<div style="font-size:12px;font-family:Share Tech Mono,monospace;font-weight:700;color:#1a2a0a;line-height:1.6;text-align:center">NOKIA</div>', delay: 1200 },
    { html: '<div style="font-size:12px;font-family:Share Tech Mono,monospace;font-weight:700;color:#1a2a0a;line-height:1.6;text-align:center">NOKIA<br><br>CONNECTING PEOPLE</div>', delay: 1400 },
    { html: '<div style="font-size:11px;font-family:Share Tech Mono,monospace;font-weight:700;color:#1a2a0a;line-height:1.5;text-align:left">Menu<br>──────────<br>1 Messages<br>2 Contacts<br>3 Computer<br>──────────<br>Select ▸</div>', delay: 1000 },
  ],
  nokiacomm: [
    { html: '<div style="font-size:13px;font-family:Space Mono,monospace;color:#b0c0a0;text-align:left;line-height:1.8">Nokia 9110 Communicator</div>', delay: 1000 },
    { html: '<div style="font-size:13px;font-family:Space Mono,monospace;color:#b0c0a0;text-align:left;line-height:1.8">Nokia 9110 Communicator<br><br>ROM Version 5.02<br><br>Initializing applications...</div>', delay: 1400 },
    { html: '<div style="font-size:13px;font-family:Space Mono,monospace;color:#b0c0a0;text-align:left;line-height:1.8">Nokia 9110 Communicator<br><br>ROM Version 5.02<br><br>Loading Computer...</div>', delay: 1000 },
  ],
};

/* ── Theme descriptions for Visual tab LLM prompt ────────────── */

const THEME_INSTRUCTIONS = {
  amiga: "Style it like an Amiga Workbench application: use colors #0055aa (blue), #ff8800 (orange), #aaaaaa (grey), #ffffff (white). Monospaced font. Beveled 3D borders, no rounded corners. Think 1987 Commodore Amiga.",
  atarist: "Style it like an Atari ST GEM application: clean black-and-white UI with green (#00aa00) accents. Sharp lines, minimal decoration, monospaced font. Think 1985 Atari ST desktop.",
  c64: "Style it like a C64 GEOS application: purple/lavender palette (#6c6c9e, #b4b4dc, #d4d4ee, #3c3c6c). Chunky pixel aesthetic, monospaced font. Think 1986 Commodore 64 GEOS desktop.",
  mac: "Style it like a classic Macintosh System 7 application: black-and-white with grey (#cccccc) chrome, 1-bit iconography feel. Chicago-like sans-serif font. Rounded corners. Think 1991 Macintosh.",
  win31: "Style it like a Windows 3.1 Program Manager application: teal (#008080) and navy (#000080) accents, silver (#c0c0c0) surfaces, beveled borders. MS Sans Serif style font. Think 1992 Windows 3.1.",
  hacker: "Style it like a hacker terminal: pure black background, electric green (#00ff41) text and borders. Monospaced font. Text should glow with text-shadow. Add subtle scanline effect. Think The Matrix.",
  modern: "Style it as a modern dark web application: dark slate background (#0f1117, #1a1d27), blue accent (#6c8cff), clean sans-serif typography, rounded corners (8px), subtle shadows. Polished SaaS aesthetic.",
  win11: "Style it like a Windows 11 application: light theme with white (#fff) surfaces, subtle grey (#e5e5e5) borders, blue (#0078d4) accent. Segoe UI font. Rounded corners (8px), Mica-like frosted feel. Clean modern Microsoft design.",
  macos: "Style it like a modern macOS dark mode application: dark (#2d2d2d) surfaces, blue (#0a84ff) accent, SF Pro-like sans-serif font. Rounded corners (10px), subtle vibrancy. Think modern Apple desktop app.",
  iphone: "Style it like an iOS dark mode app: pure black (#000) background, dark grey (#1c1c1e) surfaces, blue (#0a84ff) accent. SF Pro-like sans-serif, large rounded corners (12px). Think iOS 17 native app.",
  android: "Style it like a Material You Android app: dark (#121212) background, pastel blue (#8ab4f8) and purple (#bb86fc) accents, Roboto font. Large rounded corners (12px), subtle elevation shadows. Think Material Design 3.",
  nokia7110: "Style it like a Nokia 7110 phone screen: olive-green LCD background (#8b9f6b, #9aaf7a), dark green (#1a2a0a) text, very simple chunky layout. Monospaced font, no decoration, minimalist. Think 1999 WAP phone.",
  nokiacomm: "Style it like a Nokia Communicator (9110) app: grey-green (#d0d8c0, #dce4cc) background, dark green (#202820) text, navy blue (#004488) accent. Monospaced font, simple beveled borders. Think 1998 PDA phone.",
};

/* ── Boot ────────────────────────────────────────────────────── */

async function runBoot() {
  const screen = $("#boot-screen");
  const content = $("#boot-content");
  const steps = BOOT_SEQUENCES[currentTheme] || BOOT_SEQUENCES.amiga;

  for (const step of steps) {
    content.innerHTML = step.html;
    await sleep(step.delay);
  }

  // Fade out.
  await sleep(300);
  screen.classList.add("hiding");
  await sleep(500);
  screen.classList.add("hidden");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ── Theme switching ─────────────────────────────────────────── */

function setTheme(theme) {
  document.body.classList.remove(`theme-${currentTheme}`);
  document.body.dataset.theme = theme;
  document.body.classList.add(`theme-${theme}`);
  currentTheme = theme;
  localStorage.setItem("computer-theme", theme);
}

/* ── Theme modal ─────────────────────────────────────────────── */

const THEME_LIST = [
  { id: "amiga",      name: "Amiga Workbench", bg: "#0055aa", fg: "#ff8800", pattern: "repeating-conic-gradient(#0055aa 0% 25%, #004488 0% 50%) 0 0 / 4px 4px" },
  { id: "atarist",    name: "Atari ST GEM",    bg: "#00aa00", fg: "#000000", pattern: null },
  { id: "c64",        name: "C64 GEOS",        bg: "#6c6c9e", fg: "#d4d4ee", pattern: "repeating-conic-gradient(#6c6c9e 0% 25%, #5a5a8a 0% 50%) 0 0 / 2px 2px" },
  { id: "mac",        name: "Classic Mac",      bg: "#666699", fg: "#000000", pattern: "repeating-conic-gradient(#666699 0% 25%, #555588 0% 50%) 0 0 / 2px 2px" },
  { id: "win31",      name: "Windows 3.1",      bg: "#008080", fg: "#ffffff", pattern: null },
  { id: "hacker",     name: "Hacker",           bg: "#000000", fg: "#00ff41", pattern: null },
  { id: "modern",     name: "Modern",           bg: "#0f1117", fg: "#6c8cff", pattern: null },
  { id: "win11",      name: "Windows 11",       bg: "#f3f3f3", fg: "#0078d4", pattern: null },
  { id: "macos",      name: "macOS",            bg: "#1e1e1e", fg: "#0a84ff", pattern: null },
  { id: "iphone",     name: "iPhone",           bg: "#000000", fg: "#0a84ff", pattern: null },
  { id: "android",    name: "Android",          bg: "#121212", fg: "#8ab4f8", pattern: null },
  { id: "nokia7110",  name: "Nokia 7110",       bg: "#8b9f6b", fg: "#1a2a0a", pattern: null },
  { id: "nokiacomm",  name: "Nokia Communicator", bg: "#d0d8c0", fg: "#004488", pattern: null },
];

/* ── Custom themes (user-uploaded, stored in localStorage) ──── */

// All CSS variable keys that define a theme.
const THEME_VARS = [
  "bg", "bg-pattern", "surface", "surface-alt",
  "chrome", "chrome-hi", "chrome-lo", "chrome-dark",
  "titlebar-bg", "titlebar-fg", "titlebar-stripe",
  "text", "text-dim", "text-inv",
  "accent", "accent-alt",
  "input-bg", "input-border",
  "btn-bg", "btn-fg", "btn-active",
  "log-bg", "log-text", "log-dim",
  "led-on", "led-off",
  "scrollbar-track", "scrollbar-thumb",
  "font", "font-size", "font-weight", "line-height",
  "font-title", "font-title-size",
  "bevel", "radius",
  "win-close-w", "win-depth-w", "win-gadget-h",
];

// Template based on the Modern theme — a clean starting point.
const THEME_TEMPLATE = {
  name: "My Custom Theme",
  description: "Describe your theme style for visual generation prompts.",
  variables: {
    "bg":             "#0f1117",
    "bg-pattern":     "none",
    "surface":        "#1a1d27",
    "surface-alt":    "#222633",
    "chrome":         "#1a1d27",
    "chrome-hi":      "#2a2e3d",
    "chrome-lo":      "#13151c",
    "chrome-dark":    "#2a2e3d",
    "titlebar-bg":    "#1a1d27",
    "titlebar-fg":    "#e0e0e6",
    "titlebar-stripe": "none",
    "text":           "#e0e0e6",
    "text-dim":       "#8888a0",
    "text-inv":       "#0f1117",
    "accent":         "#6c8cff",
    "accent-alt":     "#a78bfa",
    "input-bg":       "#13151c",
    "input-border":   "#2a2e3d",
    "btn-bg":         "#6c8cff",
    "btn-fg":         "#ffffff",
    "btn-active":     "#5070dd",
    "log-bg":         "#0a0d14",
    "log-text":       "#7ee787",
    "log-dim":        "#4a6a52",
    "led-on":         "#6c8cff",
    "led-off":        "#2a2e3d",
    "scrollbar-track": "#1a1d27",
    "scrollbar-thumb": "#2a2e3d",
    "font":           "-apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif",
    "font-size":      "14px",
    "font-weight":    "400",
    "line-height":    "1.55",
    "font-title":     "var(--font)",
    "font-title-size": "13px",
    "bevel":          "1px",
    "radius":         "8px",
    "win-close-w":    "0px",
    "win-depth-w":    "0px",
    "win-gadget-h":   "0px",
  },
  boot: [
    { text: "Loading custom theme...", delay: 600 },
    { text: "Ready.", delay: 400 }
  ]
};

let customThemes = []; // Array of { id, name, description, variables, boot }

function loadCustomThemes() {
  try {
    const raw = localStorage.getItem("computer-custom-themes");
    if (raw) customThemes = JSON.parse(raw);
  } catch (e) {
    console.warn("Failed to load custom themes:", e);
    customThemes = [];
  }
}

function saveCustomThemes() {
  localStorage.setItem("computer-custom-themes", JSON.stringify(customThemes));
}

function customThemeId(name) {
  // Generate a stable CSS-safe id from the name.
  return "custom-" + name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 30);
}

function injectCustomThemeCSS(theme) {
  // Remove old style element for this theme if present.
  const existingEl = document.getElementById(`custom-theme-style-${theme.id}`);
  if (existingEl) existingEl.remove();

  // Build CSS from variables.
  let css = `.theme-${theme.id} {\n`;
  for (const [key, val] of Object.entries(theme.variables)) {
    if (THEME_VARS.includes(key)) {
      css += `  --${key}: ${val};\n`;
    }
  }
  css += `}\n`;

  // Boot screen.
  const bg = theme.variables["bg"] || "#000";
  const fg = theme.variables["text"] || "#fff";
  css += `.theme-${theme.id} #boot-screen { background: ${bg}; }\n`;
  css += `.theme-${theme.id} #boot-content { color: ${fg}; }\n`;

  // Modern-style window overrides (hide retro gadgets, rounded corners).
  const radius = theme.variables["radius"] || "8px";
  const radiusPx = parseInt(radius);
  if (radiusPx >= 6) {
    css += `.theme-${theme.id} .window { border-radius: ${radius}; box-shadow: 0 2px 12px rgba(0,0,0,0.3); }\n`;
    css += `.theme-${theme.id} .win-btn { display: none; }\n`;
    css += `.theme-${theme.id} ::-webkit-scrollbar { width: 6px; }\n`;
    css += `.theme-${theme.id} ::-webkit-scrollbar-button { display: none; }\n`;
    css += `.theme-${theme.id} ::-webkit-scrollbar-thumb { border-radius: 3px; border: none; }\n`;
  }

  const style = document.createElement("style");
  style.id = `custom-theme-style-${theme.id}`;
  style.textContent = css;
  document.head.appendChild(style);
}

function registerCustomThemeBoot(theme) {
  // Register boot sequence from the theme definition.
  const bootSteps = (theme.boot || []).map((step) => ({
    html: `<div style="font-size:14px">${esc(step.text || "")}</div>`,
    delay: step.delay || 500,
  }));
  if (bootSteps.length === 0) {
    bootSteps.push({ html: `<div style="font-size:14px">${esc(theme.name)}</div>`, delay: 600 });
  }
  BOOT_SEQUENCES[theme.id] = bootSteps;

  // Register theme instructions for visual generation.
  if (theme.description) {
    THEME_INSTRUCTIONS[theme.id] = theme.description;
  } else {
    const accent = theme.variables["accent"] || "#6c8cff";
    const bg = theme.variables["bg"] || "#000";
    THEME_INSTRUCTIONS[theme.id] = `Style it to match this custom theme: background ${bg}, accent color ${accent}. Use the aesthetic described by the theme name "${theme.name}".`;
  }
}

function downloadThemeTemplate() {
  const blob = new Blob([JSON.stringify(THEME_TEMPLATE, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "computer-theme-template.json";
  a.click();
  URL.revokeObjectURL(url);
}

function validateCustomTheme(obj) {
  if (!obj || typeof obj !== "object") return "Invalid JSON — expected an object.";
  if (!obj.name || typeof obj.name !== "string" || obj.name.trim().length < 2) {
    return 'Missing or invalid "name" (must be at least 2 characters).';
  }
  if (!obj.variables || typeof obj.variables !== "object") {
    return 'Missing "variables" object with CSS custom properties.';
  }
  // Check that at least some key variables are present.
  const required = ["bg", "text", "accent", "surface"];
  const missing = required.filter((k) => !obj.variables[k]);
  if (missing.length > 0) {
    return `Missing required variables: ${missing.join(", ")}`;
  }
  return null; // valid
}

function uploadCustomTheme(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const obj = JSON.parse(e.target.result);
      const err = validateCustomTheme(obj);
      if (err) {
        alert("Invalid theme file:\n\n" + err);
        return;
      }

      const id = customThemeId(obj.name);

      // Check for duplicates — overwrite if same name.
      const existingIdx = customThemes.findIndex((t) => t.id === id);

      const theme = {
        id,
        name: obj.name.trim(),
        description: (obj.description || "").trim(),
        variables: {},
        boot: obj.boot || [],
      };

      // Copy only valid variable keys.
      for (const key of THEME_VARS) {
        if (obj.variables[key] !== undefined) {
          theme.variables[key] = String(obj.variables[key]);
        }
      }

      if (existingIdx >= 0) {
        customThemes[existingIdx] = theme;
      } else {
        customThemes.push(theme);
      }

      saveCustomThemes();
      injectCustomThemeCSS(theme);
      registerCustomThemeBoot(theme);
      renderCustomThemeGrid();

      logEntry("system", `Custom theme "${theme.name}" ${existingIdx >= 0 ? "updated" : "uploaded"}`);

      // Auto-select the new theme.
      setTheme(theme.id);
      updateThemeModalSelection();
      setTimeout(() => closeThemeModal(), 300);

    } catch (parseErr) {
      alert("Failed to parse theme file:\n\n" + parseErr.message);
    }
  };
  reader.readAsText(file);
}

function deleteCustomTheme(id) {
  const idx = customThemes.findIndex((t) => t.id === id);
  if (idx < 0) return;

  const theme = customThemes[idx];
  customThemes.splice(idx, 1);
  saveCustomThemes();

  // Remove injected CSS.
  const styleEl = document.getElementById(`custom-theme-style-${id}`);
  if (styleEl) styleEl.remove();

  // Remove boot/instructions.
  delete BOOT_SEQUENCES[id];
  delete THEME_INSTRUCTIONS[id];

  // If this theme was active, switch to Modern.
  if (currentTheme === id) {
    setTheme("modern");
  }

  renderCustomThemeGrid();
  updateThemeModalSelection();
  logEntry("system", `Custom theme "${theme.name}" deleted`);
}

function renderCustomThemeGrid() {
  const grid = $("#custom-theme-grid");
  const empty = $("#custom-theme-empty");

  // Remove old custom cards (but keep empty placeholder).
  grid.querySelectorAll(".theme-card.custom-theme").forEach((c) => c.remove());

  if (customThemes.length === 0) {
    if (empty) empty.style.display = "";
    return;
  }

  if (empty) empty.style.display = "none";

  for (const t of customThemes) {
    const card = document.createElement("div");
    card.className = "theme-card custom-theme" + (t.id === currentTheme ? " selected" : "");
    card.dataset.theme = t.id;

    const preview = document.createElement("div");
    preview.className = "theme-card-preview";
    preview.style.background = t.variables["bg-pattern"] && t.variables["bg-pattern"] !== "none"
      ? t.variables["bg-pattern"]
      : (t.variables["bg"] || "#333");
    preview.style.color = t.variables["accent"] || "#fff";
    preview.textContent = "Aa";

    const nameRow = document.createElement("div");
    nameRow.className = "theme-card-name";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = t.name;
    nameSpan.style.flex = "1";
    nameSpan.style.overflow = "hidden";
    nameSpan.style.textOverflow = "ellipsis";

    const delBtn = document.createElement("button");
    delBtn.className = "theme-card-delete";
    delBtn.title = "Delete this theme";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm(`Delete custom theme "${t.name}"?`)) {
        deleteCustomTheme(t.id);
      }
    });

    nameRow.appendChild(nameSpan);
    nameRow.appendChild(delBtn);

    card.appendChild(preview);
    card.appendChild(nameRow);
    card.addEventListener("click", () => {
      setTheme(t.id);
      updateThemeModalSelection();
      setTimeout(() => closeThemeModal(), 200);
    });

    grid.appendChild(card);
  }
}

function updateThemeModalSelection() {
  const allCards = document.querySelectorAll("#theme-grid .theme-card, #custom-theme-grid .theme-card");
  allCards.forEach((c) => {
    c.classList.toggle("selected", c.dataset.theme === currentTheme);
  });
}

function initThemeModal() {
  const modal = $("#theme-modal");
  const grid = $("#theme-grid");
  const btn = $("#theme-btn");

  // Load custom themes from localStorage and inject their CSS.
  loadCustomThemes();
  for (const t of customThemes) {
    injectCustomThemeCSS(t);
    registerCustomThemeBoot(t);
  }

  // Build built-in theme cards.
  for (const t of THEME_LIST) {
    const card = document.createElement("div");
    card.className = "theme-card" + (t.id === currentTheme ? " selected" : "");
    card.dataset.theme = t.id;

    const preview = document.createElement("div");
    preview.className = "theme-card-preview";
    preview.style.background = t.pattern || t.bg;
    preview.style.color = t.fg;
    preview.textContent = "Aa";

    const name = document.createElement("div");
    name.className = "theme-card-name";
    name.textContent = t.name;

    card.appendChild(preview);
    card.appendChild(name);
    card.addEventListener("click", () => {
      setTheme(t.id);
      updateThemeModalSelection();
      // Close modal after a short delay so user sees the selection.
      setTimeout(() => closeThemeModal(), 200);
    });
    grid.appendChild(card);
  }

  // Render custom themes.
  renderCustomThemeGrid();

  // Download template button.
  const dlBtn = $("#theme-download-tpl");
  if (dlBtn) dlBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    downloadThemeTemplate();
  });

  // Upload button.
  const uploadInput = $("#theme-upload-input");
  if (uploadInput) uploadInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      uploadCustomTheme(file);
      e.target.value = ""; // reset so same file can be re-uploaded
    }
  });

  // Open modal.
  btn.addEventListener("click", () => {
    modal.classList.add("active");
    updateThemeModalSelection();
  });

  // Close on backdrop click.
  modal.querySelector(".modal-backdrop").addEventListener("click", closeThemeModal);
  modal.querySelector(".modal-close").addEventListener("click", closeThemeModal);

  // Close on Escape.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeThemeModal();
    }
  });
}

function closeThemeModal() {
  $("#theme-modal").classList.remove("active");
}

/* ── WebSocket connection ────────────────────────────────────── */

function connect() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    connected = true;
    $(".status-dot").classList.add("connected");
    logEntry("system", "Connected to Captain Claw");
  };

  ws.onclose = () => {
    connected = false;
    $(".status-dot").classList.remove("connected");
    logEntry("error", "Connection lost — reconnecting...");
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    logEntry("error", "WebSocket error");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      console.warn("Bad message:", e);
    }
  };
}

function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

/* ── Message handling ────────────────────────────────────────── */

function handleMessage(data) {
  switch (data.type) {
    case "welcome":
      handleWelcome(data);
      break;

    case "chat_message":
      handleChatMessage(data);
      break;

    case "monitor":
      handleMonitor(data);
      break;

    case "status":
      handleStatus(data);
      break;

    case "thinking":
      handleThinking(data);
      break;

    case "tool_output_inline":
      handleToolStream(data);
      break;

    case "error":
      logEntry("error", data.message || data.error || "Unknown error");
      break;

    case "session_info":
      if (data.name) {
        $("#session-badge").textContent = data.name;
      }
      // Update session id and reload exploration history when session changes.
      var newSid = data.id || data.session_id;
      if (newSid && newSid !== sessionId) {
        sessionId = newSid;
        explorationNodes.clear();
        currentNodeId = null;
        historyBranchNodeId = null;
        pendingExploration = null;
        lastPrompt = "";
        lastResult = "";
        agentCreatedHtmlFiles = [];
        clearAnswer();
        clearVisual();
        renderHistoryDrawer();
        $("#log-content").innerHTML = "";
        logEntry("system", "Session switched — " + (data.name || newSid.slice(0, 8)));
        loadExplorationHistory(sessionId);
      }
      break;

    case "session_switched":
      // Backend signals a full session switch (e.g. /nuke).
      // session_info broadcast follows, which handles the reload.
      break;

    case "command_result":
      if (data.command === "/btw") {
        logEntry("btw", data.content || "Noted.");
      } else {
        logEntry("system", data.content || "");
      }
      break;

    case "approval_request":
      // Auto-approve in Computer mode (can be refined later).
      send({ type: "approval_response", approved: true, request_id: data.request_id });
      logEntry("system", `Auto-approved: ${data.tool || "action"}`);
      break;
  }
}

function handleWelcome(data) {
  // Session info is nested under data.session (from ws_handler).
  const sess = data.session || {};
  if (sess.name || data.session_name) {
    $("#session-badge").textContent = sess.name || data.session_name;
  }
  // Capture session id for exploration persistence and file scoping.
  sessionId = sess.id || data.session_id || data.id || null;
  logEntry("system", `Session ready${sessionId ? ' (' + sessionId.slice(0, 8) + '\u2026)' : ''}`);

  // Apply the saved model selection to the session.
  if (selectedModel && connected) {
    send({ type: "set_model", selector: selectedModel });
    logEntry("system", `Applying saved model: ${selectedModel}`);
  }

  // Load exploration history.
  if (sessionId) {
    loadExplorationHistory(sessionId);
  }

  // Replay history into the answer panel.
  if (data.history && data.history.length > 0) {
    for (const msg of data.history) {
      if (msg.role === "assistant" && msg.content) {
        renderAnswer(msg.content, true);
      }
    }
  }
}

function handleChatMessage(data) {
  if (data.role === "assistant") {
    setProcessing(false);
    renderAnswer(data.content || "", false);
  }
  // Track HTML files created by the agent (broadcast by backend).
  if (data.role === "html_file" && data.content) {
    agentCreatedHtmlFiles.push(data.content);
    console.log('[Computer] Agent created HTML file:', data.content);
  }
}

function handleMonitor(data) {
  const toolName = data.tool_name || "unknown";
  const output = data.output || "";
  const truncated = output.length > 200 ? output.slice(0, 200) + "..." : output;
  logEntry("tool", `${toolName}: ${truncated}`, toolName);
  blinkLed();
}

function handleStatus(data) {
  const status = data.status || "";
  logEntry("status", status);

  if (status.toLowerCase().includes("thinking") || status.toLowerCase().includes("running")) {
    setProcessing(true);
  }
}

function handleThinking(data) {
  const text = data.text || "";
  const tool = data.tool || "";
  const phase = data.phase || "";

  if (phase === "done") {
    return;
  }

  if (tool) {
    logEntry("thinking", `[${tool}] ${text}`);
  } else if (text) {
    logEntry("thinking", text);
  }
  blinkLed();
}

function handleToolStream(data) {
  const chunk = data.chunk || data.text || "";
  if (chunk.trim()) {
    logEntry("stream", chunk);
  }
}

/* ── Input handling ──────────────────────────────────────────── */

function handleSend() {
  const input = $("#input-box");
  const text = input.value.trim();
  if (!text && !pendingImagePath && !pendingFilePath) return;
  if (!connected) return;

  // ── /btw command: inject additional instructions while task is running ──
  const btwMatch = text.match(/^\/btw\s+([\s\S]+)/i) || text.match(/^btw\s+([\s\S]+)/i);
  if (btwMatch && isProcessing) {
    const btwText = btwMatch[1].trim();
    if (btwText) {
      send({ type: "btw", content: btwText });
      logEntry("btw", btwText);
      input.value = "";
      input.style.height = "auto";
    }
    return;
  }

  // If the explore confirmation bar is active, treat this as an exploration send.
  const bar = $("#explore-confirm-bar");
  if (bar && bar.classList.contains("active")) {
    const topic = bar._exploreTopic || "";
    pendingExploration = {
      parentId: currentNodeId,
      edgeLabel: topic,
      source: 'click',
    };
    bar.classList.remove("active");
    bar.style.display = "none";
    $("#send-btn").style.display = "";
  }

  input.value = "";
  input.style.height = "auto";

  // Store prompt for Visual tab.
  lastPrompt = text;
  lastResult = "";

  // Clear tracked HTML files from previous turn.
  agentCreatedHtmlFiles = [];

  // Show user input in log.
  logEntry("user", text || "(attachment)");

  // Clear previous answer for fresh response.
  clearAnswer();
  clearVisual();

  // Render input decomposition.
  if (text) renderInputDecomposition(text);

  // ── History branching: rewind session context if forking from an old node ──
  let rewindTimestamp = null;
  if (historyBranchNodeId) {
    const branchNode = explorationNodes.get(historyBranchNodeId);
    if (branchNode) {
      rewindTimestamp = branchNode.created_at;
      // Prune exploration nodes newer than the branch point.
      const branchTime = branchNode.created_at || '';
      const toDelete = [];
      for (const [id, n] of explorationNodes) {
        if ((n.created_at || '') > branchTime) toDelete.push(id);
      }
      for (const id of toDelete) {
        explorationNodes.delete(id);
        // Fire-and-forget backend cleanup.
        fetch(`/api/computer/exploration/${id}`, { method: 'DELETE' }).catch(() => {});
      }
      currentNodeId = historyBranchNodeId;
      logEntry("system", `Forking from: ${branchNode.prompt.slice(0, 50)}${branchNode.prompt.length > 50 ? '…' : ''}`);
    }
    historyBranchNodeId = null;
  }

  // Send to agent.
  if (text.startsWith("/")) {
    send({ type: "command", command: text });
  } else {
    const msg = { type: "chat", content: text || "" };
    if (pendingImagePath) msg.image_path = pendingImagePath;
    if (pendingFilePath) msg.file_path = pendingFilePath;
    if (rewindTimestamp) msg.rewind_to = rewindTimestamp;
    send(msg);
  }

  clearAttachment();
  setProcessing(true);
}

/* ── Input decomposition ─────────────────────────────────────── */

function renderInputDecomposition(text) {
  const container = $("#decomposition-content");

  const words = text.split(/\s+/);
  const isQuestion = text.includes("?");
  const hasAction = /\b(create|build|write|fix|update|add|remove|delete|analyze|find|search|list|show|explain|summarize|review|refactor)\b/i.test(text);

  let intent = "query";
  if (hasAction) intent = "action";
  if (text.startsWith("/")) intent = "command";

  const actionPattern = /\b(create|build|write|fix|update|add|remove|delete|analyze|find|search|list|show|explain|summarize|review|refactor)\s+(.+?)(?:\s+(?:and|then|,)\s+|$)/gi;
  const steps = [];
  let match;
  while ((match = actionPattern.exec(text)) !== null) {
    steps.push({ action: match[1], target: match[2].trim() });
  }

  if (steps.length === 0) {
    steps.push({ action: intent, target: text });
  }

  const stepsHtml = steps.map((s) => `
    <div class="decomp-step">
      <span class="decomp-bullet">◆</span>
      <span class="decomp-action">${esc(s.action)}</span>
      <span class="decomp-target">→ ${esc(s.target)}</span>
    </div>
  `).join("");

  const complexity = words.length > 20 ? "complex" : words.length > 8 ? "moderate" : "simple";

  container.innerHTML = `
    <div class="decomp-node">
      <div class="decomp-intent">${esc(intent)}</div>
      ${stepsHtml}
      <div class="decomp-meta">
        Complexity: ${complexity} · Words: ${words.length}${isQuestion ? " · Question" : ""}
      </div>
    </div>
  `;
  container.classList.remove("decomp-empty");
}

/* ── Answer rendering ────────────────────────────────────────── */

function clearAnswer() {
  const el = $("#answer-content");
  el.innerHTML = '<div class="output-processing"><span class="processing-dots">⣾</span> Processing...</div>';
  el.classList.remove("output-empty");
}

function clearVisual() {
  const el = $("#visual-content");
  el.innerHTML = '<span class="output-placeholder">Visual rendering will appear here...</span>';
  el.classList.add("output-empty");
  const frame = $("#visual-frame");
  if (frame) {
    frame.style.display = "none";
    frame.srcdoc = "";
  }
  // Hide visual choice bar if present.
  const bar = $("#visual-choice-bar");
  if (bar) bar.style.display = "none";
}

function renderAnswer(content, isReplay) {
  const el = $("#answer-content");
  el.innerHTML = markdownToHtml(content);
  el.classList.remove("output-empty");

  // Store for Visual tab.
  lastResult = content;

  // Auto-generate blueprint from the answer.
  generateBlueprint(content);

  // Create exploration node (skip for replayed history).
  if (!isReplay && lastPrompt) {
    const node = createExplorationNode(lastPrompt, content);
    currentNodeId = node.id;
    saveExplorationNode(node);
    pendingExploration = null;

    // Update map & history drawer.
    renderMap();
    renderHistoryDrawer();
  }

  // Auto-trigger Visual generation if the Visual tab is active.
  const visualTabActive = $(".tab-btn[data-tab='visual']").classList.contains("active");
  if (visualTabActive) {
    if (agentCreatedHtmlFiles.length > 0) {
      // Agent already created an HTML file — ask user what to do.
      showVisualChoiceBar(agentCreatedHtmlFiles);
    } else {
      generateVisual();
    }
  }
}

/* ── Blueprint tab (structural decomposition) ────────────────── */

function generateBlueprint(content) {
  const el = $("#blueprint-content");

  const sections = [];
  const lines = content.split("\n");
  let currentSection = { type: "text", title: "Response", lines: [] };

  for (const line of lines) {
    const trimmed = line.trim();
    if (/^#{1,3}\s+/.test(trimmed)) {
      if (currentSection.lines.length > 0) {
        sections.push(currentSection);
      }
      currentSection = {
        type: "section",
        title: trimmed.replace(/^#+\s*/, ""),
        lines: [],
      };
    } else if (/^```/.test(trimmed)) {
      if (currentSection.lines.length > 0) {
        sections.push(currentSection);
      }
      currentSection = {
        type: "code",
        title: "Code",
        lines: [],
      };
    } else if (/^[-*]\s+/.test(trimmed)) {
      currentSection.type = "list";
      currentSection.lines.push(trimmed.replace(/^[-*]\s+/, ""));
    } else if (trimmed) {
      currentSection.lines.push(trimmed);
    }
  }
  if (currentSection.lines.length > 0) {
    sections.push(currentSection);
  }

  if (sections.length === 0) {
    el.innerHTML = '<span class="output-placeholder">No structure to decompose</span>';
    el.classList.add("output-empty");
    return;
  }

  const typeIcons = { text: "📄", section: "📑", code: "💻", list: "📋" };

  el.innerHTML = sections.map((s, i) => {
    const icon = typeIcons[s.type] || "📄";
    const bodyHtml = s.lines.map((l) => `<div>${esc(l)}</div>`).join("");
    return `
      <div class="blueprint-section">
        <div class="blueprint-header" onclick="this.parentElement.classList.toggle('collapsed')">
          ${icon} ${esc(s.title || `Section ${i + 1}`)}
          <span class="blueprint-label">${s.type}</span>
        </div>
        <div class="blueprint-body">${bodyHtml}</div>
      </div>
    `;
  }).join(`
    <div class="blueprint-step">
      <span class="blueprint-arrow">↓</span>
    </div>
  `);

  el.classList.remove("output-empty");
}

/* ── Visual tab (LLM-generated HTML) ─────────────────────────── */

async function generateVisual() {
  if (!lastPrompt || !lastResult) return;
  if (visualGenerating) return;

  visualGenerating = true;
  const el = $("#visual-content");
  const frame = $("#visual-frame");

  el.innerHTML = '<div class="output-processing"><span class="processing-dots">⣾</span> Generating visual...</div>';
  el.classList.remove("output-empty");
  frame.style.display = "none";

  logEntry("system", "Generating visual rendering...");
  blinkLed();

  const payload = {
    prompt: lastPrompt,
    result: lastResult,
    theme: currentTheme,
    theme_instructions: THEME_INSTRUCTIONS[currentTheme] || THEME_INSTRUCTIONS.modern,
    token_tier: selectedTokenTier,
    model: selectedModel || "",
  };

  const t0 = performance.now();
  console.group("%c[Computer] Visual Generation", "color:#6c8cff;font-weight:bold");
  console.log("Theme:", currentTheme);
  console.log("Tier:", selectedTokenTier, "| Model:", selectedModel || "(default)");
  console.log("Prompt:", lastPrompt.length > 120 ? lastPrompt.slice(0, 120) + "…" : lastPrompt);
  console.log("Result length:", lastResult.length, "chars");
  console.log("Request payload:", payload);

  try {
    const res = await fetch("/api/computer/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const elapsed = Math.round(performance.now() - t0);

    if (!res.ok) {
      const errBody = await res.text();
      console.error("Response:", res.status, errBody);
      console.log("Elapsed:", elapsed, "ms");
      console.groupEnd();
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    const html = data.html || "";

    console.log("Response status:", res.status);
    console.log("HTML size:", html.length, "chars");
    console.log("Elapsed:", elapsed, "ms");
    console.groupEnd();

    if (!html) {
      el.innerHTML = '<span class="output-placeholder">No visual generated</span>';
      el.classList.add("output-empty");
      return;
    }

    // Count explore links for logging.
    const linkCount = (html.match(/class="explore-link"/g) || []).length;
    if (linkCount > 0) {
      console.log(`[Computer] Found ${linkCount} explore-link elements`);
    }

    // Inject postMessage bridge and render in sandboxed iframe.
    const enrichedHtml = injectExploreBridge(html);
    el.innerHTML = "";
    el.classList.remove("output-empty");
    frame.srcdoc = enrichedHtml;
    frame.style.display = "block";

    // Store visual HTML on the current exploration node.
    if (currentNodeId && explorationNodes.has(currentNodeId)) {
      const node = explorationNodes.get(currentNodeId);
      node.visual_html = html;
      updateExplorationNodeVisual(currentNodeId, html);
    }

    logEntry("system", `Visual rendering complete (${humanTime(elapsed)}, ${humanSize(html.length)}${linkCount > 0 ? `, ${linkCount} explore links` : ''})`);

  } catch (err) {
    console.error("Visual generation failed:", err);
    console.groupEnd();
    el.innerHTML = `<span class="output-placeholder">Visual generation failed: ${esc(err.message)}</span>`;
    el.classList.add("output-empty");
    logEntry("error", `Visual generation failed: ${err.message}`);
  } finally {
    visualGenerating = false;
  }
}

/* ── Visual choice bar (agent HTML vs generate new) ──────────── */

function showVisualChoiceBar(htmlFiles) {
  const el = $("#visual-content");
  const frame = $("#visual-frame");

  // Build file list display.
  const fileNames = htmlFiles.map(f => f.split('/').pop()).join(', ');

  let bar = $("#visual-choice-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "visual-choice-bar";
    el.parentNode.insertBefore(bar, el);
  }

  bar.innerHTML = `
    <div class="visual-choice-info">
      <span class="visual-choice-icon">🌐</span>
      <span class="visual-choice-text">Agent created <strong>${esc(fileNames)}</strong> — use it as visual or generate a new themed rendering?</span>
    </div>
    <div class="visual-choice-actions">
      <button id="visual-use-file-btn" class="tab-btn" title="Use agent's HTML file">📄 Use File</button>
      <button id="visual-generate-btn" class="tab-btn" title="Generate new themed visual">🎨 Generate New</button>
    </div>
  `;
  bar.style.display = "flex";

  // Clear the visual area placeholder.
  el.innerHTML = '';
  el.classList.remove("output-empty");
  frame.style.display = "none";

  // Wire buttons.
  bar.querySelector("#visual-use-file-btn").addEventListener("click", async () => {
    bar.style.display = "none";
    await useAgentHtmlFile(htmlFiles[htmlFiles.length - 1]);
  });

  bar.querySelector("#visual-generate-btn").addEventListener("click", () => {
    bar.style.display = "none";
    generateVisual();
  });

  logEntry("system", `Agent created HTML file — choose: use it or generate new visual`);
}

async function useAgentHtmlFile(filePath) {
  const el = $("#visual-content");
  const frame = $("#visual-frame");
  const fileName = filePath.split('/').pop();

  el.innerHTML = '<div class="output-processing"><span class="processing-dots">⣾</span> Loading agent HTML...</div>';
  el.classList.remove("output-empty");

  try {
    // The path from the backend broadcast is the physical path — use it directly.
    const contentRes = await fetch(`/api/files/content?path=${encodeURIComponent(filePath)}`);
    if (!contentRes.ok) {
      // Fall back to view endpoint in iframe.
      el.innerHTML = "";
      frame.src = `/api/files/view?path=${encodeURIComponent(filePath)}`;
      frame.style.display = "block";
      logEntry("system", `Loaded agent HTML file: ${fileName}`);
      return;
    }

    const data = await contentRes.json();
    const html = data.content || "";

    if (!html) {
      throw new Error("Empty HTML file");
    }

    // Inject the explore bridge and render in the iframe.
    const enrichedHtml = injectExploreBridge(html);
    el.innerHTML = "";
    el.classList.remove("output-empty");
    frame.srcdoc = enrichedHtml;
    frame.style.display = "block";

    // Store visual HTML on the current exploration node.
    if (currentNodeId && explorationNodes.has(currentNodeId)) {
      const node = explorationNodes.get(currentNodeId);
      node.visual_html = html;
      updateExplorationNodeVisual(currentNodeId, html);
    }

    logEntry("system", `Using agent HTML file: ${fileName} (${humanSize(html.length)}) — no extra LLM call`);

  } catch (err) {
    console.warn("Failed to load agent HTML, falling back to generate:", err);
    logEntry("system", `Failed to load agent HTML (${err.message}), generating new visual...`);
    generateVisual();
  }
}

/* ── Fullscreen toggle ───────────────────────────────────────── */

let isFullscreen = false;

function toggleFullscreen() {
  const panel = $("#output-panel");
  isFullscreen = !isFullscreen;
  panel.classList.toggle("fullscreen", isFullscreen);
  document.body.classList.toggle("visual-fullscreen", isFullscreen);
  $("#fullscreen-btn").textContent = isFullscreen ? "⛶" : "⛶";
  $("#fullscreen-btn").title = isFullscreen ? "Exit fullscreen" : "Fullscreen";
}

function initFullscreen() {
  const btn = $("#fullscreen-btn");
  btn.addEventListener("click", toggleFullscreen);

  // Re-run visual button.
  const rerunBtn = $("#rerun-visual-btn");
  rerunBtn.addEventListener("click", () => {
    if (!lastPrompt || !lastResult) {
      logEntry("system", "Nothing to re-run — no prompt/result yet");
      return;
    }
    // Force re-generation by clearing the iframe first.
    const frame = $("#visual-frame");
    frame.srcdoc = "";
    frame.style.display = "none";
    visualGenerating = false;
    generateVisual();
  });

  // Escape key exits fullscreen.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isFullscreen) {
      toggleFullscreen();
    }
  });
}

/* ── Activity log ────────────────────────────────────────────── */

const MAX_LOG_ENTRIES = 200;

function logEntry(type, text, toolName) {
  const container = $("#log-content");
  const now = new Date();
  const ts = now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const icons = {
    system: "●",
    user: "▶",
    btw: "💡",
    status: "◎",
    tool: "⚙",
    thinking: "◇",
    stream: "│",
    error: "✖",
  };

  const entry = document.createElement("div");
  entry.className = `log-entry`;

  const icon = icons[type] || "·";
  const textClass = `log-${type}`;

  entry.innerHTML = `
    <span class="log-ts">${ts}</span>
    <span class="log-icon">${icon}</span>
    <span class="log-text ${textClass}">${esc(text)}</span>
  `;

  container.appendChild(entry);

  while (container.children.length > MAX_LOG_ENTRIES) {
    container.removeChild(container.firstChild);
  }

  container.scrollTop = container.scrollHeight;
}

/* ── Drive LED ───────────────────────────────────────────────── */

function blinkLed() {
  const led = $("#drive-led");
  led.classList.add("on");
  clearTimeout(ledTimer);
  ledTimer = setTimeout(() => led.classList.remove("on"), 150);
}

function setProcessing(active) {
  isProcessing = active;
  const led = $("#drive-led");
  if (active) {
    led.classList.add("on");
  } else {
    led.classList.remove("on");
  }
}

/* ── Panel resizing ──────────────────────────────────────────── */

function initResize() {
  const handle = $("#h-resize");
  const left = $("#left-column");
  const workspace = $("#workspace");
  let dragging = false;

  // Restore persisted panel width.
  const saved = localStorage.getItem("panel-width");
  if (saved) {
    const pct = parseFloat(saved);
    if (pct >= 25 && pct <= 70) left.style.width = pct + "%";
  }

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = workspace.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(25, Math.min(70, pct));
    left.style.width = clamped + "%";
  });

  document.addEventListener("mouseup", () => {
    if (dragging) {
      dragging = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      // Persist the panel width.
      const pct = (left.getBoundingClientRect().width / workspace.getBoundingClientRect().width) * 100;
      localStorage.setItem("panel-width", pct.toFixed(2));
    }
  });
}

/* ── Tab switching ───────────────────────────────────────────── */

function initTabs() {
  for (const btn of $$("#output-tabs .tab-btn")) {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;

      // Update buttons.
      for (const b of $$("#output-tabs .tab-btn")) {
        b.classList.toggle("active", b === btn);
      }

      // Update panels.
      for (const panel of $$(".output-tab")) {
        panel.classList.toggle("active", panel.id === `output-${tab}`);
      }

      // Show/hide Visual-only buttons.
      const fsBtn = $("#fullscreen-btn");
      const rerunBtn = $("#rerun-visual-btn");
      if (tab === "visual") {
        fsBtn.style.display = "inline-block";
        rerunBtn.style.display = "inline-block";
        // Auto-generate visual if we have content and haven't generated yet.
        const frame = $("#visual-frame");
        if (lastResult && (!frame.srcdoc || frame.srcdoc === "")) {
          if (agentCreatedHtmlFiles.length > 0) {
            showVisualChoiceBar(agentCreatedHtmlFiles);
          } else {
            generateVisual();
          }
        }
      } else {
        fsBtn.style.display = "none";
        rerunBtn.style.display = "none";
        // Exit fullscreen if leaving visual tab.
        if (isFullscreen) {
          toggleFullscreen();
        }
      }

      // Render map when switching to Map tab.
      if (tab === "map") {
        renderMap();
      }

      // Load files when switching to Files tab.
      if (tab === "files") {
        loadFiles();
      }
    });
  }
}

/* ═══════════════════════════════════════════════════════════════
   MAP TAB — SVG mind-map of the exploration tree
   ═══════════════════════════════════════════════════════════════ */

const MAP_MAIN_W = 200;
const MAP_MAIN_H = 60;
const MAP_SUB_W = 140;
const MAP_SUB_H = 36;
const MAP_H_GAP = 40;
const MAP_V_GAP = 80;
const MAP_SUB_V_GAP = 20;  // tighter gap for sub-steps

let mapTransform = { x: 0, y: 0, scale: 1 };
let mapDragging = false;
let mapDragStart = { x: 0, y: 0 };

function renderMap() {
  const svg = $("#map-svg");
  if (!svg) return;

  // Show all exploration nodes including text-selection-initiated ones.
  const allNodes = Array.from(explorationNodes.values());
  const nodes = allNodes;
  if (nodes.length === 0) {
    svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="var(--text-dim)" font-size="13" font-family="var(--font)">No exploration history yet. Ask a question to begin.</text>`;
    return;
  }

  // Sort chronologically.
  const sorted = [...nodes].sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));

  // Build parent→children map (only among main nodes).
  const mainIds = new Set(sorted.map(n => n.id));
  const childrenMap = new Map();
  const rootIds = [];

  for (const n of sorted) {
    // Walk up parent chain to find nearest main-node ancestor.
    let pid = n.parent_id;
    while (pid && !mainIds.has(pid)) {
      const pn = explorationNodes.get(pid);
      pid = pn ? pn.parent_id : null;
    }
    if (!pid || !mainIds.has(pid)) {
      rootIds.push(n.id);
    } else {
      if (!childrenMap.has(pid)) childrenMap.set(pid, []);
      childrenMap.get(pid).push(n.id);
    }
  }

  // Layout: scatter nodes across the stage for readability.
  // Uses a spiral/zigzag pattern so nodes spread in 2D instead of a single column.
  const padding = 40;
  const nodeW = MAP_MAIN_W;
  const nodeH = MAP_MAIN_H + 14;  // taller to fit answer snippet
  const pixelPositions = new Map();

  // Determine if the graph is mostly linear (no real branching).
  const totalBranches = [...childrenMap.values()].filter(c => c.length > 1).length;
  const isLinear = totalBranches === 0;

  if (isLinear && sorted.length > 1) {
    // Scatter layout: place nodes in a zigzag grid pattern.
    // Compute grid dimensions to be roughly square.
    const cols = Math.max(2, Math.ceil(Math.sqrt(sorted.length)));
    const cellW = nodeW + MAP_H_GAP + 20;
    const cellH = nodeH + MAP_V_GAP;

    // Deterministic jitter based on node index for organic feel.
    function jitter(i, range) {
      return ((i * 7 + 13) % 17) / 17 * range - range / 2;
    }

    sorted.forEach((n, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      // Alternate row direction for a snake/zigzag path.
      const effCol = row % 2 === 0 ? col : (cols - 1 - col);
      pixelPositions.set(n.id, {
        px: padding + effCol * cellW + jitter(i, 30),
        py: padding + row * cellH + jitter(i + 5, 20),
      });
    });
  } else {
    // Tree layout for branching graphs.
    const positions = new Map();
    let nextX = 0;

    function layoutNode(id, depth) {
      const children = childrenMap.get(id) || [];
      if (children.length === 0) {
        positions.set(id, { x: nextX, y: depth });
        nextX++;
        return;
      }
      for (const cid of children) {
        layoutNode(cid, depth + 1);
      }
      const firstChild = positions.get(children[0]);
      const lastChild = positions.get(children[children.length - 1]);
      positions.set(id, { x: (firstChild.x + lastChild.x) / 2, y: depth });
    }

    for (const rid of rootIds) {
      layoutNode(rid, 0);
    }

    let maxDepth = 0;
    for (const pos of positions.values()) maxDepth = Math.max(maxDepth, pos.y);

    const rowY = [padding];
    for (let d = 1; d <= maxDepth; d++) {
      rowY.push(rowY[d - 1] + nodeH + MAP_V_GAP);
    }

    for (const [id, pos] of positions) {
      pixelPositions.set(id, {
        px: padding + pos.x * (nodeW + MAP_H_GAP),
        py: rowY[pos.y] || padding,
      });
    }
  }

  let maxPx = 0, maxPy = 0;
  for (const p of pixelPositions.values()) {
    maxPx = Math.max(maxPx, p.px + nodeW);
    maxPy = Math.max(maxPy, p.py + nodeH);
  }
  const svgW = maxPx + padding;
  const svgH = maxPy + padding;

  let svgContent = '';

  // Edges.
  for (const n of sorted) {
    // Find effective parent among main nodes.
    let pid = n.parent_id;
    while (pid && !mainIds.has(pid)) {
      const pn = explorationNodes.get(pid);
      pid = pn ? pn.parent_id : null;
    }
    if (pid && pixelPositions.has(pid) && pixelPositions.has(n.id)) {
      const ppos = pixelPositions.get(pid);
      const cpos = pixelPositions.get(n.id);
      const x1 = ppos.px + nodeW / 2;
      const y1 = ppos.py + nodeH;
      const x2 = cpos.px + nodeW / 2;
      const y2 = cpos.py;
      const midY = (y1 + y2) / 2;

      svgContent += `<path d="M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}"
        fill="none" stroke="var(--accent)" stroke-width="2" opacity="0.5"/>`;
    }
  }

  // Nodes — show user prompt + answer snippet.
  for (const n of sorted) {
    if (!pixelPositions.has(n.id)) continue;
    const pos = pixelPositions.get(n.id);
    const isCurrent = n.id === currentNodeId;

    const strokeColor = isCurrent ? 'var(--accent)' : 'var(--chrome-lo)';
    const strokeWidth = isCurrent ? 3 : 1;
    const fillColor = isCurrent ? 'var(--surface-alt)' : 'var(--surface)';

    svgContent += `<g class="map-node" data-id="${n.id}" style="cursor:pointer">`;
    svgContent += `<rect x="${pos.px}" y="${pos.py}" width="${nodeW}" height="${nodeH}"
      rx="4" ry="4" fill="${fillColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`;

    // User prompt (bold).
    const prompt = n.prompt.length > 28 ? n.prompt.slice(0, 28) + '…' : n.prompt;
    svgContent += `<text x="${pos.px + 8}" y="${pos.py + 16}" font-size="11" font-weight="700"
      fill="var(--text)" font-family="var(--font)">▸ ${escSvg(prompt)}</text>`;

    // Answer snippet (dimmed, italic — first meaningful line).
    const answerSnippet = _mapAnswerSnippet(n.answer, 30);
    svgContent += `<text x="${pos.px + 8}" y="${pos.py + 34}" font-size="10" font-style="italic"
      fill="var(--text-dim)" font-family="var(--font)">${escSvg(answerSnippet)}</text>`;

    // Metadata line: time + indicators.
    const ts = new Date(n.created_at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
    const indicators = [ts];
    if (n.answer) indicators.push(humanSize(n.answer.length));
    if (n.visual_html) indicators.push('🖼');
    svgContent += `<text x="${pos.px + 8}" y="${pos.py + 50}" font-size="9"
      fill="var(--text-dim)" font-family="var(--font)">${escSvg(indicators.join(' · '))}</text>`;

    svgContent += `</g>`;
  }

  // Apply transform.
  const t = mapTransform;
  svg.innerHTML = `<g transform="translate(${t.x},${t.y}) scale(${t.scale})">${svgContent}</g>`;
  svg.setAttribute('viewBox', `0 0 ${Math.max(svgW, 400)} ${Math.max(svgH, 300)}`);
  svg.style.width = '100%';
  svg.style.height = '100%';

  // Attach click handlers.
  for (const g of svg.querySelectorAll('.map-node')) {
    g.addEventListener('click', () => {
      const nodeId = g.dataset.id;
      if (nodeId) {
        navigateToNode(nodeId);
        const answerBtn = $(".tab-btn[data-tab='answer']");
        if (answerBtn) answerBtn.click();
      }
    });
  }
}

/** Extract a short snippet from the answer for the map node. */
function _mapAnswerSnippet(answer, maxLen) {
  if (!answer) return '—';
  // Strip markdown heading markers, bullet points, code fences.
  const lines = answer.split('\n').map(l => l.trim()).filter(l =>
    l && !l.startsWith('#') && !l.startsWith('```') && !l.startsWith('---')
  );
  const text = (lines[0] || '').replace(/^\*+\s*/, '').replace(/\*+/g, '');
  if (text.length > maxLen) return text.slice(0, maxLen) + '…';
  return text || '—';
}

function escSvg(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function initMap() {
  const container = $("#map-container");
  if (!container) return;

  // Pan via drag.
  container.addEventListener("mousedown", (e) => {
    if (e.target.closest('.map-node')) return;
    mapDragging = true;
    mapDragStart = { x: e.clientX - mapTransform.x, y: e.clientY - mapTransform.y };
    container.style.cursor = "grabbing";
  });

  container.addEventListener("mousemove", (e) => {
    if (!mapDragging) return;
    mapTransform.x = e.clientX - mapDragStart.x;
    mapTransform.y = e.clientY - mapDragStart.y;
    renderMap();
  });

  container.addEventListener("mouseup", () => {
    mapDragging = false;
    container.style.cursor = "grab";
  });

  container.addEventListener("mouseleave", () => {
    mapDragging = false;
    container.style.cursor = "grab";
  });

  // Zoom via wheel.
  container.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    mapTransform.scale = Math.max(0.2, Math.min(3, mapTransform.scale * delta));
    renderMap();
  }, { passive: false });

  // Zoom buttons.
  const zoomIn = $("#map-zoom-in");
  const zoomOut = $("#map-zoom-out");
  const fit = $("#map-fit");

  if (zoomIn) zoomIn.addEventListener("click", () => {
    mapTransform.scale = Math.min(3, mapTransform.scale * 1.2);
    renderMap();
  });
  if (zoomOut) zoomOut.addEventListener("click", () => {
    mapTransform.scale = Math.max(0.2, mapTransform.scale / 1.2);
    renderMap();
  });
  if (fit) fit.addEventListener("click", () => {
    mapTransform = { x: 0, y: 0, scale: 1 };
    renderMap();
  });
}

/* ═══════════════════════════════════════════════════════════════
   FILES TAB — themed file cards with preview
   ═══════════════════════════════════════════════════════════════ */

const FILE_ICONS = {
  '.html': '🌐', '.htm': '🌐',
  '.js': '📜', '.ts': '📜', '.mjs': '📜',
  '.py': '🐍',
  '.json': '📋', '.yaml': '📋', '.yml': '📋', '.toml': '📋',
  '.css': '🎨', '.scss': '🎨',
  '.md': '📝', '.txt': '📝', '.log': '📝',
  '.png': '🖼', '.jpg': '🖼', '.jpeg': '🖼', '.gif': '🖼', '.webp': '🖼', '.svg': '🖼', '.bmp': '🖼',
  '.mp3': '🎵', '.wav': '🎵', '.ogg': '🎵', '.flac': '🎵',
  '.mp4': '🎬', '.webm': '🎬', '.mov': '🎬',
  '.pdf': '📕',
  '.csv': '📊', '.xlsx': '📊', '.xls': '📊',
  '.zip': '📦', '.tar': '📦', '.gz': '📦',
  '.sh': '⚙', '.bash': '⚙',
};

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp']);
const HTML_EXTS = new Set(['.html', '.htm']);

function getFileIcon(ext) {
  return FILE_ICONS[ext] || '📄';
}

function humanFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

const FOLDER_ICONS = {
  media: '🖼', downloads: '⬇', scripts: '📜', output: '📤',
  showcase: '🏆', skills: '🧩', summaries: '📝', tmp: '📁',
  tools: '🔧', other: '📂',
};

function extractFolder(file) {
  // Derive folder group from logical path (e.g. "media/photo.png" → "media").
  const logical = file.logical || '';
  if (logical) {
    const seg = logical.split('/')[0];
    if (seg && seg !== logical) return seg;
  }
  // Fall back: try to parse category from physical path segments.
  const phys = file.physical || '';
  const parts = phys.replace(/\\/g, '/').split('/');
  const savedIdx = parts.lastIndexOf('saved');
  if (savedIdx >= 0 && savedIdx + 1 < parts.length) {
    return parts[savedIdx + 1];
  }
  const outputIdx = parts.lastIndexOf('output');
  if (outputIdx >= 0) return 'output';
  return file.source || 'other';
}

function renderFileCard(f) {
  const icon = getFileIcon(f.extension || '');
  const size = f.size ? humanFileSize(f.size) : '';
  const previewable = f.is_text || IMAGE_EXTS.has(f.extension) || HTML_EXTS.has(f.extension);
  return `
    <div class="file-card" data-path="${esc(f.physical)}" data-ext="${esc(f.extension || '')}" data-text="${f.is_text ? '1' : '0'}">
      <span class="file-icon">${icon}</span>
      <div class="file-info">
        <div class="file-name">${esc(f.filename)}</div>
        <div class="file-meta">${esc(size)}${f.logical ? ' · ' + esc(f.logical) : ''}</div>
      </div>
      <div class="file-actions">
        ${previewable ? '<button class="file-action-btn file-preview-btn" title="Preview file"><span class="file-action-icon">👁</span><span class="file-action-label">Preview</span></button>' : ''}
        <a class="file-action-btn" href="/api/files/view?path=${encodeURIComponent(f.physical)}" target="_blank" title="Open in new tab"><span class="file-action-icon">↗</span><span class="file-action-label">Open</span></a>
        <a class="file-action-btn" href="/api/files/download?path=${encodeURIComponent(f.physical)}" title="Download file"><span class="file-action-icon">⬇</span><span class="file-action-label">Download</span></a>
      </div>
    </div>
  `;
}

async function loadFiles() {
  const el = $("#files-content");
  const countEl = $("#files-count");

  try {
    // Use session-scoped endpoint when session is known, fall back to global.
    const url = sessionId
      ? `/api/files/session/${encodeURIComponent(sessionId)}`
      : "/api/files";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const files = await res.json();

    if (!files || files.length === 0) {
      _filesCache = [];
      el.innerHTML = '<span class="output-placeholder">No files yet.</span>';
      el.classList.add("output-empty");
      countEl.textContent = '';
      return;
    }

    _filesCache = files;

    // Apply current search filter if any.
    const searchInput = $("#files-search");
    const q = searchInput ? searchInput.value.trim() : '';
    if (q) {
      filterFiles(q);
      return;
    }

    countEl.textContent = `${files.length} file${files.length !== 1 ? 's' : ''}`;
    el.classList.remove("output-empty");

    // Group files by folder category.
    const groups = new Map();
    for (const f of files) {
      const folder = extractFolder(f);
      if (!groups.has(folder)) groups.set(folder, []);
      groups.get(folder).push(f);
    }

    // Sort groups: put common categories first, then alpha.
    const order = ['output', 'media', 'downloads', 'scripts', 'tools', 'skills', 'summaries', 'showcase', 'tmp'];
    const sortedKeys = Array.from(groups.keys()).sort((a, b) => {
      const ia = order.indexOf(a);
      const ib = order.indexOf(b);
      if (ia >= 0 && ib >= 0) return ia - ib;
      if (ia >= 0) return -1;
      if (ib >= 0) return 1;
      return a.localeCompare(b);
    });

    let html = '';
    for (const folder of sortedKeys) {
      const groupFiles = groups.get(folder);
      const folderIcon = FOLDER_ICONS[folder] || FOLDER_ICONS.other;
      const collapsed = '';  // start expanded
      html += `
        <div class="file-group${collapsed}">
          <div class="file-group-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <span class="file-group-icon">${folderIcon}</span>
            <span class="file-group-name">${esc(folder)}</span>
            <span class="file-group-count">${groupFiles.length}</span>
            <span class="file-group-chevron">▾</span>
          </div>
          <div class="file-group-body">
            ${groupFiles.map(renderFileCard).join('')}
          </div>
        </div>
      `;
    }

    el.innerHTML = html;

    // Attach preview handlers.
    for (const btn of el.querySelectorAll('.file-preview-btn')) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = btn.closest('.file-card');
        openFilePreview(card.dataset.path, card.dataset.ext, card.dataset.text === '1');
      });
    }

  } catch (err) {
    console.warn("Failed to load files:", err);
    el.innerHTML = `<span class="output-placeholder">Failed to load files: ${esc(err.message)}</span>`;
    el.classList.add("output-empty");
  }
}

let _filesCache = [];   // cached file list for search filtering

function filterFiles(query) {
  const el = $("#files-content");
  if (!_filesCache.length) return;

  const q = (query || '').trim().toLowerCase();
  const filtered = q
    ? _filesCache.filter(f =>
        (f.filename || '').toLowerCase().includes(q) ||
        (f.extension || '').toLowerCase().includes(q) ||
        (f.logical || '').toLowerCase().includes(q))
    : _filesCache;

  const countEl = $("#files-count");
  if (q) {
    countEl.textContent = `${filtered.length}/${_filesCache.length} files`;
  } else {
    countEl.textContent = `${_filesCache.length} file${_filesCache.length !== 1 ? 's' : ''}`;
  }

  if (filtered.length === 0) {
    el.innerHTML = `<span class="output-placeholder">No files matching "${esc(q)}"</span>`;
    el.classList.add("output-empty");
    return;
  }

  el.classList.remove("output-empty");

  // Group filtered files by folder.
  const groups = new Map();
  for (const f of filtered) {
    const folder = extractFolder(f);
    if (!groups.has(folder)) groups.set(folder, []);
    groups.get(folder).push(f);
  }
  const order = ['output', 'media', 'downloads', 'scripts', 'tools', 'skills', 'summaries', 'showcase', 'tmp'];
  const sortedKeys = Array.from(groups.keys()).sort((a, b) => {
    const ia = order.indexOf(a); const ib = order.indexOf(b);
    if (ia >= 0 && ib >= 0) return ia - ib;
    if (ia >= 0) return -1;
    if (ib >= 0) return 1;
    return a.localeCompare(b);
  });

  let html = '';
  for (const folder of sortedKeys) {
    const groupFiles = groups.get(folder);
    const folderIcon = FOLDER_ICONS[folder] || FOLDER_ICONS.other;
    html += `
      <div class="file-group">
        <div class="file-group-header" onclick="this.parentElement.classList.toggle('collapsed')">
          <span class="file-group-icon">${folderIcon}</span>
          <span class="file-group-name">${esc(folder)}</span>
          <span class="file-group-count">${groupFiles.length}</span>
          <span class="file-group-chevron">▾</span>
        </div>
        <div class="file-group-body">
          ${groupFiles.map(renderFileCard).join('')}
        </div>
      </div>
    `;
  }
  el.innerHTML = html;

  // Re-attach preview handlers.
  for (const btn of el.querySelectorAll('.file-preview-btn')) {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const card = btn.closest('.file-card');
      openFilePreview(card.dataset.path, card.dataset.ext, card.dataset.text === '1');
    });
  }
}

function initFiles() {
  const refreshBtn = $("#files-refresh-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => loadFiles());
  }
  const searchInput = $("#files-search");
  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => filterFiles(searchInput.value), 150);
    });
  }
}

/* ── File preview modal ──────────────────────────────────────── */

async function openFilePreview(path, ext, isText) {
  const modal = $("#file-preview-modal");
  const title = $("#preview-title");
  const body = $("#preview-body");
  const filename = path.split('/').pop();

  title.textContent = filename;
  body.innerHTML = '<div class="output-processing"><span class="processing-dots">⣾</span> Loading preview...</div>';
  modal.classList.remove("hidden");

  const MD_EXTS = new Set(['.md', '.markdown', '.mdown', '.mkd']);

  try {
    if (IMAGE_EXTS.has(ext)) {
      // Image: render inline.
      body.innerHTML = `<img src="/api/files/view?path=${encodeURIComponent(path)}" alt="${esc(filename)}">`;
    } else if (HTML_EXTS.has(ext)) {
      // HTML: render in sandboxed iframe.
      body.innerHTML = `<iframe src="/api/files/view?path=${encodeURIComponent(path)}" sandbox="allow-scripts"></iframe>`;
    } else if (MD_EXTS.has(ext)) {
      // Markdown: fetch content and render as formatted HTML.
      const res = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      body.innerHTML = `<div class="md-preview">${markdownToHtml(data.content || '(empty)')}</div>`;
    } else if (isText) {
      // Text: fetch content and render in pre block.
      const res = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      body.innerHTML = `<pre>${esc(data.content || '(empty)')}</pre>`;
    } else {
      body.innerHTML = '<span class="output-placeholder">Preview not available for this file type.</span>';
    }
  } catch (err) {
    body.innerHTML = `<span class="output-placeholder">Preview failed: ${esc(err.message)}</span>`;
  }
}

function initFilePreview() {
  const modal = $("#file-preview-modal");
  if (!modal) return;

  // Close on backdrop click.
  const backdrop = modal.querySelector(".modal-backdrop");
  if (backdrop) backdrop.addEventListener("click", () => modal.classList.add("hidden"));

  // Close on ✕ button.
  const closeBtn = modal.querySelector(".modal-close");
  if (closeBtn) closeBtn.addEventListener("click", () => modal.classList.add("hidden"));

  // Close on Escape.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      modal.classList.add("hidden");
    }
  });
}

/* ═══════════════════════════════════════════════════════════════
   MODEL & TIER SELECTORS
   ═══════════════════════════════════════════════════════════════ */

async function loadModels() {
  try {
    const res = await fetch("/api/orchestrator/models");
    if (!res.ok) return;
    const data = await res.json();
    availableModels = data.models || [];
    renderModelGrid();
    updateModelBtnLabel();
    console.log(`[Computer] Loaded ${availableModels.length} available models`);
  } catch (e) {
    console.warn("Failed to load models:", e);
  }
}

function providerIcon(provider) {
  const p = (provider || "").toLowerCase();
  if (p.includes("anthropic") || p.includes("claude")) return "🟣";
  if (p.includes("openai") || p.includes("gpt")) return "🟢";
  if (p.includes("google") || p.includes("gemini") || p.includes("vertex")) return "🔵";
  if (p.includes("groq")) return "🟠";
  if (p.includes("mistral")) return "🔴";
  if (p.includes("ollama") || p.includes("local")) return "⚪";
  if (p.includes("openrouter")) return "🟡";
  return "⬜";
}

function renderModelGrid() {
  const grid = $("#model-grid");
  grid.innerHTML = "";

  // Default option
  const defCard = document.createElement("div");
  defCard.className = "model-card" + (!selectedModel ? " selected" : "");
  defCard.dataset.modelId = "";
  defCard.innerHTML =
    '<div class="model-card-icon">⚙️</div>' +
    '<div class="model-card-info">' +
      '<div class="model-card-name">Default Model</div>' +
      '<div class="model-card-detail">Use the session\'s default model for all operations</div>' +
    '</div>' +
    (!selectedModel ? '<div class="model-card-badges"><span class="model-badge active">Active</span></div>' : '');
  defCard.addEventListener("click", () => selectModel(""));
  grid.appendChild(defCard);

  for (const m of availableModels) {
    if (m.id === "default" && availableModels.length === 1) continue; // skip if it's the only fallback
    const card = document.createElement("div");
    card.className = "model-card" + (m.id === selectedModel ? " selected" : "");
    card.dataset.modelId = m.id;

    const name = m.description || m.id;
    const detail = `${m.provider} / ${m.model}`;
    const badges = [];
    if (m.reasoning_level) badges.push(m.reasoning_level);
    if (m.model_type && m.model_type !== "llm") badges.push(m.model_type);
    if (m.id === selectedModel) badges.push("Active");

    card.innerHTML =
      '<div class="model-card-icon">' + providerIcon(m.provider) + '</div>' +
      '<div class="model-card-info">' +
        '<div class="model-card-name">' + esc(name) + '</div>' +
        '<div class="model-card-detail">' + esc(detail) + '</div>' +
      '</div>' +
      (badges.length
        ? '<div class="model-card-badges">' +
          badges.map(b => '<span class="model-badge' + (b === "Active" ? " active" : "") + '">' + esc(b) + '</span>').join("") +
          '</div>'
        : '');

    card.addEventListener("click", () => selectModel(m.id));
    grid.appendChild(card);
  }
}

function selectModel(modelId) {
  selectedModel = modelId;
  localStorage.setItem("computer-model", selectedModel);

  // Update visuals.
  renderModelGrid();
  updateModelBtnLabel();

  // Also set the model for the chat/agent session via WebSocket.
  if (modelId && connected) {
    send({ type: "set_model", selector: modelId });
  } else if (!modelId && connected) {
    // Reset to default — send the default model id.
    send({ type: "set_model", selector: "default" });
  }

  logEntry("system", `Model: ${selectedModel || '(default)'}`);
  closeModelModal();
}

function updateModelBtnLabel() {
  const label = $("#model-btn-label");
  if (!selectedModel) {
    label.textContent = "Default";
    return;
  }
  const m = availableModels.find(m => m.id === selectedModel);
  label.textContent = m ? (m.description || m.id) : selectedModel;
}

function openModelModal() {
  renderModelGrid();
  $("#model-modal").classList.add("active");
}

function closeModelModal() {
  $("#model-modal").classList.remove("active");
}

function initModelModal() {
  $("#model-btn").addEventListener("click", openModelModal);
  $("#model-modal .modal-close").addEventListener("click", closeModelModal);
  $("#model-modal .modal-backdrop").addEventListener("click", closeModelModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("#model-modal").classList.contains("active")) {
      closeModelModal();
    }
  });
}

function initSelectors() {
  // Model modal (replaces old <select>).
  initModelModal();

  // Tier selector.
  const tierSel = $("#tier-selector");
  if (tierSel) {
    tierSel.value = selectedTokenTier;
    tierSel.addEventListener("change", (e) => {
      selectedTokenTier = e.target.value;
      localStorage.setItem("computer-token-tier", selectedTokenTier);
      logEntry("system", `Token tier: ${selectedTokenTier}`);
    });
  }
}

/* ── Markdown to HTML (basic) ────────────────────────────────── */

function markdownToHtml(md) {
  if (!md) return "";

  // Pre-process: extract tables before escaping (they need raw pipe chars).
  const tableBlocks = [];
  const TABLE_PLACEHOLDER = '\x00TABLE_';
  md = md.replace(
    /(^[ \t]*\|.+\|[ \t]*\n)([ \t]*\|[\s:]*[-:]+[-|\s:]*\|[ \t]*\n)((?:[ \t]*\|.+\|[ \t]*\n?)*)/gm,
    (match, headerLine, sepLine, bodyLines) => {
      const idx = tableBlocks.length;
      tableBlocks.push({ headerLine, sepLine, bodyLines });
      return TABLE_PLACEHOLDER + idx + '\n';
    }
  );

  let html = esc(md);

  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>");

  html = html.replace(/`(.+?)`/g, "<code>$1</code>");

  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");

  // Restore table blocks as HTML tables.
  for (let i = 0; i < tableBlocks.length; i++) {
    const tb = tableBlocks[i];
    const headerCells = parseTableRow(tb.headerLine);
    const bodyRows = tb.bodyLines.trim()
      ? tb.bodyLines.trim().split('\n').map(parseTableRow)
      : [];

    // Parse alignment from separator row.
    const sepCells = parseTableRow(tb.sepLine);
    const aligns = sepCells.map(s => {
      s = s.trim();
      if (s.startsWith(':') && s.endsWith(':')) return 'center';
      if (s.endsWith(':')) return 'right';
      return 'left';
    });

    let tableHtml = '<table><thead><tr>';
    headerCells.forEach((cell, ci) => {
      const align = aligns[ci] || 'left';
      tableHtml += `<th style="text-align:${align}">${esc(cell)}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';
    for (const row of bodyRows) {
      tableHtml += '<tr>';
      row.forEach((cell, ci) => {
        const align = aligns[ci] || 'left';
        tableHtml += `<td style="text-align:${align}">${esc(cell)}</td>`;
      });
      tableHtml += '</tr>';
    }
    tableHtml += '</tbody></table>';

    html = html.replace(TABLE_PLACEHOLDER + i, tableHtml);
  }

  html = html.replace(/^(?!<[hupblot])([\s\S]+?)(?=\n\n|$)/gm, (match) => {
    const trimmed = match.trim();
    if (!trimmed || trimmed.startsWith("<")) return match;
    return `<p>${trimmed}</p>`;
  });

  html = html.replace(/\n{2,}/g, "\n");

  return html;
}

function parseTableRow(line) {
  line = line.trim();
  if (line.startsWith('|')) line = line.slice(1);
  if (line.endsWith('|')) line = line.slice(0, -1);
  return line.split('|').map(c => c.trim());
}

/* ── Utilities ───────────────────────────────────────────────── */

function humanTime(ms) {
  if (ms < 1000) return ms + "ms";
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(1);
  return m + "m " + rem + "s";
}

function humanSize(chars) {
  if (chars < 1000) return chars + " chars";
  return (chars / 1000).toFixed(1) + "k chars";
}

function esc(s) {
  if (!s) return "";
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ── Auto-resize textarea ────────────────────────────────────── */

function initTextarea() {
  const textarea = $("#input-box");

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  });

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  $("#send-btn").addEventListener("click", () => handleSend());
}

/* ── Clear log button ────────────────────────────────────────── */

function initClearLog() {
  const clearBtn = $(".win-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      $("#log-content").innerHTML = "";
      logEntry("system", "Log cleared");
    });
  }
}

/* ── Processing animation ────────────────────────────────────── */

const SPINNER_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
let spinnerIdx = 0;

function initSpinner() {
  setInterval(() => {
    const els = $$(".processing-dots");
    for (const el of els) {
      spinnerIdx = (spinnerIdx + 1) % SPINNER_FRAMES.length;
      el.textContent = SPINNER_FRAMES[spinnerIdx];
    }
  }, 100);
}

/* ── Attachment / paste / drag-drop ──────────────────────────── */

function resizeImageFile(file, maxSide) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const w = img.width, h = img.height;
      const scale = (w > maxSide || h > maxSide) ? maxSide / Math.max(w, h) : 1;
      const nw = Math.round(w * scale), nh = Math.round(h * scale);
      const canvas = document.createElement("canvas");
      canvas.width = nw; canvas.height = nh;
      canvas.getContext("2d").drawImage(img, 0, 0, nw, nh);
      canvas.toBlob((blob) => {
        if (!blob) { resolve(file); return; }
        const outName = file.name.replace(/\.[^.]+$/, ".jpg");
        resolve(new File([blob], outName, { type: "image/jpeg" }));
      }, "image/jpeg", 0.85);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("load")); };
    img.src = url;
  });
}

async function uploadFile(file) {
  if (!file) return;
  const ext = file.name.split(".").pop().toLowerCase();
  const isImage = _IMAGE_EXTS.indexOf(ext) !== -1;
  const isData = (ext === "csv" || ext === "xlsx");

  if (!isImage && !isData) {
    logEntry("error", "Supported files: images (.png, .jpg, .jpeg, .webp, .gif, .bmp) and data (.csv, .xlsx).");
    return;
  }

  const attachBtn = $("#attach-btn");

  if (isImage) {
    attachBtn.disabled = true;
    attachBtn.textContent = "⏳";
    try { file = await resizeImageFile(file, 1024); } catch (_) {}
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("/api/image/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        logEntry("error", "Upload failed: " + (data.error || "Unknown error"));
      } else {
        showAttachmentChip(file.name, data.path, false);
      }
    } catch (err) {
      logEntry("error", "Upload failed: " + err.message);
    } finally {
      attachBtn.disabled = false;
      attachBtn.textContent = "📎";
      $("#file-input").value = "";
    }
    return;
  }

  // Data file
  attachBtn.disabled = true;
  attachBtn.textContent = "⏳";
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch("/api/file/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      logEntry("error", "Upload failed: " + (data.error || "Unknown error"));
    } else {
      showAttachmentChip(file.name, data.path, true);
    }
  } catch (err) {
    logEntry("error", "Upload failed: " + err.message);
  } finally {
    attachBtn.disabled = false;
    attachBtn.textContent = "📎";
    $("#file-input").value = "";
  }
}

function showAttachmentChip(filename, path, isFile) {
  if (isFile) { pendingFilePath = path; } else { pendingImagePath = path; }
  const bar = $("#attachment-bar");
  const chip = $("#attachment-chip");
  chip.innerHTML = "";
  const span = document.createElement("span");
  span.className = "attachment-name";
  span.textContent = (isFile ? "📄 " : "🖼️ ") + filename;
  chip.appendChild(span);
  const removeBtn = document.createElement("button");
  removeBtn.className = "attachment-remove";
  removeBtn.textContent = "✕";
  removeBtn.title = "Remove attachment";
  removeBtn.addEventListener("click", clearAttachment);
  chip.appendChild(removeBtn);
  bar.classList.remove("hidden");
}

function clearAttachment() {
  pendingImagePath = null;
  pendingFilePath = null;
  const bar = $("#attachment-bar");
  if (bar) bar.classList.add("hidden");
  const chip = $("#attachment-chip");
  if (chip) chip.innerHTML = "";
  const fi = $("#file-input");
  if (fi) fi.value = "";
}

function initAttachments() {
  const attachBtn = $("#attach-btn");
  const fileInput = $("#file-input");
  const inputBox = $("#input-box");
  const workspace = $("#workspace");

  attachBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  // Paste image from clipboard.
  inputBox.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image/") === 0) {
        e.preventDefault();
        let file = items[i].getAsFile();
        if (file) {
          if (!file.name || file.name === "image.png") {
            const ext = (file.type.split("/")[1] || "png") === "jpeg" ? "jpg" : (file.type.split("/")[1] || "png");
            const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
            file = new File([file], "pasted-" + ts + "." + ext, { type: file.type });
          }
          uploadFile(file);
        }
        return;
      }
    }
  });

  // Drag & drop on workspace.
  let dragCounter = 0;
  workspace.addEventListener("dragenter", (e) => { e.preventDefault(); dragCounter++; });
  workspace.addEventListener("dragover", (e) => { e.preventDefault(); });
  workspace.addEventListener("dragleave", (e) => { e.preventDefault(); dragCounter--; });
  workspace.addEventListener("drop", (e) => {
    e.preventDefault();
    dragCounter = 0;
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });
}

/* ── Folder browser ─────────────────────────────────────────── */

function openFolderModal() {
  $("#folder-modal").classList.add("active");
  loadFolderList();
  browseFolderDir("");
  initFolderTabs();
}

function closeFolderModal() {
  $("#folder-modal").classList.remove("active");
}

async function initFolderTabs() {
  // Check for Windows drives.
  try {
    const res = await fetch("/api/drives");
    const data = await res.json();
    const bar = $("#folder-drive-bar");
    if (data.drives && data.drives.length > 0) {
      bar.classList.remove("hidden");
      bar.innerHTML = data.drives.map(d =>
        '<button class="folder-drive-btn" data-drive="' + esc(d) + '">' + esc(d) + '</button>'
      ).join("");
      bar.querySelectorAll(".folder-drive-btn").forEach(btn => {
        btn.addEventListener("click", () => browseFolderDir(btn.dataset.drive + "\\"));
      });
    } else {
      bar.classList.add("hidden");
    }
  } catch (_) {
    $("#folder-drive-bar").classList.add("hidden");
  }

  // Check for GDrive availability.
  try {
    const res = await fetch("/api/gws-status");
    const data = await res.json();
    _gwsAvailable = !!data.available;
    $("#folder-tab-gdrive-btn").style.display = _gwsAvailable ? "" : "none";
  } catch (_) {
    _gwsAvailable = false;
    $("#folder-tab-gdrive-btn").style.display = "none";
  }

  // Tab switching.
  for (const tab of $$(".folder-modal-tab")) {
    tab.onclick = () => {
      for (const t of $$(".folder-modal-tab")) t.classList.remove("active");
      tab.classList.add("active");
      const target = tab.dataset.tab;
      $("#folder-tab-local").classList.toggle("hidden", target !== "local");
      $("#folder-tab-gdrive").classList.toggle("hidden", target !== "gdrive");
      if (target === "gdrive") loadGdriveTab();
    };
  }
}

async function loadFolderList() {
  const list = $("#folder-list");
  try {
    const res = await fetch("/api/read-folders");
    const data = await res.json();
    const dirs = data.dirs || [];
    if (!dirs.length) {
      list.innerHTML = '<div class="folder-empty">No extra folders configured</div>';
      return;
    }
    list.innerHTML = dirs.map(d => {
      const cls = d.exists ? "" : " missing";
      return '<div class="folder-entry' + cls + '">' +
        '<span class="folder-entry-path" title="' + esc(d.resolved) + '">' + esc(d.resolved) + '</span>' +
        (!d.exists ? '<span class="folder-entry-warn" title="Directory does not exist">⚠</span>' : '') +
        '<button class="folder-entry-remove" data-path="' + esc(d.resolved) + '" title="Remove">✕</button>' +
        '</div>';
    }).join("");
    list.querySelectorAll(".folder-entry-remove").forEach(btn => {
      btn.addEventListener("click", () => removeFolderDir(btn.dataset.path));
    });
  } catch (_) {
    list.innerHTML = '<div class="folder-empty">Failed to load</div>';
  }
}

async function browseFolderDir(path) {
  const dirsEl = $("#folder-dirs");
  dirsEl.innerHTML = '<div class="folder-empty">Loading...</div>';
  try {
    const url = "/api/browse" + (path ? "?path=" + encodeURIComponent(path) : "");
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) { dirsEl.innerHTML = '<div class="folder-empty">' + esc(data.error) + '</div>'; return; }
    folderCurrentPath = data.path;
    $("#folder-selected-path").textContent = data.path;
    const addBtn = $("#folder-add-btn");
    addBtn.disabled = !!data.blocked;

    renderFolderBreadcrumb(data.path);

    let html = "";
    if (data.parent) {
      html += '<div class="folder-dir-item parent" data-path="' + esc(data.parent) + '"><span class="folder-dir-icon">⬆</span> ..</div>';
    }
    if (!data.dirs.length && !data.parent) {
      html += '<div class="folder-empty">No subdirectories</div>';
    }
    data.dirs.forEach(name => {
      const sep = data.path.includes("\\") ? "\\" : "/";
      const full = data.path + (data.path.endsWith(sep) ? "" : sep) + name;
      html += '<div class="folder-dir-item" data-path="' + esc(full) + '"><span class="folder-dir-icon">📁</span> ' + esc(name) + '</div>';
    });
    dirsEl.innerHTML = html;
    dirsEl.querySelectorAll(".folder-dir-item").forEach(el => {
      el.addEventListener("click", () => browseFolderDir(el.dataset.path));
    });
  } catch (e) {
    dirsEl.innerHTML = '<div class="folder-empty">Error: ' + esc(e.message) + '</div>';
  }
}

function renderFolderBreadcrumb(fullPath) {
  const bc = $("#folder-breadcrumb");
  const sep = fullPath.includes("\\") ? "\\" : "/";
  const parts = fullPath.split(sep).filter(Boolean);
  let html = "", built = "";
  if (fullPath.startsWith("/")) {
    html += '<span class="folder-bc-item" data-path="/">/</span>';
    built = "/";
  }
  parts.forEach((p, i) => {
    if (i === 0 && !fullPath.startsWith("/")) { built = p + sep; }
    else { built += (built.endsWith(sep) ? "" : sep) + p; }
    if (i > 0 || fullPath.startsWith("/")) html += '<span class="folder-bc-sep">' + sep + '</span>';
    html += '<span class="folder-bc-item" data-path="' + esc(built) + '">' + esc(p) + '</span>';
  });
  bc.innerHTML = html;
  bc.querySelectorAll(".folder-bc-item").forEach(el => {
    el.addEventListener("click", () => browseFolderDir(el.dataset.path));
  });
}

async function addCurrentFolder() {
  if (!folderCurrentPath) return;
  const btn = $("#folder-add-btn");
  btn.disabled = true; btn.textContent = "Adding...";
  try {
    const res = await fetch("/api/read-folders", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: folderCurrentPath }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) { logEntry("error", "Add folder failed: " + (data.error || "Unknown")); }
    else { logEntry("system", "Read folder added: " + data.path); loadFolderList(); }
  } catch (e) { logEntry("error", "Add folder failed: " + e.message); }
  finally { btn.disabled = false; btn.textContent = "Add this folder"; }
}

async function removeFolderDir(path) {
  try {
    const res = await fetch("/api/read-folders", {
      method: "DELETE", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) { logEntry("error", "Remove folder failed: " + (data.error || "Unknown")); }
    else { logEntry("system", "Read folder removed: " + data.path); loadFolderList(); }
  } catch (e) { logEntry("error", "Remove folder failed: " + e.message); }
}

// ── Google Drive folder browser ──

async function loadGdriveTab() {
  loadGdriveFolderList();
  browseGdrive("root", "My Drive");
}

async function loadGdriveFolderList() {
  const list = $("#gdrive-list");
  try {
    const res = await fetch("/api/read-folders/gdrive");
    const data = await res.json();
    const folders = data.folders || [];
    if (!folders.length) { list.innerHTML = '<div class="folder-empty">No Google Drive folders configured</div>'; return; }
    list.innerHTML = folders.map(f =>
      '<div class="folder-entry"><span class="folder-entry-path" title="ID: ' + esc(f.id) + '">' +
      esc(f.name) + '</span><button class="folder-entry-remove" data-id="' + esc(f.id) + '" title="Remove">✕</button></div>'
    ).join("");
    list.querySelectorAll(".folder-entry-remove").forEach(btn => {
      btn.addEventListener("click", () => removeGdriveFolder(btn.dataset.id));
    });
  } catch (_) { list.innerHTML = '<div class="folder-empty">Failed to load</div>'; }
}

async function browseGdrive(folderId, folderName) {
  const dirsEl = $("#gdrive-dirs");
  dirsEl.innerHTML = '<div class="folder-empty">Loading...</div>';
  _gdriveCurrent = { id: folderId, name: folderName };
  $("#gdrive-selected-name").textContent = folderName;
  $("#gdrive-add-btn").disabled = (folderId === "root");

  if (folderId === "root") { _gdriveBcStack = [{ id: "root", name: "My Drive" }]; }
  else {
    const idx = _gdriveBcStack.findIndex(b => b.id === folderId);
    if (idx >= 0) _gdriveBcStack = _gdriveBcStack.slice(0, idx + 1);
    else _gdriveBcStack.push({ id: folderId, name: folderName });
  }
  renderGdriveBreadcrumb();

  try {
    const res = await fetch("/api/read-folders/gdrive/browse?folder_id=" + encodeURIComponent(folderId));
    const data = await res.json();
    if (data.error) { dirsEl.innerHTML = '<div class="folder-empty">' + esc(data.error) + '</div>'; return; }
    const folders = data.folders || [];
    const sharedDrives = data.shared_drives || [];
    let html = "";
    if (_gdriveBcStack.length > 1) {
      const parent = _gdriveBcStack[_gdriveBcStack.length - 2];
      html += '<div class="folder-dir-item parent" data-gid="' + esc(parent.id) + '" data-gname="' + esc(parent.name) + '"><span class="folder-dir-icon">⬆</span> ..</div>';
    }
    if (sharedDrives.length) {
      html += '<div class="folder-section-lbl">Shared Drives</div>';
      sharedDrives.forEach(d => {
        html += '<div class="folder-dir-item" data-gid="' + esc(d.id) + '" data-gname="' + esc(d.name) + '"><span class="folder-dir-icon">📤</span> ' + esc(d.name) + '</div>';
      });
      if (folders.length) html += '<div class="folder-section-lbl">My Drive</div>';
    }
    if (!folders.length && !sharedDrives.length && _gdriveBcStack.length <= 1) {
      html += '<div class="folder-empty">No subfolders</div>';
    }
    folders.forEach(f => {
      html += '<div class="folder-dir-item" data-gid="' + esc(f.id) + '" data-gname="' + esc(f.name) + '"><span class="folder-dir-icon">📁</span> ' + esc(f.name) + '</div>';
    });
    dirsEl.innerHTML = html;
    dirsEl.querySelectorAll(".folder-dir-item").forEach(el => {
      el.addEventListener("click", () => browseGdrive(el.dataset.gid, el.dataset.gname));
    });
  } catch (e) { dirsEl.innerHTML = '<div class="folder-empty">Error: ' + esc(e.message) + '</div>'; }
}

function renderGdriveBreadcrumb() {
  const bc = $("#gdrive-breadcrumb");
  let html = "";
  _gdriveBcStack.forEach((item, i) => {
    if (i > 0) html += '<span class="folder-bc-sep">/</span>';
    html += '<span class="folder-bc-item" data-gid="' + esc(item.id) + '" data-gname="' + esc(item.name) + '">' + esc(item.name) + '</span>';
  });
  bc.innerHTML = html;
  bc.querySelectorAll(".folder-bc-item").forEach(el => {
    el.addEventListener("click", () => browseGdrive(el.dataset.gid, el.dataset.gname));
  });
}

async function addGdriveFolder() {
  if (_gdriveCurrent.id === "root") return;
  const btn = $("#gdrive-add-btn");
  btn.disabled = true; btn.textContent = "Adding...";
  try {
    const res = await fetch("/api/read-folders/gdrive", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: _gdriveCurrent.id, name: _gdriveCurrent.name }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) { logEntry("error", "Add GDrive folder failed: " + (data.error || "Unknown")); }
    else { logEntry("system", "Google Drive folder added: " + data.name); loadGdriveFolderList(); }
  } catch (e) { logEntry("error", "Add GDrive folder failed: " + e.message); }
  finally { btn.disabled = false; btn.textContent = "Add this folder"; }
}

async function removeGdriveFolder(id) {
  try {
    const res = await fetch("/api/read-folders/gdrive", {
      method: "DELETE", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) { logEntry("error", "Remove GDrive folder failed: " + (data.error || "Unknown")); }
    else { logEntry("system", "Google Drive folder removed"); loadGdriveFolderList(); }
  } catch (e) { logEntry("error", "Remove GDrive folder failed: " + e.message); }
}

function initFolderModal() {
  const folderBtn = $("#folder-btn");
  folderBtn.addEventListener("click", openFolderModal);

  // Close handlers.
  $("#folder-modal .modal-close").addEventListener("click", closeFolderModal);
  $("#folder-modal .modal-backdrop").addEventListener("click", closeFolderModal);

  // Add buttons.
  $("#folder-add-btn").addEventListener("click", addCurrentFolder);
  $("#gdrive-add-btn").addEventListener("click", addGdriveFolder);
}

/* ═══════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  // Restore theme.
  const saved = localStorage.getItem("computer-theme");
  if (saved && saved !== currentTheme) {
    setTheme(saved);
  }

  // Theme modal.
  initThemeModal();

  // Init subsystems.
  initTextarea();
  initTabs();
  initResize();
  initClearLog();
  initSpinner();
  initFullscreen();
  initMap();
  initFiles();
  initFilePreview();
  initSelectors();
  initAttachments();
  initFolderModal();

  // Load available models for the model selector.
  loadModels();

  // Listen for postMessage from visual iframe.
  window.addEventListener("message", handleIframeMessage);

  // Boot sequence.
  await runBoot();

  // Connect to agent.
  connect();

  // Focus input.
  $("#input-box").focus();
});
