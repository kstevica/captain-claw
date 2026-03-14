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

/* ── Exploration state ───────────────────────────────────────── */

const MAX_EXPLORATION_NODES = 200;

let explorationNodes = new Map();  // id -> ExplorationNode
let currentNodeId = null;
let pendingExploration = null;     // { parentId, edgeLabel, source }
let sessionId = null;              // set on welcome
let exploreDebounceTimer = null;

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
  } catch (e) {
    console.warn('Failed to load exploration history:', e);
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

  // Hide the bar.
  bar.classList.remove("active");
  bar.style.display = "none";

  // Send whatever is in the input box (user may have edited it).
  handleSend();
}

function cancelExploration() {
  const bar = $("#explore-confirm-bar");
  if (bar) {
    bar.classList.remove("active");
    bar.style.display = "none";
  }
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

  // Update map highlighting.
  renderMap();

  logEntry("system", `Navigated to: ${node.prompt.slice(0, 60)}${node.prompt.length > 60 ? '…' : ''}`);
}

/* ── Boot sequences per theme ────────────────────────────────── */

const BOOT_SEQUENCES = {
  amiga: [
    { html: '<div class="boot-amiga-hand">🖴</div><div class="boot-text">Insert Workbench disk</div>', delay: 800 },
    { html: '<div class="boot-amiga-hand">🖴</div><div class="boot-text">Loading Kickstart...</div>', delay: 600 },
    { html: '<div class="boot-amiga-check">✓</div><div class="boot-text">Amiga Workbench 1.3</div>', delay: 700 },
  ],
  atarist: [
    { html: '<div style="font-size:32px">⚡</div><div class="boot-text">Atari ST — TOS 1.04</div>', delay: 600 },
    { html: '<div style="font-size:32px">⚡</div><div class="boot-text">GEM Desktop loading...</div>', delay: 500 },
  ],
  c64: [
    { html: '<div style="font-size:13px;text-align:left;font-family:Silkscreen,monospace;line-height:1.8">**** COMMODORE 64 GEOS V2.0 ****<br><br>64K RAM SYSTEM  38911 BASIC BYTES FREE<br><br>LOADING DESKTOP...</div>', delay: 1200 },
  ],
  mac: [
    { html: '<div style="font-size:48px">🙂</div><div class="boot-text">Welcome to Macintosh</div>', delay: 1000 },
    { html: '<div style="font-size:48px">🙂</div><div class="boot-text">Loading System 7...</div>', delay: 600 },
  ],
  win31: [
    { html: '<div style="font-size:13px;text-align:left;font-family:monospace;line-height:1.8">Microsoft(R) Windows(TM)<br>Version 3.1<br><br>Copyright (C) Microsoft Corp. 1985-1992<br><br>Loading Program Manager...</div>', delay: 1200 },
  ],
  hacker: [
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...</div>', delay: 1000 },
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...<br><br>The Matrix has you...</div>', delay: 1000 },
    { html: '<div style="font-size:12px;text-align:left;font-family:monospace;line-height:1.6;text-shadow:0 0 8px #00ff41">Wake up, Neo...<br><br>The Matrix has you...<br><br>Follow the white rabbit.<br><br>Knock, knock, Neo.</div>', delay: 800 },
    { html: '<div style="font-size:14px;font-family:monospace;text-shadow:0 0 10px #00ff41">root@computer:~# <span style="animation:boot-blink 0.7s step-end infinite">_</span></div>', delay: 600 },
  ],
  modern: [
    { html: '<div style="font-size:28px;font-weight:300;letter-spacing:2px;opacity:0.7">Computer</div>', delay: 600 },
    { html: '<div style="font-size:28px;font-weight:300;letter-spacing:2px">Computer</div><div style="margin-top:16px;width:120px;height:3px;border-radius:2px;background:#2a2e3d;overflow:hidden"><div style="width:0%;height:100%;background:#6c8cff;border-radius:2px;animation:modern-load 0.8s ease-out forwards"></div></div>', delay: 900 },
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
  if (!text || !connected) return;

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
  }

  input.value = "";
  input.style.height = "auto";

  // Store prompt for Visual tab.
  lastPrompt = text;
  lastResult = "";

  // Clear tracked HTML files from previous turn.
  agentCreatedHtmlFiles = [];

  // Show user input in log.
  logEntry("user", text);

  // Clear previous answer for fresh response.
  clearAnswer();
  clearVisual();

  // Render input decomposition.
  renderInputDecomposition(text);

  // Send to agent.
  if (text.startsWith("/")) {
    send({ type: "command", command: text });
  } else {
    send({ type: "chat", content: text });
  }

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

    // Update map.
    renderMap();
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

  const nodes = Array.from(explorationNodes.values());
  if (nodes.length === 0) {
    svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="var(--text-dim)" font-size="13" font-family="var(--font)">No exploration history yet. Ask a question to begin.</text>`;
    return;
  }

  // Sort nodes chronologically for orphan-linking.
  const sorted = [...nodes].sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));

  // Auto-link orphan click-nodes to the nearest prior manual node.
  // This fixes explore-clicks that lost their parent linkage.
  let lastManualId = null;
  for (const n of sorted) {
    const isClick = n.source === 'click';
    if (!isClick) {
      lastManualId = n.id;
    } else if (!n.parent_id || !explorationNodes.has(n.parent_id)) {
      // Orphan click node — attach to nearest prior manual node.
      if (lastManualId) {
        n.parent_id = lastManualId;
      }
    }
  }

  // Classify nodes: "main" (manual) vs "sub" (click / explore).
  const isSubNode = (n) => n.source === 'click';

  // Build tree structure.
  const childrenMap = new Map();
  const rootIds = [];

  for (const n of sorted) {
    if (!n.parent_id || !explorationNodes.has(n.parent_id)) {
      rootIds.push(n.id);
    } else {
      if (!childrenMap.has(n.parent_id)) childrenMap.set(n.parent_id, []);
      childrenMap.get(n.parent_id).push(n.id);
    }
  }

  // Layout: assign (x, y) to each node using recursive tree layout.
  // Sub-nodes use tighter vertical spacing.
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
    // Center parent above children.
    const firstChild = positions.get(children[0]);
    const lastChild = positions.get(children[children.length - 1]);
    const cx = (firstChild.x + lastChild.x) / 2;
    positions.set(id, { x: cx, y: depth });
  }

  for (const rid of rootIds) {
    layoutNode(rid, 0);
  }

  // Convert grid positions to pixel coordinates.
  // Use cumulative Y to account for mixed main/sub row heights.
  const padding = 40;
  const pixelPositions = new Map();

  // Determine max depth.
  let maxDepth = 0;
  for (const pos of positions.values()) {
    maxDepth = Math.max(maxDepth, pos.y);
  }

  // Compute Y offsets per depth row.
  const rowY = [padding];
  for (let d = 1; d <= maxDepth; d++) {
    // Check if all nodes at depth d-1 are sub-nodes.
    let prevRowHeight = MAP_MAIN_H;
    let gap = MAP_V_GAP;
    // Check if nodes at this depth are sub-nodes — use tighter spacing.
    let allSub = true;
    for (const [id, pos] of positions) {
      if (pos.y === d) {
        const node = explorationNodes.get(id);
        if (node && !isSubNode(node)) { allSub = false; break; }
      }
    }
    if (allSub) gap = MAP_SUB_V_GAP;

    // Check prev row for height.
    for (const [id, pos] of positions) {
      if (pos.y === d - 1) {
        const node = explorationNodes.get(id);
        if (node && isSubNode(node)) prevRowHeight = Math.min(prevRowHeight, MAP_SUB_H);
      }
    }
    rowY.push(rowY[d - 1] + prevRowHeight + gap);
  }

  for (const [id, pos] of positions) {
    const node = explorationNodes.get(id);
    const w = (node && isSubNode(node)) ? MAP_SUB_W : MAP_MAIN_W;
    pixelPositions.set(id, {
      px: padding + pos.x * (MAP_MAIN_W + MAP_H_GAP) + (MAP_MAIN_W - w) / 2,
      py: rowY[pos.y] || padding,
    });
  }

  // Calculate SVG viewBox dimensions.
  let maxPx = 0, maxPy = 0;
  for (const [id, p] of pixelPositions) {
    const node = explorationNodes.get(id);
    const w = (node && isSubNode(node)) ? MAP_SUB_W : MAP_MAIN_W;
    const h = (node && isSubNode(node)) ? MAP_SUB_H : MAP_MAIN_H;
    maxPx = Math.max(maxPx, p.px + w);
    maxPy = Math.max(maxPy, p.py + h);
  }
  const svgW = maxPx + padding;
  const svgH = maxPy + padding;

  // Build SVG content.
  let svgContent = '';

  // Edges.
  for (const n of sorted) {
    if (n.parent_id && pixelPositions.has(n.parent_id) && pixelPositions.has(n.id)) {
      const parentNode = explorationNodes.get(n.parent_id);
      const parentW = (parentNode && isSubNode(parentNode)) ? MAP_SUB_W : MAP_MAIN_W;
      const parentH = (parentNode && isSubNode(parentNode)) ? MAP_SUB_H : MAP_MAIN_H;
      const childW = isSubNode(n) ? MAP_SUB_W : MAP_MAIN_W;

      const parent = pixelPositions.get(n.parent_id);
      const child = pixelPositions.get(n.id);
      const x1 = parent.px + parentW / 2;
      const y1 = parent.py + parentH;
      const x2 = child.px + childW / 2;
      const y2 = child.py;
      const midY = (y1 + y2) / 2;

      // Sub-node edges: dashed, thinner, dimmer.
      const isSub = isSubNode(n);
      const dashAttr = isSub ? ' stroke-dasharray="4,3"' : '';
      const edgeWidth = isSub ? 1.5 : 2;
      const edgeOpacity = isSub ? 0.4 : 0.6;

      svgContent += `<path d="M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}"
        fill="none" stroke="var(--accent)" stroke-width="${edgeWidth}" opacity="${edgeOpacity}"${dashAttr}/>`;

      // Edge label.
      if (n.edge_label) {
        const labelX = (x1 + x2) / 2;
        const labelY = midY - 4;
        const label = n.edge_label.length > 25 ? n.edge_label.slice(0, 25) + '…' : n.edge_label;
        svgContent += `<text x="${labelX}" y="${labelY}" text-anchor="middle"
          fill="var(--text-dim)" font-size="9" font-family="var(--font)">${escSvg(label)}</text>`;
      }
    }
  }

  // Nodes.
  for (const n of sorted) {
    if (!pixelPositions.has(n.id)) continue;
    const pos = pixelPositions.get(n.id);
    const isCurrent = n.id === currentNodeId;
    const isSub = isSubNode(n);

    const w = isSub ? MAP_SUB_W : MAP_MAIN_W;
    const h = isSub ? MAP_SUB_H : MAP_MAIN_H;
    const strokeColor = isCurrent ? 'var(--accent)' : 'var(--chrome-lo)';
    const strokeWidth = isCurrent ? 3 : 1;
    const fillColor = isCurrent ? 'var(--surface-alt)' : 'var(--surface)';
    const dashAttr = isSub ? ' stroke-dasharray="3,2"' : '';
    const opacity = isSub ? ' opacity="0.7"' : '';

    svgContent += `<g class="map-node" data-id="${n.id}" style="cursor:pointer"${opacity}>`;
    svgContent += `<rect x="${pos.px}" y="${pos.py}" width="${w}" height="${h}"
      rx="${isSub ? 12 : 4}" ry="${isSub ? 12 : 4}" fill="${fillColor}" stroke="${strokeColor}" stroke-width="${strokeWidth}"${dashAttr}/>`;

    if (isSub) {
      // Compact sub-node: icon + short prompt only.
      svgContent += `<text x="${pos.px + 8}" y="${pos.py + 14}" font-size="10"
        fill="var(--text-dim)">🔗</text>`;

      const preview = n.prompt.length > 18 ? n.prompt.slice(0, 18) + '…' : n.prompt;
      svgContent += `<text x="${pos.px + 22}" y="${pos.py + 14}" font-size="10"
        fill="var(--text)" font-family="var(--font)">${escSvg(preview)}</text>`;

      // Tiny answer size.
      const ansLen = n.answer ? humanSize(n.answer.length) : '';
      svgContent += `<text x="${pos.px + 8}" y="${pos.py + 28}" font-size="8"
        fill="var(--text-dim)" font-family="var(--font)">${ansLen}${n.visual_html ? ' · 🖼' : ''}</text>`;
    } else {
      // Full main node.
      svgContent += `<text x="${pos.px + 8}" y="${pos.py + 16}" font-size="11"
        fill="var(--text-dim)">✎</text>`;

      const preview = n.prompt.length > 24 ? n.prompt.slice(0, 24) + '…' : n.prompt;
      svgContent += `<text x="${pos.px + 24}" y="${pos.py + 16}" font-size="11" font-weight="700"
        fill="var(--text)" font-family="var(--font)">${escSvg(preview)}</text>`;

      const ts = new Date(n.created_at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
      svgContent += `<text x="${pos.px + 8}" y="${pos.py + 34}" font-size="10"
        fill="var(--text-dim)" font-family="var(--font)">${ts} · ${n.theme}</text>`;

      const ansLen = n.answer ? humanSize(n.answer.length) : '';
      svgContent += `<text x="${pos.px + 8}" y="${pos.py + 50}" font-size="9"
        fill="var(--text-dim)" font-family="var(--font)">${ansLen}${n.visual_html ? ' · 🖼' : ''}</text>`;
    }

    svgContent += `</g>`;
  }

  // Apply transform.
  const t = mapTransform;
  svg.innerHTML = `<g transform="translate(${t.x},${t.y}) scale(${t.scale})">${svgContent}</g>`;
  svg.setAttribute('viewBox', `0 0 ${Math.max(svgW, 400)} ${Math.max(svgH, 300)}`);
  svg.style.width = '100%';
  svg.style.height = '100%';

  // Attach click handlers to nodes.
  for (const g of svg.querySelectorAll('.map-node')) {
    g.addEventListener('click', () => {
      const nodeId = g.dataset.id;
      if (nodeId) {
        navigateToNode(nodeId);
        // Switch to Answer tab.
        const answerBtn = $(".tab-btn[data-tab='answer']");
        if (answerBtn) answerBtn.click();
      }
    });
  }
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
        ${previewable ? '<button class="tab-btn file-preview-btn" title="Preview">👁</button>' : ''}
        <a class="tab-btn" href="/api/files/view?path=${encodeURIComponent(f.physical)}" target="_blank" title="Open in new tab">↗</a>
        <a class="tab-btn" href="/api/files/download?path=${encodeURIComponent(f.physical)}" title="Download">⬇</a>
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
      el.innerHTML = '<span class="output-placeholder">No files yet.</span>';
      el.classList.add("output-empty");
      countEl.textContent = '';
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

function initFiles() {
  const refreshBtn = $("#files-refresh-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => loadFiles());
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

    const sel = $("#model-selector");
    // Keep the default option, add available models.
    sel.innerHTML = '<option value="">(default model)</option>';
    for (const m of availableModels) {
      const label = m.description
        ? `${m.id} — ${m.description}`
        : `${m.id} (${m.provider}/${m.model})`;
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = label;
      if (m.id === selectedModel) opt.selected = true;
      sel.appendChild(opt);
    }

    console.log(`[Computer] Loaded ${availableModels.length} available models`);
  } catch (e) {
    console.warn("Failed to load models:", e);
  }
}

function initSelectors() {
  // Model selector.
  const modelSel = $("#model-selector");
  if (modelSel) {
    modelSel.addEventListener("change", (e) => {
      selectedModel = e.target.value;
      localStorage.setItem("computer-model", selectedModel);
      logEntry("system", `Visual model: ${selectedModel || '(default)'}`);
    });
  }

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

/* ═══════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  // Restore theme.
  const saved = localStorage.getItem("computer-theme");
  if (saved && saved !== currentTheme) {
    setTheme(saved);
    $("#theme-selector").value = saved;
  }

  // Theme selector.
  $("#theme-selector").addEventListener("change", (e) => {
    setTheme(e.target.value);
  });

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
