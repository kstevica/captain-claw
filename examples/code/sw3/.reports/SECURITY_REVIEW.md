# Security Review: RealmCraft RTS Game

**Date:** 2026-07-03  
**Reviewer:** Security Reviewer (CAPTAIN CLAW fleet)  
**Artifact:** `index.html` — single-file RTS game (53,356 bytes, ~1,370 lines)  
**Methodology:** Static analysis of all JS, CSS, and HTML. Dynamic analysis limited to read-only code inspection.

---

## Executive Summary

**Overall Risk: LOW**

RealmCraft is a client-side, single-file HTML/CSS/JS game with **zero external dependencies** — no CDN fetches, no npm packages, no polyfills. It has no backend, no localStorage, no cookies, no network requests, no user text input fields, and no URL parameter parsing. This gives it an extraordinarily small attack surface. The most significant finding is the use of `innerHTML` (10 occurrences) to render UI content, which — while currently safe due to all interpolated values originating from hardcoded game definitions — represents a fragile pattern that could become dangerous if the codebase evolves.

No critical, high, or medium-severity vulnerabilities were found. Two low-severity findings and one informational note are reported below.

---

## Findings Summary

| # | Title | CVSS 3.1 | Severity |
|---|-------|----------|----------|
| SEC-01 | Unsafe `innerHTML` sink for building training queue | 2.3 | Low |
| SEC-02 | Missing Content-Security-Policy | 0.0 | Informational |
| SEC-03 | Unbounded entity creation (DoS via console) | 2.5 | Low |

---

## Detailed Findings

---

### SEC-01 — Unsafe `innerHTML` Sink for Building Training Queue

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N** — Score: **2.3 (Low)**  
**CWE-79:** Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)

**Location:** `index.html:1141`, `index.html:1144`, `index.html:1179`, `index.html:1215`

#### Description

Ten instances of `innerHTML` assignment exist in the codebase. While most insert hardcoded HTML strings (safe), **four instances** interpolate dynamic game-state values into HTML without sanitization:

**1. Building info panel (line 1141–1144):**
```javascript
// Line 1141
stats+=`<div class="queue-info">Queue: ${e.queue.map(t=>UNIT_DEFS[t]?UNIT_DEFS[t].name:t).join(', ')}</div>`;
// Line 1144
infoStats.innerHTML=stats;
```
If `e.queue` ever contained an unexpected entry (not present in `UNIT_DEFS`), the fallback `t` would be injected raw as HTML. Currently the queue is populated only from `building.produces` arrays (hardcoded in `BUILDING_DEFS`), so this is **not exploitable today**.

**2. Unit info panel (line 1179):**
```javascript
let stats=`HP: ${Math.floor(e.hp)}/${e.maxHp} | Speed: ${defU.speed} | Atk: ${defU.atk.damage} | Range: ${defU.atk.range}`;
// ...
if(e.carryAmount>0) stats+=`<br>Carrying: ${e.carryAmount} ${e.carryType}`;
if(e.state==='gathering') stats+=`<br>Gathering: ${e.carryType} (${Math.floor(e.gatheringTimer/1.5*100)}%)`;
infoStats.innerHTML=stats;
```
`e.state` and `e.carryType` are set exclusively by the game's own logic functions (`issueGather`, `updateEntity`) to values from hardcoded sets (`'idle'|'moving'|'gathering'|'building'`, `'gold'|'wood'`). Not exploitable via normal gameplay.

**3. Multi-selection info panel (line 1215):**
```javascript
let stats=`Avg HP: ${Math.floor(avgPct*100)}%`;
stats+=`<div class="hp-bar-outer"><div class="hp-bar-inner" style="width:${avgPct*100}%"></div></div>`;
infoStats.innerHTML=stats;
```
All interpolated values are purely numeric. Safe.

**Remaining 6 `innerHTML` occurrences (lines 1119, 1120, 1148, 1168, 1170, 1182, 1216):** Assign hardcoded strings or `''` — safe.

#### Exploitability

| Factor | Assessment |
|--------|------------|
| Vector | Browser developer console (requires physical/local access) |
| Complexity | Low — modify `e.queue` array via console, then select the building |
| Privileges | None required beyond standard browser access |
| Impact | DOM manipulation within the game page; no data exfiltration possible (no sensitive data exists) |

#### Remediation

Replace all dynamic `innerHTML` assignments with safe alternatives:

**Option A (preferred): Use `textContent` + DOM elements:**
```javascript
// Replace line 1141–1144:
const queueSpan = document.createElement('span');
queueSpan.className = 'queue-info';
queueSpan.textContent = `Queue: ${e.queue.map(t => UNIT_DEFS[t]?.name ?? t).join(', ')}`;
infoStats.appendChild(queueSpan);
```

**Option B: Template with sanitization:**
```javascript
const sanitize = (s) => String(s).replace(/[<>&"']/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));
// Use: stats += `Queue: ${sanitize(names)}`;
```

**Option C: Add a CSP that blocks inline scripts:**
```html
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';">
```
Note: this would require converting inline `<script>` to an external file (breaking the single-file constraint) or adding a nonce/hash.

**Recommendation:** Use **Option A** — convert `infoStats` and `actionPanel` content construction to DOM API calls (`createElement`, `textContent`, `appendChild`). This is the most robust solution and adds no overhead.

---

### SEC-02 — Missing Content-Security-Policy

**CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N** — Score: **0.0 (Informational)**  
**CWE-693:** Protection Mechanism Failure

#### Description

The HTML document does not define a Content-Security-Policy via `<meta>` tag or HTTP header. There is no defense-in-depth against XSS, even though no XSS vector exists today.

#### Remediation

Even for a single-file game, a CSP meta tag provides defense-in-depth:

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;">
```

However, since all game logic is in an inline `<script>` block, you would need either:
- Move JS to an external file (breaks single-file requirement), or
- Use `'unsafe-inline'` for scripts (defeats the XSS protection), or  
- Add a `'nonce-...'` or `'sha256-...'` hash to the script-src

Given the single-file constraint and zero-dependency design, the **most practical approach** is to fix SEC-01 (remove `innerHTML` usage) and accept the missing CSP as a known limitation.

---

### SEC-03 — Unbounded Entity Creation (Console DoS)

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L** — Score: **2.5 (Low)**  
**CWE-770:** Allocation of Resources Without Limits or Throttling

#### Description

The `createUnit()` and `createBuilding()` functions have no cap on the total number of entities that can exist in `game.entities` (a `Map`). The game loop iterates over all entities every frame (`updateEntity`), and the renderer draws all visible entities. If an attacker with console access rapidly creates thousands of entities, the game would degrade to unplayable frame rates.

```javascript
function createUnit(type, x, y) {
  // ...
  return { id: genId(), type, owner: 'player', /* ... */ };
  // No check: if (game.entities.size > MAX_ENTITIES) return null;
}
```

#### Exploit Vector

Browser console: 
```javascript
for (let i=0; i<10000; i++) game.entities.set('x'+i, createUnit('peasant', 500, 500));
```

This would cause frame-rate collapse but no crash or data loss.

#### Remediation

Add a hard cap on entities:

```javascript
const MAX_ENTITIES = 500;

function createUnit(type, x, y) {
  if (game.entities.size >= MAX_ENTITIES) return null;
  // ... existing code
}

function createBuilding(type, x, y, progress = 0) {
  if (game.entities.size >= MAX_ENTITIES) return null;
  // ... existing code
}
```

---

## Positive Findings

The following security-positive design decisions deserve recognition:

| # | Finding | Impact |
|---|---------|--------|
| ✅ | **Zero external dependencies** — no CDN, npm, or polyfills | Eliminates supply-chain attacks entirely |
| ✅ | **No `eval()`, `new Function()`, or string-based timers** | No dynamic code execution vectors |
| ✅ | **`textContent` used for most UI text** — buttons, entity names, resource labels | Prevents XSS in those paths |
| ✅ | **No `localStorage`, `sessionStorage`, or cookies** | No persistent client-side data exposure |
| ✅ | **No `XMLHttpRequest`, `fetch()`, or WebSocket usage** | No network exfiltration possible |
| ✅ | **No `document.write()` or `insertAdjacentHTML()`** | No additional injection sinks |
| ✅ | **No URL parameter/hash parsing** | No DOM-based XSS via URL |
| ✅ | **`Map` used for entity registry** instead of plain `{}` | Immune to property-name collisions with `__proto__`, `constructor`, etc. |
| ✅ | **Game loop dt capping** (`Math.min(dt, 0.2)`) | Prevents spiral-of-death from large delta-time spikes |
| ✅ | **Camera coordinate clamping** | Prevents out-of-bounds memory access |

---

## Remediation Checklist

| Priority | Finding | Action |
|----------|---------|--------|
| 🔶 **Should Fix** | SEC-01: `innerHTML` sinks | Replace `infoStats.innerHTML` and queue display with DOM API (`createElement` + `textContent`) at lines 1141, 1144, 1179, 1215 |
| 🔹 **Nice to Have** | SEC-03: Unbounded entities | Add `MAX_ENTITIES = 500` guard to `createUnit()` and `createBuilding()` |
| ℹ️ **Informational** | SEC-02: Missing CSP | Add CSP meta tag if single-file constraint is relaxed, or accept risk after SEC-01 remediation |

---

## Appendix A: Full `innerHTML` Usage Inventory

| Line | Assignment | Risk | Notes |
|------|-----------|------|-------|
| 1119 | `infoStats.innerHTML = 'Select a unit...'` | ✅ Safe | Hardcoded string |
| 1120 | `actionPanel.innerHTML = '<span...>'` | ✅ Safe | Hardcoded string |
| 1144 | `infoStats.innerHTML = stats` | ⚠️ Low | `stats` contains queue entries via template literal |
| 1148 | `actionPanel.innerHTML = ''` | ✅ Safe | Empty string |
| 1168 | `actionPanel.innerHTML = '<span...>'` | ✅ Safe | Hardcoded string |
| 1170 | `actionPanel.innerHTML = '<span...>'` | ✅ Safe | Hardcoded string |
| 1179 | `infoStats.innerHTML = stats` | ⚠️ Low | `stats` contains `e.carryType`, `e.state` |
| 1182 | `actionPanel.innerHTML = ''` | ✅ Safe | Empty string |
| 1215 | `infoStats.innerHTML = stats` | ✅ Safe | All numeric interpolations |
| 1216 | `actionPanel.innerHTML = ''` | ✅ Safe | Empty string |

---

## Appendix B: Scope Boundaries

This review covers only the files present in the workspace:
- `index.html` — the complete RTS game
- `plan.md` — implementation plan (reviewed for architectural risks, no code)

Excluded from scope:
- `.captain-claw/` — internal tooling artifacts (backups, state)
- `.codemap/` — internal index database
- `saved/` — untracked runtime output directory

---

*End of Report — Generated by Security Reviewer (CAPTAIN CLAW fleet)*
