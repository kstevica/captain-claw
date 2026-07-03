# Security Review: RealmCraft RTS — Round 5 (Fresh Pass)

**Date:** 2026-07-03 08:58 UTC  
**Reviewer:** Security Reviewer (Captain Claw fleet, PHRYGIAN adversarial mode)  
**Artifacts Reviewed:** `index.html` (1,775 lines, ~68 KB), `test_game.js` (1,392 lines, ~51 KB), `plan.md` (design doc)  
**Task Under Review:** "When worker gets gold or wood and carries it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it" — the auto-build dropoff chain  
**Prior Reports:** `SECURITY_REVIEW.md` (R0), `review-r0-security-reviewer.md`, `review-r1-security-reviewer.md`, `SECURITY_REVIEW_CURRENT.md` (R1/R2/R3), `SECURITY_REVIEW_R4.md`  
**Scope:** Injection | Auth/Authz | Secrets in code | Unsafe input handling | Dependency risks | Auto-build chain deep-dive  
**Methodology:** Read-only static analysis, adversarial assumption (things WILL go wrong), line-by-line review of all dynamic code paths including the full auto-build dropoff flow, grep audit for all injection sinks, prototype pollution vectors, and unsafe patterns.  
**Delta from R4:** +5 new findings. All 9 prior findings reconfirmed.

---

## Executive Summary

**Overall Risk: LOW** (unchanged)

RealmCraft remains a single-file, zero-dependency, client-side-only RTS game with no backend, no network APIs, no text input fields, no storage, and no CDN fetches. The attack surface is minimal. **No critical, high, or medium-severity vulnerabilities exist.** All issues require browser console access to exploit — once an attacker has console access, they already own the execution environment.

This review adds **5 new low/informational findings** (SR-R5-01 through SR-R5-05), most specific to the auto-build dropoff chain under review:

| # | CVSS | Severity | Category | Finding |
|---|------|----------|----------|---------|
| **SR-R5-01** | 2.5 | Low | DoS / Unsafe input | `findAutoBuildSite` full-map fallback O(n²·m) — console-triggered DoS |
| **SR-R5-02** | 2.0 | Low | Code quality | Accidental global `cam` variable (missing `let`/`const`) in `resizeCanvas` |
| **SR-R5-03** | 1.8 | Low | Unsafe input | Missing zero-guard on `canvas.clientWidth`/`clientHeight` divisors |
| **SR-R5-04** | 0.0 | Info | Resource mgmt | Unbounded `autoBuildCooldown` object growth — memory leak in long sessions |
| **SR-R5-05** | 0.0 | Info | Dead code | Unused `dpr` variable in `screenToWorld` — correctness smell |

All **9 prior findings** (SR-NEW-01 through SR-R3-03) are reconfirmed as **present and unfixed**.

---

## Complete Findings Table (CVSS-Ranked)

| # | Title | CVSS 3.1 | Severity | Category | Status |
|---|-------|----------|----------|----------|--------|
| **SR-NEW-01** | Dynamic property key `game.resources[e.carryType]` creates prototype-pollution path | 3.5 | **Low** | Unsafe input | ⚠️ Unfixed (R2→R5) |
| SR-01 | Unsafe `innerHTML` sinks for dynamic game state (10 locations) | 2.8 | **Low** | Injection | ⚠️ Unfixed (R0→R5) |
| **SR-R5-01** | `findAutoBuildSite` full-map fallback O(n²·m) — console-triggered DoS | 2.5 | **Low** | DoS | 🔴 **New R5** |
| SR-R3-01 | NaN/Infinity passthrough in entity coordinate parameters | 2.5 | **Low** | Unsafe input | ⚠️ Unfixed (R3→R5) |
| SR-03 | Unbounded entity creation (resource exhaustion via console) | 2.5 | **Low** | Unsafe input | ⚠️ Unfixed (R0→R5) |
| **SR-R5-02** | Accidental global `cam` variable in `resizeCanvas` | 2.0 | **Low** | Code quality | 🔴 **New R5** |
| SR-NEW-02 | `Object.entries(BUILDING_DEFS)` iteration exposes prototype pollution | 2.0 | **Low** | Unsafe input | ⚠️ Unfixed (R2→R5) |
| SR-R3-02 | Mutable game-definition objects — no `Object.freeze()` | 2.0 | **Low** | Defense-in-depth | ⚠️ Unfixed (R3→R5) |
| **SR-R5-03** | Missing zero-guard on `canvas.clientWidth`/`clientHeight` division | 1.8 | **Low** | Unsafe input | 🔴 **New R5** |
| SR-04 | `isPlacementValid` lacks type validation guard for `buildingType` | 1.8 | **Low** | Unsafe input | ⚠️ Unfixed (R1→R5) |
| **SR-R5-04** | Unbounded `autoBuildCooldown` object growth | 0.0 | Info | Resource mgmt | 🔴 **New R5** |
| **SR-R5-05** | Unused `dpr` variable in `screenToWorld` | 0.0 | Info | Dead code | 🔴 **New R5** |
| SR-R3-03 | Duplicate `attackTarget:null` — dead code | 0.0 | Info | Code quality | ⚠️ Unfixed (R3→R5) |
| SR-02 | Missing Content-Security-Policy | 0.0 | Info | Defense-in-depth | ℹ️ Unfixed (R0→R5) |

---

## Auto-Build Dropoff Chain: Deep-Dive Security Analysis

The task under review is the full resource gathering → dropoff → auto-build flow. Below is a stage-by-stage security analysis:

### Stage 1: Gathering (`issueGather` → gathering loop)

**Code path:** `issueGather()` (line ~1480) → `updateEntity()` gathering state → `carryType`/`carryAmount` assignment

```javascript
// index.html ~line 1230
const curTile = tileAt(e.x, e.y);
if (curTile !== (e.carryType === 'gold' ? TERRAIN.GOLD : TERRAIN.TREE)) { ... }
// ...
e.carryAmount = 10;
e.gatheringTimer = 0;
```

**Security assessment:** Safe. `carryType` is always set to either `'gold'` or `'wood'` via `issueGather` which derives it from `tileAt(wx, wy) === TERRAIN.GOLD ? 'gold' : 'wood'`. No external data enters this path. The `carryAmount` is hardcoded to `10`.

### Stage 2: Find Dropoff (`findNearestDropoff`)

**Code path:** `findNearestDropoff(e.x, e.y, e.carryType)`

```javascript
// index.html ~line 705
function findNearestDropoff(x, y, resType) {
  for (const e of game.entities.values()) {
    if (e.owner !== 'player' || e.progress < 1) continue;
    const def = BUILDING_DEFS[e.type];
    if (!def) continue;  // <-- guards against unknown types
    if (resType === 'gold' && (def.dropoff || def.dropoff_gold)) { ... }
    if (resType === 'wood' && (def.dropoff || def.dropoff_wood)) { ... }
  }
}
```

**Security assessment:** Safe. `resType` is trusted (only 'gold' or 'wood'). The function guards against entities with types not in `BUILDING_DEFS` via `if (!def) continue`. Property access on `def` is safe because `def` comes from the hardcoded `BUILDING_DEFS` object.

### Stage 3: Auto-Build Trigger (`autoBuildDropoff`)

**Code path:** `autoBuildDropoff(peasant)` — line ~750

```javascript
function autoBuildDropoff(peasant) {
  const resType = peasant.carryType;
  if (!resType || peasant.carryAmount <= 0) return false;
  // ...
  const buildingType = getDropoffBuildingType(resType);
  if (!buildingType) return false;
  // ...
  const cost = BUILDING_DEFS[buildingType].cost;  // safe: buildingType is 'refinery' or 'lumber_mill'
  // ...
}
```

**Security assessment:** Safe with one caveat. `peasant.carryType` is trusted (set to 'gold' or 'wood' only). `getDropoffBuildingType` returns `null` for anything other than 'gold'/'wood', and the function returns early. `BUILDING_DEFS[buildingType]` is safe because `buildingType` can only be 'refinery' or 'lumber_mill'.

### Stage 4: Resource Deposit (`game.resources[e.carryType]`)

**Code path:** Line ~1260

```javascript
game.resources[e.carryType] += e.carryAmount;
```

**Security assessment:** ⚠️ **SR-NEW-01** — Dynamic property key creates a prototype pollution vector. If `carryType` were ever set to `__proto__` or `constructor` via console manipulation, this would pollute `Object.prototype`. Under normal operation, `carryType` is always 'gold' or 'wood', so this is **not exploitable through game mechanics**. Defensive fix: validate the key is in a whitelist before assignment.

### Stage 5: Auto-Build Site Search (`findAutoBuildSite`) — **NEW FINDING**

**Code path:** `findAutoBuildSite(buildingType, nearX, nearY)` — line ~730

This function has a **full-map fallback** when ring searches fail:

```javascript
// index.html ~line 740
let best = null, bestDist = Infinity;
for (let ty = 0; ty < MAP_ROWS; ty++) {
  for (let tx = 0; tx < MAP_COLS; tx++) {
    if (game.map[ty][tx] === TERRAIN.WATER || game.map[ty][tx] === TERRAIN.TREE) continue;
    const wx = tx * TILE_SIZE + TILE_SIZE / 2, wy = ty * TILE_SIZE + TILE_SIZE / 2;
    if (!isPlacementValid(buildingType, wx, wy)) continue;
    // ...
  }
}
```

`isPlacementValid` iterates **all entities** for overlap checking. Combined: 4,096 tiles × N entities = O(n²·m). With many console-created entities, this can freeze the game loop.

### Stage 6: Idle Retry Loop

**Code path:** Line ~1300

```javascript
if (e.type === 'peasant' && e.state === 'idle' && e.carryAmount > 0 && e.carryType && !e.isBuilding) {
  if (Math.floor(game.time * 10) % 20 === 0) {
    const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
    if (dropoff) { ... } else { autoBuildDropoff(e); }
  }
}
```

**Security assessment:** Safe. The ~2-second polling interval uses `game.time` arithmetic which is safe. No unbounded recursion or infinite loops.

---

## New Findings (R5 — This Review)

---

### SR-R5-01 — `findAutoBuildSite` Full-Map Fallback O(n²·m) → DoS

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L** — Score: **2.5 (Low)**  
**CWE-834:** Excessive Iteration  
**Status:** 🔴 New — specific to the auto-build dropoff chain

#### Location

`index.html:738–748` — `findAutoBuildSite()` full-map fallback loop

```javascript
// Fallback: search entire map loosely for nearest valid spot
let best=null, bestDist=Infinity;
for(let ty=0;ty<MAP_ROWS;ty++) {
  for(let tx=0;tx<MAP_COLS;tx++) {
    if(game.map[ty][tx]===TERRAIN.WATER||game.map[ty][tx]===TERRAIN.TREE) continue;
    const wx=tx*TILE_SIZE+TILE_SIZE/2, wy=ty*TILE_SIZE+TILE_SIZE/2;
    if(!isPlacementValid(buildingType, wx, wy)) continue;  // <-- iterates ALL entities
    const d=dist({x:nearX,y:nearY},{x:wx,y:wy});
    if(d<bestDist) { bestDist=d; best={x:wx,y:wy}; }
  }
}
```

**Attack vector:** An attacker with browser console access creates thousands of entities. When `findAutoBuildSite` is triggered (peasant gathers with no dropoff), the ring search fails (all perimeter tiles overlap entities), and the fallback scans all 4,096 map tiles × N entities = up to millions of overlap checks. This freezes the game loop on the next `requestAnimationFrame` tick where `updateEntity` calls `autoBuildDropoff`.

**Worst case:** 4,096 tiles × 10,000 console-created entities = ~40 million `rectsOverlap` calls, each with object property access. Estimated freeze: 500ms–2s on modern hardware.

**Remediation:**
```javascript
// Add entity count guard before full-map search
if (game.entities.size > 500) return null; // don't search full map with too many entities

// OR: Use spatial hashing (grid-based) for O(1) entity lookup in isPlacementValid
// Store entities in a 2D grid keyed by tile coordinates
```

---

### SR-R5-02 — Accidental Global `cam` Variable in `resizeCanvas`

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N** — Score: **2.0 (Low)**  
**CWE-1104:** Use of Unmaintained Third-Party Components (N/A — but analogous pattern: accidental global)  
**Status:** 🔴 New

#### Location

`index.html:1526` — `resizeCanvas()` function

```javascript
function resizeCanvas() {
  const rect=wrapper.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  canvas.width=rect.width*dpr;
  canvas.height=rect.height*dpr;
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  // Re-clamp camera after resize
  cam=game.camera;           // <-- BUG: missing 'const'/'let' — creates global 'cam'
  cam.x=clamp(cam.x,0,Math.max(0,WORLD_W-canvas.clientWidth/cam.zoom));
  cam.y=clamp(cam.y,0,Math.max(0,WORLD_H-canvas.clientHeight/cam.zoom));
}
```

**Impact:** The `cam` variable leaks into the global (`window`) scope. An attacker with console access can set `window.cam` to a malicious object that intercepts `.x`/`.y` property access. During a window resize event, `resizeCanvas` reads `cam.x` and `cam.y` from the attacker-controlled object rather than `game.camera`.

**Exploitation scenario:** 
```javascript
// Attacker in console:
cam = { get x() { /* exfiltrate data */ return 0; }, get y() { return 0; }, zoom: 1 };
// Trigger resize — resizeCanvas uses attacker's cam, not game.camera
```

**Remediation:**
```javascript
// Line 1526: add 'const'
const cam = game.camera;
```

---

### SR-R5-03 — Missing Zero-Guard on `canvas.clientWidth`/`clientHeight` Division

**CVSS:3.1/AV:P/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:L** — Score: **1.8 (Low)**  
**CWE-369:** Divide By Zero  
**Status:** 🔴 New

#### Location

`index.html:665–666` — `screenToWorld()` function

```javascript
function screenToWorld(sx, sy) {
  const cam=game.camera;
  const canvas=document.getElementById('game-canvas');
  const dpr=window.devicePixelRatio||1;  // declared but unused
  return {
    x: sx*(canvas.width/canvas.clientWidth)/cam.zoom+cam.x,   // <-- clientWidth could be 0
    y: sy*(canvas.height/canvas.clientHeight)/cam.zoom+cam.y  // <-- clientHeight could be 0
  };
}
```

**Impact:** If the game canvas is rendered in a hidden/zero-size container (e.g., `display:none`, detached DOM, zero-size iframe), `canvas.clientWidth` and `canvas.clientHeight` return 0. Division by zero produces `Infinity` or `NaN`, which propagates through the game's coordinate system. Mouse position tracking, entity targeting, and building placement all depend on `screenToWorld`. With `Infinity` coordinates, `clamp()` fails silently, `tileAt()` returns `-1`, and `isPassable()` returns `false`, breaking all interaction.

**Edge case:** This could happen during React/SPA component mounting/unmounting if the game were embedded.

**Remediation:**
```javascript
function screenToWorld(sx, sy) {
  const cam = game.camera;
  const canvas = document.getElementById('game-canvas');
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (!cw || !ch || !cam.zoom) return { x: sx, y: sy }; // safe fallback
  return {
    x: sx * (canvas.width / cw) / cam.zoom + cam.x,
    y: sy * (canvas.height / ch) / cam.zoom + cam.y
  };
}
```

---

### SR-R5-04 — Unbounded `autoBuildCooldown` Object Growth

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N** — Score: **0.0 (Informational)**  
**CWE-404:** Improper Resource Shutdown or Release  
**Status:** 🔴 New

#### Location

`index.html:137` — `game.autoBuildCooldown` state initialization
`index.html:762–813` — multiple cooldown write locations in `autoBuildDropoff`

```javascript
const game = {
  // ...
  autoBuildCooldown: {},  // <-- never pruned
  // ...
};

// In autoBuildDropoff:
game.autoBuildCooldown[buildingType] = game.time;       // on success
game.autoBuildCooldown[buildingType] = game.time + 3;   // on failure
```

**Impact:** The `autoBuildCooldown` object accumulates entries over the game session. While the only possible keys are `'refinery'` and `'lumber_mill'` (from `getDropoffBuildingType`), the object itself is never cleaned. If the auto-build function were extended to support more building types, this could grow unboundedly. Currently negligible — at most 2 keys exist.

**Remediation:** Not urgent. If the system is extended, add periodic pruning:
```javascript
// In update() or on game tick:
for (const [key, timestamp] of Object.entries(game.autoBuildCooldown)) {
  if (game.time - timestamp > 60) delete game.autoBuildCooldown[key];
}
```

---

### SR-R5-05 — Unused `dpr` Variable in `screenToWorld`

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N** — Score: **0.0 (Informational)**  
**CWE-563:** Assignment to Variable without Use  
**Status:** 🔴 New

#### Location

`index.html:663` — `screenToWorld()` function

```javascript
function screenToWorld(sx, sy) {
  const cam=game.camera;
  const canvas=document.getElementById('game-canvas');
  const dpr=window.devicePixelRatio||1;  // <-- declared but NEVER used
  return {
    x: sx*(canvas.width/canvas.clientWidth)/cam.zoom+cam.x,
    y: sy*(canvas.height/canvas.clientHeight)/cam.zoom+cam.y
  };
}
```

**Why it works anyway:** `canvas.width` was set to `rect.width * dpr` in `resizeCanvas()`, and `canvas.clientWidth` returns `rect.width`. So `canvas.width / canvas.clientWidth === dpr`. The variable is redundant.

**Risk:** This is dead code that creates confusion about the coordinate conversion logic. If someone "fixes" the function by removing `dpr` without understanding the equivalence, they might break the coordinate system. Additionally, during rapid resize events where `canvas.width` hasn't been updated yet, `dpr` and the ratio could be temporarily inconsistent — though the variable is unused so this doesn't matter.

**Remediation:**
```javascript
// Option A: Use the variable (makes intent clearer)
return {
  x: sx * dpr / cam.zoom + cam.x,
  y: sy * dpr / cam.zoom + cam.y
};

// Option B: Remove the unused declaration
// Delete line 663
```

---

## Reconfirmed Prior Findings (All Unfixed)

All 9 prior findings from R0–R4 remain unfixed. Key highlights:

### SR-NEW-01 (CVSS 3.5) — Dynamic Property Key Prototype Pollution

`game.resources[e.carryType] += e.carryAmount` at line ~1260. If `carryType` were ever `__proto__` or `constructor`, this pollutes `Object.prototype`. Under normal operation, `carryType` is always `'gold'` or `'wood'`, but there is no runtime guard.

**Remediation:**
```javascript
const VALID_RESOURCES = ['gold', 'wood', 'food'];
if (VALID_RESOURCES.includes(e.carryType)) {
  game.resources[e.carryType] += e.carryAmount;
}
```

### SR-01 (CVSS 2.8) — Unsafe `innerHTML` Sinks

10 locations in `updateUI()` use `innerHTML` with dynamically-constructed strings. Currently safe because all interpolated values come from hardcoded constants, but this is a fragile pattern.

### SR-02 — Missing Content-Security-Policy

No CSP meta tag or header. While all code is inline (making a strict CSP incompatible without refactoring), even a permissive CSP like `<meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline';">` provides defense-in-depth against future vulnerabilities.

---

## Positive Security Observations

The following security-positive patterns are worth noting (these are things the code does RIGHT):

1. **`textContent` used for all resource/simple text updates** — 4 locations in `updateUI()` use `textContent` which auto-escapes HTML
2. **No `eval()`, `new Function()`, or string-based timers** — zero code-injection vectors
3. **No external dependencies** — no CDN fetches, no npm packages, no `<script src>`
4. **No persistence** — no `localStorage`, `sessionStorage`, `document.cookie`, `IndexedDB`
5. **No network** — no `fetch()`, `XMLHttpRequest`, `WebSocket`, `postMessage`
6. **`requestAnimationFrame` for game loop** — not `setInterval`, which prevents background-tab abuse
7. **`dt` cap of 0.2** — prevents spiral-of-death from large delta-time values
8. **Type guards in entity factories** — `typeof type !== 'string'` and `Number.isFinite()` checks in `createUnit`/`createBuilding`
9. **`contextmenu` prevention** — `canvas.addEventListener('contextmenu', e => e.preventDefault())` prevents browser context menu interference
10. **`user-select: none` CSS** — prevents accidental text selection during gameplay
11. **No `prototype` manipulation** — no direct `Object.setPrototypeOf`, `__proto__`, or `constructor` assignment (only the indirect dynamic key path in SR-NEW-01)

---

## Remediation Priority

| Priority | Finding | Effort | Impact |
|----------|---------|--------|--------|
| **P1** | SR-NEW-01: Dynamic key prototype pollution | 1 line | Low likelihood, high blast radius |
| **P2** | SR-01: `innerHTML` → `textContent` + DOM | Medium refactor | Defense-in-depth |
| **P3** | SR-R5-02: Accidental global `cam` | 1 character (`const`) | Trivial fix, eliminates global leak |
| **P4** | SR-R5-01: O(n²) auto-build site search | Medium (spatial hash) | Only exploitable via console |
| **P5** | SR-R5-03: Zero-guard on canvas dimensions | 3 lines | Edge case in embedded scenarios |
| P6–P14 | All others | Varies | Informational or defense-in-depth |

---

## Test Suite (`test_game.js`) Notes

The test suite is a Node.js development artifact and does not ship to end users. It contains no secrets, no network calls, and only mock browser APIs. **No security issues** were found in the test suite. Notable:
- `global.*` assignments are standard Node.js test mocking — not a security concern
- Mock `document.getElementById` returns fixtures — no XSS risk
- `process.exit()` is standard Node.js — not applicable to browser context

---

## Conclusion

The RealmCraft RTS codebase maintains its **LOW overall risk** posture. The auto-build dropoff chain under review is functionally correct and introduces no new exploitable attack surface through normal gameplay. All 14 findings require browser console access to exploit — a bar that means the attacker already has full control of the execution environment. The most actionable fix is the accidental global `cam` variable (one character: add `const`) and the prototype-pollution guard (one-line whitelist check).
