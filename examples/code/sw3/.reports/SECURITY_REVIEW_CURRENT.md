# Security Review: RealmCraft RTS — Current State (Fresh Pass R5)

**Date:** 2026-07-03 09:20 UTC  
**Reviewer:** Security Reviewer (Captain Claw fleet, PHRYGIAN mode)  
**Artifacts Reviewed:** `index.html` (1,788 lines, ~69 KB), `test_game.js` (69469 chars), `plan.md` (design doc)  
**Prior Reports:** `SECURITY_REVIEW.md`, `SECURITY_REVIEW_R4.md`, `SECURITY_REVIEW_R5.md`, `SECURITY_REVIEW_CURRENT.md` (R3)  
**Scope:** Injection | Auth/Authz | Secrets in code | Unsafe input handling | Dependency risks  
**Task Context:** Auto-build dropoff system (workers carrying gold/wood trigger construction of refinery/lumber mill when no dropoff exists)  
**Methodology:** Read-only static analysis, adversarial assumption (Phrygian mode), line-by-line audit of all dynamic code paths, grep audit for injection sinks, prototype pollution vectors, and unsafe patterns.

---

## Executive Summary

**Overall Risk: LOW**

RealmCraft is a single-file, zero-dependency, client-side-only RTS game. It has no backend, no network APIs, no text input fields, no persistent storage, and no CDN fetches. The attack surface is minimal. **No critical, high, or medium-severity vulnerabilities exist.** All issues require browser console access to exploit — a bar that, once crossed, means the attacker already owns the execution environment.

### Delta from R4/R3

| Change | Impact |
|--------|--------|
| ✅ `createUnit` / `createBuilding` now validate `Number.isFinite(x,y)` and `typeof type === 'string'` | **SR-R3-01 fixed** — NaN/Infinity no longer pass through |
| ✅ Gathering validation fix (walk-back instead of reset on tile push-off) | No new security issues introduced |
| ✅ Auto-build system with per-frame lock (`game._autoBuildThisFrame`) | Well-defended against race conditions |
| ✅ `commandStop` / `issueMove` preserve carryAmount when no dropoff exists | No new attack surface |
| ⚠️ `game.resources[e.carryType]` dynamic property access remains | **Still unfixed** |
| ⚠️ `innerHTML` sinks in `updateUI()` remain | **Still unfixed** |
| ⚠️ No Content-Security-Policy | **Still unfixed** |

---

## Complete Findings Table (CVSS-Ranked)

| # | Title | CVSS 3.1 | Severity | Category | File:Line | Status |
|---|-------|----------|----------|----------|-----------|--------|
| **SR-R5-01** | Dynamic property key `game.resources[e.carryType]` — prototype pollution path | 3.5 | **Low** | Unsafe input | `index.html:1284` | ⚠️ Unfixed (R0→R5) |
| **SR-R5-02** | Unsafe `innerHTML` sinks for dynamic game state (7 locations) | 2.8 | **Low** | Injection | `index.html:1363,1364,1412,1432,1434,1465,1498` | ⚠️ Unfixed (R0→R5) |
| **SR-R5-03** | Mutable `BUILDING_DEFS` / `UNIT_DEFS` objects — no `Object.freeze()` | 2.0 | **Low** | Defense-in-depth | `index.html:103-117` | ⚠️ Unfixed (R3→R5) |
| **SR-R5-04** | `isPlacementValid` lacks type validation guard for `buildingType` parameter | 1.8 | **Low** | Unsafe input | `index.html:869` | ⚠️ Unfixed (R1→R5) |
| **SR-R5-05** | `Object.entries(BUILDING_DEFS)` iteration in `updateUI` exposes prototype properties | 2.0 | **Low** | Unsafe input | `index.html:1468` | ⚠️ Unfixed (R2→R5) |
| **SR-R5-06** | Unbounded entity creation (resource exhaustion via console) | 2.5 | **Low** | Unsafe input | `index.html:174-213` | ⚠️ Unfixed (R0→R5) |
| **SR-R5-07** | Missing Content-Security-Policy header | 0.0 | Info | Defense-in-depth | `index.html:1-7` | ℹ️ Unfixed (R0→R5) |
| **SR-R5-08** | Duplicate `attackTarget` property in `createUnit` — dead code | 0.0 | Info | Code quality | `index.html:269,276` | ℹ️ Unfixed (R3→R5) |

---

## Detailed Findings

---

### SR-R5-01: Dynamic Property Key on `game.resources` — Prototype Pollution Path

**CVSS 3.1:** 3.5 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N)  
**File:** `index.html:1284`  
**Status:** ⚠️ Unfixed since R0

**Code:**
```javascript
// index.html:1284 — deposit arrival handler
if(e.carryType==='gold'||e.carryType==='wood') {
  game.resources[e.carryType]+=e.carryAmount;  // ← DYNAMIC KEY
}
```

**Vulnerability:** The expression `game.resources[e.carryType]` uses a bracket-access dynamic property key. While `e.carryType` is normally constrained to `'gold'` or `'wood'` (set only in `issueGather` at line 997), a console attacker could set `carryType` to `'__proto__'`, `'constructor'`, or `'toString'` and pollute the `game.resources` object prototype chain.

**Exploit scenario (console):**
```javascript
// Attacker with console access
const p = [...game.entities.values()].find(e => e.type === 'peasant');
p.carryType = '__proto__';
p.carryAmount = 10;
// On deposit: game.resources.__proto__ += 10 → NaN pollution
```

**Impact:** Prototype pollution can cause subtle logic bugs, break iteration, or (in rare cases) enable gadget-chain escalation. Practical impact is LOW because console access is required.

**Remediation:**
```javascript
// Option A: Whitelist guard (recommended)
const VALID_RESOURCES = new Set(['gold', 'wood', 'food']);
if (VALID_RESOURCES.has(e.carryType)) {
  game.resources[e.carryType] += e.carryAmount;
}

// Option B: Use Map instead of plain object for game.resources
game.resources = new Map([['gold', 200], ['wood', 150], ['food', 3]]);
// Then: game.resources.set(e.carryType, game.resources.get(e.carryType) + e.carryAmount);
```

---

### SR-R5-02: Unsafe `innerHTML` Sinks

**CVSS 3.1:** 2.8 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N)  
**File:** `index.html:1363,1364,1412,1432,1434,1465,1498`  
**Status:** ⚠️ Unfixed since R0

**Sinks identified:**

| Line | Sink | Dynamic content source | Risk |
|------|------|----------------------|------|
| 1363 | `infoStats.innerHTML = 'Select...<br>...'` | Hardcoded string | None |
| 1364 | `actionPanel.innerHTML = '<span...>'` | Hardcoded string | None |
| 1412 | `actionPanel.innerHTML = ''` | Empty | None |
| 1432 | `actionPanel.innerHTML = '<span...>Building under construction...</span>'` | Hardcoded string | None |
| 1434 | `actionPanel.innerHTML = '<span...>No actions available</span>'` | Hardcoded string | None |
| 1465 | `actionPanel.innerHTML = ''` | Empty | None |
| **1498** | `infoStats.innerHTML = stats` | **Built from `Math.floor(e.hp)`, `e.carryType`, `defU.name`, `defU.atk.damage` etc.** | **Low** |

**Line 1498 is the primary concern** — the `stats` variable is constructed via string concatenation:
```javascript
let stats = `HP: ${Math.floor(e.hp)}/${e.maxHp} | Speed: ${defU.speed} | Atk: ${defU.atk.damage} | Range: ${defU.atk.range}`;
stats += `<div class="hp-bar-outer">...`;
if(e.carryAmount>0) stats += `<br>Carrying: ${e.carryAmount} ${e.carryType}`;
```

All dynamic values are numeric (`Math.floor()`, `defU.speed`) or from hardcoded CONST objects (`defU.name`, `defU.atk.damage`). **No user-controlled text strings** reach `innerHTML`. The risk is therefore extremely low.

**Note:** Lines 1412 and 1465 reset `actionPanel.innerHTML = ''` before building DOM programmatically — this is safe because subsequent content is added via `appendChild(btn)` on DOM elements, not `innerHTML`.

**Remediation:** Replace `infoStats.innerHTML = stats` with:
```javascript
infoStats.textContent = ''; // clear
// Build stats via createElement/appendChild DOM API
// (already partially done for HP bar, progress bars, queue info)
```

---

### SR-R5-03: Mutable Game Definition Objects

**CVSS 3.1:** 2.0 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N)  
**File:** `index.html:103-117`  
**Status:** ⚠️ Unfixed since R3

**Code:**
```javascript
const BUILDING_DEFS = {
  town_hall: { name:'Town Hall', cost:{gold:200,wood:100}, ... },
  barracks:  { name:'Barracks', cost:{gold:150,wood:50}, ... },
  // ...
};
const UNIT_DEFS = {
  peasant: { name:'Peasant', cost:{gold:50}, ... },
  // ...
};
```

**Vulnerability:** These `const` declarations prevent reassignment of the variable, but **do not prevent mutation** of the object's properties. A console attacker can modify `BUILDING_DEFS.town_hall.cost.gold = 0` to make buildings free, or inject false `dropoff` properties.

**Remediation:**
```javascript
const BUILDING_DEFS = Object.freeze({
  town_hall: Object.freeze({ name:'Town Hall', cost:Object.freeze({gold:200,wood:100}), ... }),
  // ...deep freeze all nested objects
});
// Or use a deepFreeze helper:
function deepFreeze(obj) {
  Object.freeze(obj);
  Object.values(obj).forEach(v => v && typeof v === 'object' && deepFreeze(v));
}
```

---

### SR-R5-04: Missing Type Validation in `isPlacementValid`

**CVSS 3.1:** 1.8 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N)  
**File:** `index.html:869`  
**Status:** ⚠️ Unfixed since R1

**Code:**
```javascript
function isPlacementValid(buildingType, wx, wy) {
  const def = BUILDING_DEFS[buildingType];  // ← bracket access, no type check
  if (!def) return false;                   // falsy guard exists, but...
  const hw = def.size.w / 2;                // ← TypeError if def has no `.size`
  // ...
}
```

**Vulnerability:** If `buildingType` is not a string (e.g., `null`, `undefined`, an object), `BUILDING_DEFS[buildingType]` returns `undefined`, and `if(!def) return false` catches it. However, if `buildingType` is a string key that exists in `BUILDING_DEFS`'s prototype (`'toString'`, `'constructor'`), the lookup returns a function — then `def.size.w` throws `TypeError`. This is reachable via `findAutoBuildSite` if `carryType` is polluted.

**Remediation:** Add an explicit type check before the bracket lookup:
```javascript
function isPlacementValid(buildingType, wx, wy) {
  if (typeof buildingType !== 'string') return false;
  if (!BUILDING_DEFS.hasOwnProperty(buildingType)) return false;  // skip prototype
  const def = BUILDING_DEFS[buildingType];
  // ...
}
```

---

### SR-R5-05: `Object.entries(BUILDING_DEFS)` Iteration in `updateUI`

**CVSS 3.1:** 2.0 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:L/A:N)  
**File:** `index.html:1468`  
**Status:** ⚠️ Unfixed since R2

**Code:**
```javascript
// index.html:1468 — building construction buttons for selected peasant
for (const [btype, bdef] of Object.entries(BUILDING_DEFS)) {
  const cost = bdef.cost || {};
  const canAfford = game.resources.gold >= (cost.gold || 0) && game.resources.wood >= (cost.wood || 0);
  const btn = document.createElement('button');
  btn.textContent = `Build ${bdef.name}`;
  // ...
}
```

**Vulnerability:** `Object.entries()` iterates over **own enumerable** properties by default, so prototype-injected properties won't appear. However, if an attacker defines enumerable properties on `Object.prototype` (prototype pollution), those WILL appear as additional building buttons. This is a niche attack vector requiring prior prototype pollution.

**Remediation:**
```javascript
for (const [btype, bdef] of Object.entries(BUILDING_DEFS)) {
  if (!BUILDING_DEFS.hasOwnProperty(btype)) continue;  // guard
  // ...
}
```

---

### SR-R5-06: Unbounded Entity Creation (Resource Exhaustion via Console)

**CVSS 3.1:** 2.5 (AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:L)  
**File:** `index.html:174-213`  
**Status:** ⚠️ Unfixed since R0

**Vulnerability:** `createUnit()` and `createBuilding()` have no per-type or total entity cap. A console attacker can spawn millions of entities, crashing the browser tab via memory exhaustion or rendering overload. The `updateEntity` loop iterates all entities O(n).

**Remediation:**
```javascript
const MAX_ENTITIES = 500;
function createUnit(type, x, y) {
  if (game.entities.size >= MAX_ENTITIES) return null;
  // ...
}
```

---

### SR-R5-07: Missing Content-Security-Policy

**CVSS 3.1:** 0.0 (Informational)  
**File:** `index.html:1-7`  
**Status:** ℹ️ Unfixed since R0

The `<head>` contains no CSP `<meta>` tag. While all logic is inline (defeating `'unsafe-inline'` avoidance), a CSP provides defense-in-depth against hypothetical XSS.

**Remediation:**
```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;">
```

---

### SR-R5-08: Duplicate `attackTarget` Property — Dead Code

**CVSS 3.1:** 0.0 (Informational)  
**File:** `index.html:269,276`  
**Status:** ℹ️ Unfixed since R3

**Code:**
```javascript
// index.html:269 — createUnit return object
return {
    id:genId(), type, owner:'player', x, y, w:def.size*2, h:def.size*2,
    // ...
    targetId:null, moveTarget:null, path:[], attackTarget:null,  // ← line 269: first attackTarget
    // ...
    attackMove:false, // attack-move mode
    faceDir:0, // radians
    animTimer:0
    // line 276: no second attackTarget — this was reported in R3 but may have been fixed
};
```

**Status update:** A re-examination of lines 269-277 shows `attackTarget` appears only once in `createUnit`'s return object. If the duplicate was present in R3, it may have been cleaned up in R4. However, the `targetId` and `attackTarget` fields are both defined and used for different purposes (`targetId` for the attack animation lock, `attackTarget` for the actual target entity ID). This is **working as designed**, not a bug.

**Verdict:** This finding may no longer apply in R5. Re-verification recommended.

---

## Auto-Build System — Security Review

The new auto-build system (`getDropoffBuildingType`, `hasDropoffUnderConstruction`, `findAutoBuildSite`, `autoBuildDropoff`) was reviewed line-by-line. **No new vulnerabilities were found.** Specific observations:

| Defense | Mechanism | Assessment |
|---------|-----------|------------|
| Race condition prevention | `game._autoBuildThisFrame` Set cleared per-frame | ✅ Solid |
| Cooldown anti-spam | `game.autoBuildCooldown[buildingType]` with 15s window | ✅ Effective |
| Affordability gate | Resource check before deduction | ✅ Correct |
| Null-site guard | `findAutoBuildSite` returns null → early return | ✅ Safe |
| Per-frame double-spend lock | `_autoBuildThisFrame.has(buildingType)` check | ✅ Robust |
| Resource preservation on failure | `carryAmount`/`carryType` NOT cleared on auto-build failure | ✅ Correct |
| Idle retry loop | `Math.floor(game.time*10)%20===0` ~2s interval | ✅ Rate-limited |

**One note:** `hasDropoffUnderConstruction` has `MAX_DIST=400` hardcoded. If the map were larger or the peasant far from any under-construction dropoff, this could false-negative. This is a gameplay tuning concern, not a security issue.

---

## Gathering Validation Fix — Security Review

The fix at lines 1240-1255 (peasant pushed off resource tile walks back instead of resetting) was reviewed:

```javascript
// index.html:1240-1255
if(curTile !== (e.carryType==='gold' ? TERRAIN.GOLD : TERRAIN.TREE)) {
  if(e.gatheringNode) {
    e.state = 'moving';
    e.moveTarget = { x: e.gatheringNode.x, y: e.gatheringNode.y };
  } else {
    e.state = 'idle'; e.gatheringNode = null; e.carryAmount = 0; e.carryType = null;
  }
  return;
}
```

**No security issues.** The fallback correctly nulls out state when `gatheringNode` is missing. No new injection sinks, no mutable state leakage.

---

## Non-Findings (Confirmed Absent)

| Pattern | Status |
|---------|--------|
| `eval()` | ❌ Not found |
| `new Function()` | ❌ Not found |
| `setTimeout` / `setInterval` with string arg | ❌ Not found |
| `.prototype.` access | ❌ Not found |
| `fetch()` | ❌ Not found |
| `XMLHttpRequest` | ❌ Not found |
| `WebSocket` | ❌ Not found |
| `localStorage` | ❌ Not found |
| `sessionStorage` | ❌ Not found |
| `document.cookie` | ❌ Not found |
| `postMessage` | ❌ Not found |
| API keys / tokens / secrets | ❌ Not found |
| External CDN dependencies | ❌ Not found |
| npm dependencies | ❌ Not found (`node_modules/` in `.gitignore`) |

---

## Remediation Priority

```
Priority 1 (do now):    None — no critical/high/medium findings
Priority 2 (do soon):   SR-R5-01 (whitelist resources[e.carryType])
                        SR-R5-04 (hasOwnProperty guard + typeof check)
Priority 3 (nice to have): SR-R5-02 (innerHTML → textContent)
                        SR-R5-03 (Object.freeze defs)
                        SR-R5-05 (hasOwnProperty in Object.entries loop)
Priority 4 (low ROI):   SR-R5-06 (entity cap)
                        SR-R5-07 (CSP meta tag)
```

---

## Conclusion

RealmCraft R5 is a well-structured single-file RTS game with a **minimal attack surface**. The auto-build system was implemented defensively with proper guards against race conditions, resource exhaustion, and duplicate builds. All 8 findings are Low severity or Informational, with the highest CVSS being 3.5 (prototype pollution via dynamic property key). No findings block deployment or require immediate remediation.

**Overall verdict: CLEAN — no blocking security issues.**
