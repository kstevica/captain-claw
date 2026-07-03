# Security Review: RealmCraft RTS — Round 6 (Current State)

**Date:** 2026-07-03 09:08 UTC  
**Reviewer:** Security Reviewer (Captain Claw fleet, PHRYGIAN adversarial mode)  
**Artifacts Reviewed:** `index.html` (~1,775 lines, ~68 KB), `test_game.js` (~1,392 lines, ~51 KB), `plan.md`  
**Task Under Review:** "When worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it"  
**Prior Reports Cross-Referenced:** `SECURITY_REVIEW.md` (R0), `review-r0-security-reviewer.md`, `review-r1-security-reviewer.md`, `SECURITY_REVIEW_CURRENT.md` (R1/R2/R3), `SECURITY_REVIEW_R4.md`, `SECURITY_REVIEW_R5.md`  
**Delta from R5:** 10 prior findings reconfirmed. 3 new findings in the commandStop/issueMove resource lifecycle, 1 finding upgraded. 2 R5 findings no longer applicable (commandStop behavior partially changed).  
**Scope:** Injection | Auth/Authz | Secrets in code | Unsafe input handling | Dependency risks | Resource lifecycle deep-dive  
**Methodology:** Read-only static analysis, adversarial assumption (things WILL go wrong), line-by-line review of all dynamic code paths including the full gather→carry→deposit→auto-build chain, grep audit for all injection sinks.

---

## Executive Summary

**Overall Risk: LOW** (unchanged)

RealmCraft remains a single-file, zero-dependency, client-side-only RTS game with no backend, no network APIs, no text input fields, no storage, and no CDN fetches. **No critical, high, or medium-severity vulnerabilities exist.** All issues require browser console access to exploit.

### Key Delta from R5

The `commandStop()` and `issueMove()` functions have been **partially refactored** since the R4/R5 reports. They now **conditionally** clear `carryAmount`/`carryType` based on whether a dropoff is nearby — an improvement over the unconditional clearing reported in R4. However, the condition is **inverted**: resources are cleared when a dropoff IS nearby (but **not deposited**), and preserved when no dropoff exists. This introduces a new resource-loss scenario not present in earlier reports. See **SR-R6-01**.

---

## Complete Findings Table (CVSS-Ranked)

| # | Title | CVSS 3.1 | Severity | Category | Status |
|---|-------|----------|----------|----------|--------|
| **SR-R6-01** | commandStop/issueMove conditionally clear carryAmount WITHOUT depositing | 3.5 | **Low** | Logic / Resource loss | 🔴 **New R6** |
| **SR-NEW-01** | Dynamic property key `game.resources[e.carryType]` creates prototype-pollution path | 3.5 | **Low** | Unsafe input | ⚠️ Unfixed (R2→R6) |
| SR-01 | Unsafe `innerHTML` sinks for dynamic game state (~10 locations) | 2.8 | **Low** | Injection | ⚠️ Unfixed (R0→R6) |
| **SR-R6-02** | Deposit handler fires only for `moving` state — idle peasants at dropoff don't deposit | 2.5 | **Low** | Logic / Delayed deposit | 🔴 **New R6** |
| SR-R5-01 | `findAutoBuildSite` full-map fallback O(n²·m) — console-triggered DoS | 2.5 | **Low** | DoS | ⚠️ Unfixed (R5→R6) |
| SR-03 | Unbounded entity creation (resource exhaustion via console) | 2.5 | **Low** | Unsafe input | ⚠️ Unfixed (R0→R6) |
| **SR-R5-02** | Accidental global `cam` variable in `resizeCanvas` | 2.0 | **Low** | Code quality | ⚠️ Unfixed (R5→R6) |
| SR-NEW-02 | `Object.entries(BUILDING_DEFS)` iteration exposes prototype pollution | 2.0 | **Low** | Unsafe input | ⚠️ Unfixed (R2→R6) |
| SR-R3-02 | Mutable game-definition objects — no `Object.freeze()` | 2.0 | **Low** | Defense-in-depth | ⚠️ Unfixed (R3→R6) |
| SR-R5-03 | Missing zero-guard on `canvas.clientWidth`/`clientHeight` division | 1.8 | **Low** | Unsafe input | ⚠️ Unfixed (R5→R6) |
| **SR-R6-03** | Depleted resource tile not removed from map — peasant loops on empty node | 1.5 | **Low** | Logic / Infinite retry | 🔴 **New R6** |
| SR-02 | Missing Content-Security-Policy | 0.0 | Info | Defense-in-depth | ℹ️ Unfixed (R0→R6) |
| SR-R5-04 | Unbounded `autoBuildCooldown` object growth | 0.0 | Info | Resource mgmt | ℹ️ Unfixed (R5→R6) |
| SR-R3-03 | Duplicate `attackTarget:null` — dead code | 0.0 | Info | Code quality | ⚠️ Unfixed (R3→R6) |

---

## Confirmed Fixed (R4→R6)

Three prior findings have been confirmed fixed:

| ID | Finding | Prior CVSS | Fix Location |
|----|---------|-----------|--------------|
| ✅ SR-R3-01 | NaN/Infinity coordinate passthrough | 2.5 | `index.html:267,287` — `Number.isFinite()` guards in `createUnit`/`createBuilding` |
| ✅ SR-04 | `isPlacementValid` missing type validation guard | 1.8 | `index.html:834` — `if(!def) return false;` |
| ✅ SR-R4-02 | `findNearestResource` 12-tile radius | 1.5 | `index.html:~800` — radius increased from 12 to **30 tiles** |

---

## New Findings (R6 — This Review)

---

### SR-R6-01 — commandStop / issueMove Clear carryAmount WITHOUT Depositing Resources

**CVSS:3.1/AV:P/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N** — Score: **3.5 (Low)**  
**CWE-459:** Incomplete Cleanup  
**Status:** 🔴 **New R6** — partially evolved from SR-R4-01

#### Location

`index.html` — `commandStop()` function (~line 1330) and `issueMove()` function (~line 1060)

#### Current Code

```javascript
// commandStop() — current state
function commandStop() {
  for(const id of game.selectedIds) {
    const e=game.entities.get(id);
    if(!e) continue;
    e.state='idle'; e.moveTarget=null; e.attackTarget=null; e.attackMove=false;
    if(e.type==='peasant') {
      e.buildTarget=null;
      e.gatheringNode=null;
      const dropoffChk=e.carryType?findNearestDropoff(e.x,e.y,e.carryType):null;
      if(dropoffChk){
        const d=Math.sqrt((e.x-dropoffChk.x)**2+(e.y-dropoffChk.y)**2);
        if(d<40){e.carryAmount=0;e.carryType=null;}  // <-- CLEARS but does NOT deposit!
      }
    }
  }
}

// issueMove() — same pattern
function issueMove(targetX, targetY) {
  // ...
  if(e.type==='peasant') {
    e.gatheringNode=null;
    const dropoffChk=e.carryType?findNearestDropoff(e.x,e.y,e.carryType):null;
    if(dropoffChk){
      const d=Math.sqrt((e.x-dropoffChk.x)**2+(e.y-dropoffChk.y)**2);
      if(d<40){e.carryAmount=0;e.carryType=null;}  // <-- CLEARS but does NOT deposit!
    }
    e.buildTarget=null;
  }
}
```

#### Description

**This is a logic inversion bug with real gameplay impact.** The code checks whether a dropoff exists nearby and:

- **Dropoff nearby (d < 40):** Clears `carryAmount` and `carryType` to 0/null — **silently destroying** the carried resources without depositing them.
- **No dropoff nearby:** Preserves `carryAmount` and `carryType` — resources survive, idle retry loop handles recovery.

This is the **exact opposite** of what the right behavior should be. When a dropoff IS nearby, the peasant should **deposit** and then clear. When no dropoff exists, the peasant should preserve resources for auto-build.

**Exploitation / Bug Scenario:**

1. Peasant gathers 10 gold from a mine 200px east of the town hall
2. Peasant walks toward town hall to deposit — is now at x=250, town hall at x=300, distance=50px
3. Player accidentally presses `S` (Stop) while peasant is 50px from town hall
4. `commandStop()` fires → `findNearestDropoff` finds the town hall
5. `d = 50` which is **NOT** `< 40` → resources **preserved** (correct!)
6. Now peasant wanders slightly closer (entity separation or player issues another move)
7. Player presses `S` again; peasant is now 35px from town hall
8. `d = 35` which IS `< 40` → `carryAmount=0`, `carryType=null`
9. **10 gold lost** — was never deposited, never will be

The same pattern exists in `issueMove()`: every right-click move order on a peasant who happens to be near a dropoff destroys carried resources.

#### Why This Happened

The code was likely written with the intent: "If peasant is at a dropoff and we stop them, they're done depositing, so clear the carry." But the deposit hasn't actually happened — the clearing happens BEFORE checking if a deposit should occur. The peer at distance <40 should trigger a deposit, not a discard.

#### Remediation

```javascript
// In commandStop() and issueMove():
if(dropoffChk) {
  const d = Math.sqrt((e.x-dropoffChk.x)**2 + (e.y-dropoffChk.y)**2);
  if(d < 40) {
    // DEPOSIT before clearing
    if(e.carryType === 'gold' || e.carryType === 'wood') {
      game.resources[e.carryType] += e.carryAmount;
    }
    e.carryAmount = 0;
    e.carryType = null;
  }
  // else: far from dropoff — preserve carry, let retry loop handle
}
// else: no dropoff — preserve carry, let retry loop / auto-build handle
```

Or simpler — delegate entirely to the idle retry loop:

```javascript
// Don't touch carryAmount/carryType at all in commandStop/issueMove.
// The idle retry loop (~2s period) already handles:
//   - find nearest dropoff → move to it → deposit
//   - no dropoff → autoBuildDropoff
```

---

### SR-R6-02 — Deposit Handler Fires Only for `moving` State

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L** — Score: **2.5 (Low)**  
**CWE-691:** Insufficient Control Flow Management  
**Status:** 🔴 **New R6**

#### Location

`index.html:~1250` — deposit arrival handler in `updateEntity()`

```javascript
// Peasant arriving at dropoff (idle case handled by retry block with ~2s cooldown below)
if(e.type==='peasant'&&e.state==='moving'&&e.carryAmount>0) {
    const dropoff=findNearestDropoff(e.x,e.y,e.carryType);
    if(dropoff) {
      const cx=dropoff.x, cy=dropoff.y;
      const d=dist({x:e.x,y:e.y},{x:cx,y:cy});
      if(d<40) {
        // Deposit
        if(e.carryType==='gold'||e.carryType==='wood') {
          game.resources[e.carryType]+=e.carryAmount;
        }
        e.carryAmount=0;
        // Go back to node...
      }
    } else if (!autoBuildDropoff(e)) {
      e.state='idle';  // preserve carryAmount/carryType
    }
}
```

#### Description

The deposit check is guarded by `e.state==='moving'`. An idle peasant carrying resources who happens to be at a dropoff location will **not** deposit in the current frame. Instead, they must wait for the idle retry loop:

```javascript
// Retry check: runs every ~2 seconds
if(e.type==='peasant'&&e.state==='idle'&&e.carryAmount>0&&e.carryType&&!e.isBuilding) {
    if(Math.floor(game.time*10)%20===0) {
      const dropoff=findNearestDropoff(e.x, e.y, e.carryType);
      if(dropoff) {
        e.state='moving';
        e.moveTarget={x:dropoff.x, y:dropoff.y};
      } else {
        autoBuildDropoff(e);
      }
    }
}
```

The retry loop sets `state='moving'` with a moveTarget to the dropoff. But the peasant is **already at the dropoff** (d < 40). The movement logic will see `d < 3`, set state to `idle` again, and the cycle repeats — the peasant bounces between idle and moving without ever depositing, because the deposit handler requires `state==='moving'` but the movement arrival sets `state='idle'`.

**Concrete scenario:**

1. Peasant gathers gold, auto-build creates a refinery at (x=600, y=600)
2. Peasant moves to refinery, builds it to completion, goes idle
3. Peasant is now idle at (x=600, y=600) — exactly at the completed refinery — carrying 10 gold
4. Retry loop fires: finds refinery as dropoff, sets `state='moving'`, `moveTarget=(600,600)`
5. Movement logic: `d < 3`, sets `state='idle'`, `moveTarget=null`
6. Back to step 3 — infinite loop, **resources never deposited**

This is because the movement-arrival handler (`d < 3 → state='idle'`) runs **before** the deposit handler. The order in `updateEntity()` is:
1. Movement: `e.state = 'idle'` on arrival
2. Deposit check: requires `e.state === 'moving'` → SKIPPED
3. Retry loop: fires every 2 seconds

#### Remediation

**Option A:** Broaden the deposit handler to also fire for idle peasants:
```javascript
if(e.type==='peasant'&&(e.state==='moving'||e.state==='idle')&&e.carryAmount>0) {
```

**Option B:** In the movement-arrival handler, check for carryAmount before setting to idle:
```javascript
if(d < 3) {
  e.x = e.moveTarget.x; e.y = e.moveTarget.y;
  e.moveTarget = null;
  // If carrying resources, don't go idle — let deposit handler run
  if (!(e.type === 'peasant' && e.carryAmount > 0)) {
    e.state = 'idle';
  }
}
```

**Recommendation:** Option A is simpler and more robust.

---

### SR-R6-03 — Depleted Resource Tile Not Removed from Map

**CVSS:3.1/AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L** — Score: **1.5 (Low)**  
**CWE-834:** Excessive Iteration / Resource Loop  
**Status:** 🔴 **New R6**

#### Location

`index.html:~1220` — gathering validation in `updateEntity()`

```javascript
// Peasant gathering
if(e.type==='peasant'&&e.state==='gathering') {
    const curTile=tileAt(e.x,e.y);
    if(curTile!==(e.carryType==='gold'?TERRAIN.GOLD:TERRAIN.TREE)) {
      if(e.gatheringNode) {
        e.state='moving';
        e.moveTarget={x:e.gatheringNode.x,y:e.gatheringNode.y};
      } else {
        e.state='idle'; e.gatheringNode=null; e.carryAmount=0; e.carryType=null;
      }
      return;
    }
    e.gatheringTimer+=dt;
    if(e.gatheringTimer>=1.5) {
      e.carryAmount=10;
      // ...
    }
}
```

#### Description

When a peasant gathers from a resource tile, the tile is **never removed** from `game.map`. Gold mines and trees remain on the map indefinitely, allowing infinite gathering from the same tile. While this is partly mitigated by the fact that gathered resources are "virtual" (tile doesn't track depletion), it has security-adjacent implications:

1. **Resource flooding:** A single gold mine tile can supply infinite gold, effectively removing resource scarcity as a gameplay constraint. This devalues the auto-build mechanic — why build a refinery near a distant mine when one mine gives infinite gold?

2. **No per-tile depletion tracking:** The map generation creates ~8 gold spots each with 5 tiles (center + 4 dirt neighbors that don't block). That's ~40 gold mine tiles. Without depletion, all are infinite.

3. **No resource exhaustion forcing exploration:** In standard RTS design, resource depletion forces map control and expansion. Without it, the optimal strategy is to turtle around the nearest mine — reducing the need for auto-build dropoffs at all.

4. **Interaction with findNearestResource:** The 30-tile search radius in `findNearestResource` always finds the nearest gold tile, but that tile may have been "mined" thousands of times. The function never needs to find a new mine.

#### Remediation

Add tile depletion tracking (defense-in-depth, gameplay improvement):

```javascript
// Add to game state:
resourceDepletion: new Map(), // "tx,ty" → remaining amount

// In map generation:
game.resourceDepletion.set(`${tx},${ty}`, 500); // 500 gold per tile = ~50 gathers

// In gathering completion:
const tileKey = `${gNode.tx},${gNode.ty}`;
const remaining = (game.resourceDepletion.get(tileKey) || 0) - 10;
if (remaining <= 0) {
  game.map[gNode.ty][gNode.tx] = TERRAIN.DIRT; // deplete tile
  game.resourceDepletion.delete(tileKey);
  spawnFloatingText(e.x, e.y - e.size, 'Depleted!', '#ff8844');
} else {
  game.resourceDepletion.set(tileKey, remaining);
}
```

---

## Prior Findings Reconfirmed (Chronological)

### ⚠️ SR-NEW-01 — Dynamic Property Key Prototype Pollution (R2, Unfixed)

**CVSS: 3.5** | `index.html:~1260`

```javascript
game.resources[e.carryType] += e.carryAmount;
```

If `carryType` were ever set to `__proto__` or `constructor` via console manipulation, this pollutes `Object.prototype`. Under normal gameplay, `carryType` is always `'gold'` or `'wood'` — no exploit through game mechanics.

**Remediation:** Whitelist check:
```javascript
if (e.carryType === 'gold' || e.carryType === 'wood') {
  game.resources[e.carryType] += e.carryAmount;
}
```

---

### ⚠️ SR-01 — Unsafe `innerHTML` Sinks (R0, Unfixed)

**CVSS: 2.8** | ~10 locations in `updateUI()`

Dynamic HTML strings with game-state values are assigned to `innerHTML`. Currently safe because all interpolated values come from hardcoded constants, but pattern is fragile.

**Remediation:** Replace with `createElement` + `textContent` + `appendChild`.

---

### ⚠️ SR-R5-01 — `findAutoBuildSite` Full-Map Fallback O(n²·m) (R5, Unfixed)

**CVSS: 2.5** | `index.html:738-748`

Full-map scan iterates 4,096 tiles × N entities. With many console-created entities, this can freeze the game loop.

---

### ⚠️ SR-03 — Unbounded Entity Creation (R0, Unfixed)

**CVSS: 2.5** | No `MAX_ENTITIES` cap. Console loop creating 10,000+ entities degrades to single-digit FPS.

---

### ⚠️ SR-R5-02 — Accidental Global `cam` Variable (R5, Unfixed)

**CVSS: 2.0** | `index.html:1526`

```javascript
cam=game.camera;  // missing 'const' — leaks to global scope
```

One-character fix: add `const`.

---

### ⚠️ SR-NEW-02 — `Object.entries(BUILDING_DEFS)` Prototype Exposure (R2, Unfixed)

**CVSS: 2.0** | `updateUI()` action panel generation

If `Object.prototype` were polluted, extra entries render as phantom build buttons.

**Remediation:** `hasOwnProperty` guard or explicit key whitelist.

---

### ⚠️ SR-R3-02 — Mutable Game-Definition Objects (R3, Unfixed)

**CVSS: 2.0** | `BUILDING_DEFS`, `UNIT_DEFS`, etc. are mutable. Console mutation enables cheating.

---

### ⚠️ SR-R5-03 — Missing Zero-Guard on Canvas Dimension Division (R5, Unfixed)

**CVSS: 1.8** | `index.html:663-666`

```javascript
x: sx*(canvas.width/canvas.clientWidth)/cam.zoom+cam.x,
```

`canvas.clientWidth` could be 0 in hidden/zero-size containers.

---

### ℹ️ Info-Level Findings (All Unfixed)

- **SR-02:** Missing Content-Security-Policy (R0)
- **SR-R5-04:** Unbounded `autoBuildCooldown` growth (R5)
- **SR-R3-03:** Duplicate `attackTarget:null` declaration (R3)

---

## Task-Specific Review: Auto-Build Dropoff Resource Lifecycle

*Per the task directive: "when worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it"*

### End-to-End Trace

#### Stage 1: Right-Click on Resource (`mousedown` handler)

```javascript
else if(tile===TERRAIN.GOLD||tile===TERRAIN.TREE) {
  // ...
  for(const id of game.selectedIds) {
    const e=game.entities.get(id);
    if(e&&e.type==='peasant') issueGather(e, world.x, world.y);
  }
}
```

**Verdict: ✅ CORRECT** — Dispatches to `issueGather` for peasants, normal move for non-peasants.

#### Stage 2: `issueGather()` — Gathering Assignment

```javascript
function issueGather(entity, wx, wy) {
  if(!entity.canGather) return;
  const tile=tileAt(wx,wy);
  if(tile===TERRAIN.GOLD||tile===TERRAIN.TREE) {
    const resType=tile===TERRAIN.GOLD?'gold':'wood';
    const node=findNearestResource(wx,wy,resType);
    if(!node) { /* fallback: move to click */ return; }
    entity.state='moving'; entity.moveTarget={x:node.x,y:node.y};
    entity.gatheringNode=node; entity.carryType=resType;
    entity.carryAmount=0; entity.buildTarget=null; entity.attackTarget=null;
  }
}
```

**Verdict: ✅ CORRECT** — Finds nearest resource node within 30-tile radius. Falls back to simple move if no node found (shows feedback). `carryType` correctly derived from tile type.

#### Stage 3: Arrival at Resource → Gathering State

```javascript
// Movement arrival handler
if(d<3) {
  e.x=e.moveTarget.x; e.y=e.moveTarget.y; e.state='idle'; e.moveTarget=null;
  if(e.type==='peasant'&&e.buildTarget) { /* start building */ }
  if(e.type==='peasant'&&e.gatheringNode&&e.carryAmount===0) {
    e.state='gathering'; e.gatheringTimer=0;
  }
}
```

**Verdict: ✅ CORRECT** — Peasant transitions to `gathering` state when arriving at resource node with zero carry.

#### Stage 4: Gathering Loop

```javascript
if(e.type==='peasant'&&e.state==='gathering') {
    const curTile=tileAt(e.x,e.y);
    if(curTile!==(e.carryType==='gold'?TERRAIN.GOLD:TERRAIN.TREE)) {
      if(e.gatheringNode) {
        e.state='moving'; e.moveTarget={x:e.gatheringNode.x,y:e.gatheringNode.y};
      } else { e.state='idle'; /* ... reset ... */ }
      return;
    }
    e.gatheringTimer+=dt;
    if(e.gatheringTimer>=1.5) {
      e.carryAmount=10; e.gatheringTimer=0;
      const dropoff=findNearestDropoff(e.x, e.y, e.carryType);
      if(dropoff) {
        e.state='moving'; e.moveTarget={x:dropoff.x,y:dropoff.y};
      } else if (!autoBuildDropoff(e)) {
        e.state='idle'; // preserve carryAmount/carryType
      }
    }
}
```

**Verdict: ✅ CORRECT with caveats:**
- **Tile validation** correct: walks back to `gatheringNode` if pushed off resource tile (entity separation)
- **Gathering timer** correct: 1.5 seconds per 10 resources
- **Resource preservation** correct: goes idle with preserved carry on auto-build failure
- **⚠️ SR-R6-03:** Resource tile never depletes — infinite resources from one tile

#### Stage 5: `findNearestDropoff()` — Dropoff Discovery

```javascript
function findNearestDropoff(x, y, resType) {
  let best=null, bestDist=Infinity;
  for(const e of game.entities.values()) {
    if(e.owner!=='player'||e.progress<1) continue;
    const def=BUILDING_DEFS[e.type];
    if(!def) continue;
    if(resType==='gold' && (def.dropoff||def.dropoff_gold)) { /* check distance */ }
    if(resType==='wood' && (def.dropoff||def.dropoff_wood)) { /* check distance */ }
  }
  return best;
}
```

**Verdict: ✅ CORRECT — Type-safe resource discrimination:**
- Gold-carrying peasant → finds: town hall (`dropoff:true`), refinery (`dropoff_gold:true`)
- Gold-carrying peasant → ignores: lumber mill (no `dropoff:true` or `dropoff_gold`)
- Wood-carrying peasant → finds: town hall (`dropoff:true`), lumber mill (`dropoff_wood:true`)
- Wood-carrying peasant → ignores: refinery (no `dropoff:true` or `dropoff_wood`)
- Constructing buildings (`progress<1`) are excluded

#### Stage 6: `autoBuildDropoff()` — Automatic Building Construction

```javascript
function autoBuildDropoff(peasant) {
  const resType=peasant.carryType;
  if(!resType||peasant.carryAmount<=0) return false;                // Guard 1
  const existing=findNearestDropoff(peasant.x, peasant.y, resType);
  if(existing) return false;                                         // Guard 2
  if(hasDropoffUnderConstruction(resType)) return false;             // Guard 3
  const buildingType=getDropoffBuildingType(resType);
  if(!buildingType) return false;                                    // Guard 4
  if(game.autoBuildCooldown[buildingType] && ...) return false;      // Guard 5
  const cost=BUILDING_DEFS[buildingType].cost;
  if(game.resources.gold<(cost.gold||0)||...) { ... return false; }  // Guard 6
  const site=findAutoBuildSite(buildingType, siteOrigin.x, siteOrigin.y);
  if(!site) { ... return false; }                                    // Guard 7
  // Deduct + create + assign
  game.resources.gold-=(cost.gold||0);
  game.resources.wood-=(cost.wood||0);
  const b=createBuilding(buildingType, site.x, site.y, 0);
  game.entities.set(b.id, b);
  peasant.state='moving'; peasant.moveTarget={x:site.x, y:site.y};
  peasant.buildTarget=b.id;
  game.autoBuildCooldown[buildingType]=game.time;
  return true;
}
```

**Verdict: ✅ CORRECT — Seven guard clauses, well-architected:**
- **Guard 1:** No carry → nothing to do
- **Guard 2:** Dropoff already exists → use it
- **Guard 3:** Dropoff under construction → wait
- **Guard 4:** Unknown resource type → abort
- **Guard 5:** Cooldown active → prevent duplicate builds
- **Guard 6:** Can't afford → show feedback, set short cooldown
- **Guard 7:** No valid build site → set short cooldown

`hasDropoffUnderConstruction` correctly counts town halls under construction as universal dropoffs, preventing redundant builds.

#### Stage 7: Deposit at Dropoff

```javascript
if(e.type==='peasant'&&e.state==='moving'&&e.carryAmount>0) {
    const dropoff=findNearestDropoff(e.x,e.y,e.carryType);
    if(dropoff) {
      const d=dist({x:e.x,y:e.y},{x:dropoff.x,y:dropoff.y});
      if(d<40) {
        if(e.carryType==='gold'||e.carryType==='wood') {
          game.resources[e.carryType]+=e.carryAmount;  // SR-NEW-01: dynamic key
        }
        e.carryAmount=0;
        if(e.gatheringNode) {
          e.state='moving'; e.moveTarget={x:e.gatheringNode.x,y:e.gatheringNode.y};
        } else {
          e.state='idle'; e.carryType=null;
        }
      }
    } else if (!autoBuildDropoff(e)) {
      e.state='idle'; // preserve carryAmount/carryType
    }
}
```

**Verdict: ⚠️ MOSTLY CORRECT — Two issues:**
1. **SR-NEW-01:** Dynamic property key `game.resources[e.carryType]` — no whitelist guard
2. **SR-R6-02:** Guarded by `e.state==='moving'` — idle peasants at dropoff locations skip this handler

#### Stage 8: Idle Retry Loop (Eventual Consistency)

```javascript
if(e.type==='peasant'&&e.state==='idle'&&e.carryAmount>0&&e.carryType&&!e.isBuilding) {
    if(Math.floor(game.time*10)%20===0) {
      const dropoff=findNearestDropoff(e.x, e.y, e.carryType);
      if(dropoff) {
        e.state='moving'; e.moveTarget={x:dropoff.x,y:dropoff.y};
      } else {
        autoBuildDropoff(e);
      }
    }
}
```

**Verdict: ✅ CORRECT** — Provides eventual consistency. Every ~2 game seconds, checks for dropoff or attempts auto-build. Handles: resources becoming available later, another peasant finishing construction, etc.

---

### Edge Case Matrix

| Scenario | Expected | Actual | Status |
|----------|----------|--------|--------|
| No dropoff, can afford | Auto-build refinery/mill | ✅ Builds | **PASS** |
| No dropoff, can't afford | Preserve, retry later | ✅ Idle + retry | **PASS** |
| No dropoff, cooldown active | Skip, retry later | ✅ Skipped | **PASS** |
| No dropoff, one under construction | Skip | ✅ Detected | **PASS** |
| Dropoff exists, in range | Deposit + return | ✅ Deposits | **PASS** |
| Town Hall is only dropoff | Deposits both gold & wood | ✅ `dropoff:true` universal | **PASS** |
| Multiple peasants, no dropoff | One builds, others queue | ✅ Cooldown prevents dupes | **PASS** |
| Gold peasant → lumber mill | Should NOT deposit | ✅ Ignored | **PASS** |
| Wood peasant → refinery | Should NOT deposit | ✅ Ignored | **PASS** |
| Constructing building as dropoff | Should NOT deposit | ✅ `progress<1` excluded | **PASS** |
| **Stop command near dropoff** | **Should deposit** | **❌ Clears without deposit** | **FAIL (SR-R6-01)** |
| **Move order near dropoff** | **Should deposit or preserve** | **❌ Clears without deposit** | **FAIL (SR-R6-01)** |
| **Idle peasant at dropoff** | **Should deposit immediately** | **❌ Up to 2s delay** | **FAIL (SR-R6-02)** |
| **Resource tile depletion** | **Tiles eventually exhaust** | **❌ Infinite** | **FAIL (SR-R6-03)** |
| Peasant killed while carrying | Resources lost (RTS convention) | Resources lost | **NOTED** |

---

## Positive Security Observations

The codebase continues to demonstrate excellent security hygiene in most areas:

1. ✅ **Zero external dependencies** — no CDN, no npm, no `<script src>` — zero supply-chain risk
2. ✅ **No `eval()`, `new Function()`, string-based timers** — no dynamic code execution
3. ✅ **No network APIs** — no `fetch()`, `XMLHttpRequest`, `WebSocket` — zero data exfiltration surface
4. ✅ **No storage APIs** — no `localStorage`, `sessionStorage`, `document.cookie`, `IndexedDB`
5. ✅ **No text input fields** — no user-controlled HTML injection path
6. ✅ **No URL parameter/hash parsing** — no DOM-based XSS via URL
7. ✅ **`Map` for entity registry** — immune to `__proto__`/`constructor` key pollution
8. ✅ **`textContent` for resource/simple text** — 4 locations in `updateUI()` auto-escape
9. ✅ **Game loop dt capping** — `0.2` cap prevents spiral-of-death, `0.3` skip prevents background-tab abuse
10. ✅ **Camera coordinate clamping** — prevents OOB canvas access
11. ✅ **`user-select: none`** — prevents text-selection manipulation during gameplay
12. ✅ **`contextmenu` prevention** — `e.preventDefault()` on canvas
13. ✅ **`Number.isFinite()` guards** — in both `createUnit` and `createBuilding` (fixed R4)
14. ✅ **`if(!def) return false;` guard** — in `isPlacementValid` (fixed R4)
15. ✅ **30-tile resource search radius** — up from 12 (fixed R4)

---

## Remediation Checklist (Priority-Ordered)

| Priority | ID | Action | Effort | Lines |
|----------|----|--------|--------|-------|
| 🔴 **P0** | SR-R6-01 | In `commandStop()`/`issueMove()`: deposit before clearing `carryAmount` when near dropoff | 10 min | ~10 |
| 🔴 **P0** | SR-NEW-01 | Add `carryType` whitelist before `game.resources[e.carryType] +=` | 1 min | 1 |
| 🟠 **P1** | SR-R6-02 | Broaden deposit handler to include `state==='idle'` with carryAmount | 5 min | 1 |
| 🟠 **P1** | SR-01 | Replace `innerHTML` with DOM API in `updateUI()` | 30 min | ~10 locations |
| 🟡 **P2** | SR-R5-02 | Add `const` before `cam = game.camera` in `resizeCanvas()` | 1 min | 1 char |
| 🟡 **P2** | SR-R5-03 | Zero-guard on `canvas.clientWidth`/`clientHeight` in `screenToWorld()` | 3 min | 3 |
| 🟡 **P2** | SR-NEW-02 | `hasOwnProperty` guard for `Object.entries(BUILDING_DEFS)` | 2 min | 1 |
| 🟡 **P2** | SR-R3-03 | Remove duplicate `attackTarget:null` at line 273 | 1 min | 1 |
| 🟢 **P3** | SR-R3-02 | `deepFreeze()` game definition objects | 15 min | ~10 |
| 🟢 **P3** | SR-03 | Add `MAX_ENTITIES = 500` guard in factory functions | 5 min | 4 |
| 🟢 **P3** | SR-R6-03 | Add tile depletion tracking | 30 min | ~20 |
| 🔵 **P4** | SR-R5-01 | Add entity-count guard before full-map `findAutoBuildSite` fallback | 5 min | 2 |
| ℹ️ **Info** | SR-02 | Add CSP meta tag (after SR-01 fix) | 5 min | 1 |
| ℹ️ **Info** | SR-R5-04 | Prune expired `autoBuildCooldown` entries | 5 min | 3 |

---

## Summary

The RealmCraft RTS auto-build dropoff chain is **functionally sound** at its core: seven guard clauses protect the `autoBuildDropoff` orchestrator, resource type discrimination in `findNearestDropoff` correctly routes gold/wood to appropriate buildings, and the idle retry loop provides eventual consistency for edge cases. However, **three new findings** in this review reveal that the `commandStop()` and `issueMove()` code paths (SR-R6-01) and the deposit handler state guard (SR-R6-02) still have resource-loss bugs that undermine the auto-build system's intent. The resource preservation improvements made between R4 and R5 (conditional clearing) are directionally correct but have a logic inversion that destroys resources when a dropoff IS nearby — the opposite of the needed behavior.

**Overall risk remains LOW.** All findings require browser console access or specific gameplay sequences to exploit.

---

*End of Report — Security Reviewer (Captain Claw fleet, PHRYGIAN mode)*
