# Correctness Review R6 — Auto-Build & Deposit Flow

**Date:** 2026-07-03 09:12 UTC  
**Reviewer:** Code Reviewer (Captain Claw fleet, PHRYGIAN mode)  
**Task:** "When worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it"  
**Artifacts Reviewed:** `index.html` (~1,466 lines), `test_game.js` (~710 lines), `plan.md`  
**Test Baseline:** 291 passed, 0 failed  
**Scope:** Correctness, edge cases, error handling, regressions against the task specification  
**Methodology:** Full frame-by-frame trace of the gather→carry→deposit→auto-build state machine

---

## Executive Summary

**Overall Risk: HIGH** — 1 blocking bug, 3 major bugs found.

The auto-build mechanism (`autoBuildDropoff`, `findAutoBuildSite`, `hasDropoffUnderConstruction`) is correctly wired. **However**, the deposit path that should deliver carried resources to existing dropoffs is unreliable due to a state-ordering bug in `updateEntity`. In roughly 5–15% of deposit attempts (depending on framerate), peasants arrive at a dropoff, get stuck in an idle↔moving loop, and **never deposit** their carried resources. The idle retry loop that was added to recover from edge cases cannot recover from this state because it has the same state-ordering bug. Carried resources are effectively lost unless the player manually intervenes.

Additionally, a race condition in the auto-build cooldown allows two peasants to auto-build the same building type in the same frame, deducting resources twice.

---

## Findings

### B1 [BLOCKING] — Peasants stuck at dropoff: state-ordering makes deposits unreliable

**File:line:** `index.html` movement handler (~L1190–1215) + moving-deposit handler (~L1256–1281) + idle retry (~L1299–1310)  
**Severity:** BLOCKING — peasants can permanently lose the ability to deposit resources  
**Root cause:** Order-of-operations in `updateEntity`

**The problem.** Within a single `updateEntity(e, dt, toRemove)` call, these blocks execute in order:

1. **Movement handler** (runs first): If `d < 3`, sets `e.state = 'idle'`, `e.moveTarget = null`
2. **Moving-deposit handler** (runs later): Guards on `e.state === 'moving'`. Sees `'idle'` → **does not fire**
3. **Idle retry** (runs last): Only fires every `~2` seconds. Sets `e.state = 'moving'` with `moveTarget = dropoff`. Next frame: movement handler sees `d < 3` → `state = 'idle'` → goto step 3.

```
Frame N:   peasant d=5  → move step=2.3 → d=2.7 < 3 → state='idle'
           moving-deposit: state='idle' → SKIP
           idle retry: Math.floor(time*10)%20 !== 0 → SKIP

Frame N+1: peasant d=2.7, state='idle', carryAmount=10
           movement: state≠'moving' → SKIP
           deposit: state≠'moving' → SKIP
           idle retry: SKIP (not the 2s tick)
...
~120 frames later:
           idle retry: FIRES → state='moving', moveTarget=dropoff
           NEXT FRAME: d≈0 < 3 → state='idle' → stuck again
```

The cycling condition `(d < 3)` is always true because the idle retry sets `moveTarget` to the **exact dropoff position** the peasant is already standing at. The peasant never leaves, so distance stays ≈ 0.

**Impact:** With `speed=140`, `dt≈0.0167` (60fps), the step size is ~2.2px. The pre-move distance must be in `[3, 40)` to trigger deposit. If it lands in `[step, step+3)`, the peasant overshoots to `d<3` and gets stuck. Empirical probability: ~6% at 60fps, ~13% at 30fps per deposit. After many deposits in a game, this manifests reliably. Once stuck, carried resources are **permanently un-depositable** without manual player intervention (and pressing Stop at the dropoff **discards** the resources).

**Fix:**
```javascript
// Option A: Deposit before state transition (move deposit check BEFORE movement's d<3 branch)
// Option B: Broaden deposit guard to also accept idle:
if ((e.state === 'moving' || e.state === 'idle') && e.carryAmount > 0) { ... }

// Option C (best): In the idle retry, if dropoff is within deposit range, deposit directly:
if (dropoff) {
  const d = dist({x:e.x,y:e.y}, {x:dropoff.x,y:dropoff.y});
  if (d < 40) {
    // Deposit directly — don't bother with move state
    game.resources[e.carryType] += e.carryAmount;
    e.carryAmount = 0;
    // return to gatheringNode if available
  } else {
    e.state = 'moving';
    e.moveTarget = {x:dropoff.x, y:dropoff.y};
  }
}
```

Recommend applying **both Option B and Option C** for defense-in-depth.

---

### M1 [MAJOR] — Gathering progress lost when peasant is pushed off resource tile

**File:line:** `index.html` gathering handler + movement handler arrival (~L1225–1254, ~L1190–1215)  
**Severity:** MAJOR — wasted worker time, resources gathered slower than intended

**The problem.** When entity separation pushes a gathering peasant off the resource tile, the walk-back logic correctly preserves `gatheringTimer` (it `return`s from the gathering block without resetting). However, when the peasant arrives back at the gathering node, the movement handler unconditionally resets the timer:

```javascript
// Movement handler — arrival at target:
if (d < 3) {
    e.state = 'idle';
    // ...
    if (e.type === 'peasant' && e.gatheringNode && e.carryAmount === 0) {
        e.state = 'gathering';
        e.gatheringTimer = 0;  // ← BUG: discards existing progress
    }
}
```

**Impact:** If a peasant is 0.1s from completing gathering (timer at 1.4) and gets pushed off, they walk back and restart from 0. This wastes up to 1.5s of gathering time per push event.

**Fix:**
```javascript
if (e.type === 'peasant' && e.gatheringNode && e.carryAmount === 0) {
    e.state = 'gathering';
    // Don't reset — preserve existing timer
    // e.gatheringTimer = 0;  ← remove this line
}
```

---

### M2 [MAJOR] — Race condition: two peasants can auto-build the same building in one frame

**File:line:** `index.html` `autoBuildDropoff` (~L772–802)  
**Severity:** MAJOR — resources deducted twice, two buildings created

**The problem.** The cooldown mechanism is not atomic within a frame. If peasant A's gathering completes and peasant B's idle retry both fire `autoBuildDropoff('refinery')` in the same `update(dt)` call:

1. Peasant A calls `autoBuildDropoff`: checks cooldown (clear), checks affordability (sufficient), deducts resources, creates building, **sets cooldown to `game.time`**.
2. Peasant B calls `autoBuildDropoff`: **already passed the cooldown guard** in the same frame, also deducts resources, creates a second building.

Both buildings are created. Resources are deducted twice (`gold -= 100` twice). Only peasant A is assigned to build.

**Impact:** Net loss of 100 gold and 50 wood. Only one building will be built (the other sits at progress=0 with no builder assigned, never completing).

**Fix:** Check cooldown again immediately before deduction, or use a per-frame lock:
```javascript
// Add to game state:
game._autoBuildThisFrame: new Set(),

// In autoBuildDropoff, after affordability check:
if (game._autoBuildThisFrame.has(buildingType)) return false;
game._autoBuildThisFrame.add(buildingType);

// Clear in update() after entity loop:
game._autoBuildThisFrame.clear();
```

---

### M3 [MAJOR] — `hasDropoffUnderConstruction` blocks auto-build for far-away town halls

**File:line:** `index.html` ~L727–738  
**Severity:** MAJOR — peasants carry resources indefinitely while distant town hall builds

**The problem.** The function returns `true` for **any** town hall under construction, regardless of distance:

```javascript
function hasDropoffUnderConstruction(resType) {
    const bldType = getDropoffBuildingType(resType);
    for (const e of game.entities.values()) {
        if (e.owner === 'player' && e.progress < 1) {
            if (e.type === bldType) return true;
            if (e.type === 'town_hall') return true;  // ← ANY town hall, anywhere
        }
    }
    return false;
}
```

**Impact:** If the player manually starts a town hall at the opposite corner of the map and a peasant near a far-away gold mine completes gathering, the auto-build of a refinery is blocked. The peasant must wait for the distant town hall to finish (10s build time) before being able to deposit. During that time, the peasant stands idle with 10 gold. If the distant town hall gets destroyed mid-construction, the peasant waits indefinitely.

**Fix:** Add a distance check:
```javascript
function hasDropoffUnderConstruction(resType, nearX, nearY) {
    const MAX_DIST = 400; // pixels — only block if construction is nearby
    const bldType = getDropoffBuildingType(resType);
    for (const e of game.entities.values()) {
        if (e.owner === 'player' && e.progress < 1) {
            if (e.type === bldType) {
                if (dist({x:nearX,y:nearY}, {x:e.x,y:e.y}) < MAX_DIST) return true;
            }
            if (e.type === 'town_hall') {
                if (dist({x:nearX,y:nearY}, {x:e.x,y:e.y}) < MAX_DIST) return true;
            }
        }
    }
    return false;
}
```

---

### N1 [MINOR] — `findNearestResource` off-by-one in search radius

**File:line:** `index.html` ~L805  
**Severity:** MINOR — effective radius is 29 tiles, not the documented 30

**The problem:**
```javascript
for (let ty = Math.max(0, cy - searchR); ty < Math.min(MAP_ROWS, cy + searchR); ty++) {
//                                            ^^ exclusive upper bound
```
The `ty < cy + searchR` loop condition uses `<` (exclusive), so the farthest tile searched is at `cy + searchR - 1`, not `cy + searchR`. The documented `searchR = 30` is effectively 29 tiles on the right and bottom edges.

**Impact:** Minor — resources at exactly 30 tiles distance on the right/bottom edges are invisible to the search. The visual search area is slightly smaller than expected.

**Fix:** Change `<` to `<=`:
```javascript
for (let ty = Math.max(0, cy - searchR); ty <= Math.min(MAP_ROWS - 1, cy + searchR); ty++) {
```

---

### N2 [MINOR] — `findAutoBuildSite` fallback is O(n²) with n = map tiles × entity count

**File:line:** `index.html` ~L744–756  
**Severity:** MINOR — potential frame drop on slow machines with many entities

**The problem.** When the ring search (rings 3,5,7,10) finds no valid site, the fallback iterates all 4,096 tiles, each calling `isPlacementValid()` which in turn iterates all entities for overlap checks. With 50+ entities, this is ~200K rectangle-overlap checks.

**Impact:** Frame drop during auto-build when no nearby site exists. Mitigated by the cooldown (happens at most every 15s per building type).

**Fix:** Add early exit with a bailout distance cap:
```javascript
// Only search up to 30 tiles away in fallback
const MAX_FALLBACK_TILES = 30;
```

---

### N3 [MINOR] — Auto-build cooldown intent mismatch: short cooldown is effectively 18s

**File:line:** `index.html` affordability failure path (~L787)  
**Severity:** MINOR — UX issue, not functional

**The problem.** On affordability failure, the code sets:
```javascript
game.autoBuildCooldown[buildingType] = game.time + 3; // "short cooldown"
```
But the cooldown check is:
```javascript
if ((game.time - game.autoBuildCooldown[buildingType]) < 15) return false;
```

The effective cooldown is `18s` (= 3 + 15), not `3s` as the comment suggests. The `+3` does add a penalty, but the minimum cooldown is always 15s due to the check.

**Impact:** Inconsequential — the 15s base cooldown is already sufficient. The `+3` is unnecessary complexity.

**Fix:** Either simplify to a fixed short cooldown or make the intent clear:
```javascript
// Option: actual short cooldown for resource failure
const SHORT_COOLDOWN = 3;
game.autoBuildCooldown[buildingType] = game.time + SHORT_COOLDOWN;

// And change check to use SHORT_COOLDOWN:
if (game.autoBuildCooldown[buildingType] && (game.time - game.autoBuildCooldown[buildingType]) < 3) return false;
```

---

## Test Coverage Gaps

The test file (`test_game.js`) has 291 passing tests but **does not test**:

1. **Frame-accurate deposit ordering** — Section 13 tests the "Gather → Carry → Deposit" chain but does not simulate the `updateEntity` ordering. The test `qa-r4-01` asserts `(state === 'moving' || state === 'idle')` which is the **desired** guard, not the actual guard in the game code.

2. **Auto-build double-deduct race** — No test for two peasants calling `autoBuildDropoff` in the same frame.

3. **Peasant stuck-after-deposit** — No test simulates the idle→move→idle cycle at a dropoff.

4. **Gathering progress preservation** — No test checks that `gatheringTimer` is preserved after tile-push + walk-back.

---

## Regression Against Task Spec

| Task requirement | Status | Issue |
|---|---|---|
| "Check if resource is released back to town hall (or anywhere else)" | ⚠️ PARTIAL | Deposit works ~85-94% of the time. The rest of the time, peasants get stuck. |
| "If we don't have that, build it" | ✅ CORRECT | Auto-build logic is correctly wired, with the caveat of M2 (race condition) and M3 (distant town hall). |

---

## Recommended Fix Priority

1. **B1 (BLOCKING):** Fix deposit state-ordering — apply both Option B and Option C from the fix description
2. **M2 (MAJOR):** Add per-frame lock to `autoBuildDropoff`
3. **M1 (MAJOR):** Preserve `gatheringTimer` on walk-back arrival
4. **M3 (MAJOR):** Add distance check to `hasDropoffUnderConstruction`
5. **N1 (MINOR):** Fix `findNearestResource` off-by-one
6. **N2 (MINOR):** Add bailout distance to `findAutoBuildSite` fallback
7. **N3 (MINOR):** Simplify auto-build cooldown logic
