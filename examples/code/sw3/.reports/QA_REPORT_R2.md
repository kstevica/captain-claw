# QA Report: RealmCraft RTS — Round 2 Assessment

**Date:** 2026-07-03 07:26 UTC  
**Assessor:** QA Engineer (CAPTAIN CLAW fleet, session `c78fd4a3`)  
**Artifact:** `index.html` (~1,470 lines) — single-file RTS game  
**Test Suite:** `test_game.js` (711 lines, 123 tests)  
**Methodology:** Test suite execution + static code audit + prior-bug re-verification + test/game drift analysis

---

## Executive Summary

**Overall Quality: GOOD → IMPROVING.** Four of the seven bugs from the prior QA report have been fixed. The test suite remains solid at 123/123 passing. However, I identified one new **test accuracy bug** (a drift between the test file and the game code) and two remaining correctness issues.

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 0 | — |
| 🟠 MEDIUM | 3 | Test/game code drift in `isPlacementValid`; untested `updateEntity` and commands (ongoing); 3 remaining prior bugs |
| 🟡 LOW | 4 | Missing `produces` validation; z-ordering uses center-y; dead code; UX polish gaps |

---

## 1. Test Suite Execution

```
$ node test_game.js
Running RealmCraft RTS Test Suite...

RESULTS: 123 passed, 0 failed
═══════════════════════════════════════
```

**All 123 tests pass.** The suite remains green across all 9 modules.

---

## 2. Prior Bug Re-Verification (QA-01 through QA-07)

I re-examined each previously-reported bug against the current `index.html` source:

| Bug ID | Description | Prior Severity | Current Status |
|--------|-------------|----------------|----------------|
| QA-01 | Buildings drift via entity separation | 🔴 HIGH | ✅ **FIXED** — separation loop now guards with `!e.isBuilding` |
| QA-02 | Food can go negative on unit death | 🟠 MEDIUM | ✅ **FIXED** — `Math.max(0, food - cost)` added |
| QA-03 | maxFood can fall below current food on TH death | 🟠 MEDIUM | ✅ **FIXED** — `Math.min(food, maxFood)` added |
| QA-04 | findNearestResource limited to 12 tiles | 🟠 MEDIUM | ✅ **FIXED** — `searchR` increased from 12 to 30 |
| QA-05 | isPassable missing unit check | 🟠 MEDIUM | ⚠️ **STILL PRESENT** — function checks terrain+buildings only; OK for placement but misnamed |
| QA-06 | Building completion frees peasants mid-movement | 🟡 LOW | ⚠️ **STILL PRESENT** — `buildTarget=null; state='idle'` runs on ALL peasants regardless of distance |
| QA-07 | Soft-lock when all peasants die | 🟡 LOW | ⚠️ **STILL PRESENT** — no recovery mechanic |

**Fix rate: 4/7 (57%).** The four fixed bugs were the highest-severity correctness issues. Good progress.

---

## 3. NEW Finding: Test/Game Code Drift (MEDIUM)

### 🟠 NEW-1: `isPlacementValid` in test file does not match game code

**Severity:** MEDIUM  
**Location:** `index.html` vs `test_game.js`

The game's `isPlacementValid` rejects placement on **both WATER and TREE** terrain:

```javascript
// index.html (GAME CODE)
for(const p of checkPoints) {
    const t=tileAt(p.x,p.y);
    if(t===TERRAIN.WATER||t===TERRAIN.TREE) return false;
}
```

The test file's version only checks **WATER**, omitting the TREE check entirely:

```javascript
// test_game.js (TEST CODE)
for(const p of checkPoints) {
    const tx=Math.floor(p.x/TILE_SIZE), ty=Math.floor(p.y/TILE_SIZE);
    if(tx<0||ty<0||tx>=MAP_COLS||ty>=MAP_ROWS) return false;
    if(map[ty][tx]===TERRAIN.WATER) return false;
    // ⚠️ MISSING: TERRAIN.TREE check!
}
```

**Impact:** The test for placement validation is less strict than the game. A building placed on a tree tile would pass the test but be rejected by the actual game. This is a false-negative gap — the test does not detect if someone accidentally removes the TREE check from the game code.

**Fix:** Add `map[ty][tx]===TERRAIN.TREE` to the test's `isPlacementValid`.

**Root cause:** The test file was written separately from the game code and copies ~200 lines of game logic. These two copies have drifted. The fix-and-forget pattern of copying code into tests is the real problem — any future change to the game's placement logic must be manually synced to the test file.

---

## 4. Remaining Critical Gap: `updateEntity` Still Untested (ongoing)

**Severity:** MEDIUM (ongoing from prior report)  
**Location:** `index.html`, `updateEntity()` — ~175 lines

The most complex function in the game remains completely untested. The prior QA_REPORT_FRESH identified this as CRITICAL, and it has not been addressed. This function handles:

- Entity death + particle spawning + food recalc
- Building construction progress (1+ builder multiplier)
- Training queue advancement + unit spawning + rally points
- Attack cooldown timer
- Unit movement + arrival detection + state transitions
- Peasant building state + particles
- Peasant gathering → carry → find-dropoff → deposit → return cycle
- Unit separation/push-apart logic

No test exercises any of these behaviors end-to-end.

---

## 5. Additional New Findings

### 🟡 NEW-2: `startTraining` does not validate `building.produces`

**Severity:** LOW  
**Location:** `startTraining()` in `index.html`

The training function deducts resources and queues a unit without checking whether the building can actually produce that unit type:

```javascript
function startTraining(building, unitType) {
    const def=UNIT_DEFS[unitType];
    // ... resource checks ...
    building.queue.push(unitType);  // <-- no produces check!
}
```

The UI filters available unit types per building (Town Hall shows Peasant, Barracks shows Footman/Archer, etc.), but the game logic itself has no enforcement. A console call to `startTraining(barracks, 'knight')` would succeed and queue a knight in the barracks.

**Impact:** Low — the UI prevents this in normal play. But it's a logic bypass vector and a maintenance risk if the UI is ever refactored.

**Fix:** Add `if(!building.produces || !building.produces.includes(unitType)) return;`

### 🟡 NEW-3: Entity z-ordering uses center-Y, not base-Y

**Severity:** LOW  
**Location:** `render()` — entity sorting

Entities are sorted by `e.y` (center position):

```javascript
ents.sort((a,b)=>a.y-b.y);
```

For buildings, the bottom edge is at `e.y + e.h/2`. A tall building (e.g., Town Hall at 96px height) has its sort key at its vertical center, meaning units walking behind the **bottom half** of the building will be drawn on top of it. This produces a mild but noticeable z-ordering artifact where units appear to walk "over" the lower portion of buildings.

**Impact:** Cosmetic only. Visible when units walk behind tall buildings.

**Fix:** Sort by `e.y + (e.h || 0)/2` to use the entity's base/foot position.

### 🟡 NEW-4: `commandStop` does not reset `gatheringTimer`

**Severity:** LOW  
**Location:** `commandStop()` in `index.html`

The stop command resets peasant state, buildTarget, gatheringNode, carryAmount, and carryType — but does NOT reset `gatheringTimer`:

```javascript
if(e.type==='peasant') {
    e.buildTarget=null; e.gatheringNode=null; e.carryAmount=0; e.carryType=null;
    // ⚠️ MISSING: e.gatheringTimer = 0;
}
```

If a peasant was mid-gather (timer at 1.3s out of 1.5s) and the player issues Stop, then later re-issues a gather command to the same node, the peasant retains the old `gatheringTimer` value and could instantly complete a gather cycle.

**Impact:** Very minor — the gatheringTimer is reset when a new gather node is assigned, but retained if the same node is re-targeted after a stop. Exploitable for slightly faster gathering but requires micro-management.

**Fix:** Add `e.gatheringTimer = 0;` in `commandStop`.

---

## 6. Test Coverage Summary (Updated)

| Function | Lines | Tested? | Notes |
|----------|-------|---------|-------|
| `genId` | 1 | ✅ | |
| `dist` | 1 | ✅ | |
| `clamp` | 1 | ✅ | |
| `lerp` | 1 | ❌ | Trivial, 1 line |
| `rectsOverlap` | 1 | ✅ | |
| `tileAt` | 4 | ✅ | |
| `isPassable` | 9 | ✅ | |
| `spawnParticles` | 5 | ❌ | Visual only |
| `spawnFloatingText` | 5 | ❌ | Visual only |
| `generateMap` | ~70 | ✅ | Via game init tests |
| `createUnit` | 20 | ✅ | All 4 unit types |
| `createBuilding` | 20 | ✅ | All 7 building types |
| `screenToWorld` | 6 | ❌ | **All mouse I/O depends on this** |
| `worldToScreen` | 6 | ❌ | Dead code (never called) |
| `entityAtWorld` | 10 | ✅ | |
| `entitiesInRect` | 9 | ✅ | |
| `findNearestDropoff` | 18 | ✅ | |
| `findNearestResource` | 16 | ✅ | |
| `isPlacementValid` | 19 | ⚠️ | **Test has drift (NEW-1)** |
| `startBuildingPlacement` | 7 | ❌ | UI state machine |
| `cancelPlacement` | 8 | ❌ | No guard against non-placement mode |
| `placeBuilding` | 25 | ✅ | |
| `startTraining` | 18 | ✅ | But no `produces` validation (NEW-2) |
| `getTrainTime` | 6 | ✅ | |
| `getBuildTime` | 2 | ❌ | Defined in test file, not in test blocks |
| `issueMove` | 11 | ❌ | **Core player command** |
| `issueGather` | 20 | ❌ | **Core player command** |
| `update` | ~30 | ❌ | Game loop orchestration |
| `updateEntity` | ~175 | ❌ | **Largest gap** |
| `updateUI` | ~100 | ❌ | DOM manipulation |
| `commandStop` | 13 | ❌ | **Core player command** (NEW-4 bug) |
| `render` + helpers | ~400 | ❌ | Canvas rendering |

**Tested: 18/34 logic functions (53%).** Excluding rendering/DOM: **18/24 (75%).** The 6 untested logic functions are the most critical ones.

---

## 7. Recommendations

### Immediate (fix accuracy)
1. **Fix test/game drift (NEW-1):** Add TREE terrain check to test file's `isPlacementValid`
2. **Add `produces` validation (NEW-2):** Guard `startTraining` against unsupported unit types

### Short-term (close coverage gap)
3. **Write tests for `updateEntity`:** At minimum, test death cleanup, construction progression with 1+ builders, and the peasant gather→carry→deposit cycle
4. **Write tests for `issueMove`, `issueGather`, `commandStop`:** These are the player's only interaction channels

### Polish
5. **Fix z-ordering (NEW-3):** Sort entities by base-y instead of center-y
6. **Reset gatheringTimer in commandStop (NEW-4)**
7. **Address QA-06 (mid-route cancellation):** Only free peasants physically at the construction site
8. **Address QA-07 (soft-lock):** Add a "no workers" detection with a respawn button or game-over screen
9. **Remove or wire up `worldToScreen`:** Dead code is technical debt
10. **Rename `isPassable` → `isBuildable`:** The function is only used for placement validation; the name is misleading

---

## 8. Verdict

The game is in **GOOD shape** — 4 of 7 prior bugs fixed, test suite green at 123/123, no crashes or game-breaking issues found. The main remaining risks are:

1. **Test accuracy:** The test/game drift in `isPlacementValid` means the test suite is slightly overconfident — it doesn't fully mirror game behavior. This is a maintenance hazard.
2. **Coverage gap:** `updateEntity` (175 lines) still has zero tests. This is the core game loop — every gameplay mechanic funnels through it.
3. **Three minor remaining bugs** (QA-05/06/07) are cosmetic or require deliberate edge-case triggering.

**Recommendation: Ship with the fixes above, then prioritize `updateEntity` test coverage in the next iteration.**

---

*Report generated by QA Engineer (CAPTAIN CLAW fleet), 2026-07-03 07:26 UTC.*
