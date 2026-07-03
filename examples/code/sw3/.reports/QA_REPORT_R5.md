# QA Report: RealmCraft RTS — Round 5 Assessment

**Date:** 2026-07-03 08:58 UTC
**Assessor:** QA Engineer (CAPTAIN CLAW fleet, session `17c86425`)
**Focus:** Worker resource gathering → carrying → release to town hall/dropoff → auto-build if missing
**Artifact:** `index.html` (~1,500 lines) — single-file RTS game
**Test Suite:** `test_game.js` (226 tests, all passing)

---

## Executive Summary

**Overall Quality: GOOD — CRITICAL R4-001 FIXED**

The deposit arrival handler now fires for idle peasants (condition `state==='moving'||state==='idle'`), which resolves the infinite loop where a peasant who auto-built a dropoff could never deposit at it. The full gather→carry→deposit→return chain is now correct in the code.

However, the test suite still has significant blind spots. The auto-build subsystem (`findNearestDropoff`, `findNearestResource`, `findAutoBuildSite`, `getDropoffBuildingType`) remains wholly untested with dedicated unit tests. Deposit type discrimination (gold vs wood dropoffs) is critical for correctness and untested.

| Severity | Count | Description |
|----------|-------|-------------|
| 🟠 HIGH | 4 | Deposit type discrimination untested; resource search radius untested; placeBuilding untested; issueGather tile dispatch untested |
| 🟡 MEDIUM | 3 | findNearestResource null-return path; lumber mill <-> refinery cross-deposit guard; entityAtWorld complex hit test untested |
| 🔵 LOW | 2 | findAutoBuildSite no mock; world-edge entity separation artifacts |

---

## Test Suite Results

```
$ node test_game.js
Running RealmCraft RTS Test Suite...

RESULTS: 226 passed, 0 failed
```

**All 226 tests pass.** The suite has grown from 187 (R4) to 226 (current), adding the "Gather → Carry → Deposit → Auto-Build Chain" section. But critical functions remain untested.

---

## Task-Specific Assessment

### "When worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else)"

**VERDICT: PASS — Resources are correctly released back to valid dropoff buildings.**

#### Full trace of the deposit path:

1. **Gathering completion** (`updateEntity`, ~line 1235): After 1.5s gathering, peasant gets `carryAmount=10`. Then:
   ```javascript
   const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
   if(dropoff) { e.state='moving'; e.moveTarget={x:dropoff.x,y:dropoff.y}; }
   else if(!autoBuildDropoff(e)) { e.state='idle'; /* preserves carryAmount */ }
   ```

2. **Deposit arrival** (~line 1269): Fires for peasants with `(state==='moving'||state==='idle') && carryAmount>0`:
   ```javascript
   const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
   if(dropoff && dist(peasant, dropoff) < 40) {
     game.resources[e.carryType] += e.carryAmount;  // ← RESOURCE RELEASED HERE
     e.carryAmount = 0;
     if(e.gatheringNode) { /* return to gathering */ }
   }
   ```

3. **Accepted dropoff types for gold:** Town Hall (`dropoff:true`), Refinery (`dropoff_gold:true`)
4. **Accepted dropoff types for wood:** Town Hall (`dropoff:true`), Lumber Mill (`dropoff_wood:true`)

**Resources are correctly returned to the game resource pool** via `game.resources[carryType] += carryAmount`. The peasant then returns to their gathering node to repeat the cycle.

### "And if we don't have that, build it"

**VERDICT: PASS — Auto-build correctly triggers when no appropriate dropoff exists.**

#### Full trace of the auto-build path:

1. `autoBuildDropoff(peasant)` checks (in order):
   - `carryType` is non-null and `carryAmount > 0`
   - No existing dropoff found via `findNearestDropoff`
   - No dropoff under construction via `hasDropoffUnderConstruction`
   - Cooldown not active (prevents duplicates within 15s)
   - Resources can afford the building
   - Valid build site found via `findAutoBuildSite`

2. On success: deducts resources → creates building (progress=0) → assigns peasant to build it

3. After building completes: peasant is freed (state='idle', buildTarget=null) → deposit arrival handler fires → resources are deposited at the newly-completed dropoff

**Edge case verified — town hall under construction counts as dropoff** (`hasDropoffUnderConstruction` returns true for `town_hall`), preventing redundant auto-builds.

---

## New Findings R5

### Finding R5-001 🟠 HIGH: `findNearestDropoff` type discrimination untested

**Severity:** HIGH
**Location:** `index.html` — `findNearestDropoff` function

**Details:** The function has two distinct branches that discriminate by resource type:
```javascript
if(resType==='gold' && (def.dropoff||def.dropoff_gold)) { /* match */ }
if(resType==='wood' && (def.dropoff||def.dropoff_wood)) { /* match */ }
```

A gold-carrying peasant must NOT deposit at a lumber mill, and a wood-carrying peasant must NOT deposit at a refinery. This is the core correctness of the resource pipeline, yet zero tests verify type discrimination.

**What's untested:**
- Gold peasant → lumber mill → should NOT find it as dropoff
- Wood peasant → refinery → should NOT find it as dropoff
- Gold peasant → refinery → should find it
- Wood peasant → lumber mill → should find it
- Gold peasant → town hall (universal) → should find it
- Wood peasant → town hall (universal) → should find it

**Risk:** If someone adds `dropoff:true` to lumber_mill or refinery by accident, or changes the type check logic, gold-carrying peasants would deposit at lumber mills silently, corrupting the game's economic model.

---

### Finding R5-002 🟠 HIGH: `placeBuilding` function wholly untested

**Severity:** HIGH
**Location:** `index.html` — `placeBuilding` function

**Details:** This function handles the primary building placement flow:
- Resource deduction (gold + wood)
- Placement validation
- Building creation
- Nearest idle peasant assignment
- **Soft-lock prevention**: If no peasants exist and building is `town_hall`, auto-completes it

The soft-lock prevention path is critical but has zero test coverage:
```javascript
if(peasantCount===0 && game.ui.buildingType==='town_hall') {
    b.progress=1; b.hp=b.maxHp; b.state='idle';
    if(b.provides) {
      for(const p of b.provides) {
        if(p==='food5') game.resources.maxFood+=5;
      }
    }
}
```

**What's untested:**
- Place building with sufficient resources → building created, resources deducted
- Place building with insufficient resources → rejected
- Place building on invalid terrain → rejected
- Auto-assign nearest idle peasant
- Soft-lock: place town_hall with no peasants → auto-completes
- Place building with peasants but none idle → no auto-assignment (building sits at 0%)

---

### Finding R5-003 🟠 HIGH: `issueGather` tile dispatch untested

**Severity:** HIGH
**Location:** `index.html` — `issueGather` function

**Details:** The function dispatches based on tile type:
- `TERRAIN.GOLD` → set carryType='gold', find nearest gold node
- `TERRAIN.TREE` → set carryType='wood', find nearest tree node
- Anything else → just move there

Zero tests for any dispatch path. Additionally, the `findNearestResource` null-return path (`showFeedback('No resource nearby')`) is never tested.

---

### Finding R5-004 🟠 HIGH: `findNearestResource` search radius limitation

**Severity:** HIGH
**Location:** `index.html` — `findNearestResource` function

**Details:** The search is capped at 30 tiles (960px):
```javascript
const searchR=30;
```

If the nearest resource of the requested type is more than 30 tiles away, the function returns `null`. In `issueGather`, this triggers `showFeedback('No resource nearby')` and the peasant moves to the clicked location but won't gather.

**Impact:** On a 64×64 tile map (2048×2048px), a 30-tile search radius covers roughly 22% of the map area from the center. Resources in corners or across the map are invisible to `issueGather`. The peasant walks to the click point, finds nothing, and goes idle.

**Comparison:** The `findAutoBuildSite` function has a full-map fallback after the ring search. `findNearestResource` has no such fallback.

---

### Finding R5-005 🟡 MEDIUM: Lumber mill / refinery cross-deposit guard

**Severity:** MEDIUM
**Location:** `index.html` — `findNearestDropoff` function

**Details:** The type guard works correctly (`dropoff_gold` vs `dropoff_wood`), but there's no test that a lumber mill accidentally receiving `dropoff:true` (making it universal) would be caught. The current `BUILDING_DEFS` are correct:
```javascript
lumber_mill: { dropoff_wood:true }  // only wood
refinery:     { dropoff_gold:true }  // only gold
```

But if someone adds `dropoff:true` to either in the future, the economic model breaks silently.

---

### Finding R5-006 🟡 MEDIUM: `findNearestResource` null-return causes silent failure

**Severity:** MEDIUM
**Location:** `index.html` — `issueGather` function

**Details:** When `findNearestResource` returns null:
```javascript
if(!node) {
  entity.state='moving';
  entity.moveTarget={x:wx,y:wy};
  showFeedback('No '+resType+' resource nearby');
  return;
}
```

The peasant walks to the clicked location but never gets `gatheringNode` or `carryType` set. They arrive and go idle. The `showFeedback` message is transient (floating text) and disappears in 1.2 seconds. The player has no persistent indicator that the gather command failed.

**Better:** The peasant should either (a) not move at all (reject the command), or (b) get the `gatheringNode` set to a distant resource node. Currently it's a half-measure that's confusing.

---

### Finding R5-007 🟡 MEDIUM: `entityAtWorld` hit-testing approximate for buildings vs units

**Severity:** MEDIUM
**Location:** `index.html` — `entityAtWorld` function

**Details:** The hit test uses entity `w/2` and `h/2` as half-extents:
```javascript
if(wx>=e.x-e.w/2&&wx<=e.x+e.w/2&&wy>=e.y-e.h/2&&wy<=e.y+e.h/2) return e;
```

For units, `w` is `def.size*2` (e.g., peasant size=16, w=32). For buildings, `w` is the `def.size.w` (e.g., town hall w=96). The hit box is a rectangle, which is correct for buildings but approximate for circular unit selection.

**Impact:** Clicking near the corner of a unit's bounding box (not visually overlapping) will still select the unit. This is standard RTS behavior but untested.

---

### Finding R5-008 🔵 LOW: `findAutoBuildSite` full-map fallback performance

**Severity:** LOW
**Location:** `index.html` — `findAutoBuildSite` function

**Details:** The fallback searches all 64×64=4096 tiles linearly:
```javascript
for(let ty=0;ty<MAP_ROWS;ty++) {
    for(let tx=0;tx<MAP_COLS;tx++) {
        if(!isPlacementValid(buildingType, wx, wy)) continue;
        ...
    }
}
```

Each `isPlacementValid` call checks 5 terrain points + iterates ALL entities for overlap. On a map with many entities, this is O(n*m) where n=entities, m=4096. For 50 entities: ~200K rect overlap checks. At 60fps this is fine (sub-millisecond), but the function has no test coverage and no performance guard.

---

### Finding R5-009 🔵 LOW: Entity separation on world edges creates sliding artifacts

**Severity:** LOW
**Location:** `index.html` — `updateEntity` separation + clamp

**Details:** The separation pushes entities apart, then clamps to world bounds:
```javascript
e.x=clamp(e.x,e.size,WORLD_W-e.size);
e.y=clamp(e.y,e.size,WORLD_H-e.size);
```

This means entities near world edges get pushed by separation, then "slide" along the edge boundary. This is purely cosmetic but creates visible artifacts where units appear to vibrate at world borders.

---

## Fixed from R4

### R4-001 (CRITICAL) — Auto-build → deposit infinite loop: **FIXED** ✅

The deposit arrival handler now fires for `state==='idle'` in addition to `state==='moving'`:
```javascript
// Before fix (R4):
if(e.type==='peasant'&&e.state==='moving'&&e.carryAmount>0) { ... }

// After fix (current code):
if(e.type==='peasant'&&(e.state==='moving'||e.state==='idle')&&e.carryAmount>0) { ... }
```

**Verification trace:** After auto-building a dropoff, the building completes, frees the peasant (`state='idle'`). The deposit handler fires (because `idle` is now accepted), finds the completed dropoff at the peasant's location (`d < 40`), deposits resources, and returns to gathering. ✓

### R4-002 (HIGH) — Auto-build subsystem untested: **PARTIALLY FIXED** ⚠️

Tests added for `hasDropoffUnderConstruction`, `autoBuildDropoff` cooldown, and the full gather→auto-build chain. But `findNearestDropoff`, `findNearestResource`, `findAutoBuildSite`, and `getDropoffBuildingType` still have zero dedicated tests.

### R4-003 (HIGH) — issueMove/issueGather untested: **PARTIALLY FIXED** ⚠️

`issueMove` carryAmount preservation is now tested. But `issueGather` tile dispatch (gold vs wood vs non-resource) and `findNearestResource` null-return remain untested.

### R4-004 (MEDIUM) — Idle retry timing gap: **UNCHANGED** ⚠️

The 2-second retry loop is still the only mechanism for idle peasants to find dropoffs. No change.

### R4-005 (MEDIUM) — gatheringNode not cleared during auto-build: **WON'T FIX (by design)** ✅

`gatheringNode` is intentionally preserved so peasants return to their original resource after depositing. This is correct behavior.

### R4-006 (MEDIUM) — issueMove discards resources: **FIXED** ✅

`issueMove` and `commandStop` now preserve `carryAmount`/`carryType` when no dropoff exists, and clear them when a dropoff is available.

---

## Test Coverage Gap Map (Updated R5)

```
Function                    Tested?   Lines   Notes
───────────────────────────────────────────────────────────
createUnit                  ✅        20      40 tests
createBuilding              ✅        15      8 tests
isPlacementValid            ✅        20      10 tests
startTraining               ✅        25      8 tests
updateEntity (partial)      ✅        175     13 tests
screenToWorld               ✅        5       4 tests
autoBuildDropoff (chain)    ⚠️        35      3 tests (chain only; 5 exit paths untested)
hasDropoffUnderConstruction ⚠️        8       1 test
── GAP BELOW ──
findNearestDropoff          ❌        15      0 tests
findNearestResource         ❌        14      0 tests
findAutoBuildSite           ❌        25      0 tests
getDropoffBuildingType      ❌        5       0 tests
issueGather                 ❌        18      0 tests
placeBuilding               ❌        30      0 tests
entityAtWorld               ❌        10      0 tests
generateMap                 ❌        50      0 tests
render / draw functions     ❌        250     0 tests (not testable in Node)
```

---

## Recommended Actions (Priority Order)

1. **Add `findNearestDropoff` tests** — Verify gold→refinery/townhall, gold→NOT lumbermill, wood→lumbermill/townhall, wood→NOT refinery
2. **Add `placeBuilding` tests** — Resource deduction, nearest peasant assignment, soft-lock prevention for town_hall with zero peasants
3. **Add `issueGather` tests** — Gold tile dispatch, wood tile dispatch, non-resource tile fallback, findNearestResource null-return path
4. **Add `findNearestResource` tests** — Within 30 tile radius, beyond 30 tile radius (null return), edge of search area
5. **Add `entityAtWorld` tests** — Hit detection on unit, hit detection on building, miss at corner
6. **Add `getDropoffBuildingType` tests** — gold→refinery, wood→lumber_mill, invalid→null
7. **Consider** expanding findNearestResource search radius or adding full-map fallback

---

## Overall Verdict

The core gameplay loop — worker gathers resource, carries it, releases it back to dropoff, builds dropoff if missing — is **correctly implemented**. The R4-001 infinite loop is fixed. The remaining gaps are in test coverage for the auto-build subsystem's supporting functions, which would catch regressions if the economic model changes.

**226/226 tests pass. Game logic is sound. Tests need expansion in the areas noted above.**
