# Correctness Review R1: RealmCraft RTS Game

**Date:** 2026-07-03 07:19 UTC  
**Reviewer:** Code Reviewer (dubina-code-reviewer-reason-882a55)  
**Artifact:** `index.html` — single-file RTS game (~1,468 lines)  
**Previous reviews:** `CORRECTNESS_REVIEW.md` (r0), `QA_REPORT.md`, `SECURITY_REVIEW.md`  
**Delta from r0:** Debugger applied `[fix r1] debugger` (commit `78dba7f`) — partial coordinate unification + `isBuilding` flag added.

---

## Executive Summary

The R1 fix attempted to unify the entity coordinate system to center-based positions but was **incomplete**: 6 locations that referenced building coordinates were NOT updated, creating **new regressions**. Additionally, **12 of the 25 issues** from the r0 review remain unfixed. The game is in a **mixed state** — the coordinate fix improved rendering correctness but broke dropoff navigation, unit spawning, and building death effects.

| Category | Count | New in R1 | Persisting from R0 |
|----------|-------|-----------|---------------------|
| **BLOCKING** | 3 | 1 | 2 |
| **MAJOR** | 13 | 2 | 11 |
| **MINOR** | 8 | 3 | 5 |
| **TOTAL** | **24** | **6** | **18** |

---

## BLOCKING Issues

### B1 (NEW — R1 regression). `findNearestDropoff` computes wrong building center
**File:line:** `index.html:698-706`

```javascript
function findNearestDropoff(x, y, resType) {
  // ...
  const d=dist({x,y},{x:e.x+e.w/2,y:e.y+e.h/2});  // ← WRONG after R1
```

**Root cause:** The R1 fix changed building positions from top-left to center-based (in `placeBuilding` and `initGame`). But `findNearestDropoff` still assumes top-left coordinates — it computes the building center as `e.x + e.w/2`, which for center-based `e.x` produces `center + halfWidth` (the right edge of the building, not the center).

**Impact:** Peasants returning resources compute distances to the wrong point on each building. This can cause them to walk to the wrong building (the nearest per the flawed distance calculation) or fail to recognize they're within deposit range (40px). Resource gathering breaks intermittently depending on building layout.

**Fix:** Change to `dist({x,y},{x:e.x, y:e.y})`. Buildings are now center-based — no need to add `w/2, h/2`.
```javascript
const d=dist({x,y},{x:e.x,y:e.y});
```

---

### B2 (PERSISTING from R0-B1/Q1). Entity separation pushes completed buildings
**File:line:** `index.html:998-1015`

```javascript
if(e.state==='moving'||e.state==='idle') {
    // separation pushes ALL idle entities including buildings
    e.x+=sx*dt*3;  // ← BUILDING DRIFTS
    e.y+=sy*dt*3;
}
```

**Root cause:** The R1 fix added `isBuilding: true` to building entities but **never checks it** in the separation logic. Completed buildings have `state === 'idle'`, so they participate in separation and get pushed around by nearby units. Over 30+ seconds of gameplay, buildings visibly drift from their placement positions.

**Impact:** Strategic building placement is undermined — walls and tightly-packed bases scatter. This corrupts game state silently.

**Fix:** Add the `isBuilding` guard that was created but never wired:
```javascript
if((e.state==='moving'||e.state==='idle') && !e.isBuilding) {
    // separation logic...
}
```

---

### B3 (PERSISTING from R0-B5). `issueMove` unconditionally destroys peasant state
**File:line:** `index.html:830-838`

```javascript
function issueMove(targetX, targetY) {
    for(const id of game.selectedIds) {
        const e=game.entities.get(id);
        if(!e||e.owner!=='player') continue;
        if(e.type=='peasant') {
            e.buildTarget=null; e.gatheringNode=null; e.carryType=null; e.carryAmount=0;
            // ↑ ALL progress silently wiped on ANY move command
        }
        e.state='moving'; e.moveTarget={x:targetX,y:targetY};
    }
}
```

**Impact:** Box-selecting a group that includes a carrying/gathering peasant and issuing a move command destroys all their in-progress work. This violates basic RTS conventions — in Warcraft/SC, workers retain their cargo during regular move commands and only drop it on explicit Stop.

**Fix:** Remove lines 833-835. Only `commandStop` (S key) should clear peasant state:
```javascript
function issueMove(targetX, targetY) {
    for(const id of game.selectedIds) {
        const e=game.entities.get(id);
        if(!e||e.owner!=='player') continue;
        if(e.isBuilding) continue; // buildings can't move
        e.state='moving'; e.moveTarget={x:targetX,y:targetY};
    }
}
```

---

## MAJOR Issues

### M1 (NEW — R1 regression). Training queue spawns units at wrong position
**File:line:** `index.html:985-988`

```javascript
const spawnX = e.rallyPoint ? e.rallyPoint.x : (e.x+e.w/2);
const spawnY = e.rallyPoint ? e.rallyPoint.y : (e.y+e.h);
```

**Root cause:** With center-based coordinates, the default spawn position should be: center-x = `e.x`, bottom edge = `e.y + e.h/2`. But the code computes `e.x + e.w/2` (right edge) and `e.y + e.h` (center + full height = below bottom edge by half the height).

**Impact:** Units spawn at the right-bottom corner of the building, offset by half the building height below it — completely wrong position. For a Town Hall (96×96), units spawn ~48px to the right and ~48px below where they should.

**Fix:**
```javascript
const spawnX = e.rallyPoint ? e.rallyPoint.x : e.x;
const spawnY = e.rallyPoint ? e.rallyPoint.y : (e.y + e.h/2);
```

---

### M2 (NEW — R1 regression). `isPassable` uses top-left coordinate assumptions
**File:line:** `index.html:158-162`

```javascript
function isPassable(px,py) {
    // ...
    for(const e of game.entities.values()) {
        if(e.type in BUILDING_DEFS && e.progress>=1) {
            if(px>=e.x&&px<=e.x+e.w&&py>=e.y&&py<=e.y+e.h) return false;
            // ↑ WRONG: e.x is now center, not top-left
        }
    }
}
```

**Root cause:** After R1's coordinate fix, building `e.x` is the center. The check `px>=e.x` tests against the center, not the left edge. A building centered at x=500 with w=96 has a left edge at 452, but `isPassable` returns false only for x≥500 (the right half).

**Impact:** Currently low — `isPassable` is defined but never called during movement (M9). However, it's a correctness trap: if anyone later integrates pathfinding, half the building footprint would be passable. The function name and signature suggest it *should* work.

**Fix:**
```javascript
if(px>=e.x-e.w/2 && px<=e.x+e.w/2 && py>=e.y-e.h/2 && py<=e.y+e.h/2) return false;
```

---

### M3 (PERSISTING from R0-B2). Mixed peasant+other selection → non-peasants ignored on resource right-click
**File:line:** `index.html:1292-1308`

When peasants AND footmen are selected and the player right-clicks a gold mine: peasants get `issueGather`, footmen get nothing. The player expects footmen to move there.

**Fix:** After the peasant loop, issue `issueMove` to non-peasant units:
```javascript
if(hasPeasant) {
    for(const id of game.selectedIds) {
        const e = game.entities.get(id);
        if(e && e.type === 'peasant') issueGather(e, world.x, world.y);
        else if(e && e.owner === 'player') issueMove(world.x, world.y);
    }
}
```

---

### M4 (PERSISTING from R0-B3). Carrying peasant loses resources when right-clicking non-dropoff building
**File:line:** `index.html:974-995` (deposit logic), `index.html:1310-1325` (right-click handler)

Peasant carries 10 gold → right-clicked onto Barracks → walks there → `state='idle'` → deposit check fires → nearest actual dropoff is >40px away → **resources zeroed**.

**Fix:** When deposit fails because the nearest dropoff is far away, keep the carry state and re-route to the dropoff instead of zeroing.

---

### M5 (PERSISTING from R0-B4). Building destruction doesn't clamp food to maxFood
**File:line:** `index.html:927-932`

```javascript
if(p==='food5') game.resources.maxFood-=5;
// Missing: game.resources.food = Math.min(game.resources.food, game.resources.maxFood);
```

**Impact:** Food can exceed maxFood permanently when a Town Hall is destroyed. UI shows e.g. "8/5".

**Fix:** Add clamp after maxFood reduction:
```javascript
if(p==='food5') {
    game.resources.maxFood -= 5;
    game.resources.food = Math.min(game.resources.food, game.resources.maxFood);
}
```

---

### M6 (PERSISTING from QA-02). Unit death can make food negative
**File:line:** `index.html:934-936`

```javascript
game.resources.food -= (UNIT_DEFS[e.type].food||0);
// No floor: food can become -1, -2, etc.
```

**Fix:**
```javascript
game.resources.food = Math.max(0, game.resources.food - (UNIT_DEFS[e.type].food||0));
```

---

### M7 (PERSISTING from R0-B6). `isPlacementValid` allows building on tree tiles
**File:line:** `index.html:735`

```javascript
if(t===TERRAIN.WATER) return false;
// Missing: || t===TERRAIN.TREE
```

**Impact:** Buildings can be placed on tree tiles, contradicting the plan ("not trees"). Combined with M8, this can cause peasants to try to gather from resource tiles under buildings.

**Fix:** `if(t===TERRAIN.WATER||t===TERRAIN.TREE) return false;`

---

### M8 (PERSISTING from QA-04). `findNearestResource` search limited to 12-tile radius
**File:line:** `index.html:709`

```javascript
const searchR=12; // tiles = 384px
```

On a 64×64 map, this covers ~18% of the world. A peasant >12 tiles from resources silently fails — no visual or UI feedback.

**Fix:** Increase to `searchR=30` or add fallback: if no resource found, move toward map center and show "No resources nearby" floating text.

---

### M9 (PERSISTING from R0-M2). Zero attack logic — all military units are cosmetic
**File:line:** `index.html:987-989` (attack timer), entire file (no attack code)

`attackTimer` is decremented but no code acquires targets, applies damage, or spawns projectiles. The Watch Tower (plan: "Auto-attacks nearby enemies") and all military units are non-functional.

**Fix:** Add an attack update block in `updateEntity`:
```javascript
if(e.attackDamage > 0 && e.attackTimer <= 0) {
    // auto-acquire nearest enemy target or use e.targetId
    // if target in range, apply damage, reset timer
}
```

---

### M10 (PERSISTING from R0-M3). `isPassable` never called during movement
**File:line:** `index.html:153-162` (definition), `index.html:976-984` (movement code)

Units use direct steering (`e.x += mx; e.y += my`) without checking terrain or obstacles. Units walk through walls, buildings, and water.

**Fix:** Integrate `isPassable` into movement with a simple slide-along-obstacle adjustment, or at minimum reject moves into impassable terrain.

---

### M11 (PERSISTING from R0-M6). No HiDPI support — blurry on Retina displays
**File:line:** `index.html:1239-1240`

```javascript
canvas.width = rect.width;
canvas.height = rect.height;
```

**Fix:**
```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = rect.width * dpr;
canvas.height = rect.height * dpr;
ctx.scale(dpr, dpr);
```

---

### M12 (PERSISTING from QA-07). Game soft-locks if all peasants die
**File:line:** N/A (design gap)

With all peasants dead, the player cannot gather, build, or train. Resources sit idle with no way to recover. Must refresh.

**Fix:** Detect `food === 0 && peasantCount === 0` and auto-spawn a peasant at Town Hall, or show a "Game Over" screen.

---

### M13 (PERSISTING from QA-03). `startTraining` doesn't validate building.produces
**File:line:** `index.html:807` (startTraining)

The training function is gated only by UI — the game logic doesn't check if the unit type is in the building's `produces` array. If called programmatically, you could train knights from a barracks.

**Fix:** Add validation:
```javascript
if(!building.produces || !building.produces.includes(unitType)) return;
```

---

## MINOR Issues

### m1 (NEW — R1 regression). Building death particle spawn at wrong position
**File:line:** `index.html:940`

```javascript
spawnParticles(e.x+e.w/2, e.y+e.h/2, 12, '#ff8844');
// With center-based coords: e.x+e.w/2 is right edge, not center
```

**Fix:** `spawnParticles(e.x, e.y, 12, '#ff8844');`

---

### m2 (NEW — R1 regression). Rally point line draws from wrong building position
**File:line:** `index.html:573-575`

```javascript
const bx=(e.x+e.w/2-cam.x)*z, by=(e.y+e.h/2-cam.y)*z;
// Should be (e.x-cam.x)*z, (e.y-cam.y)*z for center-based
```

**Fix:** `const bx=(e.x-cam.x)*z, by=(e.y-cam.y)*z;`

---

### m3 (NEW — R1 regression). Entity culling uses top-left bounds for buildings
**File:line:** `index.html:565`

```javascript
if(ex+e.w*z<-50||ey+e.h*z<-50||ex>vw+50||ey>vh+50) continue;
```

For center-based buildings, `ex` is the center. `ex+e.w*z` tests `center + fullWidth`, not the right edge (`center + halfWidth`). This is lenient (buildings stay on screen longer than necessary), causing minor performance waste but no visual bugs.

**Fix:** `if(ex-e.w*z/2>vw+50||ey-e.h*z/2>vh+50||ex+e.w*z/2<-50||ey+e.h*z/2<-50) continue;`

---

### m4 (PERSISTING from R0-m1). `drawHealthBar` dead empty-if branch
**File:line:** `index.html:293`

```javascript
if(ent.hp>=ent.maxHp && ent.owner==='player' && game.selectedIds.has(ent.id)) {}
if(ent.hp>=ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

The empty `if` block with no comment is confusing. It exists to prevent the second `if` from short-circuiting for selected entities at full HP.

**Fix:** Simplify to single condition:
```javascript
if(ent.hp >= ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

---

### m5 (PERSISTING from R0-m2). `game.trainingBuildings` and `game.trainingQueue` unused
**File:line:** `index.html:135-136`

Dead arrays initialized on the game state but never populated or read. Training uses `building.queue` directly.

**Fix:** Remove lines 135-136.

---

### m6 (PERSISTING from R0-m3). `worldToScreen` defined but never called
**File:line:** `index.html:658-663`

Dead code with zero call sites.

**Fix:** Remove or wire up for tooltip positioning (the `#tooltip` element also exists in DOM but is never shown).

---

### m7 (PERSISTING from R0-m4). `debug-info` span never populated
**File:line:** `index.html:62`

```html
<span id="debug-info">debug</span>
```

Present in the top bar, never updated. Clutters the UI.

**Fix:** Remove the element or populate it with useful data (FPS, entity count).

---

### m8 (PERSISTING). Rally point can be set on impassable terrain
**File:line:** `index.html:1324` (rally point assignment)

No validation that the rally point is on walkable terrain. Units can spawn on water or in trees.

**Fix:** Validate `isPassable(rallyPoint.x, rallyPoint.y)` before setting, or at spawn time adjust position.

---

## Regression Analysis vs. Implementation Plan

| Plan Feature | R0 Status | R1 Status | Delta |
|-------------|-----------|-----------|-------|
| Entity coordinate system | ❌ Inconsistent | ⚠️ Partial — center-based but 6 callsites stale | **NEW REGRESSIONS** |
| Building rendering | ⚠️ Offset bug | ✅ Fixed | DRAWING CORRECT |
| Building placement | ⚠️ Overlap detection broken | ✅ Fixed | PLACEMENT CORRECT |
| `isBuilding` flag on entities | ❌ Missing | ✅ Added but **unused** | WIRED INCORRECTLY |
| Resource gathering (dropoff nav) | ⚠️ Partial | ❌ **Broken** — wrong center calc | REGRESSION |
| Training queue unit spawn | ⚠️ Partial | ❌ **Broken** — wrong spawn pos | REGRESSION |
| Entity separation (building drift) | ❌ Bug | ❌ **Not fixed** | UNCHANGED |
| `issueMove` clears peasant state | ❌ Bug | ❌ **Not fixed** | UNCHANGED |
| `isPlacementValid` allows trees | ❌ Bug | ❌ **Not fixed** | UNCHANGED |
| Building death food clamp | ❌ Bug | ❌ **Not fixed** | UNCHANGED |
| `isPassable` coords | ⚠️ Top-left bug | ❌ **Worse** — center-based but code expects top-left | REGRESSION |
| Death particle spawn position | ⚠️ Offset | ❌ **Wrong** — right-edge instead of center | REGRESSION |
| Rally point line origin | ⚠️ Offset | ❌ **Wrong** — right-edge instead of center | REGRESSION |
| Unit attack / Watch Tower | ❌ Not implemented | ❌ **Not fixed** | UNCHANGED |
| HiDPI support | ❌ Missing | ❌ **Not fixed** | UNCHANGED |
| Peasant-soft-lock recovery | ❌ Missing | ❌ **Not fixed** | UNCHANGED |

---

## Recommended Fix Order

### Immediate (BLOCKING — game is broken)
1. **B1** — Fix `findNearestDropoff` center calculation (1 line)  
2. **B2** — Add `!e.isBuilding` guard to separation loop (1 line)  
3. **B3** — Remove peasant state clearing from `issueMove` (3 lines)  

### High Priority (MAJOR — features broken or plan-violating)
4. **M1** — Fix training queue spawn position (2 lines)  
5. **M2** — Fix `isPassable` coordinate assumptions (1 line)  
6. **M4** — Fix resource loss on wrong-dropoff right-click (~15 lines, conditional)  
7. **M5/M6** — Clamp food on building/unit death (2 lines each)  
8. **M7** — Add `TREE` to placement rejection (1 line)  
9. **M3** — Fix mixed-selection right-click on resources (~5 lines)  

### Medium Priority
10. **M8** — Increase resource search radius or add feedback  
11. **M11** — Add HiDPI support (~4 lines)  
12. **M9** — Implement attack logic (new code block, ~15 lines)  
13. **M10** — Integrate `isPassable` into movement (~10 lines)  
14. **M13** — Add `produces` validation to `startTraining` (2 lines)  

### Low Priority (MINOR — cosmetic, dead code, edge cases)
15. **m1–m3** — Fix death particles, rally line, culling for center-based  
16. **m4–m7** — Clean up dead code (drawHealthBar, trainingBuildings, worldToScreen, debug-info)  
17. **m8** — Validate rally point terrain  
18. **M12** — Soft-lock recovery for all-peasants-dead  

---

## R1 Fix Quality Assessment

The R1 debugger fix (commit `78dba7f`) was **directionally correct** but **incomplete**. It identified the root cause (coordinate system inconsistency) and applied the fix to 4 locations:

| Location | Status |
|----------|--------|
| `drawBuilding`: `ctx.translate(ex-bw/2, ey-bh/2)` | ✅ Correct |
| `drawBuilding`: progress bar positions | ✅ Correct |
| `drawBuilding`: training queue bar positions | ✅ Correct |
| `drawBuilding`: selection highlight | ✅ Correct |
| `placeBuilding`: removed `-def.size.w/2` offset | ✅ Correct |
| `initGame`: removed `-48` offset from Town Hall | ✅ Correct |
| Added `isBuilding:true/false` to entity creation | ✅ Correct addition (but unused) |

**Missed (6 locations still using old assumptions):**
- `findNearestDropoff` (`e.x+e.w/2`) — **BLOCKING**
- `isPassable` (`px>=e.x`) — MAJOR correctness trap  
- Training spawn (`e.x+e.w/2`, `e.y+e.h`) — MAJOR  
- Building death particles (`e.x+e.w/2`) — MINOR  
- Rally point line (`e.x+e.w/2, e.y+e.h/2`) — MINOR  
- Entity culling bounds — MINOR  

The fixer also did not wire the new `isBuilding` flag into any logic (separation, movement, culling), leaving it as a dead field.

---

*End of Report*
