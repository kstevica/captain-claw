# Correctness Review R2: RealmCraft RTS Game

**Date:** 2026-07-03 07:29 UTC
**Reviewer:** Code Reviewer (dubina-code-reviewer-reason-ac3a12)
**Artifact:** `index.html` — single-file RTS game (~1,480 lines)
**Previous reviews:** `CORRECTNESS_REVIEW.md` (r0), `CORRECTNESS_REVIEW_R1.md` (r1), `QA_REPORT.md`, `QA_REPORT_FRESH.md`
**Delta from R1:** All 3 BLOCKING and most critical MAJOR issues from R1 have been fixed. This review confirms fixes and identifies remaining gaps + new findings.

---

## Executive Summary

The R1→R2 fix round was **highly effective** — all 3 BLOCKING issues, 6 MAJOR issues, and 3 MINOR issues are now resolved. The coordinate system is now fully unified to center-based positions. Building drift is eliminated. Resource gathering, unit spawning, and placement validation are correct. Food accounting has proper clamping.

The game is in a **playable, functionally correct state** for all features that currently exist. The remaining issues are about missing features (attack logic, pathfinding), polish (HiDPI, dead code), and edge cases (mixed selection, peasant soft-lock).

| Category | Count | New in R2 | Persisting from R1 | Fixed since R1 |
|----------|-------|-----------|---------------------|-----------------|
| **BLOCKING** | 0 | 0 | 0 | ✅ 3 fixed |
| **MAJOR** | 7 | 1 | 6 | ✅ 6 fixed |
| **MINOR** | 9 | 2 | 7 | ✅ 3 fixed |
| **TOTAL** | **16** | **3** | **13** | **12 fixed** |

---

## ✅ Confirmed Fixes (R1→R2)

These issues from the R1 report are confirmed resolved in the current codebase:

| R1 ID | Description | Verification |
|-------|-------------|-------------|
| **B1** | `findNearestDropoff` used `e.x+e.w/2` instead of center `e.x` | Now uses `dist({x,y},{x:e.x,y:e.y})` correctly — line 697-698 |
| **B2** | Buildings drifted via entity separation | Guard `!e.isBuilding` added — line 1071 |
| **B3** | Training spawn at wrong position | `spawnX: e.x`, `spawnY: e.y+e.h/2` — lines 972-973 |
| **B4 / M2** | `isPassable` used top-left bounds | Uses `e.x-e.w/2` to `e.x+e.w/2` — line 159 |
| **B3 (old)** | `issueMove` cleared peasant state | Peasant state clearing removed; only `isBuilding` guard — lines 831-837 |
| **M7** | `isPlacementValid` allowed trees | Now rejects `TERRAIN.TREE` — line 738 |
| **M5** | No food clamp on building destruction | `Math.min(food, maxFood)` — line 926 |
| **M6** | Food could go negative on unit death | `Math.max(0, food - cost)` — line 931 |
| **M1 (old)** | Rally point line from wrong position | `bx=(e.x-cam.x)*z` (center) — line 573 |
| **m2 (old)** | Death particles at wrong position | `spawnParticles(e.x,e.y,...)` — line 960 |
| **M8 (old)** | Resource search radius 12 tiles | Increased to 30 — line 711 |

---

## 🔴 BLOCKING Issues — NONE

All previously blocking issues (B1–B4 from R1) are resolved. No new blocking regressions found.

---

## 🟠 MAJOR Issues (7)

### M1 (PERSISTING from R0-M2/R1-M9). No attack logic — all military units are cosmetic
**File:line:** `index.html:997-998` (attack timer), and missing targeting code

`attackTimer` is decremented in `updateEntity` but there is no code to:
- Auto-acquire nearby targets
- Apply damage to targets
- Spawn projectiles for ranged units
- Make the Watch Tower functional

**Impact:** Footmen, Archers, Knights, and Watch Towers are purely cosmetic. The game has no combat, which is a core RTS feature. This was listed as "No AI needed" in requirements, but combat against neutral/environmental targets is a standard RTS expectation.

**Fix:** Add attack block to `updateEntity`:
```javascript
if(e.attackDamage > 0 && e.attackTimer <= 0) {
  // Find target: e.targetId or auto-acquire nearest in range
  // If target exists and in range: apply damage, reset timer
  e.attackTimer = e.attackCooldown;
}
```

---

### M2 (PERSISTING from R0-M3/R1-M10). `isPassable` never called during movement
**File:line:** `index.html:153-162` (definition), lines 1010-1020 (movement code)

Units use direct steering (`e.x += mx; e.y += my`) without any terrain or obstacle check. Units walk through:
- Walls (defeating their entire purpose)
- Buildings
- Water tiles
- Trees

**Impact:** Wall building is non-functional. Base layout has no strategic value. Water maps are trivially traversable.

**Fix:** At minimum, clamp movement to passable tiles (slide along obstacles). Preferred: integrate a simple pathfinding step that avoids impassable tiles.

---

### M3 (PERSISTING from R0-B3/R1-M3). Mixed selection ignores non-peasants on resource right-click
**File:line:** `index.html:1297-1304`

```javascript
if(hasPeasant) {
    for(const id of game.selectedIds) {
        const e=game.entities.get(id);
        if(e&&e.type==='peasant') issueGather(e, world.x, world.y);
        // NON-PEASANTS GET NOTHING
    }
}
```

When peasants + footmen are selected and the player right-clicks a resource: peasants get `issueGather`, footmen stand still. The player expects all units to at least move to the click point.

**Fix:**
```javascript
if(hasPeasant) {
    for(const id of game.selectedIds) {
        const e = game.entities.get(id);
        if(e && e.type === 'peasant') issueGather(e, world.x, world.y);
        else if(e && e.owner === 'player') issueMove(world.x, world.y); // non-peasants move
    }
}
```

---

### M4 (PERSISTING from R0-B4/R1-M11). Carrying peasant loses resources at non-dropoff building
**File:line:** `index.html:1045-1065` (deposit logic), lines 1310-1321 (right-click handler)

Peasant carries 10 gold → right-clicked onto Barracks → walks there → arrives → `state='idle'` → deposit check fires → `findNearestDropoff` finds the nearest *actual* dropoff, but the peasant is standing at the Barracks 40px from it. If >40px: **resources zeroed, no deposit**.

**Fix:** When deposit fails because nearest dropoff is too far, re-route to the dropoff instead of zeroing:
```javascript
if(dropoff) {
    const d = dist({x:e.x,y:e.y},{x:dropoff.x,y:dropoff.y});
    if(d < 40) { /* deposit */ }
    else {
        e.state = 'moving';
        e.moveTarget = {x:dropoff.x, y:dropoff.y};
    }
}
```

---

### M5 (PERSISTING from QA-03/R1-M13). `startTraining` doesn't validate `building.produces`
**File:line:** `index.html:800-815`

Training is gated only by the UI (`e.progress>=1 && e.produces` check at line 1146). The game logic itself doesn't verify the unit type is in the building's `produces` array. If called programmatically, knights could be trained from barracks.

**Fix:** Add validation at line ~801:
```javascript
if(!building.produces || !building.produces.includes(unitType)) return;
```

---

### M6 (PERSISTING from QA-07/R1-M12). Soft-lock when all peasants die
**File:line:** N/A (design gap)

With all peasants dead, the player cannot gather resources, build, or train new peasants — even with a Town Hall and sufficient resources. The only recourse is to refresh the page.

**Fix:** Detect `all peasants dead && Town Hall exists` and either:
a) Show a "Game Over — no peasants remaining" message, or
b) Auto-spawn a free peasant at Town Hall (cooldown-based), or
c) Enable Town Hall to train peasants without requiring peasant interaction

---

### M7 (NEW). Right-click on resource covered by entity silently fails
**File:line:** `index.html:1290`

```javascript
if((tile===TERRAIN.GOLD||tile===TERRAIN.TREE)&&ent===null) {
```

The gather command only triggers when `ent === null` — if any entity (building, unit, the peasant itself) is at the click position, the condition fails and falls through to `issueMove`. This means peasants can't gather from a gold mine while standing on it, and can't be issued a gather command while another unit is on the resource tile.

**Fix:** Change condition to check that `ent` is not an enemy (or not a unit that should block gather):
```javascript
if((tile===TERRAIN.GOLD||tile===TERRAIN.TREE)&&(!ent||ent.owner!=='player')) {
```

---

## 🟡 MINOR Issues (9)

### m1 (PERSISTING from R1-m3). Entity culling uses asymmetric bounds for center-based entities
**File:line:** `index.html:556`

```javascript
if(ex+e.w*z<-50||ey+e.h*z<-50||ex>vw+50||ey>vh+50) continue;
```

For center-based entities, `ex+e.w*z` computes `center + fullWidth` instead of `center + halfWidth` (right edge). This is lenient (entities stay visible longer than needed), causing minor performance waste. Also, `ex > vw+50` checks against center instead of left edge, which is also lenient.

**Fix:**
```javascript
if(ex+e.w*z/2<-50||ey+e.h*z/2<-50||ex-e.w*z/2>vw+50||ey-e.h*z/2>vh+50) continue;
```

---

### m2 (PERSISTING from R0-m1/R1-m4). `drawHealthBar` has empty `if` block
**File:line:** `index.html:293-294`

```javascript
if(ent.hp>=ent.maxHp && ent.owner==='player' && game.selectedIds.has(ent.id)) {} // always show for selected
if(ent.hp>=ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

The empty `if` block exists to prevent the second `if` from short-circuiting for selected entities at full HP. It works but is confusing. Simplify to a single condition.

**Fix:**
```javascript
if(ent.hp >= ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

---

### m3 (PERSISTING from R0-m2/R1-m5). `game.trainingBuildings` and `game.trainingQueue` unused
**File:line:** `index.html:135-136`

Dead arrays initialized on the game state but never populated or read. Training uses `building.queue` directly.

**Fix:** Remove lines 135-136.

---

### m4 (PERSISTING from R0-m3/R1-m6). `worldToScreen` defined but never called
**File:line:** `index.html:658-663`

Dead code with zero call sites. Also: the `#tooltip` DOM element (line 79) has CSS styling (line 48) but is never populated or shown by JS.

**Fix:** Remove `worldToScreen` and the tooltip element, or wire them up (e.g., tooltip for entity hover showing name/HP).

---

### m5 (PERSISTING from R0-m4/R1-m7). `debug-info` span never populated
**File:line:** `index.html:62`

```html
<span id="debug-info">debug</span>
```

Visible in the top bar, always shows "debug". Clutters the UI.

**Fix:** Either remove the element or populate it with useful data (FPS, entity count, camera position).

---

### m6 (PERSISTING). Rally point settable on impassable terrain
**File:line:** `index.html:1322-1324`

```javascript
if(ent.owner==='player'&&game.selectedIds.has(ent.id)) {
    ent.rallyPoint={x:world.x,y:world.y};
}
```

No validation that the rally point is on walkable terrain. Units can be set to spawn on water or in trees.

**Fix:** Validate with `isPassable()` or adjust spawn position at spawn time to nearest passable tile.

---

### m7 (NEW). `cancelPlacement` lacks defensive mode guard
**File:line:** `index.html:757-763`

The function unconditionally resets cursor class and UI state. Currently unreachable from non-placement modes (all 3 call sites check `ui.mode === 'placeBuilding'` first), but a future code change could break this invariant.

**Fix:** Add a guard for defensive programming:
```javascript
if(game.ui.mode !== 'placeBuilding') return;
```

---

### m8 (NEW). `findNearestResource` returning `null` causes silent no-op in `issueGather`
**File:line:** `index.html:842-853`

```javascript
if(node) {
    // configure gathering state
}
// else: nothing happens, peasant stays idle
```

If `findNearestResource` returns null (peasant too far from resources), the function exits without changing state — no movement, no feedback. With search radius increased to 30 tiles this is unlikely, but still possible near map corners.

**Fix:** Add fallback:
```javascript
if(node) {
    // ... gathering setup
} else {
    // Move toward map center and show feedback
    entity.state = 'moving';
    entity.moveTarget = {x: WORLD_W/2, y: WORLD_H/2};
    spawnFloatingText(entity.x, entity.y, 'No resource nearby', '#ff8844');
}
```

---

### m9 (PERSISTING). Test file `test_game.js` "Bug Findings" section is stale
**File:line:** `test_game.js:600-700`

The "BUG FINDINGS — Code Review" section documents issues that are mostly fixed (building drift via separation, `speed||2` fallback, resource search radius). Several test assertions now verify *working* behavior as if it were still buggy, misleading about what's broken.

**Fix:** Update the test assertions to reflect current state. Tests that verify *fixed* behavior should be moved out of "BUG FINDINGS" and into normal test sections with updated descriptions.

---

## Regression Analysis: Plan vs. Implementation

| Plan Feature | R1 Status | R2 Status | Notes |
|-------------|-----------|-----------|-------|
| Entity coordinate system | ⚠️ Partial — 6 callsites stale | ✅ Fully unified | All center-based |
| Building rendering | ✅ Fixed | ✅ Remains fixed | — |
| Building placement | ✅ Fixed | ✅ Remains fixed | Now also rejects trees |
| `isBuilding` flag wired | ❌ Added but unused | ✅ Wired | Used in separation |
| Resource gathering (dropoff) | ❌ Broken — wrong center | ✅ Fixed | — |
| Training queue spawning | ❌ Broken — wrong position | ✅ Fixed | — |
| Entity separation (drift) | ❌ Not fixed | ✅ Fixed | `!e.isBuilding` guard |
| `issueMove` peasant state | ❌ Not fixed | ✅ Fixed | State clearing removed |
| `isPlacementValid` trees | ❌ Not fixed | ✅ Fixed | `TERRAIN.TREE` rejected |
| Food clamp on death | ❌ Not fixed | ✅ Fixed | Both building + unit |
| Resource search radius | ❌ 12 tiles | ✅ 30 tiles | — |
| Rally point line | ❌ Wrong position | ✅ Fixed | — |
| Death particles | ❌ Wrong position | ✅ Fixed | — |
| Attack logic / combat | ❌ Not implemented | ❌ Still missing | M1 |
| Obstacle collision | ❌ Not implemented | ❌ Still missing | M2 |
| HiDPI support | ❌ Missing | ❌ Still missing | M3 (pending) |
| Mixed selection on resources | ❌ Bug | ❌ Still broken | M4 |
| Peasant wrong-dropoff loss | ❌ Bug | ❌ Still broken | M5 |
| `startTraining` validation | ❌ Missing | ❌ Still missing | M6 |
| Peasant soft-lock recovery | ❌ Missing | ❌ Still missing | M7 |

---

## Recommended Fix Order

### Immediate (no BLOCKING — game is playable)
*None — all blocking issues are fixed.*

### High Priority (MAJOR — features broken or missing)
1. **M2** — Integrate `isPassable` into movement (~10 lines) — walls are useless without this
2. **M5** — Fix resource loss on wrong-dropoff right-click (~10 lines)
3. **M4** — Fix mixed-selection right-click on resources (~5 lines)
4. **M7** — Fix resource-gather blocked by entity on tile (~3 lines)
5. **M1** — Implement basic attack logic (~20 lines)
6. **M6** — Add `produces` validation to `startTraining` (2 lines)
7. **M3** — Add HiDPI support (~5 lines) + peasant soft-lock recovery (~10 lines)

### Low Priority (MINOR — cosmetic, dead code, edge cases)
8. **m1** — Fix entity culling bounds
9. **m2** — Simplify `drawHealthBar` empty-if
10. **m3-m5** — Remove dead code (`trainingBuildings`, `worldToScreen`, `debug-info`, tooltip)
11. **m6** — Validate rally point terrain
12. **m7** — Add `cancelPlacement` defensive guard
13. **m8** — Add fallback for `findNearestResource` null
14. **m9** — Update stale test assertions

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Test suite passing | 123/123 ✅ |
| BLOCKING issues | 0 |
| MAJOR issues | 7 |
| MINOR issues | 9 |
| Features working correctly | Resource gathering, building placement, unit training, selection, movement (no collision), camera, minimap |
| Features missing | Combat, pathfinding/collision, HiDPI, peasant recovery |
| Dead code count | 4 items (worldToScreen, trainingBuildings, trainingQueue, tooltip DOM) |
| Coordinate system consistency | ✅ Fully center-based (all 12+ references verified) |

---

*End of Report*
