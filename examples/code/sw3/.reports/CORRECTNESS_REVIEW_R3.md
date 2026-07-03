# Correctness Review R3: RealmCraft RTS Game

**Date:** 2026-07-03 07:42 UTC
**Reviewer:** Code Reviewer (dubina-code-reviewer-reason-e59ac4)
**Artifact:** `index.html` (59,186 chars) + `test_game.js` (26,992 chars)
**Previous reviews:** R0 → R1 → R2 (`CORRECTNESS_REVIEW_R2.md`)
**Test suite:** 125/125 passing

---

## Executive Summary

This is a **delta review** that verifies the R2 findings and identifies new issues. The R2 review contained **three inaccurate findings** (M2, M5, M7) that are actually already fixed in the current code. However, I discovered one **new BLOCKING** bug (undefined `spawnFloatingText` function) and two **new MAJOR** bugs in the resource gathering state machine.

The game is 90% functional for the implemented features. The new BLOCKING bug has a **partial recovery** (only affects one frame at a time), but must be fixed.

| Category | Total | New in R3 | Inherited from R2 | R2 Findings Rejected |
|----------|-------|-----------|--------------------|-----------------------|
| **BLOCKING** | 1 | 1 | 0 | — |
| **MAJOR** | 7 | 2 | 5 | — |
| **MINOR** | 8 | 1 | 7 | — |
| **R2 findings WRONG** | 3 | — | — | M2, M5, M7 |
| **TOTAL** | **16** | **4** | **12** | **3 rejected** |

---

## ❌ R2 Findings — Rejected (Already Fixed)

These three R2 issues are **NOT present** in the current code:

| R2 ID | Claim | Reality |
|-------|-------|---------|
| **R2-M2** | `isPassable` never called during movement | `isPassable` IS called at lines 1090, 1094-1095 with axis-aligned sliding |
| **R2-M5** | `startTraining` doesn't validate `building.produces` | Validation `!building.produces\|\|!building.produces.includes(unitType)` exists at line ~802 |
| **R2-M7** | Right-click on resource requires `ent===null` | The priority chain has no such guard; resource gathering fires on any tile regardless of entity presence after priority-1 check |

> **Note to future reviewers:** The R2 report was based on an older code version. These three findings should be removed from the backlog.

---

## 🔴 BLOCKING Issues

### B1 (NEW). `spawnFloatingText` is NEVER defined — ReferenceError at runtime
**File:line:** `index.html` — 7 call sites at lines 170, 819, 1000, 1015, 1044, 1143, and also referenced from `showFeedback` at line 170

The function `spawnFloatingText()` is called extensively but has **no definition** anywhere in the codebase. Only `spawnParticles()` (line 164) exists.

**Call sites:**
| Line | Context | Trigger |
|------|---------|---------|
| 170 | `showFeedback()` | Debug feedback on UI actions |
| 819 | `placeBuilding()` auto-complete | Building Town Hall with no peasants |
| 1000 | `updateEntity()` construction complete | Any building finishes construction |
| 1015 | `updateEntity()` unit spawn | Any unit finishes training |
| 1044 | `updateEntity()` attack damage | Any attack lands |
| 1143 | `updateEntity()` resource deposit | Peasant deposits gold/wood |

**Impact:** Every time any of these events fire, a `ReferenceError` is thrown inside `updateEntity()`. This terminates the `for...of` entity update loop for that frame — entities after the crashing one are not updated that tick. The game recovers on the next frame, but experiences:
- Frame skips whenever buildings complete or units finish training
- Missing floating feedback text (the entire purpose of the call)
- Entities iterated after the crashing entity get fewer updates

**Fix:** Add the function definition. It should create floating text objects in `game.floatingTexts`:
```javascript
function spawnFloatingText(x, y, text, color) {
  game.floatingTexts.push({x, y, yOff: 0, text, color, life: 1.2});
}
```

---

## 🟠 MAJOR Issues

### M1 (NEW). `issueMove` does not clear `gatheringNode`/`carryAmount` — ghost-gathering exploit
**File:line:** `index.html:863-872`

```javascript
function issueMove(targetX, targetY) {
  for(const id of game.selectedIds) {
    const e=game.entities.get(id);
    if(!e||e.owner!=='player') continue;
    if(e.isBuilding) continue;
    e.state='moving'; e.moveTarget={x:targetX,y:targetY};
    e.attackTarget=null;
    e.attackMove=isAttackMove;
    // BUG: gatheringNode, carryAmount, carryType, buildTarget NOT cleared
  }
}
```

Compare with `commandStop()` (line 1313-1314) which correctly clears all of these. When a gathering peasant is ordered to move, their `gatheringNode` remains set. Upon arrival at the new location (line 1082-1083), the arrival code re-enters gathering state:

```javascript
// Arrival: d<3
e.state='idle'; e.moveTarget=null;
if(e.type==='peasant'&&e.gatheringNode) {
    e.state='gathering'; e.gatheringTimer=0;  // STARTS GATHERING AT NEW LOCATION
}
```

**Exploit chain:**
1. Peasant is gathering at tree → `gatheringNode` set
2. Player issues move command to grass tile
3. Peasant walks to grass → arrival → re-enters gathering state on grass
4. 1.5s later: `carryAmount = 10` (resources from thin air!)
5. Peasant finds dropoff and deposits → **free resources generated**

**Fix:** Clear peasant-specific state in `issueMove` or, better, guard the arrival gathering check:
```javascript
// Option A: clear in issueMove (consistent with commandStop)
if(e.type==='peasant') {
    e.gatheringNode=null; e.carryAmount=0; e.carryType=null;
}
// Option B: guard in arrival code
if(e.type==='peasant'&&e.gatheringNode) {
    const gt = tileAt(e.gatheringNode.x, e.gatheringNode.y);
    if(gt===TERRAIN.GOLD||gt===TERRAIN.TREE) {
        e.state='gathering'; e.gatheringTimer=0;
    } else {
        e.gatheringNode=null; e.carryAmount=0; e.carryType=null;
    }
}
```

---

### M2 (NEW). Gathering state machine never validates resource tile
**File:line:** `index.html:1113-1117`

```javascript
if(e.type==='peasant'&&e.state==='gathering') {
    e.gatheringTimer+=dt;
    if(e.gatheringTimer>=1.5) {
      e.carryAmount=10;  // Resources awarded with NO tile check
```

The gathering timer awards `carryAmount=10` after 1.5 seconds with **no check** that the peasant is actually standing on a resource tile. This is the root cause that enables M1's ghost-gathering exploit. Even without M1, entity separation (line 1164-1176) can push a peasant off a resource tile and they'll continue generating resources from empty terrain.

**Fix:** Add a tile check before awarding resources:
```javascript
if(e.gatheringTimer>=1.5) {
    const currentTile = tileAt(e.x, e.y);
    const expectedTile = e.carryType==='gold' ? TERRAIN.GOLD : TERRAIN.TREE;
    if(currentTile === expectedTile) {
        e.carryAmount = 10;
    }
    // If not on correct tile, reset or move back to node
    e.gatheringTimer = 0;
}
```

---

### M3 (R2-M1). No attack logic — military units are cosmetic
**File:line:** `index.html:1029-1065`

Confirmed from R2. The `attackTarget` → target pursuit → damage loop exists at lines 1029-1065 but there is no code to **acquire** targets. Attack targets can only be set via right-click on an enemy entity (line 1386), and since there are no enemy entities in the game, combat is unreachable. Watch Tower attack stats (damage:15, range:160) are never used.

**Fix:** Add auto-acquire logic in `updateEntity()` for units with `attackDamage > 0`. When idle or attack-moving, scan for enemies in attackRange and auto-target.

---

### M4 (R2-M3). Mixed selection ignores non-peasants on resource right-click
**File:line:** `index.html:1398-1410`

```javascript
if(hasPeasant) {
    for(const id of game.selectedIds) {
        const e=game.entities.get(id);
        if(e&&e.type==='peasant') issueGather(e, world.x, world.y);
        // NON-PEASANTS GET NO COMMAND
    }
}
```

When peasants + military units are selected, right-clicking a resource gives gather orders to peasants but **nothing** to military units. They stand still. Expected: non-peasants should at least move to the clicked location.

**Fix:**
```javascript
if(hasPeasant) {
    for(const id of game.selectedIds) {
        const e=game.entities.get(id);
        if(e&&e.type==='peasant') issueGather(e, world.x, world.y);
        else if(e&&e.owner==='player'&&!e.isBuilding) {
            e.state='moving'; e.moveTarget={x:world.x,y:world.y};
        }
    }
}
```

---

### M5 (R2-M4). Carrying peasant loses resources when arriving at non-dropoff building
**File:line:** `index.html:1082-1083` (arrival), `1135-1154` (deposit)

When a peasant carrying resources arrives at a non-dropoff building (e.g., Barracks):

1. Movement arrival: `d<3` → `state='idle'`, `moveTarget=null`
2. `gatheringNode` still set → `state='gathering'`
3. `gatheringTimer` accumulates → `carryAmount=10` **overwrites** the carried resources
4. Resources are lost

The deposit code at line 1135 only fires when `state==='moving'&&carryAmount>0`, but state is already changed to 'gathering' by the arrival code in the same frame.

**Fix:** Either:
```javascript
// In movement arrival, before checking gatheringNode:
if(e.carryAmount>0) {
    // Don't overwrite carried resources – re-route to nearest dropoff
    const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
    if(dropoff) {
        e.state='moving'; e.moveTarget={x:dropoff.x, y:dropoff.y};
        return; // skip gatheringNode check
    }
}
```

---

### M6 (R2-M6). No peasant soft-lock recovery
**File:line:** N/A (design gap)

If all peasants die, the player cannot build, gather, or train new peasants — even with a Town Hall and sufficient resources. The only recovery mechanism (`placeBuilding` auto-complete for Town Hall) only triggers when building a *new* Town Hall, which requires having a peasant to start placement. Dead end.

**Fix:** Either show a "Game Over" message or allow Town Hall to train peasants directly (cooldown-based, no peasant interaction required) when `peasantCount === 0`.

---

### M7 (NEW). Test `startTraining` lacks `produces` validation — test gap
**File:line:** `test_game.js:201-220`

The test file's `startTraining()` function does NOT validate `building.produces`, unlike the game code (line ~802). While the existing tests only test valid unit-building combinations, this gap means regressions in the validation logic would not be caught.

**Fix:** Add the validation line to the test's `startTraining`:
```javascript
if(!building.produces || !building.produces.includes(unitType))
    return { success: false, reason: 'building_cannot_produce' };
```
And add a test case for invalid unit-building combinations.

---

## 🟡 MINOR Issues

### m1 (NEW). Duplicate `attackTarget` property in `createUnit`
**File:line:** `index.html:269, 276`

```javascript
targetId:null, moveTarget:null, path:[], attackTarget:null,  // line 269
...
attackTarget:null, // entity ID being attacked  // line 276 (duplicate)
```

The first `attackTarget` is dead code — JS object literals use the last value. Only the second assignment matters. Remove the first occurrence to avoid confusion.

---

### m2 (R2-m2). `drawHealthBar` empty if-block — confusing control flow
**File:line:** `index.html:293-294`

```javascript
if(ent.hp>=ent.maxHp && ent.owner==='player' && game.selectedIds.has(ent.id)) {}
if(ent.hp>=ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

The empty block exists only to prevent selected-but-full-HP entities from being caught by the second condition. Works but is confusing. Simplify to one condition:
```javascript
if(ent.hp >= ent.maxHp && !game.selectedIds.has(ent.id)) return;
```

---

### m3 (R2-m3). Dead arrays `game.trainingBuildings` and `game.trainingQueue`
**File:line:** `index.html:135-136`

Initialized but never populated or read. Training uses `building.queue` directly. Remove.

---

### m4 (R2-m4). Unused `worldToScreen` and `#tooltip` element
**File:line:** `index.html:658-663` (worldToScreen), line 79 (tooltip DOM), lines 48-49 (tooltip CSS)

`worldToScreen` has zero call sites. The `#tooltip` DOM element has CSS styling but is never populated or shown by JS. Either wire up tooltip functionality (entity hover showing name/HP) or remove both.

---

### m5 (R2-m5). `debug-info` span always shows "debug"
**File:line:** `index.html:62`

```html
<span id="debug-info">debug</span>
```

Visible in the top bar, always shows "debug". Clutters the UI. Either remove or populate with useful debug data (FPS, entity count, camera position).

---

### m6 (R2-m6). Rally point settable on impassable terrain
**File:line:** `index.html:1425-1427`

```javascript
if(game.selectedIds.has(ent.id)) {
    ent.rallyPoint={x:world.x,y:world.y};  // No terrain validation
}
```

Units spawned at a rally point on water/trees would be stuck. Validate with `isPassable()` or adjust spawn to nearest passable tile.

---

### m7 (R2-m8). `findNearestResource` returning null → silent no-op in `issueGather`
**File:line:** `index.html:842-853`

If no resource exists within 30 tiles, the peasant stays idle with no visual feedback after the `showFeedback` call (which itself crashes due to B1). Add a fallback movement or at minimum ensure the feedback works after B1 is fixed.

---

### m8. `resizeCanvas` does not clamp camera after resize
**File:line:** `index.html:1323-1331`

```javascript
function resizeCanvas() {
  const rect=wrapper.getBoundingClientRect();
  canvas.width=rect.width*dpr;
  canvas.height=rect.height*dpr;
  // BUG: camera not re-clamped — may go out of bounds for one frame
}
```

After resize, the camera position may be invalid relative to new viewport dimensions. The camera is only clamped in `update()` (line 928), so there's a one-frame window where the camera could show out-of-bounds area or render incorrectly.

**Fix:** Add camera clamp at end of `resizeCanvas`:
```javascript
game.camera.x = clamp(game.camera.x, 0, Math.max(0, WORLD_W - canvas.clientWidth / game.camera.zoom));
game.camera.y = clamp(game.camera.y, 0, Math.max(0, WORLD_H - canvas.clientHeight / game.camera.zoom));
```

---

## Corrected R2 Status Table

| R2 Finding | R2 Status | R3 Status | Notes |
|-----------|-----------|-----------|-------|
| M1: No attack logic | MAJOR | **Still MAJOR (M3)** | Confirmed |
| M2: isPassable unused | MAJOR | **FIXED** | Movement code calls isPassable at lines 1090, 1094-1095 |
| M3: Mixed selection | MAJOR | **Still MAJOR (M4)** | Confirmed |
| M4: Resource loss at wrong dropoff | MAJOR | **Still MAJOR (M5)** | Confirmed, mechanism clarified |
| M5: startTraining validation | MAJOR | **FIXED** | Validation at line ~802 |
| M6: Peasant soft-lock | MAJOR | **Still MAJOR (M6)** | Confirmed |
| M7: Resource click blocked by entity | MAJOR | **FIXED** | No `ent===null` guard exists in current code |
| m1: Entity culling bounds | MINOR | **Still MINOR** | Confirmed, low priority |
| m2: drawHealthBar empty if | MINOR | **Still MINOR (m2)** | Confirmed |
| m3: Dead training arrays | MINOR | **Still MINOR (m3)** | Confirmed |
| m4: worldToScreen unused | MINOR | **Still MINOR (m4)** | Confirmed |
| m5: debug-info span | MINOR | **Still MINOR (m5)** | Confirmed |
| m6: Rally point terrain | MINOR | **Still MINOR (m6)** | Confirmed |
| m7: cancelPlacement guard | MINOR | **Not reviewed** | Low priority |
| m8: findNearestResource null | MINOR | **Still MINOR (m7)** | Confirmed |
| m9: Test Bug Findings stale | MINOR | Partially addressed | Tests pass but section naming misleading |

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Test suite | 125/125 ✅ |
| BLOCKING issues | 1 (NEW) |
| MAJOR issues | 7 (2 new, 5 inherited) |
| MINOR issues | 8 (1 new, 7 inherited) |
| R2 findings rejected | 3 (M2, M5, M7) |
| Defined-but-missing function | 1 (spawnFloatingText) |
| Resource integrity bugs | 2 (M1 ghost-gathering, M2 no-tile-check) |
| Dead code items | 4 (worldToScreen, trainingBuildings, trainingQueue, tooltip DOM) |
| Features working | Resource gathering (with bugs), building placement, unit training, selection, movement+collision, camera, minimap |
| Features missing | Combat, peasant recovery |

---

## Recommended Fix Order

### Immediate (BLOCKING)
1. **B1** — Define `spawnFloatingText()` function (3 lines)

### High Priority (MAJOR — resource integrity + gameplay)
2. **M1 + M2** — Fix issueMove clearing + gathering tile validation (~8 lines) — fixes ghost-gathering exploit
3. **M5** — Fix resource loss on wrong-dropoff arrival (~6 lines)
4. **M4** — Fix mixed-selection on resource right-click (~4 lines)
5. **M7** — Add `produces` validation to test `startTraining` + test case (~6 lines)
6. **M3** — Implement basic attack auto-acquire (~15 lines)
7. **M6** — Add peasant soft-lock recovery (~10 lines)

### Low Priority (MINOR)
8. **m1** — Remove duplicate `attackTarget`
9. **m2-m5** — Clean up dead code and confusing patterns
10. **m6** — Validate rally point terrain
11. **m7** — Add fallback for null resource search
12. **m8** — Clamp camera after resize

---

*End of Report*
