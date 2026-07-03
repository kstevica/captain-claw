# Correctness Review R4: Auto-Build Dropoff System

**Date:** 2026-07-03 08:48 UTC
**Reviewer:** Code Reviewer (dubina-code-reviewer-reason-d1683e)
**Artifact:** `index.html` + `test_game.js`
**Previous reviews:** R0 → R1 → R2 → R3
**Focus:** "When worker gets gold or wood and carries it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it."

---

## Executive Summary

This review focuses specifically on the auto-build dropoff system — the peasant gathering → carrying → deposit → auto-build pipeline. The auto-build infrastructure **exists and is wired up correctly**, covering: gathering completion (line 1235), deposit arrival (line 1269), and idle retry (line 1308). However, I found **5 MAJOR** issues and **5 MINOR** issues in the system.

Several R3 findings have been **fixed** in this code version: `spawnFloatingText` is defined, `issueMove` clears peasant state, gathering tile validation exists, attack auto-acquire is present, and rally point terrain is validated. I only report issues that are *still present* or *newly discovered*.

| Category | Count |
|----------|-------|
| **MAJOR** | 5 |
| **MINOR** | 5 |
| **R3 fixes verified** | 6 |
| **TOTAL** | **10** |

---

## ✅ R3 Fixes Confirmed

To avoid rediscovery, here is what's been fixed since R3:

| R3 ID | Issue | Status |
|-------|-------|--------|
| B1 | `spawnFloatingText` undefined | ✅ Defined at lines 168-170 |
| M1 | `issueMove` doesn't clear gathering state | ✅ Clears at line 966 |
| M2 | Gathering no tile validation | ✅ Validates at lines 1220-1224 |
| M3 | No attack logic | ✅ Auto-acquire at lines 1145-1150 |
| — | `cancelPlacement` cursor clobber | ✅ Guard at line 862 |
| — | Rally point on impassable terrain | ✅ Guard at lines 1604-1606 |

---

## 🔴 MAJOR Issues

### M1. `commandStop` destroys carried resources — breaks auto-build preservation

**File:line:** `index.html:1484`

```javascript
function commandStop() {
  for(const id of game.selectedIds) {
    const e=game.entities.get(id);
    if(!e) continue;
    e.state='idle'; e.moveTarget=null; e.attackTarget=null; e.attackMove=false;
    if(e.type==='peasant') { e.buildTarget=null; e.gatheringNode=null; e.carryAmount=0; e.carryType=null; }
  }
}
```

**Impact:** The auto-build system carefully preserves `carryAmount`/`carryType` when no dropoff exists (lines 1238-1239, 1271-1272). The idle retry loop (lines 1300-1310) depends on these preserved values to retry. But if the player presses Stop (S key) at any point, `commandStop` unconditionally zeroes them — permanently destroying the resources the peasant gathered.

**Severity:** MAJOR. This creates a dissonance: the auto-build system tries to be resilient and preserve resources, but one keystroke silently destroys them. The player has no way to know they just lost 10 gold.

**Fix:** Preserve carryAmount when no dropoff exists yet:
```javascript
if(e.type==='peasant') {
    e.buildTarget=null;
    e.gatheringNode=null;
    // Only discard carried resources if a dropoff exists to deposit at
    if(e.carryAmount > 0 && findNearestDropoff(e.x, e.y, e.carryType)) {
        e.carryAmount = 0; e.carryType = null;
    }
}
```

---

### M2. Gathering timer discarded on entity separation push

**File:line:** `index.html:1220-1224`

```javascript
if(e.type==='peasant'&&e.state==='gathering') {
    const curTile=tileAt(e.x,e.y);
    if(curTile!==(e.carryType==='gold'?TERRAIN.GOLD:TERRAIN.TREE)) {
      e.state='idle'; e.gatheringNode=null; e.carryAmount=0; e.carryType=null; return;
    }
    e.gatheringTimer+=dt;
```

**Impact:** Entity separation (lines 1276-1296) can push a peasant off the resource tile. When this happens during gathering, the entire `gatheringTimer` (up to 1.4s of accumulated progress) is discarded — the peasant resets to idle and forgets both the gathering node and the timer. The peasant must be re-clicked on the resource to restart from 0.

This is particularly punishing because:
- Separation forces are applied every frame
- Even a 1-pixel nudge triggers full reset
- Other peasants gathering nearby are the most common cause of separation

**Severity:** MAJOR. Makes gathering unreliable when multiple peasants work the same node or when buildings/units are nearby.

**Fix:** Walk back to the gathering node instead of resetting:
```javascript
if(curTile!==(e.carryType==='gold'?TERRAIN.GOLD:TERRAIN.TREE)) {
    if(e.gatheringNode) {
        e.state='moving';
        e.moveTarget={x:e.gatheringNode.x, y:e.gatheringNode.y};
    } else {
        e.state='idle'; e.gatheringNode=null; e.carryAmount=0; e.carryType=null;
    }
    return;
}
```
This preserves the timer (the peasant continues gathering upon arrival) and the gatheringNode reference.

---

### M3. `hasDropoffUnderConstruction` misses `town_hall` — redundant auto-build

**File:line:** `index.html:726-731`

```javascript
function hasDropoffUnderConstruction(resType) {
  const bldType=getDropoffBuildingType(resType);
  if(!bldType) return false;
  for(const e of game.entities.values()) {
    if(e.owner==='player'&&e.type===bldType&&e.progress<1) return true;
  }
  return false;
}
```

**Impact:** `getDropoffBuildingType('gold')` returns `'refinery'` and `getDropoffBuildingType('wood')` returns `'lumber_mill'`. It does NOT return `'town_hall'`, even though a town hall (once complete) serves as a universal dropoff for BOTH gold and wood.

Scenario:
1. Town hall is under construction (progress 0.5, 5s remaining)
2. Peasant finishes gathering gold → `findNearestDropoff` → null (town hall progress<1, excluded)
3. `hasDropoffUnderConstruction('gold')` → false (only checks 'refinery', not 'town_hall')
4. `autoBuildDropoff` proceeds → builds a refinery for 100g/50w
5. Town hall completes 5s later → now have both town hall AND refinery (redundant)

**Severity:** MAJOR. Wastes 100g/50w on a redundant building the player may not want. The town hall is 200g/100w — adding a refinery on top means 300g/150w spent on dropoff infrastructure.

**Fix:** Include town_hall in the under-construction check:
```javascript
function hasDropoffUnderConstruction(resType) {
  const bldType = getDropoffBuildingType(resType);
  for(const e of game.entities.values()) {
    if(e.owner==='player'&&e.progress<1) {
      if(e.type===bldType) return true;
      if(e.type==='town_hall') return true; // town hall is universal dropoff
    }
  }
  return false;
}
```

---

### M4. `autoBuildDropoff` repeated calls on non-cooldown failures → feedback spam

**File:line:** `index.html:765-810`

```javascript
function autoBuildDropoff(peasant) {
  // ... early returns for: no carryType, dropoff exists, under construction, cooldown...
  const cost=BUILDING_DEFS[buildingType].cost;
  if(game.resources.gold<(cost.gold||0)||game.resources.wood<(cost.wood||0)) {
    showFeedback('Need more resources for '+BUILDING_DEFS[buildingType].name);
    return false;  // <-- NO COOLDOWN SET
  }
  const site=findAutoBuildSite(buildingType, peasant.x, peasant.y);
  if(!site) return false;  // <-- NO COOLDOWN SET
  // ...
  game.autoBuildCooldown[buildingType]=game.time;  // <-- ONLY SET ON SUCCESS
```

**Impact:** The cooldown is only set when the building is actually created (line 797). If the function fails due to insufficient resources or no valid placement site, it returns `false` without setting the cooldown. The idle retry loop (line 1300-1310) then calls `autoBuildDropoff` again on the next 2-second cycle. This produces:

1. Repeated `showFeedback` calls every 2 seconds — spam in the debug-info display
2. Repeated `findAutoBuildSite` calls — wasted CPU (O(n²) in the fallback path)
3. If multiple peasants are idle with resources, each triggers independently

**Severity:** MAJOR. Degrades performance and creates distracting UI spam. In the worst case (e.g., no valid build site on the entire map), every idle peasant with resources calls the full-map fallback search every 2 seconds indefinitely.

**Fix:** Set a failure cooldown after failed attempts:
```javascript
// After all checks fail:
game.autoBuildCooldown[buildingType] = game.time; // block retries for 15s regardless of outcome
return false;
```
Or use a separate failure counter:
```javascript
// Near the top of autoBuildDropoff:
autoBuildFailures[buildingType] = (autoBuildFailures[buildingType]||0);
if(autoBuildFailures[buildingType] >= 3) return false; // give up after 3 failures
// On success: autoBuildFailures[buildingType] = 0;
```

---

### M5. `findAutoBuildSite` searches near peasant position, not near resources

**File:line:** `index.html:784`

```javascript
const site=findAutoBuildSite(buildingType, peasant.x, peasant.y);
```

**Impact:** The dropoff building is placed near the peasant's current position, not near the resource the peasant was gathering. If the peasant has wandered far from the resource node (e.g., after failing to find a dropoff at one location), the refinery/lumber_mill gets built in a suboptimal location far from the resources it's meant to serve.

This is especially bad in the retry loop case (line 1308), where the peasant may be idle anywhere on the map with preserved resources.

**Severity:** MAJOR but recoverable. The building still functions (deposit range is global), but the peasant must walk farther for subsequent gathering cycles, reducing efficiency.

**Fix:** Prefer the gatheringNode position for the search center:
```javascript
const searchX = peasant.gatheringNode ? peasant.gatheringNode.x : peasant.x;
const searchY = peasant.gatheringNode ? peasant.gatheringNode.y : peasant.y;
const site = findAutoBuildSite(buildingType, searchX, searchY);
```

---

## 🟡 MINOR Issues

### m1. `isPassable` blocks TREE terrain — prevents units from walking near resources

**File:line:** `index.html:152-155`

```javascript
function isPassable(px,py) {
  const t=tileAt(px,py);
  if(t===TERRAIN.WATER||t===TERRAIN.TREE) return false;
```

The TREE terrain is used for wood resources. Since peasants walk to the tree tile to gather, and `isPassable` returns `false` for TREE tiles, the movement code at lines 1195-1203 blocks the peasant from reaching the gathering node. The axis-sliding fallback allows them to get *close*, but they can't actually stand on the tree tile.

This creates a disconnect: the gathering node is on the tree tile, but the peasant can't occupy it. The tile validation at line 1222 would then fail (peasant is on adjacent grass, not TREE), triggering the reset from M2.

**Fix:** Allow TREE in `isPassable` but not WATER, and rely on the gathering destination logic to handle resource tiles separately.

---

### m2. `hasDropoffUnderConstruction` parameter `resType` unused when bldType is null

**File:line:** `index.html:727-728`

```javascript
const bldType=getDropoffBuildingType(resType);
if(!bldType) return false;
```

If a future resource type is added, `getDropoffBuildingType` returns `null`, and the function silently returns `false`. This means no under-construction check would fire for the new resource. A fallback or log would help catch regressions.

---

### m3. `findAutoBuildSite` full-map fallback is O(n²)

**File:line:** `index.html:749-760`

```javascript
// Fallback: search entire map loosely for nearest valid spot
let best=null, bestDist=Infinity;
for(let ty=0;ty<MAP_ROWS;ty++) {
    for(let tx=0;tx<MAP_COLS;tx++) {
      // ...
      if(!isPlacementValid(buildingType, wx, wy)) continue;
```

The full-map fallback iterates all 4096 tiles and calls `isPlacementValid` for each, which in turn iterates all entities (line 845). That's 4096 × N entity iterations — O(n²). On a large entity count, this could cause a noticeable frame drop when the first auto-build fires.

**Fix:** Cache `isPlacementValid` by grid cell or sample at a coarser resolution (e.g., every 4 tiles).

---

### m4. Idle retry timing uses magic number

**File:line:** `index.html:1302`

```javascript
if(Math.floor(game.time*10)%20===0) { // check every ~2 seconds
```

The `20` is opaque — it means "20 game ticks of 0.1s each = 2 seconds." But `game.time` is updated as `game.time+=dt` where `dt` is variable (capped at 0.05s by `cappedDt`). So the actual interval depends on `dt`. If `dt` is 0.05 consistently, the check fires every `20*0.05 = 1.0s`, not 2s. Use a dedicated retry timer variable on the peasant instead.

---

### m5. `isPlacementValid` doesn't account for under-construction buildings in overlap check

**File:line:** `index.html:845`

```javascript
for(const e of game.entities.values()) {
    const eBox={x:e.x-e.w/2,y:e.y-e.h/2,w:e.w,h:e.h};
    if(rectsOverlap(box,eBox)) return false;
}
```

All entities are checked, including other under-construction buildings and dead entities queued for removal (added to `toRemove` array but not yet removed from the map). The `toRemove` issue is minimal (entities are removed at the end of the same frame), but it could cause placement failures in edge cases.

---

## Verdict on the Task

The auto-build dropoff system **works correctly for the happy path**: a peasant gathering gold/wood with no dropoff will trigger `autoBuildDropoff`, which builds a refinery/lumber_mill. The three hook points are correctly wired. Resources are preserved on failure. The cooldown prevents duplicate simultaneous builds.

**However**, the system has five significant gaps:

1. **M1**: A `commandStop` can silently destroy carried resources, undermining the entire preservation strategy
2. **M2**: Entity separation can repeatedly reset gathering, making the pipeline fragile
3. **M3**: Town hall under construction isn't recognized as a pending dropoff → redundant building waste
4. **M4**: Failed auto-build attempts cause continuous retry spam with no backoff
5. **M5**: Dropoff built near peasant, not near resources → suboptimal placement

The core task intent — "if we don't have a dropoff, build it" — is fulfilled. But the edge cases around that intent need attention for the system to be robust.

---

## Recommended Fix Priority

### Immediate
1. **M3** — Add town_hall to `hasDropoffUnderConstruction` (2 lines)
2. **M4** — Set cooldown on auto-build failure (1 line)

### High Priority
3. **M1** — Preserve carryAmount in `commandStop` when no dropoff exists (~5 lines)
4. **M2** — Walk back to gatheringNode instead of resetting timer (~6 lines)
5. **M5** — Search near gatheringNode in `findAutoBuildSite` (~3 lines)

### Low Priority
6. **m1** — Allow TREE in `isPassable` or handle gathering approach differently
7. **m3** — Optimize `findAutoBuildSite` fallback search
8. **m2, m4, m5** — Code cleanup

---

*End of Report*
