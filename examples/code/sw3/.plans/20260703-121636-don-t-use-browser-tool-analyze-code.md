# Implementation Plan: Fix Critical Bugs in RealmCraft RTS

**Date:** 2026-07-03  
**Task:** Fix blocking and major bugs identified in correctness review R6  
**Source analysis:** read + glob + codemap only (no browser tool used)  
**Files affected:** `index.html` (~10 locations, ~30 lines changed)

---

## Context

The R6 correctness review identified 1 BLOCKING, 3 MAJOR, and 3 MINOR bugs in the gather→carry→deposit→auto-build chain. This plan fixes all 7 issues in priority order. All changes target `index.html` only.

---

## Step 1 — [BLOCKING] Fix Deposit State-Ordering Bug (B1)

**Problem:** Peasants arrive at a dropoff, overshoot to `d < 3`, get stuck in an idle↔moving loop, and never deposit carried resources. The deposit handler guards on `e.state === 'moving'` but the movement handler sets `state = 'idle'` before deposit can fire.

**Files/lines:** `index.html`, deposit handler (~L1277) and idle retry loop (~L1330)

### 1a: Broaden deposit guard to also accept `idle`

Find the deposit arrival check that looks like:
```javascript
if (e.state === 'moving' && e.carryAmount > 0) {
```

Change to:
```javascript
if ((e.state === 'moving' || e.state === 'idle') && e.carryAmount > 0) {
```

### 1b: In idle retry, deposit directly when at dropoff

Find the idle retry loop (~L1330) where it sets `e.state = 'moving'` / `e.moveTarget = ...`. Before setting move state, check if the peasant is already within deposit range:

```javascript
// BEFORE the existing move-target assignment, ADD:
if (dropoff) {
  const d = dist({x:e.x, y:e.y}, {x:dropoff.x, y:dropoff.y});
  if (d < 40) {
    // Already at dropoff — deposit directly
    game.resources[e.carryType] += e.carryAmount;
    showFeedback(e.x, e.y, '+' + e.carryAmount + ' ' + e.carryType, '#ffd700');
    e.carryAmount = 0;
    e.carryType = null;
    // Return to gathering if we have a gatheringNode
    if (e.gatheringNode) {
      e.state = 'moving';
      e.moveTarget = {x: e.gatheringNode.x, y: e.gatheringNode.y};
    }
    return; // skip the rest of idle retry
  }
  // Otherwise proceed with move-to-dropoff as before
  e.state = 'moving';
  e.moveTarget = {x: dropoff.x, y: dropoff.y};
}
```

**Verification:** After fix, `SCENARIO G` and `SCENARIO I` from the test suite should still pass. Peasants at a dropoff should deposit within 1 frame, not get stuck in idle↔moving loops.

---

## Step 2 — [MAJOR] Fix Auto-Build Race Condition (M2)

**Problem:** Two peasants can call `autoBuildDropoff` in the same frame, both pass the cooldown guard, and both deduct resources and create buildings.

**Files/lines:** `index.html`, `autoBuildDropoff` function (~L772–802), game init, and main update loop

### 2a: Add per-frame lock set to game state

In the game state object (around line ~137), near `autoBuildCooldown`, add:
```javascript
_autoBuildThisFrame: new Set(),
```

### 2b: Guard autoBuildDropoff with per-frame lock

In `autoBuildDropoff`, after the affordability check (after resources are verified sufficient but BEFORE deduction), add:

```javascript
// Prevent two peasants from auto-building the same type in one frame
if (game._autoBuildThisFrame.has(buildingType)) return false;
game._autoBuildThisFrame.add(buildingType);
```

### 2c: Clear the lock each frame

In the main `update(dt)` function, at the start (before the entity loop, around ~L1061), add:
```javascript
game._autoBuildThisFrame.clear();
```

### 2d: Also clear in initGame

In `initGame()` (around ~L1745), initialize alongside other state:
```javascript
game._autoBuildThisFrame = new Set();
```

**Verification:** `SCENARIO F` (two peasants both auto-build) should still pass — but now only one building per type can be created per frame, preventing double-deduction.

---

## Step 3 — [MAJOR] Preserve Gathering Progress on Push-Back (M1)

**Problem:** When entity separation pushes a gathering peasant off the resource tile, the movement handler at arrival resets `gatheringTimer = 0`, discarding up to 1.5s of gathering progress.

**Files/lines:** `index.html`, movement handler arrival block (~L1190–1215)

### 3a: Don't reset gatheringTimer on arrival

Find the block where peasant arrives at a gathering node:
```javascript
if (e.type === 'peasant' && e.gatheringNode && e.carryAmount === 0) {
    e.state = 'gathering';
    e.gatheringTimer = 0;  // ← REMOVE THIS LINE
}
```

Change to:
```javascript
if (e.type === 'peasant' && e.gatheringNode && e.carryAmount === 0) {
    e.state = 'gathering';
    // gatheringTimer preserved — don't reset to 0
}
```

**Verification:** After fix, a peasant pushed off a resource tile at `gatheringTimer = 1.4` should resume at `1.4` after walking back, not restart from `0`.

---

## Step 4 — [MAJOR] Add Distance Check to hasDropoffUnderConstruction (M3)

**Problem:** `hasDropoffUnderConstruction` returns true for ANY town hall under construction anywhere on the map (even opposite corner), blocking auto-build for far-away peasants.

**Files/lines:** `index.html`, `hasDropoffUnderConstruction` function (~L727–738)

### 4a: Add proximity parameter and distance check

Change the function signature from taking `(resType)` to `(resType, nearX, nearY)`:

```javascript
function hasDropoffUnderConstruction(resType, nearX, nearY) {
    const MAX_DIST = 400; // pixels (~12.5 tiles)
    const bldType = getDropoffBuildingType(resType);
    for (const e of game.entities.values()) {
        if (e.owner === 'player' && e.progress < 1) {
            if (e.type === bldType) {
                if (dist({x:nearX, y:nearY}, {x:e.x, y:e.y}) < MAX_DIST) return true;
            }
            if (e.type === 'town_hall') {
                if (dist({x:nearX, y:nearY}, {x:e.x, y:e.y}) < MAX_DIST) return true;
            }
        }
    }
    return false;
}
```

### 4b: Update call sites

Find every call to `hasDropoffUnderConstruction(resType)` and add the `nearX, nearY` parameters. The main call site is in `autoBuildDropoff` (~L772). Change:

```javascript
// OLD:
if (hasDropoffUnderConstruction(e.carryType)) return false;

// NEW:
if (hasDropoffUnderConstruction(e.carryType, e.x, e.y)) return false;
```

Search for any other call sites and update them similarly.

**Verification:** A peasant near a gold mine on the left side of the map should NOT be blocked by a town hall being constructed on the far right side.

---

## Step 5 — [MINOR] Fix findNearestResource Off-by-One (N1)

**Problem:** Search loop uses `<` (exclusive upper bound) so `searchR=30` is effectively 29 on the right/bottom edges.

**Files/lines:** `index.html`, `findNearestResource` function (~L805)

### 5a: Change `<` to `<=`

Find:
```javascript
for (let ty = Math.max(0, cy - searchR); ty < Math.min(MAP_ROWS, cy + searchR); ty++) {
```

Change to:
```javascript
for (let ty = Math.max(0, cy - searchR); ty <= Math.min(MAP_ROWS - 1, cy + searchR); ty++) {
```

Also fix the tx loop similarly:
```javascript
for (let tx = Math.max(0, cx - searchR); tx < Math.min(MAP_COLS, cx + searchR); tx++) {
```
→
```javascript
for (let tx = Math.max(0, cx - searchR); tx <= Math.min(MAP_COLS - 1, cx + searchR); tx++) {
```

**Verification:** Resources at exactly 30 tiles on right/bottom edges should now be found.

---

## Step 6 — [MINOR] Add Bailout Distance to findAutoBuildSite Fallback (N2)

**Problem:** When ring search fails, the O(n²) fallback iterates all 4,096 tiles, each doing entity overlap checks — potential frame drop with many entities.

**Files/lines:** `index.html`, `findAutoBuildSite` function (~L744–756)

### 6a: Cap fallback search radius

Find the full-map fallback loop in `findAutoBuildSite`. Add a MAX_FALLBACK distance cap:

```javascript
// In the fallback loop, add:
const MAX_FALLBACK_TILES = 30;
// constrain cy, cx search area to nearX/nearY ± MAX_FALLBACK_TILES
const fy1 = Math.max(0, Math.floor(nearY / TILE_SIZE) - MAX_FALLBACK_TILES);
const fy2 = Math.min(MAP_ROWS - 1, Math.floor(nearY / TILE_SIZE) + MAX_FALLBACK_TILES);
const fx1 = Math.max(0, Math.floor(nearX / TILE_SIZE) - MAX_FALLBACK_TILES);
const fx2 = Math.min(MAP_COLS - 1, Math.floor(nearX / TILE_SIZE) + MAX_FALLBACK_TILES);
```

Apply these bounds to the fallback search loops.

**Verification:** Frame time during auto-build with many entities should remain stable.

---

## Step 7 — [MINOR] Simplify Auto-Build Cooldown Logic (N3)

**Problem:** Affordability failure sets `cooldown = game.time + 3` but the check uses `< 15`, making effective cooldown 18s. Comment says "short cooldown" but it isn't.

**Files/lines:** `index.html`, `autoBuildDropoff` affordability failure path (~L787)

### 7a: Use a dedicated short cooldown constant

At top of file near other constants, add:
```javascript
const AUTO_BUILD_SHORT_COOLDOWN = 3; // seconds — for resource-failure retry
```

### 7b: Change cooldown check to use variable

In `autoBuildDropoff`, find the cooldown check:
```javascript
if (game.autoBuildCooldown[buildingType] && (game.time - game.autoBuildCooldown[buildingType]) < 15) return false;
```

The success-path cooldown stays at 15s. On the affordability failure path, the cooldown is set AND the check needs to differentiate. Simplest fix: on affordability failure, set cooldown to a value 3s in the future but DON'T change the check — just fix the comment.

OR, better: use two separate cooldown durations:

```javascript
// After successful auto-build:
game.autoBuildCooldown[buildingType] = game.time + 15; // 15s cooldown

// On affordability failure (can't afford):
game.autoBuildCooldown[buildingType] = game.time + AUTO_BUILD_SHORT_COOLDOWN; // 3s cooldown

// Check (earlier in function):
const cooldownRemaining = game.autoBuildCooldown[buildingType] 
    ? game.time < game.autoBuildCooldown[buildingType] 
    : false;
if (cooldownRemaining) return false;
```

This makes the cooldown work as: 15s after a successful build, 3s after an affordability failure. The check becomes a simple "is cooldown still active?" rather than "is the difference less than 15?"

**Verification:** After an affordability failure, auto-build should retry after ~3s (not ~18s). After a successful build, cooldown is still 15s.

---

## Execution Order

1. **Step 1 (B1)** first — this is the blocking bug that makes deposits unreliable
2. **Step 2 (M2)** next — race condition that can waste resources
3. **Steps 3–4 (M1, M3)** — gameplay quality improvements
4. **Steps 5–7 (N1–N3)** — polish

## Verification

After all changes:
- Run `node --check index.html` (extract JS block) to verify syntax
- Run `node test_game.js` — all 341+ tests should still pass
- Manual test: gather gold with no refinery → auto-build → deposit → verify resources increase
- Manual test: gather gold with refinery already built → walk to refinery → deposit → verify no stuck peasant
- Manual test: two peasants gather gold simultaneously when no refinery exists → only one refinery built, one deposit of gold

## Files Modified

| File | Changes | Lines affected |
|------|---------|----------------|
| `index.html` | 7 bug fixes | ~30 lines total |

No other files modified.
