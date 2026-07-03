# Correctness Review: RealmCraft RTS Game

**Date:** 2026-07-03  
**Reviewer:** Code Reviewer (dubina-code-reviewer-reason-5d0f46)  
**Artifact:** `index.html` — single-file RTS game (~1,468 lines)  
**Methodology:** Full static analysis of all JS game logic, CSS, and HTML. Scored against the implementation plan (`plan.md`).

---

## Summary

| Severity | Count |
|----------|-------|
| **BLOCKING** | 8 |
| **MAJOR** | 11 |
| **MINOR** | 6 |

**BLOCKING** = game is broken in a core loop (crashes, unreachable state, data loss, fundamental feature non-functional).  
**MAJOR** = feature doesn't work correctly in common scenarios; violates plan requirements.  
**MINOR** = cosmetic, poor UX, dead code, or edge-case-only.

---

## BLOCKING Issues

### B1. Inconsistent entity coordinate system — buildings vs units
**File:line:** `index.html:260-290` (entity creation), `index.html:401-495` (building rendering), `index.html:666-676` (entityAtWorld), `index.html:725-745` (isPlacementValid)

Units store `x,y` as **center position** (evident from `ctx.arc(ex,ey,...)` in `drawSelectionCircle` and `e.x=clamp(e.x,e.size,WORLD_W-e.size)` in update). Buildings store `x,y` as **top-left** (evident from `drawBuilding` translating to `(ex,ey)` and drawing shapes from `(0,0)`; also `placeBuilding` passes `wx-def.size.w/2, wy-def.size.h/2`).

This causes **three concrete failures**:

1. **`isPlacementValid` (line 738-742)**: Computes existing-entity collision box as `eBox={x:e.x-e.w/2, y:e.y-e.h/2, w:e.w, h:e.h}`. For a building whose `e.x` is already top-left, this shifts the collision box left+up by half the building size — completely wrong. Buildings can be placed overlapping each other.

2. **`entityAtWorld` (line 666-676)**: Same `e.x-e.w/2` hit-test error. The right half of every building is unclickable; clicks on the right half register as empty terrain.

3. **Render/click z-order (line ~560)**: Entities are sorted by `a.y - b.y`. Building y (top-left) is compared to unit y (center). A building with top-left y=100 and h=96 has visual center at y=148, but sorts as y=100 — behind units that should render behind it.

**Fix:** Standardize on center positions. In `createBuilding`, compute `centerX = x + w/2, centerY = y + h/2` and store those. Update `drawBuilding`, `isPassable`, `entityAtWorld`, `entitiesInRect`, `isPlacementValid`, `findNearestDropoff`, and all collision/rect code to use center-based coordinates consistently. This is a ~30-line refactor touching ~8 locations.

---

### B2. Mixed peasant+other selection → right-click on resource ignores non-peasants
**File:line:** `index.html:1292-1308`

When a peasant AND a footman (or other non-peasant) are both selected, and the player right-clicks a gold mine or tree:

```javascript
if((tile===TERRAIN.GOLD||tile===TERRAIN.TREE)&&ent===null) {
    let hasPeasant=false;
    // ... detects peasant ...
    if(hasPeasant) {
        // Only iterates selectedIds — but only acts on e.type==='peasant'
        for(const id of game.selectedIds) {
            const e=game.entities.get(id);
            if(e&&e.type==='peasant') issueGather(e, world.x, world.y);
        }
        // NON-PEASANT UNITS GET NO COMMAND — they just stand there
    } else {
        issueMove(world.x, world.y); // only when NO peasants at all
    }
}
```

**Result:** Footmen/archers/knights in a mixed selection are silently ignored when right-clicking a resource node. The player expects them to move to the location (attack-move or just move).

**Fix:** After the peasant gather loop, also issue `issueMove` to all non-peasant selected entities:
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

### B3. Carrying peasant right-clicks non-dropoff building → resources silently lost
**File:line:** `index.html:974-995`, `index.html:1310-1325`

Scenario: Peasant carries 10 gold. Player right-clicks a Barracks (which is NOT a gold dropoff). 

1. Right-click handler (line 1314-1319): Sets `moveTarget` to Barracks center, leaves `carryAmount=10`.
2. Peasant walks to Barracks center. Movement completes → `state='idle'`.
3. Next update tick, the deposit check (line 974-995) fires:
   - `findNearestDropoff(e.x, e.y, 'gold')` finds the nearest Town Hall or Refinery.
   - If the peasant is NOT within 40px of that dropoff → the `else` branch fires: `e.state='idle'; e.carryAmount=0; e.carryType=null; e.gatheringNode=null;`
   - **Carried gold is permanently deleted.**

**Fix:** When the peasant arrives within 40px of the *intended target* (the building they were sent to) but the target isn't a valid dropoff, instead of zeroing the carry, re-route to the nearest actual dropoff:
```javascript
if(dropoff) {
    if(d < 40) { /* deposit */ }
    else { /* re-route: set moveTarget to nearest dropoff */ }
} else {
    // No dropoff exists at all — THEN lose resources (base destroyed)
}
```

---

### B4. Building destruction reduces maxFood but doesn't clamp current food
**File:line:** `index.html:927-932`

When a Town Hall is destroyed: `game.resources.maxFood -= 5`. If the player had 7 food consumed and 2 Town Halls (maxFood=10), after destruction maxFood=5 but food stays at 7 → over cap forever.

**Fix:** Add clamping after maxFood reduction:
```javascript
if(p === 'food5') {
    game.resources.maxFood -= 5;
    game.resources.food = Math.min(game.resources.food, game.resources.maxFood);
}
```

---

### B5. `issueMove` unconditionally clears peasant carry state
**File:line:** `index.html:830-838`

```javascript
function issueMove(targetX, targetY) {
  for(const id of game.selectedIds) {
    const e=game.entities.get(id);
    if(!e||e.owner!=='player') continue;
    if(e.type=='peasant') {
      e.buildTarget=null; e.gatheringNode=null; e.carryType=null; e.carryAmount=0;
    }
    e.state='moving'; e.moveTarget={x:targetX,y:targetY};
  }
}
```

Any move command — even shift-adding a peasant to a group selection and moving the group — destroys all in-progress carrying, gathering, and building state. This makes multi-unit management nearly impossible and violates RTS conventions (in Warcraft/SC, moving a carrying worker does NOT delete their cargo unless explicitly canceled).

**Fix:** Only clear peasant state when explicitly stopped (via `commandStop` or `S` key). In `issueMove`, do NOT zero peasant carry/build/gather state:
```javascript
function issueMove(targetX, targetY) {
  for(const id of game.selectedIds) {
    const e = game.entities.get(id);
    if(!e || e.owner !== 'player') continue;
    if(e.type in BUILDING_DEFS) continue; // buildings can't move
    e.state = 'moving';
    e.moveTarget = {x: targetX, y: targetY};
    // Do NOT clear peasant state here — only clear on explicit Stop
  }
}
```

---

### B6. `isPlacementValid` allows building placement on trees
**File:line:** `index.html:725-745`

The terrain validation only rejects `TERRAIN.WATER`:
```javascript
const t = tileAt(p.x, p.y);
if(t === TERRAIN.WATER) return false;
```

Trees (`TERRAIN.TREE`) pass validation. The plan explicitly states: *"On grass or dirt terrain (not water, not trees — trees can be cleared)"*.

**Fix:** Add `|| t === TERRAIN.TREE` to the rejection condition (or a separate check for tree clearing mechanics if implemented later):
```javascript
if(t === TERRAIN.WATER || t === TERRAIN.TREE) return false;
```

---

### B7. `entitiesInRect` coordinate bug — same center/top-left mismatch as B1
**File:line:** `index.html:678-688`

Identical root cause to B1: `rectsOverlap({x:rx,y:ry,w:rw,h:rh},{x:e.x-e.w/2,y:e.y-e.h/2,w:e.w,h:e.h})` treats all entities as center-positioned, but buildings are top-left. Box-drag selection misses buildings that are partially in the selection rectangle.

**Fix:** Part of the B1 coordinate unification fix.

---

### B8. `drawHealthBar` double-condition logic makes full-HP buildings never show bars when unselected (intentional but wrongly implemented)
**File:line:** `index.html:293-300`

```javascript
function drawHealthBar(ctx, ent, cam) {
  if(ent.hp>=ent.maxHp && ent.owner==='player' && game.selectedIds.has(ent.id)) {} // empty block
  if(ent.hp>=ent.maxHp && !game.selectedIds.has(ent.id)) return;
  // ... draw bar ...
}
```

The first `if` with empty body is a no-op by design — it prevents the second `if` from short-circuiting when the entity is selected. But the second `if` uses `!game.selectedIds.has(ent.id)` without checking `ent.owner==='player'`, meaning a neutral entity at full HP would also hide its bar (which is arguably fine since there are no neutral entities, but the asymmetry with the first condition is a latent bug).

**Fix:** Simplify to a single condition:
```javascript
function drawHealthBar(ctx, ent, cam) {
  const showSelected = game.selectedIds.has(ent.id);
  if(ent.hp >= ent.maxHp && !showSelected) return;
  // ...
}
```

---

## MAJOR Issues

### M1. No food cost deducted for initial starting peasants
**File:line:** `index.html:1420-1426`

Three peasants are spawned with `createUnit('peasant', ...)`. Their food cost (1 each) is pre-loaded into `game.resources.food: 3`. But this means the player never "paid" for them. If a peasant dies, food drops to 2 and the player can train a replacement "for free" (food was refunded without having been spent).

**Fix:** Start `game.resources.food = 0` then manually add food for each starting peasant, or keep `food: 3` and add a comment that this represents pre-consumed food. The current approach is confusing but functionally OK as long as death decrements are correct.

---

### M2. No attack logic implemented for any unit or Watch Tower
**File:line:** `index.html:987-989` (attack timer decrement only), entire file (no attack update)

Attack stats exist on all units and the Watch Tower, the cooldown timer is ticked, but **no code exists to actually perform an attack** — no target acquisition, no damage application, no projectile spawning. The plan lists attack as a core mechanic, and the Watch Tower is described as "Auto-attacks nearby enemies."

While the plan states "No AI or multiplayer needed," having zero attack code means military units are just more expensive peasants with no combat utility — the strategic layer of the game is absent.

**Fix:** Add an attack update block in `updateEntity`:
```javascript
// Attack logic
if(e.attackDamage > 0 && e.attackTimer <= 0 && e.targetId) {
    const target = game.entities.get(e.targetId);
    if(target && dist(e, target) <= e.attackRange) {
        target.hp -= e.attackDamage;
        e.attackTimer = e.attackCooldown;
        spawnParticles(target.x, target.y, 3, '#ff4400');
    } else { e.targetId = null; }
}
```

---

### M3. `isPassable` defined but never called — units walk through buildings and water
**File:line:** `index.html:153-162` (definition), `index.html:976-984` (movement code)

The `isPassable` function exists but the movement code (`e.x+=mx; e.y+=my;`) never consults it. Units walk through walls, buildings, and water. The plan specifies walls should "block path" and water should be "impassable."

**Fix:** Integrate `isPassable` into the movement step with a simple steering adjustment: if the move target is blocked, slide along the obstacle edge.

---

### M4. `findNearestResource` returns tree/gold tiles even when under buildings
**File:line:** `index.html:707-723`

If a building is placed over a tree tile (which B6 allows), `findNearestResource` still returns that tile as a gather node. The peasant walks to it, reaches the building wall, and gets stuck. Even without B6, gold mines might overlap with building footprints at tile boundaries.

**Fix:** In `findNearestResource`, after finding a candidate tile, verify no building overlaps its center:
```javascript
const blocked = [...game.entities.values()].some(e => 
    e.type in BUILDING_DEFS && e.progress >= 1 &&
    rx >= e.x && rx <= e.x + e.w && ry >= e.y && ry <= e.y + e.h
);
if(blocked) continue;
```

---

### M5. Camera edge-scrolling doesn't work at top-left of canvas
**File:line:** `index.html:876-877`

```javascript
if(game.mouse.x>=0&&game.mouse.x<edge) dx=-speed;
if(game.mouse.x>canvas.clientWidth-edge&&game.mouse.x<=canvas.clientWidth) dx=speed;
```

The `game.mouse.x` is relative to the canvas wrapper (`e.clientX - wrapper.getBoundingClientRect().left`). If the mouse is exactly at x=0 (leftmost pixel of the wrapper), it works. But if the mouse leaves the wrapper entirely (goes to the browser tab bar), `mousemove` stops firing and `game.mouse.x` stays at the last known position. Edge-scrolling deadlocks.

Similarly, `game.mouse.y` doesn't fire when the mouse is outside the wrapper.

**Fix:** Use `document.addEventListener('mousemove', ...)` with viewport-relative coordinates, or add padding/handle for `mouseleave`.

---

### M6. `canvas.width = rect.width` — no HiDPI support
**File:line:** `index.html:1239-1240`

```javascript
canvas.width = rect.width;
canvas.height = rect.height;
```

On Retina/HiDPI displays, the canvas renders at 1× resolution, making all graphics and text pixelated and blurry. The game looks significantly worse than expected.

**Fix:**
```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = rect.width * dpr;
canvas.height = rect.height * dpr;
ctx.scale(dpr, dpr);
```
Also update `screenToWorld` and `worldToScreen` to use `canvas.width / canvas.clientWidth` (which now correctly reflects DPR).

---

### M7. Rapid-click placement → ghost building can become stale
**File:line:** `index.html:765-795` (placeBuilding), `index.html:890-900` (ghost update)

When placement mode is active and the player clicks rapidly, `placeBuilding` calls `cancelPlacement()` at the end. But if a mousemove arrives between the build placement and the cancel, `update` re-creates `game.ghostBuilding` from the stale `game.ui.buildingType`. The ghost flickers for one frame.

**Fix:** Check `game.ui.mode` before creating ghost:
```javascript
if(game.ui.mode === 'placeBuilding' && game.ui.buildingType) {
    // ... create ghost only if not just placed
}
```

---

### M8. Minimap viewport rectangle is computed from `canvas.width` instead of CSS size
**File:line:** `index.html:643-646`

```javascript
const canvasW = document.getElementById('game-canvas').width;
const canvasH = document.getElementById('game-canvas').height;
// ...
ctx.strokeRect(cam.x/TILE_SIZE*scale, cam.y/TILE_SIZE*scale,
    (canvasW/cam.zoom)/TILE_SIZE*scale, (canvasH/cam.zoom)/TILE_SIZE*scale);
```

After the HiDPI fix (M6), `canvas.width` would be in physical pixels. The viewport rect on the minimap would appear the wrong size. Currently with 1× it works, but it's fragile.

**Fix:** Use `canvas.clientWidth` and `canvas.clientHeight` (CSS pixels) for the minimap viewport calculation.

---

### M9. `debug-info` span is never updated — suggests incomplete development
**File:line:** `index.html:62`

```html
<span style="font-size:10px;color:#666;margin-left:16px" id="debug-info">debug</span>
```

This element exists in the top bar but is never written to by any JS. It clutters the UI and suggests the developer left debugging hooks in production.

**Fix:** Either remove it or populate it with useful perf data (FPS, entity count).

---

### M10. `game.trainingBuildings` and `game.trainingQueue` initialized but never used
**File:line:** `index.html:135-136`

These arrays are created on the game state object but never populated or read. Training is handled through `building.queue` directly. Dead code.

**Fix:** Remove from game state initialization.

---

### M11. `worldToScreen` function defined but never called
**File:line:** `index.html:658-663`

The function is complete and correct but has zero call sites. Either a missing feature (tooltip positioning? minimap interaction?) or dead code.

**Fix:** Remove or use for tooltip positioning (`#tooltip` element is also in the DOM but never positioned/shown).

---

## MINOR Issues

### m1. Entity separation also separates stationary/constructing buildings
**File:line:** `index.html:998-1015`

The separation loop runs for ALL entities including buildings under construction and idle buildings. Buildings shouldn't be pushed around by nearby units.

**Fix:** Add a guard:
```javascript
if(e.type in BUILDING_DEFS) continue; // buildings don't separate
```

---

### m2. `drawHealthBar` for buildings uses incorrect bar width (`ent.w` which varies)
**File:line:** `index.html:295-300`

```javascript
const bw = ent.w * cam.zoom;  // building width (varies: 32-96 scaled)
```

For buildings, the health bar is as wide as the building, which is excessively long. For units, `ent.w` is `def.size*2` which is reasonable. Inconsistent visual scale.

**Fix:** Cap health bar width or use a fixed width for buildings:
```javascript
const bw = Math.min(ent.w, 64) * cam.zoom;
```

---

### m3. `drawHealthBar` first condition has empty body
**File:line:** `index.html:293`

```javascript
if(ent.hp>=ent.maxHp && ent.owner==='player' && game.selectedIds.has(ent.id)) {}
```

An empty `if` block is confusing. See B8 for the fix.

---

### m4. Particle spawn uses fixed `p.size*z` — too small at high zoom
**File:line:** `index.html:575-579`

Particles are drawn as fixed-size squares scaled by zoom. At zoom 2.0, particles are 2× larger; at zoom 0.5, they're 0.5×. This is arguably correct (world-space particles), but combined with the 2px base size, they're nearly invisible at default zoom.

**Fix:** Increase base particle size from 2-5 to 3-7 or add a minimum pixel size.

---

### m5. `game.time` advances even when tab is backgrounded
**File:line:** `index.html:1437-1442`

```javascript
if(dt<0.3) update(dt);
```

The guard `dt < 0.3` prevents huge time jumps, but `game.time += dt` inside `update` means if the game runs for 5 minutes at 60fps then the player switches tabs for 1 hour and comes back, `dt` is capped to 0.2 but the game timer jumps by years. Not a gameplay bug but the timer display becomes misleading.

**Fix:** Cap `game.time` increment to `cappedDt` (already computed as `Math.min(dt, 0.1)`).

---

### m6. `shiftKey` doesn't add buildings to selection on right-click
**File:line:** `index.html:1263-1275`

The right-click handler doesn't handle Shift+right-click. In Warcraft/SC convention, Shift+right-click adds waypoints. This is missing.

**Fix:** Low priority (no AI enemies anyway), but a waypoint queue would be a nice future addition.

---

## Regression Analysis Against Plan

| Plan Feature | Status | Issue |
|-------------|--------|-------|
| Camera zoom (mouse wheel) | ✅ Implemented | |
| Camera edge-scroll | ⚠️ Partial | M5 — deadlocks at edges |
| Camera middle-click drag | ✅ Implemented | |
| 64×64 tile map with terrain | ✅ Implemented | |
| Procedural water, trees, gold | ✅ Implemented | |
| 7 building types | ✅ All 7 implemented | |
| 4 unit types | ✅ All 4 implemented | |
| Building construction (peasant) | ⚠️ Partial | B1 — placement overlap bug; B6 — trees not blocked |
| Multi-peasant build speedup | ✅ Implemented | |
| Resource gathering (gold/wood) | ⚠️ Partial | B3 — resource loss on wrong dropoff; B2 — mixed selection |
| Training queue + progress | ✅ Implemented | |
| Rally points | ✅ Implemented | |
| Selection (click/drag/shift) | ⚠️ Partial | B7 — drag selection misses buildings |
| Minimap (render + click) | ✅ Implemented | M8 — fragile DPR handling |
| Keyboard shortcuts (S, A, H, ESC) | ✅ Implemented | |
| UI panel (info + actions) | ✅ Implemented | |
| Health bars / selection circles | ⚠️ Partial | B8 — confusing double-condition |
| Death + particles | ✅ Implemented | |
| Floating text on deposit/build | ✅ Implemented | |
| Stop command | ⚠️ Partial | B5 — cleared by move, not just Stop |
| Unit attack | ❌ Not implemented | M2 — zero attack code |
| Watch Tower auto-attack | ❌ Not implemented | M2 |
| Pathfinding / obstacle avoidance | ❌ Not implemented | M3 — isPassable unused |
| Wall blocks path | ❌ Not implemented | M3 — units walk through walls |
| Food system | ⚠️ Partial | B4 — no clamp on building death |
| Responsive layout | ✅ Implemented | M6 — no HiDPI support |

---

## Recommended Fix Order

1. **B1** (coordinate unification) — touches the most code, fix first
2. **B6** (tree placement block) — trivial one-line fix
3. **B2** (mixed selection) — one block rewrite
4. **B3** (resource loss on wrong dropoff) — conditional fix
5. **B4** (food clamp on building death) — one-line fix
6. **B5** (issueMove clears peasant state) — remove clearing lines
7. **B7** (selection rect) — fixed by B1
8. **B8** (health bar logic) — simplification
9. **M3** (isPassable integration) — wire up pathfinding
10. **M2** (attack logic) — new code block
11. Remaining MAJOR and MINOR issues
