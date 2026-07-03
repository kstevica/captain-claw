# Correctness Review — R5: Resource Gather → Carry → Deposit → Auto-Build

**Date:** 2026-07-03  
**Reviewer:** Code Reviewer (Captain Claw, Phrygian mode)  
**Files reviewed:** `index.html` (lines 1–1466), `test_game.js` (lines 1–710), `plan.md`  
**Test suite:** 291 passed, 0 failed

---

## Executive Summary

The auto-build dropoff system is directionally correct — peasants preserve carried resources when no dropoff exists, and the system can create refineries/lumber mills automatically. **However, there is a critical race condition in the update loop that prevents peasants from actually walking to and assisting the auto-built building.** When `autoBuildDropoff()` succeeds in the gathering handler, the deposit handler (running later in the same frame) immediately cancels the move order because it sees no *completed* dropoff (the new building is still under construction) and re-invokes `autoBuildDropoff()`, which returns `false` (under-construction detected), and resets the peasant to `idle`. The building self-completes without builder assistance, wasting ~5 seconds of peasant time.

The normal deposit path (dropoff already exists) is unaffected — the `else if` clause only fires when *no* dropoff exists.

| Severity | Count | Key Issues |
|----------|-------|------------|
| 🔴 BLOCKING | 1 | Race condition: auto-build order cancelled by deposit handler in the same frame |
| 🟠 MAJOR | 3 | Buildings pushed by entity separation; off-by-one in resource search radius; all buildings block movement (ignoring `blocking` flag) |
| 🟡 MINOR | 5 | Limited resource search range; redundant autoBuildDropoff calls; hardcoded seed; orphaned buildTarget; no resource depletion |

---

## 🔴 BLOCKING

### B1. Race condition: `autoBuildDropoff()` move order cancelled by deposit handler in same frame

**File:** `index.html`  
**Lines:** ~1290–1310 (gathering complete) / ~1320–1345 (deposit arrival handler)  
**Root cause:** After `autoBuildDropoff()` succeeds in the gathering block and sets `peasant.state = 'moving'`, the deposit-handler `if` block fires in the same frame, finds no dropoff (new building has `progress < 1` → `findNearestDropoff` skips it), calls `autoBuildDropoff()` again, gets `false` (under-construction detected by `hasDropoffUnderConstruction`), and falls into `else if` → `e.state = 'idle'`.

**Reproduction trace:**
1. Peasant gathers gold at mine, timer hits 1.5 → `carryAmount = 10`
2. `findNearestDropoff()` returns `null` (no refinery/town-hall exists)
3. `autoBuildDropoff(e)` → creates refinery at nearby site, sets `e.state = 'moving'`, `e.buildTarget = refinery.id`, returns `true`
4. **Same frame:** deposit handler fires → `state === 'moving' && carryAmount > 0` → enters block
5. `findNearestDropoff()` → returns `null` (refinery has `progress = 0 < 1`)
6. `autoBuildDropoff(e)` called again → `hasDropoffUnderConstruction()` returns `true` → returns `false`
7. `e.state = 'idle'` ← **overwrites 'moving'!**
8. Peasant stuck idle at gathering site with `carryAmount = 10`
9. Retry loop (every ~2s) also fails: `hasDropoffUnderConstruction` is still true
10. Refinery self-completes in 5s (0 actual builders), peasant finally deposits

**Impact:** Peasant wastes 5+ seconds idle instead of actively building the dropoff. Building self-builds without assistance.

**Fix:** Guard the deposit handler's fallback with a `buildTarget` check:

```javascript
// index.html, deposit arrival handler (~line 1335)
// CHANGE:
} else if (!autoBuildDropoff(e)) {
// TO:
} else if (!e.buildTarget && !autoBuildDropoff(e)) {
```

This prevents the deposit handler from interfering when the peasant already has a build assignment from a prior `autoBuildDropoff()` success.

---

## 🟠 MAJOR

### M1. Entity separation pushes completed buildings

**File:** `index.html`, `updateEntity()` entity separation block (~line 1395)  
**Root cause:** Separation logic filters by `state === 'moving' || state === 'idle' || state === 'attacking'` AND `!e.isBuilding`. But `!e.isBuilding` only checks the boolean flag — and `isBuilding` is set only for building entities defined in `BUILDING_DEFS`. If a unit with `isBuilding: false` but `state: 'idle'` is hit, it won't be pushed — which is correct. However, the earlier "Bug Findings" test (line ~409 in test_game.js) claims buildings *can* be pushed because `isBuilding` might not be checked properly. Re-examining: the separation code includes `!e.isBuilding` in its filter — **buildings ARE excluded from being pushed**. The test assertion `'completed building IS idle (vulnerable to push)'` is misleading; the guard `!e.isBuilding` prevents this.

**Verdict:** This is a **false positive in the test suite**. The guard `!e.isBuilding` correctly prevents building displacement. The test comment is inaccurate.

**However**, there is a related real issue: the `(e.speed || 2)` fallback in older code versions would give buildings speed=2, but the current code uses `dt*3` multiplication without the `speed||2` pattern. The separation formula is:
```javascript
const sepX=e.x+sx*dt*3, sepY=e.y+sy*dt*3;
```
This is unit-agnostic and doesn't use `e.speed`, so buildings wouldn't get special treatment even if they were in the separation loop. **Low severity residual.**

### M2. `findNearestResource` off-by-one in positive search bound

**File:** `index.html`, `findNearestResource()` (~line 806)  
**Lines:**
```javascript
for(let ty=Math.max(0,cy-searchR);ty<Math.min(MAP_ROWS,cy+searchR);ty++)
```
**Root cause:** Upper bound uses `<` (exclusive), meaning the search covers `[cy-searchR, cy+searchR)` — tiles at offset `+searchR` are **excluded**. With `searchR=30`, the search covers offsets -30 to +29 (60 tiles wide), not ±30 (61 tiles). A resource exactly 30 tiles away on the positive axis is missed.

**Fix:** Change `<` to `<=`:
```javascript
for(let ty=Math.max(0,cy-searchR);ty<=Math.min(MAP_ROWS-1,cy+searchR);ty++)
```

### M3. `isPassable` blocks on ALL completed buildings, `blocking` property unused

**File:** `index.html`, `isPassable()` (~line 167)  
**Root cause:** The function checks `if(e.type in BUILDING_DEFS && e.progress>=1)` and blocks passage for ALL buildings. Wall has `blocking:true` but so do non-blocking buildings (refinery, watch_tower, etc.) which all block movement equally.

**Decision needed:** Is this intentional (all buildings are solid in your RTS) or should non-wall buildings (lumber_mill, refinery, watch_tower) be passable? If intentional, remove the redundant `blocking` property from defs. If not, add `if(e.blocking)` to the passability check.

---

## 🟡 MINOR

### N1. `findNearestResource` hardcoded 30-tile radius may miss distant resources

**File:** `index.html`, `findNearestResource()` (~line 799)  
**Line:** `const searchR=30;`  
If a peasant is commanded to gather on a resource tile that's >30 tiles from the nearest actual resource (e.g., clicking on a lone gold tile far from the mine cluster), `findNearestResource` returns `null` and the peasant just walks to the clicked tile without a gathering node. On arrival, `state='moving'` with `gatheringNode=null`, so the gathering block never activates.

**Fix consideration:** Increase `searchR` to match the map size (e.g., 64) or fall back to a full-map search when the ring search returns null.

### N2. Redundant `autoBuildDropoff()` calls in deposit handler

**File:** `index.html`, deposit arrival handler (~line 1340)  
**Issue:** When no dropoff exists, the deposit handler calls `autoBuildDropoff()` **every frame** (60x/sec) for every peasant carrying resources. The cooldown mechanism (`autoBuildCooldown`) prevents actual duplicate builds, but the function body still executes up to the cooldown check each frame. This is wasteful but not harmful.

**Fix:** Consider moving this to a periodic check (like the retry loop's `~2s` interval) instead of every frame.

### N3. Hardcoded map seed (42)

**File:** `index.html`, `generateMap()` (~line 230)  
**Line:** `const rng=simpleRandom(42);`  
Every game produces identical terrain. This limits replayability and makes testing map-edge cases difficult without modifying code.

### N4. Orphaned `buildTarget` references after building destruction

**File:** `index.html`, building death logic (~line 1170)  
**Issue:** When a constructing building is destroyed (HP ≤ 0), any peasant with `buildTarget === building.id` retains the stale reference. The building handler checks `if(!b||b.progress>=1)` — which handles `null` from `game.entities.get()` correctly — but the stale ID lingers on the peasant object. Memory-wise it's just a string, but it's untidy.

**Fix:** Iterate peasants in the death handler and clear `buildTarget` if it matches the destroyed building ID.

### N5. Auto-build doesn't assign the calling peasant's `gatheringNode` to the new building's site proximity

**File:** `index.html`, `autoBuildDropoff()` (~line 760)  
**Issue:** The `findAutoBuildSite` searches near `peasant.gatheringNode || peasant`. After the building completes and the peasant deposits, the peasant returns to `gatheringNode`. This is correct behavior. However, if the `gatheringNode` becomes unreachable (surrounded by water after map changes — not possible currently since maps are static), the peasant would be stuck. **Low severity** with current map generation.

### N6. No resource depletion — gold/wood infinite

**File:** `index.html`  
**Feature gap:** Gold mines and trees are never depleted; each gather cycle yields 10 resources indefinitely. No game-balance issue for a demo, but limits strategic depth (no need to expand to new resource nodes).

### N7. `commandStop` and `issueMove` only check dropoff proximity within 40px before clearing carryAmount

**File:** `index.html`, `commandStop()` (~line 1040), `issueMove()` (~line 860)  
**Behavior:** If a peasant carrying resources is stopped/moved while within 40px of a dropoff, the resources are discarded. If the peasant is far from any dropoff, resources are preserved. This is by design per `plan.md` (qa-r4-03/04), but the asymmetric behavior could surprise players. Consider showing a confirmation or always preserving resources and requiring manual deposit.

---

## Test Suite Observations

**291 tests, 0 failures** — the test suite passes cleanly. However:

1. **Test "issueMove preserves carryAmount"** (line ~838 in test_game.js) uses inverted logic: `const hasDropoff = p.carryType ? null : true;` — this assigns `null` (falsy) when a carry type exists, which coincidentally matches the "no dropoff" case. The test passes but the intermediate logic is misleading.

2. **Test "BUG: Buildings in idle state are affected by unit separation"** (line ~397) asserts vulnerability but the actual code has `!e.isBuilding` guard. The test's assertion is technically true (`buildings have state='idle'`) but the concern about pushing is mitigated. Consider updating the test comment.

3. **No integration tests for the race condition (B1)** — the test suite simulates individual functions but doesn't test the frame-level ordering of `updateEntity` where gathering → deposit fall-through occurs.

---

## Recommendations by Priority

| Priority | Action | Lines affected |
|----------|--------|---------------|
| **P0** | Guard deposit handler fallback with `!e.buildTarget` check (fix B1) | ~1 line change |
| **P1** | Fix `findNearestResource` off-by-one (fix M2) | 2 × `<` → `<=` |
| **P1** | Clarify `isPassable` behavior — either use `blocking` flag or remove it | ~3 line change or doc |
| **P2** | Increase `findNearestResource` search radius or add full-map fallback (N1) | ~5 line change |
| **P2** | Move redundant `autoBuildDropoff` from per-frame to periodic (N2) | ~10 line restructure |
| **P3** | Add integration test for B1 race condition | ~30 lines test |
| **P3** | Clean up orphaned `buildTarget` on building death (N4) | ~5 line change |
| **P3** | Variable map seed for replayability (N3) | 1 line change |
