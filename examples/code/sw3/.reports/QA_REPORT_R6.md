# QA Report: RealmCraft RTS — Round 6 Assessment

**Date:** 2026-07-03 09:14 UTC
**Assessor:** QA Engineer (CAPTAIN CLAW fleet, session `93c795d0`)
**Focus:** Worker resource gathering → carrying → release to town hall/dropoff → auto-build if missing
**Artifact:** `index.html` (~1,698 lines JS) — single-file RTS game
**Test Suite:** `test_game.js` (291 tests, all passing)

---

## Executive Summary

**Overall Quality: GOOD — one CRITICAL bug found in the auto-build → deposit chain.**

The test suite has grown from 226 (R5) to 291 tests, addressing most previously-identified gaps. However, the core task — "when worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else), and if we don't have that, build it" — has a **critical bug** in the auto-build → deposit flow that is masked by how the E2E tests simulate behavior rather than exercising the actual game loop.

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 1 | Auto-build → deposit chain broken by execution order in `updateEntity` |
| 🟡 MEDIUM | 2 | E2E test gap; `entityAtWorld` still untested |
| 🔵 LOW | 2 | `findAutoBuildSite` no mock test; search radius limitation unchanged |

---

## Test Suite Results

```
$ node test_game.js
RESULTS: 291 passed, 0 failed
```

**All 291 tests pass.** JavaScript syntax check: **OK** (`node --check` passes cleanly).

Test growth from R5 (226) to R6 (291): **+65 new tests** covering previously-identified gaps.

---

## 🔴 Finding R6-001 (CRITICAL): Auto-build → deposit chain broken by execution order

**Severity:** CRITICAL
**Location:** `index.html` — `updateEntity()` function, lines 1110–1253

### Root Cause

The execution order in `updateEntity` prevents the deposit handler from ever firing for a peasant who auto-built a dropoff and is standing at the completed building:

```
Step 8  (line 1110): Unit movement    — if d<3, resets state='idle'
Step 9  (line 1146): Peasant building — skipped (state is 'idle')
Step 10 (line 1154): Peasant gathering — skipped (state is 'idle')
Step 11 (line 1191): Deposit handler  — requires state==='moving' → MISSES
Step 13 (line 1243): Idle retry       — fires every ~2s, sets state='moving'
```

### Full Trace

1. Peasant auto-builds a refinery/lumber mill via `autoBuildDropoff`:
   - `peasant.moveTarget = {x: site.x, y: site.y}` — peasant walks to build site
   - `peasant.buildTarget = b.id`

2. Peasant arrives at site (`d < 3`): movement code sets `state='idle'`, then `buildTarget` check triggers `state='building'`

3. Building completes (lines 1026–1030):
   ```javascript
   for (const u of game.entities.values()) {
     if (u.type==='peasant' && u.buildTarget===e.id) {
       u.buildTarget = null;
       u.state = 'idle';  // ← peasant freed, still carrying resources
     }
   }
   ```

4. Peasant is now **at `d ≈ 0`** from the completed dropoff, `state='idle'`, `carryAmount=10`, `carryType='gold'`

5. **Idle retry block** fires every ~2 seconds (line 1243):
   ```javascript
   const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
   if (dropoff) {
     e.state = 'moving';
     e.moveTarget = { x: dropoff.x, y: dropoff.y };
   }
   ```

6. **Next frame:**
   - **Step 8 (movement):** `d < 3` (peasant is at dropoff center) → `e.state='idle'`, `e.moveTarget=null`
   - Step 9–10: skipped (state is 'idle')
   - **Step 11 (deposit handler, line 1191):** `e.state==='moving'` → **FALSE** — state was reset to 'idle' at step 8. **Deposit NEVER fires.**
   - Step 13 (idle retry): timer-based, fires ~2s later

7. **Infinite loop:** Steps 5–6 repeat every ~2 seconds. The peasant oscillates between `idle` (carrying resources) and `moving` (one frame, immediately reset). Resources are **never deposited**.

### Why the Normal (non-auto-build) Path Works

For a peasant approaching a dropoff from a gathering node, the deposit handler catches them at `d < 40` **while still moving** (before `d < 3` triggers the movement reset). The 37px window (40 − 3) gives ~2.6 frames at peasant speed, which is sufficient.

The auto-build case is unique because the peasant is **already at `d ≈ 0`** when the idle retry sets `state='moving'`. There is no approach phase — the movement code instantly resets state.

### Why the E2E Test Missed This

The test "full chain with auto-build" at line ~1820 of `test_game.js` **manually simulates the deposit**:

```javascript
// Step 3: Deposit fires (peasant idle + carrying + at completed dropoff)
const d = Math.sqrt((p.x - b.x) ** 2 + (p.y - b.y) ** 2);
assert(d < 40, 'within deposit range of completed building');
resources[p.carryType] += p.carryAmount;  // ← MANUAL deposit, not game code
p.carryAmount = 0;
```

The test asserts the **intended behavior** but never exercises `updateEntity` with the actual state sequence. It passes because it bypasses the real code path.

### Recommended Fix

**Option A (minimal):** Have the idle retry block directly handle deposit when peasant is already at the dropoff:

```javascript
// In idle retry block (line 1243):
if (dropoff) {
  const d = dist({x: e.x, y: e.y}, {x: dropoff.x, y: dropoff.y});
  if (d < 40) {
    // Deposit immediately — don't delegate to next-frame handler
    game.resources[e.carryType] += e.carryAmount;
    e.carryAmount = 0;
    if (e.gatheringNode) {
      e.state = 'moving';
      e.moveTarget = { x: e.gatheringNode.x, y: e.gatheringNode.y };
    }
  } else {
    e.state = 'moving';
    e.moveTarget = { x: dropoff.x, y: dropoff.y };
  }
}
```

**Option B (structural):** Move the deposit handler (lines 1190–1216) to **before** the movement code (line 1110). This would fix both the auto-build case and any edge case where a fast unit could jump past the deposit window. However, this is a larger refactor and needs more testing.

---

## Task-Specific Assessment

### "When worker gets gold or wood, and carry it, check if this resource is released back to town hall (or anywhere else)"

**VERDICT: PASS (with caveat)** — The deposit mechanism works correctly for the normal gather→carry→deposit cycle. Resources are released via `game.resources[e.carryType] += e.carryAmount` at line 1199. The `d < 40` window catches approaching peasants before the movement code resets state.

**Caveat:** The auto-build → deposit path is broken (R6-001 above).

### "And if we don't have that, build it"

**VERDICT: PASS** — `autoBuildDropoff()` correctly triggers when no appropriate dropoff exists. The function checks all preconditions (existing dropoff, under construction, cooldown, affordability, valid site) and creates the building with peasant assignment.

**Caveat:** After the building completes, the peasant cannot deposit at it (R6-001).

---

## Changes Since R5

### R5 Gaps Addressed

| R5 Finding | Status | New Tests |
|------------|--------|-----------|
| R5-001: findNearestDropoff type discrimination untested | ✅ FIXED | 12 tests — gold→refinery, wood→lumbermill, cross-type guard, universal townhall, no-dropoff null |
| R5-002: placeBuilding untested | ✅ FIXED | 4 tests — resource deduction, insufficient resources, peasant assignment, soft-lock prevention |
| R5-003: issueGather tile dispatch untested | ✅ FIXED | 14 tests — gold dispatch, wood dispatch, non-resource move, null-resource fallback, non-gatherer guard |
| R5-004: findNearestResource search radius | ⚠️ PARTIAL | 11 tests — within radius, edge-of-search, no-match null, nearest-of-multiple, map boundaries |
| R5-005: cross-deposit guard | ✅ FIXED | Covered by R5-001 type discrimination tests |
| R5-006: findNearestResource null-return silent failure | ✅ FIXED | Test verifies null return path |

### Gaps Still Open

| R5 Finding | Status |
|------------|--------|
| R5-007: entityAtWorld hit-testing | Still untested |
| R5-008: findAutoBuildSite full-map fallback | No mock-based test |
| R5-009: world-edge entity separation artifacts | Cosmetic, unchanged |

---

## Findings Summary

### 🔴 R6-001 (CRITICAL): Auto-build → deposit chain broken
**See detailed analysis above.** After auto-building a dropoff, the peasant cannot deposit resources because the execution order in `updateEntity` causes the movement code to reset `state='idle'` before the deposit handler (which requires `state='moving'`) can fire.

**Impact:** Resources gathered by a peasant who auto-built a dropoff are permanently stuck on the peasant. The peasant oscillates between idle and moving every ~2 seconds without ever depositing.

**Fix priority:** HIGHEST — this is the core task path.

### 🟡 R6-002 (MEDIUM): E2E test bypasses actual game loop

**Location:** `test_game.js`, "full chain with auto-build" test (~line 1820)

**Details:** The E2E test manually deposits resources instead of running through `updateEntity`. This means the test passes even though the actual game code has the R6-001 bug. The test asserts the **desired behavior** rather than verifying the **actual implementation**.

**Recommended:** Add a test that calls the real `updateEntity` (or a close simulation with correct execution ordering) on a peasant who just completed auto-building a dropoff, and verifies that `game.resources` increases and `peasant.carryAmount` becomes 0.

### 🟡 R6-003 (MEDIUM): `entityAtWorld` hit-testing still untested

**Location:** `index.html` — `entityAtWorld` function

**Details:** Carried over from R5-007. The function's bounding-box hit test (unit vs building rectangles, y-sort for z-order) has zero test coverage. Clicking near the corner of a unit's bounding box could produce unexpected selection behavior.

### 🔵 R6-004 (LOW): `findAutoBuildSite` has no mock-based unit test

**Location:** `index.html` — `findAutoBuildSite` (~25 lines)

**Details:** The ring-search and full-map fallback logic is untested in isolation. A mock-based test could verify:
- Ring perimeter search finds valid sites at each radius
- Full-map fallback is triggered when no perimeter site is valid
- Water/tree tiles are correctly skipped
- `null` return when the entire map is occupied

### 🔵 R6-005 (LOW): `findNearestResource` 30-tile search radius limitation

**Location:** `index.html` — `findNearestResource`, `searchR=30`

**Details:** On a 64×64 tile map, a 30-tile search covers ~44% of the map area (π×30²/64²). Resources beyond 30 tiles are invisible. The `findAutoBuildSite` function has a full-map fallback, but `findNearestResource` does not. This is a design limitation, not a bug.

---

## Test Coverage Gap Map (Updated R6)

```
Function                    Tested?   Notes
────────────────────────────────────────────────────────────
createUnit                  ✅        40 tests
createBuilding              ✅        8 tests
isPlacementValid            ✅        10 tests
startTraining               ✅        8 tests
updateEntity (partial)      ✅        13 tests
screenToWorld               ✅        4 tests
findNearestDropoff          ✅        12 tests (NEW)
findNearestResource         ✅        11 tests (NEW)
placeBuilding               ✅        4 tests (NEW)
issueGather                 ✅        14 tests (NEW)
autoBuildDropoff (chain)    ⚠️        3 tests (chain sim, not real updateEntity)
hasDropoffUnderConstruction ⚠️        1 test
── GAP BELOW ──
entityAtWorld               ❌        0 tests
findAutoBuildSite           ❌        0 tests (standalone)
getDropoffBuildingType      ❌        0 tests (trivial, 2-line function)
render / draw functions     ❌        ~250 lines (not testable in Node)
```

---

## Recommended Actions (Priority Order)

1. **FIX R6-001:** Add inline deposit logic to the idle retry block (or restructure execution order) so peasants who auto-built a dropoff can deposit at it
2. **ADD TEST for R6-001:** Test that a peasant standing at a completed dropoff with carryAmount>0 actually deposits resources through a realistic simulation of `updateEntity`
3. Add `entityAtWorld` tests — unit hit, building hit, corner miss, z-order priority
4. Add `findAutoBuildSite` mock tests — ring search, full-map fallback, null path
5. (Optional) Expand `findNearestResource` search radius or add full-map fallback for large maps

---

## Overall Verdict

**291/291 tests pass. The core gather→carry→deposit cycle works correctly for the normal (non-auto-build) path.** However, the auto-build → deposit chain has a critical execution-order bug (R6-001) that prevents peasants from ever depositing resources at a dropoff they auto-built. This bug is masked by the E2E test which simulates the expected behavior rather than exercising the actual game loop. The fix is straightforward (add deposit logic to the idle retry block) and should be applied before any release.
