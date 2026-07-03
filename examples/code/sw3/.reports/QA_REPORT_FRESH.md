# QA Report: RealmCraft RTS — Fresh Assessment

**Date:** 2026-07-03  
**Assessor:** QA Engineer (CAPTAIN CLAW fleet, this session)  
**Artifact:** `index.html` (1,468 lines, 40 functions) — single-file RTS game  
**Test Suite:** `test_game.js` (710 lines, 17 test helpers)  
**Methodology:** Static code audit + live browser verification + test suite execution + coverage gap analysis

---

## Executive Summary

**Overall Quality: GOOD** — but with significant testing gaps in the most critical code path.

The game loads and runs correctly in a headless browser (verified via screenshot + accessibility tree). The game loop is active (timer ticks), resources render, and the UI is responsive. The existing 123-test Node.js suite passes 100%. However, **the single most complex and critical function — `updateEntity` (~175 lines) — has ZERO dedicated tests**, and several core user commands (`issueMove`, `issueGather`, `commandStop`) are also untested.

| Severity | Count | Finding |
|----------|-------|---------|
| 🔴 CRITICAL | 2 | `updateEntity` fully untested (175 lines of game logic); core commands untested |
| 🟠 MEDIUM | 2 | `screenToWorld` untested (all mouse I/O); building placement state mgmt untested |
| 🟡 LOW | 3 | Dead code (`worldToScreen`); `getBuildTime` not in test blocks; no integration test |
| 🔵 INFO | 2 | `lerp` trivial but untested; `cancelPlacement` has missing validation edge case |

---

## 1. Test Suite Results

```
═══════════════════════════════════════
RESULTS: 123 passed, 0 failed
═══════════════════════════════════════
```

**Coverage by test suite section (existing):**

| Module | Tests | Status |
|--------|-------|--------|
| Entity Creation | 40 | ✅ All pass |
| Resource System | 16 | ✅ All pass |
| Building Placement | 8 | ✅ All pass |
| Helper Functions | 8 | ✅ All pass |
| Building Properties | 12 | ✅ All pass |
| Unit Properties | 13 | ✅ All pass |
| Game State | 7 | ✅ All pass |
| Edge Cases | 10 | ✅ All pass |
| Bug Findings (code review) | 9 | ✅ All verified |

**What IS covered well:**
- Entity factory functions (`createUnit`, `createBuilding`) — all types
- Resource cost validation and deduction (`startTraining`)
- Food cap enforcement
- Building placement validation (bounds, terrain, overlap)
- Entity hit-testing (`entityAtWorld`, `entitiesInRect`)
- `isPassable()` with terrain and entity checks
- `findNearestDropoff`, `findNearestResource`
- Game initialization state
- Math helpers (`dist`, `clamp`, `rectsOverlap`, `tileAt`)

---

## 2. Live Browser Verification

| Check | Result |
|-------|--------|
| Page loads without errors | ✅ HTTP 200, title "RealmCraft — RTS" |
| Top bar renders (Gold, Wood, Food) | ✅ `🔵 200 Gold, 🪵 150 Wood, 🍖 3/5` |
| Game timer ticks | ✅ `00:09` observed (incrementing) |
| Bottom panel renders | ✅ "Select a unit or building to begin" |
| Instructions visible | ✅ "Right-click to move · Drag to box-select · Edge-scroll to pan · Scroll to zoom" |
| Canvas present and sized | ✅ Full viewport coverage |
| No console errors | ✅ Clean load |

---

## 3. Critical Findings (NEW — not in prior report)

### 🔴 CRITICAL-1: `updateEntity` (175 lines) — ZERO test coverage

**Severity:** CRITICAL  
**Location:** `index.html` lines 921–1096  

This is the single most important function in the game. It handles ALL entity behavior per frame:

- **Death handling** (HP ≤ 0): particle effects, food cap recalculation on building death, unit food tracking removal
- **Building construction** (progress < 1): builder counting, progress per frame, completion spawning, peasant release
- **Training queue** (queue): timer progression, unit spawning at rally point, queue advancement
- **Attack cooldown** (attackTimer > 0): timer decrement
- **Unit movement** (state === 'moving'): vector movement, arrival detection, state transitions to building/gathering
- **Peasant building** (state === 'building'): construction validation, particle effects
- **Peasant gathering** (state === 'gathering'): timer, carry amount 10, drop-off routing, fallback to idle
- **Drop-off deposit** (carryAmount > 0): arrival detection, resource deposit, return-to-node routing
- **Entity separation**: push-away logic for overlapping entities

The existing tests verify individual atomic operations (`createUnit`, `startTraining`) but **never exercise the update loop** that ties them together. This means the entire game flow — gather → carry → deposit → return → gather — is completely untested programmatically.

**Risk:** A single-offset error in any of these ~175 lines breaks core gameplay. The only safety net is the prior manual/visual testing.

### 🔴 CRITICAL-2: Core user commands untested

**Severity:** CRITICAL  
**Functions:** `issueMove` (line 831), `issueGather` (line 842), `commandStop` (line 1223)

These are the primary ways a player interacts with the game. None have test coverage:

- `issueMove(targetX, targetY)` — iterates selected units, sets moveTarget, resets peasant state
- `issueGather(entity, wx, wy)` — validates resource tile, finds nearest resource node, configures gathering state
- `commandStop()` — halts all selected entities

**Key untested behaviors in `issueGather`:**
- Guards (`canGather` check)
- Resource tile validation (`TERRAIN.GOLD`, `TERRAIN.TREE`)
- Node search via `findNearestResource`
- State transitions (moving → gathering → carrying → deposit)
- Edge case: what happens when `findNearestResource` returns null? The code silently does nothing — should it fall through to a regular move?

**Key untested behavior in `issueMove`:** Does NOT validate bounds — `targetX`/`targetY` can be out of map bounds. While `updateEntity` clamps position later, the entity could waste frames moving toward an unreachable target.

---

## 4. Medium Findings

### 🟠 MEDIUM-1: `screenToWorld` untested — all mouse I/O depends on it

**Severity:** MEDIUM  
**Function:** `screenToWorld` (line 651)

This function converts pixel-space mouse coordinates to world-space coordinates. It is called on **every mouse event** (click, right-click, drag, wheel). Its correctness is critical for:

- Click-to-select units
- Right-click-to-move/gather
- Box-selection
- Zoom-toward-cursor
- Building placement ghost preview

The formula `sx*(canvas.width/canvas.clientWidth)/cam.zoom+cam.x` accounts for HiDPI canvas scaling and zoom. A unit test verifying the round-trip (`screenToWorld` → `worldToScreen` back to original coordinates) would catch scaling errors.

### 🟠 MEDIUM-2: Building placement state management untested

**Severity:** MEDIUM  
**Functions:** `startBuildingPlacement` (line 750), `cancelPlacement` (line 757)

The building placement flow (click build button → ghost preview → click-to-place → confirm) is a multi-step state machine with no tests:

- Mode transitions (`ui.mode === 'placeBuilding'`)
- Cursor class changes on body
- Escape key to cancel
- Right-click to cancel
- Ghost preview rendering (visual, harder to test)
- Validation on each mousemove during placement

**Edge case found during code review:** `cancelPlacement` resets `ui.mode` and cursor class, but does NOT check if there's an incomplete sequence that should be cleaned up. For example, if `ui.buildingType` is an object reference, it's left dangling.

---

## 5. Low Findings

### 🟡 LOW-1: Dead code — `worldToScreen` never called

**Severity:** LOW  
**Function:** `worldToScreen` (line 658)

Defined but never invoked anywhere in the codebase. This is 11 lines of dead code. It's the inverse of `screenToWorld` and is syntactically correct, but its existence is misleading during maintenance. Either wire it up (it would be useful for unit-tracking UI) or remove it.

### 🟡 LOW-2: `getBuildTime` in test file but not exercised

**Severity:** LOW  

`getBuildTime` is extracted/defined in `test_game.js` (line 212) but never appears inside any `it()` test block. The function is trivial (`(BUILDING_DEFS[type]||{}).buildTime||5`), but the fallback-to-5 behavior and unknown-type handling should be verified.

### 🟡 LOW-3: No integration / end-to-end test

**Severity:** LOW  

All 123 tests are unit tests of extracted game logic. There is no test that:
- Creates a game, spawns entities, runs 60 frames, and asserts final state
- Simulates a peasant gathering cycle (move to gold → gather → carry → deposit)
- Simulates building a farm from start to completion

The infrastructure to do this exists (the test file already mocks canvas/DOM), but no integration tests were written.

### 🟡 LOW-4: `cancelPlacement` missing edge case

**Severity:** LOW  
**Location:** `cancelPlacement()` line 757

If `cancelPlacement()` is called when `ui.mode` is NOT `'placeBuilding'`, it still resets the body class to `'cursor-normal'`. This could clobber the cursor state from other modes (e.g., attack mode set by pressing 'A'). The function should guard: `if (game.ui.mode !== 'placeBuilding') return;`

---

## 6. Functions with Zero Test Coverage (Complete List)

| Function | Lines | Risk | Reason |
|----------|-------|------|--------|
| `updateEntity` | ~175 | 🔴 CRITICAL | All game behavior per frame |
| `issueMove` | 11 | 🔴 HIGH | Primary player command |
| `issueGather` | 20 | 🔴 HIGH | Resource gathering command |
| `commandStop` | 13 | 🟠 MEDIUM | Stop command |
| `screenToWorld` | 6 | 🟠 MEDIUM | All mouse I/O |
| `worldToScreen` | 6 | 🟡 LOW | Dead code (never called) |
| `startBuildingPlacement` | 7 | 🟠 MEDIUM | UI state machine |
| `cancelPlacement` | 8 | 🟠 MEDIUM | UI state machine |
| `lerp` | 1 | 🟡 LOW | Trivial math |
| `spawnParticles` | 5 | 🔵 INFO | Visual only |
| `spawnFloatingText` | 5 | 🔵 INFO | Visual only |
| `drawHealthBar` | 12 | 🔵 INFO | Canvas rendering |
| `drawSelectionCircle` | 8 | 🔵 INFO | Canvas rendering |
| `drawUnit` | 86 | 🔵 INFO | Canvas rendering |
| `drawBuilding` | 103 | 🔵 INFO | Canvas rendering |
| `render` | 118 | 🔵 INFO | Canvas rendering |
| `renderMinimap` | 30 | 🔵 INFO | Canvas rendering |
| `updateUI` | ~100 | 🔵 INFO | DOM manipulation |
| `resizeCanvas` | ~15 | 🔵 INFO | DOM/layout |

**Summary:** 19 of 40 functions (47.5%) are untested, but most are rendering/DOM functions where testing is impractical in a Node.js environment. The critical gaps are the 5 gameplay-logic functions marked CRITICAL/HIGH above.

---

## 7. Verified Existing Findings (from prior QA_REPORT.md)

All 9 previously reported bugs were verified via static analysis of current code:

| ID | Description | Status | Code Location |
|----|-------------|--------|---------------|
| BUG-01 | Buildings drift via entity separation | 🔴 Still present | `updateEntity` separation loop applies to all entities including buildings |
| BUG-02 | Food can go negative on unit death | 🟠 Still present | No `Math.max(0, ...)` guard in death handler |
| BUG-03 | Resource search radius limit (500px) | 🟠 Still present | `findNearestResource` hardcoded radius |
| BUG-04 | Food cap violation on TH death | 🟠 Still present | `maxFood` decremented without capping `food` |
| BUG-05 | `isPassable` missing unit check | 🟠 Still present | Only checks buildings for collision |
| BUG-06 | Peasant mid-route cancellation | 🟡 Still present | State not cleaned up if move interrupted |
| BUG-07 | Soft-lock on all-peasant death | 🟡 Still present | No free-peasant respawn mechanic |
| BUG-08 | No entity cap | 🔵 Still present | No limit on number of units/buildings |
| BUG-09 | Trees never removed from map data | 🔵 Still present | `mapData` unchanged after gathering |

---

## 8. Recommendations

### Immediate (before release)

1. **Add tests for `updateEntity`** — at minimum, test: death cleanup, construction progression, training queue advancement, and the peasant gather→carry→deposit cycle. These can be tested by setting up entity state and calling `updateEntity` with a delta time.

2. **Add tests for `issueMove` and `issueGather`** — verify state transitions and edge cases (null resource node, out-of-bounds targets, non-gatherable entities).

3. **Add a `cancelPlacement` guard** — prevent cursor class clobbering: `if (game.ui.mode !== 'placeBuilding') return;`

### Short-term (next iteration)

4. **Add 2–3 integration tests** — simulate a game tick sequence (e.g., spawn peasant at gold mine, run 30 frames, assert carrying state and resource amount).

5. **Run the round-trip coordinate test** — verify `screenToWorld → worldToScreen` returns the original screen coordinates within floating-point tolerance.

6. **Fix BUG-01 through BUG-05** — the medium+ severity bugs from the prior report.

### Nice-to-have

7. **Remove or wire up `worldToScreen`** — dead code is technical debt.
8. **Extract game logic into a testable module** — currently the test suite duplicates ~200 lines of game constants and logic from `index.html`. Refactoring to a shared module would prevent test drift.

---

## 9. Test vs. Game Function Coverage Matrix

```
Game Function          │ Tested? │ Test File Line │ Notes
───────────────────────┼─────────┼────────────────┼──────────────────────────
genId                  │   ✅    │ 334            │
dist                   │   ✅    │ helper tests   │
clamp                  │   ✅    │ helper tests   │
lerp                   │   ❌    │ —              │ Trivial, 1 line
rectsOverlap           │   ✅    │ helper tests   │
tileAt                 │   ✅    │ helper tests   │
isPassable             │   ✅    │ edge case tests│
spawnParticles         │   ❌    │ —              │ Visual only
spawnFloatingText      │   ❌    │ —              │ Visual only
generateMap            │   ✅    │ game state     │
createUnit             │   ✅    │ 255-295        │
createBuilding         │   ✅    │ 297-332        │
drawHealthBar          │   ❌    │ —              │ Canvas rendering
drawSelectionCircle    │   ❌    │ —              │ Canvas rendering
drawUnit               │   ❌    │ —              │ Canvas rendering
drawBuilding           │   ❌    │ —              │ Canvas rendering
render                 │   ❌    │ —              │ Canvas rendering
renderMinimap          │   ❌    │ —              │ Canvas rendering
screenToWorld          │   ❌    │ —              │ **ALL mouse I/O depends on this**
worldToScreen          │   ❌    │ —              │ **DEAD CODE** — never called
entityAtWorld          │   ✅    │ edge case tests│
entitiesInRect         │   ✅    │ edge case tests│
findNearestDropoff     │   ✅    │ 407+           │
findNearestResource    │   ✅    │ 407+           │
isPlacementValid       │   ✅    │ 415-455        │
startBuildingPlacement │   ❌    │ —              │ UI state machine
cancelPlacement        │   ❌    │ —              │ UI state machine
placeBuilding          │   ✅    │ placement tests│
startTraining          │   ✅    │ 343-405        │
getTrainTime           │   ✅    │ resource tests │
getBuildTime           │   ❌    │ —              │ Defined in test file, not in test blocks
issueMove              │   ❌    │ —              │ **Core player command**
issueGather            │   ❌    │ —              │ **Core player command**
update                 │   ❌    │ —              │ Game loop orchestration
updateEntity           │   ❌    │ —              │ **175 lines — most critical gap**
updateUI               │   ❌    │ —              │ DOM manipulation
commandStop            │   ❌    │ —              │ **Core player command**
resizeCanvas           │   ❌    │ —              │ DOM/layout
initGame               │   ✅    │ game state     │
gameLoop               │   ❌    │ —              │ rAF wrapper, hard to test meaningfully
```

**Score:** 21/40 functions have test coverage (52.5%). Excluding rendering/DOM functions (10), the effective coverage is **21/30 (70%)** — but the missing 30% (9 functions) includes the most critical code.

---

*Report generated by QA Engineer in the CAPTAIN CLAW fleet, 2026-07-03.*
