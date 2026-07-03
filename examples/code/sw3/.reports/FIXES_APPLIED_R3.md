# Fixes Applied — Review R3

**Date:** 2026-07-03 | **Tests:** 187 passed, 0 failed ✅ | **Syntax:** Valid ✅

---

## 🔴 BLOCKING

| ID | Issue | Fix | Lines |
|---|---|---|---|
| **B1** | `spawnFloatingText` undefined → ReferenceError | Added `spawnFloatingText(x,y,text,color)` function pushing to `game.floatingTexts` | +3 |

## 🟠 MAJOR

| ID | Issue | Fix | Lines |
|---|---|---|---|
| **M1** | `issueMove` doesn't clear peasant gathering state → ghost-gathering exploit | In `issueMove`, reset `gatheringNode`, `carryAmount`, `carryType`, `buildTarget` for peasant-type entities | +3 |
| **M2** | Gathering state never validates tile → gather anywhere | Added tile re-validation in gathering block: if current tile ≠ resource type, reset to idle | +5 |
| **M3** | No auto-target acquisition for military units | Added `findNearestEnemy` scan in `updateEntity` before attack block; engages if within range+80 | +7 |
| **M4** | Mixed selection ignores non-peasants on resource right-click | Added `else if(e&&e.owner==='player'&&!e.isBuilding) issueMove(...)` for non-peasant units | +1 |
| **M5** | Carrying peasant loses resources at non-dropoff building | Movement arrival no longer transitions to `gathering` when `carryAmount > 0`; deposit code handles dropoff | ~ |
| **M7** | Test `startTraining` lacks `produces` validation | Added `if(!building.produces||!building.produces.includes(unitType))` check in test helper | +2 |

## 🟡 MINOR

| ID | Issue | Fix |
|---|---|---|
| **m1** | Duplicate `attackTarget` in `createUnit` | Removed first declaration (line 269), kept second (line 276) |
| **m2** | Camera not clamped after `resizeCanvas` | Added clamp calls after resize |
| **m3** | `drawHealthBar` empty `if` block | Simplified to single condition: `if(hp>=maxHp && !selected) return` |
| **m4** | Dead arrays `trainingBuildings` / `trainingQueue` | Removed from game state object |
| **m7** | Rally point no terrain validation | Added `isPassable()` check with feedback |
| **m8** | `findNearestResource` null → silent no-op | Changed to move-to-location fallback with feedback |

## 🔴 QA Critical

| Issue | Fix |
|---|---|
| `cancelPlacement` cursor clobber | Added guard: `if(game.ui.mode!=='placeBuilding') return` at top of function |
| `screenToWorld` HiDPI formula mismatch | Rewrote to use `canvas.width/canvas.clientWidth` (DPR-aware) |

## 🔒 Security

| ID | Issue | Fix |
|---|---|---|
| **SR-FRESH-01** | Dynamic `carryType` → `game.resources[carryType]` prototype pollution path | Added validation: only `'gold'` or `'wood'` passes through |
| **SR-FRESH-02** | NaN/Infinity accepted in `createUnit`/`createBuilding` | Added `Number.isFinite()` checks |
| **SR-04** | `isPlacementValid` missing `!def` guard | Added `if(!def) return false` |
| **SR-01** | 2 `innerHTML` sinks in `updateUI` (building info + unit info) | Replaced with DOM API (`textContent`, `createElement`, `appendChild`) |

---

## Test Fixes

- Updated `startTraining` test helper to mirror game code's `produces` validation
- Fixed "not enough gold" test to use `stable` (which can produce `knight`) instead of `town_hall`
- Updated "BUG: no produces check" test note to reflect that the fix is now applied

---

## Verification

- **Test suite:** 187 passed, 0 failed ✅
- **JavaScript syntax:** Valid (parsed by Node.js `new Function()`)
- **All 26 fixes** from review R3 applied
