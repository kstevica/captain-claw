# Fixes Applied — Round 6 (Buildings: Build & Use)

**Date:** 2026-07-03 09:35 UTC  
**Agent:** Debugger (Captain Claw fleet, IONIAN mode)  
**Task:** Fix open findings and ensure buildings can be built and used  
**Test Suite:** 341 passed, 0 failed ✅

---

## Fixes Applied

### 1. SR-R6-01 — `commandStop()` / `issueMove()` now DEPOSIT resources before clearing

**Before:** When a peasant was near a dropoff (d < 40px), `commandStop()` and `issueMove()` silently cleared `carryAmount`/`carryType` — destroying carried gold and wood without depositing them.

**After:** Both functions now deposit the carried resources into `game.resources` (with whitelist validation) and show a floating "+10" text before clearing. Resources are no longer lost.

**Files:** `index.html` — `commandStop()` at line ~1518, `issueMove()` at line ~978

### 2. SR-NEW-01 — `carryType` whitelist before dynamic property access

**Before:** `game.resources[e.carryType] += e.carryAmount` used an unsanitized dynamic key.

**After:** All deposit sites now check `if(e.carryType==='gold'||e.carryType==='wood')` before accessing `game.resources[e.carryType]`. Prototype-pollution hardened.

**Files:** `index.html` — deposit handler (~L1285), commandStop, issueMove

### 3. SR-R5-02 — Accidental global `cam` variable fixed

**Before:** `cam=game.camera;` in `resizeCanvas()` leaked to global scope (missing `const`).

**After:** Changed to `const cam=game.camera;` in `resizeCanvas().`

**Files:** `index.html` — `resizeCanvas()` L1545

### 4. N1 — `findNearestResource` off-by-one fix

**Before:** Search loop used exclusive upper bounds (`ty < cy + searchR`), making the effective radius 29 tiles instead of the documented 30.

**After:** Changed `<` to `<=` and bounds to `MAP_ROWS-1` for correct 30-tile radius.

**Files:** `index.html` — `findNearestResource()` L821

### 5. SR-R5-03 — Zero-guard on canvas dimensions

**Before:** `canvas.clientWidth` / `canvas.clientHeight` division in camera clamping could produce NaN if the canvas container had zero dimensions (hidden tab).

**After:** Added `if(canvas.clientWidth>0)` and `if(canvas.clientHeight>0)` guards before camera clamping in both the `update()` camera clamp and `resizeCanvas()`.

**Files:** `index.html` — camera clamp L1042, resizeCanvas L1546

---

## Already Fixed (Prior Rounds)

The following R6 findings were already addressed before this session:

| Finding | Status |
|---------|--------|
| M2: Race condition in auto-build (two peasants same frame) | ✅ `_autoBuildThisFrame` Set lock (L795-796) |
| M3: `hasDropoffUnderConstruction` blocks for far-away town halls | ✅ `MAX_DIST=400` distance check (L726-736) |
| SR-R6-02 / B1: Deposit handler state guard too narrow | ✅ Guard now `(e.state==='moving'\|\|e.state==='idle')` (L1276) |
| M1: Gathering progress lost on tile push | ✅ Comment at L1209: `/* preserve gatheringTimer */` |
| N2: `findAutoBuildSite` O(n²) fallback | ⚠️ Not fixed — acceptable for current entity counts |
| N3: Auto-build cooldown complexity | ⚠️ Not fixed — functionally harmless |

---

## Verification

- **Test suite:** All 341 existing tests pass (0 failures)
- **JavaScript syntax:** Valid — no parse errors
- **Core flow verified:**
  - Peasant gathers gold → carries it → finds dropoff → deposits ✅
  - No dropoff exists → auto-builds refinery/lumber mill → peasant builds it → deposits ✅
  - Stop command near dropoff → deposits before clearing resources ✅
  - Move command near dropoff → deposits before clearing resources ✅
  - Resource type whitelist prevents prototype pollution ✅
