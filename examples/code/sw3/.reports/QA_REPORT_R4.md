# QA Report: RealmCraft RTS — Round 4 Assessment

**Date:** 2026-07-03 08:48 UTC  
**Assessor:** QA Engineer (CAPTAIN CLAW fleet, session `c3e28af8`)  
**Focus Area:** Worker resource gathering → carrying → depositing → auto-build dropoff  
**Artifact:** `index.html` (~1,500 lines) — single-file RTS game  
**Test Suite:** `test_game.js` (187 tests, all passing)  
**Methodology:** Test suite execution + deep forensic code trace of the gather→carry→deposit→auto-build state machine

---

## Executive Summary

**Overall Quality: GOOD** — but the auto-build dropoff path has a **critical correctness bug** that prevents peasants from ever depositing resources after auto-building a dropoff, creating an infinite loop.

The test suite is solid at **187/187 passing**, covering entity creation, resource management, placement validation, updateEntity core logic, and coordinate conversion. However, the **entire auto-build subsystem** (`autoBuildDropoff`, `findNearestDropoff`, `findNearestResource`, `findAutoBuildSite`, `hasDropoffUnderConstruction`, `getDropoffBuildingType`) has **zero dedicated tests**.

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 1 | Auto-build → deposit infinite loop: peasant stuck forever after building dropoff |
| 🟠 HIGH | 2 | Auto-build subsystem fully untested; `issueGather`/`issueMove` untested |
| 🟡 MEDIUM | 3 | Idle retry timing gap; `gatheringNode` not cleared during auto-build; `issueMove` wipes carried resources |
| 🔵 LOW | 2 | Deposit opportunistic-firing on any nearby dropoff; missing integration test |

---

## Test Suite Results

```
$ node test_game.js
Running RealmCraft RTS Test Suite...

RESULTS: 187 passed, 0 failed
```

**All 187 tests pass.** The suite has grown from 123 tests (R2) to 187 tests (R4), adding `updateEntity — Core Gameplay`, `screenToWorld`, and `cancelPlacement — Cursor Guard` sections. However, the auto-build subsystem and command functions remain untested.

---

## Finding R4-001 🔴 CRITICAL: Auto-build → deposit infinite loop

**Severity:** CRITICAL  
**Location:** `index.html` — interaction between `updateEntity` movement code (step 8), deposit arrival code (step 11), and idle retry (step 13)

### Root Cause

The `updateEntity` function processes entity states in a fixed order within a single frame:

1. **Movement code** (step 8, runs first): When peasant arrives at target (`d < 3`), sets `state='idle'` and `moveTarget=null`
2. **Deposit arrival code** (step 11, runs later): Requires `state==='moving' && moveTarget` — already cleared by step 8
3. **Idle retry** (step 13, runs last): Fires every ~2 seconds, finds dropoff, sets `state='moving'` + `moveTarget`

### The Loop

```
1. Peasant gathers gold → carryAmount=10
2. No dropoff exists → autoBuildDropoff succeeds → refinery created at site S
3. Peasant moves to S, arrives (d<3) → movement code sets state='idle', moveTarget=null
4. Peasant's buildTarget is set → state='building' → builds refinery
5. Refinery completes → peasant: buildTarget=null, state='idle', carryAmount=10
6. Idle retry fires → finds completed refinery → sets state='moving', moveTarget=refinery
7. Peasant is ALREADY at refinery center (d≈0 < 3) → movement code sets state='idle', moveTarget=null
8. Deposit check: state is 'idle' → SKIPPED (carryAmount still 10, never deposited)
9. GOTO 6 (infinite loop — peasant forever oscillates between idle and "arriving")
```

### Why it happens

The normal deposit path works because the deposit check catches peasants **approaching** a dropoff (within 40px, not yet arrived). But after auto-build, the peasant is already AT the dropoff location — there is no "approach" window. The movement code fires first and sets state='idle', starving the deposit check.

### Impact

- **Resources are never deposited** after auto-building a dropoff
- Peasant is permanently stuck carrying resources, never returning to gathering
- The auto-built dropoff is unusable by the peasant who built it
- Other peasants CAN use the completed dropoff (they approach it normally)

### Reproduction

1. Start game (3 peasants, 1 town hall)
2. Right-click a peasant on a gold mine far from the town hall
3. Wait for gather to complete → auto-build triggers → refinery appears
4. Observe: peasant never deposits gold, stays stuck near refinery

### Recommended Fix

Add a deposit check for idle peasants carrying resources that are near a dropoff, **before** the movement code in the update loop. Or, add an explicit deposit in the idle retry handler instead of just setting moveTarget:

```javascript
// In idle retry (step 13), replace moveTarget assignment with direct deposit:
const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
if (dropoff) {
  const d = dist({x:e.x,y:e.y}, {x:dropoff.x,y:dropoff.y});
  if (d < 40) {
    // Direct deposit — don't use moveTarget
    game.resources[e.carryType] += e.carryAmount;
    e.carryAmount = 0;
    if (e.gatheringNode) {
      e.state = 'moving';
      e.moveTarget = {x: e.gatheringNode.x, y: e.gatheringNode.y};
    }
  } else {
    e.state = 'moving';
    e.moveTarget = {x: dropoff.x, y: dropoff.y};
  }
}
```

---

## Finding R4-002 🟠 HIGH: Auto-build subsystem has zero test coverage

**Severity:** HIGH  
**Location:** `index.html` functions: `autoBuildDropoff`, `findNearestDropoff`, `findNearestResource`, `findAutoBuildSite`, `hasDropoffUnderConstruction`, `getDropoffBuildingType`

### Details

Six functions that implement the entire auto-build feature have no tests in `test_game.js`:

| Function | Lines | Complexity | Tests |
|----------|-------|------------|-------|
| `autoBuildDropoff` | ~35 | Decision tree (5 exit points) | 0 |
| `findNearestDropoff` | ~15 | Loop + type dispatch | 0 |
| `findNearestResource` | ~14 | Nested loop (30×30 search) | 0 |
| `findAutoBuildSite` | ~25 | Expanding ring + full-map fallback | 0 |
| `hasDropoffUnderConstruction` | ~8 | Loop + type match | 0 |
| `getDropoffBuildingType` | ~5 | Simple switch | 0 |

The auto-build function has at least 5 distinct exit conditions (none of which are tested):
- `carryType` is null or `carryAmount <= 0`
- Dropoff already exists → return false
- Dropoff under construction → return false
- Cooldown active → return false
- Cannot afford → return false
- No valid build site → return false
- Success: deduct resources, create building, assign peasant → return true

### Impact

Any change to the auto-build logic is unverified. The infinite loop bug (R4-001) would have been caught by an integration test of the gather→auto-build→deposit flow.

---

## Finding R4-003 🟠 HIGH: `issueGather` and `issueMove` commands untested

**Severity:** HIGH  
**Location:** `index.html` functions: `issueMove`, `issueGather`

### Details

These are the primary player-to-game command interfaces. Neither is tested:

- **`issueMove`:** Clears peasant state (gatheringNode, carryAmount, carryType, buildTarget) — the carry wipe on move is potentially destructive behavior that should be tested
- **`issueGather`:** Dispatches based on tile type (GOLD → gold, TREE → wood, else → move). The `findNearestResource` null-guard path is untested

### Specific concern

`issueMove` unconditionally wipes `carryAmount` and `carryType` for peasants:
```javascript
if(e.type==='peasant') {
  e.gatheringNode=null; e.carryAmount=0; e.carryType=null; e.buildTarget=null;
}
```
This means issuing a move command to a peasant mid-delivery **silently discards carried resources**. This may be intentional (design choice) but is untested and undocumented.

---

## Finding R4-004 🟡 MEDIUM: Idle retry timing gap — up to 2-second delay before deposit

**Severity:** MEDIUM  
**Location:** `index.html` idle retry loop (~line 1300)

### Details

The idle retry fires only when:
```javascript
if(Math.floor(game.time*10)%20===0) // check every ~2 seconds
```

After a peasant arrives at a dropoff and deposits via the approach window, they go back to gathering immediately. But if the deposit check misses (e.g., during auto-build aftermath), the peasant must wait up to 2 seconds for the idle retry to fire. This creates a visible "stutter" where a peasant stands idle at the dropoff before the deposit triggers.

This timing gap combines with the R4-001 bug to make the situation worse.

---

## Finding R4-005 🟡 MEDIUM: `gatheringNode` not cleared during `autoBuildDropoff` success

**Severity:** MEDIUM  
**Location:** `index.html` `autoBuildDropoff` function

### Details

When `autoBuildDropoff` succeeds, it sets:
```javascript
peasant.state = 'moving';
peasant.moveTarget = { x: site.x, y: site.y };
peasant.buildTarget = b.id;
```

But `gatheringNode` is **not cleared**. After the building completes and the peasant deposits (if R4-001 were fixed), the peasant would return to the original gathering node. This is arguably correct behavior (resume gathering where you left off), but it's undocumented state persistence that could cause confusion.

If the player manually intervenes or if the gathering node's resource state changes, the stale `gatheringNode` reference leads to undefined behavior.

---

## Finding R4-006 🟡 MEDIUM: `issueMove` silently discards carried resources

**Severity:** MEDIUM  
**Location:** `index.html` `issueMove` function

### Details

Already noted in R4-003, but worth a separate finding:
```javascript
if(e.type==='peasant') {
  e.gatheringNode=null; e.carryAmount=0; e.carryType=null; e.buildTarget=null;
}
```

This is called for ALL right-click moves, including:
- Attack-move
- Normal move
- Move-to-ally-building (which could be a deposit intent)

A peasant carrying 10 gold who is right-clicked to move 1 tile loses the gold permanently with no feedback. The player gets no warning.

**Comparison:** Warcraft/StarCraft — workers preserve carried resources when issued move commands; they only drop resources on Stop.

---

## Finding R4-007 🔵 LOW: Deposit fires opportunistically on any nearby dropoff, not just the targeted one

**Severity:** LOW  
**Location:** `index.html` deposit arrival code

### Details

The deposit check:
```javascript
const dropoff = findNearestDropoff(e.x, e.y, e.carryType);
if (dropoff) {
  if (d < 40) { /* deposit */ }
}
```

This deposits at the **nearest** dropoff, not the one the peasant's `moveTarget` points at. If a peasant is moving past a refinery to reach a town hall, they'll deposit at the refinery instead. This is arguably correct (deposit at the first dropoff you encounter), but differs from some RTS conventions where workers go to the specific dropoff assigned.

---

## Finding R4-008 🔵 LOW: No integration test for the full gather→carry→deposit loop

**Severity:** LOW  
**Location:** `test_game.js`

### Details

The test suite has excellent unit tests for:
- Entity creation
- Resource deduction (startTraining)
- Placement validation
- updateEntity sub-behaviors (movement, combat, gathering timer, building construction)

But there is **no end-to-end test** that:
1. Creates a peasant at a gold mine
2. Simulates gathering for 1.5s
3. Verifies carryAmount = 10
4. Simulates movement to dropoff
5. Verifies resources are added to game.resources
6. Verifies peasant returns to gathering

This is the single most important gameplay loop and it's not integration-tested.

---

## Summary of Findings

### The Good

- Test suite: **187/187 passing**, zero failures
- Core entity lifecycle (creation, death, food tracking) is well-tested
- `updateEntity` sub-behaviors (movement, combat, gathering timer) have tests
- Resource deduction, placement validation, and coordinate conversion are tested
- Auto-build feature design is sound: cooldown, under-construction detection, affordability gates, fallback site search — all correctly implemented

### The Bad

- **R4-001 (CRITICAL):** Auto-build → deposit infinite loop. Peasants who auto-build a dropoff can never deposit at it, getting stuck in a move→arrive→idle→retry cycle
- **R4-002 (HIGH):** Auto-build subsystem wholly untested (6 functions, 0 tests)
- **R4-003 (HIGH):** Player commands `issueMove`/`issueGather` untested; `issueMove` silently discards carried resources

### The Ugly

- **R4-004 (MEDIUM):** Idle retry introduces up to 2-second delay before deposit retry
- **R4-005 (MEDIUM):** `gatheringNode` state lingers through auto-build lifecycle
- **R4-006 (MEDIUM):** `issueMove` wipes carried resources with no player feedback

---

## Test Coverage Gap Map

```
Function                    Tested?   Lines   Notes
───────────────────────────────────────────────────────────
createUnit                  ✅        20      40 tests
createBuilding              ✅        15      8 tests
isPlacementValid            ✅        20      10 tests
startTraining               ✅        25      8 tests
updateEntity (partial)      ✅        175     13 tests (movement, combat, gathering, death)
screenToWorld               ✅        5       4 tests (R4 addition)
cancelPlacement             ✅        5       1 test (R4 addition)
── GAP BELOW ──
autoBuildDropoff            ❌        35      0 tests
findNearestDropoff          ❌        15      0 tests
findNearestResource         ❌        14      0 tests
findAutoBuildSite           ❌        25      0 tests
hasDropoffUnderConstruction ❌        8       0 tests
getDropoffBuildingType      ❌        5       0 tests
issueGather                 ❌        18      0 tests
issueMove                   ❌        12      0 tests
placeBuilding               ❌        30      0 tests
generateMap                 ❌        50      0 tests
```

---

## Recommended Actions (Priority Order)

1. **Fix R4-001:** Add direct deposit in the idle retry handler when peasant is already within 40px of a dropoff — don't route through moveTarget
2. **Add auto-build tests:** Test all 7 exit paths of `autoBuildDropoff`, plus integration test for gather→auto-build→deposit flow
3. **Add issueGather/issueMove tests:** Test resource preservation behavior, null-guard paths, and tile dispatch
4. **Consider** whether `issueMove` should preserve carried resources (player feedback wanted)
5. **Add** end-to-end gather→carry→deposit integration test
