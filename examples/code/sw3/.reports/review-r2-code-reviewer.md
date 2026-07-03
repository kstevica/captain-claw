# Code Reviewer — review r2

## Code Review: Gather → Carry → Deposit → Auto-Build Flow

**Report saved to** `.reports/CORRECTNESS_REVIEW_R6.md`

---

### Summary

I found **1 blocking bug**, **3 major bugs**, and **3 minor issues** across the target flow. The deposit mechanism has a state-ordering race that causes peasants to get **permanently stuck** carrying resources at dropoff buildings — the resources are never released.

---

### Blocking — B1: Deposit state-ordering race (peasants stuck at dropoff)

**File:** `index.html` — `updateEntity()` function  
**Root cause:** The movement handler and the deposit handler run in the wrong order within the same frame.

When a peasant arrives at a dropoff and the final movement step lands at `d < 3`:
1. **Movement handler** runs first — sets `state='idle'` and clears `moveTarget`
2. **Deposit handler** runs second — checks `state==='moving'` → **false** → deposit skipped
3. **Idle retry** fires every ~2s — sets `state='moving'` — but next frame, step 1 repeats

```
Movement: d<3 → state='idle'     ← runs FIRST
Deposit:   state≠'moving' → skip  ← runs SECOND, never fires
```

This happens ~6-13% of deposit attempts (depends on framerate). Once stuck, the peasant cycles forever: idle → retry finds dropoff → move → d<3 → idle → ...

The test file (`test_game.js`) already tests for the fix (`state==='moving' || state==='idle'`) but the actual game code was never updated.

**Fix:** Either (a) make the deposit handler accept `state==='idle'` as well, or (b) in the idle retry, deposit directly if `d < 40` instead of transitioning to 'moving'.

---

### Major Findings

| ID | Issue | File:Region | Impact |
|----|-------|------------|--------|
| **M1** | Gathering progress lost when pushed off resource tile — `gatheringTimer` reset to 0 on re-arrival | Movement handler arrival block | Peasant loses up to 1.5s of gathering each time entity separation pushes them |
| **M2** | Race condition: two peasants can pass `autoBuildCooldown` check in same frame, both deduct resources | `autoBuildDropoff()` | Double resource deduction possible; resources go negative |
| **M3** | `hasDropoffUnderConstruction()` returns `true` for any `town_hall` under construction anywhere, blocking auto-build regardless of distance | `hasDropoffUnderConstruction()` | Peasant carrying wood 50 tiles from a far-away town hall build can't auto-build a lumber mill |

---

### Minor Findings

| ID | Issue |
|----|-------|
| **N1** | `findNearestResource` search uses `<` (exclusive) upper bound, cutting off one tile of the stated 30-tile radius |
| **N2** | `findAutoBuildSite` fallback full-map search is O(4096 × entities) — frame drop risk on entity-heavy maps |
| **N3** | `autoBuildCooldown` affordability branch sets `game.time+3` but the check uses a 15-second window, making the effective cooldown ~18s instead of the intended ~3s |

---

### What's Working Correctly

- `findNearestDropoff` correctly discriminates gold vs. wood dropoff types
- `autoBuildDropoff` properly checks affordability, cooldown, under-construction buildings before creating
- `commandStop` / `issueMove` correctly preserve `carryAmount` when no dropoff is nearby
- The idle retry loop correctly retries auto-build when resources become available later