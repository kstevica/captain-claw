# QA Engineer — review r0

Report written to `.reports/QA_REPORT_R4.md` (333 lines, 14KB). Here's the summary:

---

## QA Assessment Results — Worker Resource Flow

**Test suite: 187/187 passing.** However, the resource gathering → carry → deposit → auto-build flow has **one critical bug** that breaks the entire loop after auto-building a dropoff.

### Critical Finding: Auto-Build Deposit Deadlock (qa-r4-01)

The deposit check at line ~1265 requires `state==='moving'`, but the movement code (which runs *first* in `updateEntity`) sets `state='idle'` when the peasant arrives at the dropoff (d < 3 pixels). This means:

**Normal path (dropoff already exists):** Works correctly — the peasant approaches the dropoff, and at d < 40 pixels (while still `moving`), the deposit fires. Resources are added. Peasant returns to gathering. ✓

**Auto-build path (no dropoff exists):** **BROKEN.** Here's the sequence:
1. Peasant gathers → `carryAmount=10`
2. No dropoff found → `autoBuildDropoff` creates refinery/lumber mill
3. Peasant moves to building site, arrives (d < 3), state → `building`
4. Building completes → state → `idle`, `buildTarget=null`
5. Idle retry fires → finds the completed dropoff → sets `state='moving'`, `moveTarget=dropoff`
6. **But the peasant is already AT the dropoff** (d < 3) → movement code immediately sets `state='idle'`
7. Deposit check: state is `'idle'` → **SKIPPED**
8. **Infinite loop**: goto step 5

The peasant is permanently stuck carrying resources it can never deposit.

### Other Findings

| # | Severity | Finding |
|---|----------|---------|
| qa-r4-01 | 🔴 CRITICAL | Auto-build deposit deadlock (see above) |
| qa-r4-02 | 🟠 HIGH | `autoBuildDropoff` and the entire gather→deposit chain **untested** in `test_game.js` |
| qa-r4-03 | 🟡 MEDIUM | `commandStop` clears `carryAmount=0` — if used during auto-build sequence, resources vanish |
| qa-r4-04 | 🟡 MEDIUM | `issueMove` also clears carried resources — player right-click destroys gathered gold/wood |
| qa-r4-05 | 🟢 LOW | `gatheringNode` not cleared during auto-build (minor state inconsistency, not harmful) |
| qa-r4-06 | 🔵 INFO | Town hall serves as universal dropoff (`dropoff:true`) — correctly handles both gold and wood |

### Recommended Fix for qa-r4-01

In the deposit arrival handler, remove the `state==='moving'` guard so it also catches `'idle'` peasants carrying resources near a dropoff:

```
// Before (broken):
if (e.type==='peasant'&&e.state==='moving'&&e.carryAmount>0&&e.moveTarget)

// After (fixed): check any state for carrying peasants near dropoff
if (e.type==='peasant'&&e.carryAmount>0)
```

This lets the deposit fire on the same frame the auto-built dropoff completes, avoiding the deadlock.