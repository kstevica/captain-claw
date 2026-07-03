# Code Reviewer — review r3

The report is saved at `.reports/CORRECTNESS_REVIEW_R5.md`. Here's a summary of the key findings:

---

## Findings Summary

### 🔴 BLOCKING (1)

**B1. autoBuildDropoff race: peasant order cancelled by deposit handler same-frame**
`index.html` ~line 1340 (deposit handler)

When gathering completes and `autoBuildDropoff()` successfully creates a building, sets peasant to `state='moving'` with `buildTarget`, the deposit handler fires **immediately after in the same frame**. It calls `autoBuildDropoff()` again, which returns `false` (the building is under construction → `hasDropoffUnderConstruction` returns true), then overwrites `e.state='idle'` — zapping the move-to-build order. The peasant stands idle for the full build time (5s) while the building self-completes with zero builders.

**Fix (one-line):** Add `!e.buildTarget` guard in the deposit handler:
```javascript
} else if (!e.buildTarget && !autoBuildDropoff(e)) {
```

---

### 🟠 MAJOR (3)

**M1. Entity separation pushes completed buildings**
`index.html` ~line 1319 (`updateEntity` separation logic) + test_game.js line 460

The separation loop runs for `state==='moving'||state==='idle'`. Completed buildings have `state='idle'` and `speed=0` — but `e.speed||2` evaluates to `2` (0 is falsy), so buildings drift. Fix: add `!e.isBuilding` to the separation guard.

**M2. `findNearestResource` off-by-one in upper bound**
`index.html` ~line 830, `findNearestResource()`

Uses `ty < Math.min(...)` (exclusive) instead of `<=`. With `searchR=30`, it only searches +29 tiles on the positive axis. The comment says 30-tile radius. Fix: change `<` to `<=`.

**M3. `isPassable` ignores `blocking` property**
`index.html` ~line 190, `isPassable()`

The function checks ALL completed buildings (not just `blocking:true` ones), making `blocking` on walls redundant. All buildings — including refineries, watch towers — block pathfinding. Fix: add `&& e.blocking` to the condition.

---

### 🟡 MINOR (3)

- **N1:** `findNearestResource` capped at 30 tiles — distant resources never found
- **N2:** `autoBuildDropoff` fallback O(n²) full-map scan on ring-search failure
- **N3:** Deposit handler calls `autoBuildDropoff` redundantly every frame (mitigated by cooldowns)

All 291 unit tests pass. The blocking bug was masked because the tests simulate state machines in isolation — they never exercise the same-frame sequence where `autoBuildDropoff`'s output is undone by the deposit handler.