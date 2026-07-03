# Code Reviewer — review r1

The report is saved to `.reports/CORRECTNESS_REVIEW_R5.md`. Here's a summary:

---

## Correctness Review R5 — Findings

**Test suite:** 226/226 passing ✅

### 🔴 BLOCKING (1)

**B1 — Wood gathering is completely broken** (`index.html:152-155`, `index.html:1232-1243`)

`isPassable()` returns `false` for TREE tiles, but wood-gathering requires the peasant to stand ON a tree tile. The movement snap (d<3) can never fire because the closest approach to a tree tile center from outside is ~16px. **Gold works fine** (GOLD isn't blocked). Simulation confirms: after 1000 steps, distance=18.1px, tile=grass — can never reach the tree.

**Fix:** Remove `TERRAIN.TREE` from `isPassable()`:
```javascript
// Line 154: change
if(t===TERRAIN.WATER||t===TERRAIN.TREE) return false;
// to
if(t===TERRAIN.WATER) return false;
```

### 🟠 MAJOR (2)

**M1 — Unnecessary `findNearestDropoff` + `autoBuildDropoff` calls every frame** (`index.html:1268-1292`)

Idle peasants carrying resources call `findNearestDropoff` (O(N) entity scan) every frame when no dropoff is nearby. The `autoBuildDropoff` cooldown prevents actual building spam, but the scan itself is wasted work. **Fix:** only run this check for idle peasants when the ~2s interval fires (consistent with the idle retry loop).

**M2 — `issueMove`/`commandStop` proximity check is semantically wrong** (`index.html:976`, `index.html:1507`)

`findNearestDropoff` returns the nearest dropoff **anywhere on the map**. A peasant carrying gold at (100,100) with a town hall at (1000,1000) still has resources cleared. The fix should check proximity (d<40), not existence. Currently the "preserve" branch is unreachable in normal play (town hall always exists).

### 🟡 MINOR (4)

| ID | Issue | Location |
|----|-------|----------|
| m1 | `findNearestResource` 30-tile cap misses distant resources | line ~810 |
| m2 | `autoBuildCooldown` +3 offset creates ~18s block vs 15s success | line ~786 |
| m3 | `gatheringNode` stale pointer through auto-build lifecycle | lines ~790-800 |
| m4 | Auto-build subsystem has zero integration tests | `test_game.js` |

### ✅ R4 Fixes Confirmed Working

All 5 R4 MAJOR findings are properly fixed:
- M1 (commandStop): ✅ Preserves resources when no dropoff exists
- M2 (gathering separation): ✅ Walks back to gatheringNode
- M3 (town_hall under construction): ✅ Included in `hasDropoffUnderConstruction`
- M4 (auto-build spam): ✅ Cooldown set on failure
- M5 (search origin): ✅ Uses `gatheringNode` when available
- R4-001 (deposit infinite loop): ✅ Deposit check fires for idle state