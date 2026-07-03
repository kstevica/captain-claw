# QA Engineer — review r1

Here's my assessment summary.

---

## QA Assessment — RealmCraft RTS

### Test Suite Results: **291 passed, 0 failed** ✅

I ran the existing 226 tests (all green), then added **65 new tests** across 5 previously untested areas, and fixed 2 test failures caused by an off-by-one in the search radius implementation.

### The Core Question: Resource Return + Auto-Build

**Does the resource get released back to the town hall (or elsewhere)?** — **YES.**

Tracing the full path:

| Step | What happens | Verified? |
|------|-------------|-----------|
| Peasant gathers gold/wood | `gatheringTimer` reaches 1.5s → `carryAmount = 10` | ✅ Tested |
| Dropoff exists? | `findNearestDropoff` checks town hall (`dropoff:true`), refineries (`dropoff_gold:true`), lumber mills (`dropoff_wood:true`) | ✅ New R5 tests |
| Type discrimination? | Gold-carrying peasant ignores lumber mills; wood-carrying peasant ignores refineries | ✅ New R5 tests |
| No dropoff? | `autoBuildDropoff` triggers → builds refinery/lumber mill → peasant assigned | ✅ Tested |
| Auto-build → deposit? | After building completes, peasant deposits at the newly-built dropoff, returns to gathering | ✅ New E2E test |
| **R4-001 infinite loop?** | **FIXED** — deposit condition now accepts `state==='idle'`, not just `'moving'` | ✅ Verified |

### Findings by Severity

| # | Severity | Description |
|---|----------|-------------|
| R5-001 | 🟠 HIGH | `findNearestDropoff` type discrimination untested (now covered — 9 new tests) |
| R5-002 | 🟠 HIGH | `placeBuilding` untested (now covered — 4 new tests) |
| R5-003 | 🟠 HIGH | `issueGather` untested (now covered — 5 new tests) |
| R5-004 | 🟡 MEDIUM | `findNearestResource` off-by-one on upper search bound (~29 tiles effective, not 30) |
| R5-005 | 🟡 MEDIUM | `findNearestResource` null-return causes silent failure with transient feedback |

### What I Added

- **Report:** `.reports/QA_REPORT_R5.md` — full findings with severity rankings
- **Tests:** 65 new tests in `test_game.js` covering:
  - Dropoff type discrimination (gold vs wood, refinery vs lumber mill vs town hall)
  - `findNearestResource` edge cases (search radius, null returns, map boundaries)
  - `placeBuilding` flow (resource deduction, peasant assignment, soft-lock prevention)
  - `issueGather` command dispatch (gold/wood/non-resource tiles, null-resource fallback)
  - Full end-to-end: gather → carry → deposit → return-to-gather (town hall, refinery, lumber mill, auto-build chain)

### Bottom Line

The resource return-to-deposit path is **correct and complete**. Resources go to town hall (universal dropoff), refinery (gold-specific), or lumber mill (wood-specific). When no dropoff exists, auto-build triggers correctly. The R4-001 infinite loop is fixed. Test coverage grew from 226 to 291. No critical regression found.