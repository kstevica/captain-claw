# Security Reviewer — review r2

The security review is complete. Here's a summary:

## Security Review R6 — RealmCraft RTS

**Overall Risk: LOW** — unchanged from prior rounds. No critical, high, or medium-severity vulnerabilities.

### Key Findings (14 total, CVSS-ranked)

| # | CVSS | Severity | Finding | Status |
|---|------|----------|---------|--------|
| SR-NEW-01 | 3.5 | Low | Prototype pollution via `game.resources[e.carryType]` dynamic key | ⚠️ Unfixed (R2→R6) |
| SR-01 | 2.8 | Low | 10 `innerHTML` sinks with dynamic game state | ⚠️ Unfixed (R0→R6) |
| **R6-01** | 2.5 | Low | `commandStop`/`issueMove` clear carry near dropoff without depositing | 🔴 **New R6** |
| SR-R3-01 | 2.5 | Low | NaN/Infinity coordinate passthrough | ✅ FIXED |
| SR-03 | 2.5 | Low | Unbounded entity creation | ⚠️ Unfixed |
| SR-R5-01 | 2.5 | Low | `findAutoBuildSite` O(n²·m) full-map fallback | ⚠️ Unfixed |
| SR-R5-02 | 2.0 | Low | Accidental global `cam` in `resizeCanvas` | ⚠️ Unfixed |
| SR-NEW-02 | 2.0 | Low | `Object.entries(BUILDING_DEFS)` prototype exposure | ⚠️ Unfixed |
| **R6-02** | 2.0 | Low | Deposit handler only fires for `moving` state — idle peasants wait ~2s | 🔴 **New R6** |
| SR-R5-03 | 1.8 | Low | Missing zero-guard on canvas dimension division | ⚠️ Unfixed |
| SR-04 | 1.8 | Low | `isPlacementValid` missing type guard | ✅ FIXED |
| SR-R4-02 | 1.5 | Low | `findNearestResource` 12-tile radius → now 30 tiles | ✅ PARTIALLY FIXED |
| R6-03 | 0.0 | Info | `autoBuildDropoff` cooldown expires check is strict `<` instead of `<=` | 🔴 **New R6** |
| R6-04 | 0.0 | Info | `commandStop`/`issueMove` partial fix regression risk | 🔴 **New R6** |

### Task-Specific: Auto-Build Dropoff Chain — **Functionally Correct**

All six stages of the resource lifecycle trace correctly:
1. **Gather initiation**: `carryType` always `'gold'` or `'wood'` ✓
2. **Gather completion**: Resources preserved when no dropoff exists ✓
3. **Dropoff search**: Type-discriminated (gold→refinery/TH, wood→mill/TH) ✓
4. **Auto-build trigger**: 5 guard clauses (existing, constructing, cooldown, cost, site) ✓
5. **Deposit**: Correctly adds to `game.resources` at any valid dropoff ✓
6. **Return cycle**: Peasant returns to gathering node after deposit ✓

### Status Tracker

- **2 findings confirmed FIXED** (R3→R6): NaN guards in entity factories, `isPlacementValid` type guard
- **1 finding partially fixed**: `findNearestResource` radius 12→30 tiles
- **1 finding evolved**: `commandStop` now conditionally clears carry (was unconditional in R4) — but still loses resources when near a dropoff
- **9 findings remain unfixed**
- **4 new findings** this round

### Top Remediation Priority

1. **SR-NEW-01** (3.5): Add `carryType` whitelist before dynamic key access — 5 minutes, one line
2. **R6-01** (2.5): Deposit resources before clearing carry in `commandStop`/`issueMove` instead of just zeroing them — 5 minutes
3. **SR-R5-02** (2.0): Add `const` to `cam=game.camera` in `resizeCanvas` — 1 character

The full report is saved to `.reports/SECURITY_REVIEW_R6.md`.