# Code Reviewer — review r0

Let me present the review summary.

---

## Code Review R4: Auto-Build Dropoff System

**Reviewed:** `index.html` (~1,600 lines) + `test_game.js`

### R3 Fixes Confirmed ✅
The following R3 issues are **resolved** in the current code:
- `spawnFloatingText` is defined (line 168)
- `issueMove` now properly clears peasant gathering state (line 966)
- Gathering validates the tile under the peasant (lines 1220–1224)
- `cancelPlacement` guards against non-placement mode (line 862)
- Attack auto-acquire logic is present (lines 1145–1150)

---

### 🔴 MAJOR Findings (Specific to the Task)

**M1. `commandStop` destroys carried resources — breaks auto-build preservation**
`index.html:1484` — Pressing Stop (S) on a peasant carrying resources from a failed auto-build permanently loses them. The retry loop at line 1300 depends on `carryAmount` being preserved, but `commandStop` unconditionally clears it. Fix: only clear carryAmount if a dropoff exists, or leave resources intact.

**M2. Gathering timer discarded on separation push — fragile gathering**
`index.html:1220–1224` — When entity separation pushes a peasant even 1px off the resource tile, the entire `gatheringTimer` is discarded (including at 1.4s/1.5s). The validation should walk the peasant back to `gatheringNode` instead of resetting everything.

**M3. `hasDropoffUnderConstruction` misses `town_hall` — redundant auto-build**
`index.html:726–731` — Only checks for refinery/lumber_mill under construction. When a town hall (universal dropoff) is being built, `autoBuildDropoff` may create an unnecessary refinery/lumber_mill, wasting 100g/50w. Fix: add `e.type==='town_hall'` to the check.

**M4. Failed `autoBuildDropoff` doesn't set cooldown — repeated attempts**
`index.html:765–810` — On resource- or space-failure, the cooldown is never set. The idle retry loop (line 1302) calls `autoBuildDropoff` every ~2 seconds, generating `showFeedback` spam. Fix: set a short cooldown even on failure.

**M5. `findAutoBuildSite` searched near peasant, not near resources**
`index.html:784` — Builds the dropoff near the peasant's position, not near `gatheringNode`. If the peasant walked far from the resource (e.g., retry loop), the dropoff ends up in a suboptimal location. Fix: use `gatheringNode` coordinates as search origin when available.

---

### 🟡 MINOR Findings

**m1.** `autoBuildDropoff` spawns floating text without checking if `game.floatingTexts` is valid — `index.html:807`
**m2.** Idle retry uses fragile `Math.floor(game.time*10)%20===0` — jitter or frame drops could cause missed checks — `index.html:1302`
**m3.** `autoBuildDropoff` doesn't guard against the peasant having died between gathering completion and the call — `index.html:765`
**m4.** `isPlacementValid` entity overlap check doesn't filter by owner — blocks placement on enemy buildings (future concern) — `index.html:845`
**m5.** `findAutoBuildSite` full-map fallback is O(n²) — iterates 4096 tiles × all entities; could cause frame drops — `index.html:749–760`

---

### 📊 Summary

| Category | Count |
|----------|-------|
| R3 fixes confirmed | 5 |
| MAJOR (new) | 5 |
| MINOR (new) | 5 |
| BLOCKING (new) | 0 |

The full report with concrete fix suggestions is at `.reports/CORRECTNESS_REVIEW_R4.md`.