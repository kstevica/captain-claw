# Security Reviewer — review r0

The report is saved at `.reports/SECURITY_REVIEW_R4.md`. Here's the executive summary:

---

## Security Review R4 — RealmCraft RTS

**Overall Risk: LOW** (unchanged). No critical, high, or medium-severity vulnerabilities.

### Two Fixes Confirmed Since R3

| Finding | What Changed |
|---------|-------------|
| **SR-R3-01** (NaN/Infinity passthrough) | ✅ **Fixed** — `Number.isFinite()` guards added to `createUnit` and `createBuilding` (lines 267, 287) |
| **SR-04** (isPlacementValid TypeError) | ✅ **Fixed** — `if(!def) return false;` guard added at line 834 |

### Seven Findings Still Unfixed

| # | CVSS | Title |
|---|------|-------|
| SR-NEW-01 | 3.5 | Dynamic property key `game.resources[e.carryType]` — prototype pollution path |
| SR-01 | 2.8 | Three dynamic `innerHTML` sinks in `updateUI()` |
| SR-03 | 2.5 | Unbounded entity creation (no entity limit) |
| SR-R3-03 | 2.0 | Mutable definition objects (no `Object.freeze`) |
| SR-NEW-02 | 2.0 | `Object.entries(BUILDING_DEFS)` iteration exposes prototype pollution |
| SR-R3-03b | 0.0 | Duplicate `attackTarget:null` at lines 273 and ~278 (dead code trap) |
| SR-02 | 0.0 | Missing Content-Security-Policy |

### Four New Findings (This Review)

| # | CVSS | Focus | Finding |
|---|------|-------|---------|
| **SR-R4-01** | 3.1 | **Task-critical** | `commandStop()` unconditionally discards `carryAmount`/`carryType` — contradicts auto-build resource preservation design |
| **SR-R4-02** | 2.5 | Task-relevant | `findNearestResource` hardcoded to 12-tile search radius — if all nearby resources depleted, peasants can never find new ones, auto-build never triggers |
| **SR-R4-03** | 2.0 | Task-relevant | Auto-build fallback `findAutoBuildSite` iterates 4096 tiles × all entities = O(n²) — frame spike when every idle carrying peasant retries |
| **SR-R4-04** | 1.5 | Test infra | Test-code divergence — `test_game.js` throws on unknown types (game returns null), lacks NaN guards, missing `autoBuildCooldown` state initializer |

### Task-Specific Verdict

The core auto-build flow (peasant gathers → no dropoff exists → auto-build refinery/lumber mill → peasant deposits) is **logic-complete** and has the correct guards (cooldown, under-construction detection, affordability check, periodic idle retry). The three hook points are all properly wired. **However**, SR-R4-01 (Stop discarding resources) creates a user-facing path where carried resources are silently lost — directly undermining the auto-build "don't discard resources" design intent.