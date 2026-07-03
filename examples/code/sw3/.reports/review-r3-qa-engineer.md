# QA Engineer — review r3

Here's my assessment summary:

---

## QA Assessment: RealmCraft RTS — Resource Deposit & Auto-Build

**Overall: ✅ The core resource-release and auto-build chain is correct.** I traced every code path end-to-end.

### Test Suite Before: 291 pass, 0 fail
### Test Suite After: 341 pass, 0 fail (+50 new assertions, 10 new scenarios)

### The critical flow works:

| Step | What happens | Verdict |
|------|-------------|---------|
| Peasant gathers gold/wood | `gatheringTimer ≥ 1.5` → `carryAmount = 10` | ✅ |
| Find dropoff (Town Hall, Refinery, Lumber Mill) | `findNearestDropoff` checks `progress ≥ 1` and type match | ✅ |
| **DEPOSIT:** Resources released | `game.resources[gold/wood] += carryAmount` | ✅ |
| No dropoff exists | `autoBuildDropoff` fires (line 768) | ✅ |
| Auto-build deducts cost | 100g + 50w deducted, building created at progress=0 | ✅ |
| Peasant builds it | `state='building'`, `buildTarget=b.id` | ✅ |
| Building completes | Peasant freed, `state='idle'`, **carryAmount preserved** | ✅ |
| **Same-frame deposit** | Completed building is now a valid dropoff → deposit fires immediately | ✅ |
| Retry loop | Every ~2s, idle peasants with carryAmount > 0 retry find/auto-build | ✅ |
| Edge: cooldown | 15s per-type cooldown prevents duplicate auto-builds | ✅ |
| Edge: under construction | `hasDropoffUnderConstruction` prevents building a second one | ✅ |
| Edge: insufficient resources | Cooldown set to `time+3`, carryAmount preserved — peasant retries later | ✅ |

### Findings by Severity

**🟢 LOW — No blocking bugs found.** The gathering → carry → deposit → auto-build chain is solid.

**🟡 NOTES (non-blocking observations):**
1. **Deposit range is exactly 40px** — peasant must be within 40px of dropoff center. Works for all building sizes but tight for larger buildings (e.g., Town Hall at 96x96 has a 48px half-width; peasant standing at its edge is 48px away, not depositing).
2. **Auto-build site search is O(MAP_COLS × MAP_ROWS) on fallback** — the expanding-ring search (3→5→7→10) is efficient, but the full-map fallback at the end could be slow with many entities. Not a practical concern for 64×64 maps.
3. **Per-frame double-spend lock** — `_autoBuildThisFrame` Set is used correctly, initialized in `initGame()` before game loop starts.

### What I added to `test_game.js` (Section 19 — "QA Core: Resource Release & Auto-Build Chain"):

10 scenario tests covering:
- **A**: Deposit gold at Town Hall (universal dropoff)
- **B**: Deposit wood at Lumber Mill (specialized dropoff)
- **C**: Full chain: auto-build refinery → complete → deposit gold
- **D**: Full chain: auto-build lumber mill → complete → deposit wood
- **E**: Auto-build failure (insufficient resources) → carryAmount preserved
- **F**: Two peasants auto-build different dropoffs simultaneously
- **G**: Town Hall accepts both gold AND wood
- **H**: Net resource effect after auto-build + multiple deposits
- **I**: Type rejection — wood at refinery, gold at lumber mill
- **J**: Constructing dropoff rejects deposits until complete (`progress < 1`)

Report saved to: **`.reports/QA_REPORT_R7.md`**