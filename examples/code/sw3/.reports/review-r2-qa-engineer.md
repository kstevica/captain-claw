# QA Engineer — review r2

The report is saved to `.reports/QA_REPORT_R6.md`. Here's the summary:

---

## QA Assessment — Round 6

**291 tests, 0 failures. JavaScript syntax: clean.** All R5 gaps (findNearestDropoff, findNearestResource, placeBuilding, issueGather) now have dedicated tests. Test coverage improved substantially from 226 → 291 tests.

### 🔴 CRITICAL Finding (R6-001): Auto-build → deposit chain broken

**The deposit handler at line 1191 requires `state==='moving'`, but the movement code at line 1114 resets state to `'idle'` (when `d < 3`) *before* the deposit handler runs.** Execution order in `updateEntity`:

1. **Movement** (line 1110) — `d < 3` → `state='idle'`
2. Building/Gathering blocks
3. **Deposit handler** (line 1190) — checks `state==='moving'` → **FALSE** → missed
4. Idle retry block (line 1243) — fires every ~2s → sets `state='moving'` → next frame: movement resets again → infinite 2-second oscillation

This works for *approaching* peasants because the deposit handler catches them at `d < 40` while they're still moving (the 40px deposit window is bigger than the 3px stop threshold). But when a peasant **auto-builds a dropoff at their own location**, they're already at `d ≈ 0` — the idle retry sets `state='moving'`, movement immediately resets it, and the deposit handler **never fires**.

**The E2E test masks this** — it manually simulates `resources[p.carryType] += p.carryAmount` rather than calling the real `updateEntity`. The test passes, but the actual game loop fails.

**Fix options:** (1) move deposit handler before movement code, (2) accept `state==='idle'` in the deposit handler, or (3) handle deposit directly in the idle retry block when the peasant is already at the dropoff.

### 🟡 MEDIUM: entityAtWorld, findAutoBuildSite still untested

These were R5 findings that remain unaddressed.

### 🟢 All other verified paths correct

- Resources released correctly: `game.resources[carryType] += carryAmount` ✓
- Type discrimination: gold → refinery/townhall, wood → lumber_mill/townhall ✓
- Auto-build cooldown/affordability/construction guards ✓
- Soft-lock prevention (town_hall auto-completes with zero peasants) ✓
- issueMove/commandStop preserve carryAmount unless at dropoff ✓