Everything is verified. Here's a summary of the implementation:

---

## Summary: Auto-Build Dropoff Buildings

### What was built

Implemented a complete auto-build system for the RealmCraft RTS game so that when peasants gather gold or wood and no dropoff building exists to deposit resources, the game automatically builds the appropriate dropoff building (refinery for gold, lumber mill for wood) instead of discarding the resources.

### Changes made to `index.html`

**1. New game state field** (line 137):
- `autoBuildCooldown: {}` — prevents multiple peasants from auto-building the same building type simultaneously

**2. Four new functions** inserted after `findNearestDropoff` (lines 719–802):

| Function | Purpose |
|----------|---------|
| `getDropoffBuildingType(resType)` | Maps gold→refinery, wood→lumber_mill |
| `hasDropoffUnderConstruction(resType)` | Returns true if a player-owned matching building is still being built (progress < 1) |
| `findAutoBuildSite(buildingType, nearX, nearY)` | Searches expanding ring perimeters (3→5→7→10 tiles) then falls back to full-map search for a valid placement spot |
| `autoBuildDropoff(peasant)` | Orchestrates: checks existing dropoff → checks under construction → checks cooldown → checks affordability → finds site → deducts resources → creates building → assigns peasant → shows feedback |

**3. Gathering complete handler** (~line 1235): Changed from discarding resources (`carryAmount=0`) to calling `autoBuildDropoff(e)` — if it fails, the peasant goes idle but **preserves** carried resources for retry

**4. Deposit arrival handler** (~line 1269): Same change — calls `autoBuildDropoff(e)` instead of discarding resources

**5. Idle retry loop** (~line 1300–1311): Peasants who are idle but still carrying resources check every ~2 seconds for an existing dropoff or attempt auto-build. This handles: resources becoming available later, someone else finishing construction, etc.

### How it was verified

- **JavaScript syntax**: Extracted the `<script>` block and ran `node --check` — passed cleanly, zero syntax errors
- **No duplicate functions**: Verified all 46 function declarations are unique
- **Three hook points correctly wired**: `autoBuildDropoff` is called from both resource-discard sites and the idle retry loop
- **Cooldown mechanism**: Present at game state init, checked, and set in `autoBuildDropoff`
- **All edge cases from the plan covered**: dual-trigger call pattern preserves resources on failure, cooldown prevents duplicate builds, under-construction detection avoids redundant building, affordability gate prevents impossible builds, periodic retry ensures eventual resolution