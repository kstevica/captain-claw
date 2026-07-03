# RealmCraft — Architecture Overview

**RealmCraft** is a single-file (~1,788 lines) browser-based RTS game in the tradition of Warcraft / StarCraft — build structures, train units, gather gold/wood, manage food/population on a 64×64 tile map. No AI opponents, no multiplayer, no sound. Runs entirely in-browser with zero dependencies.

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | 1,788 | Complete game: HTML shell, CSS theme, all JS game logic (~46 functions, 1 main rAF loop) |
| `test_game.js` | 2,235 | Test suite (125+ tests) covering entity creation, resources, placement, combat, auto-build |
| `plan.md` | ~280 | Design document for the auto-build dropoff feature (latest addition) |
| `canvas.clientWidth` | ~50 | Fix notes: 9 fixes applied for attack, collision, gather, placement, HiDPI |
| `.reports/` | ~20 files | QA + correctness + security reviews (R0–R4), latest: 0 BLOCKING / 7 MAJOR / 9 MINOR |

## Architecture Layers (bottom-up)

### 1. Constants & Config (line ~86)
- `TILE_SIZE=32`, map 64×64 (2048×2048 px world)
- `TERRAIN` enum: GRASS(0), DIRT(1), WATER(2), TREE(3), GOLD(4)
- `BUILDING_DEFS` dict (7 buildings) and `UNIT_DEFS` dict (4 units)
- Terrain colors for game canvas and minimap

### 2. Game State Singleton (line ~135)
- `const game = {...}` — single state object containing:
  - `entities: Map<string, object>` — all units & buildings
  - `resources: { gold, wood, food, maxFood, resourcesEarned }`
  - `selectedIds: Set<string>` — multi-select support
  - `camera: { x, y, targetX, targetY, zoom }` — smooth pan+zoom
  - `ui: { mode, buildingType, attackMode }`
  - `mouse: { sx, sy, worldX, worldY, button, drag }`
  - `selectionBox: { x1, y1, x2, y2 } | null`
  - `ghostBuilding` / `particles` / `floatingTexts` / `autoBuildCooldown`

### 3. Helper Utilities (line ~153)
- `genId()`, `dist()`, `clamp()`, `lerp()`, `rectsOverlap()`
- `tileAt(x, y)`, `isPassable(x, y, w, h)`, `spawnParticles()`, `showFeedback()`

### 4. Map Generation (line ~187)
- `generateMap()` with seed 42 — deterministic procedural generation
- Grass base → dirt paths → water bodies → tree clusters → 8 gold mines

### 5. Entity Factories (line ~280)
- `createUnit(type, x, y)` and `createBuilding(type, x, y, progress)`
- Both return flat JS objects with all fields set

### 6. Rendering Pipeline (lines ~315–650)
- Canvas 2D drawing, camera-relative transforms
- Draw order: terrain → buildings → units → selection circles → UI → minimap
- 7 sub-renderers: `drawHealthBar`, `drawSelectionCircle`, `drawUnit`, `drawBuilding`, `render`, `renderMinimap`
- Units drawn as geometric shapes (no sprites) with face-direction indicators

### 7. Camera System (line ~660)
- Edge-scroll (10px zone at viewport edges)
- Middle-click drag pan
- Zoom lerp (0.5×–2.0×), clamped to world bounds
- Smooth camera with target interpolation

### 8. Input & Selection (lines ~693–850)
- `screenToWorld` / `worldToScreen` coordinate transforms (HiDPI-aware)
- `entityAtWorld` click hit-test
- `entitiesInRect` box-selection
- `findNearestDropoff`, `findNearestResource`, `isPlacementValid`
- **Auto-build dropoff system** (lines ~760–840): `getDropoffBuildingType`, `hasDropoffUnderConstruction`, `findAutoBuildSite`, `autoBuildDropoff` — peasants auto-build refinery/lumber mill when no dropoff exists

### 9. Game Logic: update(dt) (lines ~950–1310)
- Per-entity update: movement steering, resource gathering (mine→return→deposit), building construction progress, training queue, attack (target selection + cooldown), death/cleanup, particle lifecycle
- Idle retry loop (~1300): peasants carrying resources retry auto-build every ~2s
- Delta-time capped at 50ms

### 10. UI & Events (lines ~1310–1788)
- `updateUI()` rebuilds info/action panels on selection change
- `resizeCanvas()` — HiDPI-aware responsive resize with `devicePixelRatio`
- Mouse: mousedown/move/up/click/contextmenu/wheel handlers
- Keyboard: A (attack-move), S (stop), H (hold), Escape (deselect/cancel build)
- Minimap click-to-pan
- `initGame()` bootstrap + `gameLoop()` requestAnimationFrame loop

## Key Design Patterns
- **Flat JS objects** for entities (no classes, no ECS, no OOP inheritance)
- **Map<string, Entity>** for entity storage — O(1) lookups, simple iteration
- **Camera-relative rendering** via `ctx.translate + ctx.scale` on each frame
- **Delta-time game loop** — rAF with dt cap at 50ms, fixed timestep for simulation
- **State machine** per entity: idle/moving/attacking/gathering/building/constructing
- **Procedural geometric graphics** — all visuals drawn with canvas primitives (no spritesheets, no images)
- **Auto-build on demand** — peasants automatically construct missing dropoff buildings when carrying resources

## Known Technical Debt
- `game.trainingBuildings` and `game.trainingQueue` arrays initialized but never used (dead code)
- `hasDropoffUnderConstruction` param `resType` unused when `bldType` is null (minor)
- Attack collision groups have some inconsistencies with building interactions
