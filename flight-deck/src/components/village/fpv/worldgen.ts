// The village becomes blocks (FPV plan Phase 1): a deterministic build of
// the SAME layout the isometric map draws — tiles, footprints, doors,
// roads, homes, props — so the first-person world and the 2D map can never
// disagree. 1 block = 5 village units → 1 tile = 4×4 blocks → the plot is
// a 200×200-block world, ringed by a grassy margin so the edge never shows
// the void. Flat elevation (v1).

import type { VillageMapData, VillagePlace } from '../../../services/beings'
import { folderFor } from '../places'
import { B } from './textures'

const TILE = 20                    // village units per tile
const UNITS_PER_BLOCK = 5          // village units per block
export const TPB = TILE / UNITS_PER_BLOCK // blocks per tile (4)
const GRID = 50                    // tiles per side
const PLOT_B = GRID * TPB          // 200 blocks per side
export const MARGIN = 16           // grassy blocks beyond the plot edge
export const W = PLOT_B + MARGIN * 2
export const D = PLOT_B + MARGIN * 2
export const H = 24
export const SURFACE = 3           // first air block — feet stand here

// village units → world block coordinate (array space, margin included)
export const toBlock = (units: number) => units / UNITS_PER_BLOCK + MARGIN

export interface WorldLabel {
  x0: number; z0: number; x1: number; z1: number   // block rect, inclusive
  name: string
}

// A reading stand (FPV plan Phase 4): stands inside a building whose work
// lives in a folder (Library → reports, Garden → garden pages, …). Its
// block position + the place it belongs to feed both the lectern prop and
// the proximity 'press R to read' prompt.
export interface Lectern {
  placeId: string
  placeName: string   // "the Library"
  label: string       // "the reading room" (from folderFor)
  bx: number; bz: number   // block-space stand position
}

export interface BuiltWorld {
  blocks: Uint8Array
  labels: WorldLabel[]
  lecterns: Lectern[]
  spawn: { x: number; y: number; z: number; yaw: number }
  get: (x: number, y: number, z: number) => number
}

const idx = (x: number, y: number, z: number) => (y * D + z) * W + x

// Mirrors the backend's footprint clamp (being_world._tiles_at) — same
// math the isometric map replicates in footNW.
function footNW(p: VillagePlace): [number, number, number, number] {
  const w = Math.max(1, p.w || 1), h = Math.max(1, p.h || 1)
  const cx = Math.floor(p.x / TILE), cy = Math.floor(p.y / TILE)
  const tx0 = Math.min(Math.max(1, cx - Math.floor(w / 2)), GRID - 1 - w)
  const ty0 = Math.min(Math.max(1, cy - Math.floor(h / 2)), GRID - 1 - h)
  return [tx0, ty0, w, h]
}

// …and the backend's home_tiles clamp: every cottage is 2×2 tiles.
function homeNW(hxy: [number, number]): [number, number] {
  return [Math.min(Math.max(0, Math.floor(hxy[0] / TILE)), GRID - 2),
          Math.min(Math.max(0, Math.floor(hxy[1] / TILE)), GRID - 2)]
}

// The same look table the 2D map uses (buildings.spriteForPlace): default
// places by id, everything the village raises later by first affordance.
const SPRITE_BY_ID: Record<string, string> = {
  square: 'plaza', library: 'library', workshop: 'workshop', garden: 'garden',
  well: 'well', meadow: 'meadow', 'old-bench': 'bench',
}
const SPRITE_BY_AFFORDANCE: Record<string, string> = {
  read: 'library', create: 'workshop', gather: 'pavilion', trade: 'stall',
  tend: 'garden', play: 'pond', rest: 'bench', remember: 'cairn',
}
const spriteFor = (p: VillagePlace) =>
  SPRITE_BY_ID[p.id] || SPRITE_BY_AFFORDANCE[(p.affordances || [])[0] || ''] || 'cottage'

// Home as your canvas (world-shaping plan Phase 4): the cottage dress
// vocabulary → blocks. Unknown/unset falls back to the classic cottage.
const HOME_ROOF_BLOCKS: Record<string, number> = {
  ember: B.ROOF_RED, slate: B.ROOF_SLATE, moss: B.ROOF_MOSS,
  dusk: B.ROOF_DARK,
}
const HOME_WALL_BLOCKS: Record<string, number> = {
  plaster: B.PLASTER, timber: B.TIMBER, sage: B.WALL_SAGE,
}

// deterministic per-position hash for margin trees and meadow flowers
function hash2(x: number, z: number, s: number): number {
  let h = (x * 374761393 + z * 668265263 + s * 974711) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

export function buildWorld(data: VillageMapData): BuiltWorld {
  const blocks = new Uint8Array(W * H * D)
  const labels: WorldLabel[] = []
  const lecterns: Lectern[] = []
  const set = (x: number, y: number, z: number, id: number) => {
    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) blocks[idx(x, y, z)] = id
  }
  const get = (x: number, y: number, z: number): number => {
    if (y < 0) return B.STONE
    if (y >= H) return B.AIR
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z)
    // beyond the margin the world quietly refuses: an invisible wall
    if (xi < 0 || xi >= W || zi < 0 || zi >= D) return B.STONE
    return blocks[idx(xi, yi, zi)]
  }

  // ── the ground: two of dirt, one of grass, everywhere ──────────────────
  for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) {
    set(x, 0, z, B.DIRT); set(x, 1, z, B.DIRT); set(x, 2, z, B.GRASS)
  }

  // margin woods — sparse deterministic trees so "beyond" reads as forest
  for (let z = 2; z < D - 2; z++) for (let x = 2; x < W - 2; x++) {
    const inPlot = x >= MARGIN && x < W - MARGIN && z >= MARGIN && z < D - MARGIN
    if (inPlot) continue
    if (hash2(x, z, 77) < 0.02) tree(x, z, 3 + ((hash2(x, z, 78) * 2) | 0))
  }

  // ── streets: every road tile becomes 4×4 packed earth ──────────────────
  const roads = new Set<string>()
  for (const t of data.roads ?? []) {
    roads.add(`${t[0]},${t[1]}`)
    fillTiles(t[0], t[1], 1, 1, (x, z) => set(x, 2, z, B.PATH))
  }

  // ── grounds & buildings ────────────────────────────────────────────────
  for (const p of data.places) {
    const [tx0, ty0, tw, th] = footNW(p)
    const bx0 = tx0 * TPB + MARGIN, bz0 = ty0 * TPB + MARGIN
    const bx1 = bx0 + tw * TPB - 1, bz1 = bz0 + th * TPB - 1
    labels.push({ x0: bx0, z0: bz0, x1: bx1, z1: bz1, name: `the ${p.name.replace(/^the /i, '')}` })
    // a place whose work lives in a folder gets a reading stand you can
    // step up to — set just inside a corner so it never blocks the door
    const fmap = folderFor(p)
    if (fmap) {
      lecterns.push({
        placeId: p.id, placeName: `the ${p.name.replace(/^the /i, '')}`,
        label: fmap.label, bx: bx0 + 1, bz: bz0 + 1,
      })
    }
    const sprite = spriteFor(p)
    switch (sprite) {
      case 'plaza':
        rect(bx0, bz0, bx1, bz1, (x, z) => set(x, 2, z, B.PLAZA))
        break
      case 'garden': {
        rect(bx0, bz0, bx1, bz1, (x, z) =>
          set(x, 2, z, (z - bz0) % 2 === 0 ? B.SOIL : B.FLOWERS))
        fence(bx0, bz0, bx1, bz1, set)
        break
      }
      case 'meadow':
        rect(bx0, bz0, bx1, bz1, (x, z) =>
          set(x, 2, z, hash2(x, z, 9) < 0.18 ? B.FLOWERS : B.MEADOW))
        break
      case 'pond': {
        rect(bx0, bz0, bx1, bz1, (x, z) => set(x, 2, z, B.MEADOW))
        const cx = (bx0 + bx1) >> 1, cz = (bz0 + bz1) >> 1
        rect(cx - 2, cz - 1, cx + 2, cz + 1, (x, z) => set(x, 2, z, B.WATER))
        break
      }
      case 'well': {
        // a stone ring with dark water, four posts, a little red roof
        const cx = (bx0 + bx1) >> 1, cz = (bz0 + bz1) >> 1
        rect(cx - 1, cz - 1, cx + 2, cz + 2, (x, z) => {
          const edge = x === cx - 1 || x === cx + 2 || z === cz - 1 || z === cz + 2
          if (edge) set(x, 3, z, B.STONE)
          else set(x, 2, z, B.WATER)
        })
        for (const [px, pz] of [[cx - 1, cz - 1], [cx + 2, cz - 1], [cx - 1, cz + 2], [cx + 2, cz + 2]] as const) {
          set(px, 4, pz, B.POST); set(px, 5, pz, B.POST)
        }
        rect(cx - 1, cz - 1, cx + 2, cz + 2, (x, z) => set(x, 6, z, B.ROOF_RED))
        break
      }
      case 'bench': {
        const cx = (bx0 + bx1) >> 1, cz = (bz0 + bz1) >> 1
        set(cx, 3, cz, B.PLANK); set(cx + 1, 3, cz, B.PLANK)
        break
      }
      case 'cairn': {
        const cx = (bx0 + bx1) >> 1, cz = (bz0 + bz1) >> 1
        set(cx, 3, cz, B.STONE); set(cx + 1, 3, cz, B.STONE)
        set(cx, 3, cz + 1, B.STONE); set(cx, 4, cz, B.STONE)
        break
      }
      case 'stall': {
        // an open market stall: corner posts, a counter, a flat red roof
        for (const [px, pz] of [[bx0, bz0], [bx1, bz0], [bx0, bz1], [bx1, bz1]] as const) {
          set(px, 3, pz, B.POST); set(px, 4, pz, B.POST); set(px, 5, pz, B.POST)
        }
        for (let x = bx0; x <= bx1; x++) set(x, 3, bz1, B.PLANK)
        rect(bx0, bz0, bx1, bz1, (x, z) => set(x, 6, z, B.ROOF_RED))
        break
      }
      case 'pavilion': {
        rect(bx0, bz0, bx1, bz1, (x, z) => set(x, 2, z, B.PLAZA))
        for (const [px, pz] of [[bx0, bz0], [bx1, bz0], [bx0, bz1], [bx1, bz1]] as const) {
          for (let y = 3; y <= 5; y++) set(px, y, pz, B.POST)
        }
        roofOn(bx0, bz0, bx1, bz1, 6, B.ROOF_RED, 2, set)
        break
      }
      case 'library':
        house(bx0, bz0, bx1, bz1, { wall: B.TIMBER, corner: B.TIMBER, roof: B.ROOF_DARK, wallH: 5 }, p, set)
        break
      case 'workshop':
        house(bx0, bz0, bx1, bz1, { wall: B.PLANK, corner: B.TIMBER, roof: B.ROOF_RED, wallH: 4 }, p, set)
        break
      default: // cottage — anything the village raises later
        house(bx0, bz0, bx1, bz1, { wall: B.PLASTER, corner: B.TIMBER, roof: B.ROOF_RED, wallH: 4 }, p, set)
    }
  }

  // ── every being's home: a small plastered cottage, door to the lane ────
  for (const b of data.beings) {
    if (b.kind === 'visitor' || !b.home_xy) continue   // guests keep no cottage
    const [htx, hty] = homeNW(b.home_xy)
    const bx0 = htx * TPB + MARGIN, bz0 = hty * TPB + MARGIN
    const bx1 = bx0 + 2 * TPB - 1, bz1 = bz0 + 2 * TPB - 1
    labels.push({ x0: bx0, z0: bz0, x1: bx1, z1: bz1,
                  name: b.home_name ? `“${b.home_name}” — ${b.name}'s home`
                    : `${b.name}'s home` })
    // the door faces the nearest street (the home lane, in practice)
    let best: [number, number] | null = null, bestD = Infinity
    const hcx = (htx + 1) * TPB, hcz = (hty + 1) * TPB
    for (const key of roads) {
      const [rx, rz] = key.split(',').map(Number)
      const d0 = Math.abs(rx * TPB + 2 - hcx) + Math.abs(rz * TPB + 2 - hcz)
      if (d0 < bestD) { bestD = d0; best = [rx * TPB + 2, rz * TPB + 2] }
    }
    const side = !best ? 'e'
      : Math.abs(best[0] - hcx) >= Math.abs(best[1] - hcz)
        ? (best[0] > hcx ? 'e' : 'w') : (best[1] > hcz ? 's' : 'n')
    // Home as your canvas (world-shaping plan Phase 4): the cottage
    // wears the being's chosen dress — roof and wall from the vocab.
    const look = b.home_look || {}
    const roofBlock = HOME_ROOF_BLOCKS[look.roof || ''] ?? B.ROOF_RED
    const wallBlock = HOME_WALL_BLOCKS[look.wall || ''] ?? B.PLASTER
    cottage(bx0, bz0, bx1, bz1, side, set, wallBlock, roofBlock)
  }

  // ── props: trees, bushes, flowers, lamps (same seeded payload) ─────────
  for (const pr of data.props ?? []) {
    const bx = pr.tile[0] * TPB + MARGIN, bz = pr.tile[1] * TPB + MARGIN
    const cx = bx + 1, cz = bz + 1
    if (pr.kind === 'tree') tree(cx, cz, 3 + ((hash2(cx, cz, 5) * 2) | 0))
    else if (pr.kind === 'bush') { set(cx, 3, cz, B.LEAVES); set(cx + 1, 3, cz, B.LEAVES); set(cx, 3, cz + 1, B.LEAVES) }
    else if (pr.kind === 'flowers') fillTiles(pr.tile[0], pr.tile[1], 1, 1, (x, z) => { if (hash2(x, z, 6) < 0.5) set(x, 2, z, B.FLOWERS) })
    else if (pr.kind === 'lamp') {
      // on the tile corner so the street stays walkable
      if (get(bx, 2, bz) !== B.AIR) { set(bx, 3, bz, B.POST); set(bx, 4, bz, B.POST); set(bx, 5, bz, B.LAMP) }
    }
  }

  // ── made things (world-shaping plan Phase 3): a being's placed works ───
  // Small block fixtures on their tile. Blocking kinds (cairn, sculpture,
  // fountain, shrine) fill enough of the tile to read as an obstacle —
  // parity with walk_blocked; the rest sit light and walkable-around.
  for (const o of data.objects ?? []) {
    const bx = o.tile[0] * TPB + MARGIN, bz = o.tile[1] * TPB + MARGIN
    const cx = bx + 1, cz = bz + 1
    // A beginning (instinct-build plan): the feet broke ground, the mind
    // hasn't finished it — a walkable work-site of turned soil, never a
    // solid obstacle (parity with walk_blocked, which excludes stakes).
    if (o.staked) {
      set(cx, 2, cz, B.SOIL); set(cx + 1, 2, cz, B.SOIL)
      set(cx, 2, cz + 1, B.SOIL)
      labels.push({ x0: bx, z0: bz, x1: bx + TPB - 1, z1: bz + TPB - 1,
                    name: `a beginning — a ${o.kind}` })
      continue
    }
    labels.push({ x0: bx, z0: bz, x1: bx + TPB - 1, z1: bz + TPB - 1,
                  name: `“${o.name}”` })
    switch (o.kind) {
      case 'bench':
        set(cx, 3, cz, B.PLANK); set(cx + 1, 3, cz, B.PLANK)
        break
      case 'signpost':
        set(cx, 3, cz, B.POST); set(cx, 4, cz, B.POST)
        set(cx, 5, cz, B.PLANK)
        break
      case 'planter':
        set(cx, 3, cz, B.PLANK); set(cx + 1, 3, cz, B.PLANK)
        set(cx, 4, cz, B.LEAVES); set(cx + 1, 4, cz, B.FLOWERS)
        break
      case 'lantern':
        set(cx, 3, cz, B.POST); set(cx, 4, cz, B.POST)
        set(cx, 5, cz, B.LAMP)
        break
      case 'cairn':
        set(cx, 3, cz, B.STONE); set(cx + 1, 3, cz, B.STONE)
        set(cx, 3, cz + 1, B.STONE); set(cx, 4, cz, B.STONE)
        break
      case 'sculpture':
        set(cx, 3, cz, B.STONE); set(cx, 4, cz, B.STONE)
        set(cx, 5, cz, B.STONE); set(cx + 1, 3, cz, B.STONE)
        break
      case 'fountain':
        rect(cx - 1, cz - 1, cx + 2, cz + 2, (x, z) => {
          const edge = x === cx - 1 || x === cx + 2 || z === cz - 1 || z === cz + 2
          if (edge) set(x, 3, z, B.STONE)
          else set(x, 2, z, B.WATER)
        })
        break
      case 'shrine':
        set(cx, 3, cz, B.STONE); set(cx, 4, cz, B.STONE)
        set(cx + 1, 3, cz, B.STONE); set(cx + 1, 4, cz, B.STONE)
        set(cx, 5, cz, B.ROOF_RED); set(cx + 1, 5, cz, B.ROOF_RED)
        set(cx, 3, cz + 1, B.LAMP)
        break
      default:
        set(cx, 3, cz, B.STONE)
    }
  }

  // ── spawn: a few steps south of the square, facing it ──────────────────
  const square = data.places.find((p) => p.id === 'square')
    || data.places.find((p) => (p.affordances || []).includes('gather'))
    || data.places[0]
  let spawn = { x: W / 2, y: SURFACE, z: D / 2 + 8, yaw: 0 }
  if (square) {
    const [tx0, ty0, tw, th] = footNW(square)
    spawn = {
      x: (tx0 + tw / 2) * TPB + MARGIN,
      y: SURFACE,
      z: (ty0 + th) * TPB + MARGIN + 5,
      yaw: 0, // looking north (−z) — at the square
    }
  }

  return { blocks, labels, lecterns, spawn, get }

  // ── local builders ─────────────────────────────────────────────────────
  function rect(x0: number, z0: number, x1: number, z1: number,
                fn: (x: number, z: number) => void) {
    for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) fn(x, z)
  }
  function fillTiles(tx: number, ty: number, tw: number, th: number,
                     fn: (x: number, z: number) => void) {
    rect(tx * TPB + MARGIN, ty * TPB + MARGIN,
         (tx + tw) * TPB + MARGIN - 1, (ty + th) * TPB + MARGIN - 1, fn)
  }
  function tree(cx: number, cz: number, trunkH: number) {
    for (let y = 0; y < trunkH; y++) set(cx, 3 + y, cz, B.TRUNK)
    const top = 3 + trunkH
    for (let dy = -1; dy <= 1; dy++) for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) {
      const r = dy === 1 ? 1 : 2
      if (Math.abs(dx) > r || Math.abs(dz) > r) continue
      if (Math.abs(dx) === r && Math.abs(dz) === r && hash2(cx + dx, cz + dz, 8) < 0.5) continue
      if (get(cx + dx, top + dy, cz + dz) === B.AIR) set(cx + dx, top + dy, cz + dz, B.LEAVES)
    }
    set(cx, top + 2, cz, B.LEAVES)
  }
  function fence(x0: number, z0: number, x1: number, z1: number,
                 s: typeof set) {
    const midX = (x0 + x1) >> 1, midZ = (z0 + z1) >> 1
    for (let x = x0; x <= x1; x++) {
      if (Math.abs(x - midX) > 1) { s(x, 3, z0, B.FENCE); s(x, 3, z1, B.FENCE) }
    }
    for (let z = z0; z <= z1; z++) {
      if (Math.abs(z - midZ) > 1) { s(x0, 3, z, B.FENCE); s(x1, 3, z, B.FENCE) }
    }
  }
  // a hip roof: overhangs the walls by one, each layer steps in, flat top
  function roofOn(x0: number, z0: number, x1: number, z1: number, y0: number,
                  roof: number, maxLayers: number, s: typeof set) {
    let rx0 = x0 - 1, rz0 = z0 - 1, rx1 = x1 + 1, rz1 = z1 + 1
    for (let i = 0; i < maxLayers && rx1 >= rx0 && rz1 >= rz0; i++) {
      rect(rx0, rz0, rx1, rz1, (x, z) => s(x, y0 + i, z, roof))
      rx0++; rz0++; rx1--; rz1--
    }
  }
  function house(x0: number, z0: number, x1: number, z1: number,
                 kit: { wall: number; corner: number; roof: number; wallH: number },
                 p: VillagePlace, s: typeof set) {
    // plank floor over the whole footprint
    rect(x0, z0, x1, z1, (x, z) => s(x, 2, z, B.PLANK))
    // perimeter walls, corners in timber
    rect(x0, z0, x1, z1, (x, z) => {
      const edge = x === x0 || x === x1 || z === z0 || z === z1
      if (!edge) return
      const corner = (x === x0 || x === x1) && (z === z0 || z === z1)
      for (let y = 0; y < kit.wallH; y++) s(x, 3 + y, z, corner ? kit.corner : kit.wall)
    })
    // windows: one row at eye height, every 4th block, never at corners
    const wy = 4
    for (let x = x0 + 2; x <= x1 - 2; x += 4) { s(x, wy, z0, B.WINDOW); s(x, wy, z1, B.WINDOW) }
    for (let z = z0 + 2; z <= z1 - 2; z += 4) { s(x0, wy, z, B.WINDOW); s(x1, wy, z, B.WINDOW) }
    // the door: a 2×3 opening on the outer face of the door tile
    if (p.door_x != null && p.door_y != null) {
      const dbx = p.door_x * TPB + MARGIN, dbz = p.door_y * TPB + MARGIN
      // door tile sits on a footprint edge; find which edge it touches
      let wall: 'n' | 's' | 'w' | 'e' | null = null
      if (dbz <= z0) wall = 'n'
      else if (dbz + TPB - 1 >= z1) wall = 's'
      else if (dbx <= x0) wall = 'w'
      else if (dbx + TPB - 1 >= x1) wall = 'e'
      const carve = (x: number, z: number) => { for (let y = 3; y <= 5; y++) s(x, y, z, B.AIR) }
      if (wall === 'n') { carve(dbx + 1, z0); carve(dbx + 2, z0) }
      else if (wall === 's') { carve(dbx + 1, z1); carve(dbx + 2, z1) }
      else if (wall === 'w') { carve(x0, dbz + 1); carve(x0, dbz + 2) }
      else if (wall === 'e') { carve(x1, dbz + 1); carve(x1, dbz + 2) }
    }
    roofOn(x0, z0, x1, z1, 3 + kit.wallH, kit.roof, 4, s)
  }
  function cottage(x0: number, z0: number, x1: number, z1: number,
                   side: string, s: typeof set,
                   wall: number = B.PLASTER, roof: number = B.ROOF_RED) {
    const wallH = 3
    rect(x0, z0, x1, z1, (x, z) => s(x, 2, z, B.PLANK))
    rect(x0, z0, x1, z1, (x, z) => {
      const edge = x === x0 || x === x1 || z === z0 || z === z1
      if (!edge) return
      const corner = (x === x0 || x === x1) && (z === z0 || z === z1)
      for (let y = 0; y < wallH; y++) s(x, 3 + y, z, corner ? B.TIMBER : wall)
    })
    const midX = (x0 + x1) >> 1, midZ = (z0 + z1) >> 1
    const carve = (x: number, z: number) => { s(x, 3, z, B.AIR); s(x, 4, z, B.AIR) }
    if (side === 'e') { carve(x1, midZ); carve(x1, midZ + 1); s(x0, 4, midZ, B.WINDOW) }
    else if (side === 'w') { carve(x0, midZ); carve(x0, midZ + 1); s(x1, 4, midZ, B.WINDOW) }
    else if (side === 's') { carve(midX, z1); carve(midX + 1, z1); s(midX, 4, z0, B.WINDOW) }
    else { carve(midX, z0); carve(midX + 1, z0); s(midX, 4, z1, B.WINDOW) }
    roofOn(x0, z0, x1, z1, 3 + wallH, roof, 4, s)
  }
}
