// The village in 16 pixels (FPV plan Phase 1): a procedural texture atlas
// in the same warm storybook palette as the isometric map, painted once on
// a canvas — no assets, no network, deterministic (fixed seed, so the
// world looks identical on every visit). 8×8 tiles of 16px.

import * as THREE from 'three'

// ── block ids ────────────────────────────────────────────────────────────
export const B = {
  AIR: 0, GRASS: 1, DIRT: 2, PATH: 3, PLAZA: 4, SOIL: 5, PLANK: 6,
  TIMBER: 7, PLASTER: 8, STONE: 9, ROOF_RED: 10, ROOF_DARK: 11, TRUNK: 12,
  LEAVES: 13, FENCE: 14, MEADOW: 15, FLOWERS: 16, WINDOW: 17, POST: 18,
  LAMP: 19, WATER: 20,
  // Home as your canvas (world-shaping plan Phase 4): the cottage
  // dresses a being may choose — two more roofs and a sage wall.
  ROOF_SLATE: 21, ROOF_MOSS: 22, WALL_SAGE: 23,
} as const

// tile indices into the atlas
const T = {
  GRASS_TOP: 0, GRASS_SIDE: 1, DIRT: 2, PATH: 3, PLAZA: 4, SOIL: 5,
  PLANK: 6, TIMBER: 7, PLASTER: 8, STONE: 9, ROOF_RED: 10, ROOF_DARK: 11,
  TRUNK_SIDE: 12, TRUNK_TOP: 13, LEAVES: 14, FENCE: 15, MEADOW: 16,
  FLOWERS: 17, WINDOW: 18, POST: 19, LAMP: 20, WATER: 21,
  ROOF_SLATE: 22, ROOF_MOSS: 23, WALL_SAGE: 24,
}

// id → [top, side, bottom] tiles
export const BLOCK_TILES: Record<number, [number, number, number]> = {
  [B.GRASS]: [T.GRASS_TOP, T.GRASS_SIDE, T.DIRT],
  [B.DIRT]: [T.DIRT, T.DIRT, T.DIRT],
  [B.PATH]: [T.PATH, T.DIRT, T.DIRT],
  [B.PLAZA]: [T.PLAZA, T.PLAZA, T.PLAZA],
  [B.SOIL]: [T.SOIL, T.DIRT, T.DIRT],
  [B.PLANK]: [T.PLANK, T.PLANK, T.PLANK],
  [B.TIMBER]: [T.TIMBER, T.TIMBER, T.TIMBER],
  [B.PLASTER]: [T.PLASTER, T.PLASTER, T.PLASTER],
  [B.STONE]: [T.STONE, T.STONE, T.STONE],
  [B.ROOF_RED]: [T.ROOF_RED, T.ROOF_RED, T.ROOF_RED],
  [B.ROOF_DARK]: [T.ROOF_DARK, T.ROOF_DARK, T.ROOF_DARK],
  [B.TRUNK]: [T.TRUNK_TOP, T.TRUNK_SIDE, T.TRUNK_TOP],
  [B.LEAVES]: [T.LEAVES, T.LEAVES, T.LEAVES],
  [B.FENCE]: [T.FENCE, T.FENCE, T.FENCE],
  [B.MEADOW]: [T.MEADOW, T.GRASS_SIDE, T.DIRT],
  [B.FLOWERS]: [T.FLOWERS, T.GRASS_SIDE, T.DIRT],
  [B.WINDOW]: [T.WINDOW, T.WINDOW, T.WINDOW],
  [B.POST]: [T.POST, T.POST, T.POST],
  [B.LAMP]: [T.LAMP, T.LAMP, T.LAMP],
  [B.WATER]: [T.WATER, T.WATER, T.WATER],
  [B.ROOF_SLATE]: [T.ROOF_SLATE, T.ROOF_SLATE, T.ROOF_SLATE],
  [B.ROOF_MOSS]: [T.ROOF_MOSS, T.ROOF_MOSS, T.ROOF_MOSS],
  [B.WALL_SAGE]: [T.WALL_SAGE, T.WALL_SAGE, T.WALL_SAGE],
}

export const isSolid = (id: number) => id !== B.AIR && id !== B.WATER
// lamps render in the unlit pass so they carry the night by themselves
export const isGlow = (id: number) => id === B.LAMP

// ── deterministic paint ──────────────────────────────────────────────────
function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type Ctx = CanvasRenderingContext2D
type Rng = () => number

function speckle(c: Ctx, rng: Rng, base: [number, number, number], vary: number) {
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    const m = 1 + (rng() - 0.5) * vary
    c.fillStyle = `rgb(${(base[0] * m) | 0},${(base[1] * m) | 0},${(base[2] * m) | 0})`
    c.fillRect(x, y, 1, 1)
  }
}

const ATLAS_TILES = 8 // 8×8 grid of 16px tiles

export function buildAtlas(): { texture: THREE.CanvasTexture; uvFor: (tile: number) => [number, number] } {
  const cv = document.createElement('canvas')
  cv.width = cv.height = ATLAS_TILES * 16
  const ctx = cv.getContext('2d')!

  const paint = (t: number, fn: (c: Ctx, r: Rng) => void) => {
    const rng = mulberry32(0x15EA ^ (t * 7919))
    ctx.save()
    ctx.translate((t % ATLAS_TILES) * 16, Math.floor(t / ATLAS_TILES) * 16)
    fn(ctx, rng)
    ctx.restore()
  }

  // grass — the warm meadow green of the 2D map
  paint(T.GRASS_TOP, (c, r) => {
    speckle(c, r, [154, 186, 118], 0.16)
    for (let i = 0; i < 10; i++) { c.fillStyle = 'rgba(110,146,82,.5)'; c.fillRect((r() * 16) | 0, (r() * 16) | 0, 1, 1) }
  })
  paint(T.DIRT, (c, r) => {
    speckle(c, r, [158, 126, 94], 0.2)
    for (let i = 0; i < 6; i++) { c.fillStyle = 'rgba(110,84,58,.6)'; c.fillRect((r() * 16) | 0, (r() * 16) | 0, 2, 1) }
  })
  paint(T.GRASS_SIDE, (c, r) => {
    speckle(c, r, [158, 126, 94], 0.2)
    for (let x = 0; x < 16; x++) {
      const d = 3 + ((r() * 3) | 0)
      for (let y = 0; y < d; y++) { const m = 1 + (r() - 0.5) * 0.2; c.fillStyle = `rgb(${(150 * m) | 0},${(182 * m) | 0},${(114 * m) | 0})`; c.fillRect(x, y, 1, 1) }
    }
  })
  // packed-earth street — the road color of the 2D map
  paint(T.PATH, (c, r) => {
    speckle(c, r, [205, 176, 132], 0.13)
    for (let i = 0; i < 7; i++) { c.fillStyle = 'rgba(170,140,100,.55)'; c.fillRect((r() * 15) | 0, (r() * 15) | 0, 1 + ((r() * 2) | 0), 1) }
  })
  // plaza cobbles
  paint(T.PLAZA, (c, r) => {
    speckle(c, r, [196, 180, 152], 0.1)
    for (let i = 0; i < 8; i++) {
      const x = (r() * 12) | 0, y = (r() * 12) | 0, w = 2 + ((r() * 3) | 0), h = 2 + ((r() * 2) | 0)
      const m = 0.88 + r() * 0.24
      c.fillStyle = `rgb(${(200 * m) | 0},${(184 * m) | 0},${(156 * m) | 0})`
      c.fillRect(x, y, w, h)
      c.strokeStyle = 'rgba(120,106,84,.55)'; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1)
    }
  })
  // tilled garden soil — dark rows
  paint(T.SOIL, (c, r) => {
    speckle(c, r, [126, 96, 66], 0.16)
    for (let y = 1; y < 16; y += 4) { c.fillStyle = 'rgba(84,60,40,.7)'; c.fillRect(0, y, 16, 1) }
    for (let i = 0; i < 4; i++) { c.fillStyle = 'rgba(150,190,120,.7)'; c.fillRect((r() * 15) | 0, (r() * 15) | 0, 1, 1) }
  })
  // planks — honey wood with seams
  paint(T.PLANK, (c, r) => {
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const m = 1 + (r() - 0.5) * 0.12
      const seam = (y % 4 === 3) || ((x === 7 || x === 15) && Math.floor(y / 4) % 2 === 0) || ((x === 3 || x === 11) && Math.floor(y / 4) % 2 === 1)
      const k = seam ? 0.6 : 1
      c.fillStyle = `rgb(${(206 * m * k) | 0},${(170 * m * k) | 0},${(118 * m * k) | 0})`
      c.fillRect(x, y, 1, 1)
    }
  })
  // timber — darker vertical grain (the library wears this)
  paint(T.TIMBER, (c, r) => {
    for (let x = 0; x < 16; x++) {
      const sh = x % 4 === 0 ? 0.68 : (r() < 0.25 ? 0.85 : 1)
      for (let y = 0; y < 16; y++) { const m = (1 + (r() - 0.5) * 0.15) * sh; c.fillStyle = `rgb(${(150 * m) | 0},${(114 * m) | 0},${(78 * m) | 0})`; c.fillRect(x, y, 1, 1) }
    }
  })
  // plaster — warm whitewash with flecks
  paint(T.PLASTER, (c, r) => {
    speckle(c, r, [235, 226, 206], 0.06)
    for (let i = 0; i < 5; i++) { c.fillStyle = 'rgba(196,182,152,.5)'; c.fillRect((r() * 16) | 0, (r() * 16) | 0, 1, 1) }
  })
  // stone — warm grey blocks
  paint(T.STONE, (c, r) => {
    speckle(c, r, [172, 166, 152], 0.12)
    for (let i = 0; i < 7; i++) {
      const x = (r() * 13) | 0, y = (r() * 13) | 0, w = 2 + ((r() * 3) | 0), h = 2 + ((r() * 2) | 0), m = 0.82 + r() * 0.4
      c.fillStyle = `rgb(${(176 * m) | 0},${(170 * m) | 0},${(156 * m) | 0})`; c.fillRect(x, y, w, h)
      c.strokeStyle = 'rgba(104,98,86,.6)'; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1)
    }
  })
  // terracotta shingles
  paint(T.ROOF_RED, (c, r) => {
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const row = Math.floor(y / 4)
      const seam = y % 4 === 3 || (x + (row % 2 === 0 ? 0 : 4)) % 8 === 7
      const m = (1 + (r() - 0.5) * 0.14) * (seam ? 0.62 : 1)
      c.fillStyle = `rgb(${(206 * m) | 0},${(118 * m) | 0},${(86 * m) | 0})`
      c.fillRect(x, y, 1, 1)
    }
  })
  // dusk-slate shingles (the library roof)
  paint(T.ROOF_DARK, (c, r) => {
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const row = Math.floor(y / 4)
      const seam = y % 4 === 3 || (x + (row % 2 === 0 ? 0 : 4)) % 8 === 7
      const m = (1 + (r() - 0.5) * 0.14) * (seam ? 0.6 : 1)
      c.fillStyle = `rgb(${(118 * m) | 0},${(104 * m) | 0},${(140 * m) | 0})`
      c.fillRect(x, y, 1, 1)
    }
  })
  // chosen cottage dresses (world-shaping plan Phase 4): the same
  // shingle rhythm in a cool slate and a mossy green, and a sage wall
  const shingles = (base: [number, number, number]) =>
    (c: Ctx, r: Rng) => {
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
        const row = Math.floor(y / 4)
        const seam = y % 4 === 3 || (x + (row % 2 === 0 ? 0 : 4)) % 8 === 7
        const m = (1 + (r() - 0.5) * 0.14) * (seam ? 0.62 : 1)
        c.fillStyle = `rgb(${(base[0] * m) | 0},${(base[1] * m) | 0},${(base[2] * m) | 0})`
        c.fillRect(x, y, 1, 1)
      }
    }
  paint(T.ROOF_SLATE, shingles([110, 128, 146]))
  paint(T.ROOF_MOSS, shingles([106, 136, 84]))
  paint(T.WALL_SAGE, (c, r) => {
    speckle(c, r, [205, 214, 178], 0.07)
    for (let i = 0; i < 5; i++) { c.fillStyle = 'rgba(160,172,138,.5)'; c.fillRect((r() * 16) | 0, (r() * 16) | 0, 1, 1) }
  })
  paint(T.TRUNK_SIDE, (c, r) => {
    for (let x = 0; x < 16; x++) {
      const sh = x % 5 === 0 ? 0.7 : 1
      for (let y = 0; y < 16; y++) { const m = (1 + (r() - 0.5) * 0.18) * sh; c.fillStyle = `rgb(${(138 * m) | 0},${(106 * m) | 0},${(72 * m) | 0})`; c.fillRect(x, y, 1, 1) }
    }
  })
  paint(T.TRUNK_TOP, (c, r) => {
    speckle(c, r, [172, 138, 96], 0.12)
    c.strokeStyle = 'rgba(110,82,52,.85)'
    for (let i = 1; i < 4; i++) c.strokeRect(i + 0.5, i + 0.5, 15 - i * 2, 15 - i * 2)
  })
  paint(T.LEAVES, (c, r) => {
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const g = r()
      c.fillStyle = g < 0.1 ? 'rgba(70,102,52,1)' : `rgb(${(112 + g * 34) | 0},${(158 + g * 34) | 0},${(88 + g * 26) | 0})`
      c.fillRect(x, y, 1, 1)
    }
  })
  // picket fence — pale slats over see-through-dark gaps
  paint(T.FENCE, (c, r) => {
    speckle(c, r, [96, 116, 78], 0.2)          // hedge-dark behind the slats
    for (let x = 0; x < 16; x += 4) {
      for (let y = 0; y < 16; y++) { const m = 1 + (r() - 0.5) * 0.1; c.fillStyle = `rgb(${(226 * m) | 0},${(206 * m) | 0},${(164 * m) | 0})`; c.fillRect(x, y, 2, 1) }
    }
    c.fillStyle = 'rgba(190,168,128,.9)'; c.fillRect(0, 3, 16, 1); c.fillRect(0, 10, 16, 1)
  })
  // meadow — taller, brighter grass
  paint(T.MEADOW, (c, r) => {
    speckle(c, r, [168, 198, 126], 0.14)
    for (let i = 0; i < 12; i++) { c.fillStyle = 'rgba(126,162,90,.8)'; const x = (r() * 16) | 0, y = (r() * 14) | 0; c.fillRect(x, y, 1, 2) }
  })
  // flowers — grass with ember/blush/cream dots
  paint(T.FLOWERS, (c, r) => {
    speckle(c, r, [154, 186, 118], 0.16)
    const petals = ['#e8a04b', '#e08a6d', '#f3f7d4', '#bb9dd4']
    for (let i = 0; i < 7; i++) { c.fillStyle = petals[(r() * petals.length) | 0]; c.fillRect(1 + ((r() * 14) | 0), 1 + ((r() * 14) | 0), 1, 1) }
  })
  // window — warm glass in a timber frame
  paint(T.WINDOW, (c, r) => {
    speckle(c, r, [255, 217, 138], 0.1)
    c.fillStyle = 'rgba(255,240,200,.8)'; c.fillRect(2, 2, 5, 5)
    c.strokeStyle = '#8a6a48'; c.lineWidth = 2
    c.strokeRect(1, 1, 14, 14)
    c.fillStyle = '#8a6a48'; c.fillRect(7, 0, 2, 16); c.fillRect(0, 7, 16, 2)
  })
  paint(T.POST, (c, r) => {
    for (let x = 0; x < 16; x++) {
      const sh = x % 6 === 0 ? 0.72 : 1
      for (let y = 0; y < 16; y++) { const m = (1 + (r() - 0.5) * 0.14) * sh; c.fillStyle = `rgb(${(104 * m) | 0},${(82 * m) | 0},${(56 * m) | 0})`; c.fillRect(x, y, 1, 1) }
    }
  })
  // lamp glass — bright, painted flat; lives in the unlit pass
  paint(T.LAMP, (c, r) => {
    speckle(c, r, [255, 226, 158], 0.06)
    c.fillStyle = 'rgba(255,244,208,.95)'; c.fillRect(4, 4, 8, 8)
    c.strokeStyle = 'rgba(120,94,58,.8)'; c.strokeRect(0.5, 0.5, 15, 15)
  })
  // still water
  paint(T.WATER, (c, r) => {
    speckle(c, r, [98, 152, 176], 0.12)
    for (let i = 0; i < 5; i++) { c.fillStyle = 'rgba(190,224,238,.55)'; c.fillRect((r() * 10) | 0, (r() * 16) | 0, 4 + ((r() * 5) | 0), 1) }
  })

  const texture = new THREE.CanvasTexture(cv)
  texture.magFilter = THREE.NearestFilter
  texture.minFilter = THREE.NearestFilter
  texture.generateMipmaps = false
  texture.colorSpace = THREE.SRGBColorSpace

  const uvFor = (tile: number): [number, number] =>
    [(tile % ATLAS_TILES) / ATLAS_TILES, 1 - (Math.floor(tile / ATLAS_TILES) + 1) / ATLAS_TILES]
  return { texture, uvFor }
}

export const ATLAS_STEP = 1 / ATLAS_TILES
