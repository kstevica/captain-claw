// The village's body (village-world plan Phase 3): isometric 2:1 sprites,
// warm storybook flat. Every sprite draws in LOCAL iso coordinates with the
// footprint diamond's NORTH corner at (0,0): one tile east = (+32,+16), one
// tile south = (-32,+16) — so a w×h footprint has corners N(0,0),
// E(32w,16w), S(32w-32h,16w+16h), W(-32h,16h), and walls rise in -y. The
// renderer translates each <g> to its place; nothing here positions itself.
//
// Village fixtures wear their own warm palette (they belong to the ground,
// not to a being); only the Iskre are palette-dressed.

import type { FC } from 'react'

const WALL = '#f0e3cc', WALL_D = '#d9c6a8'
const TIMBER = '#a9835a', TIMBER_D = '#8a6844'
const ROOF = '#c96f4a', ROOF_D = '#a5563a'
const SLATE = '#7c93a8', SLATE_D = '#5f7488'
const STONE = '#b9b3a6', STONE_D = '#979082'
const GRASS = '#9bc078', GRASS_D = '#86ad67'
const SOIL = '#8a6a4c', SOIL_D = '#6f543c'
const WATER = '#7db8c9', WATER_D = '#5f9db3'
const PAVE = '#d8cdb8', PAVE_D = '#c2b59c'
const INK = '#4a3a30'
const LEAF = '#6f9a55', LEAF_D = '#557a41'
const GLOW = '#ffd98a'

const pts = (...p: [number, number][]) => p.map((q) => q.join(',')).join(' ')

// ── homes + buildings ─────────────────────────────────────────────────────

// home cottage, 2×2: N(0,0) E(64,32) S(0,64) W(-64,32), rise 28.
// Home as your canvas (world-shaping plan Phase 4): the cottage may wear
// a being-chosen dress — roof + wall hue pairs from the fixed vocabulary.
export const HOME_ROOF_HUES: Record<string, [string, string]> = {
  ember: [ROOF, ROOF_D], slate: ['#7c93a8', '#5f7488'],
  moss: ['#7fa05a', '#647f45'], dusk: ['#8b7fa8', '#6d6288'],
}
export const HOME_WALL_HUES: Record<string, [string, string]> = {
  plaster: [WALL, WALL_D], timber: [TIMBER, TIMBER_D],
  sage: ['#d5dcba', '#bcc59e'],
}

export const Cottage: FC<{ look?: { roof?: string; wall?: string } | null }> =
  ({ look }) => {
    const [rf, rfD] = HOME_ROOF_HUES[look?.roof || ''] ?? HOME_ROOF_HUES.ember
    const [wl, wlD] = HOME_WALL_HUES[look?.wall || ''] ?? HOME_WALL_HUES.plaster
    return (
      <g>
        <polygon points={pts([64, 4], [64, 32], [0, 64], [0, 36])} fill={wl} />
        <polygon points={pts([-64, 4], [-64, 32], [0, 64], [0, 36])} fill={wlD} />
        <polygon points={pts([0, -44], [64, 4], [0, 36])} fill={rf} />
        <polygon points={pts([0, -44], [-64, 4], [0, 36])} fill={rfD} />
        <polygon points={pts([20, 32], [34, 42], [34, 24], [20, 15])} fill={TIMBER_D} />
        <polygon points={pts([22, 30], [32, 37], [32, 25], [22, 18.5])} fill={TIMBER} />
        <rect x="-40" y="14" width="12" height="10" rx="2" fill={GLOW} transform="skewY(-26.5)" />
        <circle cx="0" cy="-44" r="2.4" fill={rfD} />
      </g>
    )
  }

// library, 3×2: N(0,0) E(96,48) S(32,80) W(-64,32), rise 34
export const Library: FC = () => (
  <g>
    <polygon points={pts([96, 14], [96, 48], [32, 80], [32, 46])} fill={WALL} />
    <polygon points={pts([-64, -2], [-64, 32], [32, 80], [32, 46])} fill={WALL_D} />
    <polygon points={pts([16, -58], [96, 14], [32, 46])} fill={SLATE} />
    <polygon points={pts([16, -58], [-64, -2], [32, 46])} fill={SLATE_D} />
    <path d="M 56 36 L 72 48 L 72 30 Q 72 22 64 20 Q 56 22 56 28 Z" fill={TIMBER_D} />
    <path d="M 58 34 L 70 43 L 70 30 Q 70 24.5 64 23 Q 58 24.5 58 28.5 Z" fill={TIMBER} />
    <rect x="-52" y="10" width="11" height="9" rx="1.5" fill={GLOW} transform="skewY(-26.5)" />
    <rect x="-30" y="21" width="11" height="9" rx="1.5" fill={GLOW} transform="skewY(-26.5)" />
    <rect x="-12" y="30" width="11" height="9" rx="1.5" fill={GLOW} transform="skewY(-26.5)" />
    <circle cx="16" cy="-58" r="2.6" fill={SLATE_D} />
  </g>
)

// workshop, 2×2: shed roof, chimney, rise 26
export const Workshop: FC = () => (
  <g>
    <polygon points={pts([64, 6], [64, 32], [0, 64], [0, 38])} fill={TIMBER} />
    <polygon points={pts([-64, 6], [-64, 32], [0, 64], [0, 38])} fill={TIMBER_D} />
    <polygon points={pts([0, -34], [64, -6], [64, 6], [0, 38])} fill={SLATE} />
    <polygon points={pts([0, -34], [-64, 6], [0, 38])} fill={SLATE_D} />
    <rect x="30" y="-26" width="9" height="16" fill={STONE_D} transform="skewY(26.5)" />
    <polygon points={pts([16, 34], [32, 46], [32, 28], [16, 17])} fill={INK} opacity="0.75" />
    <rect x="-46" y="16" width="12" height="9" rx="1.5" fill={GLOW} transform="skewY(-26.5)" />
    <path d="M -30 44 L -22 48 L -22 58 L -30 54 Z" fill={STONE_D} />
  </g>
)

// well, 1×1: stone ring + little roof on posts, rise ~30
export const Well: FC = () => (
  <g>
    <path d="M -14 16 Q -14 26 0 26 Q 14 26 14 16 L 14 8 Q 14 15 0 15 Q -14 15 -14 8 Z"
      fill={STONE} />
    <path d="M -14 16 Q -14 26 0 26 Q 6 26 10 24 L 10 13 Q 5 15 0 15 Q -14 15 -14 8 Z"
      fill={STONE_D} opacity="0.5" />
    <ellipse cx="0" cy="8" rx="14" ry="6.5" fill={STONE_D} />
    <ellipse cx="0" cy="8" rx="9" ry="4" fill={INK} opacity="0.7" />
    <rect x="-12" y="-22" width="2.6" height="32" fill={TIMBER_D} />
    <rect x="9.4" y="-22" width="2.6" height="32" fill={TIMBER_D} />
    <polygon points={pts([0, -38], [18, -18], [0, -12])} fill={ROOF} />
    <polygon points={pts([0, -38], [-18, -18], [0, -12])} fill={ROOF_D} />
    <circle cx="0" cy="2" r="2.2" fill={TIMBER} />
  </g>
)

// the old bench, 1×1: a worn seat and a sapling that remembers
export const Bench: FC = () => (
  <g>
    <path d="M -7 -14 Q -9 -26 -2 -32 Q -1 -24 4 -21 Q -3 -19 -7 -14 Z" fill={LEAF} />
    <path d="M -7 -14 Q -8 -22 -4 -27 Q -5 -20 -7 -14 Z" fill={LEAF_D} />
    <rect x="-8.5" y="-14" width="2" height="16" fill={TIMBER_D} />
    <polygon points={pts([-12, 6], [12, -4], [12, -1], [-12, 9])} fill={TIMBER} />
    <polygon points={pts([-12, 9], [12, -1], [12, 1.5], [-12, 11.5])} fill={TIMBER_D} />
    <rect x="-11" y="9.5" width="2.2" height="7" fill={TIMBER_D} />
    <rect x="8.8" y="0.5" width="2.2" height="7" fill={TIMBER_D} />
  </g>
)

// market stall, 2×2: striped awning over a counter
export const Stall: FC = () => (
  <g>
    <polygon points={pts([50, 14], [50, 30], [0, 55], [0, 39])} fill={TIMBER} />
    <polygon points={pts([-50, 14], [-50, 30], [0, 55], [0, 39])} fill={TIMBER_D} />
    <rect x="-50" y="13" width="2.4" height="18" fill={TIMBER_D} />
    <rect x="47.6" y="13" width="2.4" height="18" fill={TIMBER_D} />
    <rect x="-2" y="-26" width="2.4" height="14" fill={TIMBER_D} />
    <polygon points={pts([0, -30], [58, -2], [50, 8], [0, -18])} fill={ROOF} />
    <polygon points={pts([0, -30], [-58, -2], [-50, 8], [0, -18])} fill={ROOF_D} />
    <polygon points={pts([14.5, -23], [29, -16], [25, -7], [11, -14.4])} fill={WALL} opacity="0.85" />
    <polygon points={pts([43.5, -9], [51, -5.4], [45.7, 4.4], [39, 0.4])} fill={WALL} opacity="0.85" />
    <polygon points={pts([-14.5, -23], [-29, -16], [-25, -7], [-11, -14.4])} fill={WALL} opacity="0.85" />
    <circle cx="24" cy="28" r="3.4" fill="#c9a23f" />
    <circle cx="31" cy="26" r="3" fill={ROOF} />
  </g>
)

// pavilion, 3×3: an open roof where people gather
export const Pavilion: FC = () => (
  <g>
    <polygon points={pts([0, 4], [96, 52], [0, 100], [-96, 52])} fill={PAVE} />
    <polygon points={pts([0, 12], [80, 52], [0, 92], [-80, 52])} fill={PAVE_D} opacity="0.45" />
    <rect x="-73" y="24" width="3" height="30" fill={TIMBER_D} />
    <rect x="70" y="24" width="3" height="30" fill={TIMBER_D} />
    <rect x="-1.5" y="-14" width="3" height="30" fill={TIMBER_D} />
    <rect x="-1.5" y="62" width="3" height="30" fill={TIMBER} />
    <polygon points={pts([0, -46], [96, 2], [0, 30])} fill={ROOF} opacity="0.96" />
    <polygon points={pts([0, -46], [-96, 2], [0, 30])} fill={ROOF_D} opacity="0.96" />
    <circle cx="0" cy="-46" r="2.6" fill={ROOF_D} />
  </g>
)

// cairn, 1×1: stacked stones for looking back
export const Cairn: FC = () => (
  <g>
    <ellipse cx="0" cy="8" rx="13" ry="6" fill={STONE} />
    <ellipse cx="0" cy="8" rx="13" ry="6" fill={STONE_D} opacity="0.35" />
    <ellipse cx="-1" cy="1" rx="9.5" ry="5" fill={STONE} />
    <ellipse cx="1" cy="-5.5" rx="7" ry="4" fill={STONE_D} />
    <ellipse cx="0" cy="-11" rx="4.6" ry="3" fill={STONE} />
    <circle cx="8" cy="12" r="2" fill={GRASS_D} />
    <circle cx="-10" cy="11" r="1.6" fill={GRASS_D} />
  </g>
)

// ── grounds decals (flat, walkable) ───────────────────────────────────────

// the square plaza, 4×4: paving + the village fountain
export const Plaza: FC = () => (
  <g>
    <polygon points={pts([0, 0], [128, 64], [0, 128], [-128, 64])} fill={PAVE} />
    <polygon points={pts([0, 10], [108, 64], [0, 118], [-108, 64])} fill="none"
      stroke={PAVE_D} strokeWidth="2" />
    <polygon points={pts([0, 34], [60, 64], [0, 94], [-60, 64])} fill="none"
      stroke={PAVE_D} strokeWidth="2" />
    <ellipse cx="0" cy="64" rx="24" ry="12" fill={STONE} />
    <ellipse cx="0" cy="62" rx="18" ry="8.5" fill={WATER} />
    <ellipse cx="0" cy="61" rx="10" ry="4.6" fill={WATER_D} />
    <rect x="-2" y="42" width="4" height="18" fill={STONE_D} />
    <ellipse cx="0" cy="42" rx="6" ry="3" fill={WATER} />
  </g>
)

// the garden, 3×3: patient rows that show the work
export const Garden: FC = () => (
  <g>
    <polygon points={pts([0, 0], [96, 48], [0, 96], [-96, 48])} fill={GRASS} />
    <polygon points={pts([-8, 12], [56, 44], [40, 52], [-24, 20])} fill={SOIL} />
    <polygon points={pts([-32, 24], [32, 56], [16, 64], [-48, 32])} fill={SOIL_D} />
    <polygon points={pts([-56, 36], [8, 68], [-8, 76], [-72, 44])} fill={SOIL} />
    <g fill={LEAF}>
      <circle cx="10" cy="26" r="2.6" /><circle cx="26" cy="34" r="2.6" />
      <circle cx="42" cy="42" r="2.6" /><circle cx="-16" cy="40" r="2.6" />
      <circle cx="0" cy="48" r="2.6" /><circle cx="16" cy="56" r="2.6" />
      <circle cx="-40" cy="52" r="2.6" /><circle cx="-24" cy="60" r="2.6" />
    </g>
    <rect x="60" y="30" width="2.4" height="16" fill={TIMBER_D} />
    <circle cx="61.2" cy="28" r="4" fill={ROOF} />
  </g>
)

// the meadow, 4×3: open grass past the houses
export const Meadow: FC = () => (
  <g>
    <polygon points={pts([0, 0], [128, 64], [32, 112], [-96, 48])} fill={GRASS} />
    <polygon points={pts([20, 26], [70, 51], [40, 66], [-10, 41])} fill={GRASS_D} opacity="0.5" />
    <g fill="#e9e2b8">
      <circle cx="-30" cy="34" r="2.2" /><circle cx="48" cy="46" r="2.2" />
      <circle cx="10" cy="70" r="2.2" /><circle cx="-56" cy="50" r="2.2" />
    </g>
    <g fill="#d98a9c">
      <circle cx="24" cy="40" r="2.2" /><circle cx="-8" cy="56" r="2.2" />
      <circle cx="64" cy="62" r="2.2" />
    </g>
    <path d="M -40 62 q 2 -8 0 -12 M -36 63 q 3 -7 1 -13 M -44 61 q 0 -7 -2 -11"
      stroke={GRASS_D} strokeWidth="1.6" fill="none" strokeLinecap="round" />
    <path d="M 80 52 q 2 -8 0 -12 M 84 53 q 3 -7 1 -13" stroke={GRASS_D}
      strokeWidth="1.6" fill="none" strokeLinecap="round" />
  </g>
)

// a pond, 3×3: somewhere to float small things
export const Pond: FC = () => (
  <g>
    <polygon points={pts([0, 4], [92, 50], [0, 96], [-92, 50])} fill={GRASS} />
    <path d="M 0 22 Q 56 30 52 54 Q 44 78 0 76 Q -50 74 -52 50 Q -50 28 0 22 Z" fill={WATER} />
    <path d="M 0 30 Q 40 34 38 52 Q 32 68 0 68 Q -36 66 -38 50 Q -36 32 0 30 Z" fill={WATER_D}
      opacity="0.55" />
    <ellipse cx="-18" cy="46" rx="6" ry="3" fill={LEAF} />
    <ellipse cx="14" cy="58" rx="5" ry="2.6" fill={LEAF_D} />
    <circle cx="-16" cy="44" r="1.6" fill="#d98a9c" />
    <path d="M 60 44 q 2 -10 -1 -16 M 65 46 q 3 -9 1 -15" stroke={LEAF_D}
      strokeWidth="1.8" fill="none" strokeLinecap="round" />
  </g>
)

// ── props (pure per-tile function draws these) ────────────────────────────

export const Tree: FC = () => (
  <g>
    <rect x="-2.6" y="-16" width="5.2" height="18" fill={TIMBER_D} />
    <circle cx="0" cy="-30" r="15" fill={LEAF} />
    <circle cx="-10" cy="-22" r="10" fill={LEAF} />
    <circle cx="10" cy="-23" r="10" fill={LEAF} />
    <circle cx="5" cy="-34" r="9" fill={GRASS} opacity="0.8" />
    <ellipse cx="0" cy="2" rx="10" ry="4" fill="#000" opacity="0.12" />
  </g>
)

export const Conifer: FC = () => (
  <g>
    <rect x="-2" y="-10" width="4" height="12" fill={TIMBER_D} />
    <polygon points={pts([0, -46], [11, -26], [-11, -26])} fill={LEAF_D} />
    <polygon points={pts([0, -36], [13, -14], [-13, -14])} fill={LEAF} />
    <polygon points={pts([0, -26], [15, -4], [-15, -4])} fill={LEAF_D} />
    <ellipse cx="0" cy="2" rx="9" ry="3.6" fill="#000" opacity="0.12" />
  </g>
)

export const Bush: FC = () => (
  <g>
    <ellipse cx="0" cy="0" rx="11" ry="6" fill={LEAF_D} />
    <ellipse cx="-4" cy="-4" rx="8" ry="5.5" fill={LEAF} />
    <ellipse cx="5" cy="-3" rx="7" ry="5" fill={GRASS} />
  </g>
)

export const Flowers: FC = () => (
  <g>
    <path d="M -6 2 q 0 -8 -2 -11 M 0 3 q 1 -9 0 -13 M 6 2 q 1 -7 3 -10"
      stroke={GRASS_D} strokeWidth="1.4" fill="none" strokeLinecap="round" />
    <circle cx="-8.5" cy="-10" r="2.6" fill="#d98a9c" />
    <circle cx="0" cy="-11.5" r="2.6" fill="#e9e2b8" />
    <circle cx="9.5" cy="-9" r="2.6" fill="#c9a23f" />
  </g>
)

export const Lamp: FC = () => (
  <g>
    <rect x="-1.4" y="-30" width="2.8" height="31" fill={INK} />
    <rect x="-4.4" y="-38" width="8.8" height="9" rx="2.4" fill={INK} />
    <rect x="-2.9" y="-36.4" width="5.8" height="5.8" rx="1.4" fill={GLOW} className="village-lamp" />
    <ellipse cx="0" cy="2" rx="5" ry="2" fill="#000" opacity="0.12" />
  </g>
)

// ── made things (world-shaping plan Phase 3) ──────────────────────────────
// Objects a being crafted and placed: small 1-tile fixtures drawn CENTERED
// on (0,0) like the props (the renderer translates them to the tile
// center). Bench and Cairn reuse the fixtures above, nudged to center.

export const Signpost: FC = () => (
  <g>
    <rect x="-1.6" y="-30" width="3.2" height="32" fill={TIMBER_D} />
    <g transform="skewY(-26.5)">
      <rect x="1" y="-24" width="20" height="8" rx="1.5" fill={TIMBER} />
      <path d="M 21 -24 L 26 -20 L 21 -16 Z" fill={TIMBER} />
      <line x1="4" y1="-21.5" x2="17" y2="-21.5" stroke={INK} strokeWidth="1.2" opacity="0.6" />
      <line x1="4" y1="-18.8" x2="14" y2="-18.8" stroke={INK} strokeWidth="1.2" opacity="0.6" />
    </g>
    <ellipse cx="0" cy="2" rx="6" ry="2.4" fill="#000" opacity="0.12" />
  </g>
)

export const Planter: FC = () => (
  <g>
    <polygon points={pts([-13, 0], [0, 6.5], [13, 0], [0, -6.5])} fill={TIMBER} />
    <polygon points={pts([-13, 0], [0, 6.5], [0, 11], [-13, 4.5])} fill={TIMBER_D} />
    <polygon points={pts([13, 0], [0, 6.5], [0, 11], [13, 4.5])} fill={TIMBER_D} opacity="0.75" />
    <polygon points={pts([-9, 0], [0, 4.5], [9, 0], [0, -4.5])} fill={SOIL_D} />
    <g fill={LEAF}>
      <circle cx="-4" cy="-6" r="3" /><circle cx="3" cy="-8" r="3.4" />
      <circle cx="0" cy="-3" r="2.6" />
    </g>
    <circle cx="-4.5" cy="-9" r="1.8" fill="#d98a9c" />
    <circle cx="4" cy="-12" r="1.8" fill="#e9e2b8" />
  </g>
)

export const Sculpture: FC = () => (
  <g>
    <polygon points={pts([-10, 4], [0, 9], [10, 4], [0, -1])} fill={STONE_D} />
    <path d="M -3 4 Q -7 -6 -2 -14 Q 4 -20 2 -28 Q 8 -22 6 -12 Q 5 -4 3 4 Z"
      fill={STONE} />
    <path d="M -3 4 Q -6 -4 -3 -11 Q -4 -3 -1 4 Z" fill={STONE_D} opacity="0.6" />
    <circle cx="2.6" cy="-28" r="2.4" fill={STONE} />
    <ellipse cx="0" cy="6" rx="9" ry="3" fill="#000" opacity="0.1" />
  </g>
)

export const Lantern: FC = () => (
  <g>
    <rect x="-1.2" y="-24" width="2.4" height="25" fill={TIMBER_D} />
    <rect x="-4" y="-31" width="8" height="8.4" rx="2" fill={INK} />
    <rect x="-2.6" y="-29.6" width="5.2" height="5.4" rx="1.2" fill={GLOW} className="village-lamp" />
    <path d="M -4 -31 L 0 -34 L 4 -31" stroke={INK} strokeWidth="1.6" fill="none" />
    <ellipse cx="0" cy="2" rx="4.6" ry="1.9" fill="#000" opacity="0.12" />
  </g>
)

export const Fountain: FC = () => (
  <g>
    <ellipse cx="0" cy="2" rx="14" ry="6.6" fill={STONE} />
    <ellipse cx="0" cy="1" rx="10.5" ry="4.8" fill={WATER} />
    <ellipse cx="0" cy="0.6" rx="6.5" ry="3" fill={WATER_D} />
    <rect x="-1.4" y="-14" width="2.8" height="14" fill={STONE_D} />
    <ellipse cx="0" cy="-14" rx="4.4" ry="2" fill={WATER} />
    <path d="M -3.4 -13 q -2.5 5 -4.5 7 M 3.4 -13 q 2.5 5 4.5 7"
      stroke={WATER} strokeWidth="1.4" fill="none" strokeLinecap="round" />
  </g>
)

export const Shrine: FC = () => (
  <g>
    <polygon points={pts([-11, 2], [0, 7.5], [11, 2], [0, -3.5])} fill={STONE_D} />
    <rect x="-8.5" y="-16" width="2.6" height="18" fill={STONE} />
    <rect x="5.9" y="-16" width="2.6" height="18" fill={STONE} />
    <polygon points={pts([0, -26], [13, -15], [0, -18.5], [-13, -15])} fill={ROOF} />
    <polygon points={pts([0, -18.5], [13, -15], [0, -11.5], [-13, -15])} fill={ROOF_D} opacity="0.55" />
    <rect x="-2.4" y="-9" width="4.8" height="5" rx="1" fill={GLOW} opacity="0.9" className="village-lamp" />
  </g>
)

const ObjBench: FC = () => <g transform="translate(0 -12)"><Bench /></g>
const ObjCairn: FC = () => <g transform="translate(0 -4)"><Cairn /></g>

export const OBJECT_SPRITES: Record<string, FC> = {
  bench: ObjBench, cairn: ObjCairn, signpost: Signpost, planter: Planter,
  sculpture: Sculpture, lantern: Lantern, fountain: Fountain, shrine: Shrine,
}

// ── the registry the renderer draws from ──────────────────────────────────

export const BUILDING_SPRITES: Record<string, FC> = {
  cottage: Cottage, library: Library, workshop: Workshop, well: Well,
  bench: Bench, stall: Stall, pavilion: Pavilion, cairn: Cairn,
  plaza: Plaza, garden: Garden, meadow: Meadow, pond: Pond,
}

export const PROP_SPRITES: Record<string, FC> = {
  tree: Tree, conifer: Conifer, bush: Bush, flowers: Flowers, lamp: Lamp,
}

const SPRITE_BY_ID: Record<string, string> = {
  square: 'plaza', library: 'library', workshop: 'workshop', garden: 'garden',
  well: 'well', meadow: 'meadow', 'old-bench': 'bench',
}
const SPRITE_BY_AFFORDANCE: Record<string, string> = {
  read: 'library', create: 'workshop', gather: 'pavilion', trade: 'stall',
  tend: 'garden', play: 'pond', rest: 'bench', remember: 'cairn',
}

// The 7 default places wear their bespoke look; anything the village
// raises later (commissions, architect drafts) looks right by affordance.
export function spriteForPlace(p: { id: string; affordances?: string[] }): string {
  return SPRITE_BY_ID[p.id]
    || SPRITE_BY_AFFORDANCE[(p.affordances || [])[0] || '']
    || 'cottage'
}
