// The Iskre themselves (village-world plan Phase 3): 10 storybook-flat
// characters, each drawn once and dressed by 4 palettes through CSS
// custom properties — 40 looks from 10 drawings. Slots: --c1 cloth,
// --c2 trim/shade, --c3 hair & spark, --c4 the face glow (they are
// sparks — the face IS the light). viewBox 0 0 48 64, feet on y≈60.

const INK = '#4a3a30'          // fixed warm dark: eyes, feet, linework
const BLUSH = '#e08a6d'

export const PALETTES: Record<string, { c1: string; c2: string; c3: string; c4: string }> = {
  ember: { c1: '#c46a3f', c2: '#8f4630', c3: '#e8a04b', c4: '#ffdfae' },
  meadow: { c1: '#6f9a55', c2: '#4a6b3a', c3: '#b5cf72', c4: '#f3f7d4' },
  sea: { c1: '#5583a3', c2: '#38607d', c3: '#8fc3d4', c4: '#e3f2f7' },
  dusk: { c1: '#83699f', c2: '#5b4775', c3: '#bb9dd4', c4: '#f0e5fa' },
}
export const PALETTE_NAMES = Object.keys(PALETTES) as (keyof typeof PALETTES & string)[]

function Face({ cx = 24, cy = 24, r = 8 }: { cx?: number; cy?: number; r?: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill="var(--c4)" />
      <circle cx={cx - r * 0.38} cy={cy} r={1.2} fill={INK} />
      <circle cx={cx + r * 0.38} cy={cy} r={1.2} fill={INK} />
      <path d={`M ${cx - 1.6} ${cy + r * 0.4} Q ${cx} ${cy + r * 0.62} ${cx + 1.6} ${cy + r * 0.4}`}
        stroke={INK} strokeWidth="0.9" fill="none" strokeLinecap="round" />
      <circle cx={cx - r * 0.72} cy={cy + r * 0.3} r={1.3} fill={BLUSH} opacity="0.55" />
      <circle cx={cx + r * 0.72} cy={cy + r * 0.3} r={1.3} fill={BLUSH} opacity="0.55" />
    </g>
  )
}

function Feet({ y = 59 }: { y?: number }) {
  return (
    <g fill={INK}>
      <ellipse cx="19.5" cy={y} rx="3.2" ry="1.8" />
      <ellipse cx="28.5" cy={y} rx="3.2" ry="1.8" />
    </g>
  )
}

// 1 — the Hood: a small wanderer, all cloak, lantern-lit face
function Hood() {
  return (
    <g>
      <path d="M 11 58 Q 10 34 24 20 Q 38 34 37 58 Z" fill="var(--c1)" />
      <path d="M 11 58 Q 12 52 24 51 Q 36 52 37 58 Z" fill="var(--c2)" />
      <circle cx="24" cy="24" r="12.5" fill="var(--c1)" />
      <Face cx={24} cy={26} r={7.5} />
      <path d="M 13 27 Q 15 15 24 13 Q 33 15 35 27 Q 30 20 24 20 Q 18 20 13 27 Z"
        fill="var(--c2)" />
    </g>
  )
}

// 2 — the Braids: a dress and two long braids of beads
function Braids() {
  return (
    <g>
      <Feet />
      <path d="M 15 58 L 18 34 Q 24 31 30 34 L 33 58 Z" fill="var(--c1)" />
      <rect x="15.5" y="53" width="17" height="4" rx="2" fill="var(--c2)" />
      <circle cx="24" cy="23" r="10" fill="var(--c3)" />
      <Face cx={24} cy={25} r={7.5} />
      <path d="M 15 20 Q 18 12 24 12 Q 30 12 33 20 Q 28 16 24 16.5 Q 20 16 15 20 Z"
        fill="var(--c3)" />
      <g fill="var(--c3)">
        <circle cx="13.5" cy="28" r="2.6" /><circle cx="13" cy="33.5" r="2.4" />
        <circle cx="12.8" cy="38.5" r="2.2" />
        <circle cx="34.5" cy="28" r="2.6" /><circle cx="35" cy="33.5" r="2.4" />
        <circle cx="35.2" cy="38.5" r="2.2" />
      </g>
      <circle cx="12.8" cy="42" r="1.4" fill="var(--c2)" />
      <circle cx="35.2" cy="42" r="1.4" fill="var(--c2)" />
    </g>
  )
}

// 3 — the Scarf: bundled up, one end of the scarf off in the wind
function Scarf() {
  return (
    <g>
      <Feet />
      <rect x="16" y="32" width="16" height="26" rx="7" fill="var(--c1)" />
      <circle cx="24" cy="22" r="9.5" fill="var(--c3)" />
      <Face cx={24} cy={24} r={7} />
      <path d="M 16 19 Q 19 12 24 12 Q 29 12 32 19 Q 27 15.5 24 15.5 Q 21 15.5 16 19 Z"
        fill="var(--c3)" />
      <rect x="15" y="29.5" width="18" height="6.5" rx="3.2" fill="var(--c2)" />
      <path d="M 31 33 Q 40 34 42 28 Q 43 34 37 37 Q 33 38.5 31 36 Z" fill="var(--c2)" />
      <rect x="38.6" y="27.2" width="4" height="2" rx="1" fill="var(--c1)" opacity="0.7" />
    </g>
  )
}

// 4 — the Hat: under a wide travelling hat, mostly hidden and happy
function Hat() {
  return (
    <g>
      <Feet />
      <path d="M 16 58 L 18 36 Q 24 33 30 36 L 32 58 Z" fill="var(--c1)" />
      <circle cx="24" cy="27" r="8.5" fill="var(--c4)" />
      <Face cx={24} cy={28} r={7} />
      <ellipse cx="24" cy="20" rx="15" ry="4.6" fill="var(--c2)" />
      <path d="M 15 20 Q 15 10 24 10 Q 33 10 33 20 Z" fill="var(--c1)" />
      <rect x="15.4" y="16.6" width="17.2" height="3" rx="1.5" fill="var(--c3)" />
    </g>
  )
}

// 5 — the Little Round: nearly all glow, a knit vest, a spark tuft
function Round() {
  return (
    <g>
      <Feet y={60} />
      <circle cx="24" cy="38" r="17" fill="var(--c4)" />
      <path d="M 8.5 42 Q 24 50 39.5 42 L 39.5 47 Q 24 56 8.5 47 Z" fill="var(--c1)" />
      <path d="M 8.5 46.5 Q 24 55 39.5 46.5 L 39.5 49 Q 24 57.5 8.5 49 Z" fill="var(--c2)" />
      <circle cx="19" cy="35" r="1.5" fill={INK} />
      <circle cx="29" cy="35" r="1.5" fill={INK} />
      <path d="M 21.6 40 Q 24 42 26.4 40" stroke={INK} strokeWidth="1" fill="none" strokeLinecap="round" />
      <circle cx="15.5" cy="39" r="1.8" fill={BLUSH} opacity="0.55" />
      <circle cx="32.5" cy="39" r="1.8" fill={BLUSH} opacity="0.55" />
      <path d="M 24 21 Q 22 15 25.5 11 Q 25 16 28 18 Q 25.5 19 24 21 Z" fill="var(--c3)" />
    </g>
  )
}

// 6 — the Tall: a long coat, a straight back, buttons all the way down
function Tall() {
  return (
    <g>
      <Feet />
      <path d="M 17 58 L 18 26 Q 24 23 30 26 L 31 58 Z" fill="var(--c1)" />
      <path d="M 23.6 27 L 24.4 27 L 24.4 56 L 23.6 56 Z" fill="var(--c2)" />
      <circle cx="24" cy="40" r="1.1" fill="var(--c2)" />
      <circle cx="24" cy="46" r="1.1" fill="var(--c2)" />
      <path d="M 18 26 L 24 29.5 L 30 26 L 30 30 L 24 33 L 18 30 Z" fill="var(--c2)" />
      <circle cx="24" cy="17" r="8.5" fill="var(--c4)" />
      <Face cx={24} cy={18} r={7} />
      <path d="M 15.5 15 Q 17 8.5 24 8.5 Q 31 8.5 32.5 15 Q 28 11.5 24 11.5 Q 20 11.5 15.5 15 Z"
        fill="var(--c3)" />
    </g>
  )
}

// 7 — the Curly: a cloud of curls that never quite agrees with itself
function Curly() {
  return (
    <g>
      <Feet />
      <path d="M 14 58 Q 14 36 24 34 Q 34 36 34 58 Z" fill="var(--c1)" />
      <path d="M 20 41 Q 24 43.5 28 41 L 28 44.5 Q 24 47 20 44.5 Z" fill="var(--c2)" />
      <circle cx="24" cy="24" r="9" fill="var(--c4)" />
      <Face cx={24} cy={25.5} r={7} />
      <g fill="var(--c3)">
        <circle cx="15" cy="21" r="4.4" />
        <circle cx="20" cy="16" r="4.8" />
        <circle cx="27" cy="15" r="4.8" />
        <circle cx="33" cy="19.5" r="4.4" />
        <circle cx="24" cy="18" r="4.6" />
        <circle cx="34" cy="25" r="3" />
        <circle cx="14" cy="26" r="3" />
      </g>
    </g>
  )
}

// 8 — the Spark: a snug suit and a small flame that never blows out
function Spark() {
  return (
    <g>
      <Feet />
      <path d="M 16.5 58 L 17.5 33 Q 24 30 30.5 33 L 31.5 58 Z" fill="var(--c1)" />
      <path d="M 20 34 Q 24 32.5 28 34 L 27.5 44 Q 24 45.5 20.5 44 Z" fill="var(--c2)" />
      <circle cx="24" cy="22" r="9" fill="var(--c4)" />
      <Face cx={24} cy={23.5} r={7.2} />
      <path d="M 23.6 12.5 L 24.4 12.5 L 24.4 8.8 L 23.6 8.8 Z" fill={INK} opacity="0.6" />
      <path d="M 24 1.5 Q 27.2 5 24 8.8 Q 20.8 5 24 1.5 Z" fill="var(--c3)" />
      <circle cx="24" cy="6.4" r="1.3" fill="var(--c4)" />
    </g>
  )
}

// 9 — the Cape: one shoulder always into the wind
function Cape() {
  return (
    <g>
      <Feet />
      <path d="M 26 30 Q 40 34 41 56 Q 33 52 26 54 Z" fill="var(--c2)" />
      <path d="M 16.5 58 L 18 33 Q 24 30 30 33 L 31.5 58 Z" fill="var(--c1)" />
      <circle cx="27.5" cy="32.5" r="2" fill="var(--c3)" />
      <circle cx="24" cy="22.5" r="9" fill="var(--c4)" />
      <Face cx={24} cy={24} r={7} />
      <path d="M 15 21 Q 16 12.5 24 12.5 Q 32 12.5 33 20 Q 34.5 15 32 12 Q 36 14 35.5 19
               Q 30 15.5 24 16 Q 19 16 15 21 Z" fill="var(--c3)" />
    </g>
  )
}

// 10 — the Apron: sleeves rolled, a day's work in the front pocket
function Apron() {
  return (
    <g>
      <Feet />
      <path d="M 14.5 58 L 17 33 Q 24 30 31 33 L 33.5 58 Z" fill="var(--c1)" />
      <path d="M 18.5 38 L 29.5 38 L 30.5 56 L 17.5 56 Z" fill="var(--c2)" />
      <path d="M 20.5 44 L 27.5 44 L 27.5 50 L 20.5 50 Z" fill="var(--c1)" />
      <path d="M 18.5 38 L 24 35 L 29.5 38" stroke="var(--c2)" strokeWidth="1.6" fill="none" />
      <circle cx="24" cy="23" r="9" fill="var(--c4)" />
      <Face cx={24} cy={24.5} r={7} />
      <path d="M 15.5 21 Q 17 13 24 13 Q 31 13 32.5 21 Q 27.5 16.5 24 16.5 Q 20.5 16.5 15.5 21 Z"
        fill="var(--c3)" />
      <circle cx="31" cy="14.5" r="3.4" fill="var(--c3)" />
    </g>
  )
}

export const CHARACTERS = [Hood, Braids, Scarf, Hat, Round, Tall, Curly, Spark, Cape, Apron]
export const CHARACTER_NAMES = ['the Hood', 'the Braids', 'the Scarf', 'the Hat',
  'the Little Round', 'the Tall', 'the Curly', 'the Spark', 'the Cape', 'the Apron']

export interface AvatarPick { c: number; p: string }

// One Iskra, dressed: character `c` (1-10) in palette `p`, `size` px wide.
export function IskraAvatar({ c, p, size = 32, className, title }: {
  c: number; p: string; size?: number; className?: string; title?: string
}) {
  const pal = PALETTES[p] ?? PALETTES.ember
  const Char = CHARACTERS[(((c || 1) - 1) % CHARACTERS.length + CHARACTERS.length) % CHARACTERS.length]
  return (
    <svg viewBox="0 0 48 64" width={size} height={(size * 64) / 48}
      className={className} aria-label={title}
      style={{ ['--c1' as string]: pal.c1, ['--c2' as string]: pal.c2,
               ['--c3' as string]: pal.c3, ['--c4' as string]: pal.c4 }}>
      {title ? <title>{title}</title> : null}
      <ellipse cx="24" cy="61" rx="10" ry="2.4" fill="#000" opacity="0.14" />
      <Char />
    </svg>
  )
}
