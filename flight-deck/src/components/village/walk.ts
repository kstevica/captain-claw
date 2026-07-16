// Client-side walk math (village-world plan Phase 2/4): position is a pure
// function of the stored course + the clock, so one map snapshot animates a
// whole walk with zero polling. Shared by the parent map and the public
// observer map so they extrapolate identically.

import type { VillageBeingPos, VillagePlace } from '../../services/beings'

export type PlaceById = Record<string, VillagePlace>

export function destOf(b: VillageBeingPos, placeById: PlaceById): [number, number] {
  if (!b.to) return b.xy
  if (b.path && b.path.length >= 2) return b.path[b.path.length - 1]
  if (b.to === 'home') return b.home_xy
  const p = placeById[b.to]
  return p ? [p.x, p.y] : b.xy
}

export function alongPath(pts: [number, number][], frac: number): [number, number] {
  let total = 0
  for (let i = 0; i < pts.length - 1; i++)
    total += Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
  if (total <= 0) return pts[pts.length - 1]
  let left = Math.max(0, Math.min(1, frac)) * total
  for (let i = 0; i < pts.length - 1; i++) {
    const seg = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    if (left <= seg || i === pts.length - 2) {
      const f = seg <= 0 ? 0 : Math.min(1, left / seg)
      return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f,
              pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f]
    }
    left -= seg
  }
  return pts[pts.length - 1]
}

// fetchedAtMs anchors the pre-world-model fallback (dead-reckon from the
// snapshot); the plotted course uses its own absolute departed_at.
export function posOf(b: VillageBeingPos, placeById: PlaceById, fetchedAtMs: number): [number, number] {
  if (!b.to) return b.xy
  if (b.path && b.path.length >= 2 && b.total_minutes && b.departed_at) {
    const t0 = Date.parse(b.departed_at)
    if (!Number.isNaN(t0)) {
      const frac = ((Date.now() - t0) / 60_000) / b.total_minutes
      return frac >= 1 ? b.path[b.path.length - 1] : alongPath(b.path, frac)
    }
  }
  const dest = destOf(b, placeById)
  const dx = dest[0] - b.xy[0], dy = dest[1] - b.xy[1]
  const dist = Math.hypot(dx, dy)
  if (dist < 1) return dest
  const walked = Math.min(dist, b.speed * ((Date.now() - fetchedAtMs) / 60_000))
  return [b.xy[0] + (dx * walked) / dist, b.xy[1] + (dy * walked) / dist]
}

export function minutesLeft(b: VillageBeingPos, placeById: PlaceById, fetchedAtMs: number): number {
  if (!b.to) return 0
  if (b.total_minutes && b.departed_at) {
    const t0 = Date.parse(b.departed_at)
    if (!Number.isNaN(t0)) return Math.max(0, b.total_minutes - (Date.now() - t0) / 60_000)
  }
  const [x, y] = posOf(b, placeById, fetchedAtMs)
  const dest = destOf(b, placeById)
  return Math.hypot(dest[0] - x, dest[1] - y) / Math.max(0.001, b.speed)
}

export function statusOf(b: VillageBeingPos, placeById: PlaceById, fetchedAtMs: number): string {
  if (b.to) {
    const name = b.to === 'home' ? 'home' : placeById[b.to]?.name ?? b.to
    const mins = Math.round(minutesLeft(b, placeById, fetchedAtMs))
    return mins < 1 ? `arriving at ${name}` : `on the road to ${name} — ~${mins} min`
  }
  if (!b.at || b.at === 'home') return 'at home'
  return `at ${placeById[b.at]?.name ?? b.at}`
}
