// The village, playable (village-world plan Phase 4): an isometric 2:1
// scene fed entirely by the village-map payload — ground, streets, grounds
// decals, then buildings, cottages, props and Iskre depth-sorted by their
// base, walking their plotted courses client-side (position is a pure
// function of the clock, so one snapshot animates without polling).
// Dark theme is evening: cooler ground, and the lamps come on.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { VillageBeingPos, VillageMapData, VillagePlace } from '../../services/beings'
import { IskraAvatar } from './avatars'
import { BUILDING_SPRITES, OBJECT_SPRITES, PROP_SPRITES, Conifer, Cottage as CottageSprite, spriteForPlace } from './buildings'

const TILE = 20
const GRID = 120     // generous clamp ceiling (matches backend GRID_MAX) —
                     // real plot size comes from the payload (grow map)
const ISO_X = 1.6           // per world unit → a 20-unit tile spans 32 px
const ISO_Y = 0.8           // …and 16 px tall: the classic 2:1 diamond

const iso = (x: number, y: number): [number, number] =>
  [(x - y) * ISO_X, (x + y) * ISO_Y]

// Replicates the backend's footprint clamp (being_world._tiles_at) so the
// picture and the physics agree on where every building stands.
function footNW(p: VillagePlace): [number, number, number, number] {
  const w = Math.max(1, p.w || 1), h = Math.max(1, p.h || 1)
  const cx = Math.floor(p.x / TILE), cy = Math.floor(p.y / TILE)
  const tx0 = Math.min(Math.max(1, cx - Math.floor(w / 2)), GRID - 1 - w)
  const ty0 = Math.min(Math.max(1, cy - Math.floor(h / 2)), GRID - 1 - h)
  return [tx0, ty0, w, h]
}

// …and the backend's home_tiles clamp: every cottage is 2×2.
function homeNW(hxy: [number, number]): [number, number] {
  return [Math.min(Math.max(0, Math.floor(hxy[0] / TILE)), GRID - 2),
          Math.min(Math.max(0, Math.floor(hxy[1] / TILE)), GRID - 2)]
}

const tileDiamond = (tx: number, ty: number): string => {
  const [sx, sy] = iso(tx * TILE, ty * TILE)
  return `${sx},${sy} ${sx + 32},${sy + 16} ${sx},${sy + 32} ${sx - 32},${sy + 16}`
}

const footDiamond = (tx0: number, ty0: number, w: number, h: number): string => {
  const [nx, ny] = iso(tx0 * TILE, ty0 * TILE)
  return `${nx},${ny} ${nx + 32 * w},${ny + 16 * w} `
    + `${nx + 32 * w - 32 * h},${ny + 16 * w + 16 * h} ${nx - 32 * h},${ny + 16 * h}`
}

const DAY = { grass: '#a8c586', grass2: '#9fbd7c', road: '#d3b489', roadEdge: '#c2a274' }
const NIGHT = { grass: '#5f7259', grass2: '#57694f', road: '#8d7f68', roadEdge: '#7d705c'.slice(0, 7) }

interface SceneProps {
  data: VillageMapData
  sel: string | null
  selBeing: string | null
  onPlace: (id: string | null) => void
  onBeing: (slug: string | null) => void
  posOf: (b: VillageBeingPos) => [number, number]
  hue: (p: VillagePlace) => string
  // fill the parent's height (fullscreen) instead of the fixed 460px card height
  fill?: boolean
  // made things (world-shaping plan Phase 3): optional so read-only maps
  // still DRAW the objects — they just aren't selectable there.
  selObject?: string | null
  onObject?: (id: string | null) => void
  // parent-build: when a kind is armed OR road mode is on, a background
  // click reports the village-unit spot instead of deselecting.
  buildKind?: string | null
  roadMode?: boolean
  onGround?: (x: number, y: number) => void
}

export function IsoScene({ data, sel, selBeing, onPlace, onBeing, posOf, hue, fill,
                           selObject, onObject, buildKind, roadMode, onGround }: SceneProps) {
  const dark = typeof document !== 'undefined'
    && document.documentElement.classList.contains('dark')
  const C = dark ? NIGHT : DAY

  // the real plot size (grow map) — the ground diamond and the home view
  // scale with it; iso() itself is plot-agnostic.
  const plot = data.grid?.plot_w || 1000
  const isoW = 1.6 * plot, isoH = 0.8 * plot     // half-width / quarter-height
  // pan + zoom: a viewBox the wheel shrinks and the pointer drags
  const HOME_VB = useMemo<[number, number, number, number]>(
    () => [-isoW - 60, -isoH * 0.175, isoW * 2 + 120, isoH * 2 + 300],
    [isoW, isoH])
  const [vb, setVb] = useState(HOME_VB)
  // a resize (new plot) re-frames the whole view
  useEffect(() => { setVb(HOME_VB) }, [HOME_VB])
  const drag = useRef<{ x: number; y: number; vb: typeof HOME_VB } | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const aspect = HOME_VB[3] / HOME_VB[2]
  const onWheel = useCallback((e: React.WheelEvent) => {
    setVb((v) => {
      const k = e.deltaY > 0 ? 1.12 : 1 / 1.12
      const w = Math.min(HOME_VB[2], Math.max(500, v[2] * k))
      const hgt = w * aspect
      const cx = v[0] + v[2] / 2, cy = v[1] + v[3] / 2
      return [cx - w / 2, cy - hgt / 2, w, hgt]
    })
  }, [HOME_VB, aspect])
  const onDown = useCallback((e: React.PointerEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, vb }
  }, [vb])
  const onMove = useCallback((e: React.PointerEvent) => {
    if (!drag.current || !svgRef.current) return
    const scale = drag.current.vb[2] / svgRef.current.clientWidth
    setVb([drag.current.vb[0] - (e.clientX - drag.current.x) * scale,
           drag.current.vb[1] - (e.clientY - drag.current.y) * scale,
           drag.current.vb[2], drag.current.vb[3]])
  }, [])
  const onUp = useCallback((e: React.PointerEvent) => {
    const d = drag.current
    drag.current = null
    // a real drag swallows the click-to-deselect
    if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) e.stopPropagation()
  }, [])

  const roads = (data.roads ?? []).map((t) => [t[0], t[1]] as [number, number])
  const props = data.props ?? []
  const places = data.places

  // depth-sorted world: buildings, cottages, props, iskre — by base sy
  type Piece = { depth: number; el: React.ReactNode }
  const pieces: Piece[] = []

  for (const p of places) {
    const key = spriteForPlace(p)
    const Sprite = BUILDING_SPRITES[key]
    if (!Sprite) continue
    const [tx0, ty0, w, h] = footNW(p)
    const [nx, ny] = iso(tx0 * TILE, ty0 * TILE)
    const isGrounds = (p.kind || '') === 'grounds'
    const depth = isGrounds ? -1000 + ny : ny + (w + h) * 16
    pieces.push({
      depth,
      el: (
        <g key={`pl-${p.id}`} className="cursor-pointer"
          onClick={(e) => { e.stopPropagation(); onBeing(null); onObject?.(null); onPlace(p.id === sel ? null : p.id) }}>
          <polygon points={footDiamond(tx0, ty0, w, h)} fill={hue(p)}
            opacity={sel === p.id ? 0.28 : 0} stroke={hue(p)}
            strokeOpacity={sel === p.id ? 0.9 : 0} strokeWidth={3} />
          <g transform={`translate(${nx} ${ny})`}><Sprite /></g>
          <polygon points={footDiamond(tx0, ty0, w, h)} fill="transparent">
            <title>{p.name}</title>
          </polygon>
        </g>
      ),
    })
    const [lx, ly] = iso(p.x, p.y)
    pieces.push({
      depth: 1e6,           // labels float above the world
      el: (
        <text key={`lb-${p.id}`} x={lx} y={ly + (w + h) * 8 + 30}
          textAnchor="middle" fontSize={30} pointerEvents="none"
          className={sel === p.id ? 'fill-zinc-100' : 'fill-zinc-300'}
          style={{ paintOrder: 'stroke', stroke: dark ? '#1b2118' : '#f4efdf',
                   strokeWidth: 5, strokeLinejoin: 'round' }}>
          {p.name}
        </text>
      ),
    })
  }

  for (const b of data.beings) {
    if (b.kind === 'visitor' || !b.home_xy) continue   // guests keep no cottage
    const [htx, hty] = homeNW(b.home_xy)
    const [nx, ny] = iso(htx * TILE, hty * TILE)
    // Home as your canvas (world-shaping plan Phase 4): the cottage wears
    // its being's chosen dress and, when named, its name.
    const title = b.home_name
      ? `“${b.home_name}” — ${b.name}'s home` : `${b.name}'s home`
    pieces.push({
      depth: ny + 4 * 16,
      el: (
        <g key={`hm-${b.slug}`} transform={`translate(${nx} ${ny}) scale(0.72)`}
          style={{ transformBox: 'fill-box' }} opacity={0.96}>
          <CottageSprite look={b.home_look} />
          <title>{title}</title>
        </g>
      ),
    })
    if (b.home_name) {
      pieces.push({
        depth: 1e6,
        el: (
          <text key={`hmlb-${b.slug}`} x={nx} y={ny + 78} textAnchor="middle"
            fontSize={19} pointerEvents="none" className="fill-zinc-400"
            style={{ paintOrder: 'stroke', stroke: dark ? '#1b2118' : '#f4efdf',
                     strokeWidth: 4, strokeLinejoin: 'round', fontStyle: 'italic' }}>
            “{b.home_name}”
          </text>
        ),
      })
    }
  }

  for (const pr of props) {
    const [tx, ty] = pr.tile
    const [bx, by] = iso((tx + 0.5) * TILE, (ty + 0.5) * TILE)
    const Sprite = pr.kind === 'tree' && (tx + ty) % 3 === 0
      ? Conifer : PROP_SPRITES[pr.kind]
    if (!Sprite) continue
    pieces.push({
      depth: by,
      el: (
        <g key={`pr-${tx}-${ty}-${pr.kind}`} transform={`translate(${bx} ${by})`}
          pointerEvents="none">
          <Sprite />
          {dark && pr.kind === 'lamp' && (
            <circle cx="0" cy="-32" r="46" fill="#ffd98a" opacity="0.16" />
          )}
        </g>
      ),
    })
  }

  // Made things (world-shaping plan Phase 3): a being's placed works, drawn
  // beside the props and clickable when the map offers a panel for them.
  for (const o of data.objects ?? []) {
    const [tx, ty] = o.tile
    const [bx, by] = iso((tx + 0.5) * TILE, (ty + 0.5) * TILE)
    const Sprite = OBJECT_SPRITES[o.kind]
    if (!Sprite) continue
    const selMe = selObject === o.id
    // A beginning (instinct-build plan): the feet broke ground but the mind
    // hasn't finished it — drawn faint and dashed, "not yet real."
    const staked = o.staked
    pieces.push({
      depth: by + 0.25,     // a made thing edges in front of a same-tile prop
      el: (
        <g key={`ob-${o.id}`} transform={`translate(${bx} ${by})`}
          className={onObject ? 'cursor-pointer' : undefined}
          onClick={onObject ? (e) => {
            e.stopPropagation(); onPlace(null); onBeing(null)
            onObject(selMe ? null : o.id)
          } : undefined}>
          {selMe && (
            <ellipse cx="0" cy="4" rx="26" ry="13" fill="#fbbf24"
              opacity="0.32" stroke="#fbbf24" strokeOpacity="0.8" />
          )}
          {staked && (
            <ellipse cx="0" cy="4" rx="20" ry="10" fill="none"
              stroke="#cbb48a" strokeWidth="1.5" strokeDasharray="3 3"
              opacity="0.75" />
          )}
          <g opacity={staked ? 0.4 : 1}
            style={staked ? { filter: 'grayscale(0.5)' } : undefined}>
            <Sprite />
          </g>
          {dark && !staked && (o.kind === 'lantern' || o.kind === 'shrine') && (
            <circle cx="0" cy="-26" r="40" fill="#ffd98a" opacity="0.14" />
          )}
          <title>{staked
            ? `a beginning — a ${o.kind} someone started${o.by_name ? ` (${o.by_name})` : ''}, not yet real`
            : `${o.name} — a ${o.kind}${o.by_name ? `, ${o.by_name}'s work` : ''}`}</title>
        </g>
      ),
    })
  }

  for (const b of data.beings) {
    const [wx, wy] = posOf(b)
    const [sx, sy] = iso(wx, wy)
    const selMe = selBeing === b.slug
    const guest = b.kind === 'visitor'
    const size = b.stage === 'infant' ? 40 : 54
    const aura = guest ? '#38bdf8' : '#8b5cf6'      // guests glow sky-blue
    pieces.push({
      depth: sy + 0.5,       // a hair in front of anything sharing the tile
      el: (
        <g key={`bg-${b.slug}`} className="cursor-pointer"
          onClick={(e) => { e.stopPropagation(); onPlace(null); onObject?.(null); onBeing(selMe ? null : b.slug) }}>
          {selMe && b.to && b.path && b.path.length >= 2 && (
            <polyline points={b.path.map(([px, py]) => iso(px, py).join(',')).join(' ')}
              fill="none" stroke="#a78bfa" strokeWidth={4} strokeOpacity={0.55}
              strokeDasharray="8 10" strokeLinecap="round" />
          )}
          <ellipse cx={sx} cy={sy} rx={selMe ? 22 : 16} ry={selMe ? 11 : 8}
            fill={selMe ? '#fbbf24' : aura} opacity={selMe ? 0.4 : guest ? 0.28 : 0.18} />
          {guest && (
            <ellipse cx={sx} cy={sy} rx={20} ry={10} fill="none" stroke={aura}
              strokeOpacity={0.8} strokeDasharray="3 5" />
          )}
          {b.to && (
            <ellipse cx={sx} cy={sy} rx={24} ry={12} fill="none" stroke="#a78bfa"
              strokeOpacity={0.7} strokeDasharray="4 6" className="animate-pulse" />
          )}
          <g transform={`translate(${sx - size / 2} ${sy - (size * 64) / 48 + 4})`}
            opacity={guest ? 0.94 : 1}>
            <IskraAvatar c={b.avatar?.c ?? 1} p={b.avatar?.p ?? 'ember'}
              size={size} title={guest ? `${b.name} — visiting from ${b.from}` : b.name} />
          </g>
          {guest && (
            <text x={sx} y={sy - (size * 64) / 48 - 6} textAnchor="middle" fontSize={17}
              pointerEvents="none" fill="#7dd3fc"
              style={{ paintOrder: 'stroke', stroke: dark ? '#0d1b2a' : '#e8f4fb',
                       strokeWidth: 4, strokeLinejoin: 'round', fontWeight: 600 }}>
              ✦ visiting
            </text>
          )}
          <text x={sx} y={sy + 24} textAnchor="middle" fontSize={24}
            pointerEvents="none" className="fill-zinc-200"
            style={{ paintOrder: 'stroke', stroke: dark ? '#1b2118' : '#f4efdf',
                     strokeWidth: 4, strokeLinejoin: 'round' }}>
            {b.name}
          </text>
        </g>
      ),
    })
  }

  pieces.sort((a, z) => a.depth - z.depth)

  // parent-build: unproject a background click (viewBox space → village
  // units) by inverting the same iso() used to draw, so the thing lands
  // where the keeper points. iso(x,y) = [(x-y)·ISO_X, (x+y)·ISO_Y].
  const onGroundClick = (e: React.MouseEvent) => {
    const svg = svgRef.current
    if (!svg) return
    const r = svg.getBoundingClientRect()
    const vx = vb[0] + ((e.clientX - r.left) / r.width) * vb[2]
    const vy = vb[1] + ((e.clientY - r.top) / r.height) * vb[3]
    const a = vx / ISO_X, b = vy / ISO_Y     // a = x−y, b = x+y
    onGround?.(Math.round((a + b) / 2), Math.round((b - a) / 2))
  }
  return (
    <svg ref={svgRef} viewBox={vb.join(' ')}
      className={`w-full touch-none rounded-md border border-zinc-800/60 ${buildKind || roadMode ? 'cursor-crosshair' : ''} ${fill ? 'h-full min-h-[460px]' : 'h-[460px]'} ${dark ? 'bg-[#20281e]' : 'bg-[#eae4cf]'}`}
      onClick={(e) => {
        if ((buildKind || roadMode) && onGround) { onGroundClick(e); return }
        onPlace(null); onBeing(null); onObject?.(null)
      }}
      onWheel={onWheel} onPointerDown={onDown} onPointerMove={onMove}
      onPointerUp={onUp} onPointerLeave={() => { drag.current = null }}
      onDoubleClick={() => setVb(HOME_VB)}>
      <defs>
        <pattern id="isoGrass" patternUnits="userSpaceOnUse" width="64" height="32">
          <rect width="64" height="32" fill={C.grass} />
          <polygon points="32,0 64,16 32,32 0,16" fill={C.grass2} />
        </pattern>
      </defs>
      <polygon points={`0,0 ${isoW},${isoH} 0,${2 * isoH} ${-isoW},${isoH}`} fill="url(#isoGrass)" />
      {roads.map(([tx, ty]) => (
        <polygon key={`rd-${tx}-${ty}`} points={tileDiamond(tx, ty)}
          fill={C.road} stroke={C.roadEdge} strokeWidth={1} />
      ))}
      {pieces.map((p) => p.el)}
    </svg>
  )
}
