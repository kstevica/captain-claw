// Enter the village (FPV plan Phases 1–3): the fullscreen first-person
// overlay — canvas + a whisper of HUD. Two ghosts can wear it: the PARENT
// (authed endpoints, wears the violet "parent" pill, may pull any sign)
// and a PUBLIC VISITOR (un-gated endpoints, wears their chosen name, may
// only plant). Both leave signed notes the Iskre later find, and both are
// quietly felt in passing (presence, cooldown-bounded, $0). Lazy-loaded
// and rendered through a portal so no stacking context paints over it.
// The overlay chrome uses fixed warm-dark colors, not zinc classes: this
// is the game's own night, whatever the app theme.

import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  getSelfFile, getSelfFiles, getVillageMap, plantVillageNote,
  postGhostBeat, postGhostLeave, postVillagePresence, pullVillageNote,
  type VillageMapData, type VillageNote, type VillagePlace,
} from '../../../services/beings'
import {
  getPublicFile, getPublicFiles, getPublicVillageMap, plantPublicNote,
  postPublicGhostBeat, postPublicGhostLeave, postPublicPresence,
} from '../../../services/beingsPublic'
import { createFPV, type FPVHandle, type FPVStatus } from './engine'
import { buildWorld } from './worldgen'

const BuildingReader = lazy(() => import('./BuildingReader'))
const MobileControls = lazy(() => import('./MobileControls'))

// a phone/tablet (coarse primary pointer) roams by touch; a mouse-primary
// device (even a touchscreen laptop) keeps the keyboard + pointer-lock
// path. `?touch=1` forces the touch controls on any device.
const IS_TOUCH = typeof window !== 'undefined'
  && (!!window.matchMedia?.('(pointer: coarse)').matches
    || /[?&]touch=1/.test(window.location.search))

const PANEL = 'rounded-xl border border-[#4a4436] bg-[#171410]/90 text-[#e8e2cf]'
const CHIP = 'rounded-md border border-[#4a4436] bg-[#171410]/75 px-2.5 py-1 text-[11px] text-[#e8e2cf] backdrop-blur-sm'
const NOTE_MAX = 280

function useClock(): string {
  const [t, setT] = useState('')
  useEffect(() => {
    const fmt = () => setT(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    fmt()
    const iv = window.setInterval(fmt, 15_000)
    return () => window.clearInterval(iv)
  }, [])
  return t
}

// who the ghost is, worn on the sleeve: violet for the parent, amber for
// a named visitor
function IdentityPill({ mode, name }: { mode: 'parent' | 'visitor'; name?: string }) {
  return mode === 'parent' ? (
    <div className={`${CHIP} border-violet-400/50 text-violet-200`}>parent</div>
  ) : (
    <div className={`${CHIP} border-amber-400/50 text-amber-200`}>{name || 'visitor'}</div>
  )
}

function AuthorPill({ note }: { note: VillageNote }) {
  return note.author_kind === 'parent' ? (
    <span className="rounded border border-violet-400/50 px-1.5 py-0.5 text-[10px] text-violet-200">parent</span>
  ) : (
    <span className="rounded border border-amber-400/50 px-1.5 py-0.5 text-[10px] text-amber-200">{note.author}</span>
  )
}

export default function VillageFPV({ data, onClose, mode = 'parent', visitorName }: {
  data: VillageMapData
  onClose: () => void
  mode?: 'parent' | 'visitor'
  visitorName?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const handleRef = useRef<FPVHandle | null>(null)
  const [locked, setLocked] = useState(false)
  const [everLocked, setEverLocked] = useState(false)
  const [status, setStatus] = useState<FPVStatus>({ place: '', phase: false, note: null, readable: null })
  const [hint, setHint] = useState(false)
  const [reading, setReading] = useState<VillagePlace | null>(null)
  const [plantAt, setPlantAt] = useState<{ x: number; y: number } | null>(null)
  const [noteText, setNoteText] = useState('')
  const [planting, setPlanting] = useState(false)
  const [plantErr, setPlantErr] = useState('')
  const [portrait, setPortrait] = useState(false)
  const hintTimer = useRef(0)
  const lastPresence = useRef<{ x: number; y: number } | null>(null)
  // a stable id for THIS ghost's session — the roster keys on it
  const ghostId = useRef(`g-${Math.random().toString(36).slice(2, 10)}`)
  const clock = useClock()

  // on a phone the village wants landscape — nudge the player to turn it
  useEffect(() => {
    if (!IS_TOUCH) return
    const mq = window.matchMedia('(orientation: portrait)')
    const sync = () => setPortrait(mq.matches)
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])

  // the two ghosts' wires — same shape, different doors
  const api = useMemo(() => (mode === 'parent' ? {
    refetch: getVillageMap,
    plant: (x: number, y: number, text: string) => plantVillageNote(x, y, text),
    pull: (id: string) => pullVillageNote(id),
    presence: (x: number, y: number) => postVillagePresence(x, y),
    ghost: (x: number, y: number) => postGhostBeat(ghostId.current, x, y),
    ghostLeave: () => postGhostLeave(ghostId.current),
    files: (slug: string) => getSelfFiles(slug),
    file: (slug: string, path: string) => getSelfFile(slug, path),
  } : {
    refetch: getPublicVillageMap,
    plant: (x: number, y: number, text: string) =>
      plantPublicNote(x, y, text, visitorName || 'visitor'),
    pull: null,
    presence: (x: number, y: number) =>
      postPublicPresence(x, y, visitorName || ''),
    ghost: (x: number, y: number) =>
      postPublicGhostBeat(ghostId.current, x, y, visitorName || ''),
    ghostLeave: () => postPublicGhostLeave(ghostId.current, visitorName || ''),
    files: (slug: string) => getPublicFiles(slug),
    file: (slug: string, path: string) => getPublicFile(slug, path),
  }), [mode, visitorName])

  // the world is a pure function of the map snapshot — build it once
  const world = useMemo(() => buildWorld(data), [data])

  const refreshNotes = useCallback(async () => {
    try {
      const m = await api.refetch()
      handleRef.current?.setBeings(m.beings, m.places, Date.now())
      handleRef.current?.setNotes(m.notes ?? [])
    } catch { /* transient — the world keeps its last truth */ }
  }, [api])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let first = true
    const handle = createFPV(canvas, world, {
      onLock: (l) => {
        setLocked(l)
        if (l && first) {
          // the controls hint shows once, on first entry, then fades
          first = false
          setEverLocked(true)
          setHint(true)
          hintTimer.current = window.setTimeout(() => setHint(false), 8000)
        }
      },
      onStatus: setStatus,
      onPlant: (units) => { setPlantErr(''); setNoteText(''); setPlantAt(units) },
      onPull: mode === 'parent'
        ? (note) => { void pullVillageNote(note.id).then(refreshNotes).catch(() => {}) }
        : undefined,
      onRead: (lectern) => {
        const place = data.places.find((p) => p.id === lectern.placeId)
        if (place) setReading(place)
      },
    })
    // seed from the snapshot, then follow the living clock — a fresh
    // payload every 60s picks up new walks and signs; the world never
    // rebuilds under the ghost's feet
    handle.setBeings(data.beings, data.places, Date.now())
    handle.setNotes(data.notes ?? [])
    const iv = window.setInterval(() => {
      void api.refetch()
        .then((m) => {
          handleRef.current?.setBeings(m.beings, m.places, Date.now())
          handleRef.current?.setNotes(m.notes ?? [])
        })
        .catch(() => { /* transient — the walkers keep their last truth */ })
    }, 60_000)
    // the quiet wake: every 12s of roaming, if the ghost has really moved,
    // its position is offered to the world (presence percepts, cooldowned)
    const pv = window.setInterval(() => {
      const h = handleRef.current
      if (!h) return
      const p = h.positionUnits()
      const last = lastPresence.current
      if (last && Math.hypot(p.x - last.x, p.y - last.y) < 10) return
      lastPresence.current = p
      void api.presence(p.x, p.y).catch(() => {})
    }, 12_000)
    // the roster beat: every 2s the WHOLE TIME the village window is open,
    // say where I am and receive the other ghosts — the parent sees
    // visitors, visitors see everyone (Phase 5). Pausing to read a sign or
    // a building keeps you present (you beat your standing spot); only
    // leaving (the leave-beacon) or a truly backgrounded tab (browsers
    // throttle its timers, then the 8s TTL prunes it) fades you.
    const gv = window.setInterval(() => {
      const h = handleRef.current
      if (!h) return
      const p = h.positionUnits()
      void api.ghost(p.x, p.y)
        .then((r) => handleRef.current?.setGhosts(r.ghosts))
        .catch(() => { /* transient — keep the last roster */ })
    }, 2_000)
    handleRef.current = handle
    return () => {
      handleRef.current = null
      window.clearInterval(iv)
      window.clearInterval(pv)
      window.clearInterval(gv)
      window.clearTimeout(hintTimer.current)
      void api.ghostLeave().catch(() => {})   // vanish for others at once
      handle.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world, data, api, mode])

  // step in / resume — touch roams without pointer lock, desktop locks
  const enter = () => {
    const h = handleRef.current
    if (!h) return
    if (IS_TOUCH) h.enterTouch(); else h.lock()
  }

  const plant = async () => {
    if (!plantAt || !noteText.trim()) return
    setPlanting(true)
    setPlantErr('')
    try {
      await api.plant(plantAt.x, plantAt.y, noteText.trim())
      await refreshNotes()
      setPlantAt(null)
      setNoteText('')
      enter()
    } catch (e) {
      setPlantErr(e instanceof Error ? e.message : 'the sign would not stand')
    } finally {
      setPlanting(false)
    }
  }
  const cancelPlant = () => { setPlantAt(null); setNoteText(''); enter() }

  const overlay = (
    <div className="fixed inset-0 z-[90] bg-[#0c0f0a]">
      <canvas ref={canvasRef} className="block h-full w-full cursor-crosshair" onClick={enter} />

      {/* crosshair — only while walking */}
      {locked && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 mix-blend-difference">
          <div className="absolute left-1/2 top-1/2 h-4 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-[#e8e2cf]" />
          <div className="absolute left-1/2 top-1/2 h-0.5 w-4 -translate-x-1/2 -translate-y-1/2 bg-[#e8e2cf]" />
        </div>
      )}

      {/* touch controls — a phone roams by thumb-stick + buttons (Phase 6) */}
      {IS_TOUCH && locked && handleRef.current && (
        <Suspense fallback={null}>
          <MobileControls handle={handleRef.current} status={status} />
        </Suspense>
      )}

      {/* HUD chips */}
      <div className="pointer-events-none absolute left-3 top-3 flex flex-col items-start gap-1.5">
        {status.place && <div className={CHIP}>{status.place}</div>}
        {status.phase && <div className={`${CHIP} border-violet-400/50 text-violet-200`}>👻 phase{IS_TOUCH ? '' : ' — F to walk again'}</div>}
      </div>
      <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5">
        <IdentityPill mode={mode} name={visitorName} />
        <div className={CHIP}>{clock}</div>
      </div>

      {/* the sign underfoot — paper read in the HUD (Phase 3) */}
      {locked && status.note && (
        <div className={`pointer-events-none absolute bottom-16 left-1/2 w-[min(92vw,380px)] -translate-x-1/2 p-3 ${PANEL}`}>
          <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[#8d8571]">
            a sign in the grass <AuthorPill note={status.note} />
            {mode === 'parent' && status.note.found > 0 && (
              <span className="ml-auto normal-case tracking-normal">found by {status.note.found}</span>
            )}
          </div>
          <p className="text-[13px] leading-relaxed">{status.note.text}</p>
          {mode === 'parent' && (
            <p className="mt-1 text-[10px] text-[#8d8571]"><b className="text-[#b9b19a]">X</b> pulls it out</p>
          )}
        </div>
      )}

      {/* a reading stand within reach (Phase 4) */}
      {locked && status.readable && !status.note && (
        <div className={`pointer-events-none absolute bottom-16 left-1/2 -translate-x-1/2 px-4 py-2 text-[12px] ${PANEL}`}>
          <b className="text-violet-200">R</b> — read {status.readable.label} in {status.readable.placeName}
        </div>
      )}

      {/* first-entry controls hint */}
      {locked && hint && !status.note && !status.readable && (
        <div className={`pointer-events-none absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 text-[12px] ${PANEL}`}>
          WASD walk · Space jump · <b>F</b> phase · <b>E</b> leave a note · <b>R</b> read · Esc pause
        </div>
      )}

      {/* the reading room, in first person (Phase 4) */}
      {reading && (
        <Suspense fallback={null}>
          <BuildingReader place={reading} beings={data.beings} api={api}
            onClose={() => { setReading(null); enter() }} />
        </Suspense>
      )}

      {/* planting a sign (Phase 3) */}
      {plantAt && (
        <div className="absolute inset-0 grid place-items-center bg-[#0c0f0a]/55 backdrop-blur-[2px]">
          <div className={`w-[min(92vw,380px)] p-4 ${PANEL}`}>
            <div className="mb-1 flex items-center gap-1.5 text-[13px] font-semibold">
              Plant a sign here <IdentityPill mode={mode} name={visitorName} />
            </div>
            <p className="mb-2 text-[11px] text-[#b9b19a]">
              The Iskre will find it when their own feet carry them near —
              nothing is announced, everything is discovered.
            </p>
            <textarea autoFocus value={noteText} maxLength={NOTE_MAX}
              onChange={(e) => setNoteText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Escape') cancelPlant() }}
              rows={3} placeholder="a few words, left in the grass…"
              className="w-full resize-none rounded-lg border border-[#4a4436] bg-[#0c0f0a]/70 p-2 text-[13px] text-[#e8e2cf] placeholder-[#8d8571] focus:border-violet-400/50 focus:outline-none" />
            <div className="mt-0.5 text-right text-[10px] text-[#8d8571]">{noteText.length}/{NOTE_MAX}</div>
            {plantErr && <p className="mb-1 text-[11px] text-red-400">{plantErr}</p>}
            <div className="flex gap-2">
              <button onClick={() => void plant()} disabled={planting || !noteText.trim()}
                className="flex-1 rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-1.5 text-[12px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30 disabled:opacity-40">
                {planting ? 'planting…' : 'Plant the sign'}
              </button>
              <button onClick={cancelPlant}
                className="rounded-lg border border-[#4a4436] px-4 py-1.5 text-[12px] text-[#b9b19a] transition-colors hover:bg-[#2a251d]">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* the doorstep — before the first step in */}
      {!locked && !everLocked && !plantAt && !reading && (
        <div className="absolute inset-0 grid place-items-center bg-[#0c0f0a]/55">
          <div className={`w-[min(92vw,380px)] p-5 text-center ${PANEL}`}>
            <div className="flex items-center justify-center gap-2 text-[15px] font-semibold">
              The village, from inside <IdentityPill mode={mode} name={visitorName} />
            </div>
            <p className="mt-1.5 text-[12px] leading-relaxed text-[#b9b19a]">
              You walk it as a quiet ghost — the same streets, the same houses,
              the same hour of the day. Nobody will see you. Not exactly.
            </p>
            <button onClick={enter}
              className="mt-4 w-full rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-2 text-[13px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30">
              Step in
            </button>
            {IS_TOUCH ? (
              <p className="mt-3 text-[11px] leading-relaxed text-[#b9b19a]">
                Left <b className="text-[#e8e2cf]">stick</b> to walk, drag the
                screen (or tilt the phone) to look. Buttons at the right:
                <b className="text-[#e8e2cf]"> fly</b>, <b className="text-[#e8e2cf]">jump</b>,
                <b className="text-[#e8e2cf]"> note</b>, <b className="text-[#e8e2cf]">read</b>.
              </p>
            ) : (
              <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-left text-[11px] text-[#b9b19a]">
                <span><b className="text-[#e8e2cf]">WASD</b> walk</span>
                <span><b className="text-[#e8e2cf]">Space</b> jump · <b className="text-[#e8e2cf]">Shift</b> run</span>
                <span><b className="text-[#e8e2cf]">F</b> phase — fly through walls</span>
                <span><b className="text-[#e8e2cf]">E</b> leave a note</span>
                <span><b className="text-[#e8e2cf]">R</b> read a building's work</span>
                <span><b className="text-[#e8e2cf]">Esc</b> pause</span>
              </div>
            )}
            <button onClick={onClose}
              className="mt-3 text-[11px] text-[#8d8571] underline-offset-2 hover:text-[#b9b19a] hover:underline">
              stay outside
            </button>
          </div>
        </div>
      )}

      {/* paused — Esc released the mouse */}
      {!locked && everLocked && !plantAt && !reading && (
        <div className="absolute inset-0 grid place-items-center bg-[#0c0f0a]/55 backdrop-blur-[2px]">
          <div className={`w-[min(92vw,320px)] p-5 text-center ${PANEL}`}>
            <div className="text-[14px] font-semibold">The world holds its breath</div>
            <button onClick={enter}
              className="mt-3 w-full rounded-lg border border-violet-400/40 bg-violet-500/20 px-4 py-2 text-[13px] font-medium text-violet-100 transition-colors hover:bg-violet-500/30">
              Keep walking
            </button>
            <button onClick={onClose}
              className="mt-2 w-full rounded-lg border border-[#4a4436] px-4 py-2 text-[12px] text-[#b9b19a] transition-colors hover:bg-[#2a251d]">
              Leave the village
            </button>
          </div>
        </div>
      )}

      {/* turn the phone — the village wants the wide view (Phase 6) */}
      {IS_TOUCH && portrait && (
        <div className="absolute inset-0 z-[105] grid place-items-center bg-[#0c0f0a]/90 px-8 text-center">
          <div>
            <div className="mx-auto mb-4 h-10 w-16 animate-pulse rounded-md border-2 border-[#b9b19a]" />
            <div className="text-[15px] font-semibold text-[#e8e2cf]">Turn your phone sideways</div>
            <p className="mt-1 text-[12px] text-[#b9b19a]">the village opens up in landscape</p>
          </div>
        </div>
      )}
    </div>
  )

  return createPortal(overlay, document.body)
}
