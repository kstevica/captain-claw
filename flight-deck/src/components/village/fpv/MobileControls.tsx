// Touch controls for the village on a phone (FPV plan Phase 6): a left
// thumb-stick to walk, a right drag-anywhere zone to look (or the phone's
// own tilt, via the gyro toggle), and a cluster of action buttons — jump,
// fly, leave a note, read. Pointer events (not touch events) so each
// finger is captured independently: walk and look at the same time. Shown
// only on touch devices, only while roaming; it hides itself the moment a
// note form or a reading panel opens (those unlock the ghost).

import { useEffect, useRef, useState } from 'react'
import { BookOpen, ChevronsUp, Compass, Feather, PenLine } from 'lucide-react'
import type { FPVHandle, FPVStatus } from './engine'

const LOOK_SENS = 0.005
const STICK_R = 52   // px — joystick travel radius

export default function MobileControls({ handle, status }: {
  handle: FPVHandle
  status: FPVStatus
}) {
  const [thumb, setThumb] = useState({ x: 0, y: 0 })
  const [gyroOn, setGyroOn] = useState(false)
  const stickRef = useRef<HTMLDivElement | null>(null)
  const stickId = useRef<number | null>(null)
  const stickCenter = useRef({ x: 0, y: 0 })
  const lookId = useRef<number | null>(null)
  const lookPt = useRef({ x: 0, y: 0 })
  // the fly button reflects the engine's phase state, which the status feed
  // carries — no local mirror to drift out of sync
  const flying = status.phase

  // leave the world → let go of any held controls
  useEffect(() => () => handle.setMove(0, 0), [handle])

  // ── the look zone (everything behind the controls) ─────────────────────
  const lookDown = (e: React.PointerEvent) => {
    if (lookId.current !== null) return
    lookId.current = e.pointerId
    lookPt.current = { x: e.clientX, y: e.clientY }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const lookMove = (e: React.PointerEvent) => {
    if (e.pointerId !== lookId.current) return
    const dx = e.clientX - lookPt.current.x, dy = e.clientY - lookPt.current.y
    lookPt.current = { x: e.clientX, y: e.clientY }
    if (!gyroOn) handle.look(dx * LOOK_SENS, dy * LOOK_SENS)
  }
  const lookUp = (e: React.PointerEvent) => {
    if (e.pointerId === lookId.current) lookId.current = null
  }

  // ── the walk stick ─────────────────────────────────────────────────────
  const stickDown = (e: React.PointerEvent) => {
    e.stopPropagation()
    const r = stickRef.current!.getBoundingClientRect()
    stickCenter.current = { x: r.left + r.width / 2, y: r.top + r.height / 2 }
    stickId.current = e.pointerId
    e.currentTarget.setPointerCapture(e.pointerId)
    stickMove(e)
  }
  const stickMove = (e: React.PointerEvent) => {
    if (e.pointerId !== stickId.current) return
    let dx = e.clientX - stickCenter.current.x
    let dy = e.clientY - stickCenter.current.y
    const d = Math.hypot(dx, dy)
    if (d > STICK_R) { dx = (dx / d) * STICK_R; dy = (dy / d) * STICK_R }
    setThumb({ x: dx, y: dy })
    handle.setMove(dx / STICK_R, -dy / STICK_R)   // screen-up = forward
  }
  const stickUp = (e: React.PointerEvent) => {
    if (e.pointerId !== stickId.current) return
    stickId.current = null
    setThumb({ x: 0, y: 0 })
    handle.setMove(0, 0)
  }

  // ── actions ────────────────────────────────────────────────────────────
  const tap = (fn: () => void) => (e: React.PointerEvent) => {
    e.stopPropagation(); e.preventDefault(); fn()
  }
  const toggleGyro = async () => {
    const next = !gyroOn
    if (next) {
      const DOE = window.DeviceOrientationEvent as unknown as
        { requestPermission?: () => Promise<string> } | undefined
      if (DOE && typeof DOE.requestPermission === 'function') {
        try { if (await DOE.requestPermission() !== 'granted') return }
        catch { return }
      }
    }
    handle.setGyro(next); setGyroOn(next)
  }

  const BTN = 'grid h-14 w-14 place-items-center rounded-full border text-[#e8e2cf] backdrop-blur-sm active:scale-95 transition-transform'
  const idle = 'border-[#4a4436] bg-[#171410]/70'
  const on = 'border-violet-400/60 bg-violet-500/25 text-violet-100'

  return (
    <div className="absolute inset-0 z-[95] touch-none select-none">
      {/* look zone — a drag anywhere that isn't a control turns the view */}
      <div className="absolute inset-0" onPointerDown={lookDown}
        onPointerMove={lookMove} onPointerUp={lookUp} onPointerCancel={lookUp} />

      {/* gyro toggle — top center */}
      <button onPointerDown={tap(() => void toggleGyro())}
        className={`absolute left-1/2 top-3 -translate-x-1/2 flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[12px] backdrop-blur-sm active:scale-95 ${gyroOn ? on : idle}`}>
        <Compass className="h-4 w-4" /> {gyroOn ? 'tilt to look' : 'drag to look'}
      </button>

      {/* walk stick — bottom left */}
      <div ref={stickRef}
        className="absolute bottom-8 left-6 h-32 w-32 rounded-full border border-[#4a4436] bg-[#171410]/45 backdrop-blur-sm"
        onPointerDown={stickDown} onPointerMove={stickMove}
        onPointerUp={stickUp} onPointerCancel={stickUp}>
        <div className="absolute left-1/2 top-1/2 h-14 w-14 rounded-full border border-[#6a6350] bg-[#e8e2cf]/25"
          style={{ transform: `translate(calc(-50% + ${thumb.x}px), calc(-50% + ${thumb.y}px))` }} />
      </div>

      {/* action buttons — bottom right */}
      <div className="absolute bottom-8 right-6 flex flex-col items-end gap-3">
        <div className="flex gap-3">
          <button onPointerDown={tap(() => handle.toggleFly())}
            className={`${BTN} ${flying ? on : idle}`} aria-label="fly">
            <Feather className="h-6 w-6" />
          </button>
          <button onPointerDown={tap(() => handle.jump())}
            className={`${BTN} ${idle}`} aria-label="jump">
            <ChevronsUp className="h-7 w-7" />
          </button>
        </div>
        <div className="flex gap-3">
          <button onPointerDown={tap(() => handle.note())}
            className={`${BTN} ${idle}`} aria-label="leave a note">
            <PenLine className="h-6 w-6" />
          </button>
          <button onPointerDown={tap(() => handle.read())}
            disabled={!status.readable}
            className={`${BTN} ${status.readable ? on : 'border-[#3a352a] bg-[#171410]/40 text-[#6a6350]'}`}
            aria-label="read">
            <BookOpen className="h-6 w-6" />
          </button>
        </div>
      </div>
    </div>
  )
}
