import { useState } from 'react'

// Drag-to-resize with localStorage persistence. axis 'x' → width (col-resize),
// axis 'y' → height (row-resize). `grow` says which drag direction makes the
// tracked size bigger: 'forward' (default) grows on a rightward/downward drag
// — the handle sits on the panel's right/bottom edge — while 'backward' grows
// on a leftward/upward drag, for a handle on the panel's LEFT/top edge (a
// column docked to the right side of the screen).
export function usePersistedSize(
  key: string,
  def: number,
  min: number,
  max: number,
  axis: 'x' | 'y',
  grow: 'forward' | 'backward' = 'forward',
  // Optional live ceiling (e.g. the container's current height). The tracked
  // size is capped by it, and a stale oversized persisted value self-heals on
  // the next mount/drag — so a divider dragged tall on a big monitor can't
  // strand itself off-screen when reopened on a small one.
  liveMax?: number,
) {
  const cap = Math.max(min, Math.min(max, liveMax ?? max))
  const [size, setSize] = useState<number>(() => {
    let v = NaN
    try { v = Number(localStorage.getItem(key)) } catch { /* private mode */ }
    return v >= min && v <= max ? v : def
  })
  // The rendered size is always capped by the live ceiling, so a stale
  // oversized persisted value can never paint off-screen; it heals to the
  // clamped value on the next drag (onUp persists Math.min(cap, ...)).
  const clamped = Math.min(size, cap)
  const onResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    const startPos = axis === 'x' ? e.clientX : e.clientY
    const startSize = clamped
    const sign = grow === 'backward' ? -1 : 1
    document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: MouseEvent) => {
      const pos = axis === 'x' ? ev.clientX : ev.clientY
      setSize(Math.min(cap, Math.max(min, startSize + sign * (pos - startPos))))
    }
    const onUp = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      setSize((s) => {
        const v = Math.min(cap, Math.max(min, s))
        try { localStorage.setItem(key, String(Math.round(v))) } catch { /* ignore */ }
        return v
      })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }
  return { size: clamped, onResizeStart }
}
