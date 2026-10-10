// Small React hooks shared by every HUD screen.

import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, MouseEvent } from 'react'
import { claimActivation, focusInitial } from './focus'

export interface ActivateProps {
  role: 'button'
  tabIndex: 0 | -1
  'aria-disabled'?: true
  onClick: (e: MouseEvent<HTMLElement>) => void
  onKeyDown: (e: KeyboardEvent<HTMLElement>) => void
}

/**
 * Props that make any element a D-pad button: focusable, activated by a
 * pinch / Enter / Space / click — exactly once per pinch (see claimActivation).
 * Spread onto a <div>: `<div className="hud-btn" {...useActivate(fn)}>`.
 */
export function useActivate(fn: (() => void) | null | undefined, opts?: { disabled?: boolean }): ActivateProps {
  const ref = useRef(fn)
  useEffect(() => { ref.current = fn })
  const disabled = !!opts?.disabled || !fn
  const run = useCallback((el: Element) => {
    if (disabled || !ref.current) return
    if (!claimActivation(el)) return
    ref.current()
  }, [disabled])
  return {
    role: 'button',
    tabIndex: disabled ? -1 : 0,
    ...(disabled ? { 'aria-disabled': true as const } : {}),
    onClick: (e) => { e.preventDefault(); run(e.currentTarget) },
    onKeyDown: (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        run(e.currentTarget)
      }
    },
  }
}

/**
 * Focus the screen's first stop (or the one Back should restore) once its
 * content is ready. Never steals focus that is already on this screen.
 */
export function useAutoFocus(ready: boolean): void {
  useEffect(() => {
    if (!ready) return
    // Two frames: let React commit and the browser lay out first.
    let raf2 = 0
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => { focusInitial() })
    })
    return () => { cancelAnimationFrame(raf1); cancelAnimationFrame(raf2) }
  }, [ready])
}

/** Current time, refreshed every `ms` (header clock, relative times). */
export function useNow(ms = 15_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(id)
  }, [ms])
  return now
}
