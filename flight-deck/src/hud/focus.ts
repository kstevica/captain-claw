// D-pad focus engine for the HUD.
//
// Input on glasses (Meta Ray-Ban Display, Rokid Lumen): swipes arrive as
// ArrowUp/Down/Left/Right keydowns, a pinch as Enter (and sometimes ALSO a
// tap/click within ~500 ms), Back as history.back(). There is no pointer and no
// physical keyboard. We follow Meta's UI toolkit (FocusNavigationProvider):
// own the arrow keys (preventDefault) and move focus geometrically, never mix
// with native spatial navigation. On top of that:
//   - a focused element taller than its scroller is read by scrolling it
//     page-by-page before focus moves on (long replies, big tables);
//   - first/last focus stops scroll the scroller to its absolute ends;
//   - "Unidentified" keydowns (ambiguous Neural Band events) are ignored;
//   - after Back or the text composer closing, the glasses reset focus to the
//     first control — pinFocus() puts it back where it belongs.
//
// Interactive things are `<div role="button" tabIndex={0}>` (see hooks.ts
// useActivate): the legacy glasses page found the Display's own focus walk
// only lands on tabindexed elements, and divs work with our engine too.

import { IS_GLASSES } from './device'
import { goBack, takeRestoreFocusKey } from './router'

export type Direction = 'up' | 'down' | 'left' | 'right'

const DIRS: Record<string, Direction> = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
}

export const FOCUSABLE_SELECTOR = [
  '[tabindex]:not([tabindex="-1"])',
  'textarea:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'button:not([disabled])',
  'a[href]',
].join(', ')

/** The one screen currently mounted (focus never leaves it). */
function screenRoot(): HTMLElement {
  return document.querySelector<HTMLElement>('.hud-screen') ?? document.body
}

function isVisible(el: HTMLElement): boolean {
  if (el.closest('[inert], [aria-hidden="true"], [hidden]')) return false
  if (el.getAttribute('aria-disabled') === 'true') return false
  const r = el.getBoundingClientRect()
  if (r.width === 0 && r.height === 0) return false
  return getComputedStyle(el).visibility !== 'hidden'
}

export function focusables(root: ParentNode = screenRoot()): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(isVisible)
}

function isTextField(el: Element | null): el is HTMLInputElement | HTMLTextAreaElement {
  if (!el) return false
  if (el instanceof HTMLTextAreaElement) return true
  if (el instanceof HTMLInputElement) {
    return !['button', 'submit', 'checkbox', 'radio', 'range', 'color', 'file', 'reset'].includes(el.type)
  }
  return (el as HTMLElement).isContentEditable === true
}

/** Nearest ancestor that scrolls vertically. */
export function scrollParent(el: Element | null): HTMLElement | null {
  let p = el?.parentElement ?? null
  while (p && p !== document.body) {
    const oy = getComputedStyle(p).overflowY
    if ((oy === 'auto' || oy === 'scroll') && p.scrollHeight > p.clientHeight + 1) return p
    p = p.parentElement
  }
  return null
}

// ── Geometric candidate search (beam + weighted distance, as in Chromium /
// Meta's FocusNavigationProvider) ──

function isCandidate(dir: Direction, s: DOMRect, d: DOMRect): boolean {
  switch (dir) {
    case 'left': return (s.right > d.right || s.left >= d.right) && s.left > d.left
    case 'right': return (s.left < d.left || s.right <= d.left) && s.right < d.right
    case 'up': return (s.bottom > d.bottom || s.top >= d.bottom) && s.top > d.top
    case 'down': return (s.top < d.top || s.bottom <= d.top) && s.bottom < d.bottom
  }
}

function inBeam(dir: Direction, s: DOMRect, d: DOMRect): boolean {
  if (dir === 'left' || dir === 'right') return d.bottom > s.top && d.top < s.bottom
  return d.right > s.left && d.left < s.right
}

function majorDist(dir: Direction, s: DOMRect, d: DOMRect): number {
  switch (dir) {
    case 'left': return Math.max(0, s.left - d.right)
    case 'right': return Math.max(0, d.left - s.right)
    case 'up': return Math.max(0, s.top - d.bottom)
    case 'down': return Math.max(0, d.top - s.bottom)
  }
}

function minorDist(dir: Direction, s: DOMRect, d: DOMRect): number {
  if (dir === 'left' || dir === 'right') return Math.abs((s.top + s.bottom) / 2 - (d.top + d.bottom) / 2)
  return Math.abs((s.left + s.right) / 2 - (d.left + d.right) / 2)
}

function better(dir: Direction, s: DOMRect, cand: DOMRect, best: DOMRect | null): boolean {
  if (!best) return true
  const cb = inBeam(dir, s, cand)
  const bb = inBeam(dir, s, best)
  if (cb && !bb) return true
  if (bb && !cb) return false
  const score = (r: DOMRect) => 13 * majorDist(dir, s, r) ** 2 + minorDist(dir, s, r) ** 2
  return score(cand) < score(best)
}

export function findNearest(current: HTMLElement, dir: Direction, candidates = focusables()): HTMLElement | null {
  const s = current.getBoundingClientRect()
  let best: HTMLElement | null = null
  let bestRect: DOMRect | null = null
  for (const c of candidates) {
    if (c === current || c.contains(current) || current.contains(c)) continue
    const r = c.getBoundingClientRect()
    if (!isCandidate(dir, s, r)) continue
    if (better(dir, s, r, bestRect)) { best = c; bestRect = r }
  }
  return best
}

// ── Scrolling helpers ──

const EDGE = 6 // px of slack before we call an element "cut off"

function pageStep(scroller: HTMLElement): number {
  return Math.max(40, Math.round(scroller.clientHeight * 0.8))
}

/**
 * Bring `el` into view inside its scroller. The first stop pins the scroller
 * to its top and a short last stop to its bottom (Meta: first/last focus must
 * reach the absolute ends). A tall element shows its top — or its bottom when
 * entered moving up, so reading continues backwards page by page.
 */
export function ensureVisible(el: HTMLElement, dir?: Direction): void {
  const scroller = scrollParent(el)
  if (!scroller) return
  const s = scroller.getBoundingClientRect()
  const r = el.getBoundingClientRect()
  const tall = r.height > s.height - 24
  const all = focusables(scroller)
  if (all[0] === el && !(tall && dir === 'up')) { scroller.scrollTop = 0; return }
  if (all[all.length - 1] === el && !tall) { scroller.scrollTop = scroller.scrollHeight; return }
  if (tall) {
    if (dir === 'up') scroller.scrollTop += r.bottom - s.bottom + 12
    else if (r.top < s.top + EDGE || r.top > s.top + 24) scroller.scrollTop += r.top - s.top - 12
    return
  }
  if (r.top < s.top + EDGE) scroller.scrollTop -= (s.top - r.top) + 12
  else if (r.bottom > s.bottom - EDGE) scroller.scrollTop += r.bottom - s.bottom + 12
}

let selfFocus = false

/** Focus `el` ourselves (no native scroll jump) and reveal it. */
export function focusEl(el: HTMLElement, dir?: Direction): void {
  selfFocus = true
  try { el.focus({ preventScroll: true }) } catch { el.focus() } finally { selfFocus = false }
  ensureVisible(el, dir)
}

/** One D-pad step. Returns true when it did something (caller prevents default). */
export function moveFocus(dir: Direction): boolean {
  const root = screenRoot()
  const active = document.activeElement instanceof HTMLElement ? document.activeElement : null
  if (!active || active === document.body || !root.contains(active)) {
    return focusInitial({ force: true })
  }
  const vertical = dir === 'up' || dir === 'down'
  const scroller = vertical ? scrollParent(active) : null
  if (scroller) {
    // Read a tall element page by page before leaving it.
    const s = scroller.getBoundingClientRect()
    const a = active.getBoundingClientRect()
    if (dir === 'down' && a.bottom > s.bottom + EDGE && a.top < s.bottom) {
      scroller.scrollTop += Math.min(a.bottom - s.bottom + 12, pageStep(scroller))
      return true
    }
    if (dir === 'up' && a.top < s.top - EDGE && a.bottom > s.top) {
      scroller.scrollTop -= Math.min(s.top - a.top + 12, pageStep(scroller))
      return true
    }
  }
  const next = findNearest(active, dir)
  if (next) { focusEl(next, dir); return true }
  if (scroller) {
    const before = scroller.scrollTop
    scroller.scrollTop += dir === 'down' ? pageStep(scroller) : -pageStep(scroller)
    return scroller.scrollTop !== before
  }
  return false
}

/**
 * Put focus somewhere sensible on the current screen: the element Back should
 * restore (data-fk), else [data-autofocus], else the first stop in the scroll
 * area, else the first stop anywhere. Leaves focus alone if it is already on
 * this screen unless `force`.
 */
export function focusInitial(opts?: { force?: boolean }): boolean {
  const root = screenRoot()
  const active = document.activeElement
  if (!opts?.force && active instanceof HTMLElement && active !== document.body && root.contains(active)) {
    return false
  }
  const key = takeRestoreFocusKey()
  let target: HTMLElement | null = null
  if (key) {
    target = Array.from(root.querySelectorAll<HTMLElement>('[data-fk]')).find((el) => el.dataset.fk === key) ?? null
    if (target && !target.matches(FOCUSABLE_SELECTOR)) {
      target = target.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)
    }
  }
  if (!target) target = root.querySelector<HTMLElement>('[data-autofocus]')
  if (!target) target = focusables(root.querySelector('.hud-scroll') ?? root)[0] ?? null
  if (!target) target = focusables(root)[0] ?? null
  if (!target) return false
  focusEl(target)
  if (key || IS_GLASSES) pinFocus(target)
  return true
}

// ── Focus pinning (glasses reset focus after Back / composer close) ──

let pin: { el: HTMLElement; until: number; arrowsAtPin: number } | null = null
let arrowCount = 0

/** Keep focus on `el` for `ms` unless the user presses an arrow meanwhile. */
export function pinFocus(el: HTMLElement, ms = 700): void {
  pin = { el, until: Date.now() + ms, arrowsAtPin: arrowCount }
}

function onFocusIn(e: FocusEvent) {
  const t = e.target
  if (!(t instanceof HTMLElement)) return
  if (pin) {
    if (Date.now() > pin.until || arrowCount !== pin.arrowsAtPin || !pin.el.isConnected) {
      pin = null
    } else if (t !== pin.el) {
      // Our own deliberate moves win over the pin; anything else (the host
      // resetting focus) is undone.
      if (selfFocus) {
        pin = null
      } else {
        const el = pin.el
        setTimeout(() => { if (el.isConnected) focusEl(el) }, 0)
        return
      }
    }
  }
  // Focus moved by the host's own walk / a tap: keep it in view.
  if (!selfFocus && screenRoot().contains(t)) ensureVisible(t)
}

// ── Activation de-duplication ──

let lastActivation: { el: Element | null; at: number } = { el: null, at: 0 }
let quietUntil = 0

/**
 * A single pinch can arrive as BOTH a click and an Enter keydown (in either
 * order, within ~500 ms). Returns false for the echo so handlers run once.
 */
export function claimActivation(el: Element): boolean {
  const now = Date.now()
  if (now < quietUntil) return false
  if (lastActivation.el === el && now - lastActivation.at < 500) return false
  lastActivation = { el, at: now }
  return true
}

/** Ignore every activation for `ms` (the late pinch that closed the composer). */
export function quietActivations(ms = 450): void {
  quietUntil = Date.now() + ms
}

// ── Global key handling ──

function onKeyDown(e: KeyboardEvent) {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
  if (e.key === 'Unidentified') return
  const dir = DIRS[e.key]
  if (dir) {
    const active = document.activeElement
    if (isTextField(active) && !IS_GLASSES) {
      // Desktop typing: keep caret movement inside the field; leave the
      // field only from its first (up) / last (down) position.
      const f = active as HTMLInputElement | HTMLTextAreaElement
      const atStart = f.selectionStart === 0 && f.selectionEnd === 0
      const atEnd = f.selectionStart === f.value.length
      if (dir === 'left' || dir === 'right') return
      if (dir === 'up' && !atStart && f.value.includes('\n')) return
      if (dir === 'down' && !atEnd && f.value.includes('\n')) return
    }
    arrowCount++
    if (moveFocus(dir)) e.preventDefault()
    return
  }
  // Desktop / simulator Back. Never on the Display itself: its shell already
  // turns the Back gesture into history.back(), so handling Escape too would
  // go back twice.
  if (!IS_GLASSES && (e.key === 'Escape' || (e.key === 'Backspace' && !isTextField(document.activeElement)))) {
    e.preventDefault()
    goBack()
  }
}

let installedEngine = false

/** Install the global listeners once. Returns an uninstaller. */
export function installFocusEngine(): () => void {
  if (installedEngine) return () => {}
  installedEngine = true
  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('focusin', onFocusIn)
  return () => {
    document.removeEventListener('keydown', onKeyDown)
    document.removeEventListener('focusin', onFocusIn)
    installedEngine = false
  }
}
