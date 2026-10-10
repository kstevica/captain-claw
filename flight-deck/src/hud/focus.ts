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
//     first control — focusInitial() overrides it after Back, and pinFocus()
//     puts it back if it comes later;
//   - a focused control that disappears (Stop when the reply ends, a chip
//     re-keyed) hands focus to its replacement or nearest neighbour, so
//     there is always one focus owner for the next pinch.
//
// Interactive things are `<div role="button" tabIndex={0}>` (see hooks.ts
// useActivate): the legacy glasses page found the Display's own focus walk
// only lands on tabindexed elements, and divs work with our engine too.

import { IS_GLASSES } from './device'
import { clearBackPending, goBack, isBackPending, takeRestoreFocusKey } from './router'

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

// Called for every candidate on every D-pad step, on a CPU ~12× slower than a
// laptop and with long documents holding hundreds of reading blocks — so no
// getComputedStyle here: an empty client-rect list covers display:none, and
// [hidden]/[inert]/aria-hidden cover the rest of what the HUD uses.
function isVisible(el: HTMLElement): boolean {
  if (el.getAttribute('aria-disabled') === 'true') return false
  if (el.getClientRects().length === 0) return false
  return !el.closest('[inert], [aria-hidden="true"], [hidden]')
}

export function focusables(root: ParentNode = screenRoot()): HTMLElement[] {
  return Array.from(stops(root)).filter(isVisible)
}

/** Every stop under `root` in DOM order, unfiltered — the per-keypress paths
 *  check visibility only on the few elements they actually pick. */
function stops(root: ParentNode): NodeListOf<HTMLElement> {
  return root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
}

/** First (or last) visible stop of `list`, scanning in from that end. */
function edgeStop(list: ArrayLike<HTMLElement>, last = false): HTMLElement | null {
  const n = list.length
  for (let k = 0; k < n; k++) {
    const el = list[last ? n - 1 - k : k]
    if (isVisible(el)) return el
  }
  return null
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

function score(dir: Direction, s: DOMRect, r: DOMRect): number {
  return 13 * majorDist(dir, s, r) ** 2 + minorDist(dir, s, r) ** 2
}

/** The best candidate so far: in-beam beats out-of-beam, then lowest score. */
interface Pick { el: HTMLElement | null; beam: boolean; score: number }

function consider(p: Pick, dir: Direction, s: DOMRect, c: HTMLElement, r: DOMRect): void {
  if (!isCandidate(dir, s, r)) return
  const beam = inBeam(dir, s, r)
  if (p.el && p.beam && !beam) return
  const sc = score(dir, s, r)
  if (p.el && p.beam === beam && sc >= p.score) return
  if (!isVisible(c)) return // only for a would-be winner: cheap on long lists
  p.el = c; p.beam = beam; p.score = sc
}

function intersect(a: DOMRect, b: DOMRect): DOMRect | null {
  const left = Math.max(a.left, b.left), right = Math.min(a.right, b.right)
  const top = Math.max(a.top, b.top), bottom = Math.min(a.bottom, b.bottom)
  return right > left && bottom > top ? new DOMRect(left, top, right - left, bottom - top) : null
}

/**
 * Nearest stop from `current` in `dir` among `list`. With `clip`, stops
 * inside `clip.area` count only by the part visible in it — content scrolled
 * out of view sits under the header / footer in viewport coordinates and must
 * not be picked from there.
 */
export function findNearest(
  current: HTMLElement, dir: Direction, list: ArrayLike<HTMLElement> = stops(screenRoot()),
  clip?: { area: HTMLElement; box: DOMRect },
): HTMLElement | null {
  const s = current.getBoundingClientRect()
  const p: Pick = { el: null, beam: false, score: Infinity }
  for (let i = 0; i < list.length; i++) {
    const c = list[i]
    if (c === current || c.contains(current) || current.contains(c)) continue
    let r: DOMRect | null = c.getBoundingClientRect()
    if (!r.width && !r.height) continue // display:none
    if (clip && clip.area.contains(c)) r = intersect(r, clip.box)
    if (r) consider(p, dir, s, c, r)
  }
  return p.el
}

/** px: in-beam stops this close in distance count as one row. */
const ROW_SLACK = 8

/**
 * findNearest for an Up/Down step among the stops of one scroller, which lay
 * out as a single column (blocks, rows, rows of chips). Reading order: the
 * nearest row in that direction wins, then the stop in it closest sideways
 * — so a narrow file-chip row under a reply is a step of its own, not
 * outscored by the wide block beyond it. Walks outward from `current` in DOM
 * order (nearest first) and stops once a whole row lies past the best one: a
 * step through a long file measures a handful of blocks, not all of them.
 */
function findNearestInFlow(current: HTMLElement, dir: 'up' | 'down', list: NodeListOf<HTMLElement>): HTMLElement | null {
  const at = Array.prototype.indexOf.call(list, current) as number
  if (at < 0) return findNearest(current, dir, list)
  const s = current.getBoundingClientRect()
  const step = dir === 'down' ? 1 : -1
  let best: HTMLElement | null = null
  let bestMajor = Infinity
  let bestMinor = Infinity
  const aside: Pick = { el: null, beam: false, score: Infinity } // out of beam, only if nothing is in it
  let limit: number | null = null
  for (let i = at + step; i >= 0 && i < list.length; i += step) {
    const c = list[i]
    if (c.contains(current) || current.contains(c)) continue
    const r = c.getBoundingClientRect()
    if (!r.width && !r.height) continue
    // Past `limit` every stop starts further away than the best row: done.
    if (limit !== null && (dir === 'down' ? r.top >= limit : r.bottom <= limit)) break
    if (best && limit === null && (dir === 'down' ? r.bottom - s.bottom : s.top - r.top) > bestMajor + ROW_SLACK) {
      limit = dir === 'down' ? r.bottom : r.top
    }
    if (!isCandidate(dir, s, r)) continue
    if (!inBeam(dir, s, r)) { if (!best) consider(aside, dir, s, c, r); continue }
    const major = majorDist(dir, s, r)
    const minor = minorDist(dir, s, r)
    const nearer = major < bestMajor - ROW_SLACK || (major <= bestMajor + ROW_SLACK && minor < bestMinor)
    if (!nearer || !isVisible(c)) continue
    best = c; bestMajor = major; bestMinor = minor
  }
  return best ?? aside.el
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
export function ensureVisible(el: HTMLElement, dir?: Direction, known?: { scroller: HTMLElement; list: NodeListOf<HTMLElement> }): void {
  const scroller = scrollParent(el)
  if (!scroller) return
  const s = scroller.getBoundingClientRect()
  const r = el.getBoundingClientRect()
  const tall = r.height > s.height - 24
  // Only the two end stops matter: scan in from each end, not the whole list.
  const list = known?.scroller === scroller ? known.list : stops(scroller)
  if (edgeStop(list) === el && !(tall && dir === 'up')) { scroller.scrollTop = 0; return }
  if (!tall && edgeStop(list, true) === el) { scroller.scrollTop = scroller.scrollHeight; return }
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
export function focusEl(el: HTMLElement, dir?: Direction, known?: { scroller: HTMLElement; list: NodeListOf<HTMLElement> }): void {
  selfFocus = true
  try { el.focus({ preventScroll: true }) } catch { el.focus() } finally { selfFocus = false }
  ensureVisible(el, dir, known)
}

/** One D-pad step. Returns true when it did something (caller prevents default). */
export function moveFocus(dir: Direction): boolean {
  const root = screenRoot()
  const active = document.activeElement instanceof HTMLElement ? document.activeElement : null
  if (!active || active === document.body || !root.contains(active)) {
    return focusInitial({ force: true })
  }
  const scroller = dir === 'up' || dir === 'down' ? scrollParent(active) : null
  if (scroller) {
    const vdir = dir as 'up' | 'down'
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
    // Stay in the scroller while it has a stop that way; else scroll it
    // (Meta: if focus can't move but the owning scroller can, scroll); only
    // at its end move on to the header tabs / footer.
    const list = stops(scroller)
    let next = findNearestInFlow(active, vdir, list)
    if (next) { focusEl(next, dir, { scroller, list }); return true }
    const before = scroller.scrollTop
    scroller.scrollTop += dir === 'down' ? pageStep(scroller) : -pageStep(scroller)
    if (scroller.scrollTop !== before) return true
    next = findNearest(active, dir, Array.from(stops(root)).filter((c) => !scroller.contains(c)))
    if (next) { focusEl(next, dir); return true }
    return false
  }
  // From the header / footer (or sideways): the whole screen, but the
  // scroller's stops only where they are actually visible.
  const area = root.querySelector<HTMLElement>('.hud-scroll')
  const clip = area && !area.contains(active) ? { area, box: area.getBoundingClientRect() } : undefined
  const next = findNearest(active, dir, stops(root), clip)
  if (next) { focusEl(next, dir); return true }
  return false
}

/** The stop carrying (or inside the holder of) data-fk `key`. */
function byFocusKey(root: HTMLElement, key: string): HTMLElement | null {
  const holder = Array.from(root.querySelectorAll<HTMLElement>('[data-fk]')).find((el) => el.dataset.fk === key) ?? null
  if (!holder || holder.matches(FOCUSABLE_SELECTOR)) return holder
  return holder.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)
}

/** A Back landed, its screen hasn't placed focus yet, and the wearer hasn't
 *  swiped since. */
function restorePending(): boolean {
  return isBackPending() && arrowCount === arrowsAtPop
}

/**
 * Put focus somewhere sensible on the current screen: the element Back should
 * restore (data-fk), else [data-autofocus], else the first stop in the scroll
 * area, else the first stop anywhere. Leaves focus alone if it is already on
 * this screen unless `force` — or unless this is the first placement after a
 * Back: the glasses reset focus to the first control right after Back, and
 * that must not beat the row the wearer came from.
 */
export function focusInitial(opts?: { force?: boolean }): boolean {
  const root = screenRoot()
  const active = document.activeElement
  const afterBack = restorePending()
  if (!opts?.force && !afterBack && active instanceof HTMLElement && active !== document.body && root.contains(active)) {
    return false
  }
  const key = takeRestoreFocusKey()
  let target: HTMLElement | null = key ? byFocusKey(root, key) : null
  if (!target) target = root.querySelector<HTMLElement>('[data-autofocus]')
  if (!target) target = edgeStop(stops(root.querySelector('.hud-scroll') ?? root))
  if (!target) target = edgeStop(stops(root))
  if (!target) return false
  clearBackPending()
  focusEl(target)
  if (key || afterBack || IS_GLASSES) pinFocus(target)
  return true
}

// ── Focus pinning (glasses reset focus after Back / composer close) ──

let pin: { el: HTMLElement; until: number; arrowsAtPin: number } | null = null
/** Arrow presses so far (the wearer's own moves), and that count at the
 *  last popstate. */
let arrowCount = 0
let arrowsAtPop = 0

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

/**
 * The focused control was removed (Stop once the reply ends, Reconnect once
 * connected, re-keyed chips, an agent row moving group): focus fell to
 * <body>, the ring vanished and the next pinch would hit nothing. Hand it to
 * the control's replacement (same data-fk) or its nearest neighbour.
 */
function onFocusOut(e: FocusEvent) {
  if (e.relatedTarget) return
  const lost = e.target
  if (!(lost instanceof HTMLElement)) return
  const root = screenRoot()
  if (!root.contains(lost)) return
  // Chromium fires this during the removal, while `lost` is still in place.
  const fk = lost.closest<HTMLElement>('[data-fk]')?.dataset.fk
  const was = lost.getBoundingClientRect()
  requestAnimationFrame(() => {
    // Still in the page: a plain blur — the host's text composer opening, the
    // WebView losing focus — not ours to undo. A replaced screen places its
    // own focus (useAutoFocus), as does one still to restore after Back.
    if (lost.isConnected || !root.isConnected || restorePending()) return
    const a = document.activeElement
    if (a && a !== document.body) return
    let target = fk ? byFocusKey(root, fk) : null
    if (!target || !isVisible(target)) {
      target = null
      let best = Infinity
      const cx = (was.left + was.right) / 2, cy = (was.top + was.bottom) / 2
      for (const c of focusables(root)) {
        const r = c.getBoundingClientRect()
        const d = Math.hypot((r.left + r.right) / 2 - cx, (r.top + r.bottom) / 2 - cy)
        if (d < best) { best = d; target = c }
      }
    }
    if (target) focusEl(target)
    else focusInitial({ force: true })
  })
}

// ── Activation de-duplication ──

const ECHO_MS = 500
let lastActivation = { at: 0, arrows: -1 }
let quietUntil = 0

/** Within the echo window of the last activation, with no swipe since. */
function isEcho(now: number): boolean {
  return now < quietUntil || (now - lastActivation.at < ECHO_MS && arrowCount === lastActivation.arrows)
}

/**
 * A single pinch can arrive as BOTH a click and an Enter keydown (in either
 * order, within ~500 ms), and both go to whatever is focused when each is
 * dispatched — which the first half may already have changed (a tab switch
 * focuses the first row, Approve focuses the composer). So the echo is
 * recognised by time, not target: any activation within ECHO_MS of the last
 * one is dropped unless the wearer swiped in between. Returns false for it.
 */
export function claimActivation(): boolean {
  const now = Date.now()
  if (isEcho(now)) return false
  lastActivation = { at: now, arrows: arrowCount }
  return true
}

/** Ignore every activation for `ms` (the late pinch that closed the composer). */
export function quietActivations(ms = 450): void {
  quietUntil = Date.now() + ms
}

/** The Enter half of a pinch that landed on a text field focused by the
 *  first half (Approve, Send, Stop → composer) would insert a newline or
 *  open the system composer: swallow it before the field sees it. */
function onKeyDownCapture(e: KeyboardEvent) {
  if (e.key !== 'Enter' || e.isComposing || !isTextField(e.target as Element | null)) return
  if (!isEcho(Date.now())) return
  e.preventDefault()
  e.stopPropagation()
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
  // go back twice. Other hosts (Rokid Lumen) may send Escape AND go back
  // themselves, so ours waits a beat and stands down if a popstate arrives.
  if (!IS_GLASSES && (e.key === 'Escape' || (e.key === 'Backspace' && !isTextField(document.activeElement)))) {
    e.preventDefault()
    if (Date.now() - lastPopAt < 400) return // the host already went back
    if (pendingBack) clearTimeout(pendingBack)
    pendingBack = setTimeout(() => { pendingBack = null; goBack() }, 250)
  }
}

let pendingBack: ReturnType<typeof setTimeout> | null = null
let lastPopAt = 0

function onPopState() {
  lastPopAt = Date.now()
  arrowsAtPop = arrowCount
  if (pendingBack) { clearTimeout(pendingBack); pendingBack = null }
}

let installedEngine = false

/** Install the global listeners once. Returns an uninstaller. */
export function installFocusEngine(): () => void {
  if (installedEngine) return () => {}
  installedEngine = true
  document.addEventListener('keydown', onKeyDownCapture, true)
  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('focusin', onFocusIn)
  document.addEventListener('focusout', onFocusOut)
  window.addEventListener('popstate', onPopState)
  return () => {
    document.removeEventListener('keydown', onKeyDownCapture, true)
    document.removeEventListener('keydown', onKeyDown)
    document.removeEventListener('focusin', onFocusIn)
    document.removeEventListener('focusout', onFocusOut)
    window.removeEventListener('popstate', onPopState)
    installedEngine = false
  }
}
