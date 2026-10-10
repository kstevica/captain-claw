// Tiny history router for the HUD.
//
// Meta Ray-Ban Display has no in-app Back button: the system Back gesture
// calls history.back() while `navigation.canGoBack`, and opens the native
// menu at the root. So every drill-down is a real history entry and the view
// is restored from `popstate`. Rules (Meta Build guide + UI toolkit):
//   - seed the first entry with replaceState; push one entry per drill-down;
//   - tab switches REPLACE (Back leaves the agent, not the tab);
//   - the shell caps a session at 5 entries — our deepest chain is 4
//     (agents → agent → rows → record);
//   - the glasses reset focus after Back, so each entry remembers which
//     element (data-fk) had focus when we left it;
//   - "up" actions (All agents, Back to list) go BACK to the ancestor entry
//     when it is in the history below (navigateUp), never stack a copy of it.
// The route also lives in the URL so a reload (universal menu → Restart)
// lands on the same screen.

import { useSyncExternalStore } from 'react'

export type Tab = 'chat' | 'files' | 'data'

export type Route =
  | { v: 'agents' }
  | { v: 'agent'; a: string; t: Tab }
  /** p = file key (physical path for own agents, file id for shared ones); n = display name */
  | { v: 'file'; a: string; p: string; n: string }
  | { v: 'rows'; a: string; tb: string }
  /** i = absolute row index in the rows screen's ordering (newest first) */
  | { v: 'record'; a: string; tb: string; i: number }

interface HistState {
  hud: 1
  route: Route
  fk?: string
  /** How many of our entries lie below this one (0 = the bottom). */
  d?: number
  /** URLs of the entries below, nearest last (capped at UP_MAX). */
  up?: string[]
}

export const TABS: Tab[] = ['chat', 'files', 'data']
export const TABLE_NAME_RE = /^[a-z0-9_]{1,128}$/

const BASE = '/hud/'
const UP_MAX = 8

let current: Route = { v: 'agents' }
let restoreKey: string | null = null
/** A Back landed and the new screen hasn't put focus anywhere yet. */
let backPending = false
/** A history.go() of ours on its way (navigateUp / collapseHistory): what
 *  to make of the entry it lands on, and what to do if it never lands. */
let traversal: { land: ((st: HistState | null) => void) | null; timer: ReturnType<typeof setTimeout> } | null = null
/** Depth of the lowest entry this document created: entries below it belong
 *  to an earlier load (before a Restart), and traversing to them reloads. */
let ownFrom = 0
const listeners = new Set<() => void>()

function notify() { listeners.forEach((fn) => fn()) }

function isTab(t: unknown): t is Tab { return t === 'chat' || t === 'files' || t === 'data' }

function str(v: string | null, max: number): string | null {
  if (!v || v.length > max) return null
  return v
}

/** Parse (and validate) a route from a query string. Anything odd → agents. */
export function parseRoute(search: string): Route | null {
  const q = new URLSearchParams(search)
  const v = q.get('v')
  if (!v) return null
  const a = str(q.get('a'), 300)
  switch (v) {
    case 'agents':
      return { v: 'agents' }
    case 'agent': {
      const t = q.get('t')
      if (!a) return { v: 'agents' }
      return { v: 'agent', a, t: isTab(t) ? t : 'chat' }
    }
    case 'file': {
      const p = str(q.get('p'), 4000)
      if (!a) return { v: 'agents' }
      if (!p) return { v: 'agent', a, t: 'files' }
      return { v: 'file', a, p, n: (q.get('n') || p.split('/').pop() || p).slice(0, 300) }
    }
    case 'rows': {
      const tb = q.get('tb') || ''
      if (!a) return { v: 'agents' }
      if (!TABLE_NAME_RE.test(tb)) return { v: 'agent', a, t: 'data' }
      return { v: 'rows', a, tb }
    }
    case 'record': {
      const tb = q.get('tb') || ''
      const i = Number(q.get('i'))
      if (!a) return { v: 'agents' }
      if (!TABLE_NAME_RE.test(tb)) return { v: 'agent', a, t: 'data' }
      if (!Number.isInteger(i) || i < 0) return { v: 'rows', a, tb }
      return { v: 'record', a, tb, i }
    }
    default:
      return { v: 'agents' }
  }
}

export function routeUrl(r: Route): string {
  const q = new URLSearchParams()
  q.set('v', r.v)
  if ('a' in r) q.set('a', r.a)
  if (r.v === 'agent') q.set('t', r.t)
  if (r.v === 'file') { q.set('p', r.p); q.set('n', r.n) }
  if (r.v === 'rows' || r.v === 'record') q.set('tb', r.tb)
  if (r.v === 'record') q.set('i', String(r.i))
  return `${BASE}?${q}`
}

/** Stable identity of a screen — used as a React key and for focus restore. */
export function routeKey(r: Route): string {
  return routeUrl(r)
}

/** The screen Back should land on. */
export function parentRoute(r: Route): Route | null {
  switch (r.v) {
    case 'agents': return null
    case 'agent': return { v: 'agents' }
    case 'file': return { v: 'agent', a: r.a, t: 'files' }
    case 'rows': return { v: 'agent', a: r.a, t: 'data' }
    case 'record': return { v: 'rows', a: r.a, tb: r.tb }
  }
}

function chainTo(r: Route): Route[] {
  const chain: Route[] = [r]
  let p = parentRoute(r)
  while (p) { chain.unshift(p); p = parentRoute(p) }
  return chain
}

function focusKeyOfActive(): string | undefined {
  const el = document.activeElement
  if (!(el instanceof HTMLElement)) return undefined
  const holder = el.closest<HTMLElement>('[data-fk]')
  return holder?.dataset.fk || undefined
}

function ours(st: unknown): HistState | null {
  const h = st as HistState | null
  return h && h.hud === 1 && h.route ? h : null
}

function onPopState(e: PopStateEvent) {
  const st = ours(e.state)
  const land = traversal?.land
  if (traversal) { clearTimeout(traversal.timer); traversal = null }
  if (land) {
    land(st)
    notify()
    return
  }
  if (st) {
    current = st.route
    restoreKey = st.fk ?? null
  } else {
    current = parseRoute(window.location.search) ?? { v: 'agents' }
    restoreKey = null
  }
  backPending = true
  notify()
}

let installed = false

/**
 * Read the route from the URL (falling back to `fallback`) and seed history.
 * On a fresh document (history.length ≤ 1) the ancestor chain is rebuilt so
 * Back walks up a deep link instead of exiting the app. Returns whether the
 * URL named a route.
 */
export function initRouter(fallback: Route): boolean {
  const fromUrl = parseRoute(window.location.search)
  current = fromUrl ?? fallback
  if (!installed) {
    window.addEventListener('popstate', onPopState)
    installed = true
  }
  if (window.history.length <= 1) {
    const chain = chainTo(current)
    const urls = chain.map(routeUrl)
    chain.forEach((r, i) => {
      const st: HistState = { hud: 1, route: r, d: i, up: urls.slice(0, i).slice(-UP_MAX) }
      if (i === 0) window.history.replaceState(st, '', urls[0])
      else window.history.pushState(st, '', urls[i])
    })
    ownFrom = 0
  } else {
    // A reload (Restart) or a link: keep this entry's place in the stack.
    const st = ours(window.history.state)
    window.history.replaceState({ hud: 1, route: current, d: st?.d ?? 0, up: st?.up ?? [] } satisfies HistState, '', routeUrl(current))
    ownFrom = st?.d ?? 0
  }
  notify()
  return fromUrl !== null
}

/** Go to a screen. Push by default; `replace` for tab/pager switches. */
export function navigate(r: Route, opts?: { replace?: boolean }): void {
  const st = ours(window.history.state)
  const d = st?.d ?? 0
  const up = st?.up ?? []
  if (opts?.replace) {
    window.history.replaceState({ hud: 1, route: r, d, up } satisfies HistState, '', routeUrl(r))
  } else {
    // Remember what had focus here so Back can put it back.
    window.history.replaceState({ ...st, hud: 1, route: current, fk: focusKeyOfActive(), d, up } satisfies HistState, '', window.location.href)
    const below = [...up, routeUrl(current)].slice(-UP_MAX)
    window.history.pushState({ hud: 1, route: r, d: d + 1, up: below } satisfies HistState, '', routeUrl(r))
  }
  current = r
  restoreKey = null
  backPending = false
  notify()
}

/** history.go(-n), then `land` instead of the plain popstate handling. If no
 *  popstate comes (the host dropped entries we counted), `fallback` runs. */
function traverseBack(n: number, land: ((st: HistState | null) => void) | null, fallback: () => void): void {
  const timer = setTimeout(() => {
    if (traversal?.timer !== timer) return
    traversal = null
    fallback()
  }, 1500)
  traversal = { land, timer }
  window.history.go(-n)
}

/**
 * Go "up" to `target` (All agents, Back to list, WebMCP's agent list). When
 * it is an entry below this one, go BACK to it — no duplicate entry, so Back
 * from there keeps meaning "leave", and its focused row comes back. When only
 * its parent is below (another tab of the agent, another agent), go back to
 * the entry right above the parent and turn it into `target`. Otherwise
 * replace this entry.
 */
export function navigateUp(target: Route): void {
  if (traversal) return // the same gesture twice; we're already on the way
  const up = ours(window.history.state)?.up ?? []
  const fallback = () => navigate(target, { replace: true })
  const i = up.lastIndexOf(routeUrl(target))
  if (i >= 0) { traverseBack(up.length - i, null, fallback); return }
  const parent = parentRoute(target)
  const p = parent ? up.lastIndexOf(routeUrl(parent)) : -1
  if (p >= 0 && p < up.length - 1) {
    traverseBack(up.length - 1 - p, (st) => {
      window.history.replaceState({ hud: 1, route: target, d: st?.d ?? 0, up: st?.up ?? [] } satisfies HistState, '', routeUrl(target))
      current = target
      restoreKey = null
      backPending = true
    }, fallback)
    return
  }
  fallback()
}

/**
 * The session ended mid-use and the sign-in screen is up: go back to the
 * lowest entry this document owns and make it a plain /hud/ launch. Back on
 * the sign-in screen then leaves the app instead of walking screens that are
 * no longer shown, and signing back in resumes like a fresh start. (Entries
 * from before a Restart are left alone — reaching them reloads the page.)
 */
export function collapseHistory(): void {
  const seed = (st: HistState | null) => {
    window.history.replaceState({ hud: 1, route: { v: 'agents' }, d: st?.d ?? 0, up: st?.up ?? [] } satisfies HistState, '', BASE)
    current = { v: 'agents' }
    restoreKey = null
    backPending = false
  }
  const here = () => { seed(ours(window.history.state)); notify() }
  if (traversal) return
  const n = (ours(window.history.state)?.d ?? 0) - ownFrom
  if (n > 0) traverseBack(n, seed, here)
  else here()
}

let lastBackAt = 0

/** Programmatic Back (desktop Escape/Backspace, WebMCP). Deduped: the glasses
 *  can deliver one gesture twice in quick succession. */
export function goBack(): void {
  const now = Date.now()
  if (now - lastBackAt < 400) return
  lastBackAt = now
  if (parentRoute(current)) window.history.back()
}

export function getRoute(): Route {
  return current
}

/** The data-fk to refocus after a Back (consumed once). */
export function takeRestoreFocusKey(): string | null {
  const k = restoreKey
  restoreKey = null
  return k
}

/** A Back landed and the screen hasn't placed its focus yet (focusInitial). */
export function isBackPending(): boolean {
  return backPending
}

export function clearBackPending(): void {
  backPending = false
}

function subscribe(fn: () => void) {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}

export function useRoute(): Route {
  return useSyncExternalStore(subscribe, getRoute)
}
