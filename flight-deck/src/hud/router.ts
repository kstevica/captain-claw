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
//     element (data-fk) had focus when we left it.
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

interface HistState { hud: 1; route: Route; fk?: string }

export const TABS: Tab[] = ['chat', 'files', 'data']
export const TABLE_NAME_RE = /^[a-z0-9_]{1,128}$/

const BASE = '/hud/'

let current: Route = { v: 'agents' }
let restoreKey: string | null = null
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

function onPopState(e: PopStateEvent) {
  const st = e.state as HistState | null
  if (st && st.hud === 1 && st.route) {
    current = st.route
    restoreKey = st.fk ?? null
  } else {
    current = parseRoute(window.location.search) ?? { v: 'agents' }
    restoreKey = null
  }
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
    window.history.replaceState({ hud: 1, route: chain[0] } satisfies HistState, '', routeUrl(chain[0]))
    for (const r of chain.slice(1)) {
      window.history.pushState({ hud: 1, route: r } satisfies HistState, '', routeUrl(r))
    }
  } else {
    window.history.replaceState({ hud: 1, route: current } satisfies HistState, '', routeUrl(current))
  }
  notify()
  return fromUrl !== null
}

/** Go to a screen. Push by default; `replace` for tab/pager switches. */
export function navigate(r: Route, opts?: { replace?: boolean }): void {
  if (opts?.replace) {
    window.history.replaceState({ hud: 1, route: r } satisfies HistState, '', routeUrl(r))
  } else {
    // Remember what had focus here so Back can put it back.
    const st = (window.history.state as HistState | null) ?? { hud: 1 as const, route: current }
    window.history.replaceState({ ...st, hud: 1, route: current, fk: focusKeyOfActive() } satisfies HistState, '', window.location.href)
    window.history.pushState({ hud: 1, route: r } satisfies HistState, '', routeUrl(r))
  }
  current = r
  restoreKey = null
  notify()
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

function subscribe(fn: () => void) {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}

export function useRoute(): Route {
  return useSyncExternalStore(subscribe, getRoute)
}
