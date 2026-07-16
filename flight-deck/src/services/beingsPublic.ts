// Token-less REST client for the Iskra public square (/fd/public/beings).
//
// Deliberately separate from beings.ts: these endpoints are un-gated, so we
// NEVER attach an Authorization header and NEVER touch the auth store (which
// would try to refresh / redirect to login for a logged-out visitor). A public
// visitor has no account — that is the whole point of the square.

const FD_BASE = '/fd/public/beings'
const VILLAGE_BASE = '/fd/public/village'

async function _fetch<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}
const pubFetch = <T>(path: string, init?: RequestInit) => _fetch<T>(FD_BASE, path, init)
const villageFetch = <T>(path: string, init?: RequestInit) => _fetch<T>(VILLAGE_BASE, path, init)

// ── Types ──

export interface PublicStats {
  messages: number
  threads: number
  answered: number
}

export interface PublicProfile {
  slug: string
  name: string
  stage: string
  state: string
  generation: number
  born_at: string
  hatched_at: string | null
  died_at: string | null
  voice: string
  interests: string[]
  temperament: Record<string, number>
  mood: string
  tick_interval_minutes: number | null
  stats: PublicStats
  latest_thought: { text: string; at: string; act: string } | null
  broadcast: { text: string; at: string } | null
}

export interface PublicFile {
  path: string
  size: number
  mtime: string
}

export interface PublicGraph {
  nodes: { path: string; group: string; degree: number }[]
  edges: { from: string; to: string; rel: string; why: string }[]
  density: number
  connected_fraction: number
}

export interface PublicThreadMessage {
  role: 'public' | 'being'
  sender_name: string
  body: string
  at: string
  read_at: string | null
  answered_at: string | null
}

export interface PublicThread {
  thread_id: string
  sender_name: string
  messages: PublicThreadMessage[]
}

// ── Endpoints ──

export interface PublicVillage {
  name: string
  description: string
  visit_secret: string
}

// A visitor as it appears in the roster: a public profile + where it lives.
// `linked` = its home machine is connected right now; `origin` may be empty
// (a NAT'd private village that set no public URL).
export type PublicVisitorCard = PublicProfile & {
  id: string; origin: string; linked: boolean
}

export const listPublicBeings = () =>
  pubFetch<{ beings: PublicProfile[]; village: PublicVillage; visitors: PublicVisitorCard[] }>('')

export const getPublicBeing = (slug: string) =>
  pubFetch<PublicProfile>(`/${slug}`)

export const getPublicFiles = (slug: string) =>
  pubFetch<{ files: PublicFile[] }>(`/${slug}/files`)

export const getPublicFile = (slug: string, path: string) =>
  pubFetch<{ path: string; text: string }>(
    `/${slug}/file?path=${encodeURIComponent(path)}`)

export const getPublicJournal = (slug: string, date = '') =>
  pubFetch<{ date: string; text: string }>(
    `/${slug}/journal${date ? `?date=${encodeURIComponent(date)}` : ''}`)

export const getPublicGraph = (slug: string) =>
  pubFetch<PublicGraph>(`/${slug}/graph`)

export const postPublicMessage = (
  slug: string, name: string, body: string, threadId?: string | null,
) =>
  pubFetch<{ thread_id: string; message_id: string }>(`/${slug}/message`, {
    method: 'POST',
    body: JSON.stringify({ name, body, thread_id: threadId || null }),
  })

export const getPublicThread = (slug: string, threadId: string) =>
  pubFetch<PublicThread>(`/${slug}/thread/${threadId}`)

// ── Visitors: beings from other machines, proxied through this village ──

export type PublicVisitorProfile = PublicProfile & {
  id: string; origin: string; visitor: true; last_seen: string; linked: boolean
}

export const getVisitorProfile = (id: string) =>
  villageFetch<PublicVisitorProfile>(`/visitors/${id}`)
export const getVisitorFiles = (id: string) =>
  villageFetch<{ files: PublicFile[] }>(`/visitors/${id}/files`)
export const getVisitorFile = (id: string, path: string) =>
  villageFetch<{ path: string; text: string }>(
    `/visitors/${id}/file?path=${encodeURIComponent(path)}`)
export const getVisitorJournal = (id: string, date = '') =>
  villageFetch<{ date: string; text: string }>(
    `/visitors/${id}/journal${date ? `?date=${encodeURIComponent(date)}` : ''}`)
export const getVisitorGraph = (id: string) =>
  villageFetch<PublicGraph>(`/visitors/${id}/graph`)
export const postVisitorMessage = (
  id: string, name: string, body: string, threadId?: string | null,
) =>
  villageFetch<{ thread_id: string; message_id: string }>(`/visitors/${id}/message`, {
    method: 'POST', body: JSON.stringify({ name, body, thread_id: threadId || null }),
  })
export const getVisitorThread = (id: string, threadId: string) =>
  villageFetch<PublicThread>(`/visitors/${id}/thread/${threadId}`)

// ── A data source shared by local beings and proxied visitors, so the detail
// page renders both identically. `key` namespaces this browser's thread id. ──

export interface PublicApi {
  key: string
  files: () => Promise<{ files: PublicFile[] }>
  file: (path: string) => Promise<{ path: string; text: string }>
  journal: (date?: string) => Promise<{ date: string; text: string }>
  graph: () => Promise<PublicGraph>
  message: (name: string, body: string, threadId?: string | null)
    => Promise<{ thread_id: string; message_id: string }>
  thread: (threadId: string) => Promise<PublicThread>
}

export function makeBeingApi(slug: string): PublicApi {
  return {
    key: `b:${slug}`,
    files: () => getPublicFiles(slug),
    file: (p) => getPublicFile(slug, p),
    journal: (d) => getPublicJournal(slug, d),
    graph: () => getPublicGraph(slug),
    message: (n, b, t) => postPublicMessage(slug, n, b, t),
    thread: (t) => getPublicThread(slug, t),
  }
}
export function makeVisitorApi(id: string): PublicApi {
  return {
    key: `v:${id}`,
    files: () => getVisitorFiles(id),
    file: (p) => getVisitorFile(id, p),
    journal: (d) => getVisitorJournal(id, d),
    graph: () => getVisitorGraph(id),
    message: (n, b, t) => postVisitorMessage(id, n, b, t),
    thread: (t) => getVisitorThread(id, t),
  }
}

// ── Per-browser identity: the thread id + chosen name live in localStorage,
// so a visitor keeps talking to the same being across reloads (no accounts). ──

const THREAD_KEY = (slug: string) => `iskra:pub:thread:${slug}`
const NAME_KEY = 'iskra:pub:name'

export function savedThreadId(slug: string): string | null {
  try { return localStorage.getItem(THREAD_KEY(slug)) } catch { return null }
}
export function saveThreadId(slug: string, id: string) {
  try { localStorage.setItem(THREAD_KEY(slug), id) } catch { /* ignore */ }
}
export function clearThreadId(slug: string) {
  try { localStorage.removeItem(THREAD_KEY(slug)) } catch { /* ignore */ }
}
export function savedName(): string {
  try { return localStorage.getItem(NAME_KEY) || '' } catch { return '' }
}
export function saveName(name: string) {
  try { localStorage.setItem(NAME_KEY, name) } catch { /* ignore */ }
}

export const PUBLIC_MSG_MAX = 64

export function cadenceLabel(minutes: number | null): string {
  if (!minutes) return 'on its own rhythm'
  if (minutes >= 60 && minutes % 60 === 0) {
    const h = minutes / 60
    return `about every ${h} hour${h > 1 ? 's' : ''}`
  }
  return `about every ${minutes} min`
}
