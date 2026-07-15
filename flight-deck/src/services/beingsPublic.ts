// Token-less REST client for the Iskra public square (/fd/public/beings).
//
// Deliberately separate from beings.ts: these endpoints are un-gated, so we
// NEVER attach an Authorization header and NEVER touch the auth store (which
// would try to refresh / redirect to login for a logged-out visitor). A public
// visitor has no account — that is the whole point of the square.

const FD_BASE = '/fd/public/beings'

async function pubFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

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
  description: string
}

export const listPublicBeings = () =>
  pubFetch<{ beings: PublicProfile[]; village: PublicVillage }>('')

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
