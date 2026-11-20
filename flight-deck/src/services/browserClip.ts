// Captain for Chrome — the deck side (captain_claw/flight_deck/browser_clip.py).
//
// A browser connects by pairing: the extension shows a code and opens
// <deck>/connect-browser#<code>; the signed-in person approves it here and
// the extension gets a personal access token named "Browser · <browser>".
// Connected browsers are listed (and disconnected) in Connections → Browser.
// Captain for Mac pairs the same way (kind "desktop", token "Desktop · <Mac>")
// and is listed beside them.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const state = useAuthStore.getState()
  if (state.authEnabled === true && !state.token) {
    if (!(await refreshAccessToken())) throw new Error('Not authenticated')
  }
  const build = (): RequestInit => ({ headers: authHeaders(), credentials: 'include', ...init })
  let res = await fetch(path, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(path, build())
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' && body.detail ? body.detail : `HTTP ${res.status}`)
  }
  return res.json()
}

export type ClipKind = 'browser' | 'desktop'

export interface ConnectedBrowser {
  id: string
  kind?: ClipKind
  name: string
  created_at: string
  last_used_at: string | null
  /** The extension's version as this browser last said it ('' = not said yet). */
  version?: string
  /** Against this deck's copy: 'update' = the deck has a newer one, 'ahead' = the browser's is newer. */
  version_state?: '' | 'same' | 'update' | 'ahead'
}

export interface PairInfo {
  code: string
  browser: string
  kind?: ClipKind
  ip: string
  user_agent: string
  created_at: string
  same_network: boolean
  expires_in: number
}

export const EXTENSION_ZIP = '/fd/clip/extension.zip'
export const MAC_APP_ZIP = '/fd/clip/mac-app.zip'
export const PAIR_PATH = '/connect-browser'

export const listBrowsers = () => fdFetch<{ browsers: ConnectedBrowser[] }>('/fd/browsers').then((r) => r.browsers)

/** The connected browsers and Macs, and whether this deck has Captain for Mac to hand out. */
export const listConnections = () =>
  fdFetch<{ browsers: ConnectedBrowser[]; mac_app?: boolean; extension_version?: string }>('/fd/browsers')
    .then((r) => ({ browsers: r.browsers, macApp: !!r.mac_app, extensionVersion: r.extension_version || '' }))

export const removeBrowser = (id: string) =>
  fdFetch<{ ok: boolean }>(`/fd/browsers/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const pairInfo = (code: string) => fdFetch<PairInfo>(`/fd/browsers/pair?code=${encodeURIComponent(code)}`)

export const decidePair = (code: string, approve: boolean) =>
  fdFetch<{ ok: boolean; status: string }>('/fd/browsers/pair', {
    method: 'POST', body: JSON.stringify({ code, approve }),
  })

/** The code a pairing link carried (<deck>/connect-browser#KQ7-M2X), read once at load. */
export function pairCodeFromLocation(): string {
  if (window.location.pathname !== PAIR_PATH) return ''
  const raw = decodeURIComponent(window.location.hash.replace(/^#/, '')).toUpperCase().replace(/[^A-Z0-9]/g, '')
  return raw.length === 6 ? `${raw.slice(0, 3)}-${raw.slice(3)}` : ''
}

// ── Recordings (captain_claw/flight_deck/browser_recordings.py) ──
//
// A process the person recorded in Captain for Chrome, an agent does again.
// Kept on the deck: its steps, the extra data every run reads first, and how
// its runs went.

export interface RecordingAsk { id: string; label: string; was: string; kind: string; options?: string[] }

export interface RecordingSummary {
  id: string
  name: string
  about: string
  site: string
  start_url: string
  steps: number
  asks: RecordingAsk[]
  asks_during: string[]
  agent_decides: string[]
  created_at: string
  updated_at: string
  runs: number
  last_run_at: string
  data?: number
}

export interface RecordingData { id: string; name: string; size: number; type: string; origin: string; added_at: string }

export interface RecordingRun {
  id: string
  recording: string
  name: string
  site: string
  agent: string
  agent_slug: string
  browser: string
  started_at: string
  ended_at: string
  /** thinking | done | error | stopped | lost (the deck restarted meanwhile) */
  status: string
  note: string
  inputs: { label: string; value: string }[]
  folder: string
  vfs: string
  extra: string[]
  steps: number
  result: string
  error: string
  steps_done?: string[]
}

export interface RecordingDetail extends Omit<RecordingSummary, 'data'> {
  step_lines: string[]
  data: RecordingData[]
  /** How its runs went, the newest first (`runs` is how many). */
  history: RecordingRun[]
  limits: { files: number; bytes: number }
}

const rec = (id: string) => `/fd/browser-recordings/${encodeURIComponent(id)}`

export const listRecordings = () =>
  fdFetch<{ recordings: RecordingSummary[]; runs: RecordingRun[] }>('/fd/browser-recordings')
export const getRecording = (id: string) => fdFetch<RecordingDetail>(rec(id))
export const deleteRecording = (id: string) => fdFetch<{ ok: boolean }>(rec(id), { method: 'DELETE' })
export const addRecordingData = (id: string, ids: string[]) =>
  fdFetch<{ added: RecordingData[]; items: RecordingData[] }>(`${rec(id)}/data`, { method: 'POST', body: JSON.stringify({ ids }) })
export const dropRecordingData = (id: string, itemId: string) =>
  fdFetch<{ items: RecordingData[] }>(`${rec(id)}/data/${encodeURIComponent(itemId)}`, { method: 'DELETE' })
/** A link that opens (or downloads) one file: the sign-in token rides along, as file links do. */
export function recordingDataUrl(id: string, itemId: string, download = false): string {
  const { token, authEnabled } = useAuthStore.getState()
  const q = new URLSearchParams()
  if (download) q.set('download', 'true')
  if (authEnabled && token) q.set('fd_token', token)
  const qs = q.toString()
  return `${rec(id)}/data/${encodeURIComponent(itemId)}/file${qs ? `?${qs}` : ''}`
}
