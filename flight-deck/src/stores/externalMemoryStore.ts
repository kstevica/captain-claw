import { create } from 'zustand'
import { useAuthStore, refreshAccessToken, registerSignOutTeardown } from './authStore'

// External memory (Membase, …) — one per user, shared by all of their agents,
// through /fd/external-memory/* (captain_claw/flight_deck/external_memory/).
// Flight Deck keeps the credentials: they never come back here, only a hint.
// On a deck with sign-in off the connection is the deck's.

export interface CredentialField {
  name: string
  label: string
  secret: boolean
  required: boolean
  placeholder: string
  help: string
}

export interface ProviderInfo {
  name: string
  label: string
  description: string
  docs_url: string
  credentials_url: string
  credential_fields: CredentialField[]
}

export interface ActionParam {
  name: string
  type: string
  description: string
  required: boolean
  enum?: string[]
}

export interface ProviderAction {
  name: string
  description: string
  params: ActionParam[]
  writes: boolean
  destructive: boolean
}

export type MirrorKind = 'insight' | 'intention' | 'reflection' | 'lesson' | 'intuition'

export interface OutboxCounts {
  pending: number
  sent: number
  failed: number
  last_sent_at: number | null
  last_error: string
}

export interface ExternalMemoryStatus {
  connected: boolean
  providers: ProviderInfo[]
  mirror: Record<MirrorKind, boolean>
  mirror_kinds: MirrorKind[]
  scope: 'user' | 'deck'
  auth_enabled: boolean
  provider?: string
  label?: string
  credential_hints?: Record<string, string>
  verified?: boolean | null
  verify_error?: 'refused' | 'unreachable' | null
  verified_at?: string | null
  updated_at?: string | null
  reaches?: string[]
  actions?: ProviderAction[]
  rules?: string[]
  outbox?: OutboxCounts
}

export interface OutboxItem {
  id: number
  kind: MirrorKind
  source_id: string
  agent: string
  title: string
  state: 'pending' | 'sent' | 'failed'
  attempts: number
  last_error: string
  remote_id: string
  created_at: number
  sent_at: number | null
}

interface ExternalMemoryStore {
  status: ExternalMemoryStatus | null
  loading: boolean
  error: string | null

  refresh: () => Promise<void>
  connect: (provider: string, credentials: Record<string, string>) => Promise<{ ok: boolean; warning?: string }>
  disconnect: () => Promise<boolean>
  test: () => Promise<{ ok: boolean; latency_ms?: number }>
  setMirror: (kinds: Partial<Record<MirrorKind, boolean>>) => Promise<boolean>
  run: (action: string, args: Record<string, unknown>) => Promise<{ text: string } | { error: string }>
  outbox: (state?: string) => Promise<OutboxItem[]>
  retryOutbox: () => Promise<number | null>
}

const _OLD_DECK = "This Flight Deck has no external memory yet — it needs an update and a restart."

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const doFetch = () => {
    const { token } = useAuthStore.getState()
    return fetch(url, {
      credentials: 'include',
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init?.headers || {}),
      },
    })
  }
  let resp = await doFetch()
  if (resp.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    resp = await doFetch()
  }
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    const detail = typeof body?.detail === 'string' ? body.detail : ''
    if (resp.status === 405 || (resp.status === 404 && (!detail || detail === 'Not Found' || detail.startsWith('No route for')))) {
      throw new Error(_OLD_DECK)
    }
    if (resp.status === 401) throw new Error('Your Flight Deck session has expired — sign in again.')
    throw new Error(detail || `Flight Deck couldn't complete the request (HTTP ${resp.status}).`)
  }
  return (await resp.json()) as T
}

const errText = (exc: unknown) =>
  exc instanceof TypeError ? "Couldn't reach Flight Deck — check the connection and try again."
    : exc instanceof Error ? exc.message : String(exc)

type WithWarning = ExternalMemoryStatus & { warning?: string }

export const useExternalMemoryStore = create<ExternalMemoryStore>((set) => ({
  status: null,
  loading: false,
  error: null,

  // Flight Deck's record only — no call to the provider.
  refresh: async () => {
    set({ loading: true, error: null })
    try {
      set({ status: await fetchJson<ExternalMemoryStatus>('/fd/external-memory/status'), loading: false })
    } catch (exc) {
      set({ loading: false, error: errText(exc) })
    }
  },

  // Flight Deck checks the credentials with the provider first: refused ones
  // aren't saved (error); ones it couldn't check are saved unverified (warning).
  connect: async (provider, credentials) => {
    set({ error: null })
    try {
      const data = await fetchJson<WithWarning>('/fd/external-memory/connection', {
        method: 'PUT',
        body: JSON.stringify({ provider, credentials }),
      })
      const { warning, ...status } = data
      set({ status })
      return { ok: true, ...(warning ? { warning } : {}) }
    } catch (exc) {
      set({ error: errText(exc) })
      return { ok: false }
    }
  },

  disconnect: async () => {
    set({ error: null })
    try {
      set({ status: await fetchJson<ExternalMemoryStatus>('/fd/external-memory/connection', { method: 'DELETE' }) })
      return true
    } catch (exc) {
      set({ error: errText(exc) })
      return false
    }
  },

  test: async () => {
    set({ error: null })
    try {
      const data = await fetchJson<ExternalMemoryStatus & { ok: boolean; latency_ms: number }>(
        '/fd/external-memory/test', { method: 'POST', body: '{}' },
      )
      const { ok, latency_ms, ...status } = data
      set({ status })
      return { ok, latency_ms }
    } catch (exc) {
      set({ error: errText(exc) })
      try {
        set({ status: await fetchJson<ExternalMemoryStatus>('/fd/external-memory/status') })
      } catch { /* keep the previous status */ }
      return { ok: false }
    }
  },

  setMirror: async (kinds) => {
    set({ error: null })
    try {
      set({
        status: await fetchJson<ExternalMemoryStatus>('/fd/external-memory/mirror', {
          method: 'PUT',
          body: JSON.stringify({ kinds }),
        }),
      })
      return true
    } catch (exc) {
      set({ error: errText(exc) })
      return false
    }
  },

  run: async (action, args) => {
    try {
      const data = await fetchJson<{ text: string }>('/fd/external-memory/run', {
        method: 'POST',
        body: JSON.stringify({ action, args }),
      })
      return { text: data.text }
    } catch (exc) {
      return { error: errText(exc) }
    }
  },

  outbox: async (state = '') => {
    try {
      const data = await fetchJson<{ items: OutboxItem[] }>(
        `/fd/external-memory/outbox?limit=30${state ? `&state=${encodeURIComponent(state)}` : ''}`,
      )
      return data.items || []
    } catch {
      return []
    }
  },

  retryOutbox: async () => {
    set({ error: null })
    try {
      const data = await fetchJson<{ requeued: number }>('/fd/external-memory/outbox/retry', { method: 'POST', body: '{}' })
      try {
        set({ status: await fetchJson<ExternalMemoryStatus>('/fd/external-memory/status') })
      } catch { /* keep the previous status */ }
      return data.requeued
    } catch (exc) {
      set({ error: errText(exc) })
      return null
    }
  },
}))

// The next person at a shared screen must not see the outgoing user's connection.
registerSignOutTeardown(() => {
  useExternalMemoryStore.setState({ status: null, loading: false, error: null })
})
