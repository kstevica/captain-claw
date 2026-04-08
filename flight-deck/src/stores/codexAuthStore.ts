import { create } from 'zustand'
import { useAuthStore } from './authStore'

export interface CodexStatus {
  configured: boolean
  connected: boolean
  reason?: string
  detail?: string
  auth_path?: string
  email?: string
  plan?: string
  account_id?: string
  expires_at?: number
  seconds_until_expiry?: number
  stale?: boolean
  access_token_preview?: string
}

interface CodexAuthStore {
  status: CodexStatus | null
  loading: boolean
  error: string | null
  lastMessage: string | null

  refresh: () => Promise<void>
  reimport: () => Promise<void>
}

function authHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const resp = await fetch(url, {
    credentials: 'include',
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers || {}),
    },
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  return resp.json()
}

export const useCodexAuthStore = create<CodexAuthStore>((set) => ({
  status: null,
  loading: false,
  error: null,
  lastMessage: null,

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const status = await fetchJson('/fd/codex/status')
      set({ status: status as CodexStatus, loading: false })
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },

  reimport: async () => {
    set({ loading: true, error: null, lastMessage: null })
    try {
      const status = await fetchJson('/fd/codex/reimport', { method: 'POST' })
      const s = status as CodexStatus
      set({
        status: s,
        loading: false,
        lastMessage: s.connected
          ? `Reimported${s.email ? ` — ${s.email}` : ''}`
          : s.detail || 'Reimport failed',
      })
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },
}))
