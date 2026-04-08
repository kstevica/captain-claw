import { create } from 'zustand'
import { useAuthStore } from './authStore'

export interface GrantedScope {
  scope: string
  label: string
}

export interface GoogleUserInfo {
  email?: string
  name?: string
  picture?: string
  [key: string]: unknown
}

export type GoogleAuthMode = 'custom'

export interface GoogleAuthStatus {
  configured: boolean
  mode: GoogleAuthMode
  supports_vertex: boolean
  connected: boolean
  user: GoogleUserInfo | null
  granted_scopes: GrantedScope[]
  redirect_uri: string
}

export interface ScopeCatalogEntry {
  scope: string
  label: string
  description: string
  sensitivity: 'none' | 'sensitive' | 'restricted'
  group: string
}

export interface GoogleAuthConfig {
  mode: GoogleAuthMode
  client_id: string
  client_id_set: boolean
  client_secret_set: boolean
  project_id: string
  location: string
  scopes: string[]
  default_scopes: string[]
  scope_catalog: ScopeCatalogEntry[]
  redirect_uri: string
}

interface GoogleAuthStore {
  status: GoogleAuthStatus | null
  config: GoogleAuthConfig | null
  loading: boolean
  error: string | null
  lastPopupMessage: string | null

  refresh: () => Promise<void>
  saveConfig: (patch: Partial<{
    client_id: string
    client_secret: string
    project_id: string
    location: string
    scopes: string[]
  }>) => Promise<boolean>
  clearCredentials: () => Promise<boolean>
  connect: () => void
  disconnect: () => Promise<void>
  startMessageListener: () => () => void
}

const emptyStatus: GoogleAuthStatus = {
  configured: false,
  mode: 'custom',
  supports_vertex: false,
  connected: false,
  user: null,
  granted_scopes: [],
  redirect_uri: '',
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

export const useGoogleAuthStore = create<GoogleAuthStore>((set, get) => ({
  status: null,
  config: null,
  loading: false,
  error: null,
  lastPopupMessage: null,

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const [status, config] = await Promise.all([
        fetchJson('/fd/google/status'),
        fetchJson('/fd/google/config'),
      ])
      set({
        status: status as GoogleAuthStatus,
        config: config as GoogleAuthConfig,
        loading: false,
      })
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
        status: get().status || emptyStatus,
      })
    }
  },

  saveConfig: async (patch) => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/config', {
        method: 'POST',
        body: JSON.stringify(patch),
      })
      await get().refresh()
      return true
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
      return false
    }
  },

  clearCredentials: async () => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/config', {
        method: 'POST',
        body: JSON.stringify({ clear: true }),
      })
      await get().refresh()
      return true
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
      return false
    }
  },

  connect: () => {
    // The login endpoint is a 302 to Google, so open it as a popup.
    // Note: /fd/google/login is intentionally unauthenticated (the user
    // already signed into Flight Deck to trigger this), so we don't
    // need to attach an Authorization header here — popups can't carry
    // custom headers anyway.
    const url = '/fd/google/login'
    const w = 520
    const h = 640
    const left = window.screenX + (window.outerWidth - w) / 2
    const top = window.screenY + (window.outerHeight - h) / 2
    window.open(
      url,
      'captain-claw-google-oauth',
      `width=${w},height=${h},left=${left},top=${top}`,
    )
  },

  disconnect: async () => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/logout', { method: 'POST' })
      await get().refresh()
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },

  startMessageListener: () => {
    const handler = (event: MessageEvent) => {
      const data = event.data
      if (!data || typeof data !== 'object') return
      if (data.type !== 'captain-claw-google-oauth') return
      set({
        lastPopupMessage:
          data.status === 'success'
            ? `Connected${data.email ? ` as ${data.email}` : ''}`
            : `Error: ${data.title || 'OAuth failed'}${data.detail ? ` — ${data.detail}` : ''}`,
      })
      setTimeout(() => { get().refresh() }, 300)
    }
    window.addEventListener('message', handler)
    return () => window.removeEventListener('message', handler)
  },
}))
