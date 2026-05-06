import { create } from 'zustand'
import { useAuthStore } from './authStore'

export type MCPTransport = 'http' | 'stdio'

export interface MCPServer {
  name: string
  // Common
  transport?: MCPTransport
  enabled: boolean
  allowed_agents?: string[]
  headers: Record<string, string>
  added_at?: number
  initialized?: boolean
  tools_count?: number
  last_error?: string | null
  // HTTP-only
  url: string
  client_id: string
  client_secret: string
  client_secret_set?: boolean
  token_endpoint: string
  // stdio-only
  command?: string
  args?: string[]
  env?: Record<string, string>
}

export interface MCPProbeResult {
  ok: boolean
  tools_count?: number
  tool_names?: string[]
  error?: string
  status_code?: number | null
}

export interface MCPServerInput {
  name: string
  transport?: MCPTransport
  enabled?: boolean
  allowed_agents?: string[]
  headers?: Record<string, string>
  // HTTP
  url?: string
  client_id?: string
  client_secret?: string
  token_endpoint?: string
  // stdio
  command?: string
  args?: string[]
  env?: Record<string, string>
}

interface MCPStore {
  servers: MCPServer[]
  loading: boolean
  saving: boolean
  testing: Record<string, boolean>
  error: string | null
  lastTestResult: Record<string, MCPProbeResult>

  refresh: () => Promise<void>
  saveServer: (input: MCPServerInput) => Promise<MCPServer | null>
  removeServer: (name: string) => Promise<boolean>
  testServer: (name: string) => Promise<MCPProbeResult>
  probeTransient: (input: MCPServerInput) => Promise<MCPProbeResult>
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

export const useMCPStore = create<MCPStore>((set, get) => ({
  servers: [],
  loading: false,
  saving: false,
  testing: {},
  error: null,
  lastTestResult: {},

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      const data = await fetchJson('/fd/mcp/servers')
      set({
        servers: (data?.servers as MCPServer[]) || [],
        loading: false,
      })
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },

  saveServer: async (input) => {
    set({ saving: true, error: null })
    try {
      const data = await fetchJson('/fd/mcp/servers', {
        method: 'POST',
        body: JSON.stringify(input),
      })
      await get().refresh()
      set({ saving: false })
      return (data?.server as MCPServer) || null
    } catch (exc) {
      set({
        saving: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
      return null
    }
  },

  removeServer: async (name) => {
    set({ error: null })
    try {
      await fetchJson(`/fd/mcp/servers/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })
      await get().refresh()
      return true
    } catch (exc) {
      set({ error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  testServer: async (name) => {
    set((state) => ({
      testing: { ...state.testing, [name]: true },
      error: null,
    }))
    try {
      const result = (await fetchJson(
        `/fd/mcp/servers/${encodeURIComponent(name)}/test`,
        { method: 'POST' },
      )) as MCPProbeResult
      set((state) => ({
        testing: { ...state.testing, [name]: false },
        lastTestResult: { ...state.lastTestResult, [name]: result },
      }))
      // Refresh status fields (initialized / tools_count) after a successful probe.
      get().refresh()
      return result
    } catch (exc) {
      const result: MCPProbeResult = {
        ok: false,
        error: exc instanceof Error ? exc.message : String(exc),
      }
      set((state) => ({
        testing: { ...state.testing, [name]: false },
        lastTestResult: { ...state.lastTestResult, [name]: result },
      }))
      return result
    }
  },

  probeTransient: async (input) => {
    try {
      const result = (await fetchJson('/fd/mcp/probe', {
        method: 'POST',
        body: JSON.stringify(input),
      })) as MCPProbeResult
      return result
    } catch (exc) {
      return {
        ok: false,
        error: exc instanceof Error ? exc.message : String(exc),
      }
    }
  },
}))
