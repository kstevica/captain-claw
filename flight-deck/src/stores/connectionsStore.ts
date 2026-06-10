import { create } from 'zustand'
import { useAuthStore } from './authStore'
import { useMCPStore, type MCPProbeResult } from './mcpStore'

// Traffic-light health for an external connection FD depends on.
//   green  — connected AND a read-only test call succeeded
//   yellow — connected but the read-only test call failed
//   red    — not connected / not configured
//   gray   — disabled or not yet checked
export type ConnectionHealth = 'green' | 'yellow' | 'red' | 'gray'

export interface ConnectionStatus {
  id: string            // 'google' | `mcp:${name}`
  kind: 'google' | 'mcp'
  label: string
  health: ConnectionHealth
  detail: string        // email / tool count / error message
  checkedAt: number | null
}

interface ConnectionsStore {
  connections: ConnectionStatus[]
  checking: boolean
  lastCheckedAt: number | null
  intervalId: ReturnType<typeof setInterval> | null

  checkAll: () => Promise<void>
  startPolling: (everyMs?: number) => void
  stopPolling: () => void
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
    headers: { ...authHeaders(), ...(init?.headers || {}) },
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  return resp.json()
}

async function probeGoogle(): Promise<ConnectionStatus> {
  const base: Omit<ConnectionStatus, 'health' | 'detail'> = {
    id: 'google',
    kind: 'google',
    label: 'Google',
    checkedAt: Date.now(),
  }
  try {
    const r = await fetchJson('/fd/google/probe')
    if (!r.connected) {
      return { ...base, health: 'red', detail: r.error || 'Not connected' }
    }
    if (!r.ok) {
      return { ...base, health: 'yellow', detail: r.error || 'Read-only test failed' }
    }
    return { ...base, health: 'green', detail: r.email ? `Connected as ${r.email}` : 'Connected' }
  } catch (exc) {
    return { ...base, health: 'red', detail: exc instanceof Error ? exc.message : String(exc) }
  }
}

export const useConnectionsStore = create<ConnectionsStore>((set, get) => ({
  connections: [],
  checking: false,
  lastCheckedAt: null,
  intervalId: null,

  checkAll: async () => {
    if (get().checking) return
    set({ checking: true })

    // Probe Google + every enabled MCP server. MCP servers come from the
    // mcpStore — refresh it so we cover newly-added servers and pick up
    // the `initialized` flag used to distinguish red (never connected)
    // from yellow (was connected, test now failing).
    const mcp = useMCPStore.getState()
    await mcp.refresh().catch(() => {})
    const servers = useMCPStore.getState().servers

    const googleResult = await probeGoogle()

    const mcpResults: ConnectionStatus[] = []
    for (const s of servers) {
      const base: Omit<ConnectionStatus, 'health' | 'detail'> = {
        id: `mcp:${s.name}`,
        kind: 'mcp',
        label: s.name,
        checkedAt: Date.now(),
      }
      if (!s.enabled) {
        mcpResults.push({ ...base, health: 'gray', detail: 'Disabled' })
        continue
      }
      const r: MCPProbeResult = await useMCPStore.getState().testServer(s.name).catch((exc) => ({
        ok: false,
        error: exc instanceof Error ? exc.message : String(exc),
      }))
      if (r.ok) {
        mcpResults.push({
          ...base,
          health: 'green',
          detail: `${r.tools_count ?? 0} tool${r.tools_count === 1 ? '' : 's'} available`,
        })
      } else if (s.initialized) {
        // Was reachable before but the live read-only test now fails.
        mcpResults.push({ ...base, health: 'yellow', detail: r.error || 'Read-only test failed' })
      } else {
        mcpResults.push({ ...base, health: 'red', detail: r.error || 'Not connected' })
      }
    }

    set({
      connections: [googleResult, ...mcpResults],
      checking: false,
      lastCheckedAt: Date.now(),
    })
  },

  startPolling: (everyMs = 600000) => {
    if (get().intervalId) return
    // Kick off an immediate check, then poll.
    get().checkAll()
    const id = setInterval(() => { get().checkAll() }, everyMs)
    set({ intervalId: id })
  },

  stopPolling: () => {
    const id = get().intervalId
    if (id) clearInterval(id)
    set({ intervalId: null })
  },
}))
