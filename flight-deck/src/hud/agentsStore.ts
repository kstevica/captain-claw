// The signed-in user's agents, loaded once per session and on demand.

import { create } from 'zustand'
import { listAgents, type HudAgent } from './api'

interface AgentsState {
  agents: HudAgent[] | null
  loading: boolean
  error: string | null
  /** Epoch ms of the last successful load. */
  loadedAt: number
  refresh: () => Promise<void>
  reset: () => void
}

let inflight: Promise<void> | null = null

export const useAgents = create<AgentsState>((set) => ({
  agents: null,
  loading: false,
  error: null,
  loadedAt: 0,
  refresh: () => {
    if (inflight) return inflight
    set({ loading: true, error: null })
    inflight = listAgents()
      .then((agents) => set({ agents, loading: false, error: null, loadedAt: Date.now() }))
      .catch((e: unknown) => set({ loading: false, error: e instanceof Error ? e.message : String(e) }))
      .finally(() => { inflight = null })
    return inflight
  },
  reset: () => set({ agents: null, loading: false, error: null, loadedAt: 0 }),
}))

export function findAgent(id: string): HudAgent | null {
  return useAgents.getState().agents?.find((a) => a.id === id) ?? null
}

const LAST_AGENT_KEY = 'hud.lastAgent.v1'

export function rememberAgent(id: string): void {
  try { localStorage.setItem(LAST_AGENT_KEY, id) } catch { /* storage blocked */ }
}

export function lastAgentId(): string | null {
  try { return localStorage.getItem(LAST_AGENT_KEY) } catch { return null }
}
