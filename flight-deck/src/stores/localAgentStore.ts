import { create } from 'zustand'

const STORAGE_KEY = 'fd:local-agents'

export interface LocalAgent {
  id: string
  name: string
  host: string
  port: number
  authToken: string
  status: 'unknown' | 'online' | 'offline'
}

interface LocalAgentStore {
  agents: LocalAgent[]
  addAgent: (name: string, host: string, port: number, authToken: string) => void
  removeAgent: (id: string) => void
  updateAgent: (id: string, patch: Partial<Pick<LocalAgent, 'name' | 'host' | 'port' | 'authToken'>>) => void
  probeAgent: (id: string) => Promise<void>
  probeAll: () => Promise<void>
}

function load(): LocalAgent[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw)
  } catch { return [] }
}

function save(agents: LocalAgent[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(agents))
}

export const useLocalAgentStore = create<LocalAgentStore>((set, get) => ({
  agents: load(),

  addAgent: (name, host, port, authToken) => {
    const id = `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const agent: LocalAgent = { id, name, host, port, authToken, status: 'unknown' }
    const agents = [...get().agents, agent]
    save(agents)
    set({ agents })
    // Probe immediately
    get().probeAgent(id)
  },

  removeAgent: (id) => {
    const agents = get().agents.filter((a) => a.id !== id)
    save(agents)
    set({ agents })
  },

  updateAgent: (id, patch) => {
    const agents = get().agents.map((a) => a.id === id ? { ...a, ...patch } : a)
    save(agents)
    set({ agents })
  },

  probeAgent: async (id) => {
    const agent = get().agents.find((a) => a.id === id)
    if (!agent) return
    try {
      // Probe via FD backend to avoid CORS issues
      const res = await fetch(`/fd/probe?host=${encodeURIComponent(agent.host)}&port=${agent.port}`)
      if (res.ok) {
        const data = await res.json()
        const status = data.ok ? 'online' : 'offline'
        const agents = get().agents.map((a) => a.id === id ? { ...a, status: status as LocalAgent['status'] } : a)
        set({ agents })
      } else {
        throw new Error('probe failed')
      }
    } catch {
      const agents = get().agents.map((a) => a.id === id ? { ...a, status: 'offline' as const } : a)
      set({ agents })
    }
  },

  probeAll: async () => {
    const agents = get().agents
    await Promise.all(agents.map((a) => get().probeAgent(a.id)))
  },
}))
