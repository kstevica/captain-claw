import { create } from 'zustand'

export interface AgentGroup {
  id: string
  name: string
  color: string
  agentIds: string[]
}

const STORAGE_KEY = 'fd:agent-groups'
const COLORS = ['violet', 'blue', 'emerald', 'amber', 'pink', 'cyan', 'red', 'indigo']

function load(): AgentGroup[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function save(groups: AgentGroup[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(groups))
}

interface GroupStore {
  groups: AgentGroup[]
  createGroup: (name: string) => string
  deleteGroup: (id: string) => void
  renameGroup: (id: string, name: string) => void
  addToGroup: (groupId: string, agentId: string) => void
  removeFromGroup: (groupId: string, agentId: string) => void
  getGroupsForAgent: (agentId: string) => AgentGroup[]
}

export const useGroupStore = create<GroupStore>((set, get) => ({
  groups: load(),

  createGroup: (name) => {
    const id = `grp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const color = COLORS[get().groups.length % COLORS.length]
    const group: AgentGroup = { id, name, color, agentIds: [] }
    const groups = [...get().groups, group]
    save(groups)
    set({ groups })
    return id
  },

  deleteGroup: (id) => {
    const groups = get().groups.filter((g) => g.id !== id)
    save(groups)
    set({ groups })
  },

  renameGroup: (id, name) => {
    const groups = get().groups.map((g) => g.id === id ? { ...g, name } : g)
    save(groups)
    set({ groups })
  },

  addToGroup: (groupId, agentId) => {
    const groups = get().groups.map((g) => {
      if (g.id !== groupId || g.agentIds.includes(agentId)) return g
      return { ...g, agentIds: [...g.agentIds, agentId] }
    })
    save(groups)
    set({ groups })
  },

  removeFromGroup: (groupId, agentId) => {
    const groups = get().groups.map((g) => {
      if (g.id !== groupId) return g
      return { ...g, agentIds: g.agentIds.filter((id) => id !== agentId) }
    })
    save(groups)
    set({ groups })
  },

  getGroupsForAgent: (agentId) => {
    return get().groups.filter((g) => g.agentIds.includes(agentId))
  },
}))
