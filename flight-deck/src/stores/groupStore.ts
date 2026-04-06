import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

export type GroupType = 'group' | 'role'

export interface AgentGroup {
  id: string
  name: string
  color: string
  type: GroupType
  agentIds: string[]
}

const STORAGE_KEY = 'fd:agent-groups'
const COLORS = ['violet', 'blue', 'emerald', 'amber', 'pink', 'cyan', 'red', 'indigo']

function load(): AgentGroup[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as AgentGroup[]
    // Migrate: default type to 'group' for existing entries
    return raw.map(g => ({ ...g, type: g.type || 'group' }))
  } catch { return [] }
}
function save(groups: AgentGroup[]) {
  const val = JSON.stringify(groups)
  if (useAuthStore.getState().authEnabled) {
    queueSave(STORAGE_KEY, val)
  } else {
    localStorage.setItem(STORAGE_KEY, val)
  }
}

interface GroupStore {
  groups: AgentGroup[]
  createGroup: (name: string, type?: GroupType) => string
  deleteGroup: (id: string) => void
  renameGroup: (id: string, name: string) => void
  addToGroup: (groupId: string, agentId: string) => void
  removeFromGroup: (groupId: string, agentId: string) => void
  getGroupsForAgent: (agentId: string) => AgentGroup[]
}

export const useGroupStore = create<GroupStore>((set, get) => ({
  groups: load(),

  createGroup: (name, type = 'group') => {
    const id = `grp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const color = COLORS[get().groups.length % COLORS.length]
    const group: AgentGroup = { id, name, color, type, agentIds: [] }
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

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try { useGroupStore.setState({ groups: JSON.parse(raw) }) } catch { /* ignore */ }
  }
})
