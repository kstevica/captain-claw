import { create } from 'zustand'
import { fetchSettingsOrNull, queueSave, registerHydrator } from '../services/settingsSync'
import { isFolder, nextGroupColor } from '../utils/agentGroups'
import { useAuthStore } from './authStore'

export type GroupType = 'group' | 'role'

export interface AgentGroup {
  id: string
  name: string
  color: string
  type: GroupType
  agentIds: string[]
  /** First in the agent list, whatever its place among the groups. One group at most. */
  pinned?: boolean
  /** Agents at the top of this group, in the order they were pinned. */
  pinnedAgentIds?: string[]
}

const STORAGE_KEY = 'fd:agent-groups'
// Which sections are folded: this device's, not synced — a phone may keep them
// shut that a desktop keeps open, and folding one must never write the groups
// themselves over what another device just changed.
const COLLAPSED_KEY = 'fd:agent-groups-collapsed'
// Back on this tab: read the groups again, at most this often.
const REFRESH_MS = 10000

function normalise(raw: AgentGroup[]): AgentGroup[] {
  // Migrate: default type to 'group' for existing entries
  const groups = raw.map(g => ({ ...g, type: g.type || 'group', agentIds: g.agentIds || [] }))
  // Only one group holds the pin, whatever was saved.
  const top = groups.find(g => g.pinned && isFolder(g))
  return groups.map(g => {
    const out: AgentGroup = { ...g }
    if (g.pinned && g !== top) delete out.pinned
    if (g.pinnedAgentIds) out.pinnedAgentIds = g.pinnedAgentIds.filter(id => g.agentIds.includes(id))
    return out
  })
}

// Agents leave a group: their pins there go with them.
function without(g: AgentGroup, gone: (id: string) => boolean): AgentGroup {
  return {
    ...g,
    agentIds: g.agentIds.filter((id) => !gone(id)),
    ...(g.pinnedAgentIds ? { pinnedAgentIds: g.pinnedAgentIds.filter((id) => !gone(id)) } : {}),
  }
}

function load(): AgentGroup[] {
  try {
    return normalise(JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as AgentGroup[])
  } catch { return [] }
}

function loadCollapsed(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(COLLAPSED_KEY) || '[]')
    return Array.isArray(raw) ? raw : []
  } catch { return [] }
}

let lastSaved = 0

function save(groups: AgentGroup[]) {
  const val = JSON.stringify(groups)
  lastSaved = Date.now()
  if (useAuthStore.getState().authEnabled) {
    queueSave(STORAGE_KEY, val)
  } else {
    localStorage.setItem(STORAGE_KEY, val)
  }
}

interface GroupStore {
  groups: AgentGroup[]
  /** Folded sections on this device. */
  collapsed: string[]
  createGroup: (name: string, type?: GroupType, color?: string) => string
  deleteGroup: (id: string) => void
  renameGroup: (id: string, name: string) => void
  updateGroup: (id: string, patch: { name?: string; color?: string }) => void
  /** A group's place among the others: before `beforeId`, or last. */
  moveGroup: (id: string, beforeId: string | null) => void
  /** This group to the top of the list and no other (null: none). It keeps its place for when it's unpinned. */
  pinGroup: (id: string | null) => void
  /** An agent to the top of this group, or back among the others. Any number of them. */
  toggleAgentPin: (groupId: string, agentId: string) => void
  addToGroup: (groupId: string, agentId: string) => void
  removeFromGroup: (groupId: string, agentId: string) => void
  /** Into this group and out of any other (null: out of them all). Roles stay. */
  placeAgent: (agentId: string, groupId: string | null) => void
  /** Exactly these agents in this group: they leave any other, the rest leave this one. */
  setGroupMembers: (groupId: string, agentIds: string[]) => void
  /** An agent removed for good: a new one under its name starts in no group. */
  forgetAgent: (agentId: string) => void
  /** The same agent under a new id (a rebuilt container): it keeps its groups and pins. */
  replaceAgent: (oldId: string, newId: string) => void
  toggleCollapsed: (id: string) => void
  expand: (id: string) => void
  getGroupsForAgent: (agentId: string) => AgentGroup[]
}

export const useGroupStore = create<GroupStore>((set, get) => {
  const commit = (groups: AgentGroup[]) => {
    save(groups)
    set({ groups })
  }
  const setCollapsed = (collapsed: string[]) => {
    try { localStorage.setItem(COLLAPSED_KEY, JSON.stringify(collapsed)) } catch { /* storage blocked: this visit only */ }
    set({ collapsed })
  }

  return {
    groups: load(),
    collapsed: loadCollapsed(),

    createGroup: (name, type = 'group', color) => {
      const id = `grp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
      const group: AgentGroup = { id, name, color: color || nextGroupColor(get().groups), type, agentIds: [] }
      commit([...get().groups, group])
      return id
    },

    deleteGroup: (id) => {
      commit(get().groups.filter((g) => g.id !== id))
      if (get().collapsed.includes(id)) setCollapsed(get().collapsed.filter((x) => x !== id))
    },

    renameGroup: (id, name) => {
      commit(get().groups.map((g) => g.id === id ? { ...g, name } : g))
    },

    updateGroup: (id, patch) => {
      commit(get().groups.map((g) => g.id === id ? { ...g, ...patch } : g))
    },

    moveGroup: (id, beforeId) => {
      const moving = get().groups.find((g) => g.id === id)
      if (!moving || id === beforeId) return
      const rest = get().groups.filter((g) => g.id !== id)
      const at = beforeId ? rest.findIndex((g) => g.id === beforeId) : -1
      rest.splice(at < 0 ? rest.length : at, 0, moving)
      commit(rest)
    },

    pinGroup: (id) => {
      commit(get().groups.map((g) => {
        if (g.id === id) return { ...g, pinned: true }
        return g.pinned ? { ...g, pinned: false } : g
      }))
    },

    toggleAgentPin: (groupId, agentId) => {
      commit(get().groups.map((g) => {
        if (g.id !== groupId || !g.agentIds.includes(agentId)) return g
        const pins = g.pinnedAgentIds ?? []
        return { ...g, pinnedAgentIds: pins.includes(agentId) ? pins.filter((id) => id !== agentId) : [...pins, agentId] }
      }))
    },

    addToGroup: (groupId, agentId) => {
      commit(get().groups.map((g) => {
        if (g.id !== groupId || g.agentIds.includes(agentId)) return g
        return { ...g, agentIds: [...g.agentIds, agentId] }
      }))
    },

    removeFromGroup: (groupId, agentId) => {
      commit(get().groups.map((g) => {
        return g.id === groupId ? without(g, (id) => id === agentId) : g
      }))
    },

    placeAgent: (agentId, groupId) => {
      commit(get().groups.map((g) => {
        if (!isFolder(g)) return g
        const inIt = g.agentIds.includes(agentId)
        if (g.id === groupId) return inIt ? g : { ...g, agentIds: [...g.agentIds, agentId] }
        return inIt ? without(g, (id) => id === agentId) : g
      }))
    },

    setGroupMembers: (groupId, agentIds) => {
      const members = new Set(agentIds)
      commit(get().groups.map((g) => {
        if (g.id === groupId) return { ...without(g, (id) => !members.has(id)), agentIds }
        if (!isFolder(g) || !g.agentIds.some((id) => members.has(id))) return g
        return without(g, (id) => members.has(id))
      }))
    },

    forgetAgent: (agentId) => {
      if (!get().groups.some((g) => g.agentIds.includes(agentId))) return
      commit(get().groups.map((g) => without(g, (id) => id === agentId)))
    },

    replaceAgent: (oldId, newId) => {
      if (!get().groups.some((g) => g.agentIds.includes(oldId))) return
      const swap = (id: string) => (id === oldId ? newId : id)
      commit(get().groups.map((g) => ({
        ...g,
        agentIds: g.agentIds.map(swap),
        ...(g.pinnedAgentIds ? { pinnedAgentIds: g.pinnedAgentIds.map(swap) } : {}),
      })))
    },

    toggleCollapsed: (id) => {
      const { collapsed } = get()
      setCollapsed(collapsed.includes(id) ? collapsed.filter((x) => x !== id) : [...collapsed, id])
    },

    expand: (id) => {
      if (get().collapsed.includes(id)) setCollapsed(get().collapsed.filter((x) => x !== id))
    },

    getGroupsForAgent: (agentId) => {
      return get().groups.filter((g) => g.agentIds.includes(agentId))
    },
  }
})

// What the server says, unless it is what we already show.
function adopt(raw: string | undefined) {
  let groups: AgentGroup[] = []
  try { groups = normalise(JSON.parse(raw || '[]')) } catch { return }
  if (JSON.stringify(groups) !== JSON.stringify(useGroupStore.getState().groups)) useGroupStore.setState({ groups })
}

// With sign-in on the server's copy is the truth: no key means no groups, not
// whatever this browser last held for someone else. Without sign-in the groups
// live in this browser only.
registerHydrator((settings) => {
  if (useAuthStore.getState().authEnabled) adopt(settings[STORAGE_KEY])
})

// A phone or a tab left open keeps its groups for days, and the next change
// there would write them over the ones made elsewhere since. Coming back to it
// reads them again.
if (typeof window !== 'undefined') {
  let lastRead = Date.now()
  const again = async () => {
    const { authEnabled, isAuthenticated } = useAuthStore.getState()
    if (!authEnabled || !isAuthenticated || document.visibilityState !== 'visible') return
    if (Date.now() - Math.max(lastRead, lastSaved) < REFRESH_MS) return
    lastRead = Date.now()
    const asked = lastSaved
    const settings = await fetchSettingsOrNull()
    // A change made meanwhile is newer than this answer.
    if (settings && lastSaved === asked) adopt(settings[STORAGE_KEY])
  }
  window.addEventListener('focus', again)
  document.addEventListener('visibilitychange', again)
}
