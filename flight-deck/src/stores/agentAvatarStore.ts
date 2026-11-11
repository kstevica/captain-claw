import { create } from 'zustand'
import { fetchSettingsOrNull, queueSave, registerHydrator } from '../services/settingsSync'
import { deleteAvatarImage } from '../services/agentAvatars'
import { useAuthStore } from './authStore'

// The picture a person gave each of their agents, by agent id: an emoji, or a
// picture they uploaded (its id at /fd/agent-avatars). No entry: its initials.
// Theirs alone and synced with their settings, like an agent's name — so it
// follows them to their phone, and someone they share an agent with picks
// their own.

export type AgentAvatar = { emoji: string } | { image: string }

const STORAGE_KEY = 'fd:agent-avatars'

function clean(raw: unknown): Record<string, AgentAvatar> {
  const out: Record<string, AgentAvatar> = {}
  if (!raw || typeof raw !== 'object') return out
  for (const [id, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!v || typeof v !== 'object') continue
    const { emoji, image } = v as { emoji?: unknown; image?: unknown }
    if (typeof emoji === 'string' && emoji && emoji.length <= 16) out[id] = { emoji }
    else if (typeof image === 'string' && /^[0-9a-f]{32}$/.test(image)) out[id] = { image }
  }
  return out
}

function load(): Record<string, AgentAvatar> {
  try { return clean(JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')) } catch { return {} }
}

function save(avatars: Record<string, AgentAvatar>) {
  const val = JSON.stringify(avatars)
  if (useAuthStore.getState().authEnabled) queueSave(STORAGE_KEY, val)
  else {
    try { localStorage.setItem(STORAGE_KEY, val) } catch { /* storage blocked: this visit only */ }
  }
}

const imageOf = (a: AgentAvatar | undefined) => (a && 'image' in a ? a.image : '')

// Changes made in this tab: a read that started before one is older than it.
let edits = 0

interface AgentAvatarStore {
  avatars: Record<string, AgentAvatar>
  /** null: back to its initials. */
  setAvatar: (agentId: string, avatar: AgentAvatar | null) => void
  /** The same agent under a new id (a rebuilt container). */
  replaceAgent: (oldId: string, newId: string) => void
  /** An agent removed for good. */
  forgetAgent: (agentId: string) => void
}

export const useAgentAvatarStore = create<AgentAvatarStore>((set, get) => {
  // An uploaded picture no agent wears any more goes from the server too.
  const commit = (avatars: Record<string, AgentAvatar>) => {
    const before = get().avatars
    edits++
    save(avatars)
    set({ avatars })
    const worn = new Set(Object.values(avatars).map(imageOf).filter(Boolean))
    for (const img of new Set(Object.values(before).map(imageOf))) {
      if (img && !worn.has(img)) void deleteAvatarImage(img)
    }
  }
  return {
    avatars: load(),

    setAvatar: (agentId, avatar) => {
      const next = { ...get().avatars }
      if (avatar) next[agentId] = avatar
      else delete next[agentId]
      commit(next)
    },

    replaceAgent: (oldId, newId) => {
      const { avatars } = get()
      if (!avatars[oldId] || oldId === newId) return
      const next = { ...avatars, [newId]: avatars[oldId] }
      delete next[oldId]
      commit(next)
    },

    forgetAgent: (agentId) => {
      if (!get().avatars[agentId]) return
      const next = { ...get().avatars }
      delete next[agentId]
      commit(next)
    },
  }
})

// What the server says, unless it is what we already show.
function adopt(raw: string | undefined) {
  let avatars: Record<string, AgentAvatar> = {}
  try { avatars = clean(JSON.parse(raw || '{}')) } catch { return }
  if (JSON.stringify(avatars) !== JSON.stringify(useAgentAvatarStore.getState().avatars)) {
    useAgentAvatarStore.setState({ avatars })
  }
}

// With sign-in the server's copy is the truth: no key means no pictures.
registerHydrator((settings) => {
  if (useAuthStore.getState().authEnabled) adopt(settings[STORAGE_KEY])
})

/** Read them again before a change (a picture picked on another device since
 *  this tab loaded must not be written over). */
export async function refreshAgentAvatars(): Promise<void> {
  const { authEnabled, isAuthenticated } = useAuthStore.getState()
  if (!authEnabled || !isAuthenticated) return
  const asked = edits
  const settings = await fetchSettingsOrNull()
  if (settings && edits === asked) adopt(settings[STORAGE_KEY])
}
