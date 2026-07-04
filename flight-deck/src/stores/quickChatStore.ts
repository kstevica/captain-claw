import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

const STORAGE_KEY = 'fd:quick-chat'

/** A lightweight agent spawned from the Quick Chat page (hidden on the Agent Desktop until promoted). */
export interface QuickChatSession {
  /** Process slug — the chat/desktop id is `proc-${slug}`. */
  slug: string
  role: string
  name: string
  tier: string
  /** True once the user promoted the agent onto the Agent Desktop. */
  promoted: boolean
  createdAt: number
}

interface QuickChatStore {
  sessions: QuickChatSession[]
  add: (s: QuickChatSession) => void
  remove: (slug: string) => void
  setPromoted: (slug: string, promoted: boolean) => void
}

function load(): QuickChatSession[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return Array.isArray(raw.sessions) ? raw.sessions : []
  } catch {
    return []
  }
}

function save(sessions: QuickChatSession[]) {
  const val = JSON.stringify({ sessions })
  if (useAuthStore.getState().authEnabled) queueSave(STORAGE_KEY, val)
  else localStorage.setItem(STORAGE_KEY, val)
}

export const useQuickChatStore = create<QuickChatStore>((set, get) => ({
  sessions: load(),

  add: (s) => {
    const sessions = [s, ...get().sessions.filter((x) => x.slug !== s.slug)]
    save(sessions)
    set({ sessions })
  },

  remove: (slug) => {
    const sessions = get().sessions.filter((x) => x.slug !== slug)
    save(sessions)
    set({ sessions })
  },

  setPromoted: (slug, promoted) => {
    const sessions = get().sessions.map((x) => (x.slug === slug ? { ...x, promoted } : x))
    save(sessions)
    set({ sessions })
  },
}))

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed.sessions)) {
        useQuickChatStore.setState({ sessions: parsed.sessions })
      }
    } catch { /* ignore */ }
  }
})
