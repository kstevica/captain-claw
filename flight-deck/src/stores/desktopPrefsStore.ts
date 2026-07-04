import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

const STORAGE_KEY = 'fd:desktop-prefs'

interface DesktopPrefsStore {
  /** Agent ids (docker c.id, `proc-<slug>`, or local a.id) hidden from the Agent Desktop canvas. */
  hiddenAgentIds: string[]
  toggleAgentHidden: (id: string) => void
  /** Deterministically hide (true) or reveal (false) an agent — avoids toggle double-flips. */
  setAgentHidden: (id: string, hidden: boolean) => void
  isAgentHidden: (id: string) => boolean
  showAllAgents: () => void
}

function load(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return Array.isArray(raw.hiddenAgentIds) ? raw.hiddenAgentIds : []
  } catch {
    return []
  }
}

function save(hiddenAgentIds: string[]) {
  const val = JSON.stringify({ hiddenAgentIds })
  if (useAuthStore.getState().authEnabled) {
    queueSave(STORAGE_KEY, val)
  } else {
    localStorage.setItem(STORAGE_KEY, val)
  }
}

export const useDesktopPrefsStore = create<DesktopPrefsStore>((set, get) => ({
  hiddenAgentIds: load(),

  toggleAgentHidden: (id) => {
    const cur = get().hiddenAgentIds
    const hiddenAgentIds = cur.includes(id)
      ? cur.filter((x) => x !== id)
      : [...cur, id]
    save(hiddenAgentIds)
    set({ hiddenAgentIds })
  },

  setAgentHidden: (id, hidden) => {
    const cur = get().hiddenAgentIds
    if (hidden === cur.includes(id)) return
    const hiddenAgentIds = hidden ? [...cur, id] : cur.filter((x) => x !== id)
    save(hiddenAgentIds)
    set({ hiddenAgentIds })
  },

  isAgentHidden: (id) => get().hiddenAgentIds.includes(id),

  showAllAgents: () => {
    save([])
    set({ hiddenAgentIds: [] })
  },
}))

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed.hiddenAgentIds)) {
        useDesktopPrefsStore.setState({ hiddenAgentIds: parsed.hiddenAgentIds })
      }
    } catch { /* ignore */ }
  }
})
