import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

export interface PinnedMessage {
  id: string
  agentId: string
  agentName: string
  content: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  model?: string
  pinnedAt: string
  tags: string[]
  note?: string
}

const STORAGE_KEY = 'fd:pinned-messages'

function load(): PinnedMessage[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function save(pins: PinnedMessage[]) {
  const val = JSON.stringify(pins)
  if (useAuthStore.getState().authEnabled) queueSave(STORAGE_KEY, val)
  else localStorage.setItem(STORAGE_KEY, val)
}

interface PinnedStore {
  pins: PinnedMessage[]
  pin: (msg: Omit<PinnedMessage, 'id' | 'pinnedAt' | 'tags' | 'note'>) => void
  unpin: (id: string) => void
  updatePin: (id: string, patch: Partial<Pick<PinnedMessage, 'tags' | 'note'>>) => void
  isPinned: (agentId: string, content: string) => boolean
}

export const usePinnedStore = create<PinnedStore>((set, get) => ({
  pins: load(),

  pin: (msg) => {
    const id = `pin-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const pin: PinnedMessage = { ...msg, id, pinnedAt: new Date().toISOString(), tags: [], note: '' }
    const pins = [pin, ...get().pins]
    save(pins)
    set({ pins })
  },

  unpin: (id) => {
    const pins = get().pins.filter((p) => p.id !== id)
    save(pins)
    set({ pins })
  },

  updatePin: (id, patch) => {
    const pins = get().pins.map((p) => p.id === id ? { ...p, ...patch } : p)
    save(pins)
    set({ pins })
  },

  isPinned: (agentId, content) => {
    return get().pins.some((p) => p.agentId === agentId && p.content === content)
  },
}))

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try { usePinnedStore.setState({ pins: JSON.parse(raw) }) } catch { /* ignore */ }
  }
})
