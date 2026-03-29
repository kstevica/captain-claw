import { create } from 'zustand'

export interface ClipboardEntry {
  id: string
  content: string
  source: string // agent name or "user"
  createdAt: string
  pinned: boolean
}

const STORAGE_KEY = 'fd:shared-clipboard'

function load(): ClipboardEntry[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function save(entries: ClipboardEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
}

interface ClipboardStore {
  entries: ClipboardEntry[]
  addEntry: (content: string, source: string) => void
  removeEntry: (id: string) => void
  togglePin: (id: string) => void
  updateEntry: (id: string, content: string) => void
  clear: () => void
}

export const useClipboardStore = create<ClipboardStore>((set, get) => ({
  entries: load(),

  addEntry: (content, source) => {
    const id = `clip-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const entry: ClipboardEntry = { id, content, source, createdAt: new Date().toISOString(), pinned: false }
    const entries = [entry, ...get().entries].slice(0, 100) // keep max 100
    save(entries)
    set({ entries })
  },

  removeEntry: (id) => {
    const entries = get().entries.filter((e) => e.id !== id)
    save(entries)
    set({ entries })
  },

  togglePin: (id) => {
    const entries = get().entries.map((e) => e.id === id ? { ...e, pinned: !e.pinned } : e)
    save(entries)
    set({ entries })
  },

  updateEntry: (id, content) => {
    const entries = get().entries.map((e) => e.id === id ? { ...e, content } : e)
    save(entries)
    set({ entries })
  },

  clear: () => {
    const entries = get().entries.filter((e) => e.pinned) // keep pinned
    save(entries)
    set({ entries })
  },
}))
