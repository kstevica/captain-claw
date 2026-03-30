import { create } from 'zustand'

export interface PinnedFile {
  id: string
  agentId: string
  agentName: string
  /** Host:port for fetching the file */
  host: string
  port: number
  auth: string
  /** File metadata */
  filename: string
  extension: string
  physical: string   // path on agent for download/view URLs
  logical: string
  size: number
  mime_type: string
  /** Pin metadata */
  pinnedAt: string
  tags: string[]
  note?: string
}

const STORAGE_KEY = 'fd:pinned-files'

function load(): PinnedFile[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function save(pins: PinnedFile[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(pins))
}

interface PinnedFilesStore {
  pins: PinnedFile[]
  pin: (file: Omit<PinnedFile, 'id' | 'pinnedAt' | 'tags' | 'note'>) => void
  unpin: (id: string) => void
  updatePin: (id: string, patch: Partial<Pick<PinnedFile, 'tags' | 'note'>>) => void
  isPinned: (agentId: string, physical: string) => boolean
}

export const usePinnedFilesStore = create<PinnedFilesStore>((set, get) => ({
  pins: load(),

  pin: (file) => {
    const id = `pf-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const pin: PinnedFile = { ...file, id, pinnedAt: new Date().toISOString(), tags: [], note: '' }
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

  isPinned: (agentId, physical) => {
    return get().pins.some((p) => p.agentId === agentId && p.physical === physical)
  },
}))
