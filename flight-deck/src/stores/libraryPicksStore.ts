import { create } from 'zustand'
import { fileLibraryApi, type LibraryPick } from '../services/fileLibraryApi'

/**
 * What of the person's shared folders is in the File library: only what they added,
 * a whole folder (`path` '') or single files. Read once, kept here so every
 * "Add to Library" button on the page agrees.
 */

const key = (folder: string, path: string) => `${folder}\u0000${path}`

interface PicksState {
  picks: Set<string>
  loaded: boolean
  loading: boolean
  load: (force?: boolean) => Promise<void>
  /** In the library: picked itself, or its whole folder is. */
  has: (folder: string, path?: string) => boolean
  /** Picked itself (what taking it out undoes). */
  own: (folder: string, path?: string) => boolean
  /** Something of the folder is in the library. */
  anyIn: (folder: string) => boolean
  add: (folder: string, path?: string) => Promise<void>
  drop: (folder: string, path?: string) => Promise<void>
}

const asSet = (list: LibraryPick[]) => new Set(list.map((p) => key(p.folder, p.path)))

export const useLibraryPicks = create<PicksState>((set, get) => ({
  picks: new Set(),
  loaded: false,
  loading: false,
  load: async (force = false) => {
    if (get().loading || (get().loaded && !force)) return
    set({ loading: true })
    try {
      set({ picks: asSet(await fileLibraryApi.picks()), loaded: true })
    } finally {
      set({ loading: false })
    }
  },
  has: (folder, path = '') => get().picks.has(key(folder, '')) || (!!path && get().picks.has(key(folder, path))),
  own: (folder, path = '') => get().picks.has(key(folder, path)),
  anyIn: (folder) => [...get().picks].some((k) => k.startsWith(`${folder}\u0000`)),
  add: async (folder, path = '') => {
    set({ picks: asSet((await fileLibraryApi.addPick(folder, path)).picks), loaded: true })
  },
  drop: async (folder, path = '') => {
    set({ picks: asSet((await fileLibraryApi.dropPick(folder, path)).picks), loaded: true })
  },
}))
