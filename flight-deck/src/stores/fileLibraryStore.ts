import { create } from 'zustand'

/** A file the library opens showing: searched for by name, then selected. */
export interface LibraryFocus {
  agent: string
  /** Path below the agent's workspace, as the library names it. */
  relPath: string
  filename: string
}

/** The kiosk's File library dialog: whether it is open, and the agent it
 *  opens filtered to (none = every agent). */
interface FileLibraryStore {
  open: boolean
  agentFilter: string | null
  focus: LibraryFocus | null
  openLibrary: (agentSlug?: string) => void
  /** Open on one file (the chat's Files menu → Show in library). */
  showFile: (focus: LibraryFocus) => void
  close: () => void
}

export const useFileLibraryStore = create<FileLibraryStore>((set) => ({
  open: false,
  agentFilter: null,
  focus: null,
  openLibrary: (agentSlug) => set({ open: true, agentFilter: agentSlug ?? null, focus: null }),
  showFile: (focus) => set({ open: true, agentFilter: focus.agent, focus }),
  close: () => set({ open: false, agentFilter: null, focus: null }),
}))
