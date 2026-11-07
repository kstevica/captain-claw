import { create } from 'zustand'
import { customToolsApi, type ToolStatus } from '../services/customToolsApi'

// Custom tools: whether the deck has them on (Admin → Settings), and the page's
// open state for the kiosk, where it opens as a dialog. A bell can ask it to
// open on a tab or a tool.

export type ToolsTab = 'mine' | 'shared' | 'catalog' | 'approvals'

// Brings a custom tool's question (Phase 7d) back after "Later" (the bell sends it).
export const SHOW_ASKS_EVENT = 'fd:custom-tool-asks'

export interface ToolsFocus {
  tab?: ToolsTab
  owner?: string
  slug?: string
}

interface CustomToolsState {
  status: ToolStatus | null
  open: boolean
  focus: ToolsFocus | null
  refreshStatus: () => Promise<void>
  openTools: (focus?: ToolsFocus) => void
  takeFocus: () => ToolsFocus | null
  close: () => void
}

let inFlight: Promise<void> | null = null

export const useCustomToolsStore = create<CustomToolsState>((set, get) => ({
  status: null,
  open: false,
  focus: null,
  refreshStatus: () => {
    if (inFlight) return inFlight
    inFlight = customToolsApi.status()
      .then((status) => set({ status }))
      .catch(() => set({ status: null }))
      .finally(() => { inFlight = null })
    return inFlight
  },
  openTools: (focus) => set({ open: true, focus: focus || null }),
  takeFocus: () => {
    const focus = get().focus
    if (focus) set({ focus: null })
    return focus
  },
  close: () => set({ open: false, focus: null }),
}))

/** Does this deck have custom tools on? (The server checks every call; this only
 *  decides what's shown.) */
export const useCustomToolsOn = () => useCustomToolsStore((s) => !!s.status?.enabled)
