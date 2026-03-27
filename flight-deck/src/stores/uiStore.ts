import { create } from 'zustand'
import type { ViewMode } from '../types'

interface UIStore {
  view: ViewMode
  sidebarOpen: boolean
  panelOpen: boolean
  setView: (v: ViewMode) => void
  toggleSidebar: () => void
  togglePanel: () => void
  setPanelOpen: (v: boolean) => void
}

export const useUIStore = create<UIStore>((set) => ({
  view: 'desktop',
  sidebarOpen: true,
  panelOpen: false,

  setView: (view) => set({ view }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setPanelOpen: (panelOpen) => set({ panelOpen }),
}))
