import { create } from 'zustand'
import type { ViewMode } from '../types'

type MobilePanel = 'none' | 'director' | 'chat' | 'tool'

interface UIStore {
  view: ViewMode
  sidebarOpen: boolean
  panelOpen: boolean
  mobilePanel: MobilePanel
  sidebarDrawerOpen: boolean
  setView: (v: ViewMode) => void
  toggleSidebar: () => void
  togglePanel: () => void
  setPanelOpen: (v: boolean) => void
  setMobilePanel: (panel: MobilePanel) => void
  toggleMobilePanel: (panel: MobilePanel) => void
  setSidebarDrawerOpen: (v: boolean) => void
}

export const useUIStore = create<UIStore>((set) => ({
  view: 'desktop',
  sidebarOpen: true,
  panelOpen: false,
  mobilePanel: 'none',
  sidebarDrawerOpen: false,

  setView: (view) => set({ view }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setPanelOpen: (panelOpen) => set({ panelOpen }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  toggleMobilePanel: (panel) => set((s) => ({ mobilePanel: s.mobilePanel === panel ? 'none' : panel })),
  setSidebarDrawerOpen: (sidebarDrawerOpen) => set({ sidebarDrawerOpen }),
}))
