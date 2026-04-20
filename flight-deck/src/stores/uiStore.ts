import { create } from 'zustand'
import type { ViewMode } from '../types'

type MobilePanel = 'none' | 'director' | 'chat' | 'tool'

interface UIStore {
  view: ViewMode
  sidebarOpen: boolean
  panelOpen: boolean
  mobilePanel: MobilePanel
  sidebarDrawerOpen: boolean
  forgeProjectId: string
  setView: (v: ViewMode) => void
  toggleSidebar: () => void
  togglePanel: () => void
  setPanelOpen: (v: boolean) => void
  setMobilePanel: (panel: MobilePanel) => void
  toggleMobilePanel: (panel: MobilePanel) => void
  setSidebarDrawerOpen: (v: boolean) => void
  setForgeProjectId: (id: string) => void
}

export const useUIStore = create<UIStore>((set) => ({
  view: 'desktop',
  sidebarOpen: true,
  panelOpen: false,
  mobilePanel: 'none',
  sidebarDrawerOpen: false,
  forgeProjectId: '',

  setView: (view) => set({ view }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setPanelOpen: (panelOpen) => set({ panelOpen }),
  setMobilePanel: (mobilePanel) => set({ mobilePanel }),
  toggleMobilePanel: (panel) => set((s) => ({ mobilePanel: s.mobilePanel === panel ? 'none' : panel })),
  setSidebarDrawerOpen: (sidebarDrawerOpen) => set({ sidebarDrawerOpen }),
  setForgeProjectId: (forgeProjectId) => set({ forgeProjectId }),
}))
