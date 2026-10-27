import { create } from 'zustand'
import { registerSignOutTeardown } from './authStore'

// Where "Your own AI keys" opens from: the kiosk Settings, and the plan bar
// once the day's allowance runs low or is spent. The dialog lives in the
// kiosk layout (`host`), so the bar offers it only there.

interface ByokStore {
  open: boolean
  /** A layout that can show the dialog is mounted. */
  host: boolean
  openDialog: () => void
  close: () => void
  setHost: (host: boolean) => void
}

export const useByokStore = create<ByokStore>((set) => ({
  open: false,
  host: false,
  openDialog: () => set({ open: true }),
  close: () => set({ open: false }),
  setHost: (host) => set({ host }),
}))

registerSignOutTeardown(() => useByokStore.setState({ open: false }))
