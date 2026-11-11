import { create } from 'zustand'

// The kiosk's Anonymize dialog: whether it is open, and the tab it opens on.

export type AnonymizeTab = 'anonymize' | 'restore' | 'vaults'

interface AnonymizeState {
  open: boolean
  tab: AnonymizeTab
  openAnonymize: (tab?: AnonymizeTab) => void
  close: () => void
}

export const useAnonymizeStore = create<AnonymizeState>((set) => ({
  open: false,
  tab: 'anonymize',
  openAnonymize: (tab = 'anonymize') => set({ open: true, tab }),
  close: () => set({ open: false }),
}))
