import { create } from 'zustand'

// The kiosk's Data rooms dialog: whether it is open.

interface RoomsState {
  open: boolean
  openRooms: () => void
  close: () => void
}

export const useRoomsStore = create<RoomsState>((set) => ({
  open: false,
  openRooms: () => set({ open: true }),
  close: () => set({ open: false }),
}))
