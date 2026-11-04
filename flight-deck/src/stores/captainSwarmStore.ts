import { create } from 'zustand'
import { useAuthStore } from './authStore'
import { isDesktopApp } from '../services/desktopShell'

/** Captain Swarm is for the deck's admins (the server refuses everyone else).
 *  A deck with sign-in off has one user, its local admin. Never in the desktop
 *  app (its deck turns Captain Swarm off): the deck says so (desktopDeck), and
 *  the page knows it before that answer arrives. */
export const useCanSwarm = () => useAuthStore((s) =>
  (s.authEnabled === false || s.user?.role === 'admin') && !s.desktopDeck && !isDesktopApp())

/** The kiosk's Captain Swarm dialog: whether it is open. */
interface CaptainSwarmStore {
  open: boolean
  openSwarm: () => void
  close: () => void
}

export const useCaptainSwarmStore = create<CaptainSwarmStore>((set) => ({
  open: false,
  openSwarm: () => set({ open: true }),
  close: () => set({ open: false }),
}))
