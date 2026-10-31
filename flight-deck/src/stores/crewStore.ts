import { create } from 'zustand'
import { useAuthStore } from './authStore'

/** The kiosk's Crew dialog: whether it is open. */
interface CrewStore {
  open: boolean
  openCrew: () => void
  close: () => void
}

export const useCrewStore = create<CrewStore>((set) => ({
  open: false,
  openCrew: () => set({ open: true }),
  close: () => set({ open: false }),
}))

/** May this user run Crew? An admin approves it, per user (the server checks it
 *  on every call; this only decides what's shown). A deck without sign-in has
 *  one user, who may. */
export const useCrewAllowed = () =>
  useAuthStore((s) => (s.authEnabled ? !!s.user?.crew : true))
