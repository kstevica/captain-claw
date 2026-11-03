import { create } from 'zustand'

/** The kiosk's Captain Swarm dialog: whether it is open. Every user may start
 *  runs (on their own models; the server checks each one), so it has no gate. */
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
