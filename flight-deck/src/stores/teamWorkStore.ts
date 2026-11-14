import { create } from 'zustand'

/** Agents working as a team, opened from the kiosk's More menu: Vatra (a team
 *  that splits one job and assembles the result), Basna (several agents answer,
 *  the best of each merged) and Council (agents discuss a topic in rounds).
 *  Vatra and Basna are one page, two modes; which one it opens on is `open`. */
export type TeamWork = 'vatra' | 'basna' | 'council'

interface TeamWorkStore {
  open: TeamWork | null
  openTeamWork: (kind: TeamWork) => void
  close: () => void
}

export const useTeamWorkStore = create<TeamWorkStore>((set) => ({
  open: null,
  openTeamWork: (kind) => set({ open: kind }),
  close: () => set({ open: null }),
}))
