import { create } from 'zustand'
import { fetchLineCounts, type LineCounts } from '../services/line'

// The Line's badge: how many items are unread, and how many still wait for
// the user. Refreshed with the bell's poll, and when the Line page reads.
interface LineStore extends LineCounts {
  refresh: () => Promise<void>
  set: (counts: LineCounts) => void
}

export const useLineStore = create<LineStore>((set) => ({
  unread: 0,
  waiting: 0,
  refresh: async () => {
    const counts = await fetchLineCounts()
    if (counts) set(counts)
  },
  set: (counts) => set(counts),
}))
