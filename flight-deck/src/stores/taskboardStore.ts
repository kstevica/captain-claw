import { create } from 'zustand'
import { registerSignOutTeardown } from './authStore'
import { taskboardApi } from '../services/taskboardApi'

/** The kiosk's Taskboard dialog: whether it is open, the agent it opens
 *  filtered to (none = every agent), the task it opens showing, and how many
 *  tasks need the user — the count on its buttons. */
interface TaskboardStore {
  open: boolean
  agentFilter: string | null
  /** Open showing this task (a task id). */
  focus: string | null
  needsMe: number
  openTaskboard: (agentSlug?: string) => void
  showTask: (taskId: string) => void
  close: () => void
  /** The Needs me count (cheap; polled for the badge). */
  fetchSummary: () => Promise<void>
}

export const useTaskboardStore = create<TaskboardStore>((set) => ({
  open: false,
  agentFilter: null,
  focus: null,
  needsMe: 0,
  openTaskboard: (agentSlug) => set({ open: true, agentFilter: agentSlug ?? null, focus: null }),
  showTask: (taskId) => set({ open: true, agentFilter: null, focus: taskId }),
  close: () => set({ open: false, agentFilter: null, focus: null }),
  fetchSummary: async () => {
    try {
      const s = await taskboardApi.summary()
      set({ needsMe: Math.max(0, Number(s?.needs_me) || 0) })
    } catch { /* keep the last count */ }
  },
}))

// The next person at a shared kiosk starts with no board open and no count.
registerSignOutTeardown(() => {
  useTaskboardStore.setState({ open: false, agentFilter: null, focus: null, needsMe: 0 })
})
