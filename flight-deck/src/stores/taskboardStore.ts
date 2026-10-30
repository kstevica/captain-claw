import { create } from 'zustand'
import { registerSignOutTeardown } from './authStore'
import { taskboardApi } from '../services/taskboardApi'

/** The kiosk's Taskboard dialog: whether it is open, the agent it opens
 *  filtered to (none = every agent), the task it opens showing, and how many
 *  tasks need the user — the count on its buttons. */
interface TaskboardStore {
  open: boolean
  agentFilter: string | null
  /** Open showing this task: its id, or a key it links back with
   *  (`sched:job:<id>`, `autonomy:<id>`) that the board looks up. */
  focus: string | null
  needsMe: number
  openTaskboard: (agentSlug?: string) => void
  showTask: (taskRef: string) => void
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
  showTask: (taskRef) => set({ open: true, agentFilter: null, focus: taskRef }),
  close: () => set({ open: false, agentFilter: null, focus: null }),
  fetchSummary: async () => {
    try {
      const s = await taskboardApi.summary()
      set({ needsMe: Math.max(0, Number(s?.needs_me) || 0) })
    } catch { /* keep the last count */ }
  },
}))

// What a link may name: a task id or a key, never anything longer or odder.
const LINKED = /^[A-Za-z0-9_.:-]{1,200}$/

/** Whether the address holds a Taskboard link (`?taskboard=…`) not taken yet. */
export function hasTaskboardLink(): boolean {
  try {
    return new URL(window.location.href).searchParams.has('taskboard')
  } catch {
    return false
  }
}

/** The task a link opens the board on (`?taskboard=<task id>`: nudges,
 *  bell items and notifications carry it), taken out of the address so a
 *  reload or Back doesn't open it again. '' when there is none. */
export function takeTaskboardLink(): string {
  try {
    const url = new URL(window.location.href)
    if (!url.searchParams.has('taskboard')) return ''
    const id = (url.searchParams.get('taskboard') || '').trim()
    url.searchParams.delete('taskboard')
    window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`)
    return LINKED.test(id) ? id : ''
  } catch {
    return ''
  }
}

// The next person at a shared kiosk starts with no board open and no count.
registerSignOutTeardown(() => {
  useTaskboardStore.setState({ open: false, agentFilter: null, focus: null, needsMe: 0 })
})
