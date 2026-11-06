import { create } from 'zustand'
import { lessonProposalsApi, type ProposalsOverview } from '../services/agentLessons'

/** Lessons to approve: what the owner's agents noticed in their chats (and members suggested), across every
 *  agent. One dialog for the whole app (LessonInboxHost); the bell's rows and the badge beside it open it. */
interface LessonInboxState {
  data: ProposalsOverview | null
  /** How many wait, for the badge (0 until the first load). */
  waiting: number
  open: boolean
  /** The agent to bring into view when it opens ("" for none). */
  focus: string
  refresh: () => Promise<void>
  show: (focus?: string) => void
  hide: () => void
}

let inFlight: Promise<void> | null = null

export const useLessonInbox = create<LessonInboxState>((set, get) => ({
  data: null,
  waiting: 0,
  open: false,
  focus: '',
  refresh: () => {
    if (inFlight) return inFlight
    inFlight = lessonProposalsApi.list()
      .then((data) => set({ data, waiting: data.waiting }))
      .catch(() => { /* an older deck without the route, or offline: keep what we had */ })
      .finally(() => { inFlight = null })
    return inFlight
  },
  show: (focus = '') => {
    set({ open: true, focus })
    void get().refresh()
  },
  hide: () => set({ open: false, focus: '' }),
}))
