import { create } from 'zustand'
import { lessonProposalsApi, type ProposalsOverview } from '../services/agentLessons'

/** Lessons to approve: what the owner's agents noticed in their chats (and members suggested), across every
 *  agent. One dialog for the app (LessonInboxHost); the badge beside the bell, the bell's lesson rows and an
 *  agent's kiosk menu open it. */
interface LessonInboxState {
  data: ProposalsOverview | null
  /** How many wait, for the badge (0 until the first load). */
  waiting: number
  open: boolean
  /** The agent to bring into view when it opens ("" for none). */
  focus: string
  /** Opened from one agent's menu: only its lessons show, under its name ("" = every agent's). */
  scopeName: string
  refresh: () => Promise<void>
  /** Open it — at *focus*, or (with *scope*) showing only that agent's. */
  show: (focus?: string, scope?: { name: string }) => void
  /** From one agent's lessons to every agent's. */
  widen: () => void
  hide: () => void
}

let inFlight: Promise<void> | null = null

export const useLessonInbox = create<LessonInboxState>((set, get) => ({
  data: null,
  waiting: 0,
  open: false,
  focus: '',
  scopeName: '',
  refresh: () => {
    if (inFlight) return inFlight
    inFlight = lessonProposalsApi.list()
      .then((data) => set({ data, waiting: data.waiting }))
      .catch(() => { /* an older deck without the route, or offline: keep what we had */ })
      .finally(() => { inFlight = null })
    return inFlight
  },
  show: (focus = '', scope) => {
    set({ open: true, focus, scopeName: focus && scope ? scope.name || focus : '' })
    void get().refresh()
  },
  widen: () => set({ scopeName: '' }),
  hide: () => set({ open: false, focus: '', scopeName: '' }),
}))
