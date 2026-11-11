import { create } from 'zustand'
import { schedulerApi, type AgentCronJob, type ScheduledJob } from '../services/schedulerApi'

/** The kiosk's Scheduled tasks dialog: whether it is open, what it opens on
 *  (a task a chat message points at, or a new task for an agent), and the
 *  task list the entry buttons count — Flight Deck's own tasks and the jobs
 *  agents scheduled for themselves with their cron tool. */
interface ScheduledStore {
  open: boolean
  /** Opened from a chat message's "Scheduled" chip: show this task. */
  focusTaskId: string | null
  /** Opened as "new task" for this agent slug. */
  newForAgent: string | null
  jobs: ScheduledJob[]
  /** The agents' own schedules (their cron), from each agent's file. */
  agentJobs: AgentCronJob[]
  loaded: boolean
  openTasks: (taskId?: string) => void
  openNew: (agentSlug?: string) => void
  close: () => void
  fetchJobs: () => Promise<void>
}

export const useScheduledStore = create<ScheduledStore>((set) => ({
  open: false,
  focusTaskId: null,
  newForAgent: null,
  jobs: [],
  agentJobs: [],
  loaded: false,
  openTasks: (taskId) => set({ open: true, focusTaskId: taskId ?? null, newForAgent: null }),
  openNew: (agentSlug) => set({ open: true, focusTaskId: null, newForAgent: agentSlug ?? '' }),
  close: () => set({ open: false, focusTaskId: null, newForAgent: null }),
  fetchJobs: async () => {
    // Each list on its own: one failing keeps the other (and what it showed).
    const [jobs, agentJobs] = await Promise.allSettled([schedulerApi.listJobs(), schedulerApi.listAgentJobs()])
    set((s) => ({
      jobs: jobs.status === 'fulfilled' && Array.isArray(jobs.value) ? jobs.value : s.jobs,
      agentJobs: agentJobs.status === 'fulfilled' && Array.isArray(agentJobs.value) ? agentJobs.value : s.agentJobs,
      loaded: true,
    }))
  },
}))

/** Tasks only (jobs the kiosk made), not WhatsApp/channel jobs from the full Scheduler page. */
export function isTask(j: ScheduledJob): boolean {
  return j.delivery_kind === 'agent'
}
