import type { ComponentType } from 'react'
import { Activity, AlarmClock, Bot, Contact, Inbox, ListChecks, Sun, Zap } from 'lucide-react'

// ── Simple layout: the outcome nav's destinations ────────────────────
//
// What the person comes for, in plain words, above their team of agents:
// Today (the briefing: what waits for the person, today's work), Agents (the
// agents that are up, as cards), Playbooks (ready-made ways to get things
// done on the web, run by an agent), Inbox (the Line: what the
// deck and the agents said on their own), Tasks (the Taskboard), People
// (the people on those tasks), Automations (Scheduled tasks) and Activity
// (every tool the agents used). Each opens the same view as before; only
// the way in changed. SimpleNav.tsx draws them. The Agent store has no entry:
// New agent opens it.
//
// They come in groups: what is for the person (Today, Inbox, Tasks, People),
// the agents' work (Agents, Playbooks, Automations, Activity), and the
// person's browser (Captain for Chrome: ChromeExtension.tsx, not an entry
// here — it is a status, not a page).

export type NavTarget = 'today' | 'agents' | 'playbooks' | 'inbox' | 'tasks' | 'people' | 'automations' | 'activity'

export type NavGroupId = 'you' | 'work' | 'browser'

/** The groups, in the order the nav shows them. `fold`: its heading folds it. */
export const NAV_GROUPS: { id: NavGroupId; label: string; fold: boolean }[] = [
  { id: 'you', label: 'For you', fold: true },
  { id: 'work', label: 'Work', fold: true },
  { id: 'browser', label: 'Browser', fold: false },
]

export interface NavEntry {
  id: NavTarget
  group: NavGroupId
  label: string
  /** The tooltip: what's there, in a few words. */
  hint: string
  icon: ComponentType<{ className?: string }>
  /** A count beside it; `hot` when it wants the person (it uses the accent). */
  count?: number
  hot?: boolean
  /** A small dot instead of a count (something running). */
  dot?: boolean
  onSelect: () => void
}

/** The destinations, in the order the nav shows them. */
export function navEntries(o: {
  onToday: () => void
  onAgents: () => void
  onPlaybooks: () => void
  onInbox: () => void
  inboxUnread: number
  inboxWaiting: number
  onTasks: () => void
  tasksNeedMe: number
  onPeople: () => void
  onAutomations: () => void
  automationsCount: number
  automationsRunning: boolean
  onActivity: () => void
}): NavEntry[] {
  const inboxCount = o.inboxWaiting || o.inboxUnread
  return [
    // For you
    { id: 'today', group: 'you', label: 'Today', icon: Sun, hint: 'What needs you, what is in progress, what got done', onSelect: o.onToday },
    {
      id: 'inbox', group: 'you', label: 'Inbox', icon: Inbox, onSelect: o.onInbox,
      hint: o.inboxWaiting ? `${o.inboxWaiting} waiting for you` : o.inboxUnread ? `${o.inboxUnread} new from your agents` : 'What your agents told you on their own',
      count: inboxCount, hot: inboxCount > 0,
    },
    {
      id: 'tasks', group: 'you', label: 'Tasks', icon: ListChecks, onSelect: o.onTasks,
      hint: o.tasksNeedMe ? `${o.tasksNeedMe} need${o.tasksNeedMe === 1 ? 's' : ''} you` : 'Everything asked and sent, and what needs you',
      count: o.tasksNeedMe, hot: o.tasksNeedMe > 0,
    },
    { id: 'people', group: 'you', label: 'People', icon: Contact, hint: 'The people your agents work with', onSelect: o.onPeople },
    // Work
    { id: 'agents', group: 'work', label: 'Agents', icon: Bot, hint: 'Your agents that are up — pick one to chat', onSelect: o.onAgents },
    { id: 'playbooks', group: 'work', label: 'Playbooks', icon: Zap, hint: 'Ready-made ways to get things done on the web — an agent runs them', onSelect: o.onPlaybooks },
    {
      id: 'automations', group: 'work', label: 'Automations', icon: AlarmClock, onSelect: o.onAutomations,
      hint: o.automationsRunning ? 'One is running now' : 'Jobs your agents do on a schedule',
      count: o.automationsCount, dot: o.automationsRunning,
    },
    { id: 'activity', group: 'work', label: 'Activity', icon: Activity, hint: 'Every tool your agents used, and how it went', onSelect: o.onActivity },
  ]
}
