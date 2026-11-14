import type { TaskItem } from '../../services/taskboardApi'

// ── The Today home's lists, as plain functions ──
//
// Kept free of React and of the services (a type import only), so
// todayLists.test.mjs runs them with plain `node --test`.

/** One source as the home holds it: its last answer and its last error. */
export interface Source {
  data: unknown
  error: string
}

/** Answered or failed: nothing more is coming from it for now. */
export const settled = (s: Source) => s.data !== null || s.error !== ''

/** Every one of *sources* has answered or failed: only then may a section
 *  say it is empty (one still loading may hold what it is waiting for). */
export const allSettled = (...sources: Source[]) => sources.every(settled)

/** Every one of *sources* failed and none has an earlier answer to show. */
export const allFailed = (...sources: Source[]) => sources.every((s) => s.data === null && s.error !== '')

/** Today's priorities: the open work that isn't the person's move (Waiting
 *  for you has that) — what the agents are working on (the Taskboard's
 *  "working" lane: in progress is no status of its own there, but an open
 *  task that is the agent's move) and what waits on someone else (its
 *  "waiting" lane). Each lane is read on its own, so a page of more urgent
 *  needs-me tasks can't push either out. Once each, most urgent first. */
export function priorityTasks(working: TaskItem[], waiting: TaskItem[]): TaskItem[] {
  const seen = new Set<string>()
  const out: TaskItem[] = []
  for (const t of [...working, ...waiting]) {
    if (seen.has(t.id) || t.status !== 'open' || t.lane === 'needs_me' || t.lane === 'snoozed') continue
    seen.add(t.id)
    out.push(t)
  }
  const at = (t: TaskItem) => t.last_event_at ?? t.updated_at
  return out.sort((a, b) => b.urgency - a.urgency || at(b) - at(a))
}
