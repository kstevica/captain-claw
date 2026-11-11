import { create } from 'zustand'
import { agentLessonsApi, type LessonMark } from '../services/agentLessons'
import { useLessonInbox } from './lessonInboxStore'

/** Which chat messages are lessons already — so "Save as a lesson?" isn't offered twice. Per chat agent
 *  (``proc-<slug>`` for the owner's own, ``shared:<ref>`` for someone's shared one), by the message's saidKey:
 *  taught, waiting in Lessons to approve (or for its owner), or dismissed there. Read once when a chat first shows a
 *  message that could be one, marked at once when one is saved, and read again when lessons or Lessons to approve
 *  change. */
interface LessonMarksState {
  marks: Record<string, Record<string, LessonMark>>
  /** Read *agentId*'s marks, unless they were read already. */
  load: (agentId: string) => void
  /** Read them again — *agentId*'s, or every agent's read so far. */
  refresh: (agentId?: string) => void
  /** A message was just saved (or suggested): mark it before Flight Deck is asked again. */
  mark: (agentId: string, key: string, mark: LessonMark) => void
}

const inFlight = new Set<string>()

function read(agentId: string): Promise<{ marks: Record<string, LessonMark> }> | null {
  if (agentId.startsWith('proc-')) return agentLessonsApi.fromChat(agentId.slice('proc-'.length))
  if (agentId.startsWith('shared:')) return agentLessonsApi.suggested(agentId.slice('shared:'.length))
  return null
}

export const useLessonMarks = create<LessonMarksState>((set, get) => {
  const fetchFor = (agentId: string) => {
    if (inFlight.has(agentId)) return
    const req = read(agentId)
    if (!req) return
    inFlight.add(agentId)
    req
      .then(({ marks }) => set((s) => ({ marks: { ...s.marks, [agentId]: marks } })))
      // An older deck without the route, or offline: offer as before (no marks), and don't ask again until a refresh.
      .catch(() => set((s) => (s.marks[agentId] ? s : { marks: { ...s.marks, [agentId]: {} } })))
      .finally(() => inFlight.delete(agentId))
  }
  return {
    marks: {},
    load: (agentId) => { if (!get().marks[agentId]) fetchFor(agentId) },
    refresh: (agentId) => {
      for (const id of agentId ? [agentId] : Object.keys(get().marks)) if (get().marks[id]) fetchFor(id)
    },
    mark: (agentId, key, mark) => {
      if (!key) return
      set((s) => ({ marks: { ...s.marks, [agentId]: { ...s.marks[agentId], [key]: mark } } }))
    },
  }
})

// Lessons to approve changed — something approved, dismissed, undone, or newly noticed: the owner's chats read their
// marks again. (Its data is read every minute; only a change in what waits or was learned counts.)
let seen = ''
useLessonInbox.subscribe((s) => {
  const now = JSON.stringify((s.data?.agents ?? []).map((a) => [a.slug, a.waiting.map((p) => p.id), a.learned.map((p) => p.id)]))
  if (now === seen) return
  const first = !seen
  seen = now
  if (first) return
  const marks = useLessonMarks.getState().marks
  for (const id of Object.keys(marks)) if (id.startsWith('proc-')) useLessonMarks.getState().refresh(id)
})
