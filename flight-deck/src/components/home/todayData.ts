import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../services/fileLibraryApi'
import { asksApi, type OpenAsk } from '../../services/asksApi'
import { schedulerApi, type TaskRun } from '../../services/schedulerApi'
import { taskboardApi, type TaskItem, type TaskProposal } from '../../services/taskboardApi'
import { priorityTasks } from './todayLists'

// ── The Today home's data ──
//
// Everything the home reads, from the routes the rest of the deck already
// uses: the Taskboard (what needs you, what is open, what got done), the
// proposals waiting for an Approve (/fd/autonomy/actions, as the bell reads
// them), the open questions (/fd/asks, at once: no long wait) and the
// scheduler's recent runs. Each source loads and fails on its own, so one
// that is down leaves the others showing. Read again every minute and when
// the tab comes back into view: seven light reads (priorities takes two),
// only while the tab is visible — kept on purpose, so what was answered
// elsewhere doesn't linger here. The questions are also waited on (one
// request held open, as PersonAsks holds its own): they change by the second.

/** How far back "today" looks for things done on their own. */
export const HANDLED_WINDOW_S = 24 * 60 * 60

const REFRESH_MS = 60_000

/** One source: its last answer (kept while it reloads) and its last error. */
export interface Loaded<T> {
  data: T | null
  error: string
}

/** A proposal waiting for an Approve, as /fd/autonomy/actions lists it. */
export interface WaitingAction {
  id: string
  source: string
  kind: string
  title: string
  /** The agent it is for (a slug), when it names one. */
  target: string
  ref_id: string
  created_at: string
  payload: TaskProposal['payload'] & { task_id?: string; to_options?: { address?: string; display?: string }[] }
}

export interface TodayData {
  needsMe: Loaded<TaskItem[]>
  open: Loaded<TaskItem[]>
  done: Loaded<TaskItem[]>
  proposals: Loaded<WaitingAction[]>
  asks: Loaded<OpenAsk[]>
  runs: Loaded<TaskRun[]>
}

const empty = <T,>(): Loaded<T> => ({ data: null, error: '' })

const INITIAL: TodayData = {
  needsMe: empty(), open: empty(), done: empty(), proposals: empty(), asks: empty(), runs: empty(),
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e)) || 'Failed'

const now = () => Date.now() / 1000

async function waitingActions(): Promise<WaitingAction[]> {
  const res = await api<{ actions?: Partial<WaitingAction>[] }>('/fd/autonomy/actions?status_filter=awaiting_approval&limit=50')
  return (res.actions ?? []).filter((a) => a.id).map((a) => ({
    id: String(a.id),
    source: String(a.source || ''),
    kind: String(a.kind || ''),
    title: String(a.title || ''),
    target: String(a.target || ''),
    ref_id: String(a.ref_id || ''),
    created_at: String(a.created_at || ''),
    payload: a.payload ?? {},
  }))
}

const LOADERS: { [K in keyof TodayData]: () => Promise<NonNullable<TodayData[K]['data']>> } = {
  needsMe: async () => (await taskboardApi.tasks({ view: 'needs_me', sort: 'urgency', limit: 20 })).items,
  // Open and not the person's move, as the Taskboard's lanes have it: what
  // the agents are working on (in progress: the "working" lane — the board
  // has no in-progress status) and what waits on someone else.
  open: async () => {
    const [working, waiting] = await Promise.all([
      taskboardApi.tasks({ view: 'working', sort: 'urgency', limit: 30 }),
      taskboardApi.tasks({ view: 'waiting', sort: 'urgency', limit: 30 }),
    ])
    return priorityTasks(working.items, waiting.items)
  },
  done: async () => {
    const since = now() - HANDLED_WINDOW_S
    return (await taskboardApi.tasks({ view: 'done', sort: 'recent', limit: 20 })).items
      .filter((t) => (t.closed_at ?? t.updated_at) >= since)
  },
  proposals: waitingActions,
  asks: async () => (await asksApi.list([], 0)).asks,
  runs: async () => {
    const since = now() - HANDLED_WINDOW_S
    return (await schedulerApi.runs({ limit: 20 }))
      .filter((r) => r.status !== 'running' && (r.finished_at ?? r.started_at) >= since)
  },
}

export function useToday(): { data: TodayData; reload: () => Promise<void> } {
  const [data, setData] = useState<TodayData>(INITIAL)
  const alive = useRef(true)

  const reload = useCallback(async () => {
    const keys = Object.keys(LOADERS) as (keyof TodayData)[]
    await Promise.all(keys.map(async (key) => {
      try {
        const value = await LOADERS[key]()
        if (alive.current) setData((d) => ({ ...d, [key]: { data: value, error: '' } }))
      } catch (e) {
        if (alive.current) setData((d) => ({ ...d, [key]: { data: d[key].data, error: message(e) } }))
      }
    }))
  }, [])

  // The questions are answered in their own window (PersonAsks), not here:
  // wait on /fd/asks the way it does, so one answered there goes at once.
  useEffect(() => {
    let watching = true
    let known: string[] = []
    const loop = async () => {
      while (watching) {
        try {
          const asks = (await asksApi.list(known, 25)).asks
          if (!watching) return
          known = asks.map((a) => a.id)
          setData((d) => ({ ...d, asks: { data: asks, error: '' } }))
        } catch {
          await new Promise((res) => window.setTimeout(res, 5000))
        }
      }
    }
    void loop()
    return () => { watching = false }
  }, [])

  useEffect(() => {
    alive.current = true
    void reload()
    const timer = window.setInterval(() => { if (document.visibilityState === 'visible') void reload() }, REFRESH_MS)
    const onShow = () => { if (document.visibilityState === 'visible') void reload() }
    document.addEventListener('visibilitychange', onShow)
    return () => {
      alive.current = false
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onShow)
    }
  }, [reload])

  return { data, reload }
}

/** "just now", "5 min ago", "3 h ago", "yesterday", or a date. */
export function ago(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return ''
  const s = Math.max(0, now() - epochSeconds)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  if (s < 2 * 86400) return 'yesterday'
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

/** An ISO time from the autonomy routes (UTC, maybe without its zone) in epoch seconds. */
export function isoSeconds(iso: string): number | null {
  if (!iso) return null
  const ms = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`)
  return Number.isFinite(ms) ? ms / 1000 : null
}

const WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

/** 2 → "Two", 12 → "12": small counts read as words. */
export function countWord(n: number): string {
  return WORDS[n] ?? String(n)
}
