import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlarmClock, AlertTriangle, ArrowLeft, ArrowRight, Ban, BellOff, Check, Clock, Hand, HelpCircle, Layers, ListChecks,
  Loader2, Lock, MessageSquare, Pencil, RotateCcw, RotateCw, Search, Settings2, SlidersHorizontal, Undo2, Wrench, X,
} from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { useTaskboardStore } from '../../stores/taskboardStore'
import { useGmailAutoReplyStore } from '../../stores/gmailAutoReplyStore'
import { HeldCard } from '../connections/GmailAutoReplySection'
import type { TabId as ConnectionsTab } from '../../pages/ConnectionsPage'
import { fmtWhen } from '../../services/schedulerApi'
import {
  taskboardApi,
  type TaskAction,
  type TaskDetail,
  type TaskEvent,
  type TaskFacetKey,
  type TaskItem,
  type TaskPage,
  type TaskParty,
  type TaskQuery,
  type TaskSort,
  type TaskView,
  type TaskboardStatus,
  type TaskboardSync,
} from '../../services/taskboardApi'
import { FullscreenButton, libraryIconBtn } from '../library/libraryParts'
import {
  AnswerButtons, ChannelIcon, DateFacet, EvidenceBadge, FoldHighlight, InstructionsChip, LaneChip, LowConfidenceHint,
  MultiFacet, NoteChip, OwesChip, ProposalButtons, ResultChip, StatusChip, SyncStatus,
} from './taskboardParts'
import { BoardSettings } from './TaskboardSettings'
import { WhyThisHere } from './TaskWhy'
import {
  DIRECTION_LABEL, EVENT_LABEL, EVIDENCE_LABEL, KIND_LABEL, NO_DATES, OWES_LABEL, RESULT_LABEL, ROLE_LABEL, STATUS_DOT,
  STATUS_LABEL, agoRelative, answeredInChat, channelLabel, dateBounds, distinctParties, foldText, isMe, isSnoozed,
  partyAddress, partyFacetLabel, partyName, taskNotes, type BoardUser, type DateRange,
} from './taskboardText'

// ── Taskboard ───────────────────────────────────────────────────────
//
// Every task your agents were given or sent, on any channel, in one place:
// whether it was finished, by whom, for whom, with which people and with
// what result. It opens on what needs you — a next step to approve, a reply
// held for your OK, something that failed or looks finished — and the rest
// is a tab away. Flight Deck builds it from the records it keeps anyway
// (scheduled runs, proposals, follow-ups, held replies, sent mail, agent
// links) and from what you ask your agents in any chat, which Jev sorts
// ("Why is this here?" says how); opening the board reads them again. What
// you change here (Done, Snooze, Rename…) holds: nothing automatic changes
// it back (but a scheduled task's next run opens it again).

const PAGE = 60

const VIEWS: { id: TaskView; label: string }[] = [
  { id: 'needs_me', label: 'Needs me' },
  { id: 'waiting', label: 'Waiting on others' },
  { id: 'working', label: 'Agents working' },
  { id: 'done', label: 'Done' },
  { id: 'all', label: 'All' },
]

const SORTS: { id: TaskSort; label: string }[] = [
  { id: 'recent', label: 'Latest activity' },
  { id: 'oldest', label: 'Oldest first' },
  { id: 'due', label: 'Due soonest' },
  { id: 'urgency', label: 'Most urgent' },
]

type Filters = Record<TaskFacetKey, string[]>
const NO_FILTERS: Filters = { status: [], channel: [], agent: [], direction: [], kind: [], evidence: [], result: [], party: [] }

const FACETS: { key: TaskFacetKey; title: string; labelOf?: (id: string) => string; dot?: (id: string) => ReactNode }[] = [
  // Where a task the board isn't sure of waits ("Why is this here?" says so).
  { key: 'status', title: 'Status', labelOf: (id) => (id === 'maybe' ? 'Maybe tasks' : STATUS_LABEL[id] || id),
    dot: (id) => <span className={`h-2 w-2 shrink-0 rounded-full ${STATUS_DOT[id] || 'bg-zinc-500'}`} /> },
  { key: 'channel', title: 'Channel', labelOf: channelLabel, dot: (id) => <ChannelIcon channel={id} className="h-3.5 w-3.5" /> },
  { key: 'agent', title: 'Agent' },
  { key: 'party', title: 'People' },
  { key: 'kind', title: 'Kind', labelOf: (id) => KIND_LABEL[id] || id },
  { key: 'direction', title: 'Direction', labelOf: (id) => DIRECTION_LABEL[id] || id },
  { key: 'evidence', title: 'Evidence', labelOf: (id) => EVIDENCE_LABEL[id] || id },
  { key: 'result', title: 'Result', labelOf: (id) => RESULT_LABEL[id] || id },
]

const EMPTY: Record<TaskView, { title: string; text: string }> = {
  needs_me: {
    title: 'Nothing needs you',
    text: 'When an agent waits for your OK on a next step, a reply is held for you, something failed or looks finished, it shows up here.',
  },
  waiting: { title: 'Nothing waiting on others', text: 'Tasks where someone else owes the next move — a reply, a follow-up — show up here.' },
  working: { title: 'No agent is on anything', text: 'Tasks an agent is still busy with — a scheduled run, an approved step — show up here.' },
  done: { title: 'Nothing done yet', text: 'Finished tasks collect here, with who did them and how it went.' },
  all: {
    title: 'No tasks yet',
    text: 'What you ask your agents in any chat, scheduled tasks, proposals, follow-ups, held replies, sent mail and messages from linked systems collect here.',
  },
}

const CLOSED = new Set(['done', 'dismissed', 'not_task'])
const OWES = ['me', 'agent', 'them', 'nobody', 'unclear'] as const

const SNOOZES = [
  { id: '1h', label: '1 hour' },
  { id: 'tomorrow', label: 'Tomorrow 9:00' },
  { id: 'week', label: 'Next week' },
] as const

/** When a snooze ends: in an hour, tomorrow at 9, or next Monday at 9. */
function snoozeUntil(which: (typeof SNOOZES)[number]['id']): number {
  const now = new Date()
  if (which === '1h') return Math.floor(now.getTime() / 1000) + 3600
  const d = new Date(now)
  d.setHours(9, 0, 0, 0)
  d.setDate(d.getDate() + (which === 'tomorrow' ? 1 : (8 - d.getDay()) % 7 || 7))
  return Math.floor(d.getTime() / 1000)
}

const iconBtn = libraryIconBtn
const sectionTitle = 'text-[11px] font-medium uppercase tracking-wider text-zinc-500'
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))
const sameList = (a: string[], b: string[]) => a.length === b.length && a.every((v, i) => v === b[i])

/** Why a task in Needs me is there, in a few words — and, when the agent
 *  couldn't do it (or did only part), why not (`more`, '' when unsaid). */
function needsReason(t: TaskItem): { text: string; more: string } | null {
  const say = (text: string, more = '') => ({ text, more })
  if (t.proposal?.status === 'awaiting_approval') return say('Approve the next step')
  // A held reply's own task, or one held on the thread a request started.
  if (t.status === 'open' && (t.kind === 'held_reply' || (t.owes_next === 'me' && t.links?.autoreply_id))) {
    return say('A reply waits for your OK')
  }
  if (t.status === 'failed') return say('It failed — take a look')
  const note = taskNotes(t).find((n) => n.id !== 'new_since_done')
  if (note) return say(note.label, note.detail)
  if (t.flags?.probably_done) return say('Probably done?')
  if (t.flags?.overdue) return say('Overdue')
  if (t.owes_next === 'me') return say('Your move')
  return t.due_at && t.due_at <= Date.now() / 1000 ? say('Due now') : null
}

/** The people on a task other than whoever did it, as one line ("You" for
 *  you; a name read out of the text that repeats one already there, once). */
function partyLine(parties: TaskParty[] | undefined, me?: BoardUser): string {
  const names: string[] = []
  for (const p of distinctParties(parties, me)) {
    const n = p.role === 'doer' ? '' : partyName(p, me)
    if (n && !names.includes(n)) names.push(n)
  }
  return names.length > 2 ? `${names.slice(0, 2).join(', ')} +${names.length - 2}` : names.join(', ')
}

/** Who is looking: the signed-in user — or, sign-in off, whoever the
 *  server marks as you on the tasks shown (the People filter's values carry
 *  the server's mark of their own, whatever the page holds). */
function useBoardUser(items: TaskItem[]): BoardUser {
  const id = useAuthStore((s) => s.user?.id ?? '')
  const name = useAuthStore((s) => s.user?.display_name ?? '')
  const marked = useMemo(() => {
    for (const t of items) for (const p of t.parties ?? []) if (p.me && p.address) return p.address
    return ''
  }, [items])
  return useMemo(() => ({ id: id || marked, name }), [id, marked, name])
}

function closedByText(t: TaskItem): string {
  if (t.closed_by === 'user') return 'by you'
  if (t.closed_by_party) return `by ${t.closed_by_party}`
  if (t.closed_by === 'jev') return 'by Jev'
  if (t.closed_by === 'evidence') return 'a send was recorded'
  if (t.closed_by === 'rule') return 'by a rule'
  return ''
}

export function TaskBoard({ phone = false, initialAgent, initialTask, onClose, onOpenScheduled, onOpenChat, onOpenConnections }: {
  phone?: boolean
  /** Open filtered to this agent's tasks. */
  initialAgent?: string | null
  /** Open showing this task (its id). */
  initialTask?: string | null
  onClose: () => void
  /** A scheduled task in Scheduled tasks, by its job id. */
  onOpenScheduled?: (jobId: string) => void
  /** The task's own conversation in its agent's chat. */
  onOpenChat?: (agentSlug: string, sessionId: string) => void
  /** Connections on a tab: Google, where a held reply the board can't show
   *  waits; Jev, from the board settings when it isn't connected. */
  onOpenConnections?: (tab: ConnectionsTab) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const fetchSummary = useTaskboardStore((s) => s.fetchSummary)
  const [status, setStatus] = useState<TaskboardStatus | null>(null)
  const [syncing, setSyncing] = useState(false)
  const [view, setView] = useState<TaskView>('needs_me')
  const [query, setQuery] = useState('')
  const [debounced, setDebounced] = useState('')
  const [filters, setFilters] = useState<Filters>(() => (initialAgent ? { ...NO_FILTERS, agent: [initialAgent] } : NO_FILTERS))
  const [dates, setDates] = useState<DateRange>(NO_DATES)
  const [sort, setSort] = useState<TaskSort>('recent')
  const [page, setPage] = useState<TaskPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<TaskItem | null>(null)
  // Bumped when a source moved the open task (a proposal answered): its detail reads again.
  const [rev, setRev] = useState(0)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  // The gear and its popover (desktop): a click outside both closes it.
  const settingsBox = useRef<HTMLSpanElement>(null)
  const reqId = useRef(0)
  const shown = useRef(PAGE)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  // The kiosk can point the open board at another agent: follow it, and let
  // go of a task that belongs to a different one.
  useEffect(() => {
    const next = initialAgent ? [initialAgent] : []
    setFilters((cur) => (sameList(cur.agent, next) ? cur : { ...cur, agent: next }))
    setSelected((cur) => (cur && initialAgent && cur.agent_slug !== initialAgent ? null : cur))
  }, [initialAgent])

  const bounds = useMemo(() => dateBounds(dates), [dates])
  const params = useMemo<TaskQuery>(
    () => ({ view, q: debounced, ...filters, from: bounds.from, to: bounds.to, sort }),
    [view, debounced, filters, bounds, sort],
  )

  const load = useCallback(async (quiet = false) => {
    const id = ++reqId.current
    if (!quiet) { setLoading(true); shown.current = PAGE }
    try {
      const res = await taskboardApi.tasks({ ...params, limit: Math.max(PAGE, shown.current), offset: 0 })
      if (id !== reqId.current) return
      setPage(res)
      setError('')
      // The open task follows its fresh copy (a source moved it on).
      setSelected((cur) => (cur ? res.items.find((i) => i.id === cur.id) ?? cur : cur))
    } catch (e) {
      if (id === reqId.current && !quiet) setError(errText(e))
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [params])

  useEffect(() => { load() }, [load])

  const loadMore = async () => {
    if (!page) return
    // Another tab, search or filter since the click: this page isn't its.
    const id = reqId.current
    setLoadingMore(true)
    try {
      const res = await taskboardApi.tasks({ ...params, limit: PAGE, offset: page.items.length })
      if (id !== reqId.current) return
      setPage((cur) => cur && {
        ...res,
        items: [...cur.items, ...res.items.filter((i) => !cur.items.some((c) => c.id === i.id))],
      })
      shown.current = page.items.length + res.items.length
    } catch (e) {
      notify('error', 'Could not load more tasks', errText(e))
    } finally {
      setLoadingMore(false)
    }
  }

  const loadRef = useRef(load)
  loadRef.current = load

  // A sync answers with the counts and the sweep, shown at once; the
  // sources come with the whole status.
  const applyStatus = useCallback((s: TaskboardSync | null | undefined) => {
    const sweep = s?.sweep
    const counts = s?.counts
    if (sweep && counts) setStatus((cur) => cur && { ...cur, sweep, counts })
    taskboardApi.status().then(setStatus).catch(() => {})
  }, [])

  // Opening the board reads every source again (no agent is woken for it).
  const syncNow = useCallback(async (loud = false) => {
    setSyncing(true)
    try {
      applyStatus(await taskboardApi.sync())
      loadRef.current(true)
      fetchSummary()
    } catch (e) {
      if (loud) notify('error', 'Could not check for new tasks', errText(e))
      taskboardApi.status().then(setStatus).catch(() => {})
    } finally {
      setSyncing(false)
    }
  }, [applyStatus, fetchSummary, notify])

  useEffect(() => { void syncNow() }, [syncNow])

  // While a sweep runs: its state often, the list as tasks land…
  const running = !!status?.sweep?.running
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => {
      taskboardApi.status().then(setStatus).catch(() => {})
      loadRef.current(true)
    }, 2500)
    return () => clearInterval(t)
  }, [running])
  // …and once more when it finishes.
  const wasRunning = useRef(false)
  useEffect(() => {
    if (wasRunning.current && !running) { loadRef.current(true); fetchSummary() }
    wasRunning.current = running
  }, [running, fetchSummary])

  // Opened on one task: shown at once, on the tab it is in.
  useEffect(() => {
    if (!initialTask) return
    let gone = false
    taskboardApi.task(initialTask)
      .then((d) => {
        if (gone) return
        setSelected(d.task)
        setView(VIEWS.some((v) => v.id === d.task.lane) ? (d.task.lane as TaskView) : 'all')
      })
      .catch((e) => { if (!gone) notify('warning', 'Task not found', errText(e)) })
    return () => { gone = true }
  }, [initialTask, notify])

  // A change shows at once; the list, its counts and the badge catch up.
  const refreshAfter = () => {
    load(true)
    taskboardApi.status().then(setStatus).catch(() => {})
    fetchSummary()
  }
  const act = async (task: TaskItem, action: TaskAction, value?: string | number) => {
    const { task: fresh } = await taskboardApi.action(task.id, action, value)
    setSelected((cur) => (cur?.id === fresh.id ? fresh : cur))
    setPage((cur) => cur && { ...cur, items: cur.items.map((i) => (i.id === fresh.id ? fresh : i)) })
    refreshAfter()
  }
  // A proposal or held reply was answered where it lives: the board reads
  // that source again, so the task moves on now rather than at the next sweep.
  const answered = async () => {
    try { applyStatus(await taskboardApi.sync()) } catch { /* the next sweep catches up */ }
    setRev((n) => n + 1)
    load(true)
    fetchSummary()
  }

  // Escape: the settings close first, then the Sources sheet, then the
  // filters (phone), then the open task (phone), then the board. Not while
  // writing (a held reply's text), nor when something inside used it to
  // close itself (the Snooze picker, "Why is this here?").
  const settingsRef = useRef(settingsOpen)
  settingsRef.current = settingsOpen
  const sourcesRef = useRef(sourcesOpen)
  sourcesRef.current = sourcesOpen
  const filtersRef = useRef(filtersOpen)
  filtersRef.current = filtersOpen
  const selectedRef = useRef(selected)
  selectedRef.current = selected
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      const el = e.target as HTMLElement | null
      if (el instanceof HTMLTextAreaElement || el?.isContentEditable) return
      setTimeout(() => {
        if (e.defaultPrevented) return
        if (settingsRef.current) { setSettingsOpen(false); return }
        if (sourcesRef.current) { setSourcesOpen(false); return }
        if (filtersRef.current) { setFiltersOpen(false); return }
        if (phone && selectedRef.current) { setSelected(null); return }
        onClose()
      }, 0)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, phone])

  // The settings popover closes on a click anywhere else (the phone's sheet
  // covers the board, and closes by its own buttons).
  useEffect(() => {
    if (!settingsOpen || phone) return
    const onDown = (e: MouseEvent) => {
      if (settingsBox.current && !settingsBox.current.contains(e.target as Node)) setSettingsOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [settingsOpen, phone])

  const items = useMemo(() => page?.items ?? [], [page])
  const me = useBoardUser(items)
  const counts = page?.counts ?? status?.counts
  const terms = useMemo(() => foldText(debounced).split(/\s+/).filter(Boolean), [debounced])
  // "Between…" with no dates picked yet filters nothing.
  const datesOn = bounds.from !== undefined || bounds.to !== undefined
  const filterCount = Object.values(filters).reduce((n, v) => n + v.length, 0) + (datesOn ? 1 : 0)
  const filtered = !!debounced || filterCount > 0
  // Only one agent's tasks (how an agent's menu opens it): an empty list
  // means that agent has none yet, not that a search missed.
  const agentOnly = filters.agent.length === 1 && filterCount === 1 && !debounced
  const clearFilters = () => { setFilters(NO_FILTERS); setDates(NO_DATES) }
  // "Connect Jev" in the settings: they close, and Connections opens on Jev.
  const connectJev = onOpenConnections && (() => { setSettingsOpen(false); onOpenConnections('jev') })
  const setFacet = (key: TaskFacetKey, values: string[]) => setFilters((cur) => ({ ...cur, [key]: values }))

  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-500/15 text-teal-600 dark:text-teal-400">
        <ListChecks className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">Taskboard</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Everything your agents were asked or sent — what's done, what waits, and what needs you.</p>}
      </div>
      {!phone && <SyncStatus status={status} syncing={syncing} />}
      <button onClick={() => syncNow(true)} disabled={syncing || running} className={iconBtn} title="Check for new tasks now" aria-label="Check for new tasks now">
        <RotateCw className={`h-4 w-4 ${syncing || running ? 'animate-spin' : ''}`} />
      </button>
      <button onClick={() => setSourcesOpen(true)} className={iconBtn} title="Sources — what the Taskboard reads" aria-label="Sources">
        <Layers className="h-4 w-4" />
      </button>
      <span ref={settingsBox} className="relative">
        <button
          onClick={() => setSettingsOpen((o) => !o)}
          className={iconBtn}
          title="Board settings — closing by itself, daily limits, Jev"
          aria-label="Board settings"
          aria-expanded={settingsOpen}
        >
          <Settings2 className="h-4 w-4" />
        </button>
        {settingsOpen && !phone && (
          <div className="absolute right-0 top-full z-40 mt-2 w-[340px]">
            <BoardSettings phone={false} onClose={() => setSettingsOpen(false)} onConnectJev={connectJev} />
          </div>
        )}
      </span>
      <FullscreenButton />
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const toolbar = (
    <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2.5 wide:px-4">
      <div className="relative min-w-0 flex-1">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={phone ? 'Search tasks…' : 'Search titles, what was asked, results, people and agents…'}
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-2 pl-8 pr-8 text-sm text-zinc-200 placeholder-zinc-500 focus:border-teal-500/60 focus:outline-none"
        />
        {query && (
          <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200" aria-label="Clear search">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      {phone ? (
        <button
          onClick={() => setFiltersOpen(true)}
          className={`relative flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-sm ${
            filterCount ? 'border-teal-500/50 text-teal-700 dark:text-teal-300' : 'border-zinc-800 text-zinc-400'}`}
        >
          <SlidersHorizontal className="h-4 w-4" />Filters{filterCount ? ` · ${filterCount}` : ''}
        </button>
      ) : (
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as TaskSort)}
          aria-label="Sort"
          className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-2 text-sm text-zinc-300 focus:border-teal-500/60 focus:outline-none"
        >
          {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
      )}
    </div>
  )

  const tabs = (
    <div role="tablist" aria-label="Views" className="flex shrink-0 gap-1 overflow-x-auto border-b border-zinc-800 px-3 py-2 [scrollbar-width:none] wide:px-4">
      {VIEWS.map((v) => {
        const n = counts?.[v.id]
        const on = view === v.id
        return (
          <button
            key={v.id}
            role="tab"
            aria-selected={on}
            onClick={() => setView(v.id)}
            className={`flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors ${
              on ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'}`}
          >
            {v.label}
            {n !== undefined && (
              <span className={`min-w-[18px] rounded-full px-1.5 text-center text-[10px] font-semibold leading-[18px] tabular-nums ${
                v.id === 'needs_me' && n > 0 ? 'bg-amber-500 text-white' : on ? 'bg-zinc-700 text-zinc-200' : 'bg-zinc-800 text-zinc-400'}`}>
                {n}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )

  const filterList = (
    <Filters
      facets={page?.facets}
      failed={!!error && !page}
      filters={filters}
      dates={dates}
      me={me}
      onFacet={setFacet}
      onDates={setDates}
    />
  )

  const list = (
    <ResultList
      view={view}
      items={items}
      total={page?.total ?? 0}
      query={debounced}
      gathering={syncing || running}
      loading={loading}
      error={error}
      filtered={filtered}
      agentOnly={agentOnly}
      terms={terms}
      me={me}
      selectedId={selected?.id ?? null}
      loadingMore={loadingMore}
      onSelect={setSelected}
      onMore={loadMore}
      onClear={() => { setQuery(''); clearFilters() }}
      onRetry={() => load()}
      onAll={() => setView('all')}
      onAnswered={answered}
      onAct={act}
    />
  )

  const detail = selected && (
    <TaskDetailPane
      key={selected.id}
      task={selected}
      rev={rev}
      terms={terms}
      me={me}
      phone={phone}
      onClose={() => setSelected(null)}
      onAct={(action, value) => act(selected, action, value)}
      onAgent={() => setFacet('agent', [selected.agent_slug])}
      onAnswered={answered}
      onOpenScheduled={onOpenScheduled}
      onOpenChat={onOpenChat}
      onOpenConnections={onOpenConnections}
    />
  )

  const sources = sourcesOpen && <SourcesSheet status={status} onClose={() => setSourcesOpen(false)} />

  if (phone) {
    return (
      <div className="relative flex h-full flex-col bg-zinc-950">
        {header}
        <div className="border-b border-zinc-800 px-3 py-2"><SyncStatus status={status} syncing={syncing} /></div>
        {toolbar}
        {tabs}
        <div className="min-h-0 flex-1">{list}</div>
        {selected && <div className="absolute inset-0 z-10 flex flex-col bg-zinc-950">{detail}</div>}
        {filtersOpen && (
          <div className="absolute inset-0 z-20 flex flex-col bg-zinc-950">
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <h3 className="flex-1 text-base font-semibold text-zinc-100">Filters</h3>
              <button onClick={() => setFiltersOpen(false)} className={iconBtn} aria-label="Close filters"><X className="h-5 w-5" /></button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="px-4 pt-3">
                <label className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">Sort</label>
                <select
                  value={sort}
                  onChange={(e) => setSort(e.target.value as TaskSort)}
                  className="h-11 w-full rounded-xl border border-zinc-800 bg-zinc-900 px-3 text-base text-zinc-200"
                >
                  {SORTS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                </select>
              </div>
              {filterList}
            </div>
            <div className="flex gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
              <button onClick={clearFilters} className="h-11 flex-1 rounded-xl border border-zinc-800 text-sm font-medium text-zinc-300">
                Clear
              </button>
              <button onClick={() => setFiltersOpen(false)} className="h-11 flex-1 rounded-xl bg-teal-600 text-sm font-medium text-white">
                Show {page?.total ?? 0} task{page?.total === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        )}
        {settingsOpen && (
          <div className="absolute inset-0 z-20 flex flex-col bg-zinc-950">
            <BoardSettings phone onClose={() => setSettingsOpen(false)} onConnectJev={connectJev} />
          </div>
        )}
        {sources}
      </div>
    )
  }

  return (
    <div className="relative flex h-full flex-col bg-zinc-950">
      {header}
      {toolbar}
      <div className="flex min-h-0 flex-1">
        <aside className="w-56 shrink-0 overflow-y-auto border-r border-zinc-800 bg-zinc-900/40">{filterList}</aside>
        <section className="flex min-w-0 flex-1 flex-col">
          {tabs}
          <div className="min-h-0 flex-1">{list}</div>
        </section>
        {selected && <aside className="w-[380px] shrink-0 border-l border-zinc-800 bg-zinc-900/30">{detail}</aside>}
      </div>
      {sources}
    </div>
  )
}

// ── Filters ──

function Filters({ facets, failed, filters, dates, me, onFacet, onDates }: {
  facets: TaskPage['facets'] | undefined
  /** The tasks didn't load: there is nothing to filter yet. */
  failed: boolean
  filters: Filters
  dates: DateRange
  /** People names you "You", and never by a deck user's id. */
  me: BoardUser
  onFacet: (key: TaskFacetKey, values: string[]) => void
  onDates: (v: DateRange) => void
}) {
  if (!facets && failed) return <p className="px-4 py-8 text-center text-xs text-zinc-500">No filters until the tasks load.</p>
  if (!facets) return <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
  return (
    <div className="flex flex-col gap-4 px-2 py-3">
      {FACETS.map(({ key, title, labelOf, dot }) => {
        const list = facets[key] ?? []
        // One value is nothing to pick between — unless it is ticked.
        if (list.length < 2 && !filters[key].length) return null
        return (
          <MultiFacet
            key={key} title={title} facets={list} values={filters[key]}
            onChange={(v) => onFacet(key, v)} dot={dot}
            labelOf={key === 'party' ? (id, label, f) => partyFacetLabel(id, label, me, f.me) : labelOf}
          />
        )
      })}
      <DateFacet value={dates} onChange={onDates} />
    </div>
  )
}

// ── Results ──

function ResultList({
  view, items, total, query, gathering, loading, error, filtered, agentOnly, terms, me, selectedId, loadingMore,
  onSelect, onMore, onClear, onRetry, onAll, onAnswered, onAct,
}: {
  view: TaskView; items: TaskItem[]; total: number; query: string
  /** The sources are being read: more may come. */
  gathering: boolean
  loading: boolean; error: string; filtered: boolean; agentOnly: boolean; terms: string[]; me: BoardUser
  selectedId: string | null; loadingMore: boolean
  onSelect: (t: TaskItem) => void; onMore: () => void; onClear: () => void; onRetry: () => void
  onAll: () => void
  onAnswered: () => Promise<void>
  /** A quick answer on a row (Is this a task? Probably done?). */
  onAct: (task: TaskItem, action: TaskAction) => Promise<void>
}) {
  if (loading && items.length === 0) {
    return <div className="flex h-full justify-center pt-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  }
  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-zinc-500">
        <AlertTriangle className="h-5 w-5 text-red-500" />{error}
        <button onClick={onRetry} className="text-sm font-medium text-teal-600 hover:underline dark:text-teal-400">Try again</button>
      </div>
    )
  }
  if (items.length === 0) {
    const empty = EMPTY[view]
    const missed = filtered && !agentOnly
    return (
      <div className="flex h-full flex-col items-center justify-center px-6 text-center">
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-500/15 text-teal-600 dark:text-teal-400">
          {gathering ? <Loader2 className="h-7 w-7 animate-spin" /> : <ListChecks className="h-7 w-7" />}
        </span>
        <h3 className="mt-4 text-base font-semibold text-zinc-100">
          {missed ? 'Nothing matches' : gathering ? 'Gathering your tasks…' : agentOnly ? `${empty.title} for this agent` : empty.title}
        </h3>
        <p className="mt-1.5 max-w-sm text-sm leading-relaxed text-zinc-400">
          {missed ? 'Try other words, or clear the filters.' : gathering ? 'Tasks show up here as soon as they\'re found.' : empty.text}
        </p>
        {filtered && (
          <button onClick={onClear} className="mt-4 text-sm font-medium text-teal-600 hover:underline dark:text-teal-400">
            {agentOnly ? 'Show every agent\'s tasks' : 'Clear search and filters'}
          </button>
        )}
        {!filtered && view !== 'all' && (
          <button onClick={onAll} className="mt-4 text-sm font-medium text-teal-600 hover:underline dark:text-teal-400">See all tasks</button>
        )}
      </div>
    )
  }
  return (
    <div className="h-full overflow-y-auto px-3 py-3 wide:px-4">
      <div className="mb-2 px-1 text-[11px] text-zinc-500">
        {total} task{total === 1 ? '' : 's'}{query ? <> for “{query}”</> : null}
      </div>
      <ul className="flex flex-col gap-1.5">
        {items.map((t) => (
          <li key={t.id}>
            <TaskRow
              task={t} terms={terms} me={me} active={t.id === selectedId} onClick={() => onSelect(t)} onAnswered={onAnswered}
              onAct={(action) => onAct(t, action)}
            />
          </li>
        ))}
      </ul>
      {items.length < total && (
        <div className="flex justify-center py-4">
          <button
            onClick={onMore}
            disabled={loadingMore}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-800/60 disabled:opacity-60"
          >
            {loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            Show more ({total - items.length} left)
          </button>
        </div>
      )}
    </div>
  )
}

type Answer = { label: string; action: TaskAction; primary?: boolean; title?: string }

// The two questions the board asks you about a task, and their answers.
const MAYBE_ANSWERS: Answer[] = [
  { label: 'Yes', action: 'confirm_task', primary: true, title: 'It is a task — the agent has the next move' },
  { label: 'No', action: 'not_task', title: 'Not a task — it leaves the board' },
]
const DONE_ANSWERS: Answer[] = [
  { label: 'Done', action: 'done', primary: true, title: 'It is finished' },
  { label: 'Not yet', action: 'not_done', title: 'Not finished — it stays open' },
]

/** Jev wasn't sure enough of one: whether it is a task at all, or whether
 *  an open one is finished. */
const isMaybe = (t: TaskItem) => t.status === 'maybe'
const isProbablyDone = (t: TaskItem) => t.status === 'open' && !!t.flags?.probably_done

function TaskRow({ task, terms, me, active, onClick, onAnswered, onAct }: {
  task: TaskItem; terms: string[]; me: BoardUser; active: boolean; onClick: () => void; onAnswered: () => Promise<void>
  onAct: (action: TaskAction) => Promise<void>
}) {
  const waiting = task.proposal?.status === 'awaiting_approval' ? task.proposal : null
  const maybe = !waiting && isMaybe(task)
  // Its question is the reason line: "Probably done?", with the answers.
  const probablyDone = !waiting && !maybe && isProbablyDone(task)
  const reason = task.lane === 'needs_me' && !probablyDone ? needsReason(task) : null
  // Done, and someone wrote again since (Done and All show it).
  const since = taskNotes(task).find((n) => n.id === 'new_since_done')
  const flags = task.flags ?? {}
  const people = partyLine(task.parties, me)
  const agentLabel = task.agent_name || task.agent_slug
  return (
    <div className={`overflow-hidden rounded-xl border transition-colors ${
      active ? 'border-teal-500/40 bg-teal-500/5' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-900/80'}`}>
      <button onClick={onClick} className="flex w-full gap-3 px-3 py-2.5 text-left">
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80">
          <ChannelIcon channel={task.channel} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-100">
              <FoldHighlight text={task.title || 'Untitled task'} terms={terms} />
            </span>
            {flags.contains_instructions && <InstructionsChip compact />}
            <EvidenceBadge evidence={task.evidence} />
            {answeredInChat(flags) && <ResultChip result={task.result} answeredInChat />}
            <LaneChip task={task} />
          </span>
          <span className="mt-0.5 block truncate text-[11px] text-zinc-500">
            {agentLabel && <><FoldHighlight text={agentLabel} terms={terms} />{' · '}</>}
            {people ? <><FoldHighlight text={people} terms={terms} />{' · '}</> : null}
            {agoRelative(task.last_event_at ?? task.created_at)}
            {flags.low_confidence && <span className="italic" title="Jev wasn't sure this is a task, so it isn't counted on the Taskboard button"> · low confidence</span>}
          </span>
          {task.ask && (
            <span className={`mt-1 block text-xs leading-relaxed text-zinc-400 ${task.result_text ? 'line-clamp-1' : 'line-clamp-2'}`}>
              <FoldHighlight text={task.ask} terms={terms} />
            </span>
          )}
          {task.result_text && (
            <span className={`mt-0.5 block text-xs leading-relaxed text-zinc-500 ${task.ask ? 'line-clamp-1' : 'line-clamp-2'}`}>
              <ArrowRight className="mr-1 inline h-3 w-3 align-[-2px]" />
              <FoldHighlight text={task.result_text} terms={terms} />
            </span>
          )}
          {reason && !waiting && (
            <span className="mt-1 flex min-w-0 items-center gap-1 text-[11px] font-medium text-amber-700 dark:text-amber-400">
              <Hand className="h-3 w-3 shrink-0" />
              <span className="min-w-0 truncate">
                {reason.text}
                {reason.more && <span className="font-normal text-zinc-400"> — <FoldHighlight text={reason.more} terms={terms} /></span>}
              </span>
            </span>
          )}
          {since && (
            <span className="mt-1 flex items-center gap-1 text-[11px] font-medium text-sky-700 dark:text-sky-400">
              <MessageSquare className="h-3 w-3 shrink-0" />{since.label}
            </span>
          )}
        </span>
      </button>
      {waiting && (
        <div className="flex flex-wrap items-center gap-2 border-t border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <span className="min-w-0 flex-1 text-xs font-medium text-amber-800 dark:text-amber-300">
            Needs you: approve next step
            {waiting.title && <span className="font-normal text-zinc-400"> — {waiting.title}</span>}
          </span>
          <ProposalButtons id={waiting.id} onAnswered={onAnswered} />
        </div>
      )}
      {(maybe || probablyDone) && (
        <div className="flex flex-wrap items-center gap-2 border-t border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <span className="min-w-0 flex-1 text-xs font-medium text-amber-800 dark:text-amber-300">
            {maybe ? 'Is this a task?' : 'Probably done?'}
            {maybe && flags.why && <span className="font-normal text-zinc-400"> — {flags.why}</span>}
          </span>
          <AnswerButtons choices={maybe ? MAYBE_ANSWERS : DONE_ANSWERS} onAct={onAct} />
        </div>
      )}
    </div>
  )
}

// ── One task ──

function TaskDetailPane({
  task, rev, terms, me, phone, onClose, onAct, onAgent, onAnswered, onOpenScheduled, onOpenChat, onOpenConnections,
}: {
  task: TaskItem; rev: number; terms: string[]; me: BoardUser; phone: boolean
  onClose: () => void
  onAct: (action: TaskAction, value?: string | number) => Promise<void>
  /** Only this agent's tasks. */
  onAgent: () => void
  /** A proposal or held reply was answered. */
  onAnswered: () => Promise<void>
  onOpenScheduled?: (jobId: string) => void
  onOpenChat?: (agentSlug: string, sessionId: string) => void
  onOpenConnections?: (tab: ConnectionsTab) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [detail, setDetail] = useState<TaskDetail | null>(null)
  const [detailError, setDetailError] = useState('')
  const [busy, setBusy] = useState<TaskAction | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [name, setName] = useState(task.title)
  const [snoozeOpen, setSnoozeOpen] = useState(false)

  // Its timeline, people and links — read again whenever it moves.
  useEffect(() => {
    let gone = false
    taskboardApi.task(task.id)
      .then((d) => { if (!gone) { setDetail(d); setDetailError('') } })
      .catch((e) => { if (!gone) setDetailError(errText(e)) })
    return () => { gone = true }
  }, [task.id, task.updated_at, task.last_event_at, rev])

  // Escape closes the Snooze picker, not the board (it marks the key used).
  useEffect(() => {
    if (!snoozeOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); setSnoozeOpen(false) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [snoozeOpen])

  const run = async (action: TaskAction, value?: string | number) => {
    setBusy(action)
    try {
      await onAct(action, value)
      setSnoozeOpen(false)
      setRenaming(false)
    } catch (e) {
      notify('error', 'Could not change the task', errText(e))
    } finally {
      setBusy(null)
    }
  }
  const startRename = () => { setName(task.title); setRenaming(true) }
  const saveName = () => {
    const v = name.trim()
    if (!v || v === task.title) { setRenaming(false); return }
    void run('rename', v.slice(0, 160))
  }

  const flags = task.flags ?? {}
  const locks = Object.keys(task.user_locks ?? {})
  const closed = CLOSED.has(task.status)
  const failed = task.status === 'failed'
  const snoozed = isSnoozed(task.snooze_until)
  const waiting = task.proposal?.status === 'awaiting_approval'
    ? task.proposal
    : detail?.proposals?.find((p) => p.status === 'awaiting_approval') ?? null
  // A held auto-reply waits for your OK: a held reply's own task, or one on
  // the thread a request of yours started (the auto-reply's latest state says).
  const lastReply = [...(detail?.events ?? [])].reverse()
    .find((e) => e.source === 'gmail_autoreplies' && typeof e.refs?.state === 'string')
  const held = task.status === 'open' && task.owes_next === 'me'
    && (task.kind === 'held_reply' || lastReply?.refs.state === 'held')
  const parties = distinctParties(detail?.parties ?? task.parties, me)
  const notes = taskNotes(task)
  // The agent couldn't do it, or did only part: what stopped it.
  const shortOf = notes.find((n) => n.id !== 'new_since_done' && n.detail)
  const jobId = task.links?.scheduled_job_id || ''
  const sessionId = task.links?.session_id || ''
  const sessionAgent = task.links?.session_agent || task.agent_slug
  const canScheduled = !!(onOpenScheduled && jobId)
  const canChat = !!(onOpenChat && sessionId && sessionAgent)
  // Undo takes back your last change (or Jev closing it).
  const canUndo = locks.length > 0 || task.closed_by === 'jev'
  // A scheduled task's Done or Reopen holds until its next run.
  const untilNextRun = task.kind === 'scheduled' && locks.includes('status') && (task.status === 'done' || task.status === 'open')
  const channels = task.channels?.length ? task.channels : [task.channel]
  const by = closedByText(task)

  const meta: [string, ReactNode][] = []
  // A task no agent handled (a proposal, a follow-up) has no Agent row.
  if (task.agent_name || task.agent_slug) meta.push(['Agent', (
    <button onClick={onAgent} className="truncate text-left text-teal-600 hover:underline dark:text-teal-400" title="Show only this agent's tasks">
      {task.agent_name || task.agent_slug}
    </button>
  )])
  meta.push(
    ['Channel', channels.map(channelLabel).join(', ')],
    ['Kind', `${KIND_LABEL[task.kind] || task.kind}${DIRECTION_LABEL[task.direction] ? ` · ${DIRECTION_LABEL[task.direction].toLowerCase()}` : ''}`],
    ['Started', fmtWhen(task.created_at)],
    ['Last activity', fmtWhen(task.last_event_at ?? task.created_at)],
  )
  if (task.due_at) meta.push(['Due', fmtWhen(task.due_at)])
  if (snoozed) meta.push(['Snoozed', `until ${fmtWhen(task.snooze_until!)}`])
  if (task.closed_at && closed) meta.push([task.status === 'done' ? 'Done' : 'Closed', `${fmtWhen(task.closed_at)}${by ? ` · ${by}` : ''}`])

  const btn = 'flex items-center justify-center gap-1.5 rounded-lg border border-zinc-800 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-50'
  const primary = 'flex items-center justify-center gap-1.5 rounded-lg bg-teal-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-teal-500 disabled:opacity-60'
  const linkBtn = 'flex items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-xs font-medium text-teal-600 transition-colors hover:bg-teal-500/10 dark:text-teal-400'
  const icon = (a: TaskAction, el: ReactNode) => (busy === a ? <Loader2 className="h-4 w-4 animate-spin" /> : el)

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
        {phone ? (
          <button onClick={onClose} className="flex flex-1 items-center gap-1.5 text-sm text-teal-600 dark:text-teal-400">
            <ArrowLeft className="h-4 w-4" />All tasks
          </button>
        ) : (
          <>
            <span className="flex-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">Task</span>
            <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close task"><X className="h-4 w-4" /></button>
          </>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800/80">
            <ChannelIcon channel={task.channel} className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            {renaming ? (
              <form onSubmit={(e) => { e.preventDefault(); saveName() }} className="flex flex-col gap-1.5">
                <input
                  autoFocus
                  value={name}
                  maxLength={160}
                  onChange={(e) => setName(e.target.value)}
                  // Escape leaves the name as it was, not the board.
                  onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); setRenaming(false) } }}
                  aria-label="Task name"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 focus:border-teal-500/60 focus:outline-none"
                />
                <div className="flex gap-1.5">
                  <button type="submit" disabled={busy === 'rename' || !name.trim()} className="flex items-center gap-1 rounded-md bg-teal-600 px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50">
                    {busy === 'rename' && <Loader2 className="h-3 w-3 animate-spin" />}Save
                  </button>
                  <button type="button" onClick={() => setRenaming(false)} className="rounded-md px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800">Cancel</button>
                </div>
              </form>
            ) : (
              <h3 className="break-words text-base font-semibold leading-snug text-zinc-100">
                <FoldHighlight text={task.title || 'Untitled task'} terms={terms} />
              </h3>
            )}
            <div className="mt-0.5 truncate text-xs text-zinc-500">{KIND_LABEL[task.kind] || task.kind} · {channelLabel(task.channel)}</div>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <LaneChip task={task} />
          {/* Out of Needs me (seen), the lane chip says Failed itself. */}
          {failed && task.lane === 'needs_me' && <StatusChip status={task.status} />}
          {!closed && !failed && <OwesChip owes={task.owes_next} />}
          <EvidenceBadge evidence={task.evidence} />
          {/* "Partly done" once: the note says it when the result would too. */}
          {!(task.result === 'partial' && notes.some((n) => n.id === 'partly_done')) && (
            <ResultChip result={task.result} answeredInChat={answeredInChat(flags)} />
          )}
          {notes.map((n) => <NoteChip key={n.id} note={n} />)}
          {flags.contains_instructions && <InstructionsChip />}
          {flags.overdue && <span className="rounded-md bg-red-500/10 px-1.5 py-0.5 text-[10px] font-medium text-red-700 dark:text-red-300">Overdue</span>}
          {flags.stale && <span className="rounded-md bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:text-amber-300">Gone quiet</span>}
          {snoozed && task.lane !== 'snoozed' && (
            <span className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">Snoozed</span>
          )}
          {locks.length > 0 && (
            <span
              title={`You changed ${locks.join(', ').replace(/_/g, ' ')} — ${untilNextRun
                ? 'its next run opens it again' : 'nothing automatic changes it back'}`}
              className="flex items-center gap-0.5 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400"
            >
              <Lock className="h-2.5 w-2.5" />Changed by you
            </span>
          )}
          {flags.low_confidence && <LowConfidenceHint />}
        </div>
        <WhyThisHere task={task} phone={phone} />

        {waiting && (
          <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
            <div className="text-xs font-medium text-amber-800 dark:text-amber-300">Needs you: approve the next step</div>
            {waiting.title && <p className="mt-1 break-words text-sm text-zinc-300">{waiting.title}</p>}
            <div className="mt-2"><ProposalButtons id={waiting.id} onAnswered={onAnswered} /></div>
          </div>
        )}
        {held && (
          <HeldReply task={task} detail={detail} detailFailed={!!detailError} onDone={onAnswered} onOpenConnections={onOpenConnections} />
        )}
        {failed && task.lane === 'needs_me' && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-sm text-red-700 dark:text-red-300">
            <div className="flex items-center gap-1.5 font-medium"><AlertTriangle className="h-4 w-4" />It failed</div>
            <p className="mt-1 text-xs opacity-80">What happened is below. “Got it” takes it off Needs me; Reopen puts it back to work.</p>
          </div>
        )}
        {isMaybe(task) && !waiting && (
          <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
            <div className="flex items-center gap-1.5 text-xs font-medium text-amber-800 dark:text-amber-300">
              <HelpCircle className="h-3.5 w-3.5" />Is this a task?
            </div>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">
              {flags.why ? `${flags.why} ` : ''}Yes puts it on the board with the agent's move next; No takes it off.
            </p>
            <div className="mt-2"><AnswerButtons choices={MAYBE_ANSWERS} onAct={(a) => onAct(a)} /></div>
          </div>
        )}
        {isProbablyDone(task) && !waiting && (
          <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
            <div className="text-xs font-medium text-amber-800 dark:text-amber-300">Probably done?</div>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">It looks finished. Done closes it; Not yet keeps it open.</p>
            <div className="mt-2"><AnswerButtons choices={DONE_ANSWERS} onAct={(a) => onAct(a)} /></div>
          </div>
        )}

        {task.ask && (
          <section className="mt-4">
            <h4 className={sectionTitle}>What was asked</h4>
            <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-300">
              <FoldHighlight text={task.ask} terms={terms} />
            </p>
          </section>
        )}
        {task.result_text && (
          <section className="mt-4">
            <h4 className={sectionTitle}>Result</h4>
            <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-300">
              <FoldHighlight text={task.result_text} terms={terms} />
            </p>
          </section>
        )}
        {shortOf && (
          <section className="mt-4">
            <h4 className={sectionTitle}>{shortOf.id === 'agent_declined' ? 'Why the agent couldn\'t do it' : 'What\'s still missing'}</h4>
            <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-zinc-300">
              <FoldHighlight text={shortOf.detail} terms={terms} />
            </p>
          </section>
        )}

        {parties.length > 0 && (
          <section className="mt-4">
            <h4 className={sectionTitle}>People</h4>
            <ul className="mt-1.5 flex flex-col gap-1.5">
              {parties.map((p, i) => {
                // You are "You", and a deck user's id is never shown.
                const address = partyAddress(p, me)
                return (
                  <li key={`${p.role}:${p.kind}:${p.address}:${i}`} className="flex min-w-0 items-center gap-2 text-xs">
                    <span className="w-[68px] shrink-0">
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{ROLE_LABEL[p.role] || p.role}</span>
                    </span>
                    <span className={`min-w-0 truncate ${isMe(p, me) ? 'font-medium text-zinc-200' : 'text-zinc-300'}`} title={address || undefined}>
                      <FoldHighlight text={partyName(p, me)} terms={terms} />
                    </span>
                    {address && <span className="min-w-0 truncate text-zinc-500">{address}</span>}
                    {!!p.verified && (
                      <span title="Taken from the message itself, not from what anyone wrote" className="shrink-0">
                        <Check className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                      </span>
                    )}
                  </li>
                )
              })}
            </ul>
          </section>
        )}

        <dl className="mt-5 grid grid-cols-[84px_1fr] gap-x-3 gap-y-2 text-xs">
          {meta.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="min-w-0 text-zinc-300">{v}</dd>
            </div>
          ))}
        </dl>

        {(canScheduled || canChat) && (
          <div className="mt-4 flex flex-wrap gap-2">
            {canScheduled && (
              <button onClick={() => onOpenScheduled!(jobId)} className={linkBtn} title="The task, its schedule and every run, in Scheduled tasks">
                <AlarmClock className="h-3.5 w-3.5" />Open in Scheduled
              </button>
            )}
            {canChat && (
              <button onClick={() => onOpenChat!(sessionAgent, sessionId)} className={linkBtn} title="Open this task's conversation in the agent's chat and keep talking about it">
                <MessageSquare className="h-3.5 w-3.5" />Continue in chat
              </button>
            )}
          </div>
        )}

        <section className="mt-5">
          <h4 className={sectionTitle}>What happened</h4>
          {detail ? <Timeline events={detail.events ?? []} total={detail.events_total} terms={terms} />
            : detailError ? <p className="mt-2 text-xs text-red-600 dark:text-red-400">{detailError}</p>
              : <Loader2 className="mt-3 h-4 w-4 animate-spin text-zinc-500" />}
        </section>
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-zinc-800 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] wide:pb-3">
        {snoozeOpen ? (
          <>
            <div className="col-span-2 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              Snooze until{busy === 'snooze' && <Loader2 className="h-3 w-3 animate-spin" />}
            </div>
            {SNOOZES.map((s) => (
              <button key={s.id} onClick={() => run('snooze', snoozeUntil(s.id))} disabled={!!busy} className={btn}>
                {s.label}
              </button>
            ))}
            <button onClick={() => setSnoozeOpen(false)} disabled={!!busy} className={btn}>Cancel</button>
          </>
        ) : (
          <>
            {closed || failed ? (
              <button onClick={() => run('reopen')} disabled={!!busy} className={primary} title="Back to open, for the agent or you to work on">
                {icon('reopen', <RotateCcw className="h-4 w-4" />)}Reopen
              </button>
            ) : (
              <button
                onClick={() => run('done')}
                disabled={!!busy}
                className={primary}
                title={task.kind === 'scheduled' ? 'Done until its next run, which opens it again' : undefined}
              >
                {icon('done', <Check className="h-4 w-4" />)}Done
              </button>
            )}
            {failed && (
              <button onClick={() => run('done')} disabled={!!busy} className={btn}>
                {icon('done', <Check className="h-4 w-4" />)}Done
              </button>
            )}
            {failed && task.lane === 'needs_me' && (
              <button onClick={() => run('ack_failed')} disabled={!!busy} className={btn} title="Seen — take it off Needs me">
                {icon('ack_failed', <Check className="h-4 w-4" />)}Got it
              </button>
            )}
            {task.status !== 'not_task' && (
              <button onClick={() => run('not_task')} disabled={!!busy} className={btn} title="Nothing anyone had to do — it leaves the board">
                {icon('not_task', <Ban className="h-4 w-4" />)}Not a task
              </button>
            )}
            {/* A failure leaves Needs me by "Got it", not by a snooze or a next move. */}
            {snoozed ? (
              <button onClick={() => run('unsnooze')} disabled={!!busy} className={btn}>
                {icon('unsnooze', <BellOff className="h-4 w-4" />)}Unsnooze
              </button>
            ) : !closed && !failed && (
              <button onClick={() => setSnoozeOpen(true)} disabled={!!busy} className={btn} title="Off Needs me for a while">
                <Clock className="h-4 w-4" />Snooze
              </button>
            )}
            <button onClick={startRename} disabled={!!busy || renaming} className={btn}>
              <Pencil className="h-4 w-4" />Rename
            </button>
            {!closed && !failed && (
              <label className="col-span-2 flex items-center gap-2 rounded-lg border border-zinc-800 px-3 py-1.5 text-sm">
                <span className="shrink-0 text-zinc-500">Next move</span>
                <select
                  value={OWES.includes(task.owes_next) ? task.owes_next : 'unclear'}
                  onChange={(e) => run('owes_next', e.target.value)}
                  disabled={!!busy}
                  aria-label="Who moves next"
                  className="min-w-0 flex-1 bg-zinc-950 py-0.5 text-zinc-200 focus:outline-none disabled:opacity-50"
                >
                  {OWES.map((o) => <option key={o} value={o}>{OWES_LABEL[o]}</option>)}
                </select>
                {busy === 'owes_next' && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-zinc-500" />}
              </label>
            )}
            {canUndo && (
              <button onClick={() => run('undo')} disabled={!!busy} className={`${btn} col-span-2`} title="Take back the last change to this task">
                {icon('undo', <Undo2 className="h-4 w-4" />)}Undo last change
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/** Everything that happened, oldest first: the latest few, the rest of
 *  what the server sent (its latest 500) on a click. */
const TIMELINE_SHOWN = 20

/** The tool a tool-call event ran, from its refs (the register's `tool`). */
function toolName(e: TaskEvent): string {
  const refs = e.refs ?? {}
  for (const k of ['tool', 'tool_name', 'name']) {
    const v = refs[k]
    if (typeof v === 'string' && v) return v
  }
  return ''
}

function Timeline({ events, total, terms }: { events: TaskEvent[]; total?: number; terms: string[] }) {
  const [all, setAll] = useState(false)
  if (!events.length) return <p className="mt-2 text-xs text-zinc-500">Nothing recorded yet.</p>
  const sorted = [...events].sort((a, b) => a.at - b.at || Number(a.id) - Number(b.id))
  const shown = all ? sorted : sorted.slice(-TIMELINE_SHOWN)
  const older = Math.max(0, (total ?? events.length) - events.length)
  return (
    <>
      {shown.length < sorted.length ? (
        <button onClick={() => setAll(true)} className="mt-2 text-[11px] text-teal-600 hover:underline dark:text-teal-400">
          Show {sorted.length - shown.length} earlier
        </button>
      ) : older > 0 && (
        <p className="mt-2 text-[11px] text-zinc-500">{older} earlier event{older === 1 ? '' : 's'} not shown</p>
      )}
      <ol className="ml-2.5 mt-3 flex flex-col gap-3 border-l border-zinc-800 pl-4">
        {shown.map((e) => {
          const actor = e.actor_kind === 'me' ? 'You' : e.actor_kind === 'jev' ? 'Jev' : e.actor
          const tool = e.kind === 'tool_call' ? toolName(e) : ''
          return (
            <li key={e.id} className="relative">
              <span className="absolute -left-[27px] top-0 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-900 ring-1 ring-zinc-800">
                {e.kind === 'tool_call'
                  ? <Wrench className="h-3 w-3 shrink-0 text-zinc-400" />
                  : <ChannelIcon channel={e.channel || 'agent'} className="h-3 w-3" />}
              </span>
              <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11px] text-zinc-500">
                <span className="font-medium text-zinc-200">{EVENT_LABEL[e.kind] || e.kind}</span>
                {tool && <span className="rounded bg-zinc-800 px-1 font-mono text-[10px] text-zinc-300">{tool}</span>}
                <span className="min-w-0 truncate" title={fmtWhen(e.at)}>{[actor, agoRelative(e.at)].filter(Boolean).join(' · ')}</span>
                {e.kind === 'tool_call' && e.outcome === 'ok' && <span className="font-medium text-emerald-700 dark:text-emerald-400">ok</span>}
                {e.evidence === 'recorded' && (
                  <span className="flex items-center gap-0.5 text-emerald-700 dark:text-emerald-400" title="Flight Deck recorded it">
                    <Check className="h-3 w-3" />recorded
                  </span>
                )}
                {e.evidence === 'said' && <span className="italic" title="Only the agent's word">agent says</span>}
                {e.outcome === 'error' && <span className="font-medium text-red-600 dark:text-red-400">failed</span>}
                {e.outcome === 'blocked' && <span className="font-medium text-amber-700 dark:text-amber-400">blocked</span>}
              </div>
              {e.summary && (
                <p className="mt-0.5 whitespace-pre-wrap break-words text-xs leading-relaxed text-zinc-400">
                  <FoldHighlight text={e.summary} terms={terms} />
                </p>
              )}
            </li>
          )
        })}
      </ol>
    </>
  )
}

/** A held auto-reply on this task's thread: read, edit, send or dismiss it
 *  here — the same card as under Connections → Google. Found by the task's
 *  Gmail thread, or by the auto-reply row its events came from. */
function HeldReply({ task, detail, detailFailed, onDone, onOpenConnections }: {
  task: TaskItem; detail: TaskDetail | null
  /** The task's detail didn't load: look by the thread alone. */
  detailFailed: boolean
  onDone: () => Promise<void>
  onOpenConnections?: (tab: ConnectionsTab) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const activity = useGmailAutoReplyStore((s) => s.activity)
  const fetchActivity = useGmailAutoReplyStore((s) => s.fetchActivity)
  const [asked, setAsked] = useState(false)
  useEffect(() => {
    fetchActivity().finally(() => setAsked(true))
  }, [fetchActivity, task.updated_at])
  const row = useMemo(() => {
    if (!detail && !detailFailed) return null
    const keys = (detail?.links ?? []).map((l) => l.key)
    const threads = new Set([
      task.links?.gmail_thread_id || '',
      ...keys.filter((k) => k.startsWith('gmail:thread:')).map((k) => k.slice('gmail:thread:'.length)),
    ].filter(Boolean))
    const refs = [
      task.links?.autoreply_id || '',
      ...keys.filter((k) => k.startsWith('autoreply:')).map((k) => k.slice('autoreply:'.length)),
      ...[...(detail?.events ?? []), task].filter((x) => /autorepl/i.test(x.source || '')).map((x) => x.source_ref),
    ].filter(Boolean)
    return activity.find((r) => r.state === 'held'
      && (threads.has(r.thread_id) || refs.some((ref) => ref === r.id || ref.startsWith(`${r.id}:`)))) ?? null
  }, [activity, detail, detailFailed, task])
  return (
    <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
      <div className="text-xs font-medium text-amber-800 dark:text-amber-300">Needs you: a reply waits for your OK</div>
      {row ? (
        <>
          <div className="mt-0.5 truncate text-[11px] text-zinc-500">To {row.from || 'the sender'}{row.subject ? ` · ${row.subject}` : ''}</div>
          <HeldCard
            row={row}
            gmailUrl=""
            onDone={async (message) => {
              if (message) notify('success', 'Held reply', message)
              await onDone()
            }}
          />
        </>
      ) : !asked || (!detail && !detailFailed) ? (
        <div className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" />Finding the reply…</div>
      ) : (
        <p className="mt-1 text-xs text-zinc-400">
          It waits under Connections → Google → Topic auto-replies.
          {onOpenConnections && (
            <button onClick={() => onOpenConnections('google')} className="ml-1 font-medium text-teal-600 hover:underline dark:text-teal-400">Open it there</button>
          )}
        </p>
      )}
    </div>
  )
}

// ── Sources ──

/** What the board reads, whether it does yet, and what it can't (the
 *  server's `not_covered` entry, said under the list). */
function SourcesSheet({ status, onClose }: { status: TaskboardStatus | null; onClose: () => void }) {
  const sources = (status?.sources ?? []).filter((s) => s.key !== 'not_covered')
  const uncovered = status?.sources?.find((s) => s.key === 'not_covered')
  return (
    <div className="absolute inset-0 z-30 flex items-stretch justify-center bg-black/50 wide:items-center wide:p-6" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Sources"
        onClick={(e) => e.stopPropagation()}
        className="flex w-full flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:max-h-full wide:max-w-lg wide:rounded-xl wide:border wide:border-zinc-800"
      >
        <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-500/15 text-teal-600 dark:text-teal-400">
            <Layers className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">Sources</h3>
            <p className="truncate text-xs text-zinc-500">Where the Taskboard finds its tasks</p>
          </div>
          <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close sources"><X className="h-4 w-4" /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {!status ? (
            <div className="flex justify-center py-8 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>
          ) : sources.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-zinc-500">No sources yet.</p>
          ) : (
            <ul className="divide-y divide-zinc-800/70">
              {sources.map((s) => {
                const state = s.tracked ? (s.on ? 'Tracked' : 'Off') : 'Not tracked yet'
                return (
                  <li key={s.key} className="flex items-start gap-3 px-4 py-3">
                    <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${s.tracked && s.on ? 'bg-emerald-500' : s.tracked ? 'bg-zinc-500' : 'bg-zinc-700'}`} />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-zinc-200">{s.label}</div>
                      {s.note && <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">{s.note}</div>}
                    </div>
                    <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium ${
                      s.tracked && s.on ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : 'bg-zinc-800 text-zinc-400'}`}>
                      {state}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
          {uncovered?.note && (
            <p className="border-t border-zinc-800 px-4 py-3 text-[11px] leading-relaxed text-zinc-500">
              <span className="font-medium text-zinc-400">{uncovered.label || 'Not tracked'}:</span> {uncovered.note}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
