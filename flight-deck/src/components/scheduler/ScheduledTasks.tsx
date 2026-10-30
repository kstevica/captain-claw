import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlarmClock,
  AlertTriangle,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleSlash,
  Copy,
  History,
  Inbox,
  ListChecks,
  Loader2,
  Mail,
  MessageSquare,
  Pause,
  Pencil,
  Play,
  Plug,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
  XCircle,
} from 'lucide-react'
import { useProcessStore } from '../../stores/processStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { useScheduledStore } from '../../stores/scheduledStore'
import { isManagedAgent } from '../../utils/managedAgents'
import {
  DAY_SHORT,
  buildSchedule,
  defaultScheduleForm,
  describeSchedule,
  explainStatus,
  fmtDuration,
  fmtRelative,
  fmtWhen,
  isFinishedOneShot,
  localTimeZone,
  parseSchedule,
  runTone,
  schedulerApi,
  type DeliveryOptions,
  type Frequency,
  type RunDelivery,
  type ScheduleForm,
  type ScheduledJob,
  type TaskDelivery,
  type TaskRun,
} from '../../services/schedulerApi'

// ── Scheduled tasks ─────────────────────────────────────────────────
//
// Your agents, on a timer. A task is a prompt one of your agents runs on a
// schedule — every run in the task's own agent session, so its history reads
// as one conversation — with the result delivered to the agent's chat, your
// email and/or your Telegram. Left: "All activity" (every run, newest first,
// searchable) and the tasks; right: the selected task with its run history,
// or the editor.

interface AgentChoice { slug: string; name: string; running: boolean }

type View = { kind: 'all' } | { kind: 'task'; id: string } | { kind: 'edit'; id: string | null }

const TEMPLATES: { title: string; prompt: string; form: Partial<ScheduleForm> }[] = [
  {
    title: 'Morning briefing',
    prompt: 'Give me a short briefing for today: my meetings, the emails that need me, and anything I should prepare for. Keep it to what matters.',
    form: { freq: 'weekdays', time: '08:00' },
  },
  {
    title: 'Weekly inbox digest',
    prompt: 'Summarise the week in my inbox: decisions waiting on me, open threads, and follow-ups I owe — grouped by person.',
    form: { freq: 'weekly', days: [4], time: '16:00' },
  },
  {
    title: 'News watch',
    prompt: 'Find today\'s most important news about our industry and competitors. Five bullets, each with a link and why it matters to us.',
    form: { freq: 'daily', time: '09:00' },
  },
  {
    title: 'Check-in reminder',
    prompt: 'Remind me to follow up on the proposal I sent, and draft a short, friendly follow-up I can send.',
    form: { freq: 'once', time: '09:00' },
  },
]

const iconBtn = 'rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50'

export function ScheduledTasks({
  initialTaskId, newForAgent, defaultAgentSlug, phone = false, onClose, onOpenChat, onOpenConnections, onOpenTaskboard,
}: {
  initialTaskId?: string | null
  /** Open straight on a new task (for this agent when not empty). */
  newForAgent?: string | null
  /** The agent a new task starts with (the chat in front). */
  defaultAgentSlug?: string
  phone?: boolean
  onClose: () => void
  /** Continue a task's conversation: open the agent's chat on its session. */
  onOpenChat?: (agentSlug: string, sessionId: string) => void
  onOpenConnections?: () => void
  /** A run's "Open task": the Taskboard on this task's own task there (by its job id). */
  onOpenTaskboard?: (jobId: string) => void
}) {
  const jobs = useScheduledStore((s) => s.jobs)
  const loaded = useScheduledStore((s) => s.loaded)
  const fetchJobs = useScheduledStore((s) => s.fetchJobs)
  const processes = useProcessStore((s) => s.processes)
  const notify = useNotificationStore((s) => s.add)

  const agents: AgentChoice[] = useMemo(() => processes
    .filter((p) => !isManagedAgent(p.slug, p.description || ''))
    .map((p) => ({ slug: p.slug, name: p.name || p.slug, running: p.status === 'running' }))
    .sort((a, b) => Number(b.running) - Number(a.running) || a.name.localeCompare(b.name)), [processes])
  const agentName = useCallback((slug: string) => agents.find((a) => a.slug === slug)?.name || slug, [agents])

  const [view, setView] = useState<View>(() =>
    newForAgent != null ? { kind: 'edit', id: null } : initialTaskId ? { kind: 'task', id: initialTaskId } : { kind: 'all' })
  // On a phone the list and the detail take turns.
  const [phoneDetail, setPhoneDetail] = useState(newForAgent != null || !!initialTaskId)
  const [template, setTemplate] = useState<(typeof TEMPLATES)[number] | null>(null)
  const [query, setQuery] = useState('')
  const [options, setOptions] = useState<DeliveryOptions | null>(null)

  useEffect(() => {
    fetchJobs()
    schedulerApi.deliveryOptions().then(setOptions).catch(() => setOptions(null))
    const t = setInterval(fetchJobs, 10_000)
    return () => clearInterval(t)
  }, [fetchJobs])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && view.kind !== 'edit') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, view.kind])

  const sorted = useMemo(() => [...jobs].sort((a, b) =>
    (b.enabled - a.enabled) || ((a.next_run_at ?? Infinity) - (b.next_run_at ?? Infinity)) || a.name.localeCompare(b.name)),
  [jobs])
  const selected = view.kind === 'task' ? jobs.find((j) => j.id === view.id) ?? null : null
  const editing = view.kind === 'edit' ? (view.id ? jobs.find((j) => j.id === view.id) ?? null : null) : null

  const go = (v: View) => { setView(v); setPhoneDetail(true); if (v.kind !== 'edit') setTemplate(null) }

  const startNew = (t?: (typeof TEMPLATES)[number]) => { setTemplate(t ?? null); go({ kind: 'edit', id: null }) }

  const header = (
    <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-500/15 text-amber-600 dark:text-amber-400">
        <AlarmClock className="h-[18px] w-[18px]" />
      </span>
      <div className="min-w-0 flex-1">
        <h2 className="truncate text-base font-semibold text-zinc-100">Scheduled tasks</h2>
        {!phone && <p className="truncate text-xs text-zinc-500">Your agents on a timer — results in chat, your inbox or Telegram.</p>}
      </div>
      <button
        onClick={() => startNew()}
        className="flex shrink-0 items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500"
      >
        <Plus className="h-4 w-4" />{phone ? 'New' : 'New task'}
      </button>
      <button onClick={onClose} className={iconBtn} title="Close" aria-label="Close">
        <X className="h-4 w-4" />
      </button>
    </div>
  )

  const searchBox = (
    <div className="relative">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search every result…"
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-2 pl-8 pr-8 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/60 focus:outline-none"
      />
      {query && (
        <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-zinc-500 hover:text-zinc-200" aria-label="Clear search">
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  )

  const list = (
    <div className="flex min-h-0 flex-col">
      <div className="px-3 pt-3">{searchBox}</div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        <button
          onClick={() => go({ kind: 'all' })}
          className={`mb-1 flex w-full items-center gap-3 rounded-lg px-2.5 py-2.5 text-left transition-colors ${
            view.kind === 'all' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-300 hover:bg-zinc-800/50'}`}
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-500 dark:text-violet-400">
            <Inbox className="h-4 w-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-medium">All activity</span>
            <span className="block truncate text-[11px] text-zinc-500">Every run, newest first</span>
          </span>
        </button>
        <div className="px-2.5 pb-1 pt-3 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          Tasks{jobs.length ? ` (${jobs.length})` : ''}
        </div>
        {!loaded && <div className="flex justify-center py-6 text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /></div>}
        {loaded && jobs.length === 0 && (
          <p className="px-2.5 py-3 text-xs leading-relaxed text-zinc-500">No tasks yet. Start with “New task” or a suggestion on the right.</p>
        )}
        <ul className="flex flex-col gap-0.5">
          {sorted.map((j) => (
            <li key={j.id}>
              <TaskRow job={j} agentName={agentName(j.agent_slug)} active={view.kind !== 'all' && (view.kind === 'task' || view.kind === 'edit') && view.id === j.id} onClick={() => go({ kind: 'task', id: j.id })} />
            </li>
          ))}
        </ul>
      </div>
    </div>
  )

  let detail: ReactNode
  if (view.kind === 'edit') {
    detail = (
      <TaskEditor
        key={view.id || `new-${template?.title || ''}`}
        job={editing}
        template={template}
        agents={agents}
        defaultAgent={newForAgent || defaultAgentSlug || ''}
        options={options}
        onOpenConnections={onOpenConnections}
        onCancel={() => {
          if (view.id) { go({ kind: 'task', id: view.id }); return }
          setView({ kind: 'all' })
          setPhoneDetail(false)
        }}
        onSaved={async (job, runNow) => {
          await fetchJobs()
          go({ kind: 'task', id: job.id })
          if (runNow) {
            try { await schedulerApi.runNow(job.id); fetchJobs() } catch (e) {
              notify('error', 'Could not run the task', e instanceof Error ? e.message : String(e))
            }
          }
        }}
      />
    )
  } else if (view.kind === 'task' && selected) {
    detail = (
      <TaskDetail
        key={selected.id}
        job={selected}
        agentName={agentName(selected.agent_slug)}
        agentRunning={agents.find((a) => a.slug === selected.agent_slug)?.running ?? false}
        query={query}
        onEdit={() => go({ kind: 'edit', id: selected.id })}
        onDeleted={async () => { await fetchJobs(); go({ kind: 'all' }) }}
        onChanged={fetchJobs}
        onOpenChat={onOpenChat}
        onOpenTaskboard={onOpenTaskboard}
      />
    )
  } else if (loaded && jobs.length === 0 && !query) {
    detail = <EmptyState onStart={startNew} />
  } else {
    detail = <ActivityFeed query={query} jobs={jobs} onOpenTask={(id) => go({ kind: 'task', id })} onOpenTaskboard={onOpenTaskboard} />
  }

  if (phone) {
    return (
      <div className="flex h-full flex-col bg-zinc-950">
        {header}
        {phoneDetail ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <button onClick={() => setPhoneDetail(false)} className="flex items-center gap-1.5 px-4 pt-3 text-sm text-violet-600 dark:text-violet-400">
              <ArrowLeft className="h-4 w-4" />All tasks
            </button>
            <div className="min-h-0 flex-1">{detail}</div>
          </div>
        ) : (
          <div className="min-h-0 flex-1">{list}</div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {header}
      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/40">{list}</aside>
        <section className="min-w-0 flex-1">{detail}</section>
      </div>
    </div>
  )
}

// ── List row ──

function TaskRow({ job, agentName, active, onClick }: {
  job: ScheduledJob; agentName: string; active: boolean; onClick: () => void
}) {
  const done = isFinishedOneShot(job)
  const paused = job.enabled !== 1 && !done
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-3 rounded-lg px-2.5 py-2.5 text-left transition-colors ${
        active ? 'bg-zinc-800' : 'hover:bg-zinc-800/50'} ${paused ? 'opacity-60' : ''}`}
    >
      <JobStatusIcon job={job} />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-zinc-100">{job.name || 'Untitled task'}</span>
          {(paused || done) && (
            <span className="shrink-0 rounded bg-zinc-800 px-1 py-0.5 text-[9px] font-medium uppercase tracking-wide text-zinc-400">{done ? 'Done' : 'Paused'}</span>
          )}
        </span>
        <span className="block truncate text-[11px] text-zinc-500">
          {describeSchedule(job.schedule, job.next_run_at)} · {agentName}
        </span>
      </span>
      <span className="flex shrink-0 flex-col items-end gap-1">
        <span className="text-[10px] text-zinc-500">
          {job.running ? 'running' : !paused && job.next_run_at ? fmtRelative(job.next_run_at) : ''}
        </span>
        <DeliveryIcons job={job} />
      </span>
    </button>
  )
}

function DeliveryIcons({ job }: { job: ScheduledJob }) {
  const d = job.delivery
  const cls = 'h-3 w-3 text-zinc-500'
  if (!d) return <span className="text-[9px] uppercase tracking-wide text-zinc-500">{job.delivery_kind}</span>
  return (
    <span className="flex items-center gap-1">
      {d.chat && <MessageSquare className={cls} aria-label="Chat" />}
      {d.email && <Mail className={cls} aria-label="Email" />}
      {d.telegram && <Send className={cls} aria-label="Telegram" />}
    </span>
  )
}

function JobStatusIcon({ job }: { job: ScheduledJob }) {
  const base = 'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg'
  if (job.running) {
    return <span className={`${base} bg-amber-500/15 text-amber-600 dark:text-amber-400`}><Loader2 className="h-4 w-4 animate-spin" /></span>
  }
  const tone = job.last_status ? runTone(job.last_status) : null
  if (isFinishedOneShot(job) && tone !== 'error') {
    return <span className={`${base} bg-emerald-500/10 text-emerald-600 dark:text-emerald-400`}><CheckCircle2 className="h-4 w-4" /></span>
  }
  if (job.enabled !== 1) return <span className={`${base} bg-zinc-800 text-zinc-500`}><Pause className="h-4 w-4" /></span>
  if (tone === 'error') return <span className={`${base} bg-red-500/10 text-red-600 dark:text-red-400`}><AlertTriangle className="h-4 w-4" /></span>
  if (tone === 'partial') return <span className={`${base} bg-amber-500/10 text-amber-600 dark:text-amber-400`}><AlertTriangle className="h-4 w-4" /></span>
  return <span className={`${base} bg-amber-500/10 text-amber-600 dark:text-amber-400`}><AlarmClock className="h-4 w-4" /></span>
}

// ── Empty state ──

function EmptyState({ onStart }: { onStart: (t?: (typeof TEMPLATES)[number]) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center overflow-y-auto px-6 py-8 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-500/15 text-amber-600 dark:text-amber-400">
        <AlarmClock className="h-7 w-7" />
      </span>
      <h3 className="mt-4 text-lg font-semibold text-zinc-100">Let an agent work while you don't</h3>
      <p className="mt-1.5 max-w-md text-sm leading-relaxed text-zinc-400">
        Every morning, every Friday, or once at a set time: your agent does the job and sends you the
        result — in its chat, by email or on Telegram. Each task keeps its own history you can search.
      </p>
      <div className="mt-6 grid w-full max-w-lg grid-cols-1 gap-2.5 wide:grid-cols-2">
        {TEMPLATES.map((t) => (
          <button
            key={t.title}
            onClick={() => onStart(t)}
            className="flex flex-col items-start gap-1 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3.5 text-left transition-colors hover:border-violet-500/50 hover:bg-zinc-800/60"
          >
            <span className="flex items-center gap-1.5 text-sm font-medium text-zinc-200">
              <Sparkles className="h-3.5 w-3.5 text-violet-500 dark:text-violet-400" />{t.title}
            </span>
            <span className="line-clamp-2 text-[11px] leading-snug text-zinc-500">{t.prompt}</span>
          </button>
        ))}
      </div>
      <button onClick={() => onStart()} className="mt-4 text-sm font-medium text-violet-600 hover:underline dark:text-violet-400">
        Or start from scratch
      </button>
    </div>
  )
}

// ── Runs: loading + polling ──

function useRuns(jobId: string | null, query: string, live: boolean) {
  const [runs, setRuns] = useState<TaskRun[]>([])
  const [loading, setLoading] = useState(true)
  const [hasMore, setHasMore] = useState(false)
  const reqId = useRef(0)
  const PAGE = 30

  const load = useCallback(async (quiet = false) => {
    const id = ++reqId.current
    if (!quiet) setLoading(true)
    try {
      const rows = await schedulerApi.runs({ jobId: jobId || undefined, q: query.trim() || undefined, limit: PAGE })
      if (id !== reqId.current) return
      setRuns((prev) => {
        // A refresh keeps pages already loaded below the first.
        if (!quiet || prev.length <= PAGE) return rows
        const ids = new Set(rows.map((r) => r.id))
        return [...rows, ...prev.filter((r) => !ids.has(r.id) && r.id < (rows[rows.length - 1]?.id ?? 0))]
      })
      if (!quiet) setHasMore(rows.length === PAGE)
    } catch {
      if (id === reqId.current && !quiet) setRuns([])
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [jobId, query])

  const loadMore = useCallback(async () => {
    const last = runs[runs.length - 1]
    if (!last) return
    const rows = await schedulerApi.runs({ jobId: jobId || undefined, q: query.trim() || undefined, limit: PAGE, before: last.id })
    setRuns((prev) => [...prev, ...rows.filter((r) => !prev.some((p) => p.id === r.id))])
    setHasMore(rows.length === PAGE)
  }, [runs, jobId, query])

  // Search as you type, a beat after the last key.
  useEffect(() => {
    const t = setTimeout(() => load(), query ? 250 : 0)
    return () => clearTimeout(t)
  }, [load, query])

  const anyRunning = live || runs.some((r) => r.status === 'running')
  useEffect(() => {
    const t = setInterval(() => load(true), anyRunning ? 3000 : 15000)
    return () => clearInterval(t)
  }, [load, anyRunning])

  return { runs, loading, hasMore, loadMore, reload: () => load(true) }
}

function ActivityFeed({ query, jobs, onOpenTask, onOpenTaskboard }: {
  query: string; jobs: ScheduledJob[]; onOpenTask: (id: string) => void; onOpenTaskboard?: (jobId: string) => void
}) {
  const { runs, loading, hasMore, loadMore } = useRuns(null, query, jobs.some((j) => j.running))
  return (
    <div className="h-full overflow-y-auto px-4 py-4 wide:px-6">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold text-zinc-100">
          {query ? <>Results for “{query}”</> : 'All activity'}
        </h3>
        {!loading && <span className="text-[11px] text-zinc-500">{runs.length}{hasMore ? '+' : ''} run{runs.length === 1 ? '' : 's'}</span>}
      </div>
      <RunList runs={runs} loading={loading} hasMore={hasMore} onMore={loadMore} showTask onOpenTask={onOpenTask} query={query}
        onOpenTaskboard={onOpenTaskboard}
        empty={query ? 'Nothing matches that search.' : 'No runs yet. A task\'s results show up here as soon as it runs.'} />
    </div>
  )
}

function RunList({ runs, loading, hasMore, onMore, showTask, onOpenTask, onOpenTaskboard, empty, query }: {
  runs: TaskRun[]; loading: boolean; hasMore: boolean; onMore: () => void
  showTask?: boolean; onOpenTask?: (id: string) => void; onOpenTaskboard?: (jobId: string) => void
  empty: string; query: string
}) {
  const [more, setMore] = useState(false)
  if (loading && runs.length === 0) {
    return <div className="flex justify-center py-10 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  }
  if (runs.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-zinc-800 px-4 py-10 text-center text-sm text-zinc-500">
        <History className="mx-auto mb-2 h-5 w-5 text-zinc-600" />{empty}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {runs.map((r) => <RunCard key={r.id} run={r} showTask={showTask} onOpenTask={onOpenTask} onOpenTaskboard={onOpenTaskboard} query={query} />)}
      {hasMore && (
        <button
          onClick={async () => { setMore(true); try { await onMore() } finally { setMore(false) } }}
          className="mx-auto flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          {more ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ChevronDown className="h-3.5 w-3.5" />}Earlier runs
        </button>
      )}
    </div>
  )
}

// ── One run ──

const TONE_PILL: Record<string, { label: string; cls: string; Icon: typeof Check }> = {
  ok: { label: 'Delivered', cls: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400', Icon: CheckCircle2 },
  partial: { label: 'Partly delivered', cls: 'bg-amber-500/10 text-amber-700 dark:text-amber-400', Icon: AlertTriangle },
  error: { label: 'Failed', cls: 'bg-red-500/10 text-red-700 dark:text-red-400', Icon: XCircle },
  running: { label: 'Running', cls: 'bg-amber-500/10 text-amber-700 dark:text-amber-400', Icon: Loader2 },
  skipped: { label: 'Skipped', cls: 'bg-zinc-800 text-zinc-400', Icon: CircleSlash },
}

const VIA: Record<string, { label: string; Icon: typeof Mail }> = {
  chat: { label: 'Chat', Icon: MessageSquare },
  email: { label: 'Email', Icon: Mail },
  telegram: { label: 'Telegram', Icon: Send },
  whatsapp: { label: 'WhatsApp', Icon: MessageSquare },
  channel: { label: 'Channel', Icon: MessageSquare },
}

function RunCard({ run, showTask, onOpenTask, onOpenTaskboard, query }: {
  run: TaskRun; showTask?: boolean; onOpenTask?: (id: string) => void
  /** The Taskboard on the task this run is part of. */
  onOpenTaskboard?: (jobId: string) => void
  query: string
}) {
  const tone = runTone(run.status)
  const pill = TONE_PILL[tone]
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const long = run.result.length > 600 || run.result.split('\n').length > 10
  const hits = useMemo(() => countHits(run.result, query), [run.result, query])

  const copy = async () => {
    try { await navigator.clipboard.writeText(run.result); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* ignore */ }
  }

  return (
    <article className="rounded-xl border border-zinc-800 bg-zinc-900/50">
      <header className="flex flex-wrap items-center gap-x-2 gap-y-1 px-4 pt-3">
        <span className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${pill.cls}`}>
          <pill.Icon className={`h-3 w-3 ${tone === 'running' ? 'animate-spin' : ''}`} />{pill.label}
        </span>
        {showTask && (
          <button onClick={() => onOpenTask?.(run.job_id)} className="truncate text-sm font-medium text-zinc-100 hover:underline">
            {run.job_name || 'Untitled task'}
          </button>
        )}
        <span className="text-xs text-zinc-500" title={new Date(run.started_at * 1000).toLocaleString()}>
          {fmtWhen(run.started_at)}
        </span>
        {run.trigger === 'manual' && <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">Run now</span>}
        {fmtDuration(run) && <span className="text-[11px] text-zinc-600">· {fmtDuration(run)}</span>}
        {hits > 0 && <span className="rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] font-medium text-violet-700 dark:text-violet-300">{hits} match{hits === 1 ? '' : 'es'}</span>}
        <span className="ml-auto flex items-center gap-1">
          {onOpenTaskboard && run.job_id && (
            <button
              onClick={() => onOpenTaskboard(run.job_id)}
              className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              title="This task on the Taskboard — every run, what it delivered, and whether it needs you"
            >
              <ListChecks className="h-3.5 w-3.5" />Open task
            </button>
          )}
          {run.result && (
            <button onClick={copy} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200" title="Copy result">
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          )}
        </span>
      </header>

      {run.deliveries.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-4 pt-2">
          {run.deliveries.map((d, i) => <DeliveryChip key={i} d={d} />)}
        </div>
      )}

      <div className="px-4 pb-3 pt-2">
        {tone === 'running' ? (
          <p className="text-sm text-zinc-400">Working on it — the result appears here when the agent is done.</p>
        ) : run.result ? (
          <div className="relative">
            <div className={`fd-markdown text-sm text-zinc-300 ${long && !open ? 'max-h-48 overflow-hidden' : ''}`}>
              <Markdown remarkPlugins={[remarkGfm]}>{run.result}</Markdown>
            </div>
            {long && !open && <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-zinc-950/90 to-transparent" />}
            {long && (
              <button onClick={() => setOpen((v) => !v)} className="mt-1 text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
                {open ? 'Show less' : 'Show the full result'}
              </button>
            )}
          </div>
        ) : null}
        {(tone === 'error' || tone === 'skipped' || tone === 'partial') && (
          <p className={`mt-2 rounded-lg px-3 py-2 text-xs ${tone === 'error' ? 'bg-red-500/10 text-red-700 dark:text-red-300' : 'bg-zinc-800/60 text-zinc-400'}`}>
            {explainStatus(run.status)}
          </p>
        )}
      </div>
    </article>
  )
}

function countHits(text: string, query: string): number {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length || !text) return 0
  const low = text.toLowerCase()
  let n = 0
  for (const w of words) {
    let at = low.indexOf(w)
    while (at >= 0) { n++; at = low.indexOf(w, at + w.length) }
  }
  return n
}

function DeliveryChip({ d }: { d: RunDelivery }) {
  const v = VIA[d.via] || { label: d.via, Icon: MessageSquare }
  const detail = d.via === 'email' && d.to ? `${v.label} → ${d.to}` : v.label
  return (
    <span
      title={d.note || undefined}
      className={`flex max-w-full items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] ${
        d.ok
          ? 'border-zinc-800 text-zinc-400'
          : 'border-red-500/30 bg-red-500/5 text-red-700 dark:text-red-300'}`}
    >
      <v.Icon className="h-3 w-3 shrink-0" />
      <span className="truncate">{detail}</span>
      {d.ok ? <Check className="h-3 w-3 shrink-0 text-emerald-600 dark:text-emerald-400" /> : <X className="h-3 w-3 shrink-0" />}
      {!d.ok && d.note && <span className="truncate text-[10px] opacity-80">· {d.note}</span>}
    </span>
  )
}

// ── One task ──

function TaskDetail({ job, agentName, agentRunning, query, onEdit, onDeleted, onChanged, onOpenChat, onOpenTaskboard }: {
  job: ScheduledJob; agentName: string; agentRunning: boolean; query: string
  onEdit: () => void; onDeleted: () => void; onChanged: () => void
  onOpenChat?: (agentSlug: string, sessionId: string) => void
  onOpenTaskboard?: (jobId: string) => void
}) {
  const notify = useNotificationStore((s) => s.add)
  const [busy, setBusy] = useState<'' | 'run' | 'toggle' | 'delete'>('')
  const { runs, loading, hasMore, loadMore, reload } = useRuns(job.id, query, job.running)
  const [promptOpen, setPromptOpen] = useState(false)
  const done = isFinishedOneShot(job)
  const paused = job.enabled !== 1 && !done
  const isTask = job.delivery_kind === 'agent'

  const act = async (kind: 'run' | 'toggle' | 'delete', fn: () => Promise<unknown>, failTitle: string) => {
    setBusy(kind)
    try { await fn(); onChanged(); reload() } catch (e) {
      notify('error', failTitle, e instanceof Error ? e.message : String(e))
    } finally { setBusy('') }
  }

  const remove = () => {
    if (!confirm(`Delete “${job.name || 'this task'}”? Its history goes with it.`)) return
    setBusy('delete')
    schedulerApi.deleteJob(job.id).then(onDeleted).catch((e) => {
      notify('error', 'Could not delete the task', e instanceof Error ? e.message : String(e))
      setBusy('')
    })
  }

  const d = job.delivery
  const deliversTo = d
    ? [d.chat && `${agentName}'s chat`, d.email && (d.email_to ? `email to ${d.email_to}` : 'your email'), d.telegram && 'Telegram']
      .filter(Boolean).join(' · ')
    : `${job.delivery_kind} ${job.delivery_target}`

  return (
    <div className="h-full overflow-y-auto px-4 py-4 wide:px-6">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-semibold leading-tight text-zinc-100">{job.name || 'Untitled task'}</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            {describeSchedule(job.schedule, job.next_run_at)}{job.tz ? ` · ${job.tz.replace(/_/g, ' ')}` : ''} · {agentName}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => act('run', () => schedulerApi.runNow(job.id), 'Could not run the task')}
            disabled={!!busy || job.running}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-sm font-medium text-zinc-200 transition-colors hover:border-violet-500/50 hover:bg-zinc-800 disabled:opacity-50"
            title={agentRunning ? 'Run it once now (the schedule stays as it is)' : 'The agent is off — power it on first'}
          >
            {busy === 'run' || job.running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {job.running ? 'Running…' : 'Run now'}
          </button>
          {isTask && (
            <button onClick={onEdit} disabled={!!busy} className={iconBtn} title="Edit" aria-label="Edit">
              <Pencil className="h-4 w-4" />
            </button>
          )}
          <button onClick={remove} disabled={!!busy} className={`${iconBtn} hover:text-red-600 dark:hover:text-red-400`} title="Delete" aria-label="Delete">
            {busy === 'delete' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* At a glance */}
      <div className="mt-4 grid grid-cols-1 gap-2 wide:grid-cols-3">
        <Glance label="Next run">
          {done ? <span className="text-zinc-500">Done — it ran once</span>
            : paused ? <span className="text-zinc-500">Paused</span>
            : job.running ? <span className="text-amber-700 dark:text-amber-400">Running now</span>
            : job.next_run_at ? <>{fmtWhen(job.next_run_at)} <span className="text-zinc-500">· {fmtRelative(job.next_run_at)}</span></>
            : <span className="text-zinc-500">Done</span>}
        </Glance>
        <Glance label="Delivers to">{deliversTo}</Glance>
        <Glance label="Status">
          {done ? (
            <button onClick={onEdit} className="text-violet-600 hover:underline dark:text-violet-400">Schedule it again</button>
          ) : (
            <label className="flex cursor-pointer items-center gap-2">
              <Switch on={!paused} busy={busy === 'toggle'} onChange={(on) =>
                act('toggle', () => schedulerApi.updateJob(job.id, { enabled: on }), on ? 'Could not resume' : 'Could not pause')} />
              <span>{paused ? 'Paused' : 'Active'}</span>
            </label>
          )}
        </Glance>
      </div>

      {!agentRunning && !paused && (
        <p className="mt-3 flex items-start gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {agentName} is off. Tasks only run while their agent is on — power it on in the agent list.
        </p>
      )}

      <button onClick={() => setPromptOpen((v) => !v)} className="mt-4 block w-full rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-left">
        <span className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-zinc-500">
          What it does
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${promptOpen ? 'rotate-180' : ''}`} />
        </span>
        <span className={`mt-1 block whitespace-pre-wrap text-sm text-zinc-300 ${promptOpen ? '' : 'line-clamp-2'}`}>{job.prompt}</span>
      </button>

      <div className="mb-3 mt-6 flex flex-wrap items-center gap-2">
        <h4 className="text-sm font-semibold text-zinc-100">
          {query ? <>Runs matching “{query}”</> : 'History'}
        </h4>
        <span className="text-[11px] text-zinc-500">{job.run_count} run{job.run_count === 1 ? '' : 's'} kept</span>
        {onOpenChat && job.agent_session_id && (
          <button
            onClick={() => onOpenChat(job.agent_slug, job.agent_session_id)}
            className="ml-auto flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium text-violet-600 hover:bg-violet-500/10 dark:text-violet-400"
            title="Open this task's conversation in the agent's chat and keep talking about it"
          >
            <MessageSquare className="h-3.5 w-3.5" />Continue in chat
          </button>
        )}
      </div>
      <RunList runs={runs} loading={loading} hasMore={hasMore} onMore={loadMore} query={query} onOpenTaskboard={onOpenTaskboard}
        empty={query ? 'No run of this task matches.' : 'It hasn\'t run yet. Use “Run now” to try it.'} />
    </div>
  )
}

function Glance({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3.5 py-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mt-0.5 truncate text-sm text-zinc-200">{children}</div>
    </div>
  )
}

function Switch({ on, busy, onChange }: { on: boolean; busy?: boolean; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={busy}
      onClick={() => onChange(!on)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-60 ${on ? 'bg-violet-600' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${on ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  )
}

// ── The editor ──

const FREQS: { id: Frequency; label: string }[] = [
  { id: 'daily', label: 'Every day' },
  { id: 'weekdays', label: 'Weekdays' },
  { id: 'weekly', label: 'Weekly' },
  { id: 'every', label: 'Repeat every…' },
  { id: 'once', label: 'Once' },
]

function TaskEditor({ job, template, agents, defaultAgent, options, onOpenConnections, onCancel, onSaved }: {
  job: ScheduledJob | null
  template: (typeof TEMPLATES)[number] | null
  agents: AgentChoice[]
  defaultAgent: string
  options: DeliveryOptions | null
  onOpenConnections?: () => void
  onCancel: () => void
  onSaved: (job: ScheduledJob, runNow: boolean) => void
}) {
  const tz = job?.tz || localTimeZone()
  const [prompt, setPrompt] = useState(job?.prompt ?? template?.prompt ?? '')
  const [name, setName] = useState(job?.name ?? template?.title ?? '')
  const [picked, setAgent] = useState(() => {
    if (job) return job.agent_slug
    if (defaultAgent && agents.some((a) => a.slug === defaultAgent)) return defaultAgent
    return ''
  })
  // Until one is picked: the first agent that's on (the list may still be loading).
  const agent = picked || agents.find((a) => a.running)?.slug || agents[0]?.slug || ''
  const [form, setForm] = useState<ScheduleForm>(() =>
    job ? parseSchedule(job.schedule, job.next_run_at) : { ...defaultScheduleForm(), ...(template?.form || {}) })
  const [delivery, setDelivery] = useState<TaskDelivery>(() =>
    job?.delivery ?? { chat: true, email: false, telegram: false, email_to: '' })
  const [editTo, setEditTo] = useState(!!job?.delivery?.email_to)
  const [saving, setSaving] = useState<'' | 'save' | 'run'>('')
  const [error, setError] = useState('')
  const [preview, setPreview] = useState<{ next: number[]; error: string } | null>(null)
  const promptRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => { if (!job) promptRef.current?.focus() }, [job])

  const schedule = buildSchedule(form)
  useEffect(() => {
    let cancelled = false
    const t = setTimeout(() => {
      schedulerApi.preview(schedule, tz)
        .then((p) => { if (!cancelled) setPreview({ next: p.next, error: p.error }) })
        .catch(() => { if (!cancelled) setPreview(null) })
    }, 200)
    return () => { cancelled = true; clearTimeout(t) }
  }, [schedule, tz])

  const email = options?.email
  const telegram = options?.telegram
  const nothingPicked = !delivery.chat && !delivery.email && !delivery.telegram
  const canSave = !!prompt.trim() && !!agent && !nothingPicked && !preview?.error && !saving

  const submit = async (runNow: boolean) => {
    if (!canSave) return
    setSaving(runNow ? 'run' : 'save')
    setError('')
    const body = {
      name: name.trim() || autoName(prompt),
      schedule, tz, agent_slug: agent, prompt: prompt.trim(),
      delivery: { ...delivery, email_to: editTo ? delivery.email_to : '' },
      // A one-shot that already ran comes back on when it gets a new time;
      // a paused task stays paused.
      ...(job && isFinishedOneShot(job) ? { enabled: true } : {}),
    }
    try {
      const saved = job ? await schedulerApi.updateJob(job.id, body) : await schedulerApi.createTask(body)
      onSaved(saved, runNow)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSaving('')
    }
  }

  const label = 'mb-1.5 block text-xs font-medium uppercase tracking-wider text-zinc-500'
  const field = 'rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-violet-500/60 focus:outline-none'
  const input = `w-full ${field}`

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-6">
        <h3 className="mb-4 text-lg font-semibold text-zinc-100">{job ? 'Edit task' : 'New task'}</h3>

        {/* What */}
        <section className="mb-5">
          <label className={label} htmlFor="task-prompt">What should the agent do?</label>
          <textarea
            id="task-prompt"
            ref={promptRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="e.g. Every morning, tell me what's on my calendar and which emails need an answer today."
            className={`${input} resize-y leading-relaxed`}
          />
          {!job && !prompt && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {TEMPLATES.map((t) => (
                <button
                  key={t.title}
                  onClick={() => { setPrompt(t.prompt); if (!name) setName(t.title); setForm((f) => ({ ...f, ...t.form })) }}
                  className="flex items-center gap-1 rounded-full border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 transition-colors hover:border-violet-500/50 hover:text-zinc-100"
                >
                  <Sparkles className="h-3 w-3 text-violet-500 dark:text-violet-400" />{t.title}
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Who */}
        <section className="mb-5">
          <span className={label}>Which agent</span>
          {agents.length === 0 ? (
            <p className="text-sm text-zinc-500">You have no agents yet — create one first.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {agents.map((a) => (
                <button
                  key={a.slug}
                  onClick={() => setAgent(a.slug)}
                  className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors ${
                    agent === a.slug
                      ? 'border-violet-500/60 bg-violet-500/10 text-zinc-100'
                      : 'border-zinc-700 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-800/60'}`}
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${a.running ? 'bg-emerald-500' : 'bg-zinc-600'}`} />
                  {a.name}
                </button>
              ))}
            </div>
          )}
          {agent && !agents.find((a) => a.slug === agent)?.running && agents.length > 0 && (
            <p className="mt-1.5 text-[11px] text-amber-700 dark:text-amber-400">This agent is off — it has to be on when the task runs.</p>
          )}
        </section>

        {/* When */}
        <section className="mb-5">
          <span className={label}>When</span>
          <div className="flex flex-wrap gap-1 rounded-lg border border-zinc-800 bg-zinc-900/60 p-1">
            {FREQS.map((f) => (
              <button
                key={f.id}
                onClick={() => setForm((s) => ({ ...s, freq: f.id }))}
                className={`flex-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  form.freq === f.id ? 'bg-zinc-700 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`}
              >{f.label}</button>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {form.freq === 'weekly' && (
              <div className="flex gap-1">
                {DAY_SHORT.map((d, i) => {
                  const on = form.days.includes(i)
                  return (
                    <button
                      key={d}
                      onClick={() => setForm((s) => ({
                        ...s, days: on ? (s.days.length > 1 ? s.days.filter((x) => x !== i) : s.days) : [...s.days, i],
                      }))}
                      className={`h-9 w-10 rounded-lg text-xs font-medium transition-colors ${
                        on ? 'bg-violet-600 text-white' : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}
                    >{d.slice(0, 2)}</button>
                  )
                })}
              </div>
            )}
            {form.freq === 'every' ? (
              <div className="flex items-center gap-2 text-sm text-zinc-400">
                Every
                <input
                  type="number" min={1} max={999}
                  value={form.every}
                  onChange={(e) => setForm((s) => ({ ...s, every: Number(e.target.value) || 1 }))}
                  className={`${field} w-20`}
                />
                <select value={form.unit} onChange={(e) => setForm((s) => ({ ...s, unit: e.target.value as ScheduleForm['unit'] }))} className={field}>
                  <option value="m">minutes</option>
                  <option value="h">hours</option>
                  <option value="d">days</option>
                </select>
              </div>
            ) : (
              <>
                {form.freq === 'once' && (
                  <input type="date" value={form.date} onChange={(e) => setForm((s) => ({ ...s, date: e.target.value }))} className={field} />
                )}
                <span className="text-sm text-zinc-400">at</span>
                <input type="time" value={form.time} onChange={(e) => setForm((s) => ({ ...s, time: e.target.value }))} className={field} />
              </>
            )}
          </div>
          {form.freq === 'every' && form.unit === 'm' && form.every < 15 && (
            <p className="mt-1.5 text-[11px] text-amber-700 dark:text-amber-400">That's very often — each run costs a full agent turn.</p>
          )}
          <p className={`mt-2 text-xs ${preview?.error ? 'text-red-600 dark:text-red-400' : 'text-zinc-500'}`}>
            {preview?.error
              ? capitalize(preview.error)
              : preview?.next.length
                ? <>Next: <span className="text-zinc-300">{fmtWhen(preview.next[0])}</span>
                    {preview.next.slice(1).map((t) => <span key={t}>, then {fmtWhen(t)}</span>)}
                    {tz && <span className="text-zinc-600"> · {tz.replace(/_/g, ' ')} time</span>}</>
                : ' '}
          </p>
        </section>

        {/* Where */}
        <section className="mb-5">
          <span className={label}>Send the result to</span>
          <div className="grid grid-cols-1 gap-2 wide:grid-cols-3">
            <DeliveryOption
              icon={MessageSquare} title="Agent chat"
              detail={`In your chat with ${agents.find((a) => a.slug === agent)?.name || 'the agent'}`}
              on={delivery.chat} onToggle={() => setDelivery((d) => ({ ...d, chat: !d.chat }))}
            />
            <DeliveryOption
              icon={Mail} title="Email"
              detail={email?.available ? `To ${editTo && delivery.email_to ? delivery.email_to : email.to || 'you'}` : (email?.reason || 'Checking…')}
              on={delivery.email}
              disabled={!email?.available}
              onToggle={() => setDelivery((d) => ({ ...d, email: !d.email }))}
              action={!email?.available && onOpenConnections ? { label: 'Connect', icon: Plug, onClick: onOpenConnections } : undefined}
            />
            <DeliveryOption
              icon={Send} title="Telegram"
              detail={telegram?.available ? 'Your chat with the deck\'s bot' : (telegram?.reason || 'Checking…')}
              on={delivery.telegram}
              disabled={!telegram?.available}
              onToggle={() => setDelivery((d) => ({ ...d, telegram: !d.telegram }))}
              action={!telegram?.available && onOpenConnections ? { label: 'Link', icon: Plug, onClick: onOpenConnections } : undefined}
            />
          </div>
          {delivery.email && email?.available && (
            <div className="mt-2">
              {editTo ? (
                <input
                  value={delivery.email_to}
                  onChange={(e) => setDelivery((d) => ({ ...d, email_to: e.target.value }))}
                  placeholder={`${email.to || 'you@example.com'}, colleague@example.com`}
                  className={input}
                />
              ) : (
                <button onClick={() => setEditTo(true)} className="text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
                  Send to other addresses
                </button>
              )}
              <p className="mt-1 text-[11px] text-zinc-500">Sent from {email.from || 'your connected mailbox'}.</p>
            </div>
          )}
          {nothingPicked && <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">Pick at least one place for the result.</p>}
        </section>

        {/* Name */}
        <section className="mb-2">
          <label className={label} htmlFor="task-name">Name</label>
          <input id="task-name" value={name} onChange={(e) => setName(e.target.value)} placeholder={autoName(prompt) || 'Morning briefing'} className={input} />
          <p className="mt-1 text-[11px] text-zinc-500">The task's conversation is kept in its own session, named after it.</p>
        </section>

        {error && (
          <p className="mt-3 flex items-start gap-2 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{capitalize(error)}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-zinc-800 px-4 py-3 wide:px-6">
        <button onClick={onCancel} className="rounded-lg px-3 py-2 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">Cancel</button>
        <button
          onClick={() => submit(true)}
          disabled={!canSave}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800 disabled:opacity-50"
        >
          {saving === 'run' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {job ? 'Save & run now' : 'Create & run now'}
        </button>
        <button
          onClick={() => submit(false)}
          disabled={!canSave}
          className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-50"
        >
          {saving === 'save' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
          {job ? 'Save' : 'Create task'}
        </button>
      </div>
    </div>
  )
}

function DeliveryOption({ icon: Icon, title, detail, on, disabled, onToggle, action }: {
  icon: typeof Mail; title: string; detail: string; on: boolean; disabled?: boolean
  onToggle: () => void
  action?: { label: string; icon: typeof Mail; onClick: () => void }
}) {
  return (
    <div
      role="checkbox"
      aria-checked={on}
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
      onClick={() => { if (!disabled) onToggle() }}
      onKeyDown={(e) => { if (!disabled && (e.key === ' ' || e.key === 'Enter')) { e.preventDefault(); onToggle() } }}
      className={`relative flex cursor-pointer flex-col gap-1 rounded-xl border p-3 text-left transition-colors ${
        disabled ? 'cursor-default border-zinc-800 opacity-70'
          : on ? 'border-violet-500/60 bg-violet-500/10' : 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800/40'}`}
    >
      <span className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${on && !disabled ? 'text-violet-600 dark:text-violet-400' : 'text-zinc-500'}`} />
        <span className="text-sm font-medium text-zinc-200">{title}</span>
        <span className={`ml-auto flex h-4 w-4 items-center justify-center rounded border ${
          on && !disabled ? 'border-violet-600 bg-violet-600 text-white' : 'border-zinc-600'}`}>
          {on && !disabled && <Check className="h-3 w-3" />}
        </span>
      </span>
      <span className="line-clamp-2 text-[11px] leading-snug text-zinc-500">{detail}</span>
      {action && (
        <button
          onClick={(e) => { e.stopPropagation(); action.onClick() }}
          className="mt-0.5 flex w-fit items-center gap-1 rounded-md bg-zinc-800 px-2 py-1 text-[11px] font-medium text-zinc-200 hover:bg-zinc-700"
        >
          <action.icon className="h-3 w-3" />{action.label}
        </button>
      )}
    </div>
  )
}

function autoName(prompt: string): string {
  const words = prompt.trim().replace(/\s+/g, ' ').split(' ').slice(0, 6).join(' ')
  return words.length > 48 ? `${words.slice(0, 47)}…` : words
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s
}
