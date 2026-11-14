import { useMemo, useState } from 'react'
import { Flame, Loader2, Plus, Search, Sparkles, Square, Trash2, Users, Wand2, X, Network, PanelLeftClose, PanelLeftOpen, Share2 } from 'lucide-react'
import type { BasnaSession } from '../../stores/basnaStore'
import { Pill } from '../console/parts'
import { agentOrigin, confColor, DIFFICULTY_COLOR, isVatra, STATUS_DOT, timeAgo } from './shared'

// ── Runs sidebar: search + filters + day-grouped run list ───────────────────

type ModeFilter = 'all' | 'basna' | 'vatra' | 'agent'
type StatusFilter = 'all' | 'running' | 'done' | 'error'

const ACTIVE_STATUSES = ['routing', 'routed', 'running']

// A run's state as a word, where the dot alone says too little.
const STATUS_PILL: Record<string, { label: string; tone: 'zinc' | 'sky' | 'amber' | 'red' }> = {
  routing: { label: 'Routing', tone: 'sky' },
  routed: { label: 'Ready', tone: 'zinc' },
  running: { label: 'Running', tone: 'amber' },
  planning: { label: 'Planning', tone: 'sky' },
  awaiting_plan: { label: 'Review plan', tone: 'amber' },
  error: { label: 'Failed', tone: 'red' },
  cancelled: { label: 'Stopped', tone: 'zinc' },
}

function SessionCard({ s, active, showMode, onOpen, onDelete, onCancel, onShare }: {
  s: BasnaSession; active: boolean; showMode: boolean; onOpen: () => void; onDelete: () => void; onCancel: () => void
  onShare?: (s: BasnaSession) => void
}) {
  // Only show the spinner + stop control while genuinely active. 'routed' is the
  // idle prepared state (Basna routed, or a Vatra plan awaiting Run) — not running.
  const working = s.status === 'running' || s.status === 'routing'
  const origin = agentOrigin(s.config)
  const vatra = isVatra(s.config)
  const pill = STATUS_PILL[s.status]
  return (
    <div
      onClick={onOpen}
      className={`group relative cursor-pointer rounded-xl border px-3 py-2.5 transition-colors ${
        active
          ? 'border-violet-500/50 bg-violet-500/[0.07] shadow-sm'
          : 'border-transparent hover:border-zinc-800 hover:bg-zinc-900/70'
      }`}
    >
      {active && <span className="absolute inset-y-2.5 left-0 w-[3px] rounded-full bg-violet-500" />}
      <div className="flex items-start gap-2.5">
        <span className="mt-[5px] flex h-3 w-3 shrink-0 items-center justify-center">
          {working
            ? <Loader2 className="h-3 w-3 animate-spin text-amber-500" />
            : <span className={`block h-2 w-2 rounded-full ${STATUS_DOT[s.status] || 'bg-zinc-500'}`} />}
        </span>
        <div className="min-w-0 flex-1">
          <p className={`line-clamp-1 text-sm font-medium leading-snug ${active ? 'text-zinc-100' : 'text-zinc-200'}`} title={s.intent || ''}>
            {s.title || s.intent || '(untitled)'}
          </p>
          {s.title && s.intent && (
            <p className="mt-0.5 line-clamp-1 text-xs leading-snug text-zinc-500" title={s.intent}>{s.intent}</p>
          )}
        </div>
        <div className="-mr-1 -mt-0.5 flex shrink-0 items-center">
          {working && (
            <button
              onClick={(e) => { e.stopPropagation(); onCancel() }}
              title="Stop this run"
              aria-label="Stop this run"
              className="rounded-lg p-1 text-amber-500 transition-colors hover:bg-red-500/10 hover:text-red-500"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          )}
          {!s.shared && onShare && (
            <button
              onClick={(e) => { e.stopPropagation(); onShare(s) }}
              title="Share with other users"
              aria-label="Share with other users"
              className="rounded-lg p-1 text-zinc-500 transition-all hover:bg-zinc-800 hover:text-zinc-200 focus-visible:opacity-100 wide:opacity-0 wide:group-hover:opacity-100"
            >
              <Share2 className="h-3.5 w-3.5" />
            </button>
          )}
          {!s.shared && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete() }}
              title="Delete run"
              aria-label="Delete run"
              className="rounded-lg p-1 text-zinc-500 transition-all hover:bg-red-500/10 hover:text-red-600 focus-visible:opacity-100 wide:opacity-0 wide:group-hover:opacity-100 dark:hover:text-red-400"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
      {s.shared && (
        <div className="mt-1.5 flex items-center gap-1 pl-[22px] text-[11px] text-sky-700 dark:text-sky-300" title={`Shared by ${s.owner_name || s.owner_email} · ${s.access}`}>
          <Users className="h-3 w-3" /> shared · {s.access}
        </div>
      )}
      <div className="mt-2 flex min-w-0 items-center gap-1.5 pl-[22px] text-[11px] text-zinc-500">
        {showMode && (
          <Pill tone={vatra ? 'violet' : 'sky'} icon={vatra ? Flame : Network} title={vatra ? 'Collaborative (Vatra) run' : 'Ensemble (Basna) run'}>
            {vatra ? 'Vatra' : 'Basna'}
          </Pill>
        )}
        {origin && (
          <Pill tone="violet" icon={Sparkles} title={`Started by an agent${origin !== 'agent' ? ` from ${origin}` : ''}`}>
            agent{origin !== 'agent' ? `·${origin}` : ''}
          </Pill>
        )}
        {pill && <Pill tone={pill.tone}>{pill.label}</Pill>}
        {s.status === 'done' && s.confidence > 0 && (
          <span className={`shrink-0 font-medium tabular-nums ${confColor(s.confidence)}`} title="Confidence">{Math.round(s.confidence * 100)}%</span>
        )}
        {s.domain && <span className="min-w-0 truncate">{s.domain}</span>}
        {s.difficulty && <span className={`shrink-0 font-medium ${DIFFICULTY_COLOR[s.difficulty] || 'text-zinc-400'}`}>{s.difficulty}</span>}
        <span className="ml-auto shrink-0 tabular-nums text-zinc-600">{timeAgo(s.updated_at || s.created_at)}</span>
      </div>
    </div>
  )
}

// Bucket a run's timestamp into a display group.
function dayGroup(iso?: string): string {
  if (!iso) return 'Earlier'
  const t = new Date(iso)
  if (isNaN(t.getTime())) return 'Earlier'
  const now = new Date()
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const diffDays = Math.floor((startOfDay(now) - startOfDay(t)) / 86400000)
  if (diffDays <= 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return 'This week'
  return 'Earlier'
}

const GROUP_ORDER = ['Today', 'Yesterday', 'This week', 'Earlier']

export function RunsSidebar({ mode: pageMode, sessions, activeId, onSelect, onDelete, onCancel, onNew, onWizard, onShare }: {
  /** The page is locked to one mode: no Basna/Vatra filters or marks. */
  mode?: 'basna' | 'vatra'
  sessions: BasnaSession[]
  activeId?: string
  onSelect: (id: string) => void
  onDelete: (s: BasnaSession) => void
  onCancel: (id: string) => void
  onNew: () => void
  onWizard: () => void
  onShare?: (s: BasnaSession) => void
}) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<ModeFilter>('all')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem('basna.runsCollapsed') === '1' } catch { return false }
  })
  const toggleCollapsed = (v: boolean) => {
    setCollapsed(v)
    try { localStorage.setItem('basna.runsCollapsed', v ? '1' : '0') } catch { /* ignore */ }
  }

  const hasAgentRuns = sessions.some((s) => agentOrigin(s.config))
  const runningCount = sessions.filter((s) => ACTIVE_STATUSES.includes(s.status)).length

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return sessions.filter((s) => {
      if (mode === 'basna' && isVatra(s.config)) return false
      if (mode === 'vatra' && !isVatra(s.config)) return false
      if (mode === 'agent' && !agentOrigin(s.config)) return false
      if (status === 'running' && !ACTIVE_STATUSES.includes(s.status)) return false
      if (status === 'done' && s.status !== 'done') return false
      if (status === 'error' && s.status !== 'error') return false
      if (q && !`${s.title} ${s.intent} ${s.domain}`.toLowerCase().includes(q)) return false
      return true
    })
  }, [sessions, query, mode, status])

  // Group the filtered runs by day bucket, preserving the store's order inside each.
  const groups = useMemo(() => {
    const by = new Map<string, BasnaSession[]>()
    for (const s of visible) {
      const g = dayGroup(s.updated_at || s.created_at)
      const arr = by.get(g) || []
      arr.push(s)
      by.set(g, arr)
    }
    return GROUP_ORDER.filter((g) => by.has(g)).map((g) => ({ label: g, items: by.get(g) as BasnaSession[] }))
  }, [visible])

  const chipCls = (on: boolean) =>
    `flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors ${
      on ? 'border-zinc-600 bg-zinc-800 text-zinc-100' : 'border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-200'}`

  const modeChip = (id: ModeFilter, label: React.ReactNode, title: string) => (
    <button
      key={id}
      onClick={() => setMode((m) => (m === id ? 'all' : id))}
      title={title}
      aria-pressed={mode === id}
      className={chipCls(mode === id)}
    >
      {label}
    </button>
  )

  const statusChip = (id: StatusFilter, label: string, cls: string) => (
    <button
      key={id}
      onClick={() => setStatus((s) => (s === id ? 'all' : id))}
      title={`Show only ${label} runs`}
      aria-pressed={status === id}
      className={chipCls(status === id)}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${cls}`} />
      {label}
      {id === 'running' && runningCount > 0 && <span className="tabular-nums">{runningCount}</span>}
    </button>
  )

  // Collapsed: a thin rail with expand + new + a live-run count, reclaiming the
  // width for the workspace (the flex-1 detail pane grows automatically). On a
  // phone, a bar across the top instead.
  if (collapsed) {
    return (
      <div className="flex w-full shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-2.5 @3xl:w-12 @3xl:flex-col @3xl:gap-3 @3xl:border-b-0 @3xl:border-r @3xl:px-0 @3xl:py-4">
        <button
          onClick={() => toggleCollapsed(false)}
          title="Expand runs"
          aria-label="Expand runs"
          className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          <PanelLeftOpen className="h-4 w-4" />
        </button>
        <button
          onClick={onNew}
          title="New run"
          aria-label="New run"
          className="rounded-lg border border-zinc-700 bg-zinc-900/60 p-1.5 text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800"
        >
          <Plus className="h-4 w-4" />
        </button>
        {runningCount > 0 && (
          <span
            title={`${runningCount} running`}
            className="flex items-center gap-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[11px] font-semibold text-amber-700 dark:text-amber-300"
          >
            <Loader2 className="h-3 w-3 animate-spin" />{runningCount}
          </span>
        )}
        <button
          onClick={() => toggleCollapsed(false)}
          title="Expand runs"
          className="ml-auto text-[11px] font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300 @3xl:ml-0 @3xl:mt-1 @3xl:rotate-180 @3xl:[writing-mode:vertical-rl]"
        >
          Runs · {sessions.length}
        </button>
      </div>
    )
  }

  return (
    <div className="flex w-full shrink-0 flex-col border-b border-zinc-800 @3xl:w-72 @3xl:overflow-hidden @3xl:border-b-0 @3xl:border-r @5xl:w-80 @7xl:w-96">
      {/* Pinned actions: start a new run from the list itself. */}
      <div className="flex items-center gap-2 px-4 pb-2 pt-4">
        <button
          onClick={() => toggleCollapsed(true)}
          title="Collapse runs"
          aria-label="Collapse runs"
          className="-ml-1.5 rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
        <span className="text-sm font-semibold text-zinc-200">Runs</span>
        {visible.length > 0 && (
          <span className="rounded-full bg-zinc-800 px-1.5 text-[11px] font-medium tabular-nums text-zinc-400">{visible.length}</span>
        )}
        <button
          onClick={onWizard}
          title={pageMode ? 'Guided setup — recommends options for your task' : 'Guided setup — recommends Basna vs Vatra and options for your task'}
          className="ml-auto inline-flex items-center gap-1.5 rounded-xl bg-violet-600 px-2.5 py-1.5 text-xs font-medium text-white shadow-sm shadow-violet-900/20 transition-colors hover:bg-violet-500"
        >
          <Wand2 className="h-3.5 w-3.5" /> Wizard
        </button>
        <button
          onClick={onNew}
          className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-2.5 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800"
        >
          <Plus className="h-3.5 w-3.5" /> New
        </button>
      </div>

      {/* Search + filters */}
      <div className="space-y-2 px-4 pb-2 pt-1">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search runs"
            className="w-full rounded-xl border border-zinc-800 bg-zinc-950 py-1.5 pl-8 pr-7 text-sm text-zinc-100 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15"
          />
          {query && (
            <button onClick={() => setQuery('')} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-200">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {!pageMode && modeChip('basna', <><Network className="h-3 w-3" /> Basna</>, 'Show only Basna (ensemble) runs')}
          {!pageMode && modeChip('vatra', <><Flame className="h-3 w-3" /> Vatra</>, 'Show only Vatra (collaborative) runs')}
          {hasAgentRuns && modeChip('agent', <><Sparkles className="h-3 w-3" /> Agent</>, 'Show only agent-started runs')}
          {(!pageMode || hasAgentRuns) && <span className="mx-0.5 h-3.5 w-px bg-zinc-800" />}
          {statusChip('running', 'live', 'bg-amber-400')}
          {statusChip('done', 'done', 'bg-emerald-500')}
          {statusChip('error', 'failed', 'bg-rose-500')}
        </div>
      </div>

      {/* Day-grouped list */}
      <div className="max-h-80 space-y-3 overflow-auto px-2 pb-4 pt-1 @3xl:max-h-none @3xl:flex-1">
        {visible.length === 0 && (
          <p className="px-2 py-3 text-xs leading-relaxed text-zinc-500">
            {sessions.length === 0 ? 'No runs yet — describe a task and run a team.' : 'No runs match the filters.'}
          </p>
        )}
        {groups.map((g) => (
          <div key={g.label} className="space-y-0.5">
            <div className="sticky top-0 z-10 bg-zinc-950/90 px-2 pb-1 pt-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500 backdrop-blur">
              {g.label}
            </div>
            {g.items.map((s) => (
              <SessionCard
                key={s.id}
                s={s}
                active={activeId === s.id}
                showMode={!pageMode}
                onOpen={() => onSelect(s.id)}
                onDelete={() => onDelete(s)}
                onCancel={() => onCancel(s.id)}
                onShare={onShare}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
