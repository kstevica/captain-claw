import { useMemo, useState } from 'react'
import { Loader2, Plus, Search, Sparkles, Square, Trash2, Users, Wand2, X, Network, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import type { BasnaSession } from '../../stores/basnaStore'
import { agentOrigin, confColor, DIFFICULTY_COLOR, isVatra, STATUS_DOT, timeAgo } from './shared'

// ── Runs sidebar: search + filters + day-grouped run list ───────────────────

type ModeFilter = 'all' | 'basna' | 'vatra' | 'agent'
type StatusFilter = 'all' | 'running' | 'done' | 'error'

const ACTIVE_STATUSES = ['routing', 'routed', 'running']

function SessionCard({ s, active, onOpen, onDelete, onCancel }: {
  s: BasnaSession; active: boolean; onOpen: () => void; onDelete: () => void; onCancel: () => void
}) {
  // Only show the spinner + stop control while genuinely active. 'routed' is the
  // idle prepared state (Basna routed, or a Vatra plan awaiting Run) — not running.
  const working = s.status === 'running' || s.status === 'routing'
  const origin = agentOrigin(s.config)
  return (
    <div
      onClick={onOpen}
      className={`group relative cursor-pointer rounded-lg border p-2.5 pl-3 transition-colors ${
        active ? 'border-sky-600/60 bg-sky-950/30' : 'border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/50'
      }`}
    >
      {active && <span className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-sky-500" />}
      <div className="flex items-start gap-2">
        <span className="mt-0.5 shrink-0">
          {working
            ? <Loader2 className="h-3 w-3 animate-spin text-amber-400" />
            : <span className={`block h-2 w-2 rounded-full ${STATUS_DOT[s.status] || 'bg-zinc-500'}`} />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-1 text-xs font-medium leading-snug text-zinc-200" title={s.intent || ''}>{s.title || s.intent || '(untitled)'}</p>
          {s.title && s.intent && (
            <p className="mt-0.5 line-clamp-1 text-[10px] leading-snug text-zinc-500" title={s.intent}>{s.intent}</p>
          )}
        </div>
        {working && (
          <button
            onClick={(e) => { e.stopPropagation(); onCancel() }}
            title="Stop this run"
            className="shrink-0 rounded p-0.5 text-amber-500 transition-colors hover:text-rose-400"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="shrink-0 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 pl-5 text-[10px] text-zinc-500">
        {origin && (
          <span
            title={`Started by an agent${origin !== 'agent' ? ` from ${origin}` : ''}`}
            className="flex shrink-0 items-center gap-1 rounded bg-violet-500/15 border border-violet-500/25 px-1.5 py-0.5 font-medium text-violet-700 dark:text-violet-300"
          >
            <Sparkles className="h-2.5 w-2.5" />agent{origin !== 'agent' ? `·${origin}` : ''}
          </span>
        )}
        {isVatra(s.config) && (
          <span title="Collaborative (Vatra) run" className="flex shrink-0 items-center gap-1 rounded border border-violet-500/25 bg-violet-500/15 px-1.5 py-0.5 font-medium text-violet-700 dark:text-violet-300">
            <Users className="h-2.5 w-2.5" />vatra
          </span>
        )}
        {s.domain && <span className="truncate rounded bg-zinc-800/70 px-1.5 py-0.5 text-zinc-400">{s.domain}</span>}
        {s.difficulty && <span className={`font-medium ${DIFFICULTY_COLOR[s.difficulty] || 'text-zinc-400'}`}>{s.difficulty}</span>}
        {s.status === 'done' && s.confidence > 0 && (
          <span className={`font-medium ${confColor(s.confidence)}`}>· {Math.round(s.confidence * 100)}%</span>
        )}
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

export function RunsSidebar({ sessions, activeId, onSelect, onDelete, onCancel, onNew, onWizard }: {
  sessions: BasnaSession[]
  activeId?: string
  onSelect: (id: string) => void
  onDelete: (s: BasnaSession) => void
  onCancel: (id: string) => void
  onNew: () => void
  onWizard: () => void
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

  const modeChip = (id: ModeFilter, label: React.ReactNode, title: string) => (
    <button
      key={id}
      onClick={() => setMode((m) => (m === id ? 'all' : id))}
      title={title}
      className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium border transition-colors ${
        mode === id
          ? 'border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-300'
          : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {label}
    </button>
  )

  const statusChip = (id: StatusFilter, label: string, cls: string) => (
    <button
      key={id}
      onClick={() => setStatus((s) => (s === id ? 'all' : id))}
      title={`Show only ${label} runs`}
      className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium border transition-colors ${
        status === id
          ? 'border-zinc-500 bg-zinc-700/60 text-zinc-200'
          : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${cls}`} />
      {label}
      {id === 'running' && runningCount > 0 && <span className="tabular-nums">{runningCount}</span>}
    </button>
  )

  // Collapsed: a thin rail with expand + new + a live-run count, reclaiming the
  // width for the workspace (the flex-1 detail pane grows automatically).
  if (collapsed) {
    return (
      <div className="flex w-11 shrink-0 flex-col items-center gap-3 border-r border-zinc-800 py-3">
        <button
          onClick={() => toggleCollapsed(false)}
          title="Expand runs"
          className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          <PanelLeftOpen className="h-4 w-4" />
        </button>
        <button
          onClick={onNew}
          title="New run"
          className="rounded-lg border border-zinc-700 p-1.5 text-zinc-300 transition-colors hover:bg-zinc-800"
        >
          <Plus className="h-4 w-4" />
        </button>
        {runningCount > 0 && (
          <span
            title={`${runningCount} running`}
            className="flex items-center gap-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-amber-500"
          >
            <Loader2 className="h-2.5 w-2.5 animate-spin" />{runningCount}
          </span>
        )}
        <button
          onClick={() => toggleCollapsed(false)}
          title="Expand runs"
          className="mt-1 [writing-mode:vertical-rl] rotate-180 text-[10px] font-semibold uppercase tracking-wide text-zinc-600 hover:text-zinc-400"
        >
          Runs · {sessions.length}
        </button>
      </div>
    )
  }

  return (
    <div className="flex w-80 shrink-0 flex-col overflow-hidden border-r border-zinc-800 lg:w-96">
      {/* Pinned actions: start a new run from the list itself. */}
      <div className="flex items-center gap-2 border-b border-zinc-800/70 px-3 py-2.5">
        <button
          onClick={() => toggleCollapsed(true)}
          title="Collapse runs"
          className="rounded p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
        >
          <PanelLeftClose className="h-3.5 w-3.5" />
        </button>
        <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Runs</span>
        {visible.length > 0 && <span className="text-[10px] tabular-nums text-zinc-600">{visible.length}</span>}
        <button
          onClick={onWizard}
          title="Guided setup — recommends Basna vs Vatra and options for your task"
          className="ml-auto flex items-center gap-1.5 rounded-lg bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500"
        >
          <Wand2 className="h-3 w-3" /> Wizard
        </button>
        <button
          onClick={onNew}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-[11px] font-medium text-zinc-200 hover:bg-zinc-800"
        >
          <Plus className="h-3 w-3" /> New
        </button>
      </div>

      {/* Search + filters */}
      <div className="space-y-1.5 px-3 pt-2.5 pb-1.5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search runs"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 py-1 pl-7 pr-6 text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
          />
          {query && (
            <button onClick={() => setQuery('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-600 hover:text-zinc-300">
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {modeChip('basna', <><Network className="h-2.5 w-2.5" /> basna</>, 'Show only Basna (ensemble) runs')}
          {modeChip('vatra', <><Users className="h-2.5 w-2.5" /> vatra</>, 'Show only Vatra (collaborative) runs')}
          {hasAgentRuns && modeChip('agent', <><Sparkles className="h-2.5 w-2.5" /> agent</>, 'Show only agent-started runs')}
          <span className="mx-0.5 h-3 w-px bg-zinc-800" />
          {statusChip('running', 'live', 'bg-amber-400')}
          {statusChip('done', 'done', 'bg-emerald-500')}
          {statusChip('error', 'failed', 'bg-rose-500')}
        </div>
      </div>

      {/* Day-grouped list */}
      <div className="flex-1 space-y-1.5 overflow-auto px-3 pb-3 pt-1">
        {visible.length === 0 && (
          <p className="px-1 py-2 text-xs text-zinc-600">
            {sessions.length === 0 ? 'No runs yet — describe a task and run a team.' : 'No runs match the filters.'}
          </p>
        )}
        {groups.map((g) => (
          <div key={g.label} className="space-y-1.5">
            <div className="sticky top-0 z-10 -mx-1 bg-zinc-950/90 px-1 pt-1.5 pb-0.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-600 backdrop-blur">
              {g.label}
            </div>
            {g.items.map((s) => (
              <SessionCard
                key={s.id}
                s={s}
                active={activeId === s.id}
                onOpen={() => onSelect(s.id)}
                onDelete={() => onDelete(s)}
                onCancel={() => onCancel(s.id)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
