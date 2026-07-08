import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle, Check, ChevronDown, ChevronRight, Clock, Download, Loader2,
  Network, Sparkles, Square, Users, Wrench, Coins,
} from 'lucide-react'
import type { BasnaSession, ProgressEvent, VatraSubtask, RunCost } from '../../stores/basnaStore'
import { VatraBlackboard } from '../VatraDelegation'
import { downloadMarkdown, formatProgress, fmtTok, type LiveAgent } from './shared'
import { RunFilesPanel, RunDatastorePanel } from './RunArtifacts'

// ── Run workspace: live header + two-column board/agents/progress layout ────

// One live agent card — its status, running token usage, and streaming activity.
function LiveAgentCard({ a, onSkip }: { a: LiveAgent; onSkip?: (role: string) => void }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {a.done
            ? (a.ok === false
                ? <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                : <Check className="h-3.5 w-3.5 shrink-0 text-emerald-500" />)
            : <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-sky-400" />}
          <span className="truncate text-sm font-medium text-zinc-200">{a.role}</span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {a.usage && (
            <span className="font-mono text-[10px] tabular-nums text-zinc-500" title="LLM tokens (prompt → completion)">
              {fmtTok(a.usage.prompt_tokens)}→{fmtTok(a.usage.completion_tokens)} tok
            </span>
          )}
          {onSkip && !a.done && (
            <button
              onClick={() => onSkip(a.role)}
              title="Skip this agent — cancel its current turn and move on"
              className="rounded p-0.5 text-zinc-500 transition-colors hover:text-rose-400"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          )}
        </div>
      </div>
      <div className="mt-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-600">
        Activity ({a.actions.length})
      </div>
      <div className="mt-1 max-h-48 space-y-0.5 overflow-auto">
        {a.actions.slice(-14).map((ev) => (
          <div key={ev.i} className="flex items-baseline gap-2 text-[11px]">
            {ev.stage === 'narration'
              ? <Sparkles className="h-3 w-3 shrink-0 text-zinc-500" />
              : <Wrench className="h-3 w-3 shrink-0 text-zinc-600" />}
            <span className="shrink-0 font-mono text-zinc-400">{ev.tool}</span>
            {ev.detail && <span className="truncate text-zinc-600">{ev.detail}</span>}
          </div>
        ))}
        {a.actions.length === 0 && <div className="text-[11px] text-zinc-600">working…</div>}
      </div>
    </div>
  )
}

// A responsive grid of live agent cards (Basna's main column).
function LiveAgentGrid({ agents, onSkip }: { agents: LiveAgent[]; onSkip?: (role: string) => void }) {
  return (
    <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
      {agents.map((a) => <LiveAgentCard key={a.role} a={a} onSkip={onSkip} />)}
    </div>
  )
}

// Live per-agent panels, sectioned by execution group when the run has them.
export function LiveAgentsPanel({ agents, onSkip }: { agents: LiveAgent[]; onSkip?: (role: string) => void }) {
  const letters = useMemo(
    () => [...new Set(agents.map((a) => a.group).filter(Boolean) as string[])].sort(),
    [agents],
  )
  if (letters.length === 0) return <LiveAgentGrid agents={agents} onSkip={onSkip} />
  return (
    <div className="space-y-3">
      {letters.map((letter) => {
        const inGroup = agents.filter((a) => a.group === letter)
        const working = inGroup.some((a) => !a.done)
        return (
          <div key={letter}>
            <div className="mb-1.5 flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded bg-sky-500/15 text-[11px] font-bold text-sky-300">{letter}</span>
              <span className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Group {letter}</span>
              <span className="text-[10px] text-zinc-600">{inGroup.length} agent(s)</span>
              {working
                ? <Loader2 className="h-3 w-3 animate-spin text-sky-400" />
                : <Check className="h-3 w-3 text-emerald-500" />}
            </div>
            <LiveAgentGrid agents={inGroup} onSkip={onSkip} />
          </div>
        )
      })}
    </div>
  )
}

// Compact agent roster for the rail (Vatra): grouped rows, expandable to the
// agent's live activity feed.
function AgentRoster({ agents, onSkip }: { agents: LiveAgent[]; onSkip?: (role: string) => void }) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const letters = useMemo(
    () => [...new Set(agents.map((a) => a.group).filter(Boolean) as string[])].sort(),
    [agents],
  )
  const grouped: { label: string | null; items: LiveAgent[] }[] = letters.length
    ? [
        ...letters.map((l) => ({ label: l, items: agents.filter((a) => a.group === l) })),
        ...(agents.some((a) => !a.group) ? [{ label: null, items: agents.filter((a) => !a.group) }] : []),
      ]
    : [{ label: null, items: agents }]

  const row = (a: LiveAgent) => {
    const isOpen = !!open[a.role]
    return (
      <div key={a.role}>
        <div className="flex w-full items-center gap-1.5 py-1 text-left">
          <button
            onClick={() => setOpen((o) => ({ ...o, [a.role]: !o[a.role] }))}
            className="flex min-w-0 flex-1 items-center gap-1.5"
            title={isOpen ? 'Hide activity' : 'Show activity'}
          >
            {a.done
              ? (a.ok === false
                  ? <span className="h-2 w-2 shrink-0 rounded-full bg-rose-500" />
                  : <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />)
              : a.actions.length > 0 || a.usage
                ? <Loader2 className="h-2.5 w-2.5 shrink-0 animate-spin text-sky-400" />
                : <span className="h-2 w-2 shrink-0 rounded-full bg-zinc-600" />}
            <span className="truncate text-xs text-zinc-200">{a.role}</span>
            {isOpen ? <ChevronDown className="h-3 w-3 shrink-0 text-zinc-600" /> : <ChevronRight className="h-3 w-3 shrink-0 text-zinc-600" />}
          </button>
          {a.usage && (
            <span className="shrink-0 font-mono text-[10px] tabular-nums text-zinc-600" title="LLM tokens (total)">
              {fmtTok(a.usage.total_tokens ?? ((a.usage.prompt_tokens || 0) + (a.usage.completion_tokens || 0)))}
            </span>
          )}
          {onSkip && !a.done && (
            <button
              onClick={() => onSkip(a.role)}
              title="Skip this agent — cancel its current turn and move on"
              className="shrink-0 rounded p-0.5 text-zinc-600 transition-colors hover:text-rose-400"
            >
              <Square className="h-3 w-3 fill-current" />
            </button>
          )}
        </div>
        {isOpen && (
          <div className="mb-1 ml-3.5 max-h-40 space-y-0.5 overflow-auto border-l border-zinc-800 pl-2">
            {a.actions.slice(-12).map((ev) => (
              <div key={ev.i} className="flex items-baseline gap-1.5 text-[10px]">
                {ev.stage === 'narration'
                  ? <Sparkles className="h-2.5 w-2.5 shrink-0 text-zinc-500" />
                  : <Wrench className="h-2.5 w-2.5 shrink-0 text-zinc-600" />}
                <span className="shrink-0 font-mono text-zinc-400">{ev.tool}</span>
                {ev.detail && <span className="truncate text-zinc-600">{ev.detail}</span>}
              </div>
            ))}
            {a.actions.length === 0 && <div className="text-[10px] text-zinc-600">working…</div>}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
      <div className="mb-1.5 flex items-center gap-2">
        <Users className="h-3.5 w-3.5 text-zinc-500" />
        <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Agents</span>
        <span className="ml-auto text-[10px] tabular-nums text-zinc-600">{agents.filter((a) => a.done).length}/{agents.length} done</span>
      </div>
      {grouped.map((g) => (
        <div key={g.label ?? '_'}>
          {g.label && (
            <div className="mt-1.5 mb-0.5 text-[9px] font-semibold uppercase tracking-wider text-zinc-600">Group {g.label}</div>
          )}
          {g.items.map(row)}
        </div>
      ))}
      {agents.length === 0 && <p className="text-[11px] text-zinc-600">Waiting for the team to spawn…</p>}
    </div>
  )
}

// Collapsible progress feed. `fill` makes it a tall pane (side-by-side layout)
// rather than the compact rail card.
export function ProgressFeed({ progress, running, defaultOpen = true, fill = false }: {
  progress: ProgressEvent[]; running: boolean; defaultOpen?: boolean; fill?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const ref = useRef<HTMLDivElement>(null)
  // Keep the log pinned to the newest line as events stream in.
  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [progress.length, open])
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
      <div className="flex items-center gap-2">
        <button onClick={() => setOpen((o) => !o)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          {running
            ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-sky-400" />
            : <Check className="h-3.5 w-3.5 shrink-0 text-emerald-500" />}
          <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Progress</span>
          {progress.length > 0 && <span className="text-[10px] tabular-nums text-zinc-600">{progress.length}</span>}
          {open ? <ChevronDown className="h-3 w-3 text-zinc-600" /> : <ChevronRight className="h-3 w-3 text-zinc-600" />}
        </button>
        <button
          onClick={() => downloadMarkdown('basna-progress.md', formatProgress(progress))}
          title="Export progress log"
          className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <Download className="h-3 w-3" />
        </button>
      </div>
      {open && (
        <div ref={ref} className={`mt-2 space-y-1 overflow-auto ${fill ? 'max-h-[62vh] min-h-[320px]' : 'max-h-64'}`}>
          {progress.slice(-40).map((ev) => (
            <div key={ev.i} className="flex items-baseline gap-1.5 text-[11px]">
              <span className="shrink-0 font-mono text-[9px] tabular-nums text-zinc-600">
                {ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''}
              </span>
              <span className={
                ev.stage === 'phase' ? 'text-sky-700 dark:text-sky-300 font-semibold'
                  : ev.stage === 'narration' ? 'text-zinc-200'
                  : ev.ok === false ? 'text-rose-400'
                  : 'text-zinc-500'
              }>{ev.message}</span>
            </div>
          ))}
          {running && progress.length === 0 && (
            <div className="text-[11px] text-zinc-500">Starting…</div>
          )}
        </div>
      )}
    </div>
  )
}

function fmtElapsed(sec: number): string {
  if (sec < 60) return `${sec}s`
  const m = Math.floor(sec / 60)
  if (m < 60) return `${m}m ${sec % 60}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

// Execution-group stepper for the run header: A ✓ → B (1/3) → C.
function GroupStepper({ agents }: { agents: LiveAgent[] }) {
  const letters = useMemo(
    () => [...new Set(agents.map((a) => a.group).filter(Boolean) as string[])].sort(),
    [agents],
  )
  if (letters.length < 2) return null
  // The active group: the first with unfinished agents.
  const activeIdx = letters.findIndex((l) => agents.some((a) => a.group === l && !a.done))
  return (
    <div className="flex flex-wrap items-center gap-1 text-[11px]">
      {letters.map((l, i) => {
        const inGroup = agents.filter((a) => a.group === l)
        const done = inGroup.length > 0 && inGroup.every((a) => a.done)
        const active = i === activeIdx
        return (
          <span key={l} className="flex items-center gap-1">
            {i > 0 && <ChevronRight className="h-3 w-3 text-zinc-600" />}
            <span className={`flex items-center gap-1 rounded px-2 py-0.5 font-medium ${
              done ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
                : active ? 'bg-violet-500/20 text-violet-700 dark:text-violet-300'
                : 'bg-zinc-800/70 text-zinc-500'
            }`}>
              {l}
              {done && <Check className="h-3 w-3" />}
              {active && <span className="tabular-nums">{inGroup.filter((a) => a.done).length}/{inGroup.length}</span>}
            </span>
          </span>
        )
      })}
    </div>
  )
}

// A horizontal two-pane split with a draggable divider. The ratio persists and
// the layout stacks vertically on narrow screens.
export function ResizableSplit({ storageKey, left, right }: {
  storageKey: string; left: React.ReactNode; right: React.ReactNode
}) {
  const [pct, setPct] = useState(() => {
    try { const v = Number(localStorage.getItem(storageKey)); return v >= 20 && v <= 80 ? v : 50 } catch { return 50 }
  })
  const [wide, setWide] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia('(min-width: 1024px)').matches)
  const [dragging, setDragging] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)')
    const h = () => setWide(mq.matches)
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [])
  useEffect(() => { try { localStorage.setItem(storageKey, String(Math.round(pct))) } catch { /* ignore */ } }, [pct, storageKey])

  const onDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setDragging(true)
    const move = (ev: MouseEvent) => {
      const r = ref.current?.getBoundingClientRect()
      if (!r || r.width === 0) return
      setPct(Math.max(20, Math.min(80, ((ev.clientX - r.left) / r.width) * 100)))
    }
    const up = () => {
      setDragging(false)
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  if (!wide) return <div className="space-y-3">{left}{right}</div>
  return (
    <div ref={ref} className={`flex items-stretch ${dragging ? 'select-none' : ''}`}>
      <div style={{ width: `${pct}%` }} className="min-w-0">{left}</div>
      <div
        onMouseDown={onDown}
        title="Drag to resize"
        className="group relative mx-1.5 w-1.5 shrink-0 cursor-col-resize"
      >
        <div className={`absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 rounded ${
          dragging ? 'bg-sky-500' : 'bg-zinc-800 group-hover:bg-sky-600/60'}`} />
      </div>
      <div style={{ width: `${100 - pct}%` }} className="min-w-0">{right}</div>
    </div>
  )
}

export function RunWorkspace({
  session, vatraMode, running, subtasks, liveAgents, progress, currentPhase, runCost, project, onSkip, onStop,
}: {
  session: BasnaSession
  vatraMode: boolean
  running: boolean
  subtasks?: VatraSubtask[]
  liveAgents: LiveAgent[]
  progress: ProgressEvent[]
  currentPhase: string | null
  runCost?: RunCost | null
  project: string
  onSkip?: (role: string) => void
  onStop: () => void
}) {
  // Elapsed clock: first progress timestamp → last (or now, ticking, while live).
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [running])
  const firstTs = progress.find((e) => e.ts)?.ts
  let lastTs: number | undefined
  for (let i = progress.length - 1; i >= 0; i--) { if (progress[i].ts) { lastTs = progress[i].ts; break } }
  const elapsedSec = firstTs
    ? Math.max(0, Math.floor((running ? now / 1000 : (lastTs || firstTs)) - firstTs))
    : 0

  // Live token total across the fleet (cumulative per-agent usage events).
  const totalTokens = liveAgents.reduce(
    (acc, a) => acc + (a.usage?.total_tokens ?? ((a.usage?.prompt_tokens || 0) + (a.usage?.completion_tokens || 0))),
    0,
  )
  const usd = runCost?.usd

  return (
    <div className="space-y-3">
      {/* Run header: what's running, where it is, what it costs so far. */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3.5">
        <div className="flex flex-wrap items-center gap-2">
          {running
            ? <Loader2 className="h-4 w-4 shrink-0 animate-spin text-sky-400" />
            : session.status === 'error'
              ? <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
              : <Check className="h-4 w-4 shrink-0 text-emerald-500" />}
          <span className="min-w-0 truncate text-sm font-semibold text-zinc-100">
            {session.title || session.intent || 'Run'}
          </span>
          <span className={`flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
            vatraMode
              ? 'border-violet-500/25 bg-violet-500/15 text-violet-700 dark:text-violet-300'
              : 'border-sky-500/25 bg-sky-500/15 text-sky-700 dark:text-sky-300'
          }`}>
            {vatraMode ? <Users className="h-2.5 w-2.5" /> : <Network className="h-2.5 w-2.5" />}
            {vatraMode ? 'vatra' : 'basna'}
          </span>
          <div className="ml-auto flex shrink-0 items-center gap-3 text-[11px] text-zinc-500">
            {elapsedSec > 0 && (
              <span className="flex items-center gap-1 tabular-nums" title="Elapsed time">
                <Clock className="h-3 w-3" /> {fmtElapsed(elapsedSec)}
              </span>
            )}
            {totalTokens > 0 && (
              <span className="flex items-center gap-1 font-mono tabular-nums" title="LLM tokens so far (fleet total)">
                <Coins className="h-3 w-3" /> {fmtTok(totalTokens)}{typeof usd === 'number' ? ` · $${usd.toFixed(2)}` : ''}
              </span>
            )}
            {running && (
              <button
                onClick={onStop}
                title="Stop this run"
                className="flex items-center gap-1.5 rounded-lg border border-rose-500/40 px-2.5 py-1 text-[11px] font-medium text-rose-600 hover:bg-rose-500/10 dark:text-rose-400"
              >
                <Square className="h-3 w-3 fill-current" /> Stop
              </button>
            )}
          </div>
        </div>
        {(currentPhase || liveAgents.some((a) => a.group)) && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <GroupStepper agents={liveAgents} />
            {currentPhase && (
              <span className="inline-flex items-center gap-1 rounded-full border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-700 dark:text-sky-300">
                {running && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
                {currentPhase}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Workspace: Progress ⇔ board/agents (resizable 50:50), + a rail for
          the roster and artifacts. */}
      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0">
          <ResizableSplit
            storageKey="basna.runSplit"
            left={<ProgressFeed progress={progress} running={running} fill />}
            right={vatraMode
              ? <VatraBlackboard sessionId={session.id} subtasks={subtasks} active={running} />
              : (liveAgents.length > 0
                  ? <LiveAgentsPanel agents={liveAgents} onSkip={onSkip} />
                  : <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 text-xs text-zinc-500">Spawning the team…</div>)}
          />
        </div>
        <div className="space-y-3">
          {vatraMode && <AgentRoster agents={liveAgents} onSkip={onSkip} />}
          {project && <RunFilesPanel project={project} live={running} />}
          {project && <RunDatastorePanel project={project} live={running} hideWhenEmpty />}
        </div>
      </div>
    </div>
  )
}
