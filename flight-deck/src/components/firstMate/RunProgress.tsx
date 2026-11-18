import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Bot, Check, ChevronDown, Circle, FolderOpen, Loader2, Minus, X } from 'lucide-react'
import { getRunProgress, type RunProgress as Progress } from '../../services/firstMate'
import type { ProgressEvent } from '../../stores/basnaStore'
import { RunFiles } from './RunFiles'
import { errText } from './ui'

// What a run is doing, inside its row on the Runs tab: the plan's steps as they
// go, then, folded, the agents that worked on it (one per step; Bat removes
// each when its step ends), the latest activity, and its files. Reads again
// every few seconds while the run is going.

const POLL_MS = 4000
const KEEP = 400

type StepState = 'done' | 'running' | 'failed' | 'skipped' | 'pending'

function stateOf(status: string): StepState {
  if (status === 'done') return 'done'
  if (status === 'running' || status === 'retrying') return 'running'
  if (status === 'failed' || status === 'error') return 'failed'
  if (status === 'skipped') return 'skipped'
  return 'pending'
}

function StateIcon({ state }: { state: StepState }) {
  if (state === 'done') return <Check className="h-3.5 w-3.5 shrink-0 text-emerald-500" aria-label="Done" />
  if (state === 'running') return <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-violet-500" aria-label="Working" />
  if (state === 'failed') return <X className="h-3.5 w-3.5 shrink-0 text-red-500" aria-label="Failed" />
  if (state === 'skipped') return <Minus className="h-3.5 w-3.5 shrink-0 text-zinc-500" aria-label="Skipped" />
  return <Circle className="h-3 w-3 shrink-0 text-zinc-600" aria-label="Not started" />
}

function clock(ts?: number): string {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''
}

/** One line for an event: what was done, in words. */
function line(e: ProgressEvent): string {
  if (e.stage === 'action') return e.detail ? `${e.message} · ${e.detail}` : e.message
  if (e.stage === 'narration') return e.detail || e.message
  return e.message
}

function Fold({ icon: Icon, label, count, children }: {
  icon: typeof Bot; label: string; count?: number; children: () => React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-t border-zinc-800/80 pt-2">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg px-1 py-1 text-left text-xs font-medium text-zinc-400 transition-colors hover:text-zinc-200">
        <Icon className="h-3.5 w-3.5 shrink-0" />
        <span>{label}</span>
        {count !== undefined && <span className="text-zinc-500">{count}</span>}
        <ChevronDown className={`ml-auto h-3.5 w-3.5 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <div className="mt-1.5">{children()}</div>}
    </div>
  )
}

interface Agent { key: string; title: string; state: StepState; name: string; now: boolean; actions: ProgressEvent[] }

function AgentRow({ a }: { a: Agent }) {
  const [open, setOpen] = useState(false)
  const last = a.actions[a.actions.length - 1]
  return (
    <li>
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-start gap-2 rounded-lg px-1.5 py-1.5 text-left transition-colors hover:bg-[var(--fd-hover)]">
        <span className="mt-0.5"><StateIcon state={a.now ? 'running' : a.state} /></span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-[13px] text-zinc-200" title={a.title}>{a.title}</span>
            {a.now && <span className="shrink-0 rounded-full bg-violet-500/15 px-1.5 text-[10px] font-medium text-violet-700 dark:text-violet-300">working now</span>}
          </span>
          <span className="block truncate text-[11px] text-zinc-500">
            {a.name ? `${a.name} · ` : ''}{a.actions.length} {a.actions.length === 1 ? 'action' : 'actions'}
            {last ? ` · last: ${line(last)}` : ''}
          </span>
        </span>
        <ChevronDown className={`mt-1 h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        a.actions.length ? (
          <ol className="mb-1 ml-7 max-h-56 space-y-0.5 overflow-y-auto pr-1 font-mono text-[11px] leading-relaxed text-zinc-400">
            {a.actions.slice(-40).map((e) => (
              <li key={e.i} className="flex gap-2">
                <span className="shrink-0 text-zinc-600">{clock(e.ts)}</span>
                <span className="min-w-0 break-words">{line(e)}</span>
              </li>
            ))}
          </ol>
        ) : <p className="mb-1 ml-7 text-[11px] text-zinc-500">Nothing recorded yet.</p>
      )}
    </li>
  )
}

export function RunProgress({ runId, going }: { runId: string; going: boolean }) {
  const [p, setP] = useState<Progress | null>(null)
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [error, setError] = useState('')
  const next = useRef(0)

  useEffect(() => {
    let stop = false
    next.current = 0
    setEvents([])
    const load = async () => {
      try {
        const got = await getRunProgress(runId, next.current)
        if (stop) return
        setP(got)
        setError('')
        next.current = got.next
        if (got.events.length) {
          setEvents((old) => {
            const seen = new Set(old.map((e) => e.i))
            return [...old, ...got.events.filter((e) => !seen.has(e.i))].slice(-KEEP)
          })
        }
      } catch (e) {
        if (!stop) setError(errText(e))
      }
    }
    void load()
    if (!going) return () => { stop = true }
    const t = setInterval(load, POLL_MS)
    return () => { stop = true; clearInterval(t) }
  }, [runId, going])

  const agents = useMemo<Agent[]>(() => {
    if (!p) return []
    const steps = new Map(p.steps.map((s) => [s.key, s]))
    const order: string[] = []
    const acts = new Map<string, ProgressEvent[]>()
    for (const e of events) {
      if (!e.agent) continue
      if (!acts.has(e.agent)) { acts.set(e.agent, []); order.push(e.agent) }
      if (e.stage === 'action' || e.stage === 'narration') acts.get(e.agent)!.push(e)
    }
    for (const w of p.workers) if (!acts.has(w.step)) { acts.set(w.step, []); order.push(w.step) }
    return order.map((key) => {
      const step = steps.get(key)
      const worker = p.workers.find((w) => w.step === key)
      return {
        key, title: step?.title || key, state: stateOf(step?.status || ''),
        name: worker?.name || '', now: !!worker?.running, actions: acts.get(key) || [],
      }
    })
  }, [p, events])

  if (error && !p) return <p className="mt-2 text-xs text-red-600 dark:text-red-300">{error}</p>
  if (!p) return <p className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Loading its progress…</p>
  if (!p.available) {
    return <p className="mt-2 text-xs text-zinc-500">{going ? 'It hasn’t started working yet.' : 'There’s no progress to show for this run.'}</p>
  }

  const done = p.steps.filter((s) => s.status === 'done').length
  const current = p.steps.find((s) => stateOf(s.status) === 'running')
  const activity = events.filter((e) => e.stage !== 'usage')

  return (
    <div className="mt-2.5 space-y-2 rounded-xl bg-[var(--fd-hover)] px-3 py-2.5">
      {p.steps.length === 0 ? (
        <p className="text-xs text-zinc-500">{going ? 'Planning its steps…' : 'It had no steps.'}</p>
      ) : (
        <div>
          <div className="flex items-center gap-2 text-xs text-zinc-400">
            <span className="font-medium text-zinc-300">{done} of {p.steps.length} steps done</span>
            {current && <span className="min-w-0 truncate">· now: {current.title}</span>}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-zinc-800" role="progressbar"
            aria-valuemin={0} aria-valuemax={p.steps.length} aria-valuenow={done} aria-label="Steps done">
            <div className="h-full rounded-full bg-violet-500 transition-[width] duration-500"
              style={{ width: `${Math.round((done / p.steps.length) * 100)}%` }} />
          </div>
          <ol className="mt-2 space-y-1">
            {p.steps.map((s) => (
              <li key={s.key} className="flex items-start gap-2 text-[13px]">
                <span className="mt-0.5"><StateIcon state={stateOf(s.status)} /></span>
                <span className="min-w-0 flex-1">
                  <span className={stateOf(s.status) === 'pending' ? 'text-zinc-500' : 'text-zinc-200'}>{s.title}</span>
                  {s.attempt > 1 && <span className="ml-1.5 text-[11px] text-zinc-500">try {s.attempt}</span>}
                  {s.error && stateOf(s.status) === 'failed' && (
                    <span className="mt-0.5 block text-[11px] text-red-600 dark:text-red-300">{s.error}</span>
                  )}
                  {s.note && (
                    <span className="mt-1 block max-h-24 overflow-hidden whitespace-pre-wrap rounded-lg border-l-2 border-violet-500/40 pl-2 text-[11px] leading-relaxed text-zinc-400"
                      title="Its progress note: where its worker got to">
                      {s.note}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
      <Fold icon={Bot} label="Agents" count={agents.length}>
        {() => agents.length
          ? <ul className="space-y-0.5">{agents.map((a) => <AgentRow key={a.key} a={a} />)}</ul>
          : <p className="px-1 text-xs text-zinc-500">No agent has worked on it yet.</p>}
      </Fold>
      <Fold icon={Activity} label="Activity" count={activity.length}>
        {() => activity.length ? (
          <ol className="max-h-64 space-y-0.5 overflow-y-auto px-1 font-mono text-[11px] leading-relaxed text-zinc-400">
            {activity.slice(-80).map((e) => (
              <li key={e.i} className="flex gap-2">
                <span className="shrink-0 text-zinc-600">{clock(e.ts)}</span>
                <span className="min-w-0 break-words">{line(e)}</span>
              </li>
            ))}
          </ol>
        ) : <p className="px-1 text-xs text-zinc-500">Nothing yet.</p>}
      </Fold>
      {p.files && (
        <Fold icon={FolderOpen} label="Files and data">
          {() => <RunFiles project={p.files} live={going} />}
        </Fold>
      )}
    </div>
  )
}
