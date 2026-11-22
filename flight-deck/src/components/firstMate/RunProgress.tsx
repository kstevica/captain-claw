import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Activity, Bot, Check, ChevronDown, Database, FolderOpen, ListChecks, Loader2, Minus, NotebookPen, X, type LucideIcon } from 'lucide-react'
import { getRunProgress, type RunProgress as Progress } from '../../services/firstMate'
import type { ProgressEvent } from '../../stores/basnaStore'
import { RunFiles, RunNotes } from './RunFiles'
import { useRunFolder } from './useRunFolder'
import { TimingChips } from './FactoryResults'
import { RunDataList } from './FactoryData'
import { useRunData } from './useRunData'
import { NoteText } from './RunResult'
import { FunnelStrip } from './FunnelStrip'
import { Pill, errText, type PillTone } from './ui'

// What a run is doing, inside its row on the Runs tab: how far it got (a bar with one
// segment per step), then tabs across the whole width: its steps with their progress
// notes, the agents that worked on it (one per step; Bat removes each when its step
// ends), the latest activity, its files, the workers' notes, and the extra data it was
// given. Reads again
// every few seconds while the run is going.

const POLL_MS = 4000
const KEEP = 400

type StepState = 'done' | 'running' | 'failed' | 'skipped' | 'pending'
type Tab = 'steps' | 'agents' | 'activity' | 'files' | 'notes' | 'data'

function stateOf(status: string): StepState {
  if (status === 'done') return 'done'
  if (status === 'running' || status === 'retrying') return 'running'
  if (status === 'failed' || status === 'error') return 'failed'
  if (status === 'skipped') return 'skipped'
  return 'pending'
}

const STATE: Record<StepState, { label: string; tone: PillTone; badge: string; bar: string }> = {
  done: { label: 'Done', tone: 'green', badge: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400', bar: 'bg-emerald-500' },
  running: { label: 'Working', tone: 'violet', badge: 'bg-violet-500/15 text-violet-600 dark:text-violet-300', bar: 'bg-violet-500 animate-pulse' },
  failed: { label: 'Failed', tone: 'red', badge: 'bg-red-500/12 text-red-600 dark:text-red-300', bar: 'bg-red-500' },
  skipped: { label: 'Skipped', tone: 'zinc', badge: 'bg-zinc-500/10 text-zinc-500', bar: 'bg-zinc-600' },
  pending: { label: 'Not started', tone: 'zinc', badge: 'bg-zinc-500/15 text-zinc-500', bar: 'bg-zinc-500/25' },
}

function Badge({ state, n }: { state: StepState; n: number }) {
  const cls = `flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold tabular-nums ${STATE[state].badge}`
  if (state === 'done') return <span className={cls} aria-label="Done"><Check className="h-3.5 w-3.5" strokeWidth={2.5} /></span>
  if (state === 'running') return <span className={cls} aria-label="Working"><Loader2 className="h-3.5 w-3.5 animate-spin" /></span>
  if (state === 'failed') return <span className={cls} aria-label="Failed"><X className="h-3.5 w-3.5" strokeWidth={2.5} /></span>
  if (state === 'skipped') return <span className={cls} aria-label="Skipped"><Minus className="h-3.5 w-3.5" /></span>
  return <span className={cls} aria-label="Not started">{n}</span>
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

/** "step-3" → "Step 3"; anything else as it is. */
function stepName(key: string): string {
  const m = /^step-(\d+)$/.exec(key)
  return m ? `Step ${m[1]}` : key
}

/** A log that stays at its newest line while you're there, and holds still when you scroll up. */
function Log({ events, tall = false, showStep = false }: { events: ProgressEvent[]; tall?: boolean; showStep?: boolean }) {
  const ref = useRef<HTMLOListElement>(null)
  const pinned = useRef(true)
  useLayoutEffect(() => {
    const el = ref.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [events])
  return (
    <ol ref={ref} onScroll={(e) => {
      const el = e.currentTarget
      pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
    }}
      className={`${tall ? 'max-h-[28rem]' : 'max-h-64'} space-y-px overflow-y-auto rounded-xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] p-2 font-mono text-[11px] leading-relaxed text-zinc-400`}>
      {events.map((e) => (
        <li key={e.i} className="flex gap-3 rounded-md px-1.5 py-0.5 hover:bg-[var(--fd-hover)]">
          <span className="shrink-0 tabular-nums text-zinc-600">{clock(e.ts)}</span>
          {showStep && e.agent && <span className="w-12 shrink-0 truncate text-violet-600/80 dark:text-violet-300/80">{stepName(e.agent)}</span>}
          <span className={`min-w-0 break-words ${e.stage === 'narration' ? 'italic text-zinc-300' : ''}`}>{line(e)}</span>
        </li>
      ))}
    </ol>
  )
}

interface Agent { key: string; title: string; state: StepState; name: string; now: boolean; actions: ProgressEvent[] }

function AgentRow({ a }: { a: Agent }) {
  const [open, setOpen] = useState(a.now)
  const last = a.actions[a.actions.length - 1]
  const state = a.now ? 'running' : a.state
  return (
    <li className={`rounded-xl border transition-colors ${a.now ? 'border-violet-500/30 bg-violet-500/[0.05]' : 'border-[var(--fd-line)] bg-[var(--fd-surface-raised)]'}`}>
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-[var(--fd-hover)]">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[0.7rem] bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]">
          <Bot className="h-4 w-4" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className="truncate text-[13px] font-medium text-zinc-200" title={a.title}>{a.name || stepName(a.key)}</span>
            <Pill tone={STATE[state].tone}>{a.now ? 'Working now' : STATE[state].label}</Pill>
          </span>
          <span className="block truncate text-[11px] text-zinc-500" title={a.title}>
            {stepName(a.key)}: {a.title}
          </span>
        </span>
        <span className="hidden shrink-0 text-right text-[11px] text-zinc-500 @3xl:block">
          <span className="block tabular-nums text-zinc-300">{a.actions.length} {a.actions.length === 1 ? 'action' : 'actions'}</span>
          {last && <span className="block">last at {clock(last.ts)}</span>}
        </span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="px-3 pb-3">
          {a.actions.length ? <Log events={a.actions.slice(-60)} /> : <p className="text-[11px] text-zinc-500">Nothing recorded yet.</p>}
        </div>
      )}
    </li>
  )
}

function TabButton({ icon: Icon, label, count, on, onClick }: { icon: LucideIcon; label: string; count?: number; on: boolean; onClick: () => void }) {
  return (
    <button role="tab" aria-selected={on} onClick={onClick}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
        on ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-[var(--fd-hover)] hover:text-zinc-200'}`}>
      <Icon className="hidden h-3.5 w-3.5 @2xl:block" />
      {label}
      {count != null && count > 0 && <span className={`fd-count ${on ? 'fd-count-hot' : ''}`}>{count > 999 ? '999+' : count}</span>}
    </button>
  )
}

export function RunProgress({ runId, going, stopped }: { runId: string; going: boolean; stopped: boolean }) {
  const [p, setP] = useState<Progress | null>(null)
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [error, setError] = useState('')
  const [tab, setTab] = useState<Tab>('steps')
  const data = useRunData(runId)
  const folder = useRunFolder(p?.files ?? '', going && (tab === 'files' || tab === 'notes'))
  const next = useRef(0)

  useEffect(() => {
    let stop = false
    next.current = 0
    const load = async () => {
      try {
        const got = await getRunProgress(runId, next.current)
        if (stop) return
        const fresh = next.current === 0
        setP(got)
        setError('')
        next.current = got.next
        if (fresh || got.events.length) {
          setEvents((old) => {
            const base = fresh ? [] : old
            const seen = new Set(base.map((e) => e.i))
            return [...base, ...got.events.filter((e) => !seen.has(e.i))].slice(-KEEP)
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
    }).sort((a, b) => Number(b.now) - Number(a.now))
  }, [p, events])

  if (error && !p) return <p className="mt-3 text-xs text-red-600 dark:text-red-300">{error}</p>
  if (!p) return <p className="mt-3 flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" /> Loading its progress…</p>

  const steps = p.available ? p.steps : []
  const done = steps.filter((s) => s.status === 'done').length
  const current = steps.find((s) => stateOf(s.status) === 'running')
  const activity = events.filter((e) => e.stage !== 'usage')
  const dataCount = data.data?.files.length ?? 0
  const shown: Tab = (tab === 'files' || tab === 'notes') && !p.files ? 'steps' : tab

  return (
    <div className="mt-3 overflow-hidden rounded-2xl border border-[var(--fd-line)] bg-[var(--fd-hover)]">
      <div className="space-y-3 px-4 pb-3 pt-3.5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          {/* A basis, so a narrow drawer puts the timing chips on a line of their own instead of over the count */}
          <div className="min-w-0 flex-[1_1_15rem]">
            {steps.length ? (
              <p className="flex min-w-0 items-baseline gap-2">
                <span className="shrink-0 whitespace-nowrap text-sm font-semibold text-zinc-100">{done} of {steps.length} steps done</span>
                <span className="shrink-0 text-xs tabular-nums text-zinc-500">{Math.round((done / steps.length) * 100)}%</span>
              </p>
            ) : (
              <p className="text-sm font-semibold text-zinc-100">
                {!p.available ? (going ? 'It hasn’t started working yet' : 'No progress to show') : going ? 'Planning its steps…' : 'It had no steps'}
              </p>
            )}
            {current && (
              <p className="mt-0.5 truncate text-xs text-zinc-500" title={current.title}>
                <span className="font-medium text-violet-700 dark:text-violet-300">Now</span> · {current.title}
              </p>
            )}
          </div>
          <TimingChips timing={p.timing} going={going} />
        </div>
        {steps.length > 0 && (
          <div className="flex gap-1" role="progressbar" aria-valuemin={0} aria-valuemax={steps.length} aria-valuenow={done} aria-label="Steps done">
            {steps.map((s, i) => (
              <span key={s.key} title={`${i + 1}. ${s.title} · ${STATE[stateOf(s.status)].label}`}
                className={`h-1.5 min-w-1 flex-1 rounded-full transition-colors ${STATE[stateOf(s.status)].bar}`} />
            ))}
          </div>
        )}
        {p.funnel && <FunnelStrip counts={p.funnel} />}
      </div>

      <div role="tablist" aria-label="This run" className="flex flex-wrap gap-1 border-y border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 py-2">
        <TabButton icon={ListChecks} label="Steps" count={steps.length} on={shown === 'steps'} onClick={() => setTab('steps')} />
        <TabButton icon={Bot} label="Agents" count={agents.length} on={shown === 'agents'} onClick={() => setTab('agents')} />
        <TabButton icon={Activity} label="Activity" count={activity.length} on={shown === 'activity'} onClick={() => setTab('activity')} />
        {p.files && <TabButton icon={FolderOpen} label="Files" count={folder.results.length} on={shown === 'files'} onClick={() => setTab('files')} />}
        {p.files && <TabButton icon={NotebookPen} label="Notes" count={folder.notes.length} on={shown === 'notes'} onClick={() => setTab('notes')} />}
        <TabButton icon={Database} label="Extra data" count={dataCount} on={shown === 'data'} onClick={() => setTab('data')} />
      </div>

      <div role="tabpanel" className="bg-[var(--fd-surface-raised)] p-3 @2xl:p-4">
        {shown === 'steps' && (steps.length ? (
          <ol className="space-y-1.5">
            {steps.map((s, i) => {
              const state = stateOf(s.status)
              return (
                <li key={s.key} className={`rounded-xl border px-3 py-2.5 transition-colors ${
                  state === 'running' ? 'border-violet-500/30 bg-violet-500/[0.05]' : 'border-transparent hover:bg-[var(--fd-hover)]'}`}>
                  <div className="flex items-start gap-3">
                    <Badge state={state} n={i + 1} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-start gap-3">
                        <p className={`min-w-0 flex-1 text-[13px] leading-relaxed ${state === 'pending' ? 'text-zinc-500' : 'text-zinc-200'}`}>{s.title}</p>
                        <span className="mt-0.5 flex shrink-0 items-center gap-1.5">
                          {s.attempt > 1 && <span className="text-[11px] text-zinc-500">try {s.attempt}</span>}
                          <Pill tone={STATE[state].tone}>{STATE[state].label}</Pill>
                        </span>
                      </div>
                      {s.error && state === 'failed' && <p className="mt-0.5 text-[11px] text-red-600 dark:text-red-300">{s.error}</p>}
                      {s.note && <NoteText text={s.note} startOpen={state === 'running'} />}
                    </div>
                  </div>
                </li>
              )
            })}
          </ol>
        ) : (
          <p className="text-xs text-zinc-500">
            {!p.available ? (going ? 'Its steps show here once it starts.' : 'There’s no progress to show for this run.')
              : going ? 'Captain is planning its steps; they show here in a moment.' : 'It had no steps.'}
          </p>
        ))}
        {shown === 'agents' && (agents.length
          ? <ul className="grid gap-2">{agents.map((a) => <AgentRow key={a.key} a={a} />)}</ul>
          : <p className="text-xs text-zinc-500">No agent has worked on it yet.</p>)}
        {shown === 'activity' && (activity.length
          ? <Log events={activity.slice(-150)} tall showStep />
          : <p className="text-xs text-zinc-500">Nothing yet.</p>)}
        {shown === 'files' && p.files && <RunFiles folder={folder} live={going} />}
        {shown === 'notes' && p.files && <RunNotes folder={folder} />}
        {shown === 'data' && <RunDataList runId={runId} going={going} stopped={stopped} state={data} />}
      </div>
    </div>
  )
}
