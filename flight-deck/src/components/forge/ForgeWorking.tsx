import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, BellRing, Check, Hammer, Loader2, RefreshCw } from 'lucide-react'
import { forgeApi, type ForgeFeedEvent, type ForgePlan, type ForgeWorkStep } from '../../services/agentForge'
import type { Archetype } from '../../services/tierConfig'
import { Button } from '../console/parts'
import { AgentBadge, Elapsed, ErrorNote, Fold } from './forgeParts'
import { errorOf, roleOf } from './forgeUtil'

// While Captain works: what it's doing now in plain words, what's done and
// what's next, and the team taking shape as it's chosen. The raw log is one click
// away for whoever wants it, and closing this loses nothing.

interface Stage { id: ForgeWorkStep; label: string; doing: string }

const STAGES: Stage[] = [
  { id: 'read', label: 'Reading your files', doing: 'Opening what you attached, so the plan starts from your material.' },
  { id: 'lead', label: 'Choosing your agents', doing: 'Working out the fewest agents that cover what you asked for.' },
  { id: 'plan', label: "Writing each agent's job", doing: 'What each one owns, delivers and hands to the others.' },
  { id: 'create', label: 'Creating your agents', doing: 'Starting each one with its job and its teammates.' },
  { id: 'recommend', label: 'Recommending what comes next', doing: 'What should run by itself, and what software is worth building.' },
]

function stagesOf(plan: ForgePlan): Stage[] {
  if (plan.approved.length) return STAGES.filter((s) => s.id === 'create' || s.id === 'recommend')
  return STAGES.filter((s) => (s.id === 'read' ? plan.files.length > 0 : s.id === 'lead' || s.id === 'plan'))
}

function Mark({ state }: { state: 'done' | 'now' | 'next' }) {
  if (state === 'done') return <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-600 text-white"><Check className="h-3.5 w-3.5" /></span>
  if (state === 'now') return <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-500/15 text-violet-700 ring-1 ring-violet-500/50 dark:text-violet-300"><Loader2 className="h-3.5 w-3.5 animate-spin" /></span>
  return <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-zinc-800"><span className="h-1.5 w-1.5 rounded-full bg-zinc-700" /></span>
}

export function ForgeWorking({ plan, archetypes, onChanged, onStartOver, onClose }: {
  plan: ForgePlan
  archetypes: Archetype[]
  onChanged: () => Promise<void>
  onStartOver: () => void
  onClose: () => void
}) {
  const [events, setEvents] = useState<ForgeFeedEvent[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [logOpen, setLogOpen] = useState(false)
  const failed = plan.status === 'error'
  const stopped = failed || plan.stalled

  useEffect(() => {
    let live = true
    const read = () => forgeApi.feed(plan.id).then((f) => { if (live) setEvents(f.events || []) }).catch(() => {})
    void read()
    if (stopped) return () => { live = false }
    const t = window.setInterval(read, 1200)
    return () => { live = false; window.clearInterval(t) }
  }, [plan.id, stopped])

  const stages = useMemo(() => stagesOf(plan), [plan])
  const reached = useMemo(() => {
    let at = plan.status === 'recommending' ? 'recommend' : plan.status === 'creating' ? 'create' : ''
    for (const e of events) if (e.step) at = e.step
    return Math.max(0, stages.findIndex((s) => s.id === at))
  }, [events, plan.status, stages])
  const now = stages[reached] ?? stages[0]
  const chosen = useMemo(() => [...events].reverse().find((e) => e.agents)?.agents ?? [], [events])
  const log = useMemo(() => events.filter((e) => e.message && !['usage', 'llm', 'action', 'narration'].includes(e.stage)), [events])

  // What carries on from here depends on how far it got. Going on with an approved team makes
  // only the agents it doesn't have yet.
  const resume = async () => {
    setBusy(true)
    setErr('')
    try {
      if (plan.status === 'recommending') await forgeApi.recommend(plan.id)
      else if (plan.approved.length) await forgeApi.approve(plan.id, [], plan.overview)
      else await forgeApi.replan(plan.id, '', [])
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setBusy(false)
    }
  }

  const made = new Map(plan.created.map((a) => [a.id, a]))
  const total = plan.approved.length
  const doneCount = plan.created.filter((a) => a.ok).length

  return (
    <div className="min-h-0 flex-1 overflow-y-auto" aria-live="polite">
      <div className="mx-auto w-full max-w-2xl space-y-5 px-4 py-6 wide:py-10">
        {stopped ? (
          <section className="fd-rise rounded-2xl border border-red-500/30 bg-red-500/5 p-5">
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-red-500/15 text-red-700 dark:text-red-300"><AlertTriangle className="h-5 w-5" /></span>
            <h2 className="mt-3 text-xl font-semibold tracking-tight text-zinc-100">{failed ? 'That stopped partway' : 'Flight Deck restarted partway'}</h2>
            <p className="mt-1 text-sm leading-relaxed text-zinc-400">
              {failed ? plan.error || "It didn't say why." : 'Nothing is lost. Carry on from where it was.'}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button variant="primary" icon={RefreshCw} busy={busy} onClick={resume}>Try again</Button>
              <Button onClick={onStartOver} disabled={busy}>New request</Button>
            </div>
          </section>
        ) : (
          <section className="fd-rise flex items-center gap-4">
            <span className="relative flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500/20 to-sky-500/10 text-violet-600 dark:text-violet-300">
              <span className="absolute inset-0 animate-ping rounded-2xl bg-violet-500/10 [animation-duration:2.4s] motion-reduce:hidden" />
              <Hammer className="relative h-6 w-6" />
            </span>
            <div className="min-w-0">
              <h2 className="text-2xl font-semibold tracking-tight text-zinc-100">{now.label}</h2>
              <p className="mt-0.5 text-sm leading-relaxed text-zinc-500">{now.doing}</p>
            </div>
          </section>
        )}

        {err && <ErrorNote>{err}</ErrorNote>}

        {total > 0 && (
          <section aria-label="Agents" className="fd-rise rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <div className="flex items-baseline justify-between gap-2 text-sm">
              <span className="font-medium text-zinc-200">Your agents</span>
              <span className="tabular-nums text-zinc-500">{doneCount} of {total} created</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
              <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-500" style={{ width: `${Math.max(4, (doneCount / total) * 100)}%` }} />
            </div>
            <ul className="mt-3 space-y-1.5">
              {plan.approved.map((a) => {
                const row = made.get(a.id)
                const nextUp = !row && !stopped && plan.approved.find((x) => !made.has(x.id))?.id === a.id
                return (
                  <li key={a.id} className="flex items-center gap-3 rounded-xl px-1 py-1">
                    <AgentBadge archetype={a.archetype} archetypes={archetypes} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-zinc-100">{row?.name || a.name}</span>
                      <span className={`block truncate text-xs ${row?.error ? 'text-red-700 dark:text-red-300' : 'text-zinc-500'}`}>{row?.error || roleOf(a.archetype, archetypes)}</span>
                    </span>
                    {row?.ok ? <span className="fd-rise flex items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-300"><Check className="h-3.5 w-3.5" /> Created</span>
                      : row?.error ? <AlertTriangle className="h-4 w-4 shrink-0 text-red-500" />
                        : nextUp ? <span className="flex items-center gap-1 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Creating</span>
                          : <span className="text-xs text-zinc-600">{stopped ? 'Not made' : 'Waiting'}</span>}
                  </li>
                )
              })}
            </ul>
          </section>
        )}

        <ol className="fd-rise rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4" aria-label="Progress">
          {stages.map((s, i) => {
            const state = i < reached ? 'done' : i === reached && !stopped ? 'now' : 'next'
            return (
              <li key={s.id} className="flex items-stretch gap-3">
                <span className="flex flex-col items-center">
                  <Mark state={state} />
                  {i < stages.length - 1 && <span className={`my-1 w-px flex-1 ${state === 'done' ? 'bg-violet-500/50' : 'bg-zinc-800'}`} />}
                </span>
                <span className={`min-w-0 flex-1 ${i < stages.length - 1 ? 'pb-4' : ''}`}>
                  <span className={`block text-sm leading-6 ${state === 'now' ? 'font-medium text-zinc-100' : state === 'done' ? 'text-zinc-300' : 'text-zinc-600'}`}>{s.label}</span>
                  {s.id === 'lead' && chosen.length > 0 && (
                    <span className="fd-rise mt-1 block text-xs text-zinc-500">{chosen.length} agent{chosen.length === 1 ? '' : 's'}: {chosen.map((a) => a.name).join(', ')}</span>
                  )}
                </span>
              </li>
            )
          })}
        </ol>

        {!total && !stopped && (
          <section aria-label="Your team" className="grid gap-3 wide:grid-cols-2">
            {chosen.length > 0
              ? chosen.map((a, i) => (
                  <div key={a.name + i} className="fd-rise flex items-center gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-3" style={{ animationDelay: `${i * 80}ms` }}>
                    <AgentBadge archetype={a.archetype} archetypes={archetypes} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-zinc-100">{a.name}</span>
                      <span className="block truncate text-xs text-zinc-500">{roleOf(a.archetype, archetypes)}</span>
                    </span>
                  </div>
                ))
              : [0, 1].map((i) => (
                  <div key={i} className="flex items-center gap-3 rounded-2xl border border-zinc-800/70 p-3" aria-hidden>
                    <span className="fd-shimmer h-9 w-9 shrink-0 rounded-xl" />
                    <span className="min-w-0 flex-1 space-y-1.5">
                      <span className="fd-shimmer block h-3 w-2/3 rounded" />
                      <span className="fd-shimmer block h-2.5 w-1/2 rounded" />
                    </span>
                  </div>
                ))}
          </section>
        )}

        {!stopped && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-zinc-800/70 px-4 py-3">
            <p className="flex min-w-0 items-center gap-2 text-xs leading-relaxed text-zinc-500">
              <BellRing className="h-3.5 w-3.5 shrink-0" />
              <span>You don't have to wait here. It carries on, and the bell tells you when it needs you.</span>
            </p>
            <div className="flex items-center gap-3">
              {events[0]?.ts && <span className="text-xs text-zinc-600"><Elapsed since={events[0].ts} /></span>}
              <Button variant="ghost" onClick={onClose} className="!px-2.5 !py-1 !text-xs">Close</Button>
            </div>
          </div>
        )}

        {log.length > 0 && (
          <Fold label="What it's doing, in detail" open={logOpen} onToggle={() => setLogOpen((v) => !v)}>
            <ol className="max-h-64 space-y-1 overflow-y-auto rounded-xl border border-zinc-800/70 p-3 text-xs text-zinc-500">
              {log.slice(-60).map((e, i) => (
                <li key={i} className={e.stage === 'phase' ? 'pt-1 font-medium text-zinc-300 first:pt-0' : e.ok === false ? 'text-amber-700 dark:text-amber-300' : ''}>{e.message}</li>
              ))}
            </ol>
          </Fold>
        )}
      </div>
    </div>
  )
}
