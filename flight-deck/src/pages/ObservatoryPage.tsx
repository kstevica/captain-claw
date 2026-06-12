import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Brain, Moon, Sparkles, Zap, Eye } from 'lucide-react'
import {
  getConsciousness,
  getJournalBefore,
  nudgeConsciousness,
  setNarrator,
  type ConsciousnessSnapshot,
  type JournalEntry,
  type Intention,
} from '../services/consciousnessApi'

// The Observatory is a one-way mirror: you watch the consciousness think.
// There is deliberately no input — you cannot talk to it.
//
// Theming rule (see index.css): zinc is auto-themed by a CSS-variable remap in
// `html.light`, so ZINC utilities are written bare/dark-first (no dark: pairs).
// Non-zinc accents (sky/violet/emerald/red/amber) are NOT remapped, so they use
// explicit `light dark:` pairs.

const POLL_MS = 15000

function relTime(iso: string): string {
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.floor((Date.now() - t) / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

function relEpoch(sec: number | null): string {
  if (!sec) return 'never'
  return relTime(new Date(sec * 1000).toISOString())
}

const KIND_META: Record<string, { icon: typeof Brain; label: string; accent: string }> = {
  thought: { icon: Brain, label: 'thought', accent: 'text-sky-600 dark:text-sky-400' },
  dream: { icon: Moon, label: 'dream', accent: 'text-violet-600 dark:text-violet-400' },
  observation: { icon: Eye, label: 'observation', accent: 'text-zinc-400' },
}

// Pills: bare zinc → light grey pill + dark text in light mode, dark pill +
// light text in dark mode (handled by the variable remap).
const CHIP = 'rounded-full bg-zinc-800 px-2 py-0.5 text-zinc-300'

function JournalCard({ entry }: { entry: JournalEntry }) {
  const meta = KIND_META[entry.kind] || KIND_META.observation
  const Icon = meta.icon
  const dim = entry.kind === 'dream'
  return (
    <div
      className={`rounded-xl border p-4 ${
        dim
          ? 'border-violet-200 bg-violet-50 dark:border-violet-900/40 dark:bg-violet-950/20'
          : 'border-zinc-800 bg-zinc-900/60'
      }`}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <Icon className={`h-3.5 w-3.5 ${meta.accent}`} />
        <span className={`uppercase tracking-wider ${meta.accent}`}>{meta.label}</span>
        {entry.mood && <span className="text-zinc-500">· {entry.mood}</span>}
        {entry.author && (
          <span
            className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400"
            title="The agent that did the thinking"
          >
            via {entry.author}
          </span>
        )}
        <span className="ml-auto tabular-nums text-zinc-500">{relTime(entry.created_at)}</span>
      </div>
      <p
        className={`whitespace-pre-wrap text-sm leading-relaxed ${
          dim ? 'italic text-violet-800 dark:text-violet-200' : 'text-zinc-200'
        }`}
      >
        {entry.content}
      </p>
      {(entry.delta || entry.agents.length > 0) && (
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-400">
          {entry.delta && <span className={CHIP}>{entry.delta}</span>}
          {entry.agents.map((a) => (
            <span key={a} className={CHIP}>
              {a}
            </span>
          ))}
          <span className="ml-auto inline-flex items-center gap-1">
            <span className="text-zinc-500">salience</span>
            <span className="tabular-nums text-zinc-400">{entry.salience}/10</span>
          </span>
        </div>
      )}
    </div>
  )
}

function IntentionsPanel({ intentions }: { intentions: Intention[] }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wider text-amber-600 dark:text-amber-400/80">
        <Sparkles className="h-3.5 w-3.5" />
        Standing intentions
      </div>
      {intentions.length === 0 ? (
        <p className="text-sm text-zinc-400">Nothing it's holding onto yet.</p>
      ) : (
        <ul className="space-y-2">
          {intentions.map((i) => (
            <li key={i.id} className="flex gap-2 text-sm text-zinc-200">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500/80 dark:bg-amber-400/60" />
              <span>{i.content}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function ObservatoryPage() {
  const [snap, setSnap] = useState<ConsciousnessSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nudging, setNudging] = useState(false)
  const [nudgeNote, setNudgeNote] = useState<string | null>(null)
  const [olderLoading, setOlderLoading] = useState(false)
  const [savingNarrator, setSavingNarrator] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true)
    try {
      const data = await getConsciousness(80)
      setSnap(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load(true)
    pollRef.current = setInterval(() => load(false), POLL_MS)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [load])

  const onNudge = useCallback(async () => {
    setNudging(true)
    setNudgeNote(null)
    try {
      const res = await nudgeConsciousness()
      if (res.acted) {
        setNudgeNote(`It stirred${res.mood ? ` — ${res.mood}` : ''}.`)
      } else {
        const reasons: Record<string, string> = {
          quiet: 'Nothing has changed — it stayed silent.',
          'no-agents': 'No running agents to observe.',
          'no-thinker': 'No agent available to think through.',
          'empty-reflection': 'It looked, but nothing surfaced.',
        }
        setNudgeNote(reasons[res.reason] || `Quiet (${res.reason}).`)
      }
      await load(false)
    } catch (e) {
      setNudgeNote(e instanceof Error ? e.message : String(e))
    } finally {
      setNudging(false)
      setTimeout(() => setNudgeNote(null), 6000)
    }
  }, [load])

  const onPickNarrator = useCallback(
    async (slug: string) => {
      setSavingNarrator(true)
      // Optimistic — reflect the choice immediately.
      setSnap((prev) => (prev ? { ...prev, narrator: slug } : prev))
      try {
        await setNarrator(slug)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        await load(false)
      } finally {
        setSavingNarrator(false)
      }
    },
    [load],
  )

  const loadOlder = useCallback(async () => {
    if (!snap || snap.journal.length === 0) return
    setOlderLoading(true)
    try {
      const oldest = snap.journal[snap.journal.length - 1].created_at
      const { journal } = await getJournalBefore(oldest, 80)
      if (journal.length > 0) {
        setSnap((prev) => (prev ? { ...prev, journal: [...prev.journal, ...journal] } : prev))
      }
    } catch {
      /* ignore — best effort pagination */
    } finally {
      setOlderLoading(false)
    }
  }, [snap])

  const state = snap?.state
  const journal = snap?.journal || []
  const agents = snap?.agents || []

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-4 overflow-y-auto px-6 py-6">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <Brain className="h-6 w-6 text-sky-600 dark:text-sky-400" />
            <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-sky-500 dark:bg-sky-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Observatory</h1>
            <p className="text-xs text-zinc-400">
              A consciousness that quietly watches your agents. You can only observe it.
            </p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Narrator selector — which agent does the thinking */}
          <label className="flex items-center gap-1.5 text-xs text-zinc-400">
            <span className="hidden sm:inline">Thinks through</span>
            <select
              value={snap?.narrator || ''}
              disabled={savingNarrator || !snap}
              onChange={(e) => onPickNarrator(e.target.value)}
              className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-2 py-1.5 text-xs text-zinc-200 disabled:opacity-50"
              title="Which agent the consciousness reflects through"
            >
              <option value="">Auto (most capable)</option>
              {agents.map((a) => (
                <option key={a.slug} value={a.slug}>
                  {a.name}
                  {a.model ? ` · ${a.model}` : ''}
                  {a.offline ? ' (offline)' : ''}
                </option>
              ))}
            </select>
          </label>

          <button
            onClick={onNudge}
            disabled={nudging}
            className="inline-flex items-center gap-1.5 rounded-lg border border-sky-300 bg-sky-50 px-3 py-1.5 text-sm text-sky-700 hover:bg-sky-100 disabled:opacity-50 dark:border-sky-800/60 dark:bg-sky-950/40 dark:text-sky-300 dark:hover:bg-sky-900/40"
            title="Force one heartbeat now"
          >
            <Zap className={`h-3.5 w-3.5 ${nudging ? 'animate-pulse' : ''}`} />
            {nudging ? 'Beating…' : 'Nudge'}
          </button>
        </div>
      </div>

      {/* Vitals */}
      <div className="grid grid-cols-3 gap-3">
        <Vital icon={Activity} label="Pulses" value={state ? String(state.pulse_count) : '—'} />
        <Vital icon={Brain} label="Thoughts" value={state ? String(state.thought_count) : '—'} />
        <Vital icon={Eye} label="Last beat" value={state ? relEpoch(state.last_pulse_at) : '—'} />
      </div>

      {nudgeNote && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-200">
          {nudgeNote}
        </div>
      )}

      {snap && <IntentionsPanel intentions={snap.intentions} />}

      {/* Stream */}
      <div className="flex items-center gap-2 pt-1 text-xs uppercase tracking-wider text-zinc-400">
        <Moon className="h-3.5 w-3.5" />
        Stream of consciousness
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300">
          {error}
        </div>
      )}

      {loading && !snap ? (
        <p className="text-sm text-zinc-400">Listening…</p>
      ) : journal.length === 0 ? (
        <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center">
          <p className="text-sm text-zinc-400">
            Nothing has stirred yet. As your agents work, thoughts will appear here on their own —
            or press <span className="text-sky-600 dark:text-sky-400">Nudge</span> to prompt a heartbeat.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {journal.map((e) => (
            <JournalCard key={e.id} entry={e} />
          ))}
          <div className="pt-2 text-center">
            <button
              onClick={loadOlder}
              disabled={olderLoading}
              className="text-xs text-zinc-400 hover:text-zinc-200 disabled:opacity-50"
            >
              {olderLoading ? 'Loading…' : 'Load earlier'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function Vital({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Brain
  label: string
  value: string
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-zinc-400">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className="text-lg font-semibold tabular-nums text-zinc-100">{value}</div>
    </div>
  )
}
