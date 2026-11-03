// Captain Swarm (docs/captain-swarm-plan.md): agents that get measurably better at real work.
// Runs: keep improving a measured number, one idea at a time (Phase 2, on code).
// Practice sets: a set frozen from the owner's own history (inbox triage: did you reply?),
// and an agent measured on it on its private exam lane — nothing sent, nothing kept (Phase 1).
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Loader2, Mail, Square, Target, Trash2 } from 'lucide-react'
import { captainSwarmApi } from '../services/captainSwarm'
import type { Measurement, PracticeSet } from '../services/captainSwarm'
import { useProcessStore } from '../stores/processStore'
import { RunsPanel } from '../components/captainSwarm/RunsPanel'

const pct = (n?: number) => `${Math.round((n || 0) * 100)}%`
const num = (n?: number, d = 2) => (n ?? 0).toFixed(d)

function Field({ label, value, onChange, min, max, unit }: {
  label: string; value: number; onChange: (n: number) => void; min: number; max: number; unit?: string
}) {
  // Typed freely, kept in range when the field is left: clamping on every key
  // would turn "12" into "52" on the way.
  const [text, setText] = useState(String(value))
  useEffect(() => { setText(String(value)) }, [value])
  const commit = () => {
    const n = Math.max(min, Math.min(max, parseInt(text || '0', 10) || min))
    setText(String(n))
    onChange(n)
  }
  return (
    <label className="flex items-center gap-1.5 whitespace-nowrap text-xs text-zinc-400">
      {label}
      <input
        type="number" min={min} max={max} value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit() }}
        className="w-16 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-zinc-100"
      />
      {unit && <span className="text-zinc-500">{unit}</span>}
    </label>
  )
}

function BuildTriage({ onBuilt, busy }: { onBuilt: () => void; busy: boolean }) {
  const [windowDays, setWindowDays] = useState(90)
  const [replyDays, setReplyDays] = useState(3)
  const [holdoutDays, setHoldoutDays] = useState(21)
  const [maxCases, setMaxCases] = useState(400)
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)

  const build = async () => {
    setError('')
    setSending(true)
    try {
      await captainSwarmApi.buildTriage({
        window_days: windowDays, reply_days: replyDays, holdout_days: holdoutDays, max_cases: maxCases,
      })
      onBuilt()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
        <Mail size={15} className="text-sky-400" /> Inbox triage: did you reply?
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
        Captain reads a sample of the mail you received (read-only) and labels each email by whether you wrote in
        its thread within a few days. Those labels are what you actually did, so they cost you no time. The copy
        stays on this deck in a locked folder, only for you; the most recent weeks are held back to check gains
        later. Delete the set and the copy is gone.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Field label="last" value={windowDays} onChange={setWindowDays} min={14} max={365} unit="days" />
        <Field label="replied within" value={replyDays} onChange={setReplyDays} min={1} max={14} unit="days" />
        <Field label="hold back the last" value={holdoutDays} onChange={setHoldoutDays} min={7} max={90} unit="days" />
        <Field label="at most" value={maxCases} onChange={setMaxCases} min={20} max={1000} unit="emails" />
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          onClick={build}
          disabled={sending || busy}
          className="inline-flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
        >
          {sending ? <Loader2 size={14} className="animate-spin" /> : <Mail size={14} />}
          Build from my Gmail
        </button>
        {busy && <span className="text-xs text-zinc-500">A set is being built.</span>}
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </div>
  )
}

function SetCard({ set, active, onSelect, onDelete }: {
  set: PracticeSet; active: boolean; onSelect: () => void; onDelete: () => void
}) {
  const p = set.progress
  return (
    <div
      onClick={set.state === 'ready' ? onSelect : undefined}
      className={`rounded-lg border p-3 text-sm ${active ? 'border-sky-600 bg-zinc-800/70' : 'border-zinc-800 bg-zinc-900/40'}
        ${set.state === 'ready' ? 'cursor-pointer hover:border-zinc-600' : ''}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-medium text-zinc-100">{set.title || set.id}</span>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          title="Delete this set and its copy of your mail"
          className="text-zinc-500 hover:text-red-400"
        >
          <Trash2 size={14} />
        </button>
      </div>
      {set.state === 'building' && (
        <div className="mt-1 flex items-center gap-1.5 text-xs text-zinc-400">
          <Loader2 size={12} className="animate-spin" />
          {p?.stage === 'reading' && p.total ? `Reading mail ${p.done}/${p.total}` : 'Listing your mail…'}
        </div>
      )}
      {set.state === 'error' && <div className="mt-1 text-xs text-red-400">{set.error}</div>}
      {set.state === 'ready' && set.open && set.holdout && (
        <div className="mt-1 space-y-0.5 text-xs text-zinc-400">
          <div>{set.open.cases} open · {set.holdout.cases} held back</div>
          <div>You replied to {pct(set.open.base_rate)} of the open part, {pct(set.holdout.base_rate)} of the held-back part</div>
          <div className="text-zinc-500">
            {set.source?.window?.from?.slice(0, 10)} → {set.source?.window?.to?.slice(0, 10)}
            {set.source?.gmail ? ` · ${set.source.gmail}` : ''}
          </div>
        </div>
      )}
    </div>
  )
}

function MeasurementRow({ m, onStop }: { m: Measurement; onStop: () => void }) {
  const r = m.result
  return (
    <tr className="border-t border-zinc-800 text-xs">
      <td className="py-2 pr-3 text-zinc-200">{m.agent_name}</td>
      <td className="pr-3 text-zinc-400">{m.split === 'holdout' ? 'held back' : 'open'}</td>
      <td className="pr-3 text-zinc-400">{m.cases} × {m.repeat}</td>
      <td className="pr-3">
        {m.state === 'running' ? (
          <span className="inline-flex items-center gap-1.5 text-zinc-400">
            <Loader2 size={12} className="animate-spin" />
            {m.progress?.total ? `${m.progress.done}/${m.progress.total}` : 'starting'}
            <button onClick={onStop} title="Stop" className="text-zinc-500 hover:text-red-400"><Square size={12} /></button>
          </span>
        ) : m.state === 'error' ? (
          <span className="text-red-400">{m.error}</span>
        ) : r ? (
          <span className="font-medium text-zinc-100">
            {num(r.f1.mean)} <span className="font-normal text-zinc-500">± {num(r.f1.stdev)}</span>
            {m.state === 'stopped' && <span className="ml-1 font-normal text-amber-400">stopped</span>}
          </span>
        ) : null}
      </td>
      <td className="pr-3 text-zinc-400">{r ? `${num(r.pooled.precision)} / ${num(r.pooled.recall)}` : ''}</td>
      <td className="pr-3 text-zinc-400">{r ? `${r.guardrails.seconds_per_case}s · ${Math.round(r.guardrails.tokens_per_case)} tok` : ''}</td>
      <td className="text-zinc-500">{r && r.pooled.unanswered ? `${r.pooled.unanswered} unanswered` : ''}</td>
    </tr>
  )
}

function Measure({ set }: { set: PracticeSet }) {
  const processes = useProcessStore((s) => s.processes)
  const fetchProcesses = useProcessStore((s) => s.fetchProcesses)
  const [agent, setAgent] = useState('')
  const [split, setSplit] = useState<'open' | 'holdout'>('open')
  const [cases, setCases] = useState(50)
  const [repeat, setRepeat] = useState(3)
  const [rows, setRows] = useState<Measurement[]>([])
  const [error, setError] = useState('')

  useEffect(() => { fetchProcesses() }, [fetchProcesses])
  useEffect(() => {
    if (!agent && processes.length) setAgent(processes[0].slug)
  }, [agent, processes])

  const load = useCallback(async () => {
    try {
      setRows((await captainSwarmApi.measurements(set.id)).measurements)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [set.id])

  const running = rows.some((m) => m.state === 'running')
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!running) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [running, load])

  const start = async () => {
    setError('')
    try {
      await captainSwarmApi.measure(set.id, { agent, split, cases, repeat })
      load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const available = split === 'holdout' ? set.holdout?.cases || 0 : set.open?.cases || 0
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="text-sm font-medium text-zinc-100">Measure an agent on it</div>
      <p className="mt-1 text-xs text-zinc-400">
        The agent sees each email and answers YES or NO: does it need your reply? It answers on its private exam
        lane, so it can't send or change anything, and it keeps nothing. Each case is asked several times, so
        the score comes with its noise. The score is F1 on "needs your reply".
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <select
          value={agent}
          onChange={(e) => setAgent(e.target.value)}
          className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-100"
        >
          {processes.map((p) => (
            <option key={p.slug} value={p.slug}>
              {p.name}{p.status !== 'running' && !p.asleep ? ' (stopped)' : ''}
            </option>
          ))}
        </select>
        <div className="inline-flex overflow-hidden rounded-md border border-zinc-700 text-xs">
          {(['open', 'holdout'] as const).map((s) => (
            <button key={s} onClick={() => setSplit(s)}
              className={`px-2.5 py-1 ${split === s ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}>
              {s === 'open' ? 'Open' : 'Held back'}
            </button>
          ))}
        </div>
        <Field label="cases" value={cases} onChange={setCases} min={5} max={300} />
        <Field label="runs" value={repeat} onChange={setRepeat} min={1} max={5} />
        <button
          onClick={start}
          disabled={!agent || running}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50"
        >
          <Target size={14} /> Measure
        </button>
        <span className="text-xs text-zinc-500">
          {Math.min(cases, available)} of {available} cases × {repeat} = {Math.min(cases, available) * repeat} questions
        </span>
      </div>
      {error && <div className="mt-2 text-xs text-red-400">{error}</div>}
      {rows.length > 0 && (
        <table className="mt-4 w-full text-left">
          <thead>
            <tr className="text-[11px] uppercase tracking-wide text-zinc-500">
              <th className="pb-1 font-medium">Agent</th>
              <th className="pb-1 font-medium">Part</th>
              <th className="pb-1 font-medium">Cases × runs</th>
              <th className="pb-1 font-medium">F1</th>
              <th className="pb-1 font-medium">Precision / recall</th>
              <th className="pb-1 font-medium">Per case</th>
              <th className="pb-1 font-medium" />
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <MeasurementRow key={m.id} m={m} onStop={() => captainSwarmApi.stop(set.id, m.id).then(load)} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export function CaptainSwarmPage() {
  const [tab, setTab] = useState<'runs' | 'practice'>('runs')
  return (
    <div className="h-full overflow-y-auto bg-zinc-950">
      <div className="mx-auto max-w-6xl space-y-4 p-4 sm:p-6">
        <div>
          <h1 className="flex items-center gap-2 text-lg font-semibold text-zinc-100">
            <Target size={18} className="text-amber-400" /> Captain Swarm
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Agents that get measurably better at real work. A run keeps improving one number: it tries ideas one
            at a time, measures every attempt itself, and keeps only gains bigger than the noise.
          </p>
        </div>
        <div className="inline-flex overflow-hidden rounded-md border border-zinc-700 text-xs">
          {(['runs', 'practice'] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1.5 ${tab === t ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}>
              {t === 'runs' ? 'Runs' : 'Practice sets'}
            </button>
          ))}
        </div>
        {tab === 'runs' ? <RunsPanel /> : <PracticeSets />}
      </div>
    </div>
  )
}

function PracticeSets() {
  const [sets, setSets] = useState<PracticeSet[]>([])
  const [activeId, setActiveId] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setSets((await captainSwarmApi.sets()).sets)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const building = sets.some((s) => s.state === 'building')
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!building) return
    const t = setInterval(load, 2500)
    return () => clearInterval(t)
  }, [building, load])

  const active = useMemo(
    () => sets.find((s) => s.id === activeId && s.state === 'ready') || sets.find((s) => s.state === 'ready'),
    [sets, activeId],
  )

  const remove = async (s: PracticeSet) => {
    if (!window.confirm('Delete this practice set? Its copy of your mail and its measurements go with it.')) return
    try {
      await captainSwarmApi.deleteSet(s.id)
      load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="space-y-4">
        {error && <div className="text-xs text-red-400">{error}</div>}
        <BuildTriage onBuilt={load} busy={building} />
        {sets.length > 0 && (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {sets.map((s) => (
              <SetCard key={s.id} set={s} active={active?.id === s.id}
                onSelect={() => setActiveId(s.id)} onDelete={() => remove(s)} />
            ))}
          </div>
        )}
        {active && <Measure key={active.id} set={active} />}
    </div>
  )
}
