import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Mountain, Code2, Brain, Boxes, Play, Loader2, CheckCircle2, XCircle, AlertTriangle,
} from 'lucide-react'
import {
  getTiers, startCoder, startReason, startIntent, getTargets, getRun, listRuns,
  type Track, type Tier, type DubinaRun, type TargetOption,
} from '../services/dubinaApi'

const POLL_MS = 1200

export function DubinaPage() {
  const [track, setTrack] = useState<Track>('coder')
  const [tiers, setTiers] = useState<Tier[]>([])
  const [defaults, setDefaults] = useState<{ coder: string[]; reason: string[] }>({ coder: [], reason: [] })

  // Shared controls
  const [task, setTask] = useState('')
  const [baseTier, setBaseTier] = useState('')
  const [maxTier, setMaxTier] = useState('')
  const [budget, setBudget] = useState('')
  const [samples, setSamples] = useState(3)
  const [fixes, setFixes] = useState(2)

  // Coder-only
  const [workspace, setWorkspace] = useState('')
  const [testCommand, setTestCommand] = useState('pytest -q')
  const [solutionPath, setSolutionPath] = useState('solution.py')
  const [testPath, setTestPath] = useState('')
  const [spec, setSpec] = useState('')

  // Reasoning / Intent
  const [stakes, setStakes] = useState('normal')
  const [threshold, setThreshold] = useState(0.6)

  // Intent-only: the run target (archetype or live agent)
  const [target, setTarget] = useState('')
  const [targets, setTargets] = useState<TargetOption[]>([])

  const [run, setRun] = useState<DubinaRun | null>(null)
  const [history, setHistory] = useState<DubinaRun[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const poll = useRef<ReturnType<typeof setInterval> | null>(null)

  // Load tiers + run-targets once.
  useEffect(() => {
    getTiers()
      .then((r) => {
        setTiers(r.tiers)
        setDefaults(r.default_ladders)
      })
      .catch((e) => setError(String(e)))
    getTargets().then(setTargets).catch(() => {})
  }, [])

  useEffect(() => {
    if (!target && targets.length) setTarget(targets[0].value)
  }, [targets, target])

  // When the track (or defaults) change, seed the base/max selectors.
  useEffect(() => {
    const ladder = (defaults as Record<string, string[]>)[track] || defaults.reason
    if (ladder && ladder.length) {
      setBaseTier((b) => (ladder.includes(b) ? b : ladder[0]))
      setMaxTier((m) => (ladder.includes(m) ? m : ladder[ladder.length - 1]))
    } else if (tiers.length) {
      setBaseTier((b) => b || tiers[0].id)
      setMaxTier((m) => m || tiers[tiers.length - 1].id)
    }
  }, [track, defaults, tiers])

  const refreshHistory = useCallback(() => {
    listRuns(track).then((r) => setHistory(r.runs)).catch(() => {})
  }, [track])

  useEffect(() => {
    refreshHistory()
  }, [refreshHistory])

  useEffect(() => () => { if (poll.current) clearInterval(poll.current) }, [])

  const watch = useCallback((t: Track, id: string) => {
    if (poll.current) clearInterval(poll.current)
    poll.current = setInterval(async () => {
      try {
        const r = await getRun(t, id)
        setRun(r)
        if (r.status !== 'running') {
          if (poll.current) clearInterval(poll.current)
          setBusy(false)
          refreshHistory()
        }
      } catch {
        /* keep polling */
      }
    }, POLL_MS)
  }, [refreshHistory])

  const submit = async () => {
    setError('')
    setBusy(true)
    setRun(null)
    try {
      const computeBudget = budget ? Number(budget) : 0
      if (track === 'intent') {
        // Intent runs inline server-side and returns the finished run.
        const finished = await startIntent({
          task, target, base_tier: baseTier, max_tier: maxTier, compute_budget: computeBudget,
          max_step_samples: samples, max_fix_attempts: fixes, stakes, agreement_threshold: threshold,
        })
        setRun(finished)
        setBusy(false)
        refreshHistory()
        return
      }
      let started: { run_id: string; track: Track }
      if (track === 'coder') {
        started = await startCoder({
          task, workspace, test_command: testCommand, solution_path: solutionPath,
          test_path: testPath, spec, base_tier: baseTier, max_tier: maxTier,
          compute_budget: computeBudget, max_step_samples: samples, max_fix_attempts: fixes,
        })
      } else {
        started = await startReason({
          task, base_tier: baseTier, max_tier: maxTier, compute_budget: computeBudget,
          max_step_samples: samples, max_fix_attempts: fixes, stakes,
          agreement_threshold: threshold,
        })
      }
      watch(started.track, started.run_id)
    } catch (e) {
      setError(String(e))
      setBusy(false)
    }
  }

  const llmTiers = tiers
  const canRun = task.trim() && baseTier && maxTier && !busy
    && (track !== 'coder' || workspace.trim())
    && (track !== 'intent' || target)

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl px-6 py-6">
        {/* Header */}
        <div className="mb-6 flex items-start gap-2">
          <Mountain className="mt-0.5 h-5 w-5 text-violet-400" />
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Frontier Horizon</h1>
            <p className="mt-0.5 text-sm text-zinc-500">
              Simulate a top-frontier model on a cheaper tier — test-time compute, verifier-gated,
              escalating up the ladder only when a verifier demands it.
            </p>
          </div>
        </div>

        {/* Track tabs */}
        <div className="mb-4 inline-flex items-center gap-1 rounded-xl border border-zinc-800 p-1 text-sm">
          <TrackTab active={track === 'coder'} onClick={() => setTrack('coder')} icon={<Code2 className="h-3.5 w-3.5" />} label="Coder" />
          <TrackTab active={track === 'reason'} onClick={() => setTrack('reason')} icon={<Brain className="h-3.5 w-3.5" />} label="Reasoning" />
          <TrackTab active={track === 'intent'} onClick={() => setTrack('intent')} icon={<Boxes className="h-3.5 w-3.5" />} label="Intent" />
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {/* Controls */}
          <div className="space-y-3 md:col-span-2">
            <Field label="Task">
              <textarea
                value={task} onChange={(e) => setTask(e.target.value)} rows={3}
                placeholder={track === 'coder' ? 'Implement add(a, b) so the tests pass…' : 'What is the time complexity of…'}
                className="w-full resize-y rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
            </Field>

            {track === 'coder' && (
              <>
                <Field label="Workspace (project dir the tests run in)">
                  <Text value={workspace} onChange={setWorkspace} placeholder="/path/to/project" />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Test command"><Text value={testCommand} onChange={setTestCommand} /></Field>
                  <Field label="Solution file"><Text value={solutionPath} onChange={setSolutionPath} /></Field>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Test file (for spec→tests)"><Text value={testPath} onChange={setTestPath} placeholder="test_solution.py" /></Field>
                  <Field label="Spec (synthesize tests if missing)"><Text value={spec} onChange={setSpec} placeholder="optional" /></Field>
                </div>
              </>
            )}

            {track === 'intent' && (
              <Field label="Run target (archetype or live agent)">
                <Select value={target} onChange={setTarget}
                  options={targets.length ? targets.map((t) => [t.value, t.label])
                                          : [['', 'no archetypes or running agents found']]} />
              </Field>
            )}

            {(track === 'reason' || track === 'intent') && (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Stakes">
                  <Select value={stakes} onChange={setStakes} options={[['normal', 'normal'], ['high', 'high (always run critics)']]} />
                </Field>
                <Field label={`Agreement threshold (${threshold.toFixed(2)})`}>
                  <input type="range" min={0} max={1} step={0.05} value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-violet-500" />
                </Field>
              </div>
            )}
          </div>

          {/* Ladder + budget */}
          <div className="space-y-3">
            <Field label="Base tier (runs the whole process)">
              <Select value={baseTier} onChange={setBaseTier} options={llmTiers.map((t) => [t.id, t.id])} />
            </Field>
            <Field label="Max tier (escalation ceiling)">
              <Select value={maxTier} onChange={setMaxTier} options={llmTiers.map((t) => [t.id, t.id])} />
            </Field>
            <Field label="Compute budget (blank = unbounded)">
              <Text value={budget} onChange={setBudget} placeholder="e.g. 50" />
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Samples (vote)"><Num value={samples} onChange={setSamples} /></Field>
              <Field label="Fix attempts"><Num value={fixes} onChange={setFixes} /></Field>
            </div>
            <button
              onClick={submit} disabled={!canRun}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {busy ? 'Running…' : 'Run'}
            </button>
            {baseTier && (
              <p className="text-[11px] text-zinc-600">
                {tiers.find((t) => t.id === baseTier)?.description || ''}
              </p>
            )}
          </div>
        </div>

        {/* Live run */}
        {run && <RunView run={run} />}

        {/* History */}
        {history.length > 0 && (
          <div className="mt-8">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-600">
              Recent {track} runs
            </h2>
            <div className="divide-y divide-zinc-900 rounded-xl border border-zinc-900">
              {history.map((h) => (
                <button key={h.id} onClick={() => { setRun(null); getRun(track, h.id).then(setRun) }}
                  className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-zinc-900/50">
                  <StatusDot status={h.status} />
                  <span className="flex-1 truncate text-sm text-zinc-300">{h.task || '(untitled)'}</span>
                  <span className="text-[11px] text-zinc-600">{h.base_tier} → {h.max_tier}</span>
                  <span className="text-[11px] text-zinc-600">{h.cost_spent.toFixed(0)}u</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Live run view ────────────────────────────────────────────────────

function RunView({ run }: { run: DubinaRun }) {
  const summary = run.result || {}
  const answer = (summary.answer as string) || (summary.code as string) || ''
  return (
    <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="mb-3 flex items-center gap-2">
        <StatusDot status={run.status} />
        <span className="text-sm font-medium text-zinc-200 capitalize">{run.status}</span>
        {run.stopped_reason && <span className="text-[11px] text-amber-400">({run.stopped_reason})</span>}
        <span className="ml-auto text-[11px] text-zinc-600">
          tier {String(summary.tier_used ?? '—')} · rung {String(summary.rung_reached ?? '—')} · {run.cost_spent.toFixed(0)}u
        </span>
      </div>

      {run.steps && run.steps.length > 0 && (
        <div className="mb-3 overflow-x-auto">
          <table className="w-full text-left text-[11px]">
            <thead className="text-zinc-600">
              <tr><th className="py-1 pr-3">#</th><th className="pr-3">tier</th><th className="pr-3">rung</th><th className="pr-3">kind</th><th className="pr-3">samples</th><th className="pr-3">conf</th><th>pass</th></tr>
            </thead>
            <tbody className="text-zinc-400">
              {run.steps.map((s) => (
                <tr key={s.seq} className="border-t border-zinc-900">
                  <td className="py-1 pr-3">{s.seq}</td>
                  <td className="pr-3 text-zinc-300">{s.tier}</td>
                  <td className="pr-3">{s.rung}</td>
                  <td className="pr-3">{s.kind}</td>
                  <td className="pr-3">{s.samples}</td>
                  <td className="pr-3">{s.confidence.toFixed(2)}</td>
                  <td>{s.passed ? <CheckCircle2 className="h-3 w-3 text-emerald-400" /> : <XCircle className="h-3 w-3 text-zinc-700" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {answer && (
        <pre className="max-h-72 overflow-auto rounded-lg border border-zinc-900 bg-black/30 p-3 text-xs text-zinc-300">{answer}</pre>
      )}
      {run.error && <p className="text-xs text-red-300">{run.error}</p>}
    </div>
  )
}

// ── Small UI helpers ─────────────────────────────────────────────────

function TrackTab({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button onClick={onClick}
      className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-colors ${active ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}>
      {icon}{label}
    </button>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium text-zinc-500">{label}</span>
      {children}
    </label>
  )
}

const inputCls = 'w-full rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'

function Text({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className={inputCls} />
}

function Num({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return <input type="number" min={1} value={value} onChange={(e) => onChange(Math.max(1, Number(e.target.value) || 1))} className={inputCls} />
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: [string, string][] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={inputCls}>
      {options.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
    </select>
  )
}

function StatusDot({ status }: { status: string }) {
  if (status === 'running') return <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-400" />
  if (status === 'passed') return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
  if (status === 'error') return <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
  return <XCircle className="h-3.5 w-3.5 text-zinc-600" />
}
