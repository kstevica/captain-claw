// Captain Swarm runs (docs/captain-swarm-plan.md): start one from a template, watch it
// try ideas one at a time, and see every attempt's verdict. Two kinds so far:
// - inbox triage rules (Phase 3a): an agent's rules, tried on its exam lane against
//   what the owner actually did; the best rules reach it only when the owner teaches them;
// - code (Phase 2): the best commit sits on a branch; the owner opens the PR, Captain never merges.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { BookOpen, Check, GitPullRequest, Loader2, Play, Square, X } from 'lucide-react'
import { captainSwarmRuns } from '../../services/captainSwarm'
import type { Experiment, MravTemplate, RunDetail, RunSummary, TriageTemplate } from '../../services/captainSwarm'
import { useProcessStore } from '../../stores/processStore'

const ACTIVE = ['awaiting_go', 'baseline', 'running']
const STATUS_TEXT: Record<string, string> = {
  awaiting_go: 'waiting for your Go', baseline: 'measuring the start', running: 'running', done: 'done',
  stalled: 'stalled', stopped: 'stopped', failed: 'failed', cancelled: 'cancelled', declined: 'declined',
}
const STATUS_DOT: Record<string, string> = {
  awaiting_go: 'bg-fuchsia-400', baseline: 'bg-sky-400', running: 'bg-sky-400', done: 'bg-emerald-400',
  stalled: 'bg-amber-400', stopped: 'bg-amber-400', failed: 'bg-red-400', cancelled: 'bg-zinc-500', declined: 'bg-zinc-500',
}
const EXP_TEXT: Record<string, string> = {
  proposed: 'starting', running: 'worker busy', screening: 'screening', measuring: 'measuring',
  kept: 'kept', discarded: 'discarded', disqualified: 'thrown out', failed: 'failed',
}
const EXP_COLOR: Record<string, string> = {
  kept: 'text-emerald-400', discarded: 'text-zinc-400', disqualified: 'text-red-400', failed: 'text-amber-400',
}
// Chart marks, as SVG fills: the same meaning as the ledger's colours.
const MARK: Record<string, string> = {
  kept: '#34d399', discarded: '#a1a1aa', disqualified: '#f87171', failed: '#fbbf24',
}

const f3 = (n?: number | null) => (n === null || n === undefined ? '—' : n.toFixed(3))
const usd = (n?: number) => `$${(n || 0).toFixed(2)}`

function TextField({ label, value, onChange, wide }: {
  label: string; value: string; onChange: (v: string) => void; wide?: boolean
}) {
  return (
    <label className={`flex flex-col gap-1 text-xs text-zinc-400 ${wide ? 'sm:col-span-2' : ''}`}>
      {label}
      <input value={value} onChange={(e) => onChange(e.target.value)}
        className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100" />
    </label>
  )
}

function StartRun({ onStarted }: { onStarted: (id: string) => void }) {
  const [templates, setTemplates] = useState<{ mrav: MravTemplate; triage: TriageTemplate } | null>(null)
  const [which, setWhich] = useState<'triage' | 'mrav'>('triage')
  const [canStart, setCanStart] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    captainSwarmRuns.templates().then((t) => {
      setTemplates(t.templates)
      setCanStart(t.can_start)
      if (!t.templates.triage.sets.length && t.can_start) setWhich('mrav')
    }).catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  if (!templates) return error ? <div className="text-xs text-red-400">{error}</div> : null
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3 inline-flex overflow-hidden rounded-md border border-zinc-700 text-xs">
        {([['triage', 'Inbox triage rules'], ['mrav', 'Mrav eval (code)']] as const).map(([k, label]) => (
          <button key={k} onClick={() => setWhich(k)}
            className={`px-2.5 py-1 ${which === k ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}>
            {label}
          </button>
        ))}
      </div>
      {which === 'triage'
        ? <StartTriage tpl={templates.triage} onStarted={onStarted} />
        : <StartMrav tpl={templates.mrav} canStart={canStart} onStarted={onStarted} />}
    </div>
  )
}

function useStart(template: 'mrav' | 'triage', onStarted: (id: string) => void) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const start = async (options: Record<string, string | number>) => {
    setBusy(true)
    setError('')
    try {
      const r = await captainSwarmRuns.start(template, options)
      onStarted(r.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return { busy, error, start }
}

function StartTriage({ tpl, onStarted }: { tpl: TriageTemplate; onStarted: (id: string) => void }) {
  const processes = useProcessStore((s) => s.processes)
  const fetchProcesses = useProcessStore((s) => s.fetchProcesses)
  const [form, setForm] = useState<Record<string, string>>(
    () => Object.fromEntries(Object.entries(tpl.defaults).map(([k, v]) => [k, String(v)])))
  const { busy, error, start } = useStart('triage', onStarted)
  useEffect(() => { fetchProcesses() }, [fetchProcesses])
  useEffect(() => {
    setForm((f) => ({ ...f, set_id: f.set_id || tpl.sets[0]?.id || '', agent: f.agent || processes[0]?.slug || '' }))
  }, [tpl.sets, processes])
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  const agent = processes.find((p) => p.slug === form.agent)
  return (
    <>
      <div className="text-sm font-medium text-zinc-100">{tpl.title}</div>
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">{tpl.about}</p>
      {tpl.sets.length === 0 ? (
        <div className="mt-3 text-xs text-amber-400">Build a practice set first, on the Practice sets tab.</div>
      ) : (
        <>
          <div className="mt-3 grid gap-3 sm:grid-cols-4">
            <label className="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
              Practice set
              <select value={form.set_id} onChange={(e) => set('set_id')(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
                {tpl.sets.map((x) => <option key={x.id} value={x.id}>{x.title} · {x.open} open, {x.holdout} held back</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
              Agent
              <select value={form.agent} onChange={(e) => set('agent')(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
                {processes.map((p) => <option key={p.slug} value={p.slug}>{p.name}</option>)}
              </select>
            </label>
            <TextField label="Open cases each time" value={form.cases || ''} onChange={set('cases')} />
            <TextField label="Runs per case" value={form.repeat || ''} onChange={set('repeat')} />
            <TextField label="Held-out cases (0 = all)" value={form.holdout_cases || ''} onChange={set('holdout_cases')} />
            <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
            <TextField label="Budget ($)" value={form.budget_usd || ''} onChange={set('budget_usd')} />
            <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
            <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
          </div>
          <p className="mt-2 text-xs text-zinc-500">
            Each measurement asks {agent?.name || 'the agent'} {Number(form.cases || 0) * Number(form.repeat || 0)} questions
            on its own model; a model also reads your open cases to suggest rules.
          </p>
          <div className="mt-3 flex items-center gap-3">
            <button onClick={() => start({ ...form, agent_name: agent?.name || '' })} disabled={busy || !form.set_id || !form.agent}
              className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50">
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start the run
            </button>
            {error && <span className="text-xs text-red-400">{error}</span>}
          </div>
        </>
      )}
    </>
  )
}

function StartMrav({ tpl, canStart, onStarted }: { tpl: MravTemplate; canStart: boolean; onStarted: (id: string) => void }) {
  const [form, setForm] = useState<Record<string, string>>(
    () => Object.fromEntries(Object.entries(tpl.defaults).map(([k, v]) => [k, String(v)])))
  const { busy, error, start } = useStart('mrav', onStarted)
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  return (
    <>
      <div className="text-sm font-medium text-zinc-100">{tpl.title}</div>
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">
        {tpl.about} Each idea is tried by a coding worker in its own branch; Captain measures every attempt
        itself and keeps only gains bigger than the noise. Nothing is merged: the best commit waits on a branch.
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <TextField label="Repository" value={form.repo_path || ''} onChange={set('repo_path')} wide />
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Model measured
          <input list="cs-local-models" value={form.model || ''} onChange={(e) => set('model')(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100" />
          <datalist id="cs-local-models">{tpl.local_models.map((m) => <option key={m} value={m} />)}</datalist>
        </label>
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Coding worker
          <select value={form.harness || 'claude'} onChange={(e) => set('harness')(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
            {['claude', 'codex', 'opencode', 'crush', 'pi'].map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </label>
        <TextField label="Budget ($)" value={form.budget_usd || ''} onChange={set('budget_usd')} />
        <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
        <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
        <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
      </div>
      {tpl.local_models.length > 0 && !tpl.local_models.includes(form.model) && (
        <div className="mt-2 text-xs text-amber-400">
          {form.model} isn't installed in this computer's Ollama. Installed: {tpl.local_models.join(', ')}.
        </div>
      )}
      {!form.holdout_file && (
        <div className="mt-2 text-xs text-amber-400">There are no held-out cases on this deck yet.</div>
      )}
      <div className="mt-3 flex items-center gap-3">
        <button onClick={() => start(form)} disabled={busy || !canStart}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start the run
        </button>
        {!canStart && <span className="text-xs text-zinc-500">Measuring runs a command on this deck: admins only.</span>}
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </>
  )
}

function Chart({ run }: { run: RunDetail }) {
  const points = run.ledger.filter((e) => e.dev?.stats)
  const base = run.baseline?.dev?.stats?.mean
  if (base === undefined || points.length === 0) return null
  const values = [base, ...points.map((e) => e.dev.stats!.mean)]
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const pad = Math.max((hi - lo) * 0.15, 0.02)
  const [y0, y1] = [lo - pad, hi + pad]
  const W = 560
  const H = 140
  const x = (i: number) => 24 + (i + 0.5) * ((W - 32) / Math.max(points.length, 1))
  const y = (v: number) => H - 18 - ((v - y0) / (y1 - y0)) * (H - 30)
  // The best so far, as a step line: it only moves at a kept attempt.
  let best = base
  const steps: string[] = [`M 24 ${y(base)}`]
  points.forEach((e, i) => {
    if (e.status === 'kept') {
      steps.push(`L ${x(i)} ${y(best)}`, `L ${x(i)} ${y(e.dev.stats!.mean)}`)
      best = e.dev.stats!.mean
    }
  })
  steps.push(`L ${W - 8} ${y(best)}`)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full" role="img"
      aria-label="Each attempt's score on the open cases, and the best so far">
      <line x1={24} x2={W - 8} y1={y(base)} y2={y(base)} stroke="#52525b" strokeDasharray="3 3" />
      <text x={26} y={y(base) - 4} fill="#71717a" fontSize="10">start {base.toFixed(3)}</text>
      <path d={steps.join(' ')} fill="none" stroke="#34d399" strokeWidth={1.5} />
      {points.map((e, i) => (
        <circle key={e.id} cx={x(i)} cy={y(e.dev.stats!.mean)} r={4} fill={MARK[e.status] || '#a1a1aa'}>
          <title>{`#${e.round}.${e.n} ${EXP_TEXT[e.status] || e.status}: ${e.dev.stats!.mean.toFixed(3)}`}</title>
        </circle>
      ))}
    </svg>
  )
}

function Ledger({ exps }: { exps: Experiment[] }) {
  const [open, setOpen] = useState('')
  if (exps.length === 0) return <p className="text-xs text-zinc-500">No attempts yet.</p>
  return (
    <table className="w-full text-left text-xs">
      <thead>
        <tr className="whitespace-nowrap text-[11px] uppercase tracking-wide text-zinc-500">
          <th className="pb-1 pr-2 font-medium">#</th>
          <th className="pb-1 pr-3 font-medium">Idea</th>
          <th className="pb-1 pr-3 font-medium">Verdict</th>
          <th className="pb-1 pr-3 font-medium">Open</th>
          <th className="pb-1 pr-3 font-medium">Held-out</th>
          <th className="pb-1 font-medium">Cost</th>
        </tr>
      </thead>
      <tbody>
        {[...exps].reverse().map((e) => (
          <tr key={e.id} onClick={() => setOpen(open === e.id ? '' : e.id)}
            className="cursor-pointer border-t border-zinc-800 align-top hover:bg-zinc-800/40">
            <td className="py-2 pr-2 text-zinc-500">{e.round}.{e.n}</td>
            <td className="py-2 pr-3 text-zinc-200">
              <div className={open === e.id ? '' : 'line-clamp-2'}>{e.hypothesis}</div>
              {open === e.id && (
                <div className="mt-1 space-y-1 text-zinc-400">
                  {e.why && <div>Why: {e.why}</div>}
                  {(e.change?.add?.length || e.change?.remove?.length) ? (
                    <div className="space-y-0.5">
                      {(e.change?.add || []).map((r) => <div key={`a${r}`} className="text-emerald-400/90">+ {r}</div>)}
                      {(e.change?.remove || []).map((r) => <div key={`r${r}`} className="text-red-400/90">− rule {String(r)}</div>)}
                    </div>
                  ) : null}
                  {e.verdict_reason && <div>Verdict: {e.verdict_reason}</div>}
                  {e.lesson && <div>Lesson: {e.lesson}</div>}
                  {e.diff_stat && <div>Change: {e.diff_stat}{e.lane === 'crew' ? ` · ${e.changed_files.join(', ')}` : ''}</div>}
                  {e.head_sha && <div className="font-mono text-[11px] text-zinc-500">{e.head_sha.slice(0, 12)}</div>}
                </div>
              )}
            </td>
            <td className={`whitespace-nowrap py-2 pr-3 ${EXP_COLOR[e.status] || 'text-sky-400'}`}>
              {['proposed', 'running', 'screening', 'measuring'].includes(e.status) && (
                <Loader2 size={11} className="mr-1 inline animate-spin" />
              )}
              {EXP_TEXT[e.status] || e.status}
            </td>
            <td className="whitespace-nowrap py-2 pr-3 text-zinc-300">
              {e.dev?.stats ? <>{f3(e.dev.stats.mean)} <span className="text-zinc-500">± {f3(e.dev.stats.stdev)}</span></> : '—'}
            </td>
            <td className="py-2 pr-3 text-zinc-300">{e.holdout?.stats ? f3(e.holdout.stats.mean) : '—'}</td>
            <td className="py-2 text-zinc-400">{e.cost_usd ? usd(e.cost_usd) : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function RunView({ id, onChanged }: { id: string; onChanged: () => void }) {
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')

  const load = useCallback(async () => {
    try {
      setRun(await captainSwarmRuns.get(id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [id])
  useEffect(() => { load() }, [load])
  const active = run ? ACTIVE.includes(run.status) : false
  useEffect(() => {
    if (!active) return
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [active, load])

  const act = async (fn: () => Promise<unknown>, done?: string) => {
    setError('')
    try {
      await fn()
      if (done) setNote(done)
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  if (!run) return error ? <div className="text-xs text-red-400">{error}</div> : <Loader2 size={16} className="animate-spin text-zinc-500" />
  const b = run.budget
  return (
    <div className="space-y-4 rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-zinc-100">
            <span className={`h-2 w-2 rounded-full ${STATUS_DOT[run.status]}`} />
            {run.title}
          </div>
          <div className="mt-0.5 text-xs text-zinc-500">
            {STATUS_TEXT[run.status]}{run.stopped_reason ? `: ${run.stopped_reason}` : ''} · started by {run.started_by}
          </div>
        </div>
        <div className="flex gap-2">
          {run.status === 'awaiting_go' && (
            <>
              <button onClick={() => act(() => captainSwarmRuns.go(id))}
                className="inline-flex items-center gap-1 rounded-md bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-500">
                <Check size={13} /> Go
              </button>
              <button onClick={() => act(() => captainSwarmRuns.decline(id))}
                className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800">
                <X size={13} /> Decline
              </button>
            </>
          )}
          {['baseline', 'running'].includes(run.status) && (
            <button onClick={() => act(() => captainSwarmRuns.cancel(id))}
              className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800">
              <Square size={12} /> Stop
            </button>
          )}
          {!active && run.improved && run.kind === 'code' && (
            <button onClick={() => act(async () => {
              const r = await captainSwarmRuns.pr(id)
              if (r.delivery.url) setNote(`Pull request: ${r.delivery.url}`)
            })}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-500">
              <GitPullRequest size={13} /> Open a pull request
            </button>
          )}
          {!active && run.improved && run.kind === 'practice' && (
            <button onClick={() => act(async () => {
              const r = await captainSwarmRuns.teach(id)
              setNote(r.added ? `Taught ${r.added} rule(s) to ${run.objective.subject.agent_name}.` : 'It knew those rules already.')
            })}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-500">
              <BookOpen size={13} /> Teach these rules to {run.objective.subject.agent_name}
            </button>
          )}
        </div>
      </div>
      {(error || note) && <div className={`text-xs ${error ? 'text-red-400' : 'text-emerald-400'}`}>{error || note}</div>}
      <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-4">
        <div className="rounded-md bg-zinc-950/60 p-2.5">
          <div className="text-zinc-500">Start</div>
          <div className="mt-0.5 whitespace-nowrap text-sm text-zinc-200">{f3(run.baseline_dev)}
            <span className="text-xs text-zinc-500"> ± {f3(run.baseline?.dev?.stats?.stdev)}</span></div>
          <div className="text-zinc-500">held-out {f3(run.baseline_holdout)}</div>
        </div>
        <div className="rounded-md bg-zinc-950/60 p-2.5">
          <div className="text-zinc-500">Best so far</div>
          <div className={`mt-0.5 whitespace-nowrap text-sm ${run.improved ? 'text-emerald-400' : 'text-zinc-200'}`}>{f3(run.champion_dev)}
            <span className="text-xs text-zinc-500"> ± {f3(run.champion?.dev?.stats?.stdev)}</span></div>
          <div className="text-zinc-500">held-out {f3(run.champion_holdout)}</div>
        </div>
        <div className="rounded-md bg-zinc-950/60 p-2.5">
          <div className="text-zinc-500">Attempts</div>
          <div className="mt-0.5 text-sm text-zinc-200">{run.experiments}{b ? ` of ${b.max_experiments}` : ''}</div>
          <div className="text-zinc-500">round {run.round || 0}</div>
        </div>
        <div className="rounded-md bg-zinc-950/60 p-2.5">
          <div className="text-zinc-500">Spent</div>
          <div className="mt-0.5 text-sm text-zinc-200">{usd(run.spent_usd)}{b ? ` of ${usd(b.usd)}` : ''}</div>
          <div className="text-zinc-500">{Math.round(run.spent_tokens / 1000)}k tokens</div>
        </div>
      </div>
      <Chart run={run} />
      {run.kind === 'practice' && run.rules && (
        <div className="rounded-md bg-zinc-950/60 p-3 text-xs">
          <div className="mb-1 text-zinc-500">{run.improved ? 'The best rules so far' : 'Rules'}</div>
          {run.rules.champion.length === 0 ? (
            <div className="text-zinc-500">None: the agent as it is.</div>
          ) : (
            <ol className="list-decimal space-y-0.5 pl-4 text-zinc-200">
              {run.rules.champion.map((r) => (
                <li key={r} className={run.rules!.baseline.includes(r) ? '' : 'text-emerald-300'}>{r}</li>
              ))}
            </ol>
          )}
        </div>
      )}
      <Ledger exps={run.ledger} />
      <details className="text-xs text-zinc-400">
        <summary className="cursor-pointer text-zinc-500">What it measures and what it may change</summary>
        <div className="mt-2 space-y-1">
          <div>{run.objective.goal}</div>
          {run.kind === 'code' ? (
            <>
              <div>Repository: {run.objective.subject.repo_path}</div>
              <div>May change: {(run.objective.subject.editable || []).join(', ')}</div>
              <div>Never changes: {(run.objective.subject.protected || []).join(', ')}</div>
              <div className="break-all font-mono text-[11px] text-zinc-500">{run.objective.metric.run}</div>
              <div>Coding worker: {run.objective.lane.harness}{run.objective.lane.model ? ` (${run.objective.lane.model})` : ''}</div>
            </>
          ) : (
            <>
              <div>Agent: {run.objective.subject.agent_name} · practice set {run.objective.subject.set_id}</div>
              <div>Each measurement: {run.objective.metric.cases} open cases × {run.objective.metric.repeat} runs, on the
                agent's exam lane (it can't send or change anything). Scored by F1 on "needs your reply".</div>
            </>
          )}
        </div>
      </details>
      <details className="text-xs text-zinc-400" open={active}>
        <summary className="cursor-pointer text-zinc-500">Log</summary>
        <ul className="mt-2 max-h-56 space-y-0.5 overflow-y-auto">
          {[...run.events].reverse().map((ev) => (
            <li key={ev.id}>
              <span className="text-zinc-600">{new Date(ev.ts * 1000).toLocaleTimeString()}</span> {ev.message}
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}

export function RunsPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [activeId, setActiveId] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    try {
      setRuns((await captainSwarmRuns.list()).runs)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])
  useEffect(() => { load() }, [load])
  const anyActive = runs.some((r) => ACTIVE.includes(r.status))
  useEffect(() => {
    if (!anyActive) return
    const t = setInterval(load, 8000)
    return () => clearInterval(t)
  }, [anyActive, load])
  const selected = useMemo(() => activeId || runs[0]?.id || '', [activeId, runs])

  return (
    <div className="space-y-4">
      {error && <div className="text-xs text-red-400">{error}</div>}
      <StartRun onStarted={(id) => { setActiveId(id); load() }} />
      {runs.length > 0 && (
        <div className="grid gap-4 xl:grid-cols-[16rem_1fr]">
          <ul className="space-y-1">
            {runs.map((r) => (
              <li key={r.id}>
                <button onClick={() => setActiveId(r.id)}
                  className={`w-full rounded-md px-2.5 py-2 text-left text-sm ${selected === r.id ? 'bg-zinc-800' : 'hover:bg-zinc-800/60'}`}>
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${STATUS_DOT[r.status]}`} />
                    <span className="truncate text-zinc-100">{r.title}</span>
                  </div>
                  <div className="mt-0.5 pl-4 text-[11px] text-zinc-500">
                    {STATUS_TEXT[r.status]} · {f3(r.baseline_dev)} → {f3(r.champion_dev)} · {r.experiments} attempts
                  </div>
                </button>
              </li>
            ))}
          </ul>
          {selected && <RunView key={selected} id={selected} onChanged={load} />}
        </div>
      )}
    </div>
  )
}
