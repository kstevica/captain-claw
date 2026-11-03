// Captain Swarm runs (docs/captain-swarm-plan.md): start one from a template, watch it
// try ideas — one at a time, or several at once in waves (Phase 4) — and see every
// attempt's verdict. Three kinds so far:
// - inbox triage rules (Phase 3a): an agent's rules, tried on its exam lane against
//   what the owner actually did; the best rules reach it only when the owner teaches them;
// - code (Phase 2): the best commit sits on a branch; the owner opens the PR, Captain never merges;
// - one task's best result (Phase 3b): attempts in rehearsal, every claim re-checked by Captain,
//   the winner carried out once by a Bat run, through Bat's gate, and saved as a playbook if wanted.
// Every run learns (Phase 6): lessons are kept per kind of task, and the next run of that kind
// starts from them; the owner sees them and can forget any.
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  BookMarked, BookOpen, Check, ExternalLink, FileDown, GitPullRequest, Hand, Loader2, Play, Square, X,
} from 'lucide-react'
import { captainSwarmRuns } from '../../services/captainSwarm'
import type {
  CodeTemplate, DocumentTemplate, Experiment, Lesson, MravTemplate, OutcomeTemplate, Playbook, RunDetail, RunSummary,
  TriageTemplate,
} from '../../services/captainSwarm'
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
  measured: 'waiting for the wave', kept: 'kept', discarded: 'discarded', disqualified: 'thrown out', failed: 'failed',
}
const LIVE_EXP = ['proposed', 'running', 'screening', 'measuring', 'measured']
const EXP_COLOR: Record<string, string> = {
  kept: 'text-emerald-400', discarded: 'text-zinc-400', disqualified: 'text-red-400', failed: 'text-amber-400',
}
// Chart marks, as SVG fills: the same meaning as the ledger's colours.
const MARK: Record<string, string> = {
  kept: '#34d399', discarded: '#a1a1aa', disqualified: '#f87171', failed: '#fbbf24',
}

// "Nothing yet" (a one-task run's start is a sentinel far past any real number) reads as a dash.
const f3 = (n?: number | null) => (n === null || n === undefined || Math.abs(n) >= 1e11 ? '—' : n.toFixed(3))
const usd = (n?: number) => `$${(n || 0).toFixed(2)}`

// Caps are entered and shown in tokens: "80M", "500k" or a number.
function fmtTokens(n?: number): string {
  const v = n || 0
  for (const [unit, size] of [['B', 1e9], ['M', 1e6], ['k', 1e3]] as const) {
    if (v >= size) return `${(v / size).toFixed(1).replace(/\.0$/, '')}${unit}`
  }
  return String(Math.round(v))
}

function parseTokens(s: string): number | null {
  const m = s.replace(/[_\s]/g, '').match(/^(\d+(?:[.,]\d+)?)([kKmMbBgG]?)$/)
  if (!m) return null
  const unit = { '': 1, k: 1e3, m: 1e6, b: 1e9, g: 1e9 }[m[2].toLowerCase() as '' | 'k' | 'm' | 'b' | 'g']
  return Math.round(parseFloat(m[1].replace(',', '.')) * unit)
}

const TOKEN_KEYS = ['attempt_tokens', 'budget_tokens']

// A template's defaults as form text, token caps written short ("15M").
const formOf = (defaults: Record<string, string | number>) => Object.fromEntries(Object.entries(defaults).map(
  ([k, v]) => [k, TOKEN_KEYS.includes(k) && typeof v === 'number' ? fmtTokens(v) : String(v)]))

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

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60)

type Families = Record<string, { lessons: number; runs: number }>

// The kind of task a run's lessons are kept under: runs of the same kind learn from each other.
function FamilyField({ value, onChange, fallback, families }: {
  value: string; onChange: (v: string) => void; fallback: string; families: Families
}) {
  const known = families[slug(value) || fallback]
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-400">
      Kind of task
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={fallback}
        className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600" />
      <span className="text-[11px] text-zinc-500">
        {known ? `${known.lessons} lesson${known.lessons === 1 ? '' : 's'} from ${known.runs} earlier run${known.runs === 1 ? '' : 's'}`
          : 'nothing learned yet'}
      </span>
    </label>
  )
}

function TokenField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const n = parseTokens(value)
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-400">
      {label}
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder="e.g. 15M"
        className={`rounded border bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 ${
          value.trim() && n === null ? 'border-red-500/70' : 'border-zinc-700'}`} />
      <span className="text-[11px] text-zinc-500">
        {n === null ? 'Write 80M, 500k or a number' : `${n.toLocaleString()} tokens`}
      </span>
    </label>
  )
}

// How many attempts run at once: each kind has its own ceiling on this deck.
function AtOnce({ value, onChange, most }: { value: string; onChange: (v: string) => void; most: number }) {
  return <TextField label={`At once (up to ${most})`} value={value} onChange={onChange} />
}

function StartRun({ onStarted }: { onStarted: (id: string) => void }) {
  const [templates, setTemplates] = useState<{
    mrav: MravTemplate; triage: TriageTemplate; outcome: OutcomeTemplate; document: DocumentTemplate; code: CodeTemplate
  } | null>(null)
  const [which, setWhich] = useState<'outcome' | 'document' | 'triage' | 'code' | 'mrav'>('document')
  const [canStart, setCanStart] = useState(false)
  const [error, setError] = useState('')
  const [families, setFamilies] = useState<Families>({})

  useEffect(() => {
    captainSwarmRuns.templates().then((t) => {
      setTemplates(t.templates)
      setCanStart(t.can_start)

    }).catch((e) => setError(e instanceof Error ? e.message : String(e)))
    captainSwarmRuns.lessons().then((l) => setFamilies(
      Object.fromEntries(l.families.map((f) => [f.family, { lessons: f.lessons, runs: f.runs }]))))
      .catch(() => setFamilies({}))
  }, [])

  if (!templates) return error ? <div className="text-xs text-red-400">{error}</div> : null
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="mb-3 inline-flex flex-wrap overflow-hidden rounded-md border border-zinc-700 text-xs">
        {([['document', 'Write or research'], ['outcome', 'Best result for a task'], ['triage', 'Rules from your habits'],
          ['code', 'Any repository'], ['mrav', 'Mrav eval']] as const).map(([k, label]) => (
          <button key={k} onClick={() => setWhich(k)}
            className={`px-2.5 py-1 ${which === k ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800'}`}>
            {label}
          </button>
        ))}
      </div>
      {which === 'document' && <StartDocument tpl={templates.document} families={families} onStarted={onStarted} />}
      {which === 'outcome' && <StartOutcome tpl={templates.outcome} families={families} onStarted={onStarted} />}
      {which === 'code' && <StartCode tpl={templates.code} families={families} canStart={canStart} onStarted={onStarted} />}
      {which === 'triage' && <StartTriage tpl={templates.triage} families={families} onStarted={onStarted} />}
      {which === 'mrav' && <StartMrav tpl={templates.mrav} families={families} canStart={canStart} onStarted={onStarted} />}
    </div>
  )
}

function useStart(template: 'mrav' | 'triage' | 'outcome' | 'document', onStarted: (id: string) => void) {
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

function StartOutcome({ tpl, families, onStarted }: {
  tpl: OutcomeTemplate; families: Families; onStarted: (id: string) => void
}) {
  const [form, setForm] = useState<Record<string, string>>(
    () => ({ task: '', max_value: '', must_include: '', currency: '', ...formOf(tpl.defaults) }))
  const { busy, error, start } = useStart('outcome', onStarted)
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  return (
    <>
      <div className="text-sm font-medium text-zinc-100">{tpl.title}</div>
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">{tpl.about}</p>
      <label className="mt-3 flex flex-col gap-1 text-xs text-zinc-400">
        What to find
        <textarea value={form.task} onChange={(e) => set('task')(e.target.value)} rows={2}
          placeholder="The cheapest new paperback copy of The Pragmatic Programmer, 20th Anniversary Edition, shipped to Croatia"
          className="resize-y rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600" />
      </label>
      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <TextField label="Ranked by" value={form.label || ''} onChange={set('label')} />
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Better is
          <select value={form.direction || 'min'} onChange={(e) => set('direction')(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
            <option value="min">lower</option>
            <option value="max">higher</option>
          </select>
        </label>
        <TextField label="At most (optional)" value={form.max_value || ''} onChange={set('max_value')} />
        <TextField label="Currency (optional)" value={form.currency || ''} onChange={set('currency')} />
        <TextField label="Page must say (comma-separated)" value={form.must_include || ''} onChange={set('must_include')} wide />
        <TokenField label="Most tokens per attempt" value={form.attempt_tokens || ''} onChange={set('attempt_tokens')} />
        <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
        <AtOnce value={form.parallel || ''} onChange={set('parallel')} most={tpl.max_parallel} />
        <TokenField label="Token budget" value={form.budget_tokens || ''} onChange={set('budget_tokens')} />
        <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
        <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
        <FamilyField value={form.family || ''} onChange={set('family')} families={families}
          fallback={slug(`find-${form.label || 'price'}`)} />
      </div>
      <p className="mt-2 text-xs text-zinc-500">
        Attempts that run at once share what's left of the budget equally, and each uses its own sites: only one
        at a time may use a site or an account in its browser. Every attempt is a Bat run, so they count toward
        your Bat runs.
      </p>
      <div className="mt-3 flex items-center gap-3">
        <button onClick={() => start(form)} disabled={busy || form.task.trim().length < 10}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start the run
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </>
  )
}

// One document — writing or research — tried several ways, each graded against a checklist from the brief.
function StartDocument({ tpl, families, onStarted }: {
  tpl: DocumentTemplate; families: Families; onStarted: (id: string) => void
}) {
  const [form, setForm] = useState<Record<string, string>>(
    () => ({ task: '', audience: '', criteria: '', ...formOf(tpl.defaults) }))
  const { busy, error, start } = useStart('document', onStarted)
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  const sources = form.sources !== '0'
  return (
    <>
      <div className="text-sm font-medium text-zinc-100">{tpl.title}</div>
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">{tpl.about}</p>
      <label className="mt-3 flex flex-col gap-1 text-xs text-zinc-400">
        The brief
        <textarea value={form.task} onChange={(e) => set('task')(e.target.value)} rows={3}
          placeholder="A two-page brief for our studio on what the EU AI Act asks of a small design agency that uses AI image tools, and what we should do this quarter"
          className="resize-y rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600" />
      </label>
      <label className="mt-3 flex flex-col gap-1 text-xs text-zinc-400">
        It must (one per line, optional)
        <textarea value={form.criteria} onChange={(e) => set('criteria')(e.target.value)} rows={2}
          placeholder={'Name the deadlines that apply to us\nEnd with a checklist we can act on'}
          className="resize-y rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600" />
      </label>
      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <TextField label="Written for (optional)" value={form.audience || ''} onChange={set('audience')} wide />
        <TextField label="About how many words (0 = any)" value={form.words || ''} onChange={set('words')} />
        <label className="flex items-center gap-2 self-end pb-1.5 text-xs text-zinc-300">
          <input type="checkbox" checked={sources} onChange={(e) => set('sources')(e.target.checked ? '1' : '0')} />
          Research: back claims with sources
        </label>
        <TokenField label="Most tokens per attempt" value={form.attempt_tokens || ''} onChange={set('attempt_tokens')} />
        <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
        <AtOnce value={form.parallel || ''} onChange={set('parallel')} most={tpl.max_parallel} />
        <TokenField label="Token budget" value={form.budget_tokens || ''} onChange={set('budget_tokens')} />
        <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
        <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
        <FamilyField value={form.family || ''} onChange={set('family')} families={families}
          fallback={sources ? 'research' : 'writing'} />
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button onClick={() => start(form)} disabled={busy || form.task.trim().length < 15}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start the run
        </button>
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </>
  )
}

const lines = (s: string) => s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean)

// Any repository with a command that prints a number: the objective, written out.
function StartCode({ tpl, families, canStart, onStarted }: {
  tpl: CodeTemplate; families: Families; canStart: boolean; onStarted: (id: string) => void
}) {
  const [form, setForm] = useState<Record<string, string>>(
    () => ({ repo_path: '', goal: '', editable: '', protected: '', run: '', read: '', direction: 'max',
      holdout_run: '', holdout_file: '', target: '', ...formOf(tpl.defaults) }))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  const name = form.repo_path.split('/').filter(Boolean).pop() || 'the repository'
  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await captainSwarmRuns.startObjective({
        kind: 'code', family: form.family || 'code', title: `${form.read || 'The number'} in ${name}`, goal: form.goal,
        subject: { repo_path: form.repo_path.trim(), editable: lines(form.editable), protected: lines(form.protected),
          snapshot: { paths: [], exclude: [] } },
        metric: { kind: 'command', run: form.run.trim(), read: form.read.trim(), direction: form.direction,
          timeout_s: 3600 },
        holdout: { run: form.holdout_run.trim(), file: form.holdout_file.trim() },
        guardrails: [], noise: { min_gain: Number(form.min_gain || 0.01) },
        lane: { kind: 'crew', harness: form.harness || 'claude', per_round: Number(form.per_round || 2),
          parallel: Number(form.parallel || 1) },
        budget: { tokens: form.budget_tokens, hours: Number(form.budget_hours),
          max_experiments: Number(form.budget_max_experiments) },
        stop: { target: form.target.trim() === '' ? null : Number(form.target), stall_rounds: 3, remeasure_every: 3 },
      })
      onStarted(r.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <div className="text-sm font-medium text-zinc-100">{tpl.title}</div>
      <p className="mt-1 text-xs leading-relaxed text-zinc-400">{tpl.about}</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <TextField label="Repository (full path on this computer)" value={form.repo_path} onChange={set('repo_path')} wide />
        <TextField label="What to improve, in a sentence" value={form.goal} onChange={set('goal')} wide />
        <TextField label="May change (paths, comma-separated)" value={form.editable} onChange={set('editable')} wide />
        <TextField label="Never change (paths)" value={form.protected} onChange={set('protected')} wide />
      </div>
      <label className="mt-3 flex flex-col gap-1 text-xs text-zinc-400">
        The measuring command: it prints JSON; {'{checkout}'} is the copy being measured
        <input value={form.run} onChange={(e) => set('run')(e.target.value)}
          placeholder="cd {checkout} && npx vitest run --reporter=json --outputFile=/dev/stdout"
          className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-xs text-zinc-100 placeholder:text-zinc-600" />
      </label>
      <div className="mt-3 grid gap-3 sm:grid-cols-4">
        <TextField label="The number in its JSON (e.g. numPassedTests)" value={form.read} onChange={set('read')} />
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Better is
          <select value={form.direction} onChange={(e) => set('direction')(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
            <option value="max">higher</option>
            <option value="min">lower</option>
          </select>
        </label>
        <TextField label="Smallest gain that counts" value={form.min_gain || ''} onChange={set('min_gain')} />
        <TextField label="Target (optional)" value={form.target} onChange={set('target')} />
        <TextField label="Held-out command (optional, may use {holdout})" value={form.holdout_run} onChange={set('holdout_run')} wide />
        <TextField label="Held-out cases file (outside the repository)" value={form.holdout_file} onChange={set('holdout_file')} wide />
        <label className="flex flex-col gap-1 text-xs text-zinc-400">
          Coding worker
          <select value={form.harness || 'claude'} onChange={(e) => set('harness')(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100">
            {['claude', 'codex', 'opencode', 'crush', 'pi'].map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </label>
        <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
        <AtOnce value={form.parallel || ''} onChange={set('parallel')} most={tpl.max_parallel} />
        <TokenField label="Token budget" value={form.budget_tokens || ''} onChange={set('budget_tokens')} />
        <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
        <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
        <FamilyField value={form.family || ''} onChange={set('family')} families={families} fallback="code" />
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button onClick={start} disabled={busy || !canStart || !form.repo_path.trim() || !form.run.trim() || !form.read.trim()}
          className="inline-flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-50">
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start the run
        </button>
        {!canStart && <span className="text-xs text-zinc-500">Measuring runs a command on this deck: admins only.</span>}
        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>
    </>
  )
}

function StartTriage({ tpl, families, onStarted }: {
  tpl: TriageTemplate; families: Families; onStarted: (id: string) => void
}) {
  const processes = useProcessStore((s) => s.processes)
  const fetchProcesses = useProcessStore((s) => s.fetchProcesses)
  const [form, setForm] = useState<Record<string, string>>(
    () => formOf(tpl.defaults))
  const { busy, error, start } = useStart('triage', onStarted)
  useEffect(() => { fetchProcesses() }, [fetchProcesses])
  useEffect(() => {
    setForm((f) => ({ ...f, set_id: f.set_id || tpl.sets[0]?.id || '', agent: f.agent || processes[0]?.slug || '' }))
  }, [tpl.sets, processes])
  const set = (k: string) => (v: string) => setForm((f) => ({ ...f, [k]: v }))
  const agent = processes.find((p) => p.slug === form.agent)
  const calendar = tpl.sets.find((x) => x.id === form.set_id)?.kind === 'calendar_invites'
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
            <AtOnce value={form.parallel || ''} onChange={set('parallel')} most={tpl.max_parallel} />
            <TokenField label="Token budget" value={form.budget_tokens || ''} onChange={set('budget_tokens')} />
            <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
            <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
            <FamilyField value={form.family || ''} onChange={set('family')} families={families}
              fallback={calendar ? 'calendar-invites' : 'mail-triage'} />
          </div>
          <p className="mt-2 text-xs text-zinc-500">
            Each measurement asks {agent?.name || 'the agent'} {Number(form.cases || 0) * Number(form.repeat || 0)} questions
            on its own model; a model also reads your open cases to suggest rules. The agent answers one question at
            a time, so ideas tried at once are measured in turn; with three or more, a quick look on a third of the
            cases sends only the better half on.
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

function StartMrav({ tpl, families, canStart, onStarted }: {
  tpl: MravTemplate; families: Families; canStart: boolean; onStarted: (id: string) => void
}) {
  const [form, setForm] = useState<Record<string, string>>(
    () => formOf(tpl.defaults))
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
        <TokenField label="Token budget" value={form.budget_tokens || ''} onChange={set('budget_tokens')} />
        <TextField label="Hours" value={form.budget_hours || ''} onChange={set('budget_hours')} />
        <TextField label="Most attempts" value={form.budget_max_experiments || ''} onChange={set('budget_max_experiments')} />
        <TextField label="Ideas per round" value={form.per_round || ''} onChange={set('per_round')} />
        <AtOnce value={form.parallel || ''} onChange={set('parallel')} most={tpl.max_parallel} />
        <FamilyField value={form.family || ''} onChange={set('family')} families={families} fallback="mrav" />
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
  // "Nothing found" (a one-task run's start, an attempt that verified nothing) isn't a point on it.
  const real = (v?: number) => v !== undefined && Math.abs(v) < 1e11
  const points = run.ledger.filter((e) => e.dev?.stats && e.dev.stage !== 'slice' && real(e.dev.stats.mean))
  const start = run.baseline?.dev?.stats?.mean
  const base = real(start) ? start : undefined
  if (start === undefined || points.length === 0) return null
  const values = [...(base === undefined ? [] : [base]), ...points.map((e) => e.dev.stats!.mean)]
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
  const steps: string[] = base === undefined ? [] : [`M 24 ${y(base)}`]
  points.forEach((e, i) => {
    if (e.status === 'kept') {
      const v = e.dev.stats!.mean
      steps.push(...(best === undefined ? [`M ${x(i)} ${y(v)}`] : [`L ${x(i)} ${y(best)}`, `L ${x(i)} ${y(v)}`]))
      best = v
    }
  })
  if (best !== undefined) steps.push(`L ${W - 8} ${y(best)}`)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full" role="img"
      aria-label="Each attempt's score on the open cases, and the best so far">
      {base !== undefined && (
        <>
          <line x1={24} x2={W - 8} y1={y(base)} y2={y(base)} stroke="#52525b" strokeDasharray="3 3" />
          <text x={26} y={y(base) - 4} fill="#71717a" fontSize="10">start {base.toFixed(3)}</text>
        </>
      )}
      <path d={steps.join(' ')} fill="none" stroke="#34d399" strokeWidth={1.5} />
      {points.map((e, i) => (
        <circle key={e.id} cx={x(i)} cy={y(e.dev.stats!.mean)} r={4} fill={MARK[e.status] || '#a1a1aa'}>
          <title>{`#${e.round}.${e.n} ${EXP_TEXT[e.status] || e.status}: ${e.dev.stats!.mean.toFixed(3)}`}</title>
        </circle>
      ))}
    </svg>
  )
}

function Ledger({ exps, intents }: { exps: Experiment[]; intents?: RunDetail['intents'] }) {
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
          <th className="pb-1 font-medium">Tokens</th>
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
                  {(e.change?.sites || []).length > 0 && <div>Sites: {e.change!.sites!.join(', ')}</div>}
                  {e.change?.again_of && <div>Tried again on top of the new best, after its wave.</div>}
                  {e.dev?.slice !== undefined && <div>Quick look first: {f3(e.dev.slice)}</div>}
                  {e.verdict_reason && <div>Verdict: {e.verdict_reason}</div>}
                  {(intents?.[e.id] || []).length > 0 && (
                    <div>
                      <div className="text-zinc-500">Recorded instead of done:</div>
                      {(intents?.[e.id] || []).slice(0, 12).map((i, k) => (
                        <div key={k} className={i.kind === 'commit' ? 'text-amber-300/90' : 'text-zinc-400'}>
                          {i.kind === 'commit' ? 'stopped at' : 'intent'}: {i.what}
                        </div>
                      ))}
                    </div>
                  )}
                  {e.lesson && <div>Lesson: {e.lesson}</div>}
                  {e.diff_stat && <div>Change: {e.diff_stat}{e.lane === 'crew' ? ` · ${e.changed_files.join(', ')}` : ''}</div>}
                  {e.head_sha && <div className="font-mono text-[11px] text-zinc-500">{e.head_sha.slice(0, 12)}</div>}
                </div>
              )}
            </td>
            <td className={`whitespace-nowrap py-2 pr-3 ${EXP_COLOR[e.status] || 'text-sky-400'}`}>
              {LIVE_EXP.includes(e.status) && (
                <Loader2 size={11} className="mr-1 inline animate-spin" />
              )}
              {EXP_TEXT[e.status] || e.status}
            </td>
            <td className="whitespace-nowrap py-2 pr-3 text-zinc-300">
              {e.dev?.stage === 'slice' ? <span className="text-zinc-500">quick {f3(e.dev.value)}</span>
                : e.dev?.stats ? <>{f3(e.dev.stats.mean)} <span className="text-zinc-500">± {f3(e.dev.stats.stdev)}</span></> : '—'}
            </td>
            <td className="py-2 pr-3 text-zinc-300">{e.holdout?.stats ? f3(e.holdout.stats.mean) : '—'}</td>
            <td className="py-2 text-zinc-400" title={e.cost_usd ? `about ${usd(e.cost_usd)}` : undefined}>
              {e.tokens ? fmtTokens(e.tokens) : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// The wave under way: each attempt's lane, its turn in the measurement queue, and the
// sites or accounts it holds in its browser right now.
function InFlight({ run }: { run: RunDetail }) {
  const live = run.live
  if (!live) return null
  const wave = live.wave.map((id) => run.ledger.find((e) => e.id === id)).filter((e): e is Experiment => !!e)
  const best = live.measuring['']
  if (wave.length === 0 && !best) return null
  const queue = (id: string) => live.measuring[id] === 'waiting' ? 'waiting to be measured'
    : live.measuring[id] === 'measuring' ? 'being measured' : ''
  return (
    <div className="rounded-md bg-zinc-950/60 p-3 text-xs">
      <div className="mb-1.5 text-zinc-500">
        {wave.length > 1 ? `${wave.length} attempts at once` : 'In flight'}
        {live.queued > 0 ? ` · ${live.queued} more idea${live.queued === 1 ? '' : 's'} waiting` : ''}
      </div>
      {best && <div className="text-zinc-400">The best so far is {best === 'waiting' ? 'waiting to be measured' : 'being measured'}.</div>}
      <ul className="space-y-1">
        {wave.map((e) => (
          <li key={e.id} className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-zinc-500">{e.round}.{e.n}</span>
            <span className="min-w-0 flex-1 truncate text-zinc-200">{e.hypothesis}</span>
            <span className={EXP_COLOR[e.status] || 'text-sky-400'}>
              {LIVE_EXP.includes(e.status) && <Loader2 size={11} className="mr-1 inline animate-spin" />}
              {queue(e.id) || EXP_TEXT[e.status] || e.status}
            </span>
            {(live.sites[e.id] || []).map((s) => (
              <span key={s} className="rounded border border-zinc-700 px-1 text-[11px] text-zinc-400">
                {s.startsWith('account:') ? `account ${s.slice(8)}` : s}
              </span>
            ))}
          </li>
        ))}
      </ul>
    </div>
  )
}

// A document run's best document, how it did on each checklist item, and its sources.
function DocumentPanel({ doc }: { doc: NonNullable<RunDetail['document']> }) {
  const [open, setOpen] = useState(false)
  const best = doc.best
  return (
    <div className="space-y-2 rounded-md bg-zinc-950/60 p-3 text-xs">
      <div className="flex flex-wrap items-center gap-2 text-zinc-500">
        {best ? `The best document so far · ${best.words} words` : 'No document has passed its checks yet.'}
        {doc.saved && <span className="text-sky-300">saved to {doc.saved}</span>}
      </div>
      {best && (
        <>
          <div className={`whitespace-pre-wrap rounded border border-zinc-800 bg-zinc-900/60 p-3 text-[13px] leading-relaxed text-zinc-200 ${open ? '' : 'max-h-72 overflow-hidden'}`}>
            {best.text}{best.cut ? '\n…' : ''}
          </div>
          <button onClick={() => setOpen((v) => !v)} className="text-sky-400 hover:underline">
            {open ? 'Show less' : 'Read it all'}
          </button>
        </>
      )}
      {doc.checklist.length > 0 && (
        <details>
          <summary className="cursor-pointer text-zinc-500">
            The checklist it is graded on ({doc.checklist.length} items, made from your brief; {doc.held_items} more
            held back, unseen by anyone planning)
          </summary>
          <ul className="mt-1 space-y-0.5">
            {doc.checklist.map((c, i) => (
              <li key={i} className={c.grade === null ? 'text-zinc-400' : c.grade >= 1 ? 'text-emerald-300/90' : c.grade > 0 ? 'text-amber-300/90' : 'text-zinc-500'}>
                {c.grade === null ? '·' : c.grade >= 1 ? '✓' : c.grade > 0 ? '½' : '✗'} {c.item}
              </li>
            ))}
          </ul>
        </details>
      )}
      {best && best.sources.length > 0 && (
        <details>
          <summary className="cursor-pointer text-zinc-500">
            Sources Captain opened: {best.sources.filter((x) => x.ok).length} of {best.sources.length} hold
          </summary>
          <ul className="mt-1 space-y-0.5">
            {best.sources.map((x) => (
              <li key={x.url} className={x.ok ? 'text-zinc-300' : 'text-zinc-500'}>
                {x.ok ? '✓' : '✗'}{' '}
                <a href={x.url} target="_blank" rel="noreferrer" className="break-all text-sky-400 hover:underline">{x.url}</a>
                {x.via === 'browser' ? ' (seen in a browser)' : ''}{!x.ok && x.why ? ` — ${x.why}` : ''}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

// What the run learned (Phase 6): what it started with from earlier runs of the same kind of
// task, the planner's readings, a strategy round's directions, and how each strategy has done.
function Learning({ run }: { run: RunDetail }) {
  const l = run.learning
  const [earlier, setEarlier] = useState<Lesson[] | null>(null)
  const [error, setError] = useState('')
  if (!l) return null
  const nothing = !l.earlier.lessons && !l.readings.length && !l.strategy.length && !l.weights.length && !l.playbook
  if (nothing) return null
  const show = async () => {
    try {
      setEarlier(((await captainSwarmRuns.lessons(l.family)).lessons || []).filter((x) => x.run_id !== run.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const forget = async (lesson: Lesson) => {
    try {
      await captainSwarmRuns.forget(lesson.id)
      setEarlier((xs) => (xs || []).filter((x) => x.id !== lesson.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  return (
    <div className="space-y-2 rounded-md bg-zinc-950/60 p-3 text-xs">
      <div className="flex flex-wrap items-center gap-2 text-zinc-500">
        What it learned
        <span className="rounded border border-zinc-700 px-1 text-[11px] text-zinc-400">{l.family}</span>
        {l.earlier.lessons > 0 && (
          <button onClick={() => (earlier ? setEarlier(null) : show())} className="text-sky-400 hover:underline">
            started from {l.earlier.lessons} lessons of {l.earlier.runs} earlier run{l.earlier.runs === 1 ? '' : 's'}
          </button>
        )}
      </div>
      {error && <div className="text-red-400">{error}</div>}
      {earlier && (
        <ul className="max-h-56 space-y-0.5 overflow-y-auto">
          {earlier.map((x) => (
            <li key={x.id} className="flex gap-2">
              <span className={x.outcome === 'won' ? 'text-emerald-400' : x.outcome === 'lost' ? 'text-zinc-500' : 'text-sky-400'}>
                {x.kind === 'attempt' ? (x.outcome === 'won' ? 'worked' : x.outcome === 'lost' ? "didn't" : 'tried') : x.kind === 'strategy' ? 'direction' : 'reading'}
              </span>
              <span className="min-w-0 flex-1 text-zinc-300">{x.text}</span>
              <button onClick={() => forget(x)} title="Forget this lesson: no later run reads it"
                className="shrink-0 text-zinc-500 hover:text-red-400"><X size={12} /></button>
            </li>
          ))}
        </ul>
      )}
      {l.strategy.length > 0 && (
        <div>
          <div className="text-zinc-500">It stalled; a strategy round suggested:</div>
          {l.strategy.map((x, i) => <div key={i} className="text-amber-200/90">→ {x.text}</div>)}
        </div>
      )}
      {l.readings.length > 0 && (
        <div>
          <div className="text-zinc-500">The planner's readings, newest first:</div>
          {l.readings.slice(0, 4).map((x, i) => <div key={i} className="text-zinc-300">{x.text}</div>)}
        </div>
      )}
      {l.weights.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {l.weights.map((w) => (
            <span key={w.tag} title={`won ${w.won} of ${w.won + w.lost} on this kind of task`}
              className="rounded border border-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-400">
              {w.tag} <span className={w.weight >= 0.5 ? 'text-emerald-400' : 'text-zinc-500'}>{w.won}/{w.won + w.lost}</span>
            </span>
          ))}
        </div>
      )}
      {l.playbook && <div className="text-sky-300">Saved as a playbook on {l.playbook.agent}.</div>}
    </div>
  )
}

// A one-task run's ways that worked and didn't, as a playbook on one of the owner's agents.
function PlaybookPanel({ id, onSaved }: { id: string; onSaved: (agent: string) => void }) {
  const processes = useProcessStore((s) => s.processes)
  const fetchProcesses = useProcessStore((s) => s.fetchProcesses)
  const [playbook, setPlaybook] = useState<Playbook | null>(null)
  const [agent, setAgent] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    fetchProcesses()
    captainSwarmRuns.playbook(id).then((r) => setPlaybook(r.playbook))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [id, fetchProcesses])
  const chosen = agent || processes[0]?.slug || ''
  const save = async () => {
    setBusy(true)
    setError('')
    try {
      await captainSwarmRuns.savePlaybook(id, chosen)
      onSaved(processes.find((p) => p.slug === chosen)?.name || chosen)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="space-y-2 rounded-md border border-zinc-800 bg-zinc-950/60 p-3 text-xs">
      <div className="text-zinc-400">
        The agent reads this when it meets a task like this one. Nothing is saved until you click Save.
      </div>
      {playbook ? (
        <div className="space-y-1 text-zinc-300">
          <div className="text-sm text-zinc-100">{playbook.name}</div>
          <div className="text-zinc-500">{playbook.trigger_description}</div>
          {playbook.do_pattern && <pre className="whitespace-pre-wrap text-emerald-300/90">{playbook.do_pattern}</pre>}
          {playbook.dont_pattern && <pre className="whitespace-pre-wrap text-zinc-400">{playbook.dont_pattern}</pre>}
        </div>
      ) : !error && <Loader2 size={14} className="animate-spin text-zinc-500" />}
      <div className="flex flex-wrap items-center gap-2">
        <select value={chosen} onChange={(e) => setAgent(e.target.value)}
          className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-100">
          {processes.map((p) => <option key={p.slug} value={p.slug}>{p.name}</option>)}
        </select>
        <button onClick={save} disabled={busy || !playbook || !chosen}
          className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 font-medium text-white hover:bg-sky-500 disabled:opacity-50">
          {busy ? <Loader2 size={12} className="animate-spin" /> : <BookMarked size={12} />} Save
        </button>
        {processes.length === 0 && <span className="text-zinc-500">You have no agents to give it to.</span>}
        {error && <span className="text-red-400">{error}</span>}
      </div>
    </div>
  )
}

function RunView({ id, onChanged }: { id: string; onChanged: () => void }) {
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [showPlaybook, setShowPlaybook] = useState(false)

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
          {!active && run.improved && run.kind === 'outcome' && !run.outcome?.carried_out && (
            <button onClick={() => act(async () => {
              const r = await captainSwarmRuns.carryOut(id)
              setNote(`Carrying it out once: Bat run ${r.bat_run}. Bat asks for your Go before it spends, sends or signs up.`)
            })}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-500">
              <Hand size={13} /> Carry it out
            </button>
          )}
          {!active && run.improved && run.kind === 'document' && !run.document?.saved && (
            <button onClick={() => act(async () => {
              const r = await captainSwarmRuns.saveDocument(id)
              setNote(`Saved to your files: ${r.saved}`)
            })}
              className="inline-flex items-center gap-1 rounded-md bg-sky-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-sky-500">
              <FileDown size={13} /> Save to my files
            </button>
          )}
          {!active && run.improved && (run.kind === 'outcome' || run.kind === 'document') && !run.learning?.playbook && (
            <button onClick={() => setShowPlaybook((v) => !v)}
              className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800">
              <BookMarked size={13} /> Save as a playbook
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
      {showPlaybook && (
        <PlaybookPanel id={id} onSaved={(agent) => { setShowPlaybook(false); setNote(`Saved as a playbook on ${agent}.`); load() }} />
      )}
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
          <div className="text-zinc-500">Used</div>
          {b?.usd && !b.tokens ? (
            <>
              <div className="mt-0.5 text-sm text-zinc-200">{usd(run.spent_usd)} of {usd(b.usd)}</div>
              <div className="text-zinc-500">{fmtTokens(run.spent_tokens)} tokens</div>
            </>
          ) : (
            <>
              <div className="mt-0.5 text-sm text-zinc-200">
                {fmtTokens(run.spent_tokens)}{b?.tokens ? ` of ${fmtTokens(b.tokens)}` : ''} tokens
              </div>
              <div className="text-zinc-500">about {usd(run.spent_usd)}</div>
            </>
          )}
        </div>
      </div>
      <Chart run={run} />
      {run.kind === 'outcome' && run.outcome && (
        <div className="space-y-2 rounded-md bg-zinc-950/60 p-3 text-xs">
          <div className="text-zinc-500">Best result Captain could check itself</div>
          {run.outcome.best ? (
            <div className="text-zinc-200">
              <span className="text-sm text-emerald-300">{run.outcome.best.value} {run.outcome.best.currency}</span>
              {' '}· {run.outcome.best.title}{' '}
              <a href={run.outcome.best.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 text-sky-400 hover:underline">
                page <ExternalLink size={11} />
              </a>
              {run.outcome.best.next_step && <div className="text-zinc-500">Would commit by: {run.outcome.best.next_step}</div>}
            </div>
          ) : <div className="text-zinc-500">Nothing verified yet.</div>}
          {run.outcome.checks.length > 0 && (
            <ul className="space-y-0.5">
              {run.outcome.checks.map((c) => (
                <li key={c.url} className={c.verified ? 'text-zinc-300' : 'text-zinc-500'}>
                  {c.verified ? '✓' : '✗'} {c.value} {c.currency} · {c.title.slice(0, 80)}
                  {c.via === 'browser' ? ' (seen in a browser)' : ''}
                  {!c.verified && c.why ? ` — ${c.why}` : ''}
                </li>
              ))}
            </ul>
          )}
          {run.outcome.carried_out && <div className="text-sky-300">Being carried out by Bat run {run.outcome.carried_out}.</div>}
        </div>
      )}
      {run.kind === 'document' && run.document && <DocumentPanel doc={run.document} />}
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
      {active && <InFlight run={run} />}
      <Learning run={run} />
      <Ledger exps={run.ledger} intents={run.intents} />
      <details className="text-xs text-zinc-400">
        <summary className="cursor-pointer text-zinc-500">What it measures and what it may change</summary>
        <div className="mt-2 space-y-1">
          <div>{run.objective.goal}</div>
          <div>
            {(run.objective.lane.parallel || 1) > 1
              ? `Up to ${run.objective.lane.parallel} attempts at once, all started from the best so far; one keeper per wave.`
              : 'One attempt at a time, each started from the best so far.'}
          </div>
          {run.kind === 'code' ? (
            <>
              <div>Repository: {run.objective.subject.repo_path}</div>
              <div>May change: {(run.objective.subject.editable || []).join(', ')}</div>
              <div>Never changes: {(run.objective.subject.protected || []).join(', ')}</div>
              <div className="break-all font-mono text-[11px] text-zinc-500">{run.objective.metric.run}</div>
              <div>Coding worker: {run.objective.lane.harness}{run.objective.lane.model ? ` (${run.objective.lane.model})` : ''}</div>
            </>
          ) : run.kind === 'document' ? (
            <>
              <div>Brief: {run.objective.subject.task}</div>
              <div>Each attempt is a Bat run in rehearsal that searches, reads and writes; nothing is sent or published.
                A judge on a model the workers don't use grades every document against a checklist made from the brief,
                twice; a second checklist nobody planning sees confirms a gain
                {run.objective.subject.sources ? ', and Captain opens every source a document cites' : ''}. These are
                model grades, not yours: read the documents.</div>
            </>
          ) : run.kind === 'outcome' ? (
            <>
              <div>Task: {run.objective.subject.task}</div>
              <div>Each attempt is a Bat run in rehearsal: nothing is bought, sent, booked or submitted, and Captain
                opens every result's page itself. Before one becomes the best, a model reviewer reads the task and
                what the page says around the number.</div>
            </>
          ) : (
            <>
              <div>Agent: {run.objective.subject.agent_name} · practice set {run.objective.subject.set_id}</div>
              <div>Each measurement: {run.objective.metric.cases} open cases × {run.objective.metric.repeat} runs, on the
                agent's exam lane (it can't send or change anything). Scored by F1 on
                {run.objective.subject.set_kind === 'calendar_invites' ? ' "you accept"' : ' "needs your reply"'}.</div>
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
        <div className="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[16rem_minmax(0,1fr)]">
          {/* minmax(0, …): a long run title truncates instead of widening the column past the page. */}
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
                    {STATUS_TEXT[r.status]} · {f3(r.baseline_dev)} → {f3(r.champion_dev)} · {r.experiments} attempt{r.experiments === 1 ? '' : 's'}
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
