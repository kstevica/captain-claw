import { useCallback, useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Mountain, Code2, Brain, Boxes, Play, Loader2, CheckCircle2, XCircle, AlertTriangle,
  Square, Trash2, Download, Eye, FileText, Maximize2, Minimize2, X,
} from 'lucide-react'
import {
  getTiers, startCoder, startReason, startIntent, getTargets, getRun, getProgress, listRuns,
  stopRun, cleanupAgents, fetchRunFileText, downloadRunFile,
  type Track, type Tier, type DubinaRun, type TargetOption, type ProgressEvent, type RunFile,
} from '../services/dubinaApi'

const POLL_MS = 1200

// What each track is and — crucially — how its verifier differs. Shown as a banner
// under the tabs so the three modes are never confused.
const TRACK_INFO: Record<Track, { blurb: string; verifier: string; placeholder: string }> = {
  coder: {
    blurb: 'Writes code and proves it against your project’s real tests.',
    verifier: 'Ground-truth verifier: a step passes only when the tests pass. Escalates to a stronger tier only when tests keep failing — never guesses.',
    placeholder: 'Implement add(a, b) in solution.py so the tests pass…',
  },
  reason: {
    blurb: 'Answers a question and checks itself — no external ground truth.',
    verifier: 'Statistical verifier: samples several answers and passes on self-consistency (agreement). On low agreement, diverse-lens critics must survive before it passes.',
    placeholder: 'Pick ONE data store for an append-only event log and give the single most important reason…',
  },
  intent: {
    blurb: 'Runs the task through one of your agents/archetypes instead of a raw model.',
    verifier: 'Same statistical verifier as Reasoning, but the generator is a tool-using fleet agent. Critics still use a real Library model, so the agent never grades its own answer.',
    placeholder: 'Scan the market for competitors to X; end with one go / no-go / phased recommendation…',
  },
}

// Per-knob explanations of how each control affects the escalation process.
const HINT = {
  base: 'Every step starts here — the cheap substrate that does the bulk of the work.',
  max: 'Ceiling the ladder may climb to when the verifier keeps rejecting. Set equal to base to pin a single tier.',
  budget: 'Spend cap in cost units (each rung up costs ~2× more). Blank = unbounded; on exhaustion you get the best verified-so-far, never a silent truncation.',
  samples: 'Parallel answers drawn when a single pass fails — more samples = stronger self-consistency vote, more cost.',
  fixes: 'Feedback-driven retries at a tier before climbing — the verifier’s feedback is fed into the next attempt.',
  stakes: 'high = always run critics, even when the samples already agree. normal = critics only when agreement is low.',
  threshold: 'Fraction of samples that must agree to pass without critics. Higher = stricter self-consistency.',
  targetReq: 'Which agent or archetype runs the task (its tools & system prompt apply).',
  targetOpt: 'Optional — run via one of your agents/archetypes instead of a raw tier model. Critics still use a real model.',
}

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
  const [runTrack, setRunTrack] = useState<Track>('coder') // track of the displayed run
  const [progress, setProgress] = useState<ProgressEvent[]>([])
  const [history, setHistory] = useState<DubinaRun[]>([])
  const [busy, setBusy] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [cleanMsg, setCleanMsg] = useState('')
  const [error, setError] = useState('')
  const [modal, setModal] = useState<{ title: string; content: string; mode: ViewMode } | null>(null)
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

  // Intent requires a target → seed the first one. Coder/Reason default to the
  // Library tier model (empty target), so don't auto-select there.
  useEffect(() => {
    if (track === 'intent' && !target && targets.length) setTarget(targets[0].value)
  }, [targets, target, track])

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
    setRunTrack(t)
    if (poll.current) clearInterval(poll.current)
    poll.current = setInterval(async () => {
      try {
        const [r, p] = await Promise.all([getRun(t, id), getProgress(t, id).catch(() => null)])
        setRun(r)
        if (p) setProgress(p.events || [])
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
    setProgress([])
    try {
      const computeBudget = budget ? Number(budget) : 0
      let started: { run_id: string; track: Track }
      if (track === 'coder') {
        started = await startCoder({
          task, workspace, test_command: testCommand, solution_path: solutionPath,
          test_path: testPath, spec, base_tier: baseTier, max_tier: maxTier,
          compute_budget: computeBudget, max_step_samples: samples, max_fix_attempts: fixes,
          target,
        })
      } else if (track === 'reason') {
        started = await startReason({
          task, base_tier: baseTier, max_tier: maxTier, compute_budget: computeBudget,
          max_step_samples: samples, max_fix_attempts: fixes, stakes,
          agreement_threshold: threshold, target,
        })
      } else {
        started = await startIntent({
          task, target, base_tier: baseTier, max_tier: maxTier, compute_budget: computeBudget,
          max_step_samples: samples, max_fix_attempts: fixes, stakes, agreement_threshold: threshold,
        })
      }
      watch(started.track, started.run_id)
    } catch (e) {
      setError(String(e))
      setBusy(false)
    }
  }

  const doStop = async () => {
    if (!run) return
    setStopping(true)
    try {
      await stopRun(runTrack, run.id)
      const r = await getRun(runTrack, run.id)
      setRun(r)
    } catch (e) {
      setError(String(e))
    }
    if (poll.current) clearInterval(poll.current)
    setStopping(false)
    setBusy(false)
    refreshHistory()
  }

  const doCleanup = async () => {
    setCleaning(true)
    setCleanMsg('')
    try {
      const r = await cleanupAgents()
      setCleanMsg(r.count ? `Stopped ${r.count} spawned agent${r.count > 1 ? 's' : ''}` : 'No spawned agents to remove')
      getTargets().then(setTargets).catch(() => {})
    } catch (e) {
      setCleanMsg(String(e))
    }
    setCleaning(false)
  }

  const viewFile = async (name: string) => {
    if (!run) return
    const text = await fetchRunFileText(runTrack, run.id, name)
    setModal({ title: name, content: text, mode: viewModeForFile(name) })
  }

  // Export the run as a single Markdown file: result answer/code, the ladder steps,
  // and the full execution log.
  const exportRun = () => {
    if (!run) return
    downloadText(`dubina-${runTrack}-${run.id}.md`, formatRunExport(run, progress))
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
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {cleanMsg && <span className="text-[11px] text-zinc-500">{cleanMsg}</span>}
            <button
              onClick={doCleanup} disabled={cleaning} title="Stop any leftover agents Dubina spawned for archetype runs"
              className="flex items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-[11px] text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-zinc-200 disabled:opacity-40"
            >
              {cleaning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
              Remove spawned agents
            </button>
          </div>
        </div>

        {/* Track tabs */}
        <div className="mb-3 inline-flex items-center gap-1 rounded-xl border border-zinc-800 p-1 text-sm">
          <TrackTab active={track === 'coder'} onClick={() => setTrack('coder')} icon={<Code2 className="h-3.5 w-3.5" />} label="Coder" />
          <TrackTab active={track === 'reason'} onClick={() => setTrack('reason')} icon={<Brain className="h-3.5 w-3.5" />} label="Reasoning" />
          <TrackTab active={track === 'intent'} onClick={() => setTrack('intent')} icon={<Boxes className="h-3.5 w-3.5" />} label="Intent" />
        </div>

        {/* Track explainer — what this track does + how its verifier differs */}
        <div className="mb-4 rounded-lg border border-zinc-800 bg-zinc-900/30 px-3 py-2.5">
          <p className="text-sm text-zinc-300">{TRACK_INFO[track].blurb}</p>
          <p className="mt-1 text-[12px] leading-snug text-zinc-500">{TRACK_INFO[track].verifier}</p>
          <p className="mt-1.5 text-[11px] text-zinc-600">
            Coder = verified by tests · Reasoning = verified by self-consistency · Intent = a tool-using agent, verified by self-consistency
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Task — full width, generous height */}
        <div className="mb-4">
          <Field label="Task / intent" hint="Describe what to produce. The run drives this to a verified result, escalating tiers only when the verifier demands it.">
            <textarea
              value={task} onChange={(e) => setTask(e.target.value)} rows={8}
              placeholder={TRACK_INFO[track].placeholder}
              className="min-h-[11rem] w-full resize-y rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
          </Field>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Left — what runs the task (generator + verifier knobs) */}
          <div className="space-y-3">
            <SectionLabel>Generator</SectionLabel>
            <Field
              label={track === 'intent' ? 'Run target (archetype or live agent)' : 'Run target'}
              hint={track === 'intent' ? HINT.targetReq : HINT.targetOpt}>
              <Select value={target} onChange={setTarget}
                options={track === 'intent'
                  ? (targets.length ? targets.map((t) => [t.value, t.label])
                                    : [['', 'no archetypes or running agents found']])
                  : [['', '— Library tier model —'], ...targets.map((t) => [t.value, t.label] as [string, string])]} />
            </Field>

            {track === 'coder' && (
              <>
                <Field label="Workspace" hint="Project directory the verifier runs the tests in.">
                  <Text value={workspace} onChange={setWorkspace} placeholder="/path/to/project" />
                </Field>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Test command" hint="Run to verify a step."><Text value={testCommand} onChange={setTestCommand} /></Field>
                  <Field label="Solution file" hint="Where generated code is written."><Text value={solutionPath} onChange={setSolutionPath} /></Field>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Test file" hint="Target for spec→tests synthesis."><Text value={testPath} onChange={setTestPath} placeholder="test_solution.py" /></Field>
                  <Field label="Spec" hint="If set and tests are missing, they’re synthesized first."><Text value={spec} onChange={setSpec} placeholder="optional" /></Field>
                </div>
              </>
            )}

            {(track === 'reason' || track === 'intent') && (
              <>
                <SectionLabel>Verifier</SectionLabel>
                <Field label="Stakes" hint={HINT.stakes}>
                  <Select value={stakes} onChange={setStakes} options={[['normal', 'normal'], ['high', 'high (always run critics)']]} />
                </Field>
                <Field label={`Agreement threshold (${threshold.toFixed(2)})`} hint={HINT.threshold}>
                  <input type="range" min={0} max={1} step={0.05} value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-violet-500" />
                </Field>
              </>
            )}
          </div>

          {/* Right — the escalation ladder + budget */}
          <div className="space-y-3">
            <SectionLabel>Escalation ladder</SectionLabel>
            <Field label="Base tier" hint={HINT.base}>
              <Select value={baseTier} onChange={setBaseTier} options={llmTiers.map((t) => [t.id, t.id])} />
            </Field>
            <Field label="Max tier" hint={HINT.max}>
              <Select value={maxTier} onChange={setMaxTier} options={llmTiers.map((t) => [t.id, t.id])} />
            </Field>
            <Field label="Compute budget" hint={HINT.budget}>
              <Text value={budget} onChange={setBudget} placeholder="e.g. 50 (blank = unbounded)" />
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Samples (vote)" hint={HINT.samples}><Num value={samples} onChange={setSamples} /></Field>
              <Field label="Fix attempts" hint={HINT.fixes}><Num value={fixes} onChange={setFixes} /></Field>
            </div>
            <button
              onClick={submit} disabled={!canRun}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {busy ? 'Running…' : 'Run'}
            </button>
            {baseTier && !target && (
              <p className="text-[11px] text-zinc-600">
                {tiers.find((t) => t.id === baseTier)?.description || ''}
              </p>
            )}
          </div>
        </div>

        {/* Live run */}
        {run && (
          <RunView run={run} progress={progress} onStop={doStop}
            stopping={stopping} onView={viewFile} onDownload={(n) => downloadRunFile(runTrack, run.id, n)}
            onExport={exportRun} />
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="mt-8">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-600">
              Recent {track} runs
            </h2>
            <div className="divide-y divide-zinc-900 rounded-xl border border-zinc-900">
              {history.map((h) => (
                <button key={h.id} onClick={() => { setRun(null); setProgress([]); setRunTrack(track); getRun(track, h.id).then(setRun); getProgress(track, h.id).then((p) => setProgress(p.events || [])).catch(() => {}) }}
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
      {modal && <FileModal title={modal.title} content={modal.content} mode={modal.mode} onClose={() => setModal(null)} />}
    </div>
  )
}

// ── Live run view ────────────────────────────────────────────────────

function RunView({ run, progress, onStop, stopping, onView, onDownload, onExport }: {
  run: DubinaRun; progress: ProgressEvent[]
  onStop: () => void; stopping: boolean
  onView: (name: string) => void; onDownload: (name: string) => void; onExport: () => void
}) {
  const summary = run.result || {}
  const answer = (summary.answer as string) || ''
  const code = (summary.code as string) || ''
  const body = code || answer
  const files = (summary.files as RunFile[] | undefined) || []
  const done = run.status !== 'running'
  return (
    <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
      <div className="mb-3 flex items-center gap-2">
        <StatusDot status={run.status} />
        <span className="text-sm font-medium text-zinc-200 capitalize">{run.status}</span>
        {run.stopped_reason && <span className="text-[11px] text-amber-400">({run.stopped_reason})</span>}
        {run.status === 'running' && (
          <button
            onClick={onStop} disabled={stopping} title="Stop this run (and dispose any agents it spawned)"
            className="flex items-center gap-1 rounded-md border border-red-500/30 bg-red-500/10 px-2 py-0.5 text-[11px] text-red-400 transition-colors hover:bg-red-500/20 disabled:opacity-40"
          >
            {stopping ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />}
            Stop
          </button>
        )}
        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={onExport} title="Export the result + execution log as Markdown"
            className="flex items-center gap-1 rounded-md border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-zinc-200"
          >
            <Download className="h-3 w-3" /> Export
          </button>
          <span className="text-[11px] text-zinc-600">
            tier {String(summary.tier_used ?? '—')} · rung {String(summary.rung_reached ?? '—')} · {run.cost_spent.toFixed(0)}u
          </span>
        </div>
      </div>

      {progress.length > 0 && <ExecutionLog events={progress} />}

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

      {/* Generated files — agents that answered by writing documents */}
      {files.length > 0 && (
        <div className="mb-3 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Generated files</div>
          <div className="space-y-1">
            {files.map((f) => (
              <div key={f.name} className="flex items-center gap-2 text-xs">
                <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                <span className="truncate text-zinc-300">{f.name}</span>
                <span className="shrink-0 text-zinc-600">{Math.max(1, Math.round(f.size / 1024))}kb</span>
                <div className="ml-auto flex shrink-0 items-center gap-0.5">
                  {isViewable(f.name) && (
                    <button onClick={() => onView(f.name)} title="View" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <button onClick={() => onDownload(f.name)} title="Download" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                    <Download className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Result / answer */}
      {body ? (
        <div>
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            {code ? 'Result — code' : 'Result — answer'}
          </div>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-zinc-800 bg-zinc-950/40 p-3 text-xs text-zinc-300">{body}</pre>
        </div>
      ) : done && !run.error && (
        <p className="text-xs text-zinc-500">
          No textual answer{files.length ? ' — see generated files above.' : ' was produced.'}
        </p>
      )}
      {run.error && <p className="text-xs text-red-300">{run.error}</p>}
    </div>
  )
}

// ── Files: viewer + export helpers (mirrors Basna's FileModal) ───────

type ViewMode = 'markdown' | 'html' | 'text'

const VIEWABLE_EXTS = new Set([
  'md', 'markdown', 'txt', 'text', 'log', 'html', 'htm', 'json', 'csv', 'tsv',
  'xml', 'yaml', 'yml', 'toml', 'ini', 'py', 'sh', 'bash', 'js', 'mjs', 'ts', 'tsx', 'jsx', 'css', 'sql',
])
function fileExt(name: string): string { const m = name.toLowerCase().match(/\.([a-z0-9]+)$/); return m ? m[1] : '' }
function isViewable(name: string): boolean { return VIEWABLE_EXTS.has(fileExt(name)) }
function viewModeForFile(name: string): ViewMode {
  const e = fileExt(name)
  if (e === 'md' || e === 'markdown') return 'markdown'
  if (e === 'html' || e === 'htm') return 'html'
  return 'text'
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// One Markdown document: header + result + generated-file list + full log.
function formatRunExport(run: DubinaRun, progress: ProgressEvent[]): string {
  const s = run.result || {}
  const body = (s.code as string) || (s.answer as string) || ''
  const files = (s.files as RunFile[] | undefined) || []
  const out: string[] = [
    `# Dubina run — ${run.status}`, '',
    `- Task: ${run.task || '(untitled)'}`,
    `- Ladder: ${run.base_tier} → ${run.max_tier}`,
    `- Tier used: ${String(s.tier_used ?? '—')} · rung ${String(s.rung_reached ?? '—')} · ${run.cost_spent.toFixed(0)}u`,
  ]
  if (run.stopped_reason) out.push(`- Stopped: ${run.stopped_reason}`)
  out.push('')
  if (body) out.push(s.code ? '## Result — code' : '## Result — answer', '', body, '')
  if (run.error) out.push('## Error', '', run.error, '')
  if (files.length) out.push('## Generated files', '', ...files.map((f) => `- ${f.name} (${Math.max(1, Math.round(f.size / 1024))}kb)`), '')
  out.push('## Execution log', '')
  for (const e of progress) {
    const t = e.ts ? new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false }) : ''
    out.push(`- \`${t}\` **${e.stage}** ${e.agent ? `${e.agent}: ` : ''}${e.message}`)
  }
  return out.join('\n')
}

function FileModal({ title, content, mode, onClose }: {
  title: string; content: string; mode: ViewMode; onClose: () => void
}) {
  const [maximized, setMaximized] = useState(false)
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl ${
          maximized ? 'h-[96vh] w-[97vw] max-w-none' : 'max-h-[90vh] w-full max-w-4xl'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="truncate text-sm font-medium text-zinc-200">{title}</span>
          <div className="flex items-center gap-2">
            <button onClick={() => setMaximized((m) => !m)} title={maximized ? 'Restore' : 'Maximise'}
              className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              {maximized ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
            <button onClick={() => downloadText(title, content)}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-2.5 py-1 text-xs text-zinc-200 hover:bg-zinc-800">
              <Download className="h-3.5 w-3.5" /> Download
            </button>
            <button onClick={onClose} className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        {mode === 'html' ? (
          <iframe title={title} sandbox="" srcDoc={content} className="min-h-[60vh] w-full flex-1 rounded-b-xl bg-white" />
        ) : mode === 'markdown' ? (
          <div className="fd-markdown flex-1 overflow-auto p-5 text-sm leading-relaxed text-zinc-200">
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          </div>
        ) : (
          <pre className="flex-1 overflow-auto whitespace-pre-wrap rounded-b-xl p-4 font-mono text-xs text-zinc-300">{content}</pre>
        )}
      </div>
    </div>
  )
}

// ── Execution log (live narration; mirrors Basna's progress panel) ───

const STAGE_COLOR: Record<string, string> = {
  start: 'text-violet-400',
  attempt: 'text-zinc-400',
  action: 'text-sky-500 dark:text-sky-400',
  llm: 'text-amber-600 dark:text-amber-400',
  narration: 'text-zinc-500',
  done: 'text-emerald-500 dark:text-emerald-400',
  error: 'text-red-500 dark:text-red-400',
}

function fmtTime(ts?: number): string {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString([], { hour12: false })
}

// The short tag in the gutter: tool name for an agent action, else the stage.
function badgeFor(e: ProgressEvent): string {
  if (e.stage === 'action') return e.tool || 'tool'
  return e.stage
}

function ExecutionLog({ events }: { events: ProgressEvent[] }) {
  const endRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'nearest' }) }, [events.length])
  return (
    <div className="mb-3">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-zinc-600">Execution log</div>
      <div className="max-h-72 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950/40 p-2 font-mono text-[11px] leading-relaxed">
        {events.map((e) => (
          <div key={e.i} className="flex gap-2">
            <span className="shrink-0 text-zinc-600">{fmtTime(e.ts)}</span>
            <span className={`w-20 shrink-0 truncate ${STAGE_COLOR[e.stage] || 'text-zinc-500'}`} title={badgeFor(e)}>{badgeFor(e)}</span>
            {e.agent && <span className="shrink-0 text-zinc-500">{e.agent}:</span>}
            <span className={e.stage === 'narration' ? 'italic text-zinc-500' : 'text-zinc-400'}>{e.message}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
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

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium text-zinc-500">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] leading-snug text-zinc-600">{hint}</span>}
    </label>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">{children}</div>
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
