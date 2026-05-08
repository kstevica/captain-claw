/**
 * PlanCard — live monitor for an in-flight plan-mode execution.
 *
 * Three regions:
 *   1. Header: status pill, elapsed time, step counter, dismiss
 *   2. Progress bar (verified | completed | failed | pending)
 *   3. Active-step spotlight (latest task_step text + tool + phase)
 *   4. Step list with per-step status, runtime, and expandable detail
 *
 * Driven entirely by the planState reducer in chatStore — see the
 * plan_* and orchestrator_event handlers there.
 */

import { useEffect, useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  RotateCw,
  ListChecks,
  AlertTriangle,
  X,
  Wrench,
  Sparkles,
  Activity,
  FileText,
  Download,
  Eye,
  RefreshCw,
  StopCircle,
} from 'lucide-react'
import { useChatStore, type PlanStep, type PlanState } from '../../stores/chatStore'
import {
  listAgentFilesSince,
  getDownloadUrl,
  formatSize,
  isViewable,
  type AgentFile,
} from '../../services/fileTransfer'
import { FileViewer } from './FileViewer'

interface PlanCardProps {
  containerId: string
}

export function PlanCard({ containerId }: PlanCardProps) {
  const session = useChatStore((s) => s.sessions.get(containerId))
  const togglePlanCardCollapsed = useChatStore((s) => s.togglePlanCardCollapsed)
  const dismissPlan = useChatStore((s) => s.dismissPlan)
  const cancelTask = useChatStore((s) => s.cancelTask)

  // Tab state lives on the component (not in the persisted store) — switching
  // tabs is ephemeral UI; on next plan run we want to default back to Steps.
  const [activeTab, setActiveTab] = useState<'steps' | 'files'>('steps')

  if (!session?.planState) return null

  const plan = session.planState
  const collapsed = session.planCardCollapsed
  const finished = plan.status !== 'running'
  const activeStep = plan.activeStepId ? plan.steps.find((s) => s.id === plan.activeStepId) : undefined

  const counts = aggregateCounts(plan)

  return (
    <div className="mx-3 my-2 flex max-h-[45vh] shrink-0 flex-col overflow-hidden rounded-lg border border-violet-500/30 bg-gradient-to-b from-violet-950/40 to-zinc-950/60 shadow-lg shadow-violet-950/20 dark:border-violet-500/30">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-violet-500/20 bg-violet-900/20 px-3 py-2">
        <button
          onClick={() => togglePlanCardCollapsed(containerId)}
          className="flex flex-1 items-center gap-2 text-left"
        >
          {collapsed ? (
            <ChevronRight className="h-3.5 w-3.5 text-violet-300" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5 text-violet-300" />
          )}
          <ListChecks className="h-4 w-4 text-violet-300" />
          <span className="text-xs font-semibold text-violet-100">Plan monitor</span>
          <PlanStatusBadge plan={plan} />
        </button>

        <ProgressCounters counts={counts} />
        <ElapsedTimer startedAtMs={plan.startedAtMs} running={!finished} />

        {plan.status === 'running' && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              cancelTask(containerId)
            }}
            title="Stop plan execution"
            className="flex items-center gap-1 rounded border border-red-500/40 bg-red-950/30 px-1.5 py-0.5 text-[10px] font-medium text-red-200 hover:bg-red-900/50 hover:text-red-100"
          >
            <StopCircle className="h-3 w-3" />
            <span>Stop</span>
          </button>
        )}

        {finished && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              dismissPlan(containerId)
            }}
            title="Dismiss plan card"
            className="rounded p-1 text-violet-300 hover:bg-violet-800/50 hover:text-violet-100"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Progress bar */}
      <ProgressBar counts={counts} total={plan.steps.length} />

      {!collapsed && (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {/* Tab bar — Steps + Files. Files tab loads a since-filtered listing
              of files written under the agent's saved/output dirs after the
              plan started. */}
          <div className="flex items-center gap-1 border-b border-violet-500/20 bg-violet-950/30 px-2">
            <TabButton
              active={activeTab === 'steps'}
              onClick={() => setActiveTab('steps')}
              icon={<ListChecks className="h-3 w-3" />}
              label="Steps"
            />
            <TabButton
              active={activeTab === 'files'}
              onClick={() => setActiveTab('files')}
              icon={<FileText className="h-3 w-3" />}
              label="Files"
            />
          </div>

          {activeTab === 'steps' ? (
            <>
              {/* Active step spotlight */}
              {activeStep && plan.status === 'running' && <ActiveStepSpotlight step={activeStep} />}

              {/* Body — internal scroll keeps long step lists from pushing chat off screen */}
              <div className="overflow-y-auto px-3 py-2">
                {plan.errorMessage && (
                  <div className="mb-2 flex items-start gap-1.5 rounded border border-red-700/40 bg-red-950/30 px-2 py-1.5 text-[11px] text-red-300">
                    <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                    <span className="break-words">{plan.errorMessage}</span>
                  </div>
                )}

                <ol className="space-y-1">
                  {plan.steps.map((step, idx) => (
                    <PlanStepRow
                      key={step.id}
                      step={step}
                      index={idx + 1}
                      isActive={step.id === plan.activeStepId}
                      revisions={plan.revisions}
                    />
                  ))}
                </ol>

                {plan.verificationNotes && plan.status === 'failed' && (
                  <div className="mt-2 rounded border border-amber-700/40 bg-amber-950/20 px-2 py-1.5 text-[11px] text-amber-300">
                    <span className="font-medium">Verification notes:</span> {plan.verificationNotes}
                  </div>
                )}
              </div>
            </>
          ) : (
            <PlanFilesTab
              host={session.host}
              port={session.port}
              auth={session.auth}
              startedAtMs={plan.startedAtMs}
              running={plan.status === 'running'}
            />
          )}
        </div>
      )}
    </div>
  )
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
        active
          ? 'border-b-2 border-violet-400 text-violet-100'
          : 'border-b-2 border-transparent text-violet-300/60 hover:text-violet-200'
      }`}
    >
      {icon}
      {label}
    </button>
  )
}

// ── Files tab ────────────────────────────────────────────────────────────────

function PlanFilesTab({
  host,
  port,
  auth,
  startedAtMs,
  running,
}: {
  host: string
  port: number
  auth: string
  startedAtMs: number
  running: boolean
}) {
  const [files, setFiles] = useState<AgentFile[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [viewerIdx, setViewerIdx] = useState<number | null>(null)

  const fetchFiles = async () => {
    setLoading(true)
    setError('')
    try {
      const list = await listAgentFilesSince(host, port, auth, startedAtMs)
      // Newest first — the most recent step's output is usually what the user
      // wants to inspect.
      list.sort((a, b) => (b.modified || 0) - (a.modified || 0))
      setFiles(list)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  // Initial load + auto-refresh while plan is running. Poll cheaply (every 4s)
  // so newly-written files surface without forcing the user to hit refresh.
  useEffect(() => {
    fetchFiles()
    if (!running) return
    const id = setInterval(fetchFiles, 4000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host, port, auth, startedAtMs, running])

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-violet-500/10 px-3 py-1.5 text-[10px] text-violet-300/70">
        <span>
          {files.length} file{files.length === 1 ? '' : 's'} since plan started
        </span>
        <button
          onClick={fetchFiles}
          disabled={loading}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-violet-300 hover:bg-violet-800/40 disabled:opacity-40"
          title="Refresh"
        >
          <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        {error && (
          <div className="rounded border border-red-700/40 bg-red-950/30 px-2 py-1.5 text-[11px] text-red-300">
            {error}
          </div>
        )}
        {!error && !loading && files.length === 0 && (
          <div className="px-2 py-6 text-center text-[11px] text-violet-300/50">
            No files written yet. New files will appear here as steps run.
          </div>
        )}
        {files.length > 0 && (
          <ul className="space-y-0.5">
            {files.map((f, idx) => (
              <li
                key={f.physical}
                className="group flex items-center gap-2 rounded px-2 py-1 text-[12px] hover:bg-violet-900/20"
              >
                <FileText className="h-3.5 w-3.5 shrink-0 text-violet-300/70" />
                <span className="flex-1 truncate text-zinc-200" title={f.logical}>
                  {f.filename}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                  {formatSize(f.size)}
                </span>
                {isViewable(f) && (
                  <button
                    onClick={() => setViewerIdx(idx)}
                    className="rounded p-1 text-zinc-500 opacity-0 hover:bg-zinc-700 hover:text-zinc-200 group-hover:opacity-100"
                    title="View"
                  >
                    <Eye className="h-3 w-3" />
                  </button>
                )}
                <a
                  href={getDownloadUrl(host, port, f.physical, auth)}
                  download={f.filename}
                  className="rounded p-1 text-zinc-500 opacity-0 hover:bg-zinc-700 hover:text-zinc-200 group-hover:opacity-100"
                  title="Download"
                >
                  <Download className="h-3 w-3" />
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>

      {viewerIdx !== null && files[viewerIdx] && (
        <FileViewer
          file={files[viewerIdx]}
          host={host}
          port={port}
          auth={auth}
          onClose={() => setViewerIdx(null)}
          onPrev={() => setViewerIdx((i) => (i !== null && i > 0 ? i - 1 : i))}
          onNext={() =>
            setViewerIdx((i) => (i !== null && i < files.length - 1 ? i + 1 : i))
          }
          hasPrev={viewerIdx > 0}
          hasNext={viewerIdx < files.length - 1}
        />
      )}
    </div>
  )
}

// ── Header pieces ────────────────────────────────────────────────────────────

function PlanStatusBadge({ plan }: { plan: PlanState }) {
  const map: Record<PlanState['status'], { label: string; cls: string }> = {
    running: { label: 'running', cls: 'bg-violet-500/30 text-violet-100 ring-1 ring-violet-400/40' },
    verified: { label: 'verified', cls: 'bg-emerald-500/30 text-emerald-100 ring-1 ring-emerald-400/40' },
    completed: { label: 'completed', cls: 'bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-400/30' },
    failed: { label: 'failed', cls: 'bg-red-500/30 text-red-100 ring-1 ring-red-400/40' },
    cancelled: { label: 'cancelled', cls: 'bg-zinc-500/30 text-zinc-200 ring-1 ring-zinc-400/40' },
  }
  const m = map[plan.status]
  return (
    <span className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${m.cls}`}>
      {plan.status === 'running' && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
      {m.label}
    </span>
  )
}

function ProgressCounters({ counts }: { counts: ReturnType<typeof aggregateCounts> }) {
  return (
    <div className="flex items-center gap-2 text-[10px] font-mono text-violet-200/80">
      <span className="flex items-center gap-0.5">
        <CheckCircle2 className="h-3 w-3 text-emerald-400" />
        {counts.verified + counts.completed}
      </span>
      {counts.failed > 0 && (
        <span className="flex items-center gap-0.5">
          <XCircle className="h-3 w-3 text-red-400" />
          {counts.failed}
        </span>
      )}
      <span className="text-violet-300/50">/ {counts.total}</span>
    </div>
  )
}

function ElapsedTimer({ startedAtMs, running }: { startedAtMs: number; running: boolean }) {
  const [now, setNow] = useState<number>(() => Date.now())
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [running])
  const elapsed = Math.max(0, now - startedAtMs)
  return <span className="font-mono text-[10px] text-violet-200/70">{formatDuration(elapsed)}</span>
}

function ProgressBar({
  counts,
  total,
}: {
  counts: ReturnType<typeof aggregateCounts>
  total: number
}) {
  if (total === 0) return null
  const verifiedPct = (counts.verified / total) * 100
  const completedPct = (counts.completed / total) * 100
  const runningPct = (counts.running / total) * 100
  const failedPct = (counts.failed / total) * 100

  return (
    <div className="flex h-1 overflow-hidden bg-zinc-900/60">
      {verifiedPct > 0 && <div style={{ width: `${verifiedPct}%` }} className="bg-emerald-400" />}
      {completedPct > 0 && <div style={{ width: `${completedPct}%` }} className="bg-emerald-500/60" />}
      {runningPct > 0 && (
        <div style={{ width: `${runningPct}%` }} className="bg-violet-400 animate-pulse" />
      )}
      {failedPct > 0 && <div style={{ width: `${failedPct}%` }} className="bg-red-500" />}
    </div>
  )
}

// ── Active step spotlight ────────────────────────────────────────────────────

function ActiveStepSpotlight({ step }: { step: PlanStep }) {
  const phase = (step.currentPhase || '').toLowerCase()
  const isToolPhase = phase === 'tool' || Boolean(step.currentTool)
  const PhaseIcon = isToolPhase ? Wrench : phase === 'thinking' ? Sparkles : Activity

  return (
    <div className="mx-3 mt-3 mb-1 rounded-md border border-violet-500/30 bg-violet-950/40 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <div className="relative">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-300" />
        </div>
        <span className="text-[10px] font-bold uppercase tracking-wider text-violet-300">Now running</span>
        <span className="text-[11px] font-medium text-violet-50">{step.title || step.id}</span>
      </div>

      {(step.currentText || step.currentTool || phase) && (
        <div className="mt-1.5 flex items-start gap-1.5 text-[11px] text-violet-100/80">
          <PhaseIcon className="mt-0.5 h-3 w-3 shrink-0 text-violet-300/80" />
          <span className="flex-1 break-words">
            {step.currentTool && (
              <span className="font-mono text-violet-200">
                {step.currentTool}
                {step.currentText ? ' · ' : ''}
              </span>
            )}
            {step.currentText || (!step.currentTool && phase ? phase : '')}
          </span>
        </div>
      )}
    </div>
  )
}

// ── Step list ────────────────────────────────────────────────────────────────

function PlanStepRow({
  step,
  index,
  isActive,
  revisions,
}: {
  step: PlanStep
  index: number
  isActive: boolean
  revisions: PlanState['revisions']
}) {
  const [expanded, setExpanded] = useState(false)
  const stepRevisions = revisions.filter((r) => r.task_id === step.id)
  const runtime = step.startedAt && step.endedAt ? step.endedAt - step.startedAt : 0

  const hasDetails =
    step.acceptance_criteria ||
    step.verificationNotes ||
    stepRevisions.length > 0 ||
    step.depends_on.length > 0 ||
    step.error ||
    step.tokensIn ||
    step.tokensOut

  return (
    <li className={`text-[12px] ${isActive ? 'rounded bg-violet-900/20 ring-1 ring-violet-500/30' : ''}`}>
      <button
        onClick={() => hasDetails && setExpanded(!expanded)}
        className={`flex w-full items-start gap-2 rounded px-1.5 py-1 text-left ${
          hasDetails ? 'hover:bg-violet-900/20' : 'cursor-default'
        }`}
      >
        <span className="mt-0.5 w-4 shrink-0 text-right text-[10px] font-mono text-violet-400/60">{index}</span>
        <StepIcon status={step.status} />
        <span
          className={`flex-1 ${
            step.status === 'failed'
              ? 'text-red-300'
              : step.status === 'running'
                ? 'text-violet-100 font-medium'
                : 'text-zinc-200'
          }`}
        >
          {step.title || step.id}
        </span>
        {step.step_kind === 'orchestrate' && (
          <span className="rounded bg-zinc-700/50 px-1 py-0.5 text-[9px] text-zinc-400">orchestrate</span>
        )}
        {step.revisionCount > 0 && (
          <span className="flex items-center gap-0.5 rounded bg-amber-700/30 px-1 py-0.5 text-[9px] text-amber-300">
            <RotateCw className="h-2.5 w-2.5" />
            {step.revisionCount}
          </span>
        )}
        {runtime > 0 && (
          <span className="font-mono text-[9px] text-zinc-500">{formatDuration(runtime)}</span>
        )}
        {hasDetails && (
          <span className="text-zinc-500">
            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </span>
        )}
      </button>

      {expanded && hasDetails && (
        <div className="ml-9 mt-1 space-y-1.5 rounded bg-zinc-900/40 px-2 py-1.5 text-[11px]">
          {step.acceptance_criteria && (
            <div>
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Acceptance</div>
              <div className="text-zinc-300">{step.acceptance_criteria}</div>
            </div>
          )}
          {step.depends_on.length > 0 && (
            <div className="text-zinc-400">
              <span className="text-zinc-500">depends on:</span>{' '}
              <span className="font-mono text-[10px]">{step.depends_on.join(', ')}</span>
            </div>
          )}
          {step.error && (
            <div className="rounded border border-red-800/40 bg-red-950/30 px-1.5 py-1 text-red-300">
              <span className="font-medium">Error:</span> {step.error}
            </div>
          )}
          {(step.tokensIn || step.tokensOut) && (
            <div className="font-mono text-[10px] text-zinc-500">
              tokens: {step.tokensIn ?? 0} in / {step.tokensOut ?? 0} out
            </div>
          )}
          {step.verificationNotes && (
            <div>
              <div className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Verification {step.verificationPassed ? '✓' : '✗'}
              </div>
              <div className={step.verificationPassed ? 'text-emerald-300' : 'text-red-300'}>
                {step.verificationNotes}
              </div>
            </div>
          )}
          {stepRevisions.map((r, i) => (
            <div key={i} className="rounded border border-amber-800/30 bg-amber-950/20 px-1.5 py-1">
              <div className="text-[10px] font-medium uppercase tracking-wider text-amber-400">
                Revision {r.revision_count}
              </div>
              {r.rationale && <div className="mt-0.5 text-amber-200">{r.rationale}</div>}
              {r.revised_description && (
                <div className="mt-1 text-zinc-300">
                  <span className="text-zinc-500">→ </span>
                  {r.revised_description}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </li>
  )
}

function StepIcon({ status }: { status: PlanStep['status'] }) {
  const cls = 'mt-0.5 h-3.5 w-3.5 shrink-0'
  switch (status) {
    case 'verified':
      return <CheckCircle2 className={`${cls} text-emerald-400`} />
    case 'completed':
      return <CheckCircle2 className={`${cls} text-emerald-400/60`} />
    case 'failed':
      return <XCircle className={`${cls} text-red-400`} />
    case 'running':
      return <Loader2 className={`${cls} animate-spin text-violet-300`} />
    case 'revising':
      return <Loader2 className={`${cls} animate-spin text-amber-400`} />
    case 'pending':
    default:
      return <Clock className={`${cls} text-zinc-500`} />
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function aggregateCounts(plan: PlanState) {
  let verified = 0
  let completed = 0
  let running = 0
  let failed = 0
  let pending = 0
  for (const s of plan.steps) {
    switch (s.status) {
      case 'verified':
        verified++
        break
      case 'completed':
        completed++
        break
      case 'running':
        running++
        break
      case 'failed':
        failed++
        break
      default:
        pending++
        break
    }
  }
  return { verified, completed, running, failed, pending, total: plan.steps.length }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const totalSec = Math.floor(ms / 1000)
  if (totalSec < 60) return `${totalSec}s`
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  if (min < 60) return `${min}m ${sec}s`
  const hr = Math.floor(min / 60)
  return `${hr}h ${min % 60}m`
}
