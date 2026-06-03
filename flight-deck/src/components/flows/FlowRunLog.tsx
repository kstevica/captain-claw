import { useEffect } from 'react'
import {
  ArrowLeft,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  CircleDot,
  Bot,
} from 'lucide-react'
import { useFlowsStore } from '../../stores/flowsStore'
import type { FlowRunStep } from '../../services/flowsApi'

function StatusIcon({ status }: { status: string }) {
  const s = (status || '').toLowerCase()
  if (s === 'done' || s === 'ok' || s === 'success') return <CheckCircle2 className="h-4 w-4 text-emerald-400" />
  if (s === 'error' || s === 'failed') return <XCircle className="h-4 w-4 text-red-400" />
  if (s === 'running') return <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
  if (s === 'parked' || s === 'waiting') return <Clock className="h-4 w-4 text-amber-400" />
  return <CircleDot className="h-4 w-4 text-zinc-500" />
}

function fmtTime(t?: string): string {
  if (!t) return ''
  const d = new Date(t)
  if (isNaN(d.getTime())) return t
  return d.toLocaleString()
}

export function FlowRunLog() {
  const { activeRunId, runDetail, error, openList, refreshRun } = useFlowsStore()

  // Poll while running
  useEffect(() => {
    if (!activeRunId) return
    const isRunning = (runDetail?.run.status || '').toLowerCase() === 'running'
    if (!isRunning) return
    const t = setInterval(() => refreshRun(activeRunId), 2000)
    return () => clearInterval(t)
  }, [activeRunId, runDetail?.run.status, refreshRun])

  const run = runDetail?.run
  const steps: FlowRunStep[] = runDetail?.steps || []

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-6 py-6">
        {/* Header */}
        <div className="mb-5 flex items-center justify-between">
          <button
            onClick={openList}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to flows
          </button>
          {activeRunId && (
            <button
              onClick={() => refreshRun(activeRunId)}
              className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Run summary */}
        {run && (
          <div className="mb-5 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3">
            <div className="flex items-center gap-2.5">
              <StatusIcon status={run.status} />
              <span className="text-sm font-medium text-zinc-200">Run {run.id.slice(0, 8)}</span>
              <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
                {run.status}
              </span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-zinc-500">
              <div>Started: <span className="text-zinc-400">{fmtTime(run.started_at) || '—'}</span></div>
              <div>Ended: <span className="text-zinc-400">{fmtTime(run.ended_at) || '—'}</span></div>
            </div>
            {run.error && (
              <div className="mt-2 rounded-lg bg-red-500/[0.06] px-2.5 py-1.5 text-[11px] text-red-300">
                {run.error}
              </div>
            )}
          </div>
        )}

        {!runDetail && !error && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
            <span className="ml-2 text-sm text-zinc-500">Loading run…</span>
          </div>
        )}

        {/* Step timeline */}
        {runDetail && (
          <div className="space-y-0">
            {steps.length === 0 ? (
              <p className="rounded-xl border border-dashed border-zinc-800 px-6 py-10 text-center text-xs text-zinc-600">
                No steps recorded for this run yet.
              </p>
            ) : (
              steps.map((step, i) => (
                <div key={`${step.step_id}-${step.seq}-${i}`} className="flex gap-3">
                  {/* Timeline rail */}
                  <div className="flex flex-col items-center">
                    <div className="mt-1.5"><StatusIcon status={step.status} /></div>
                    {i < steps.length - 1 && <div className="my-1 w-px flex-1 bg-zinc-800" />}
                  </div>

                  {/* Step card */}
                  <div className="mb-3 min-w-0 flex-1 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3.5 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-zinc-200">{step.step_id}</span>
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
                        {step.status}
                      </span>
                      {typeof step.ms === 'number' && (
                        <span className="ml-auto text-[10px] text-zinc-500">{step.ms} ms</span>
                      )}
                    </div>
                    {step.agent && (
                      <div className="mt-1 flex items-center gap-1.5 text-[11px] text-zinc-500">
                        <Bot className="h-3 w-3 text-violet-400/70" /> {step.agent}
                      </div>
                    )}
                    {step.output_text && (
                      <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded-lg bg-zinc-950/70 px-2.5 py-2 text-[11px] leading-relaxed text-zinc-300">
                        {step.output_text}
                      </pre>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}
