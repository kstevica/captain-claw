import { useMemo } from 'react'
import { useTraceStore } from '../../stores/traceStore'
import type { TraceSpan } from '../../types'

// Colour mapping per span type
const TYPE_COLORS: Record<string, string> = {
  decompose: 'bg-purple-500',
  execution: 'bg-blue-500',
  task: 'bg-emerald-500',
  synthesize: 'bg-amber-500',
  validation: 'bg-cyan-500',
  llm_call: 'bg-indigo-500',
  tool_execution: 'bg-pink-500',
  workspace_op: 'bg-teal-500',
  orchestration: 'bg-slate-500',
}

const STATUS_ICON: Record<string, string> = {
  completed: '\u2713',
  failed: '\u2717',
  running: '\u25cf',
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function SpanBar({ span, minTime, totalRange }: {
  span: TraceSpan
  minTime: number
  totalRange: number
}) {
  const leftPct = totalRange > 0
    ? ((span.started_at - minTime) / totalRange) * 100
    : 0
  const widthPct = totalRange > 0 && span.duration_ms > 0
    ? (span.duration_ms / 1000 / totalRange) * 100
    : 0.5  // min visible width for running/instant spans

  const barColor = TYPE_COLORS[span.span_type] || 'bg-gray-500'
  const isFailed = span.status === 'failed'
  const isRunning = span.status === 'running'

  return (
    <div className="flex items-center gap-2 py-0.5 group">
      {/* Label */}
      <div className="w-44 shrink-0 truncate text-xs text-zinc-400 flex items-center gap-1" title={span.name}>
        <span className={`text-[10px] ${isFailed ? 'text-red-400' : isRunning ? 'text-yellow-400 animate-pulse' : 'text-emerald-400'}`}>
          {STATUS_ICON[span.status] || '\u25cb'}
        </span>
        <span className="truncate">{span.name}</span>
      </div>

      {/* Timeline bar */}
      <div className="flex-1 relative h-4 bg-zinc-800/50 rounded overflow-hidden">
        <div
          className={`absolute top-0.5 bottom-0.5 rounded ${barColor} ${isRunning ? 'animate-pulse opacity-70' : 'opacity-80'} ${isFailed ? 'opacity-50' : ''}`}
          style={{
            left: `${Math.min(leftPct, 99)}%`,
            width: `${Math.max(widthPct, 0.5)}%`,
          }}
        />
      </div>

      {/* Duration */}
      <div className="w-16 shrink-0 text-right text-[10px] text-zinc-500 tabular-nums">
        {span.duration_ms > 0 ? formatMs(span.duration_ms) : isRunning ? '...' : '-'}
      </div>
    </div>
  )
}

export default function TraceTimeline() {
  const { spans, summary } = useTraceStore()

  // Sort spans by start time
  const sortedSpans = useMemo(
    () => [...spans].sort((a, b) => a.started_at - b.started_at),
    [spans],
  )

  // Compute timeline range
  const { minTime, totalRange } = useMemo(() => {
    if (sortedSpans.length === 0) return { minTime: 0, totalRange: 0 }
    const min = sortedSpans[0].started_at
    const maxEnd = Math.max(...sortedSpans.map((s) =>
      s.ended_at > 0 ? s.ended_at : s.started_at + (s.duration_ms / 1000 || 0.001)
    ))
    return { minTime: min, totalRange: maxEnd - min || 1 }
  }, [sortedSpans])

  if (spans.length === 0) {
    return (
      <div className="text-xs text-zinc-500 italic p-3">
        No trace data available. Run an orchestration to see spans.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 p-3">
      {/* Summary header */}
      {summary && (
        <div className="flex items-center gap-3 text-[10px] text-zinc-500 mb-2 flex-wrap">
          <span>{summary.span_count} spans</span>
          <span className="text-emerald-500">{summary.completed} done</span>
          {summary.failed > 0 && <span className="text-red-400">{summary.failed} failed</span>}
          {summary.running > 0 && <span className="text-yellow-400">{summary.running} running</span>}
          <span>{formatMs(summary.total_duration_ms)} total</span>
          {(summary.total_input_tokens > 0 || summary.total_output_tokens > 0) && (
            <span className="text-zinc-600">
              {summary.total_input_tokens.toLocaleString()}+{summary.total_output_tokens.toLocaleString()} tokens
            </span>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-2 text-[9px] text-zinc-600 mb-1 flex-wrap">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <span key={type} className="flex items-center gap-0.5">
            <span className={`inline-block w-2 h-2 rounded-sm ${color}`} />
            {type}
          </span>
        ))}
      </div>

      {/* Span bars */}
      <div className="flex flex-col">
        {sortedSpans.map((span) => (
          <SpanBar
            key={span.span_id}
            span={span}
            minTime={minTime}
            totalRange={totalRange}
          />
        ))}
      </div>
    </div>
  )
}
