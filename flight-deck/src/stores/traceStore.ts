import { create } from 'zustand'
import type { TraceSpan, TraceSummary } from '../types'
import { getTracesFromAgent } from '../services/api'

/** Stable empty array to avoid new-reference re-renders. */
const EMPTY_SPANS: TraceSpan[] = []

function computeSummary(spans: TraceSpan[]): TraceSummary | null {
  if (spans.length === 0) return null

  const byType: Record<string, number> = {}
  let totalDuration = 0
  let totalInput = 0
  let totalOutput = 0
  let completed = 0
  let failed = 0
  let running = 0

  for (const s of spans) {
    byType[s.span_type] = (byType[s.span_type] || 0) + 1
    totalDuration += s.duration_ms || 0
    totalInput += (s.attributes?.input_tokens as number) || 0
    totalOutput += (s.attributes?.output_tokens as number) || 0
    if (s.status === 'completed') completed++
    else if (s.status === 'failed') failed++
    else running++
  }

  return {
    trace_id: spans[0].trace_id,
    span_count: spans.length,
    by_type: byType,
    total_duration_ms: Math.round(totalDuration * 10) / 10,
    total_input_tokens: totalInput,
    total_output_tokens: totalOutput,
    completed,
    failed,
    running,
  }
}

interface AgentTraces {
  spans: TraceSpan[]
  summary: TraceSummary | null
}

interface TraceStore {
  /** Per-agent trace data keyed by containerId */
  agents: Record<string, AgentTraces>

  isLoading: boolean
  error: string | null

  // Actions
  fetchTraces: (containerId: string, agentSlug: string) => Promise<void>
  clearTraces: (containerId: string) => void
  clearAllTraces: () => void

  /** Handle a real-time trace_span WebSocket event for a specific agent. */
  handleSpanEvent: (containerId: string, span: TraceSpan) => void
}

export const useTraceStore = create<TraceStore>((set) => ({
  agents: {},
  isLoading: false,
  error: null,

  fetchTraces: async (containerId: string, agentSlug: string) => {
    set({ isLoading: true, error: null })
    try {
      const res = await getTracesFromAgent(agentSlug)
      if (res.traces) {
        set((state) => ({
          agents: {
            ...state.agents,
            [containerId]: {
              spans: res.traces!.spans,
              summary: res.traces!.summary,
            },
          },
          isLoading: false,
        }))
      } else {
        set({ isLoading: false })
      }
    } catch (err) {
      set({ error: String(err), isLoading: false })
    }
  },

  clearTraces: (containerId: string) => {
    set((state) => {
      const { [containerId]: _, ...rest } = state.agents
      return { agents: rest }
    })
  },

  clearAllTraces: () => {
    set({ agents: {}, error: null })
  },

  handleSpanEvent: (containerId: string, span: TraceSpan) => {
    set((state) => {
      const existing = state.agents[containerId] || { spans: [], summary: null }
      const idx = existing.spans.findIndex((s) => s.span_id === span.span_id)
      let newSpans: TraceSpan[]
      if (idx >= 0) {
        newSpans = [...existing.spans]
        newSpans[idx] = span
      } else {
        newSpans = [...existing.spans, span]
      }

      return {
        agents: {
          ...state.agents,
          [containerId]: {
            spans: newSpans,
            summary: computeSummary(newSpans),
          },
        },
      }
    })
  },
}))

// ── Selector helpers (use these in components) ──

/** Select spans for a specific agent. Returns a stable empty array when no data. */
export function selectSpans(state: TraceStore, containerId: string): TraceSpan[] {
  return state.agents[containerId]?.spans ?? EMPTY_SPANS
}

/** Select summary for a specific agent. */
export function selectSummary(state: TraceStore, containerId: string): TraceSummary | null {
  return state.agents[containerId]?.summary ?? null
}

/** Select span count for a specific agent. */
export function selectSpanCount(state: TraceStore, containerId: string): number {
  return state.agents[containerId]?.spans.length ?? 0
}
