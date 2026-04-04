import { create } from 'zustand'
import type { TraceSpan, TraceSummary } from '../types'
import { getTracesFromAgent } from '../services/api'

interface TraceStore {
  // Data
  spans: TraceSpan[]
  summary: TraceSummary | null
  isLoading: boolean
  error: string | null

  // Actions
  fetchTraces: (agentSlug: string) => Promise<void>
  clearTraces: () => void

  /** Handle a real-time trace_span WebSocket event. */
  handleSpanEvent: (span: TraceSpan) => void
}

export const useTraceStore = create<TraceStore>((set) => ({
  spans: [],
  summary: null,
  isLoading: false,
  error: null,

  fetchTraces: async (agentSlug: string) => {
    set({ isLoading: true, error: null })
    try {
      const res = await getTracesFromAgent(agentSlug)
      if (res.traces) {
        set({
          spans: res.traces.spans,
          summary: res.traces.summary,
          isLoading: false,
        })
      } else {
        set({ spans: [], summary: null, isLoading: false })
      }
    } catch (err) {
      set({ error: String(err), isLoading: false })
    }
  },

  clearTraces: () => {
    set({ spans: [], summary: null, error: null })
  },

  handleSpanEvent: (span: TraceSpan) => {
    set((state) => {
      const idx = state.spans.findIndex((s) => s.span_id === span.span_id)
      let newSpans: TraceSpan[]
      if (idx >= 0) {
        // Update existing span (e.g. end_span call)
        newSpans = [...state.spans]
        newSpans[idx] = span
      } else {
        // New span
        newSpans = [...state.spans, span]
      }

      // Recompute summary from spans
      const byType: Record<string, number> = {}
      let totalDuration = 0
      let totalInput = 0
      let totalOutput = 0
      let completed = 0
      let failed = 0
      let running = 0

      for (const s of newSpans) {
        byType[s.span_type] = (byType[s.span_type] || 0) + 1
        totalDuration += s.duration_ms || 0
        totalInput += (s.attributes?.input_tokens as number) || 0
        totalOutput += (s.attributes?.output_tokens as number) || 0
        if (s.status === 'completed') completed++
        else if (s.status === 'failed') failed++
        else running++
      }

      const summary: TraceSummary = {
        trace_id: span.trace_id,
        span_count: newSpans.length,
        by_type: byType,
        total_duration_ms: Math.round(totalDuration * 10) / 10,
        total_input_tokens: totalInput,
        total_output_tokens: totalOutput,
        completed,
        failed,
        running,
      }

      return { spans: newSpans, summary }
    })
  },
}))
