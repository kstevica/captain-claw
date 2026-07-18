// UI state for the Local inference worker (mrav Phase 2). The logic lives
// in src/inference/localInference.ts — this store only mirrors its state
// so the panel re-renders; keep it dumb.
import { create } from 'zustand'

export type InferenceStatus = 'off' | 'starting' | 'ready' | 'error'

// One served completion — newest first. Kept in memory only (a worker log,
// not persistent history); capped so a long-lived tab can't grow unbounded.
export interface JobLogEntry {
  ts: number // Date.now() when the job finished
  model: string
  promptTokens: number
  completionTokens: number
  seconds: number
  tps: number // decode tokens/sec (completion / seconds)
  ok: boolean
  error?: string
}

const JOB_LOG_CAP = 100

interface InferenceState {
  status: InferenceStatus
  modelId: string
  ctxWindow: number // engine window (input + output tokens)
  progress: string
  error: string
  wsConnected: boolean
  jobsDone: number
  lastCall: string // e.g. "1843→92 tok · 1.4s"
  jobs: JobLogEntry[]
  patch: (partial: Partial<Omit<InferenceState, 'patch' | 'logJob' | 'clearJobs'>>) => void
  logJob: (entry: JobLogEntry) => void
  clearJobs: () => void
}

export const useInferenceStore = create<InferenceState>((set) => ({
  status: 'off',
  modelId: '',
  ctxWindow: 0,
  progress: '',
  error: '',
  wsConnected: false,
  jobsDone: 0,
  lastCall: '',
  jobs: [],
  patch: (partial) => set(partial),
  logJob: (entry) => set((s) => ({ jobs: [entry, ...s.jobs].slice(0, JOB_LOG_CAP) })),
  clearJobs: () => set({ jobs: [] }),
}))
