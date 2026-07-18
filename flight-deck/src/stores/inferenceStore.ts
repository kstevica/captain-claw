// UI state for the Local inference worker (mrav Phase 2). The logic lives
// in src/inference/localInference.ts — this store only mirrors its state
// so the panel re-renders; keep it dumb.
import { create } from 'zustand'

export type InferenceStatus = 'off' | 'starting' | 'ready' | 'error'

interface InferenceState {
  status: InferenceStatus
  modelId: string
  ctxWindow: number // engine window (input + output tokens)
  progress: string
  error: string
  wsConnected: boolean
  jobsDone: number
  lastCall: string // e.g. "1843→92 tok · 1.4s"
  patch: (partial: Partial<Omit<InferenceState, 'patch'>>) => void
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
  patch: (partial) => set(partial),
}))
