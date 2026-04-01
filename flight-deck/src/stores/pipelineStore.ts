import { create } from 'zustand'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

export interface PipelineStep {
  agentId: string
  prompt?: string // optional prompt to prepend — if empty, sends raw output
}

export interface Pipeline {
  id: string
  name: string
  enabled: boolean
  steps: PipelineStep[] // ordered chain: step[0] triggers step[1], etc.
  createdAt: string
}

const STORAGE_KEY = 'fd:pipelines'

function load(): Pipeline[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function save(pipelines: Pipeline[]) {
  const val = JSON.stringify(pipelines)
  if (useAuthStore.getState().authEnabled) queueSave(STORAGE_KEY, val)
  else localStorage.setItem(STORAGE_KEY, val)
}

interface PipelineStore {
  pipelines: Pipeline[]
  createPipeline: (name: string) => string
  deletePipeline: (id: string) => void
  updatePipeline: (id: string, patch: Partial<Pick<Pipeline, 'name' | 'enabled' | 'steps'>>) => void
  addStep: (pipelineId: string, step: PipelineStep) => void
  removeStep: (pipelineId: string, index: number) => void
  reorderStep: (pipelineId: string, from: number, to: number) => void
  updateStep: (pipelineId: string, index: number, patch: Partial<PipelineStep>) => void
  getPipelinesForAgent: (agentId: string) => Pipeline[]
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  pipelines: load(),

  createPipeline: (name) => {
    const id = `pipe-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const pipeline: Pipeline = { id, name, enabled: true, steps: [], createdAt: new Date().toISOString() }
    const pipelines = [...get().pipelines, pipeline]
    save(pipelines)
    set({ pipelines })
    return id
  },

  deletePipeline: (id) => {
    const pipelines = get().pipelines.filter((p) => p.id !== id)
    save(pipelines)
    set({ pipelines })
  },

  updatePipeline: (id, patch) => {
    const pipelines = get().pipelines.map((p) => p.id === id ? { ...p, ...patch } : p)
    save(pipelines)
    set({ pipelines })
  },

  addStep: (pipelineId, step) => {
    const pipelines = get().pipelines.map((p) => {
      if (p.id !== pipelineId) return p
      return { ...p, steps: [...p.steps, step] }
    })
    save(pipelines)
    set({ pipelines })
  },

  removeStep: (pipelineId, index) => {
    const pipelines = get().pipelines.map((p) => {
      if (p.id !== pipelineId) return p
      const steps = p.steps.filter((_, i) => i !== index)
      return { ...p, steps }
    })
    save(pipelines)
    set({ pipelines })
  },

  reorderStep: (pipelineId, from, to) => {
    const pipelines = get().pipelines.map((p) => {
      if (p.id !== pipelineId) return p
      const steps = [...p.steps]
      const [item] = steps.splice(from, 1)
      steps.splice(to, 0, item)
      return { ...p, steps }
    })
    save(pipelines)
    set({ pipelines })
  },

  updateStep: (pipelineId, index, patch) => {
    const pipelines = get().pipelines.map((p) => {
      if (p.id !== pipelineId) return p
      const steps = p.steps.map((s, i) => i === index ? { ...s, ...patch } : s)
      return { ...p, steps }
    })
    save(pipelines)
    set({ pipelines })
  },

  getPipelinesForAgent: (agentId) => {
    return get().pipelines.filter((p) => p.enabled && p.steps.some((s) => s.agentId === agentId))
  },
}))

registerHydrator((settings) => {
  const raw = settings[STORAGE_KEY]
  if (raw) {
    try { usePipelineStore.setState({ pipelines: JSON.parse(raw) }) } catch { /* ignore */ }
  }
})
