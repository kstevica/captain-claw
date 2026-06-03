import { create } from 'zustand'
import type {
  Flow,
  FlowInput,
  FlowRunSummary,
  FlowRunDetail,
  FlowTestStep,
} from '../services/flowsApi'
import * as api from '../services/flowsApi'

type FlowsView = 'list' | 'builder' | 'runlog'

interface FlowsStore {
  // Data
  flows: Flow[]
  loading: boolean
  error: string | null

  // Navigation within the Flows section
  view: FlowsView
  editingId: string | null // null = creating new; set = editing existing

  // Run log
  activeRunId: string | null
  runDetail: FlowRunDetail | null
  runs: FlowRunSummary[]

  // Test trace (in builder)
  testSteps: FlowTestStep[] | null
  testing: boolean

  // Actions
  fetchFlows: () => Promise<void>
  openList: () => void
  openNew: () => void
  openEdit: (id: string) => void
  openRunLog: (runId: string) => Promise<void>

  saveFlow: (input: FlowInput, id: string | null) => Promise<void>
  removeFlow: (id: string) => Promise<void>
  toggleEnabled: (id: string, enabled: boolean) => Promise<void>
  runNow: (id: string) => Promise<string>
  testFlow: (id: string, payload: Record<string, unknown>) => Promise<void>
  clearTest: () => void
  refreshRun: (runId: string) => Promise<void>
  fetchRunsFor: (id: string) => Promise<void>
}

export const useFlowsStore = create<FlowsStore>((set, get) => ({
  flows: [],
  loading: false,
  error: null,

  view: 'list',
  editingId: null,

  activeRunId: null,
  runDetail: null,
  runs: [],

  testSteps: null,
  testing: false,

  fetchFlows: async () => {
    set({ loading: true, error: null })
    try {
      const flows = await api.listFlows()
      set({ flows, loading: false })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false })
    }
  },

  openList: () => set({ view: 'list', editingId: null, testSteps: null }),
  openNew: () => set({ view: 'builder', editingId: null, testSteps: null }),
  openEdit: (id) => set({ view: 'builder', editingId: id, testSteps: null }),

  openRunLog: async (runId) => {
    set({ view: 'runlog', activeRunId: runId, runDetail: null, error: null })
    try {
      const detail = await api.getFlowRun(runId)
      set({ runDetail: detail })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  saveFlow: async (input, id) => {
    set({ error: null })
    try {
      if (id) {
        const existing = get().flows.find((f) => f.id === id)
        const full: Flow = { ...(existing as Flow), ...input, id }
        await api.updateFlow(id, full)
      } else {
        await api.createFlow(input)
      }
      await get().fetchFlows()
      set({ view: 'list', editingId: null, testSteps: null })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
      throw e
    }
  },

  removeFlow: async (id) => {
    set({ error: null })
    try {
      await api.deleteFlow(id)
      set({ flows: get().flows.filter((f) => f.id !== id) })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  toggleEnabled: async (id, enabled) => {
    // optimistic
    set({ flows: get().flows.map((f) => (f.id === id ? { ...f, enabled } : f)) })
    try {
      await api.enableFlow(id, enabled)
    } catch (e) {
      // revert
      set({
        flows: get().flows.map((f) => (f.id === id ? { ...f, enabled: !enabled } : f)),
        error: e instanceof Error ? e.message : String(e),
      })
    }
  },

  runNow: async (id) => {
    set({ error: null })
    const { run_id } = await api.runFlow(id)
    // refresh list so last_run updates eventually
    get().fetchFlows()
    return run_id
  },

  testFlow: async (id, payload) => {
    set({ testing: true, testSteps: null, error: null })
    try {
      const r = await api.testFlow(id, payload)
      set({ testSteps: r.steps || [], testing: false })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), testing: false })
    }
  },

  clearTest: () => set({ testSteps: null }),

  refreshRun: async (runId) => {
    try {
      const detail = await api.getFlowRun(runId)
      set({ runDetail: detail })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  fetchRunsFor: async (id) => {
    try {
      const runs = await api.listFlowRuns(id)
      set({ runs })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },
}))
