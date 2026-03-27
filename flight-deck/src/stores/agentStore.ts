import { create } from 'zustand'
import type { InstanceInfo, Concern, BotPortStats } from '../types'
import * as api from '../services/api'

interface AgentStore {
  // Data
  instances: InstanceInfo[]
  concerns: Concern[]
  stats: BotPortStats | null
  selectedInstanceId: string | null
  wsConnected: boolean

  // Actions
  fetchInstances: () => Promise<void>
  fetchConcerns: (activeOnly?: boolean) => Promise<void>
  fetchStats: () => Promise<void>
  selectInstance: (id: string | null) => void
  setWsConnected: (v: boolean) => void

  // Real-time updates
  updateInstanceActivity: (instanceId: string, stepType: string, data: Record<string, unknown>) => void
  updateInstanceStatus: (instanceId: string, status: 'connected' | 'disconnected') => void
  upsertInstance: (instance: InstanceInfo) => void
  removeInstance: (instanceId: string) => void
}

export const useAgentStore = create<AgentStore>((set, get) => ({
  instances: [],
  concerns: [],
  stats: null,
  selectedInstanceId: null,
  wsConnected: false,

  fetchInstances: async () => {
    try {
      const instances = await api.getInstances()
      set({ instances })
    } catch (e) {
      console.error('Failed to fetch instances:', e)
    }
  },

  fetchConcerns: async (activeOnly = false) => {
    try {
      const concerns = await api.getConcerns(activeOnly)
      set({ concerns })
    } catch (e) {
      console.error('Failed to fetch concerns:', e)
    }
  },

  fetchStats: async () => {
    try {
      const stats = await api.getStats()
      set({ stats })
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  },

  selectInstance: (id) => set({ selectedInstanceId: id }),
  setWsConnected: (wsConnected) => set({ wsConnected }),

  updateInstanceActivity: (instanceId, stepType, data) => {
    set({
      instances: get().instances.map((inst) =>
        inst.id === instanceId
          ? { ...inst, activity: { ...inst.activity, [stepType]: { ...data, updated_at: new Date().toISOString() } } }
          : inst
      ),
    })
  },

  updateInstanceStatus: (instanceId, status) => {
    set({
      instances: get().instances.map((inst) =>
        inst.id === instanceId ? { ...inst, status } : inst
      ),
    })
  },

  upsertInstance: (instance) => {
    const existing = get().instances.find((i) => i.id === instance.id)
    if (existing) {
      set({
        instances: get().instances.map((i) =>
          i.id === instance.id ? { ...i, ...instance } : i
        ),
      })
    } else {
      set({ instances: [...get().instances, instance] })
    }
  },

  removeInstance: (instanceId) => {
    set({ instances: get().instances.filter((i) => i.id !== instanceId) })
    if (get().selectedInstanceId === instanceId) {
      set({ selectedInstanceId: null })
    }
  },
}))
