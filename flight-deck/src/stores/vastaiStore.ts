import { create } from 'zustand'
import { useAuthStore } from './authStore'

// ── Types ──

export interface VastOffer {
  id: number
  gpu_name: string
  gpu_ram_gb: number
  num_gpus: number
  cpu_cores: number
  ram_gb: number
  disk_gb: number
  dph_total: number
  storage_cost_per_gb_month: number
  reliability: number
  inet_down_mbps: number
  inet_up_mbps: number
  cuda_version: number
  direct_port_count: number
  geolocation: string
  verified: boolean
}

export interface VastInstance {
  id: number
  offer_id: number
  gpu_name: string
  num_gpus: number
  gpu_ram_gb: number
  state: string
  public_ip: string
  ollama_port: number
  ssh_port: number
  auth_token: string
  dph_total: number
  disk_gb: number
  created_at: string
  label: string
  auto_stop_minutes: number
  last_activity_at: string
  ollama_ready: boolean
  ollama_error: string
  models_loaded: string[]
}

export interface VastOfferFilter {
  gpu_name?: string
  min_gpu_ram_gb?: number
  max_price_per_hour?: number
  min_reliability?: number
  num_gpus?: number
  limit?: number
}

export interface VastConnectionInfo {
  provider: string
  base_url: string
  api_key: string
  ollama_ready: boolean
  models: string[]
}

export interface OllamaModel {
  name: string
  size?: number
  digest?: string
  modified_at?: string
  details?: Record<string, unknown>
}

interface VastAIStore {
  // Status
  configured: boolean
  balance: number | null
  email: string

  // Data
  instances: VastInstance[]
  offers: VastOffer[]

  // UI state
  loading: boolean
  offersLoading: boolean
  error: string | null
  actionLoading: string | null  // instance_id:action being performed

  // Actions
  refreshStatus: () => Promise<void>
  refreshInstances: () => Promise<void>
  searchOffers: (filters?: VastOfferFilter) => Promise<void>
  setApiKey: (key: string) => Promise<boolean>
  removeApiKey: () => Promise<boolean>
  createInstance: (offerId: number, label?: string, diskGb?: number, prePullModel?: string) => Promise<VastInstance | null>
  stopInstance: (id: number) => Promise<boolean>
  startInstance: (id: number) => Promise<boolean>
  destroyInstance: (id: number) => Promise<boolean>
  setAutoStop: (id: number, minutes: number) => Promise<boolean>
  getConnectionInfo: (id: number) => Promise<VastConnectionInfo | null>
  listModels: (id: number) => Promise<OllamaModel[]>
  pullModel: (id: number, model: string) => Promise<boolean>
  deleteModel: (id: number, model: string) => Promise<boolean>
  clearError: () => void
}

// ── Helpers ──

function authHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const resp = await fetch(url, {
    credentials: 'include',
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers || {}),
    },
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  return resp.json()
}

// ── Store ──

export const useVastAIStore = create<VastAIStore>((set, get) => ({
  configured: false,
  balance: null,
  email: '',
  instances: [],
  offers: [],
  loading: false,
  offersLoading: false,
  error: null,
  actionLoading: null,

  clearError: () => set({ error: null }),

  refreshStatus: async () => {
    set({ loading: true, error: null })
    try {
      const data = await fetchJson('/fd/vastai/status')
      set({
        configured: data.configured ?? false,
        balance: data.balance ?? null,
        email: data.email ?? '',
        loading: false,
      })
    } catch (exc) {
      set({ loading: false, error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  refreshInstances: async () => {
    try {
      const data = await fetchJson('/fd/vastai/instances')
      set({ instances: data.instances ?? [] })
    } catch (exc) {
      set({ error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  searchOffers: async (filters?: VastOfferFilter) => {
    set({ offersLoading: true, error: null })
    try {
      const data = await fetchJson('/fd/vastai/offers/search', {
        method: 'POST',
        body: JSON.stringify(filters ?? {}),
      })
      set({ offers: data.offers ?? [], offersLoading: false })
    } catch (exc) {
      set({ offersLoading: false, error: exc instanceof Error ? exc.message : String(exc) })
    }
  },

  setApiKey: async (key: string) => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/vastai/api-key', {
        method: 'PUT',
        body: JSON.stringify({ api_key: key }),
      })
      set({ configured: true, loading: false })
      // Refresh status to get balance
      get().refreshStatus()
      return true
    } catch (exc) {
      set({ loading: false, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  removeApiKey: async () => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/vastai/api-key', { method: 'DELETE' })
      set({ configured: false, balance: null, email: '', instances: [], loading: false })
      return true
    } catch (exc) {
      set({ loading: false, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  createInstance: async (offerId, label, diskGb, prePullModel) => {
    set({ actionLoading: 'create', error: null })
    try {
      const inst = await fetchJson('/fd/vastai/instances', {
        method: 'POST',
        body: JSON.stringify({
          offer_id: offerId,
          label: label ?? '',
          disk_gb: diskGb ?? 64,
          pre_pull_model: prePullModel ?? '',
        }),
      })
      set((s) => ({
        instances: [...s.instances, inst],
        actionLoading: null,
      }))
      return inst
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return null
    }
  },

  stopInstance: async (id) => {
    set({ actionLoading: `${id}:stop`, error: null })
    try {
      const updated = await fetchJson(`/fd/vastai/instances/${id}/stop`, { method: 'POST' })
      set((s) => ({
        instances: s.instances.map((i) => (i.id === id ? updated : i)),
        actionLoading: null,
      }))
      return true
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  startInstance: async (id) => {
    set({ actionLoading: `${id}:start`, error: null })
    try {
      const updated = await fetchJson(`/fd/vastai/instances/${id}/start`, { method: 'POST' })
      set((s) => ({
        instances: s.instances.map((i) => (i.id === id ? updated : i)),
        actionLoading: null,
      }))
      return true
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  destroyInstance: async (id) => {
    set({ actionLoading: `${id}:destroy`, error: null })
    try {
      await fetchJson(`/fd/vastai/instances/${id}`, { method: 'DELETE' })
      set((s) => ({
        instances: s.instances.map((i) =>
          i.id === id ? { ...i, state: 'destroyed', ollama_ready: false } : i,
        ),
        actionLoading: null,
      }))
      return true
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  setAutoStop: async (id, minutes) => {
    try {
      const updated = await fetchJson(`/fd/vastai/instances/${id}/auto-stop`, {
        method: 'PUT',
        body: JSON.stringify({ auto_stop_minutes: minutes }),
      })
      set((s) => ({
        instances: s.instances.map((i) => (i.id === id ? updated : i)),
      }))
      return true
    } catch (exc) {
      set({ error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  getConnectionInfo: async (id) => {
    try {
      return await fetchJson(`/fd/vastai/instances/${id}/connection`)
    } catch {
      return null
    }
  },

  listModels: async (id) => {
    try {
      const data = await fetchJson(`/fd/vastai/instances/${id}/models`)
      return data.models ?? []
    } catch {
      return []
    }
  },

  pullModel: async (id, model) => {
    set({ actionLoading: `${id}:pull`, error: null })
    try {
      await fetchJson(`/fd/vastai/instances/${id}/models/pull`, {
        method: 'POST',
        body: JSON.stringify({ model }),
      })
      // Refresh instance list to get updated models_loaded
      get().refreshInstances()
      set({ actionLoading: null })
      return true
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },

  deleteModel: async (id, model) => {
    set({ actionLoading: `${id}:delete-model`, error: null })
    try {
      await fetchJson(`/fd/vastai/instances/${id}/models/${encodeURIComponent(model)}`, {
        method: 'DELETE',
      })
      get().refreshInstances()
      set({ actionLoading: null })
      return true
    } catch (exc) {
      set({ actionLoading: null, error: exc instanceof Error ? exc.message : String(exc) })
      return false
    }
  },
}))
