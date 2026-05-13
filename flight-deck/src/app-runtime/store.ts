import { create } from 'zustand'
import type { AgentManifest } from './types'
import { fetchAppList, fetchManifest, type AppSummary } from './manifests'

const SELECTED_AGENT_KEY = 'fd:app-runtime:selected-agent'

function loadSelected(): string | null {
  try { return localStorage.getItem(SELECTED_AGENT_KEY) } catch { return null }
}
function persistSelected(id: string | null) {
  try {
    if (id) localStorage.setItem(SELECTED_AGENT_KEY, id)
    else localStorage.removeItem(SELECTED_AGENT_KEY)
  } catch { /* ignore */ }
}

interface AppRuntimeState {
  manifest: AgentManifest | null
  agentId: string | null
  surfaceId: string | null
  selectedEntity: { type: string; id: string; data?: Record<string, unknown> } | null

  availableApps: AppSummary[]
  appsLoading: boolean
  manifestLoading: boolean
  error: string | null

  // One-shot request: when set, AppHost opens the authoring dialog and clears it.
  pendingAuthoring: 'new' | 'edit' | null

  refreshAppList: () => Promise<void>
  loadAgent: (agentId: string) => Promise<void>
  setSurface: (surfaceId: string) => void
  selectEntity: (type: string, id: string, data?: Record<string, unknown>) => void
  clearEntity: () => void
  requestAuthoring: (mode: 'new' | 'edit') => void
  clearAuthoringRequest: () => void
}

export const useAppRuntime = create<AppRuntimeState>((set, get) => ({
  manifest: null,
  agentId: loadSelected(),
  surfaceId: null,
  selectedEntity: null,

  availableApps: [],
  appsLoading: false,
  manifestLoading: false,
  error: null,
  pendingAuthoring: null,

  refreshAppList: async () => {
    set({ appsLoading: true, error: null })
    try {
      const apps = await fetchAppList()
      set({ availableApps: apps, appsLoading: false })
    } catch (exc) {
      set({
        availableApps: [],
        appsLoading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },

  loadAgent: async (agentId) => {
    set({ manifestLoading: true, error: null })
    const m = await fetchManifest(agentId)
    if (!m) {
      persistSelected(null)
      set({
        manifest: null,
        agentId: null,
        surfaceId: null,
        selectedEntity: null,
        manifestLoading: false,
        error: `Manifest "${agentId}" not found`,
      })
      return
    }
    persistSelected(agentId)
    set({
      manifest: m,
      agentId,
      surfaceId: m.home_surface ?? Object.keys(m.surfaces)[0] ?? null,
      selectedEntity: null,
      manifestLoading: false,
    })
  },

  setSurface: (surfaceId) => {
    const m = get().manifest
    if (!m || !m.surfaces[surfaceId]) return
    set({ surfaceId, selectedEntity: null })
  },

  selectEntity: (type, id, data) => {
    const m = get().manifest
    if (!m) return
    const entitySurface = Object.values(m.surfaces).find(
      (s) => s.layout === 'entity' && s.entity === type,
    )
    set({
      selectedEntity: { type, id, data },
      surfaceId: entitySurface?.id ?? get().surfaceId,
    })
  },

  clearEntity: () => set({ selectedEntity: null }),

  requestAuthoring: (mode) => set({ pendingAuthoring: mode }),
  clearAuthoringRequest: () => set({ pendingAuthoring: null }),
}))

// Apply $entity.id / $entity.<field> templates against the current entity.
export function applyTemplates(
  values: Record<string, string> | undefined,
  entity: { id: string; data?: Record<string, unknown> } | null,
): Record<string, unknown> {
  if (!values) return {}
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(values)) {
    if (typeof v !== 'string' || !v.startsWith('$')) {
      out[k] = v
      continue
    }
    if (v === '$entity.id') {
      out[k] = entity?.id ?? ''
      continue
    }
    if (v.startsWith('$entity.') && entity?.data) {
      const field = v.slice('$entity.'.length)
      out[k] = entity.data[field] ?? ''
      continue
    }
    out[k] = v
  }
  return out
}
