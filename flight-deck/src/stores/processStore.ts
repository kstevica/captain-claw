import { create } from 'zustand'
import type { ProcessInfo } from '../services/docker'
import * as dockerApi from '../services/docker'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'

const DESC_KEY = 'fd:process-descriptions'
const NAME_KEY = 'fd:process-names'
const FWD_KEY = 'fd:process-forwarding-tasks'
const APPROVAL_KEY = 'fd:process-consult-approval'
const FLEET_INSTRUCTIONS_KEY = 'fd:process-fleet-instructions'
const COGNITIVE_MODE_KEY = 'fd:process-cognitive-modes'
const ECO_MODE_KEY = 'fd:process-eco-modes'
const MODEL_OVERRIDE_KEY = 'fd:process-model-overrides'

function loadMap(key: string): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}
function persistMap(key: string, m: Record<string, string | boolean>) {
  const val = JSON.stringify(m)
  if (useAuthStore.getState().authEnabled) queueSave(key, val)
  else localStorage.setItem(key, val)
}
function loadBoolMap(key: string): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}

interface ProcessStore {
  processes: ProcessInfo[]
  loading: boolean
  descriptionOverrides: Record<string, string>
  nameOverrides: Record<string, string>
  forwardingTaskOverrides: Record<string, string>
  consultApprovalOverrides: Record<string, boolean>
  fleetInstructionsOverrides: Record<string, string>
  cognitiveModeOverrides: Record<string, string>
  ecoModeOverrides: Record<string, boolean>
  modelOverrides: Record<string, { provider: string; model: string }>

  fetchProcesses: () => Promise<void>
  stopProcess: (slug: string) => Promise<void>
  startProcess: (slug: string) => Promise<void>
  restartProcess: (slug: string) => Promise<void>
  removeProcess: (slug: string) => Promise<void>
  cloneProcess: (slug: string, newName: string) => Promise<void>
  setDescription: (slug: string, description: string) => void
  setNameOverride: (slug: string, name: string) => void
  setForwardingTask: (slug: string, task: string) => void
  getForwardingTask: (slug: string) => string
  setConsultApproval: (slug: string, required: boolean) => void
  getConsultApproval: (slug: string) => boolean
  setFleetInstructions: (slug: string, instructions: string) => void
  getFleetInstructions: (slug: string) => string
  setCognitiveMode: (slug: string, mode: string) => void
  getCognitiveMode: (slug: string) => string
  setEcoMode: (slug: string, enabled: boolean) => void
  getEcoMode: (slug: string) => boolean
  setModelOverride: (slug: string, provider: string, model: string) => void
  getModelOverride: (slug: string) => { provider: string; model: string } | null
}

export const useProcessStore = create<ProcessStore>((set, get) => ({
  processes: [],
  loading: false,
  descriptionOverrides: loadMap(DESC_KEY),
  nameOverrides: loadMap(NAME_KEY),
  forwardingTaskOverrides: loadMap(FWD_KEY),
  consultApprovalOverrides: loadBoolMap(APPROVAL_KEY),
  fleetInstructionsOverrides: loadMap(FLEET_INSTRUCTIONS_KEY),
  cognitiveModeOverrides: loadMap(COGNITIVE_MODE_KEY),
  ecoModeOverrides: loadBoolMap(ECO_MODE_KEY),
  modelOverrides: (() => { try { return JSON.parse(localStorage.getItem(MODEL_OVERRIDE_KEY) || '{}') } catch { return {} } })(),

  fetchProcesses: async () => {
    try {
      const processes = await dockerApi.listProcesses()
      const descOverrides = get().descriptionOverrides
      const nameOverrides = get().nameOverrides
      const modelOverrides = get().modelOverrides
      const merged = processes.map((p) => ({
        ...p,
        ...(descOverrides[p.slug] != null ? { description: descOverrides[p.slug] } : {}),
        ...(nameOverrides[p.slug] != null ? { name: nameOverrides[p.slug] } : {}),
        ...(modelOverrides[p.slug] != null ? { provider: modelOverrides[p.slug].provider, model: modelOverrides[p.slug].model } : {}),
      }))
      set({ processes: merged })
    } catch {
      set({ processes: [] })
    }
  },

  stopProcess: async (slug) => {
    await dockerApi.stopProcess(slug)
    get().fetchProcesses()
  },

  startProcess: async (slug) => {
    await dockerApi.startProcess(slug)
    get().fetchProcesses()
  },

  restartProcess: async (slug) => {
    await dockerApi.restartProcess(slug)
    get().fetchProcesses()
  },

  removeProcess: async (slug) => {
    await dockerApi.removeProcess(slug)
    get().fetchProcesses()
  },

  cloneProcess: async (slug, newName) => {
    await dockerApi.cloneProcess(slug, newName)
    get().fetchProcesses()
  },

  setDescription: (slug, description) => {
    const overrides = { ...get().descriptionOverrides, [slug]: description }
    persistMap(DESC_KEY, overrides)
    const processes = get().processes.map((p) => p.slug === slug ? { ...p, description } : p)
    set({ descriptionOverrides: overrides, processes })
  },

  setNameOverride: (slug, name) => {
    const overrides = { ...get().nameOverrides, [slug]: name }
    persistMap(NAME_KEY, overrides)
    const processes = get().processes.map((p) => p.slug === slug ? { ...p, name } : p)
    set({ nameOverrides: overrides, processes })
  },

  setForwardingTask: (slug, task) => {
    const overrides = { ...get().forwardingTaskOverrides, [slug]: task }
    persistMap(FWD_KEY, overrides)
    set({ forwardingTaskOverrides: overrides })
  },

  getForwardingTask: (slug) => get().forwardingTaskOverrides[slug] || '',

  setConsultApproval: (slug, required) => {
    const overrides = { ...get().consultApprovalOverrides, [slug]: required }
    persistMap(APPROVAL_KEY, overrides)
    set({ consultApprovalOverrides: overrides })
  },

  getConsultApproval: (slug) => get().consultApprovalOverrides[slug] ?? false,

  setFleetInstructions: (slug, instructions) => {
    const overrides = { ...get().fleetInstructionsOverrides, [slug]: instructions }
    persistMap(FLEET_INSTRUCTIONS_KEY, overrides)
    set({ fleetInstructionsOverrides: overrides })
  },

  getFleetInstructions: (slug) => get().fleetInstructionsOverrides[slug] || '',

  setCognitiveMode: (slug, mode) => {
    const overrides = { ...get().cognitiveModeOverrides, [slug]: mode }
    persistMap(COGNITIVE_MODE_KEY, overrides)
    set({ cognitiveModeOverrides: overrides })
  },

  getCognitiveMode: (slug) => get().cognitiveModeOverrides[slug] || 'neutra',

  setEcoMode: (slug, enabled) => {
    const overrides = { ...get().ecoModeOverrides, [slug]: enabled }
    persistMap(ECO_MODE_KEY, overrides)
    set({ ecoModeOverrides: overrides })
  },

  getEcoMode: (slug) => get().ecoModeOverrides[slug] || false,

  setModelOverride: (slug, provider, model) => {
    const overrides = { ...get().modelOverrides, [slug]: { provider, model } }
    const val = JSON.stringify(overrides)
    if (useAuthStore.getState().authEnabled) queueSave(MODEL_OVERRIDE_KEY, val)
    else localStorage.setItem(MODEL_OVERRIDE_KEY, val)
    const processes = get().processes.map((p) => p.slug === slug ? { ...p, provider, model } : p)
    set({ modelOverrides: overrides, processes })
  },

  getModelOverride: (slug) => get().modelOverrides[slug] || null,
}))

registerHydrator((settings) => {
  const updates: Record<string, any> = {}
  for (const [key, field] of [
    [DESC_KEY, 'descriptionOverrides'],
    [NAME_KEY, 'nameOverrides'],
    [FWD_KEY, 'forwardingTaskOverrides'],
    [APPROVAL_KEY, 'consultApprovalOverrides'],
    [FLEET_INSTRUCTIONS_KEY, 'fleetInstructionsOverrides'],
    [COGNITIVE_MODE_KEY, 'cognitiveModeOverrides'],
    [ECO_MODE_KEY, 'ecoModeOverrides'],
    [MODEL_OVERRIDE_KEY, 'modelOverrides'],
  ] as const) {
    const raw = settings[key]
    if (raw) {
      try { updates[field] = JSON.parse(raw) } catch { /* ignore */ }
    }
  }
  if (Object.keys(updates).length > 0) {
    useProcessStore.setState(updates)
  }
})
