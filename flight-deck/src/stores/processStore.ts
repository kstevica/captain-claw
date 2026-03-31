import { create } from 'zustand'
import type { ProcessInfo } from '../services/docker'
import * as dockerApi from '../services/docker'

const DESC_KEY = 'fd:process-descriptions'
const NAME_KEY = 'fd:process-names'
const FWD_KEY = 'fd:process-forwarding-tasks'
const APPROVAL_KEY = 'fd:process-consult-approval'

function loadMap(key: string): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}
function saveMap(key: string, m: Record<string, string>) {
  localStorage.setItem(key, JSON.stringify(m))
}
function loadBoolMap(key: string): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}
function saveBoolMap(key: string, m: Record<string, boolean>) {
  localStorage.setItem(key, JSON.stringify(m))
}

interface ProcessStore {
  processes: ProcessInfo[]
  loading: boolean
  descriptionOverrides: Record<string, string>
  nameOverrides: Record<string, string>
  forwardingTaskOverrides: Record<string, string>
  consultApprovalOverrides: Record<string, boolean>

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
}

export const useProcessStore = create<ProcessStore>((set, get) => ({
  processes: [],
  loading: false,
  descriptionOverrides: loadMap(DESC_KEY),
  nameOverrides: loadMap(NAME_KEY),
  forwardingTaskOverrides: loadMap(FWD_KEY),
  consultApprovalOverrides: loadBoolMap(APPROVAL_KEY),

  fetchProcesses: async () => {
    try {
      const processes = await dockerApi.listProcesses()
      const descOverrides = get().descriptionOverrides
      const nameOverrides = get().nameOverrides
      const merged = processes.map((p) => ({
        ...p,
        ...(descOverrides[p.slug] != null ? { description: descOverrides[p.slug] } : {}),
        ...(nameOverrides[p.slug] != null ? { name: nameOverrides[p.slug] } : {}),
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
    saveMap(DESC_KEY, overrides)
    const processes = get().processes.map((p) => p.slug === slug ? { ...p, description } : p)
    set({ descriptionOverrides: overrides, processes })
  },

  setNameOverride: (slug, name) => {
    const overrides = { ...get().nameOverrides, [slug]: name }
    saveMap(NAME_KEY, overrides)
    const processes = get().processes.map((p) => p.slug === slug ? { ...p, name } : p)
    set({ nameOverrides: overrides, processes })
  },

  setForwardingTask: (slug, task) => {
    const overrides = { ...get().forwardingTaskOverrides, [slug]: task }
    saveMap(FWD_KEY, overrides)
    set({ forwardingTaskOverrides: overrides })
  },

  getForwardingTask: (slug) => get().forwardingTaskOverrides[slug] || '',

  setConsultApproval: (slug, required) => {
    const overrides = { ...get().consultApprovalOverrides, [slug]: required }
    saveBoolMap(APPROVAL_KEY, overrides)
    set({ consultApprovalOverrides: overrides })
  },

  getConsultApproval: (slug) => get().consultApprovalOverrides[slug] ?? false,
}))
