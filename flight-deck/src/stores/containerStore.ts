import { create } from 'zustand'
import type { ContainerInfo } from '../services/docker'
import * as dockerApi from '../services/docker'

const DESC_OVERRIDES_KEY = 'fd:container-descriptions'

function loadDescOverrides(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(DESC_OVERRIDES_KEY) || '{}') } catch { return {} }
}
function saveDescOverrides(m: Record<string, string>) {
  localStorage.setItem(DESC_OVERRIDES_KEY, JSON.stringify(m))
}

interface ContainerStore {
  containers: ContainerInfo[]
  selectedContainerId: string | null
  dockerAvailable: boolean
  loading: boolean
  descriptionOverrides: Record<string, string>

  fetchContainers: () => Promise<void>
  selectContainer: (id: string | null) => void
  stopContainer: (id: string) => Promise<void>
  startContainer: (id: string) => Promise<void>
  restartContainer: (id: string) => Promise<void>
  removeContainer: (id: string) => Promise<void>
  checkHealth: () => Promise<void>
  setDescription: (id: string, description: string) => void
}

export const useContainerStore = create<ContainerStore>((set, get) => ({
  containers: [],
  selectedContainerId: null,
  dockerAvailable: false,
  loading: false,
  descriptionOverrides: loadDescOverrides(),

  fetchContainers: async () => {
    try {
      const containers = await dockerApi.listContainers()
      // Merge in local description overrides
      const overrides = get().descriptionOverrides
      const merged = containers.map((c) => {
        const override = overrides[c.id] ?? overrides[c.name]
        return override != null ? { ...c, description: override } : c
      })
      set({ containers: merged, dockerAvailable: true })
    } catch {
      set({ containers: [], dockerAvailable: false })
    }
  },

  selectContainer: (id) => set({ selectedContainerId: id }),

  stopContainer: async (id) => {
    await dockerApi.stopContainer(id)
    get().fetchContainers()
  },

  startContainer: async (id) => {
    await dockerApi.startContainer(id)
    get().fetchContainers()
  },

  restartContainer: async (id) => {
    await dockerApi.restartContainer(id)
    get().fetchContainers()
  },

  removeContainer: async (id) => {
    await dockerApi.removeContainer(id, true)
    if (get().selectedContainerId === id) set({ selectedContainerId: null })
    get().fetchContainers()
  },

  checkHealth: async () => {
    try {
      const h = await dockerApi.healthCheck()
      set({ dockerAvailable: h.ok && h.docker })
    } catch {
      set({ dockerAvailable: false })
    }
  },

  setDescription: (id, description) => {
    const overrides = { ...get().descriptionOverrides, [id]: description }
    saveDescOverrides(overrides)
    const containers = get().containers.map((c) => c.id === id ? { ...c, description } : c)
    set({ descriptionOverrides: overrides, containers })
  },
}))
