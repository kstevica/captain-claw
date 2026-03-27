import { create } from 'zustand'
import type { ContainerInfo } from '../services/docker'
import * as dockerApi from '../services/docker'

interface ContainerStore {
  containers: ContainerInfo[]
  selectedContainerId: string | null
  dockerAvailable: boolean
  loading: boolean

  fetchContainers: () => Promise<void>
  selectContainer: (id: string | null) => void
  stopContainer: (id: string) => Promise<void>
  startContainer: (id: string) => Promise<void>
  restartContainer: (id: string) => Promise<void>
  removeContainer: (id: string) => Promise<void>
  checkHealth: () => Promise<void>
}

export const useContainerStore = create<ContainerStore>((set, get) => ({
  containers: [],
  selectedContainerId: null,
  dockerAvailable: false,
  loading: false,

  fetchContainers: async () => {
    try {
      const containers = await dockerApi.listContainers()
      set({ containers, dockerAvailable: true })
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
}))
