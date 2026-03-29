import { create } from 'zustand'
import type { ContainerInfo } from '../services/docker'
import * as dockerApi from '../services/docker'
import { usePipelineStore } from './pipelineStore'
import { useGroupStore } from './groupStore'

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
  rebuildContainer: (id: string) => Promise<void>
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

  rebuildContainer: async (id) => {
    // Send current description to preserve it
    const desc = get().descriptionOverrides[id] || get().containers.find(c => c.id === id)?.description || ''
    const result = await dockerApi.rebuildContainer(id, desc)

    // Migrate description override from old ID to new ID
    if (result.old_container_id && result.container_id !== result.old_container_id) {
      const oldId = result.old_container_id
      const newId = result.container_id
      const overrides = { ...get().descriptionOverrides }
      if (overrides[oldId]) {
        overrides[newId] = overrides[oldId]
        delete overrides[oldId]
        saveDescOverrides(overrides)
        set({ descriptionOverrides: overrides })
      }

      // Migrate pipeline step references from old ID to new ID
      const pipelineStore = usePipelineStore.getState()
      for (const pipeline of pipelineStore.pipelines) {
        const hasOldRef = pipeline.steps.some(s => s.agentId === oldId)
        if (hasOldRef) {
          const newSteps = pipeline.steps.map(s =>
            s.agentId === oldId ? { ...s, agentId: newId } : s
          )
          pipelineStore.updatePipeline(pipeline.id, { steps: newSteps })
        }
      }

      // Migrate group memberships from old ID to new ID
      const groupStore = useGroupStore.getState()
      for (const group of groupStore.groups) {
        if (group.agentIds.includes(oldId)) {
          groupStore.removeFromGroup(group.id, oldId)
          groupStore.addToGroup(group.id, newId)
        }
      }

      // Migrate agent card positions
      try {
        const posData = JSON.parse(localStorage.getItem('fd:agent-positions') || '{}')
        if (posData[oldId]) {
          posData[newId] = posData[oldId]
          delete posData[oldId]
          localStorage.setItem('fd:agent-positions', JSON.stringify(posData))
        }
      } catch { /* ignore */ }
    }

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
