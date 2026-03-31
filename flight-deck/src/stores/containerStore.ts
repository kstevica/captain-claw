import { create } from 'zustand'
import type { ContainerInfo } from '../services/docker'
import * as dockerApi from '../services/docker'
import { usePipelineStore } from './pipelineStore'
import { useGroupStore } from './groupStore'

const DESC_OVERRIDES_KEY = 'fd:container-descriptions'
const NAME_OVERRIDES_KEY = 'fd:container-names'
const FWD_TASK_OVERRIDES_KEY = 'fd:container-forwarding-tasks'
const CONSULT_APPROVAL_KEY = 'fd:container-consult-approval'

function loadDescOverrides(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(DESC_OVERRIDES_KEY) || '{}') } catch { return {} }
}
function saveDescOverrides(m: Record<string, string>) {
  localStorage.setItem(DESC_OVERRIDES_KEY, JSON.stringify(m))
}
function loadNameOverrides(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(NAME_OVERRIDES_KEY) || '{}') } catch { return {} }
}
function saveNameOverrides(m: Record<string, string>) {
  localStorage.setItem(NAME_OVERRIDES_KEY, JSON.stringify(m))
}
function loadFwdTaskOverrides(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(FWD_TASK_OVERRIDES_KEY) || '{}') } catch { return {} }
}
function saveFwdTaskOverrides(m: Record<string, string>) {
  localStorage.setItem(FWD_TASK_OVERRIDES_KEY, JSON.stringify(m))
}
function loadConsultApproval(): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(CONSULT_APPROVAL_KEY) || '{}') } catch { return {} }
}
function saveConsultApproval(m: Record<string, boolean>) {
  localStorage.setItem(CONSULT_APPROVAL_KEY, JSON.stringify(m))
}

interface ContainerStore {
  containers: ContainerInfo[]
  selectedContainerId: string | null
  dockerAvailable: boolean
  loading: boolean
  descriptionOverrides: Record<string, string>
  nameOverrides: Record<string, string>
  forwardingTaskOverrides: Record<string, string>
  consultApprovalOverrides: Record<string, boolean>

  fetchContainers: () => Promise<void>
  selectContainer: (id: string | null) => void
  stopContainer: (id: string) => Promise<void>
  startContainer: (id: string) => Promise<void>
  restartContainer: (id: string) => Promise<void>
  removeContainer: (id: string) => Promise<void>
  rebuildContainer: (id: string) => Promise<void>
  cloneContainer: (id: string, newName: string) => Promise<void>
  setNameOverride: (id: string, name: string) => void
  checkHealth: () => Promise<void>
  setDescription: (id: string, description: string) => void
  setForwardingTask: (id: string, task: string) => void
  getForwardingTask: (id: string) => string
  setConsultApproval: (id: string, required: boolean) => void
  getConsultApproval: (id: string) => boolean
}

export const useContainerStore = create<ContainerStore>((set, get) => ({
  containers: [],
  selectedContainerId: null,
  dockerAvailable: false,
  loading: false,
  descriptionOverrides: loadDescOverrides(),
  nameOverrides: loadNameOverrides(),
  forwardingTaskOverrides: loadFwdTaskOverrides(),
  consultApprovalOverrides: loadConsultApproval(),

  fetchContainers: async () => {
    try {
      const containers = await dockerApi.listContainers()
      // Merge in local description and name overrides
      const descOverrides = get().descriptionOverrides
      const nameOverrides = get().nameOverrides
      const merged = containers.map((c) => {
        const descOverride = descOverrides[c.id] ?? descOverrides[c.name]
        const nameOverride = nameOverrides[c.id] ?? nameOverrides[c.name]
        return {
          ...c,
          ...(descOverride != null ? { description: descOverride } : {}),
          ...(nameOverride != null ? { agent_name: nameOverride } : {}),
        }
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

      // Migrate name override from old ID to new ID
      const nameOverrides = { ...get().nameOverrides }
      if (nameOverrides[oldId]) {
        nameOverrides[newId] = nameOverrides[oldId]
        delete nameOverrides[oldId]
        saveNameOverrides(nameOverrides)
        set({ nameOverrides })
      }

      // Migrate forwarding task override from old ID to new ID
      const fwdTaskOverrides = { ...get().forwardingTaskOverrides }
      if (fwdTaskOverrides[oldId]) {
        fwdTaskOverrides[newId] = fwdTaskOverrides[oldId]
        delete fwdTaskOverrides[oldId]
        saveFwdTaskOverrides(fwdTaskOverrides)
        set({ forwardingTaskOverrides: fwdTaskOverrides })
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

  cloneContainer: async (id, newName) => {
    await dockerApi.cloneContainer(id, newName)
    get().fetchContainers()
  },

  setNameOverride: (id, name) => {
    const overrides = { ...get().nameOverrides, [id]: name }
    saveNameOverrides(overrides)
    const containers = get().containers.map((c) => c.id === id ? { ...c, agent_name: name } : c)
    set({ nameOverrides: overrides, containers })
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

  setForwardingTask: (id, task) => {
    const overrides = { ...get().forwardingTaskOverrides, [id]: task }
    saveFwdTaskOverrides(overrides)
    set({ forwardingTaskOverrides: overrides })
  },

  getForwardingTask: (id) => {
    return get().forwardingTaskOverrides[id] || ''
  },

  setConsultApproval: (id, required) => {
    const overrides = { ...get().consultApprovalOverrides, [id]: required }
    saveConsultApproval(overrides)
    set({ consultApprovalOverrides: overrides })
  },

  getConsultApproval: (id) => {
    return get().consultApprovalOverrides[id] ?? false
  },
}))
