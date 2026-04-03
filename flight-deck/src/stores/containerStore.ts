import { create } from 'zustand'
import type { ContainerInfo } from '../services/docker'
import * as dockerApi from '../services/docker'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { useAuthStore } from './authStore'
import { usePipelineStore } from './pipelineStore'
import { useGroupStore } from './groupStore'

const DESC_OVERRIDES_KEY = 'fd:container-descriptions'
const NAME_OVERRIDES_KEY = 'fd:container-names'
const FWD_TASK_OVERRIDES_KEY = 'fd:container-forwarding-tasks'
const CONSULT_APPROVAL_KEY = 'fd:container-consult-approval'
const FLEET_INSTRUCTIONS_KEY = 'fd:container-fleet-instructions'

function loadMap(key: string): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}
function loadBoolMap(key: string): Record<string, boolean> {
  try { return JSON.parse(localStorage.getItem(key) || '{}') } catch { return {} }
}
function saveMap(key: string, m: Record<string, string | boolean>) {
  const val = JSON.stringify(m)
  if (useAuthStore.getState().authEnabled) queueSave(key, val)
  else localStorage.setItem(key, val)
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
  fleetInstructionsOverrides: Record<string, string>

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
  setFleetInstructions: (id: string, instructions: string) => void
  getFleetInstructions: (id: string) => string
}

export const useContainerStore = create<ContainerStore>((set, get) => ({
  containers: [],
  selectedContainerId: null,
  dockerAvailable: false,
  loading: false,
  descriptionOverrides: loadMap(DESC_OVERRIDES_KEY),
  nameOverrides: loadMap(NAME_OVERRIDES_KEY),
  forwardingTaskOverrides: loadMap(FWD_TASK_OVERRIDES_KEY),
  consultApprovalOverrides: loadBoolMap(CONSULT_APPROVAL_KEY),
  fleetInstructionsOverrides: loadMap(FLEET_INSTRUCTIONS_KEY),

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
        saveMap(DESC_OVERRIDES_KEY, overrides)
        set({ descriptionOverrides: overrides })
      }

      // Migrate name override from old ID to new ID
      const nameOverrides = { ...get().nameOverrides }
      if (nameOverrides[oldId]) {
        nameOverrides[newId] = nameOverrides[oldId]
        delete nameOverrides[oldId]
        saveMap(NAME_OVERRIDES_KEY, nameOverrides)
        set({ nameOverrides })
      }

      // Migrate forwarding task override from old ID to new ID
      const fwdTaskOverrides = { ...get().forwardingTaskOverrides }
      if (fwdTaskOverrides[oldId]) {
        fwdTaskOverrides[newId] = fwdTaskOverrides[oldId]
        delete fwdTaskOverrides[oldId]
        saveMap(FWD_TASK_OVERRIDES_KEY, fwdTaskOverrides)
        set({ forwardingTaskOverrides: fwdTaskOverrides })
      }

      // Migrate fleet instructions override from old ID to new ID
      const fleetInstOverrides = { ...get().fleetInstructionsOverrides }
      if (fleetInstOverrides[oldId]) {
        fleetInstOverrides[newId] = fleetInstOverrides[oldId]
        delete fleetInstOverrides[oldId]
        saveMap(FLEET_INSTRUCTIONS_KEY, fleetInstOverrides)
        set({ fleetInstructionsOverrides: fleetInstOverrides })
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
    saveMap(NAME_OVERRIDES_KEY, overrides)
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
    saveMap(DESC_OVERRIDES_KEY, overrides)
    const containers = get().containers.map((c) => c.id === id ? { ...c, description } : c)
    set({ descriptionOverrides: overrides, containers })
  },

  setForwardingTask: (id, task) => {
    const overrides = { ...get().forwardingTaskOverrides, [id]: task }
    saveMap(FWD_TASK_OVERRIDES_KEY, overrides)
    set({ forwardingTaskOverrides: overrides })
  },

  getForwardingTask: (id) => {
    return get().forwardingTaskOverrides[id] || ''
  },

  setConsultApproval: (id, required) => {
    const overrides = { ...get().consultApprovalOverrides, [id]: required }
    saveMap(CONSULT_APPROVAL_KEY, overrides)
    set({ consultApprovalOverrides: overrides })
  },

  getConsultApproval: (id) => {
    return get().consultApprovalOverrides[id] ?? false
  },

  setFleetInstructions: (id, instructions) => {
    const overrides = { ...get().fleetInstructionsOverrides, [id]: instructions }
    saveMap(FLEET_INSTRUCTIONS_KEY, overrides)
    set({ fleetInstructionsOverrides: overrides })
  },

  getFleetInstructions: (id) => {
    return get().fleetInstructionsOverrides[id] || ''
  },
}))

registerHydrator((settings) => {
  const updates: Partial<Record<string, any>> = {}
  for (const [key, field] of [
    [DESC_OVERRIDES_KEY, 'descriptionOverrides'],
    [NAME_OVERRIDES_KEY, 'nameOverrides'],
    [FWD_TASK_OVERRIDES_KEY, 'forwardingTaskOverrides'],
    [CONSULT_APPROVAL_KEY, 'consultApprovalOverrides'],
    [FLEET_INSTRUCTIONS_KEY, 'fleetInstructionsOverrides'],
  ] as const) {
    const raw = settings[key]
    if (raw) {
      try { updates[field] = JSON.parse(raw) } catch { /* ignore */ }
    }
  }
  if (Object.keys(updates).length > 0) {
    useContainerStore.setState(updates as any)
  }
})
