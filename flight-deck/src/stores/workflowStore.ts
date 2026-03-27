import { create } from 'zustand'
import type { Swarm, SwarmTask, SwarmEdge, SwarmProject } from '../types'
import * as api from '../services/api'

interface WorkflowStore {
  // Data
  projects: SwarmProject[]
  swarms: Swarm[]
  activeSwarmId: string | null
  tasks: SwarmTask[]
  edges: SwarmEdge[]

  // Project actions
  fetchProjects: () => Promise<void>
  createProject: (name: string, description?: string) => Promise<SwarmProject>

  // Swarm actions
  fetchSwarms: (projectId?: string) => Promise<void>
  selectSwarm: (id: string | null) => Promise<void>
  createSwarm: (data: Partial<Swarm>) => Promise<Swarm>
  startSwarm: (id: string) => Promise<void>
  pauseSwarm: (id: string) => Promise<void>
  resumeSwarm: (id: string) => Promise<void>
  cancelSwarm: (id: string) => Promise<void>
  decomposeSwarm: (id: string) => Promise<void>

  // Task actions
  createTask: (data: Partial<SwarmTask>) => Promise<SwarmTask>
  updateTask: (taskId: string, data: Partial<SwarmTask>) => Promise<void>
  deleteTask: (taskId: string) => Promise<void>
  approveTask: (taskId: string) => Promise<void>
  retryTask: (taskId: string) => Promise<void>
  skipTask: (taskId: string) => Promise<void>

  // Edge actions
  createEdge: (from: string, to: string, type?: 'dependency' | 'data_flow') => Promise<void>
  deleteEdge: (edgeId: number) => Promise<void>

  // Real-time
  updateTaskStatus: (taskId: string, status: string) => void
  updateSwarmStatus: (swarmId: string, status: string) => void
}

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
  projects: [],
  swarms: [],
  activeSwarmId: null,
  tasks: [],
  edges: [],

  fetchProjects: async () => {
    const projects = await api.getProjects()
    set({ projects })
  },

  createProject: async (name, description) => {
    const p = await api.createProject({ name, description })
    set({ projects: [...get().projects, p] })
    return p
  },

  fetchSwarms: async (projectId) => {
    const swarms = await api.getSwarms(projectId)
    set({ swarms })
  },

  selectSwarm: async (id) => {
    set({ activeSwarmId: id })
    if (id) {
      const [tasks, swarm] = await Promise.all([
        api.getTasks(id),
        api.getSwarm(id),
      ])
      // Extract edges from the swarm response or fetch separately
      set({
        tasks,
        swarms: get().swarms.map((s) => (s.id === id ? swarm : s)),
      })
    } else {
      set({ tasks: [], edges: [] })
    }
  },

  createSwarm: async (data) => {
    const swarm = await api.createSwarm(data)
    set({ swarms: [...get().swarms, swarm] })
    return swarm
  },

  startSwarm: async (id) => {
    await api.startSwarm(id)
    const swarm = await api.getSwarm(id)
    set({ swarms: get().swarms.map((s) => (s.id === id ? swarm : s)) })
  },

  pauseSwarm: async (id) => {
    await api.pauseSwarm(id)
    get().updateSwarmStatus(id, 'paused')
  },

  resumeSwarm: async (id) => {
    await api.resumeSwarm(id)
    get().updateSwarmStatus(id, 'running')
  },

  cancelSwarm: async (id) => {
    await api.cancelSwarm(id)
    get().updateSwarmStatus(id, 'cancelled')
  },

  decomposeSwarm: async (id) => {
    await api.decomposeSwarm(id)
    // Refetch tasks after decomposition
    const tasks = await api.getTasks(id)
    set({ tasks })
  },

  createTask: async (data) => {
    const swarmId = get().activeSwarmId
    if (!swarmId) throw new Error('No active swarm')
    const task = await api.createTask(swarmId, data)
    set({ tasks: [...get().tasks, task] })
    return task
  },

  updateTask: async (taskId, data) => {
    const task = await api.updateTask(taskId, data)
    set({ tasks: get().tasks.map((t) => (t.id === taskId ? { ...t, ...task } : t)) })
  },

  deleteTask: async (taskId) => {
    await api.deleteTask(taskId)
    set({
      tasks: get().tasks.filter((t) => t.id !== taskId),
      edges: get().edges.filter((e) => e.from_task_id !== taskId && e.to_task_id !== taskId),
    })
  },

  approveTask: async (taskId) => {
    await api.approveTask(taskId)
    set({ tasks: get().tasks.map((t) => (t.id === taskId ? { ...t, approval_status: 'approved' as const, status: 'waiting' as const } : t)) })
  },

  retryTask: async (taskId) => {
    await api.retryTask(taskId)
    set({ tasks: get().tasks.map((t) => (t.id === taskId ? { ...t, status: 'retrying' as const } : t)) })
  },

  skipTask: async (taskId) => {
    await api.skipTask(taskId)
    set({ tasks: get().tasks.map((t) => (t.id === taskId ? { ...t, status: 'skipped' as const } : t)) })
  },

  createEdge: async (from, to, type = 'dependency') => {
    const swarmId = get().activeSwarmId
    if (!swarmId) return
    const edge = await api.createEdge(swarmId, { from_task_id: from, to_task_id: to, edge_type: type })
    set({ edges: [...get().edges, edge] })
  },

  deleteEdge: async (edgeId) => {
    await api.deleteEdge(edgeId)
    set({ edges: get().edges.filter((e) => e.id !== edgeId) })
  },

  updateTaskStatus: (taskId, status) => {
    set({ tasks: get().tasks.map((t) => (t.id === taskId ? { ...t, status: status as SwarmTask['status'] } : t)) })
  },

  updateSwarmStatus: (swarmId, status) => {
    set({ swarms: get().swarms.map((s) => (s.id === swarmId ? { ...s, status: status as Swarm['status'] } : s)) })
  },
}))
