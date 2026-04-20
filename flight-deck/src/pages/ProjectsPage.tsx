import { useState, useEffect, useCallback } from 'react'
import {
  FolderKanban, Plus, Loader2, Trash2, Users, Target,
  AlertTriangle, CheckCircle2, Clock, Wand2, UserPlus, UserMinus,
  FileText, Activity, ArrowRight, Pause, Archive,
} from 'lucide-react'
import { useUIStore } from '../stores/uiStore'

// ── Types ──

interface Project {
  id: string
  name: string
  description: string
  status: string
  goals: Goal[]
  config: Record<string, unknown>
  lead_agent_id: string
  created_at: string
  updated_at: string
  completed_at: string | null
  metadata: Record<string, unknown>
}

interface Goal {
  goal: string
  success_criteria: string
  priority: string
  status: string
  notes?: string
}

interface Member {
  id: string
  project_id: string
  agent_id: string
  agent_name: string
  role: string
  expertise_tags: string[]
  joined_at: string
  left_at: string | null
  contribution_summary: string
}

interface Artifact {
  id: string
  project_id: string
  kind: string
  title: string
  content: string
  created_by: string
  session_id: string
  status: string
  created_at: string
  updated_at: string
}

interface ActivityEvent {
  id: number
  project_id: string
  event_type: string
  agent_id: string
  detail: Record<string, unknown>
  created_at: string
}

// ── API helpers ──

const api = (path: string, init?: RequestInit) =>
  fetch(`/fd/projects${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  }).then(async (r) => {
    if (!r.ok) throw new Error(await r.text().catch(() => r.statusText))
    return r.json()
  })

// ── Main page ──

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [selected, setSelected] = useState<Project | null>(null)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const { setView, setForgeProjectId } = useUIStore()

  const fetchProjects = useCallback(async () => {
    try {
      const data = await api('')
      setProjects(data)
    } catch (e) {
      console.error('Failed to fetch projects', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchProjects() }, [fetchProjects])

  const createProject = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      const project = await api('', {
        method: 'POST',
        body: JSON.stringify({ name: newName.trim(), description: newDesc.trim() }),
      })
      setProjects((p) => [project, ...p])
      setSelected(project)
      setNewName('')
      setNewDesc('')
    } catch (e) {
      alert(`Failed to create: ${e}`)
    } finally {
      setCreating(false)
    }
  }

  const deleteProject = async (id: string) => {
    if (!confirm('Delete this project and all its data?')) return
    try {
      await fetch(`/fd/projects/${id}`, { method: 'DELETE' })
      setProjects((p) => p.filter((x) => x.id !== id))
      if (selected?.id === id) setSelected(null)
    } catch (e) {
      alert(`Failed to delete: ${e}`)
    }
  }

  const updateStatus = async (id: string, status: string) => {
    try {
      const updated = await api(`/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ status }),
      })
      setProjects((p) => p.map((x) => (x.id === id ? updated : x)))
      if (selected?.id === id) setSelected(updated)
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const statusIcon = (s: string) => {
    if (s === 'active') return <CheckCircle2 size={14} className="text-emerald-400" />
    if (s === 'paused') return <Pause size={14} className="text-amber-400" />
    if (s === 'completed') return <CheckCircle2 size={14} className="text-blue-400" />
    if (s === 'archived') return <Archive size={14} className="text-zinc-500" />
    return <Clock size={14} className="text-zinc-500" />
  }

  return (
    <div className="flex h-full min-h-0">
      {/* Sidebar: project list */}
      <div className="flex w-64 flex-shrink-0 flex-col border-r border-zinc-800 bg-zinc-950">
        <div className="border-b border-zinc-800 p-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
            <FolderKanban size={16} />
            Projects
          </div>
        </div>

        {/* Create form */}
        <div className="border-b border-zinc-800 p-3 space-y-2">
          <input
            className="w-full rounded bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 border border-zinc-700 focus:border-blue-500 focus:outline-none"
            placeholder="Project name..."
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && createProject()}
          />
          <input
            className="w-full rounded bg-zinc-900 px-2 py-1.5 text-xs text-zinc-400 border border-zinc-700 focus:border-blue-500 focus:outline-none"
            placeholder="Description (optional)"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
          />
          <button
            onClick={createProject}
            disabled={!newName.trim() || creating}
            className="flex w-full items-center justify-center gap-1.5 rounded bg-blue-600 px-2 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {creating ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
            Create Project
          </button>
        </div>

        {/* Project list */}
        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center p-8">
              <Loader2 size={20} className="animate-spin text-zinc-500" />
            </div>
          )}
          {projects.map((p) => (
            <div
              key={p.id}
              onClick={() => setSelected(p)}
              className={`flex cursor-pointer items-center gap-2 border-b border-zinc-800/50 px-3 py-2.5 text-xs transition-colors ${
                selected?.id === p.id
                  ? 'bg-zinc-800/80 text-zinc-100'
                  : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
              }`}
            >
              {statusIcon(p.status)}
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{p.name}</div>
                {p.description && (
                  <div className="truncate text-[10px] text-zinc-500">{p.description}</div>
                )}
              </div>
            </div>
          ))}
          {!loading && projects.length === 0 && (
            <div className="p-4 text-center text-xs text-zinc-600">
              No projects yet. Create one above.
            </div>
          )}
        </div>
      </div>

      {/* Main: project detail */}
      <div className="flex-1 overflow-y-auto bg-zinc-950 p-6">
        {selected ? (
          <ProjectDetail
            project={selected}
            onUpdate={(p) => {
              setSelected(p)
              setProjects((list) => list.map((x) => (x.id === p.id ? p : x)))
            }}
            onDelete={() => deleteProject(selected.id)}
            onStatusChange={(s) => updateStatus(selected.id, s)}
            onForge={() => {
              setForgeProjectId(selected.id)
              setView('forge')
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-zinc-600">
              <FolderKanban size={48} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">Select a project or create a new one</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Project detail view ──

interface FleetAgent {
  name: string
  kind: string
  host: string
  port: number
  status: string
  description: string
}

function ProjectDetail({
  project,
  onUpdate,
  onDelete,
  onStatusChange,
  onForge,
}: {
  project: Project
  onUpdate: (p: Project) => void
  onDelete: () => void
  onStatusChange: (s: string) => void
  onForge: () => void
}) {
  const [members, setMembers] = useState<Member[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [activity, setActivity] = useState<ActivityEvent[]>([])
  const [agentActivity, setAgentActivity] = useState<{ agent_name: string; role: string; content: string; timestamp: string; tool_name?: string }[]>([])
  const [agentActivityLoading, setAgentActivityLoading] = useState(false)
  const [tab, setTab] = useState<'goals' | 'team' | 'artifacts' | 'activity' | 'agents'>('goals')

  // New goal form
  const [newGoal, setNewGoal] = useState('')
  const [addingGoal, setAddingGoal] = useState(false)

  // Add member
  const [fleet, setFleet] = useState<FleetAgent[]>([])
  const [showAddMember, setShowAddMember] = useState(false)
  const [addingMember, setAddingMember] = useState(false)

  // Goal checking
  const [checking, setChecking] = useState(false)
  const [goalCheck, setGoalCheck] = useState<{
    goals: { goal: string; status: string; progress_pct: number; evidence: string; remaining: string; agents_involved: string[] }[];
    summary: string;
  } | null>(null)

  // Dispatch: two-step (plan → review → execute) — hidden but kept
  const [_planning, setPlanning] = useState(false)
  const [_executing, setExecuting] = useState(false)
  const [dispatchPlan, setDispatchPlan] = useState<{
    plan: { assignments: { agent_name: string; goals: string[]; subtasks: string[]; rationale: string; priority_order?: string }[]; coordination_notes: string };
    goals_count: number; matched_agents: string[]; unmatched_agents: string[];
  } | null>(null)
  const [_dispatchResult, setDispatchResult] = useState<{
    dispatched: number; results: { agent: string; status: string; error?: string }[];
  } | null>(null)

  // New artifact form
  const [artKind, setArtKind] = useState('note')
  const [artTitle, setArtTitle] = useState('')
  const [artContent, setArtContent] = useState('')
  const [addingArt, setAddingArt] = useState(false)

  useEffect(() => {
    api(`/${project.id}/members`).then(setMembers).catch(() => {})
    api(`/${project.id}/artifacts`).then(setArtifacts).catch(() => {})
    api(`/${project.id}/activity`).then(setActivity).catch(() => {})
  }, [project.id])

  const fetchFleet = () => {
    fetch('/fd/fleet').then((r) => r.json()).then(setFleet).catch(() => {})
  }

  useEffect(() => {
    if (showAddMember) fetchFleet()
  }, [showAddMember])

  const fetchAgentActivity = () => {
    setAgentActivityLoading(true)
    api(`/${project.id}/agent-activity?limit=100`)
      .then(setAgentActivity)
      .catch(() => {})
      .finally(() => setAgentActivityLoading(false))
  }

  useEffect(() => {
    if (tab === 'agents') fetchAgentActivity()
  }, [tab, project.id])

  const memberAgentIds = new Set(members.filter((m) => !m.left_at).map((m) => m.agent_name || m.agent_id))
  const availableAgents = fleet.filter((a) => a.status === 'running' && !memberAgentIds.has(a.name))

  const addMember = async (agent: FleetAgent) => {
    setAddingMember(true)
    try {
      await api(`/${project.id}/members`, {
        method: 'POST',
        body: JSON.stringify({
          agent_id: `${agent.kind}-${agent.name}`,
          agent_name: agent.name,
          role: 'contributor',
        }),
      })
      const updated = await api(`/${project.id}/members`)
      setMembers(updated)
      setShowAddMember(false)
    } catch (e) {
      alert(`Failed to add: ${e}`)
    } finally {
      setAddingMember(false)
    }
  }

  const removeMember = async (m: Member) => {
    try {
      await fetch(`/fd/projects/${project.id}/members/${m.agent_id}`, { method: 'DELETE' })
      const updated = await api(`/${project.id}/members`)
      setMembers(updated)
    } catch (e) {
      alert(`Failed to remove: ${e}`)
    }
  }

  // @ts-expect-error — kept for upcoming dispatch UI
  const _generatePlan = async () => {
    setPlanning(true)
    setDispatchPlan(null)
    setDispatchResult(null)
    try {
      let llmConfig = { provider: 'anthropic', model: 'claude-sonnet-4-20250514', api_key: '' }
      try {
        const saved = JSON.parse(localStorage.getItem('fd:forge-llm-config') || '{}')
        if (saved.provider) llmConfig = saved
      } catch { /* use defaults */ }

      const result = await api(`/${project.id}/dispatch/plan`, {
        method: 'POST',
        body: JSON.stringify({
          provider: llmConfig.provider,
          model: llmConfig.model,
          api_key: llmConfig.api_key,
        }),
      })
      setDispatchPlan(result)
    } catch (e) {
      alert(`Planning failed: ${e}`)
    } finally {
      setPlanning(false)
    }
  }

  // @ts-expect-error — kept for upcoming dispatch UI
  const _executePlan = async () => {
    if (!dispatchPlan) return
    setExecuting(true)
    try {
      const result = await api(`/${project.id}/dispatch/execute`, {
        method: 'POST',
        body: JSON.stringify({ plan: dispatchPlan.plan }),
      })
      setDispatchResult(result)
      setDispatchPlan(null)
      const updated = await api(`/${project.id}`)
      onUpdate(updated)
      api(`/${project.id}/activity`).then(setActivity).catch(() => {})
    } catch (e) {
      alert(`Dispatch failed: ${e}`)
    } finally {
      setExecuting(false)
    }
  }

  const checkGoals = async () => {
    setChecking(true)
    setGoalCheck(null)
    try {
      let llmConfig = { provider: 'anthropic', model: 'claude-sonnet-4-20250514', api_key: '' }
      try {
        const saved = JSON.parse(localStorage.getItem('fd:forge-llm-config') || '{}')
        if (saved.provider) llmConfig = saved
      } catch { /* use defaults */ }

      const result = await api(`/${project.id}/goals/check`, {
        method: 'POST',
        body: JSON.stringify({
          provider: llmConfig.provider,
          model: llmConfig.model,
          api_key: llmConfig.api_key,
        }),
      })
      setGoalCheck(result)
    } catch (e) {
      alert(`Goal check failed: ${e}`)
    } finally {
      setChecking(false)
    }
  }

  const addGoal = async () => {
    if (!newGoal.trim()) return
    setAddingGoal(true)
    try {
      await api(`/${project.id}/goals`, {
        method: 'POST',
        body: JSON.stringify({ goal: newGoal.trim() }),
      })
      const updated = await api(`/${project.id}`)
      onUpdate(updated)
      setNewGoal('')
    } catch (e) {
      alert(`Failed: ${e}`)
    } finally {
      setAddingGoal(false)
    }
  }

  const updateGoalStatus = async (idx: number, status: string) => {
    try {
      await api(`/${project.id}/goals`, {
        method: 'PUT',
        body: JSON.stringify({ goal_index: idx, status }),
      })
      const updated = await api(`/${project.id}`)
      onUpdate(updated)
    } catch (e) {
      alert(`Failed: ${e}`)
    }
  }

  const addArtifact = async () => {
    if (!artTitle.trim()) return
    setAddingArt(true)
    try {
      const art = await api(`/${project.id}/artifacts`, {
        method: 'POST',
        body: JSON.stringify({ kind: artKind, title: artTitle.trim(), content: artContent }),
      })
      setArtifacts((a) => [art, ...a])
      setArtTitle('')
      setArtContent('')
    } catch (e) {
      alert(`Failed: ${e}`)
    } finally {
      setAddingArt(false)
    }
  }

  const goalStatusIcon = (s: string) => {
    if (s === 'done') return <CheckCircle2 size={14} className="text-emerald-400" />
    if (s === 'in_progress') return <ArrowRight size={14} className="text-blue-400" />
    if (s === 'blocked') return <AlertTriangle size={14} className="text-red-400" />
    return <Clock size={14} className="text-zinc-500" />
  }

  const kindColor = (k: string) => {
    if (k === 'decision') return 'text-blue-400 bg-blue-400/10'
    if (k === 'milestone') return 'text-emerald-400 bg-emerald-400/10'
    if (k === 'blocker') return 'text-red-400 bg-red-400/10'
    if (k === 'deliverable') return 'text-amber-400 bg-amber-400/10'
    return 'text-zinc-400 bg-zinc-400/10'
  }

  return (
    <div className="max-w-4xl space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-zinc-100">{project.name}</h1>
          {project.description && (
            <p className="mt-1 text-sm text-zinc-400">{project.description}</p>
          )}
          <div className="mt-2 flex items-center gap-3 text-xs text-zinc-500">
            <span>Created {new Date(project.created_at).toLocaleDateString()}</span>
            <span>ID: {project.id}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {project.status === 'active' && members.length > 0 && (
            <button
              onClick={checkGoals}
              disabled={checking}
              className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-50"
              title="Analyze agent activity to check goal progress"
            >
              {checking ? <Loader2 size={12} className="animate-spin" /> : <Target size={12} />}
              {checking ? 'Checking...' : 'Check Goals'}
            </button>
          )}
          <select
            value={project.status}
            onChange={(e) => onStatusChange(e.target.value)}
            className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300"
          >
            <option value="active">Active</option>
            <option value="paused">Paused</option>
            <option value="completed">Completed</option>
            <option value="archived">Archived</option>
          </select>
          <button
            onClick={onDelete}
            className="rounded p-1.5 text-zinc-500 hover:bg-red-900/30 hover:text-red-400"
            title="Delete project"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Goal check results */}
      {goalCheck && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-500/5 px-4 py-3 text-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-blue-300 font-medium">Goal Progress Report</span>
            <button onClick={() => setGoalCheck(null)} className="text-zinc-500 hover:text-zinc-300 text-xs">dismiss</button>
          </div>

          {goalCheck.summary && (
            <div className="text-xs text-zinc-300 bg-zinc-900/50 rounded p-2">{goalCheck.summary}</div>
          )}

          {goalCheck.goals.map((g, i) => {
            const barColor =
              g.status === 'done' ? 'bg-emerald-500'
              : g.status === 'blocked' ? 'bg-red-500'
              : g.progress_pct > 0 ? 'bg-blue-500'
              : 'bg-zinc-700'
            const statusLabel =
              g.status === 'done' ? 'Done'
              : g.status === 'blocked' ? 'Blocked'
              : g.status === 'in_progress' ? 'In Progress'
              : 'Not Started'
            const statusColor =
              g.status === 'done' ? 'text-emerald-400'
              : g.status === 'blocked' ? 'text-red-400'
              : g.status === 'in_progress' ? 'text-blue-400'
              : 'text-zinc-500'

            return (
              <div key={i} className="rounded border border-zinc-800 bg-zinc-900/80 p-3 space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs text-zinc-200 flex-1">{g.goal}</span>
                  <span className={`text-[10px] font-medium shrink-0 ${statusColor}`}>{statusLabel}</span>
                </div>

                {/* Progress bar */}
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${g.progress_pct}%` }} />
                  </div>
                  <span className="text-[10px] text-zinc-500 w-8 text-right">{g.progress_pct}%</span>
                </div>

                {g.evidence && (
                  <div className="text-[11px] text-zinc-400">
                    <span className="text-zinc-500 font-medium">Done: </span>{g.evidence}
                  </div>
                )}
                {g.remaining && (
                  <div className="text-[11px] text-zinc-400">
                    <span className="text-zinc-500 font-medium">Remaining: </span>{g.remaining}
                  </div>
                )}
                {g.agents_involved?.length > 0 && (
                  <div className="flex gap-1 mt-1">
                    {g.agents_involved.map((a, j) => (
                      <span key={j} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{a}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-zinc-800">
        {([
          { id: 'goals' as const, icon: Target, label: 'Goals', count: project.goals?.length || 0 },
          { id: 'team' as const, icon: Users, label: 'Team', count: members.length },
          { id: 'artifacts' as const, icon: FileText, label: 'Artifacts', count: artifacts.length },
          { id: 'agents' as const, icon: Activity, label: 'Agent Activity' },
          { id: 'activity' as const, icon: Clock, label: 'Project Log' },
        ]).map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-3 py-2 text-xs font-medium transition-colors ${
              tab === t.id
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <t.icon size={12} className="mr-1 inline" />
            {t.label}
            {'count' in t ? ` (${t.count})` : ''}
          </button>
        ))}
      </div>

      {/* Goals tab */}
      {tab === 'goals' && (
        <div className="space-y-3">
          {(project.goals || []).map((g, i) => (
            <div
              key={i}
              className="flex items-start gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3"
            >
              <button
                onClick={() =>
                  updateGoalStatus(
                    i,
                    g.status === 'done'
                      ? 'pending'
                      : g.status === 'in_progress'
                      ? 'done'
                      : 'in_progress',
                  )
                }
                className="mt-0.5"
              >
                {goalStatusIcon(g.status)}
              </button>
              <div className="flex-1 min-w-0">
                <div className={`text-sm ${g.status === 'done' ? 'text-zinc-500 line-through' : 'text-zinc-200'}`}>
                  {g.goal}
                </div>
                {g.success_criteria && (
                  <div className="mt-1 text-xs text-zinc-500">Criteria: {g.success_criteria}</div>
                )}
              </div>
              <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
                {g.priority}
              </span>
            </div>
          ))}
          <div className="flex gap-2">
            <input
              className="flex-1 rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
              placeholder="Add a goal..."
              value={newGoal}
              onChange={(e) => setNewGoal(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addGoal()}
            />
            <button
              onClick={addGoal}
              disabled={!newGoal.trim() || addingGoal}
              className="rounded bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
            >
              {addingGoal ? <Loader2 size={12} className="animate-spin" /> : 'Add'}
            </button>
          </div>
        </div>
      )}

      {/* Team tab */}
      {tab === 'team' && (
        <div className="space-y-3">
          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setShowAddMember(!showAddMember)}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
            >
              <UserPlus size={12} /> Add Agent
            </button>
            <button
              onClick={onForge}
              className="flex items-center gap-1.5 rounded-lg border border-violet-600/50 bg-violet-600/10 px-3 py-1.5 text-xs text-violet-300 hover:bg-violet-600/20"
            >
              <Wand2 size={12} /> Forge Team
            </button>
          </div>

          {/* Add member dropdown */}
          {showAddMember && (
            <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-3 space-y-2">
              <div className="text-xs font-medium text-zinc-400">Select a running agent to add:</div>
              {availableAgents.length === 0 ? (
                <p className="text-xs text-zinc-600">No available agents found. Start agents first or check the fleet.</p>
              ) : (
                <div className="space-y-1">
                  {availableAgents.map((a) => (
                    <button
                      key={`${a.kind}-${a.name}`}
                      onClick={() => addMember(a)}
                      disabled={addingMember}
                      className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                    >
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">{a.kind}</span>
                      <span className="flex-1">{a.name}</span>
                      {a.description && <span className="text-[10px] text-zinc-600 truncate max-w-48">{a.description}</span>}
                      <Plus size={12} className="text-zinc-500" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {members.length === 0 && !showAddMember && (
            <p className="text-xs text-zinc-500">
              No members yet. Add agents above or forge a new team.
            </p>
          )}
          {members.map((m) => (
            <div
              key={m.id}
              className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-3"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-zinc-800 text-xs font-bold text-zinc-400">
                {(m.agent_name || m.agent_id).charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-zinc-200">
                  {m.agent_name || m.agent_id}
                </div>
                <div className="text-xs text-zinc-500">
                  {m.role}
                  {m.expertise_tags.length > 0 && ` — ${m.expertise_tags.join(', ')}`}
                </div>
              </div>
              <div className="text-[10px] text-zinc-600">
                Joined {new Date(m.joined_at).toLocaleDateString()}
              </div>
              {m.left_at ? (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
                  Left
                </span>
              ) : (
                <button
                  onClick={() => removeMember(m)}
                  className="rounded p-1 text-zinc-600 hover:bg-red-900/30 hover:text-red-400"
                  title="Remove from project"
                >
                  <UserMinus size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Artifacts tab */}
      {tab === 'artifacts' && (
        <div className="space-y-3">
          {/* Add form */}
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 space-y-2">
            <div className="flex gap-2">
              <select
                value={artKind}
                onChange={(e) => setArtKind(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300"
              >
                <option value="note">Note</option>
                <option value="decision">Decision</option>
                <option value="milestone">Milestone</option>
                <option value="deliverable">Deliverable</option>
                <option value="blocker">Blocker</option>
              </select>
              <input
                className="flex-1 rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
                placeholder="Title..."
                value={artTitle}
                onChange={(e) => setArtTitle(e.target.value)}
              />
              <button
                onClick={addArtifact}
                disabled={!artTitle.trim() || addingArt}
                className="rounded bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-700 disabled:opacity-50"
              >
                {addingArt ? <Loader2 size={12} className="animate-spin" /> : 'Add'}
              </button>
            </div>
            <textarea
              className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-300 focus:border-blue-500 focus:outline-none"
              placeholder="Content (optional)..."
              rows={2}
              value={artContent}
              onChange={(e) => setArtContent(e.target.value)}
            />
          </div>

          {/* List */}
          {artifacts.map((a) => (
            <div
              key={a.id}
              className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3"
            >
              <div className="flex items-center gap-2">
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${kindColor(a.kind)}`}>
                  {a.kind}
                </span>
                <span className="text-sm font-medium text-zinc-200">{a.title}</span>
                {a.status !== 'active' && (
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
                    {a.status}
                  </span>
                )}
                <span className="ml-auto text-[10px] text-zinc-600">
                  {new Date(a.created_at).toLocaleDateString()}
                </span>
              </div>
              {a.content && (
                <p className="mt-2 text-xs text-zinc-400 whitespace-pre-wrap">{a.content}</p>
              )}
              {a.created_by && (
                <div className="mt-1 text-[10px] text-zinc-600">by {a.created_by}</div>
              )}
            </div>
          ))}
          {artifacts.length === 0 && (
            <p className="text-xs text-zinc-500">No artifacts yet.</p>
          )}
        </div>
      )}

      {/* Agent Activity tab */}
      {tab === 'agents' && (
        <div className="space-y-1">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-zinc-500">Live activity from project agents</span>
            <button
              onClick={fetchAgentActivity}
              disabled={agentActivityLoading}
              className="flex items-center gap-1 rounded border border-zinc-700 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-50"
            >
              {agentActivityLoading ? <Loader2 size={10} className="animate-spin" /> : <ArrowRight size={10} />}
              Refresh
            </button>
          </div>
          {agentActivity.length === 0 && !agentActivityLoading && (
            <p className="text-xs text-zinc-500">No agent activity yet. Dispatch goals first, then refresh.</p>
          )}
          {agentActivityLoading && agentActivity.length === 0 && (
            <div className="flex items-center gap-2 py-4 text-xs text-zinc-500">
              <Loader2 size={14} className="animate-spin" /> Loading agent activity...
            </div>
          )}
          {agentActivity.map((evt, i) => (
            <div
              key={i}
              className={`rounded-lg border p-2.5 text-xs ${
                evt.role === 'assistant'
                  ? 'border-zinc-800 bg-zinc-900/50'
                  : evt.role === 'tool'
                  ? 'border-zinc-800/50 bg-zinc-950'
                  : 'border-zinc-800/30 bg-transparent'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 text-[9px] font-bold text-zinc-400">
                  {evt.agent_name.charAt(0).toUpperCase()}
                </span>
                <span className="font-medium text-zinc-300">{evt.agent_name}</span>
                {evt.role === 'tool' && evt.tool_name && (
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-amber-400">{evt.tool_name}</span>
                )}
                {evt.role === 'user' && (
                  <span className="rounded bg-violet-500/10 px-1.5 py-0.5 text-[10px] text-violet-300">task</span>
                )}
                {evt.role === 'assistant' && (
                  <span className="rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-300">response</span>
                )}
                {evt.timestamp && (
                  <span className="ml-auto text-[10px] text-zinc-600">
                    {new Date(evt.timestamp).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                )}
              </div>
              <div className="text-zinc-400 whitespace-pre-wrap break-words leading-relaxed pl-7">
                {evt.content}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Activity tab */}
      {tab === 'activity' && (
        <div className="space-y-1">
          {activity.length === 0 && (
            <p className="text-xs text-zinc-500">No activity yet.</p>
          )}
          {activity.map((a) => {
            const detail = a.detail || {}
            const detailStr = Object.entries(detail)
              .filter(([, v]) => v)
              .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
              .join(', ')
            return (
              <div
                key={a.id}
                className="flex items-start gap-3 rounded py-1.5 px-2 text-xs hover:bg-zinc-900/50"
              >
                <span className="w-32 flex-shrink-0 text-zinc-600">
                  {new Date(a.created_at).toLocaleString(undefined, {
                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                  })}
                </span>
                <span className="font-medium text-zinc-300">{a.event_type.replace(/_/g, ' ')}</span>
                {a.agent_id && (
                  <span className="text-zinc-500">by {a.agent_id.slice(0, 8)}</span>
                )}
                {detailStr && (
                  <span className="truncate text-zinc-600">{detailStr}</span>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
