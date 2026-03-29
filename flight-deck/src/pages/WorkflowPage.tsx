import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  BackgroundVariant,
} from '@xyflow/react'
import {
  Play,
  Pause,
  Square,
  RefreshCw,
  Plus,
  Zap,
  GitBranch,
  Trash2,
  Power,
  ArrowDown,
  Edit3,
  Box,
  Settings2 as Cog,
  X,
  ChevronDown,
  ChevronRight,
  Workflow,
} from 'lucide-react'
import { useWorkflowStore } from '../stores/workflowStore'
import { usePipelineStore, type Pipeline, type PipelineStep } from '../stores/pipelineStore'
import { useContainerStore } from '../stores/containerStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { TaskNode } from '../components/workflow/TaskNode'
import { StatusBadge } from '../components/common/StatusBadge'

const nodeTypes = { task: TaskNode }

type Tab = 'workflows' | 'pipelines'

export function WorkflowPage() {
  const [tab, setTab] = useState<Tab>('pipelines')

  return (
    <div className="flex h-full flex-col">
      {/* Tab bar */}
      <div className="flex items-center gap-1 border-b border-zinc-800 bg-zinc-900/30 px-4 py-1.5">
        <TabButton active={tab === 'pipelines'} onClick={() => setTab('pipelines')} icon={GitBranch} label="Pipelines" />
        <TabButton active={tab === 'workflows'} onClick={() => setTab('workflows')} icon={Workflow} label="Swarm Workflows" />
      </div>

      {tab === 'workflows' ? <WorkflowsTab /> : <PipelinesTab />}
    </div>
  )
}

function TabButton({ active, onClick, icon: Icon, label }: { active: boolean; onClick: () => void; icon: typeof Play; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
        active ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
      }`}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}

// ═══════════════════════════════════════════════════════════
//  PIPELINES TAB
// ═══════════════════════════════════════════════════════════

function PipelinesTab() {
  const { pipelines, createPipeline, deletePipeline, updatePipeline, addStep, removeStep, updateStep } = usePipelineStore()
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const [newName, setNewName] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const allAgents = [
    ...containers.filter((c) => c.web_port).map((c) => ({
      id: c.id, name: c.agent_name || c.name, kind: 'docker' as const,
    })),
    ...localAgents.map((a) => ({ id: a.id, name: a.name, kind: 'local' as const })),
  ]

  const agentName = (id: string) => allAgents.find((a) => a.id === id)?.name || id.slice(0, 8)
  const agentKind = (id: string) => allAgents.find((a) => a.id === id)?.kind || 'docker'

  const handleCreate = () => {
    if (!newName.trim()) return
    const id = createPipeline(newName.trim())
    setNewName('')
    setExpandedId(id)
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-6">
        {/* Header */}
        <div className="mb-6">
          <h2 className="flex items-center gap-2 text-lg font-semibold text-zinc-100">
            <GitBranch className="h-5 w-5 text-emerald-400" />
            Agent Pipelines
          </h2>
          <p className="mt-1 text-sm text-zinc-500">
            Chain agents together — when one responds, its output automatically flows to the next.
          </p>
        </div>

        {/* Create new pipeline */}
        <div className="mb-6 flex gap-2">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            placeholder="New pipeline name..."
            className="flex-1 rounded-xl border border-zinc-700/50 bg-zinc-900/50 px-4 py-2.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/20"
          />
          <button
            onClick={handleCreate}
            disabled={!newName.trim()}
            className="flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
          >
            <Plus className="h-4 w-4" />
            Create
          </button>
        </div>

        {/* Pipeline cards */}
        {pipelines.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-zinc-800 px-8 py-16 text-center">
            <GitBranch className="mx-auto h-10 w-10 text-zinc-700" />
            <p className="mt-3 text-sm text-zinc-500">No pipelines yet</p>
            <p className="mt-1 text-xs text-zinc-600">Create one to chain agent outputs automatically.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {pipelines.map((pipeline) => (
              <PipelineCard
                key={pipeline.id}
                pipeline={pipeline}
                expanded={expandedId === pipeline.id}
                onToggle={() => setExpandedId(expandedId === pipeline.id ? null : pipeline.id)}
                onDelete={() => deletePipeline(pipeline.id)}
                onToggleEnabled={() => updatePipeline(pipeline.id, { enabled: !pipeline.enabled })}
                onRename={(name) => updatePipeline(pipeline.id, { name })}
                onAddStep={(step) => addStep(pipeline.id, step)}
                onRemoveStep={(idx) => removeStep(pipeline.id, idx)}
                onUpdateStep={(idx, patch) => updateStep(pipeline.id, idx, patch)}
                allAgents={allAgents}
                agentName={agentName}
                agentKind={agentKind}
              />
            ))}
          </div>
        )}

        {/* Info footer */}
        <div className="mt-6 rounded-xl bg-zinc-900/40 border border-zinc-800/50 px-4 py-3">
          <p className="text-xs text-zinc-500">
            <span className="font-medium text-zinc-400">How it works:</span> When an agent at step N responds, its output is automatically forwarded to the agent at step N+1
            with context about which pipeline and agent it came from.
          </p>
        </div>
      </div>
    </div>
  )
}

function PipelineCard({
  pipeline, expanded, onToggle, onDelete, onToggleEnabled, onRename,
  onAddStep, onRemoveStep, onUpdateStep, allAgents, agentName, agentKind,
}: {
  pipeline: Pipeline
  expanded: boolean
  onToggle: () => void
  onDelete: () => void
  onToggleEnabled: () => void
  onRename: (name: string) => void
  onAddStep: (step: PipelineStep) => void
  onRemoveStep: (idx: number) => void
  onUpdateStep: (idx: number, patch: Partial<PipelineStep>) => void
  allAgents: { id: string; name: string; kind: 'docker' | 'local' }[]
  agentName: (id: string) => string
  agentKind: (id: string) => string
}) {
  const [renaming, setRenaming] = useState(false)
  const [nameInput, setNameInput] = useState(pipeline.name)
  const [addingStep, setAddingStep] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState<number | null>(null)
  const [promptInput, setPromptInput] = useState('')

  const handleRename = () => {
    if (nameInput.trim()) onRename(nameInput.trim())
    setRenaming(false)
  }

  return (
    <div className={`rounded-2xl border transition-all ${
      expanded
        ? 'border-violet-500/30 bg-zinc-900/50 shadow-lg shadow-violet-500/5'
        : pipeline.enabled
          ? 'border-zinc-800 bg-zinc-900/30 hover:border-zinc-700'
          : 'border-zinc-800/50 bg-zinc-950/30 opacity-60 hover:opacity-80'
    }`}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 cursor-pointer" onClick={onToggle}>
        <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${
          pipeline.enabled ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-600'
        }`}>
          <GitBranch className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1">
          {renaming ? (
            <input
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenaming(false) }}
              onBlur={handleRename}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              onClick={(e) => e.stopPropagation()}
              autoFocus
            />
          ) : (
            <div className="text-sm font-medium text-zinc-200">{pipeline.name}</div>
          )}

          {/* Mini flow preview */}
          {pipeline.steps.length > 0 && !expanded && (
            <div className="mt-1 flex items-center gap-1 text-[11px] text-zinc-500">
              {pipeline.steps.map((s, i) => (
                <span key={i} className="flex items-center gap-1">
                  {i > 0 && <span className="text-zinc-700">→</span>}
                  <span className={`rounded px-1.5 py-0.5 ${
                    agentKind(s.agentId) === 'docker'
                      ? 'bg-blue-500/10 text-blue-400'
                      : 'bg-amber-500/10 text-amber-400'
                  }`}>
                    {agentName(s.agentId)}
                  </span>
                </span>
              ))}
            </div>
          )}
          {pipeline.steps.length === 0 && !expanded && (
            <div className="mt-0.5 text-[11px] text-zinc-600">No steps — click to configure</div>
          )}
        </div>

        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
            pipeline.enabled ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-500'
          }`}>
            {pipeline.enabled ? 'Active' : 'Disabled'}
          </span>
          <button
            onClick={onToggleEnabled}
            className={`rounded-lg p-1.5 transition-colors ${
              pipeline.enabled ? 'text-emerald-400 hover:bg-emerald-500/10' : 'text-zinc-600 hover:bg-zinc-800'
            }`}
            title={pipeline.enabled ? 'Disable' : 'Enable'}
          >
            <Power className="h-4 w-4" />
          </button>
          <button
            onClick={() => { setRenaming(true); setNameInput(pipeline.name) }}
            className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
          >
            <Edit3 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onDelete}
            className="rounded-lg p-1.5 text-zinc-500 hover:bg-red-500/10 hover:text-red-400 transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
          {expanded ? <ChevronDown className="h-4 w-4 text-zinc-600" /> : <ChevronRight className="h-4 w-4 text-zinc-600" />}
        </div>
      </div>

      {/* Expanded: step editor */}
      {expanded && (
        <div className="border-t border-zinc-800/50 px-4 py-4">
          {/* Visual flow */}
          <div className="flex flex-col items-center gap-0">
            {pipeline.steps.map((step, i) => (
              <div key={i} className="flex flex-col items-center w-full max-w-md mx-auto">
                {/* Step card */}
                <div className="w-full rounded-xl border border-zinc-700/50 bg-zinc-950/60 p-3 relative group">
                  <div className="flex items-center gap-3">
                    {/* Step number badge */}
                    <div className={`flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-bold ${
                      agentKind(step.agentId) === 'docker'
                        ? 'bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/20'
                        : 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20'
                    }`}>
                      {i + 1}
                    </div>

                    {/* Agent info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        {agentKind(step.agentId) === 'docker'
                          ? <Box className="h-3.5 w-3.5 text-blue-400/70" />
                          : <Cog className="h-3.5 w-3.5 text-amber-400/70" />}
                        <span className="text-sm font-medium text-zinc-200">{agentName(step.agentId)}</span>
                        <span className={`text-[10px] rounded px-1 py-0.5 ${
                          agentKind(step.agentId) === 'docker'
                            ? 'bg-blue-500/10 text-blue-400/70'
                            : 'bg-amber-500/10 text-amber-400/70'
                        }`}>
                          {agentKind(step.agentId)}
                        </span>
                      </div>
                      {step.prompt && (
                        <p className="mt-1 text-[11px] text-violet-400/80 truncate" title={step.prompt}>
                          Prompt: {step.prompt}
                        </p>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => { setEditingPrompt(editingPrompt === i ? null : i); setPromptInput(step.prompt || '') }}
                        className="rounded-lg p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                        title="Edit prompt prefix"
                      >
                        <Edit3 className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => onRemoveStep(i)}
                        className="rounded-lg p-1 text-zinc-600 hover:bg-red-500/10 hover:text-red-400"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  </div>

                  {/* Prompt editor inline */}
                  {editingPrompt === i && (
                    <div className="mt-3 border-t border-zinc-800/50 pt-3">
                      <label className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Prompt Prefix</label>
                      <textarea
                        value={promptInput}
                        onChange={(e) => setPromptInput(e.target.value)}
                        placeholder="Optional: prepend this prompt before forwarded output..."
                        className="mt-1 w-full resize-none rounded-lg border border-zinc-700/50 bg-zinc-900/50 px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                        rows={3}
                      />
                      <div className="mt-2 flex gap-2">
                        <button
                          onClick={() => { onUpdateStep(i, { prompt: promptInput.trim() || undefined }); setEditingPrompt(null) }}
                          className="rounded-lg bg-violet-600 px-3 py-1 text-xs text-white hover:bg-violet-500"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => setEditingPrompt(null)}
                          className="rounded-lg px-3 py-1 text-xs text-zinc-500 hover:text-zinc-300"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Arrow connector */}
                {i < pipeline.steps.length - 1 && (
                  <div className="flex flex-col items-center py-1">
                    <div className="h-4 w-px bg-gradient-to-b from-violet-500/40 to-violet-500/10" />
                    <ArrowDown className="h-3.5 w-3.5 text-violet-500/40 -mt-0.5" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Add step */}
          <div className="mt-4 flex flex-col items-center">
            {pipeline.steps.length > 0 && (
              <div className="flex flex-col items-center pb-2">
                <div className="h-3 w-px bg-zinc-700/40" />
              </div>
            )}

            {addingStep ? (
              <div className="w-full max-w-md rounded-xl border border-dashed border-violet-500/30 bg-violet-500/5 p-3">
                <p className="mb-2 text-xs font-medium text-violet-400">Select an agent to add:</p>
                <div className="flex flex-wrap gap-1.5">
                  {allAgents.map((a) => (
                    <button
                      key={a.id}
                      onClick={() => { onAddStep({ agentId: a.id }); setAddingStep(false) }}
                      className="flex items-center gap-1.5 rounded-lg border border-zinc-700/50 bg-zinc-900/50 px-2.5 py-1.5 text-xs text-zinc-300 hover:border-violet-500/30 hover:bg-violet-500/10 hover:text-violet-400 transition-colors"
                    >
                      {a.kind === 'docker'
                        ? <Box className="h-3 w-3 text-blue-400/70" />
                        : <Cog className="h-3 w-3 text-amber-400/70" />}
                      {a.name}
                    </button>
                  ))}
                </div>
                <button onClick={() => setAddingStep(false)} className="mt-2 text-[10px] text-zinc-500 hover:text-zinc-300">
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setAddingStep(true)}
                className="flex items-center gap-1.5 rounded-xl border border-dashed border-zinc-700 px-4 py-2 text-xs text-zinc-500 hover:border-violet-500/30 hover:bg-violet-500/5 hover:text-violet-400 transition-colors"
              >
                <Plus className="h-3.5 w-3.5" />
                Add Step
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════
//  WORKFLOWS TAB (existing swarm workflows)
// ═══════════════════════════════════════════════════════════

function WorkflowsTab() {
  const {
    swarms,
    activeSwarmId,
    tasks,
    edges: swarmEdges,
    fetchSwarms,
    selectSwarm,
    startSwarm,
    pauseSwarm,
    cancelSwarm,
    decomposeSwarm,
    createEdge,
    createTask,
  } = useWorkflowStore()

  useEffect(() => { fetchSwarms() }, [fetchSwarms])

  const activeSwarm = swarms.find((s) => s.id === activeSwarmId)

  const flowNodes: Node[] = useMemo(
    () =>
      tasks.map((t) => ({
        id: t.id,
        type: 'task',
        position: { x: t.position_x || 0, y: t.position_y || 0 },
        data: { task: t },
      })),
    [tasks]
  )

  const flowEdges: Edge[] = useMemo(
    () =>
      swarmEdges.map((e) => ({
        id: String(e.id),
        source: e.from_task_id,
        target: e.to_task_id,
        animated: e.edge_type === 'data_flow',
        style: {
          stroke: e.edge_type === 'data_flow' ? '#8b5cf6' : '#3f3f46',
          strokeWidth: 2,
        },
      })),
    [swarmEdges]
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges)

  useEffect(() => { setNodes(flowNodes) }, [flowNodes, setNodes])
  useEffect(() => { setEdges(flowEdges) }, [flowEdges, setEdges])

  const onConnect = useCallback(
    (conn: Connection) => {
      setEdges((eds) => addEdge({ ...conn, style: { stroke: '#3f3f46', strokeWidth: 2 } }, eds))
      if (conn.source && conn.target) {
        createEdge(conn.source, conn.target)
      }
    },
    [setEdges, createEdge]
  )

  const handleAddTask = async () => {
    if (!activeSwarmId) return
    await createTask({
      name: 'New Task',
      description: '',
      position_x: 200 + Math.random() * 200,
      position_y: 100 + Math.random() * 200,
    })
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/30 px-4 py-2">
        <div className="flex items-center gap-3">
          <select
            value={activeSwarmId ?? ''}
            onChange={(e) => selectSwarm(e.target.value || null)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          >
            <option value="">Select workflow...</option>
            {swarms.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name || s.original_task?.slice(0, 40) || s.id.slice(0, 8)}
              </option>
            ))}
          </select>

          {activeSwarm && <StatusBadge status={activeSwarm.status} />}
        </div>

        {activeSwarm && (
          <div className="flex items-center gap-1.5">
            <ToolbarButton icon={Plus} label="Add Task" onClick={handleAddTask} />
            <ToolbarButton icon={Zap} label="Decompose" onClick={() => decomposeSwarm(activeSwarm.id)} />
            <div className="mx-1 h-5 w-px bg-zinc-800" />
            {activeSwarm.status === 'ready' || activeSwarm.status === 'draft' ? (
              <ToolbarButton icon={Play} label="Start" onClick={() => startSwarm(activeSwarm.id)} accent />
            ) : activeSwarm.status === 'running' ? (
              <>
                <ToolbarButton icon={Pause} label="Pause" onClick={() => pauseSwarm(activeSwarm.id)} />
                <ToolbarButton icon={Square} label="Cancel" onClick={() => cancelSwarm(activeSwarm.id)} danger />
              </>
            ) : activeSwarm.status === 'paused' ? (
              <>
                <ToolbarButton icon={Play} label="Resume" onClick={() => startSwarm(activeSwarm.id)} accent />
                <ToolbarButton icon={Square} label="Cancel" onClick={() => cancelSwarm(activeSwarm.id)} danger />
              </>
            ) : null}
            <ToolbarButton icon={RefreshCw} label="Refresh" onClick={() => selectSwarm(activeSwarm.id)} />
          </div>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1">
        {activeSwarmId ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            className="bg-zinc-950"
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#27272a" />
            <Controls />
            <MiniMap
              nodeStrokeColor="#3f3f46"
              nodeColor="#18181b"
              maskColor="rgba(0,0,0,0.7)"
            />
          </ReactFlow>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="text-zinc-500">Select a workflow or create a new one</p>
              <p className="mt-1 text-sm text-zinc-600">
                Workflows let you chain agent tasks in a DAG
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ToolbarButton({
  icon: Icon,
  label,
  onClick,
  accent,
  danger,
}: {
  icon: typeof Play
  label: string
  onClick: () => void
  accent?: boolean
  danger?: boolean
}) {
  let cls = 'rounded-lg px-2.5 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors '
  if (accent) cls += 'bg-violet-600 text-white hover:bg-violet-500'
  else if (danger) cls += 'text-red-400 hover:bg-red-500/10'
  else cls += 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'

  return (
    <button onClick={onClick} title={label} className={cls}>
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}
