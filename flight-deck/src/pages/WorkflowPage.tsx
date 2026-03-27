import { useCallback, useEffect, useMemo } from 'react'
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
import { Play, Pause, Square, RefreshCw, Plus, Zap } from 'lucide-react'
import { useWorkflowStore } from '../stores/workflowStore'
import { TaskNode } from '../components/workflow/TaskNode'
import { StatusBadge } from '../components/common/StatusBadge'

const nodeTypes = { task: TaskNode }

export function WorkflowPage() {
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

  // Convert SwarmTasks to React Flow nodes
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

  // Convert SwarmEdges to React Flow edges
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

  // Sync when data changes
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
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/30 px-4 py-2">
        <div className="flex items-center gap-3">
          {/* Swarm selector */}
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
