import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { SwarmTask } from '../../types'
import { StatusBadge } from '../common/StatusBadge'
import { Play, Pause, RotateCcw, SkipForward, CheckCircle } from 'lucide-react'
import { useWorkflowStore } from '../../stores/workflowStore'

type TaskNodeData = {
  task: SwarmTask
}

export const TaskNode = memo(function TaskNode({ data }: NodeProps & { data: TaskNodeData }) {
  const { approveTask, retryTask, skipTask } = useWorkflowStore()
  const task = data.task

  const borderColor: Record<string, string> = {
    running: 'border-blue-500/50',
    completed: 'border-emerald-500/30',
    failed: 'border-red-500/30',
    pending_approval: 'border-orange-500/50',
    queued: 'border-zinc-700',
    waiting: 'border-amber-500/30',
    paused: 'border-zinc-700',
    skipped: 'border-zinc-800',
    retrying: 'border-amber-500/30',
  }

  return (
    <div
      className={`min-w-[200px] max-w-[280px] rounded-xl border bg-zinc-900 shadow-lg ${
        borderColor[task.status] ?? 'border-zinc-700'
      }`}
    >
      <Handle type="target" position={Position.Top} className="!bg-zinc-600 !border-zinc-500 !w-2.5 !h-2.5" />

      <div className="p-3">
        {/* Header */}
        <div className="flex items-start justify-between mb-1.5">
          <h4 className="text-sm font-medium text-zinc-200 leading-tight pr-2">{task.name || 'Untitled task'}</h4>
          <StatusBadge status={task.status} />
        </div>

        {/* Description */}
        {task.description && (
          <p className="text-xs text-zinc-500 line-clamp-2 mb-2">{task.description}</p>
        )}

        {/* Assigned */}
        {task.assigned_persona && (
          <div className="text-xs text-zinc-600 mb-2">
            <span className="text-zinc-500">Agent:</span> {task.assigned_persona}
          </div>
        )}

        {/* Error */}
        {task.error_message && (
          <div className="mb-2 rounded bg-red-500/10 px-2 py-1 text-xs text-red-400 line-clamp-2">
            {task.error_message}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-1">
          {task.status === 'pending_approval' && (
            <NodeButton icon={CheckCircle} label="Approve" onClick={() => approveTask(task.id)} color="text-emerald-400" />
          )}
          {task.status === 'failed' && (
            <NodeButton icon={RotateCcw} label="Retry" onClick={() => retryTask(task.id)} color="text-amber-400" />
          )}
          {(task.status === 'failed' || task.status === 'queued') && (
            <NodeButton icon={SkipForward} label="Skip" onClick={() => skipTask(task.id)} color="text-zinc-400" />
          )}
          {task.status === 'running' && (
            <NodeButton icon={Pause} label="Pause" onClick={() => {}} color="text-zinc-400" />
          )}
          {task.status === 'paused' && (
            <NodeButton icon={Play} label="Resume" onClick={() => {}} color="text-blue-400" />
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-zinc-600 !border-zinc-500 !w-2.5 !h-2.5" />
    </div>
  )
})

function NodeButton({
  icon: Icon,
  label,
  onClick,
  color,
}: {
  icon: typeof Play
  label: string
  onClick: () => void
  color: string
}) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick() }}
      title={label}
      className={`rounded p-1 hover:bg-zinc-800 ${color}`}
    >
      <Icon className="h-3.5 w-3.5" />
    </button>
  )
}
