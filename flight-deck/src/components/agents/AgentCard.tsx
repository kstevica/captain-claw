import { Monitor, Cpu, Clock, MessageSquare } from 'lucide-react'
import type { InstanceInfo } from '../../types'
import { StatusBadge } from '../common/StatusBadge'
import { useAgentStore } from '../../stores/agentStore'

export function AgentCard({ instance }: { instance: InstanceInfo }) {
  const { selectInstance, selectedInstanceId } = useAgentStore()
  const isSelected = selectedInstanceId === instance.id
  const activity = instance.activity

  const statusText = activity?.status?.text as string | undefined
  const thinkingText = activity?.thinking?.text as string | undefined

  return (
    <div
      onClick={() => selectInstance(isSelected ? null : instance.id)}
      className={`group cursor-pointer rounded-xl border p-4 transition-all ${
        isSelected
          ? 'border-violet-500/50 bg-violet-500/5 shadow-lg shadow-violet-500/5'
          : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900'
      }`}
    >
      {/* Header */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <div className={`rounded-lg p-1.5 ${isSelected ? 'bg-violet-500/20' : 'bg-zinc-800'}`}>
            <Monitor className="h-4 w-4 text-zinc-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">{instance.name || instance.id.slice(0, 12)}</h3>
            <span className="text-xs text-zinc-500 font-mono">{instance.id.slice(0, 8)}</span>
          </div>
        </div>
        <StatusBadge status={instance.status} />
      </div>

      {/* Activity */}
      {(statusText || thinkingText) && (
        <div className="mb-3 rounded-lg bg-zinc-950/50 px-3 py-2">
          {statusText && (
            <p className="text-xs text-zinc-400 truncate">
              <span className="text-zinc-600 mr-1">Status:</span>{statusText}
            </p>
          )}
          {thinkingText && (
            <p className="text-xs text-violet-400/80 truncate mt-0.5">
              <span className="text-zinc-600 mr-1">Thinking:</span>{thinkingText}
            </p>
          )}
        </div>
      )}

      {/* Personas */}
      {instance.personas.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1">
          {instance.personas.map((p) => (
            <span
              key={p.name}
              className="rounded-md bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400"
              title={p.description}
            >
              {p.name}
            </span>
          ))}
        </div>
      )}

      {/* Footer stats */}
      <div className="flex items-center gap-4 text-xs text-zinc-500">
        <span className="flex items-center gap-1">
          <MessageSquare className="h-3 w-3" />
          {instance.active_concerns} active
        </span>
        <span className="flex items-center gap-1">
          <Cpu className="h-3 w-3" />
          {instance.max_concurrent} max
        </span>
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {timeAgo(instance.last_heartbeat)}
        </span>
      </div>
    </div>
  )
}

function timeAgo(iso: string): string {
  if (!iso) return '--'
  const diff = Date.now() - new Date(iso).getTime()
  const secs = Math.floor(diff / 1000)
  if (secs < 60) return `${secs}s ago`
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m ago`
  return `${Math.floor(mins / 60)}h ago`
}
