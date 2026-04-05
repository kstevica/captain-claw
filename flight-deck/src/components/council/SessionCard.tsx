import { Users, Trash2, MessageSquare, Clock } from 'lucide-react'
import type { CouncilSessionSummary } from '../../stores/councilStore'

const TYPE_COLORS: Record<string, string> = {
  debate: 'bg-red-500/20 text-red-400',
  brainstorm: 'bg-green-500/20 text-green-400',
  review: 'bg-blue-500/20 text-blue-400',
  planning: 'bg-yellow-500/20 text-yellow-400',
  interview: 'bg-cyan-500/20 text-cyan-400',
  troubleshoot: 'bg-orange-500/20 text-orange-400',
  critique: 'bg-rose-500/20 text-rose-400',
  freeform: 'bg-purple-500/20 text-purple-400',
}

const STATUS_COLORS: Record<string, string> = {
  setup: 'bg-zinc-500/20 text-zinc-400',
  active: 'bg-emerald-500/20 text-emerald-400',
  synthesizing: 'bg-violet-500/20 text-violet-400',
  concluded: 'bg-zinc-500/20 text-zinc-400',
}

interface SessionCardProps {
  session: CouncilSessionSummary
  onOpen: (id: string) => void
  onDelete: (id: string) => void
}

export function SessionCard({ session, onOpen, onDelete }: SessionCardProps) {
  const agentCount = (() => {
    try { return JSON.parse(session.agents || '[]').length } catch { return 0 }
  })()

  const date = new Date(session.created_at).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  return (
    <button
      onClick={() => onOpen(session.id)}
      className="group flex flex-col gap-2 rounded-xl border border-zinc-700/50 bg-zinc-800/50 p-4 text-left transition-all hover:border-violet-500/40 hover:bg-zinc-800"
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-medium text-zinc-200 line-clamp-1">{session.title || 'Untitled'}</h3>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(session.id) }}
          className="shrink-0 rounded p-1 text-zinc-500 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-red-400 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      <p className="text-xs text-zinc-400 line-clamp-2">{session.topic}</p>

      <div className="flex flex-wrap items-center gap-2 mt-auto pt-1">
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium uppercase ${TYPE_COLORS[session.session_type] || TYPE_COLORS.brainstorm}`}>
          {session.session_type}
        </span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${STATUS_COLORS[session.status] || STATUS_COLORS.setup}`}>
          {session.status}
        </span>
        <span className="flex items-center gap-1 text-[10px] text-zinc-500">
          <Users className="h-3 w-3" /> {agentCount}
        </span>
        <span className="flex items-center gap-1 text-[10px] text-zinc-500">
          <MessageSquare className="h-3 w-3" /> R{session.current_round}/{session.max_rounds}
        </span>
        <span className="flex items-center gap-1 text-[10px] text-zinc-500 ml-auto">
          <Clock className="h-3 w-3" /> {date}
        </span>
      </div>
    </button>
  )
}
