import { Users, Trash2, MessageSquare, Clock, Square, Share2 } from 'lucide-react'
import type { CouncilSessionSummary } from '../../stores/councilStore'
import { Pill } from '../console/parts'
import { IconSquare, RoundBar, StatusPill, TypePill } from './councilLook'
import { typeLook } from './councilStyles'

interface SessionCardProps {
  session: CouncilSessionSummary
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onCancel?: (id: string) => void
  onShare?: (session: CouncilSessionSummary) => void
}

// Shown on hover where there is one; always on a touch screen.
const CARD_ACTION = 'rounded-lg p-1.5 text-zinc-500 transition-all hover:bg-zinc-800 [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100 [@media(hover:hover)]:group-focus-visible:opacity-100'

export function SessionCard({ session, onOpen, onDelete, onCancel, onShare }: SessionCardProps) {
  const running = session.status === 'active' || session.status === 'synthesizing'
  const agentCount = (() => {
    try { return JSON.parse(session.agents || '[]').length } catch { return 0 }
  })()

  const date = new Date(session.created_at).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
  const look = typeLook(session.session_type)

  return (
    <button
      onClick={() => onOpen(session.id)}
      className="group relative flex min-w-0 flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-all hover:-translate-y-px hover:border-zinc-700 hover:bg-zinc-900/70 hover:shadow-lg hover:shadow-black/10 focus-visible:border-violet-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40"
    >
      <div className="flex items-start gap-3">
        <IconSquare icon={look.icon} tone={look.tone} />
        <div className="min-w-0 flex-1 pt-0.5">
          <h3 className="line-clamp-2 text-[15px] font-semibold leading-snug text-zinc-100">{session.title || 'Untitled'}</h3>
          <div className="mt-0.5 flex items-center gap-1 text-xs text-zinc-500">
            <Clock className="h-3 w-3 shrink-0" /> <span className="truncate">{date}</span>
          </div>
        </div>
        <div className="-mr-1 -mt-1 flex shrink-0 items-center">
          {!session.shared && onShare && (
            <button
              onClick={(e) => { e.stopPropagation(); onShare(session) }}
              title="Share with other users"
              className={`${CARD_ACTION} hover:text-sky-600 dark:hover:text-sky-300`}
            >
              <Share2 className="h-3.5 w-3.5" />
            </button>
          )}
          {!session.shared && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(session.id) }}
              title="Delete"
              className={`${CARD_ACTION} hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400`}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
          {running && onCancel && (
            <button
              onClick={(e) => { e.stopPropagation(); onCancel(session.id) }}
              title="Stop this deliberation"
              className="ml-1 rounded-lg border border-zinc-700 p-1.5 text-zinc-400 transition-colors hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400"
            >
              <Square className="h-3 w-3 fill-current" />
            </button>
          )}
        </div>
      </div>

      {session.shared && (
        <div className="mt-3">
          <Pill tone="sky" icon={Users}>
            shared by {session.owner_name || session.owner_email || 'another user'} · {session.access}
          </Pill>
        </div>
      )}

      <p className="mt-3 line-clamp-2 text-sm leading-relaxed text-zinc-400">{session.topic}</p>

      <div className="mt-auto pt-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <StatusPill status={session.status} />
          <TypePill type={session.session_type} />
          <span className="ml-auto flex items-center gap-3 text-xs tabular-nums text-zinc-500">
            <span className="flex items-center gap-1" title={`${agentCount} agent${agentCount === 1 ? '' : 's'}`}>
              <Users className="h-3.5 w-3.5" /> {agentCount}
            </span>
            <span className="flex items-center gap-1" title={`Round ${session.current_round} of ${session.max_rounds}`}>
              <MessageSquare className="h-3.5 w-3.5" /> R{session.current_round}/{session.max_rounds}
            </span>
          </span>
        </div>
        <RoundBar current={session.current_round} max={session.max_rounds} className="mt-3" />
      </div>
    </button>
  )
}
