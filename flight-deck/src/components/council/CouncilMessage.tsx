import { Pin, PinOff, ArrowRight, FileText, Share2 } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { CouncilMessage as CouncilMessageType } from '../../stores/councilStore'
import { suitabilityLabel } from '../../utils/suitability'

const ACTION_BADGES: Record<string, { label: string; color: string }> = {
  answer: { label: 'Answer', color: 'bg-blue-500/20 text-blue-400' },
  respond: { label: 'Respond', color: 'bg-green-500/20 text-green-400' },
  challenge: { label: 'Challenge', color: 'bg-red-500/20 text-red-400' },
  refine: { label: 'Refine', color: 'bg-amber-500/20 text-amber-400' },
  broaden: { label: 'Broaden', color: 'bg-purple-500/20 text-purple-400' },
  pass: { label: 'Pass', color: 'bg-zinc-500/20 text-zinc-400' },
  inject: { label: 'Inject', color: 'bg-violet-500/20 text-violet-400' },
}

const ROLE_COLORS: Record<string, string> = {
  agent: 'border-l-violet-500',
  user: 'border-l-emerald-500',
  system: 'border-l-zinc-500',
  moderator: 'border-l-amber-500',
  synthesis: 'border-l-cyan-500',
}

interface CouncilMessageProps {
  message: CouncilMessageType
  onPin: (id: number) => void
  agentNames: Map<string, string>
}

export function CouncilMessageBubble({ message, onPin, agentNames }: CouncilMessageProps) {
  const isSystem = message.role === 'system'
  const badge = ACTION_BADGES[message.action]

  if (isSystem) {
    // Special card for Flight Deck file-share notices
    const meta = message.metadata as
      | {
          kind?: string
          fileName?: string
          srcPath?: string
          srcAgentName?: string
          recipients?: { agentId: string; agentName: string; path: string }[]
        }
      | undefined
    if (meta?.kind === 'file_share' && meta.fileName && meta.recipients) {
      return (
        <div className="flex justify-center py-1.5">
          <div className="w-full max-w-3xl rounded-lg border border-emerald-500/30 bg-zinc-900/80 px-3 py-2 text-[11px] shadow-sm">
            <div className="flex items-center gap-2 text-zinc-100">
              <Share2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
              <span className="font-medium text-emerald-400">Flight Deck</span>
              <span className="text-zinc-400">shared a file from</span>
              <span className="font-medium text-zinc-100">{meta.srcAgentName}</span>
            </div>
            <div className="mt-1.5 flex items-center gap-1.5 text-zinc-300">
              <FileText className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
              <span
                className="truncate font-mono text-zinc-200"
                title={meta.srcPath}
              >
                {meta.fileName}
              </span>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {meta.recipients.map((r) => (
                <span
                  key={r.agentId}
                  className="inline-flex items-center gap-1 rounded-full bg-zinc-800/70 px-2 py-0.5 text-[10px] text-zinc-300"
                  title={r.path}
                >
                  <ArrowRight className="h-2.5 w-2.5 text-emerald-400" />
                  {r.agentName}
                </span>
              ))}
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="flex justify-center py-1">
        <span className="rounded-full bg-zinc-800/50 px-3 py-1 text-[11px] text-zinc-500">
          {message.content}
        </span>
      </div>
    )
  }

  return (
    <div className={`group relative rounded-lg border-l-2 bg-zinc-800/40 p-3 ${ROLE_COLORS[message.role] || ROLE_COLORS.agent}`}>
      {/* Header */}
      <div className="mb-1.5 flex items-center gap-2 text-xs">
        <span className="font-medium text-zinc-200">{message.agentName}</span>

        {badge && (
          <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${badge.color}`}>
            {badge.label}
          </span>
        )}

        {message.targetAgentId && (
          <span className="flex items-center gap-1 text-zinc-500">
            <ArrowRight className="h-3 w-3" />
            {agentNames.get(message.targetAgentId) || message.targetAgentId}
          </span>
        )}

        {message.role === 'agent' && (
          <div
            className="flex items-center gap-1 ml-auto"
            title="Self-rated suitability for the topic from this contribution. Each agent reports it at the top of every turn; the moderator uses it to pick who speaks next."
          >
            <div className="h-1.5 w-16 rounded-full bg-zinc-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-violet-500 transition-all"
                style={{ width: `${message.suitability * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-zinc-500">
              {suitabilityLabel(message.suitability)} ({Math.round(message.suitability * 100)}%)
            </span>
          </div>
        )}

        {message.role === 'moderator' && (
          <span className="text-[10px] text-amber-400 ml-auto">moderator</span>
        )}

        {message.role === 'synthesis' && (
          <span className="text-[10px] text-cyan-400 ml-auto">synthesis</span>
        )}
      </div>

      {/* Content */}
      <div className="fd-markdown text-sm text-zinc-300 leading-relaxed">
        <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
      </div>

      {/* Pin button */}
      {message.id > 0 && (
        <button
          onClick={() => onPin(message.id)}
          className={`absolute right-2 top-2 rounded p-1 transition-all ${
            message.pinned
              ? 'text-amber-400 opacity-100'
              : 'text-zinc-500 opacity-0 group-hover:opacity-100 hover:text-amber-400'
          }`}
        >
          {message.pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
        </button>
      )}
    </div>
  )
}
