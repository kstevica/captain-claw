import { Pin, PinOff, ArrowRight, FileText, Share2 } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { CouncilMessage as CouncilMessageType } from '../../stores/councilStore'
import { suitabilityLabel } from '../../utils/suitability'
import { AgentAvatar } from './councilLook'
import { type CouncilTone } from './councilStyles'

const ACTION_BADGES: Record<string, { label: string; color: string }> = {
  answer: { label: 'Answer', color: 'bg-sky-500/10 text-sky-700 dark:text-sky-300' },
  respond: { label: 'Respond', color: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' },
  challenge: { label: 'Challenge', color: 'bg-rose-500/10 text-rose-700 dark:text-rose-300' },
  refine: { label: 'Refine', color: 'bg-amber-500/10 text-amber-700 dark:text-amber-300' },
  broaden: { label: 'Broaden', color: 'bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300' },
  pass: { label: 'Pass', color: 'bg-zinc-800 text-zinc-400' },
  inject: { label: 'Inject', color: 'bg-violet-500/10 text-violet-700 dark:text-violet-300' },
}

// The card's look by who wrote it: agents plain, the person, the moderator
// and the synthesis tinted.
const ROLE_CARD: Record<string, string> = {
  agent: 'border-zinc-800 bg-zinc-900/40',
  user: 'border-emerald-500/25 bg-emerald-500/[0.05]',
  system: 'border-zinc-800 bg-zinc-900/40',
  moderator: 'border-amber-500/25 bg-amber-500/[0.05]',
  synthesis: 'border-cyan-500/25 bg-cyan-500/[0.05]',
}

const ROLE_TONE: Record<string, CouncilTone | undefined> = {
  user: 'emerald',
  moderator: 'amber',
  synthesis: 'cyan',
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
        <div className="flex justify-center py-1">
          <div className="w-full max-w-3xl rounded-xl border border-emerald-500/25 bg-emerald-500/[0.04] px-3.5 py-2.5 text-xs">
            <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-zinc-100">
              <Share2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
              <span className="font-medium text-emerald-700 dark:text-emerald-400">Flight Deck</span>
              <span className="text-zinc-500">shared a file from</span>
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
            <div className="mt-2 flex flex-wrap gap-1">
              {meta.recipients.map((r) => (
                <span
                  key={r.agentId}
                  className="inline-flex items-center gap-1 rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300"
                  title={r.path}
                >
                  <ArrowRight className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                  {r.agentName}
                </span>
              ))}
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className="flex justify-center py-0.5">
        <span className="max-w-full rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-center text-xs text-zinc-500">
          {message.content}
        </span>
      </div>
    )
  }

  const partial = (message.metadata as { partial?: boolean })?.partial

  return (
    <div className="group flex gap-3">
      <AgentAvatar
        name={message.agentName || (message.role === 'user' ? 'You' : '?')}
        id={message.agentId || message.agentName}
        tone={ROLE_TONE[message.role]}
      />
      <div className={`relative min-w-0 flex-1 rounded-2xl rounded-tl-md border px-4 py-3 ${ROLE_CARD[message.role] || ROLE_CARD.agent}`}>
        {/* Header */}
        <div className="mb-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 pr-6 text-xs">
          <span className="text-sm font-semibold text-zinc-100">{message.agentName}</span>

          {partial && (
            <span
              className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300"
              title="This turn was interrupted (server stopped mid-response). Partial text shown — restart the round to regenerate."
            >
              interrupted
            </span>
          )}

          {badge && (
            <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${badge.color}`}>
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
              className="ml-auto flex items-center gap-1.5"
              title="Self-rated suitability for the topic from this contribution. Each agent reports it at the top of every turn; the moderator uses it to pick who speaks next."
            >
              <div className="h-1.5 w-14 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-violet-500 transition-all"
                  style={{ width: `${message.suitability * 100}%` }}
                />
              </div>
              <span className="text-[11px] tabular-nums text-zinc-500">
                {suitabilityLabel(message.suitability)} ({Math.round(message.suitability * 100)}%)
              </span>
            </div>
          )}

          {message.role === 'moderator' && (
            <span className="ml-auto text-[11px] font-medium text-amber-700 dark:text-amber-400">moderator</span>
          )}

          {message.role === 'synthesis' && (
            <span className="ml-auto text-[11px] font-medium text-cyan-700 dark:text-cyan-400">synthesis</span>
          )}
        </div>

        {/* Content */}
        <div className="fd-markdown text-sm leading-relaxed text-zinc-300 [&_ol]:list-decimal [&_ul]:list-disc">
          <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
        </div>

        {/* Pin button */}
        {message.id > 0 && (
          <button
            onClick={() => onPin(message.id)}
            title={message.pinned ? 'Unpin' : 'Pin'}
            className={`absolute right-2 top-2 rounded-lg p-1.5 transition-all ${
              message.pinned
                ? 'text-amber-600 opacity-100 dark:text-amber-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-amber-600 dark:hover:text-amber-400 [@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover:opacity-100'
            }`}
          >
            {message.pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
          </button>
        )}
      </div>
    </div>
  )
}
