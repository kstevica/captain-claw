import { useRef, useEffect, useMemo } from 'react'
import { MessagesSquare } from 'lucide-react'
import { CouncilMessageBubble } from './CouncilMessage'
import { AgentAvatar } from './councilLook'
import type { CouncilMessage, CouncilSession } from '../../stores/councilStore'

interface CouncilDiscussionProps {
  session: CouncilSession
  onPin: (id: number) => void
  speaking: string
}

export function CouncilDiscussion({ session, onPin, speaking }: CouncilDiscussionProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  const agentNames = useMemo(() => {
    const map = new Map<string, string>()
    for (const a of session.agents) map.set(a.id, a.name)
    return map
  }, [session.agents])

  // Group messages by round
  const rounds = useMemo(() => {
    const groups = new Map<number, CouncilMessage[]>()
    for (const msg of session.messages) {
      const r = msg.round
      if (!groups.has(r)) groups.set(r, [])
      groups.get(r)!.push(msg)
    }
    return Array.from(groups.entries()).sort((a, b) => a[0] - b[0])
  }, [session.messages])

  // Track whether user is scrolled near the bottom
  const isNearBottom = useRef(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    // Consider "near bottom" if within 120px of the end
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  // Auto-scroll on new messages or when a new speaker starts (only if already at bottom)
  useEffect(() => {
    if (isNearBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [session.messages.length, speaking])

  const speakingAgent = session.agents.find(a => a.id === speaking)

  return (
    <div ref={scrollRef} onScroll={handleScroll} className="min-h-0 flex-1 space-y-6 overflow-auto px-4 py-4 wide:px-6">
      {rounds.map(([round, messages]) => (
        <div key={round} className="space-y-3">
          <div className="sticky -top-4 z-10 -mx-4 flex items-center gap-3 bg-zinc-950/95 px-4 py-1.5 backdrop-blur-sm wide:-mx-6 wide:px-6">
            <div className="h-px flex-1 bg-zinc-800" />
            <span className="rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-zinc-500 shadow-sm">
              Round {round}
            </span>
            <div className="h-px flex-1 bg-zinc-800" />
          </div>

          {messages.map(msg => (
            <CouncilMessageBubble
              key={msg.localId || msg.id}
              message={msg}
              onPin={onPin}
              agentNames={agentNames}
            />
          ))}
        </div>
      ))}

      {speaking && speakingAgent && (
        <div className="flex items-center gap-3">
          <AgentAvatar name={speakingAgent.name} id={speakingAgent.id} />
          <div className="flex items-center gap-2.5 rounded-2xl border border-zinc-800 bg-zinc-900/40 px-3.5 py-2.5 text-sm text-zinc-400">
            <div className="flex gap-1">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" style={{ animationDelay: '0.2s' }} />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" style={{ animationDelay: '0.4s' }} />
            </div>
            <span>{speakingAgent.name} is speaking...</span>
          </div>
        </div>
      )}

      {session.messages.length === 0 && (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-zinc-500">
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-zinc-900 text-zinc-500">
            <MessagesSquare className="h-5 w-5" />
          </span>
          Council discussion will appear here once started.
        </div>
      )}
    </div>
  )
}
