import { useRef, useEffect, useMemo } from 'react'
import { CouncilMessageBubble } from './CouncilMessage'
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
    <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-auto px-4 py-3 space-y-4">
      {rounds.map(([round, messages]) => (
        <div key={round} className="space-y-2">
          <div className="sticky top-0 z-10 flex items-center gap-2 py-1">
            <div className="h-px flex-1 bg-zinc-700/50" />
            <span className="rounded-full bg-zinc-800 px-3 py-0.5 text-[10px] font-medium text-zinc-400 border border-zinc-700/50">
              Round {round}
            </span>
            <div className="h-px flex-1 bg-zinc-700/50" />
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
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-400">
          <div className="flex gap-1">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" />
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: '0.2s' }} />
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" style={{ animationDelay: '0.4s' }} />
          </div>
          <span>{speakingAgent.name} is speaking...</span>
        </div>
      )}

      {session.messages.length === 0 && (
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">
          Council discussion will appear here once started.
        </div>
      )}
    </div>
  )
}
