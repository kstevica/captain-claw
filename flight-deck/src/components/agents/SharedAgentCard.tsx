import { useState } from 'react'
import { Users, MessageSquare, LogOut, Loader2 } from 'lucide-react'
import type { SharedAgent } from '../../services/sharedAgents'
import { useChatStore } from '../../stores/chatStore'
import { useSharedAgentStore } from '../../stores/sharedAgentStore'
import { useNotificationStore } from '../../stores/notificationStore'

// An agent another deck user shared with you, on the Agent Desktop. Chat only:
// no power, config, files or memory — those stay with its owner.

export function SharedAgentCard({ agent }: { agent: SharedAgent }) {
  const openSharedChat = useChatStore((s) => s.openSharedChat)
  const leave = useSharedAgentStore((s) => s.leave)
  const [leaving, setLeaving] = useState(false)
  const running = agent.status === 'running'
  const owner = agent.owner_name || agent.owner_email || 'another user'
  const name = agent.name || agent.slug

  const handleLeave = async () => {
    if (!confirm(`Leave ${name}? You'll lose access until ${owner} shares it again.`)) return
    setLeaving(true)
    try {
      await leave(agent.agent_ref, agent.owner_id)
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not leave agent',
        `${name}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setLeaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${running ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-zinc-600'}`}
          title={running ? 'Running' : 'Stopped'}
        />
        <span className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">{name}</span>
        <span
          className="flex shrink-0 items-center gap-0.5 rounded border border-sky-500/25 bg-sky-500/15 px-1 py-0.5 text-[9px] font-medium text-sky-700 dark:text-sky-300"
          title={`Shared by ${owner}`}
        >
          <Users className="h-2.5 w-2.5" />shared
        </span>
      </div>
      <div className="text-[11px] text-zinc-500">
        Shared by {owner}{running ? '' : ' · Stopped'}
      </div>
      {agent.description && (
        <p className="line-clamp-2 text-xs text-zinc-400">{agent.description}</p>
      )}
      <div className="mt-1 flex items-center gap-2">
        <button
          onClick={() => openSharedChat(agent.agent_ref, name, agent.owner_name || owner)}
          disabled={!running}
          title={running ? `Chat with ${name}` : `${name} is stopped — ask ${owner} to start it`}
          className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Chat
        </button>
        <button
          onClick={handleLeave}
          disabled={leaving}
          title={`Leave ${name}`}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-40 dark:hover:text-red-400"
        >
          {leaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogOut className="h-3.5 w-3.5" />}
          Leave
        </button>
      </div>
    </div>
  )
}
