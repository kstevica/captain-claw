import { MessageSquare, Trash2, RefreshCw, Cpu, ExternalLink, Loader2, FolderOpen } from 'lucide-react'
import type { LocalAgent } from '../../stores/localAgentStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useChatStore } from '../../stores/chatStore'

const statusStyles: Record<string, string> = {
  online: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  offline: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  unknown: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
}

export function LocalAgentCard({ agent, onBrowseFiles }: { agent: LocalAgent; onBrowseFiles?: () => void }) {
  const { removeAgent, probeAgent } = useLocalAgentStore()
  const openChat = useChatStore((s) => s.openChat)
  const session = useChatStore((s) => s.sessions.get(agent.id))
  const busy = session?.busy ?? false
  const statusText = session?.statusText ?? ''

  return (
    <div className={`rounded-xl border bg-zinc-900/50 p-5 ${busy ? 'border-violet-500/40' : 'border-zinc-800'}`}>
      {/* Header */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-zinc-800 p-2">
            <Cpu className="h-5 w-5 text-zinc-400" />
          </div>
          <div>
            <h3 className="text-base font-semibold">{agent.name}</h3>
            <span className="text-xs text-zinc-500 font-mono">{agent.host}:{agent.port}</span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {agent.status === 'online' && (
            <>
              <button
                onClick={() => openChat(agent.id, agent.name, agent.host, agent.port, agent.authToken)}
                className="flex items-center gap-1 rounded-lg bg-violet-600/20 px-2 py-0.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30"
              >
                <MessageSquare className="h-3 w-3" />
                Chat
              </button>
              <button
                onClick={() => window.open(`http://${agent.host}:${agent.port}/chat`, '_blank')}
                className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <ExternalLink className="h-3 w-3" />
                Open
              </button>
            </>
          )}
          <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${statusStyles[agent.status]}`}>
            {agent.status === 'online' && (
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
              </span>
            )}
            {agent.status}
          </span>
        </div>
      </div>

      {/* Persona / Model override (visible when chat connected) */}
      {session?.connected && (session.models.length > 0 || session.personalities.length > 0) && (
        <div className="mb-3 flex flex-wrap gap-2">
          {session.personalities.length > 0 && (
            <select
              value={session.activePersonality}
              onChange={(e) => useChatStore.getState().setPersonality(agent.id, e.target.value)}
              className="flex-1 min-w-[120px] rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              <option value="">Default persona</option>
              {session.personalities.map((p) => (
                <option key={p.id || p.name} value={p.id || p.name}>{p.name}</option>
              ))}
            </select>
          )}
          {session.models.length > 0 && (
            <select
              value={session.activeModel}
              onChange={(e) => useChatStore.getState().setModel(agent.id, e.target.value)}
              className="flex-1 min-w-[120px] rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              <option value="">Default model</option>
              {session.models.map((m) => (
                <option key={m.selector || m.id} value={m.selector || m.id}>{m.label || m.id}</option>
              ))}
            </select>
          )}
        </div>
      )}

      {/* Busy indicator */}
      {busy && (
        <div className="mb-3 flex items-center gap-2 rounded-lg bg-violet-500/10 px-3 py-1.5">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-400" />
          <span className="text-xs text-violet-300">{statusText || 'Working...'}</span>
        </div>
      )}

      {/* Actions */}
      <div className="mt-4 flex items-center gap-1">
        {onBrowseFiles && (
          <button
            onClick={onBrowseFiles}
            className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <FolderOpen className="h-3.5 w-3.5" />
            Files
          </button>
        )}
        <button
          onClick={() => probeAgent(agent.id)}
          className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Probe
        </button>
        <div className="flex-1" />
        <button
          onClick={() => { if (confirm(`Remove '${agent.name}'?`)) removeAgent(agent.id) }}
          className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-red-400/60 hover:bg-red-500/10 hover:text-red-400"
        >
          <Trash2 className="h-3.5 w-3.5" />
          Remove
        </button>
      </div>
    </div>
  )
}
