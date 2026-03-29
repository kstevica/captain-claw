import { useState, useMemo, useCallback } from 'react'
import {
  Box,
  Settings2 as Cog,
  Circle,
  Clock,
  Activity,
  Send,
  StopCircle,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Filter,
  ArrowUpDown,
  Megaphone,
  Loader2,
  Wrench,
  MessageSquare,
  X,
} from 'lucide-react'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useChatStore } from '../../stores/chatStore'

// ── Types ──

interface UnifiedAgentRow {
  id: string
  kind: 'docker' | 'local'
  name: string
  description: string
  status: string
  port: number | null
  host: string
  auth: string
  // from chat session
  chatConnected: boolean
  busy: boolean
  statusText: string
  lastMessageTime: string | null
  messageCount: number
  created: string
}

type SortField = 'name' | 'status' | 'lastActive' | 'type'
type FilterStatus = 'all' | 'running' | 'stopped' | 'offline'

// ── Helpers ──

function relativeTime(iso: string | null): string {
  if (!iso) return 'never'
  const diff = Date.now() - new Date(iso).getTime()
  if (diff < 0) return 'just now'
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}

function statusColor(status: string): string {
  if (/running|online/i.test(status)) return 'text-emerald-400'
  if (/exited|stopped|offline/i.test(status)) return 'text-zinc-500'
  if (/error|dead/i.test(status)) return 'text-red-400'
  if (/created|restarting/i.test(status)) return 'text-amber-400'
  return 'text-zinc-500'
}

function statusPriority(status: string): number {
  if (/running|online/i.test(status)) return 0
  if (/created|restarting/i.test(status)) return 1
  if (/exited|stopped|offline/i.test(status)) return 2
  return 3
}

// ── Component ──

export function DirectorPanel() {
  const containers = useContainerStore((s) => s.containers)
  const { stopContainer, restartContainer } = useContainerStore()
  const localAgents = useLocalAgentStore((s) => s.agents)
  const { probeAll } = useLocalAgentStore()
  const chatSessions = useChatStore((s) => s.sessions)
  const { openChat, sendMessage } = useChatStore()

  const [sortField, setSortField] = useState<SortField>('status')
  const [sortAsc, setSortAsc] = useState(true)
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all')
  const [showBroadcast, setShowBroadcast] = useState(false)
  const [broadcastMsg, setBroadcastMsg] = useState('')
  const [broadcastSending, setBroadcastSending] = useState(false)
  const [expandedActivity, setExpandedActivity] = useState<Set<string>>(new Set())
  const [showFilters, setShowFilters] = useState(false)

  // Build unified list
  const agents: UnifiedAgentRow[] = useMemo(() => {
    const rows: UnifiedAgentRow[] = []

    for (const c of containers) {
      const session = chatSessions.get(c.id)
      const lastMsg = session?.messages?.length
        ? session.messages[session.messages.length - 1]
        : null
      rows.push({
        id: c.id,
        kind: 'docker',
        name: c.agent_name || c.name,
        description: c.description || '',
        status: c.status,
        port: c.web_port,
        host: 'localhost',
        auth: c.web_auth || '',
        chatConnected: session?.connected ?? false,
        busy: session?.busy ?? false,
        statusText: session?.statusText ?? '',
        lastMessageTime: lastMsg?.timestamp ?? null,
        messageCount: session?.messages?.length ?? 0,
        created: c.created,
      })
    }

    for (const a of localAgents) {
      const session = chatSessions.get(a.id)
      const lastMsg = session?.messages?.length
        ? session.messages[session.messages.length - 1]
        : null
      rows.push({
        id: a.id,
        kind: 'local',
        name: a.name,
        description: a.description || '',
        status: a.status === 'online' ? 'running' : a.status === 'offline' ? 'stopped' : 'unknown',
        port: a.port,
        host: a.host,
        auth: a.authToken || '',
        chatConnected: session?.connected ?? false,
        busy: session?.busy ?? false,
        statusText: session?.statusText ?? '',
        lastMessageTime: lastMsg?.timestamp ?? null,
        messageCount: session?.messages?.length ?? 0,
        created: '',
      })
    }

    return rows
  }, [containers, localAgents, chatSessions])

  // Filter
  const filtered = useMemo(() => {
    if (filterStatus === 'all') return agents
    return agents.filter((a) => {
      if (filterStatus === 'running') return /running|online/i.test(a.status)
      if (filterStatus === 'stopped') return /exited|stopped/i.test(a.status)
      if (filterStatus === 'offline') return /offline|unknown/i.test(a.status)
      return true
    })
  }, [agents, filterStatus])

  // Sort
  const sorted = useMemo(() => {
    const list = [...filtered]
    list.sort((a, b) => {
      let cmp = 0
      switch (sortField) {
        case 'name':
          cmp = a.name.localeCompare(b.name)
          break
        case 'status':
          cmp = statusPriority(a.status) - statusPriority(b.status)
          break
        case 'lastActive':
          cmp = (a.lastMessageTime ?? '').localeCompare(b.lastMessageTime ?? '')
          break
        case 'type':
          cmp = a.kind.localeCompare(b.kind)
          break
      }
      return sortAsc ? cmp : -cmp
    })
    return list
  }, [filtered, sortField, sortAsc])

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  const toggleActivity = (id: string) => {
    setExpandedActivity((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // Broadcast
  const handleBroadcast = useCallback(async () => {
    if (!broadcastMsg.trim()) return
    setBroadcastSending(true)
    const running = agents.filter((a) => /running|online/i.test(a.status))

    for (const agent of running) {
      // Ensure chat is connected
      const session = chatSessions.get(agent.id)
      if (!session) {
        openChat(agent.id, agent.name, agent.host, agent.port ?? 0, agent.auth)
        // Small delay for WS to connect
        await new Promise((r) => setTimeout(r, 500))
      }
      sendMessage(agent.id, broadcastMsg.trim())
    }

    setBroadcastSending(false)
    setBroadcastMsg('')
    setShowBroadcast(false)
  }, [broadcastMsg, agents, chatSessions, openChat, sendMessage])

  // Bulk actions
  const runningCount = agents.filter((a) => /running|online/i.test(a.status)).length
  const dockerRunning = agents.filter((a) => a.kind === 'docker' && /running/i.test(a.status))

  const handleStopAll = async () => {
    for (const a of dockerRunning) await stopContainer(a.id)
  }

  const handleRestartAll = async () => {
    for (const a of dockerRunning) await restartContainer(a.id)
    probeAll()
  }

  const handleConnectChat = (agent: UnifiedAgentRow) => {
    if (agent.port) {
      openChat(agent.id, agent.name, agent.host, agent.port, agent.auth)
    }
  }

  // Activity feed: last N messages across all sessions
  const activityFeed = useMemo(() => {
    const items: { agentId: string; agentName: string; role: string; content: string; timestamp: string; toolName?: string }[] = []
    for (const agent of agents) {
      const session = chatSessions.get(agent.id)
      if (!session?.messages?.length) continue
      // Take last 5 messages per agent
      const recent = session.messages.slice(-5)
      for (const m of recent) {
        items.push({
          agentId: agent.id,
          agentName: agent.name,
          role: m.role,
          content: m.content || m.tool_name || '',
          timestamp: m.timestamp,
          toolName: m.tool_name,
        })
      }
    }
    items.sort((a, b) => b.timestamp.localeCompare(a.timestamp))
    return items.slice(0, 30)
  }, [agents, chatSessions])

  const [activeTab, setActiveTab] = useState<'agents' | 'activity'>('agents')

  return (
    <div className="flex h-full flex-col border-r border-zinc-800 bg-zinc-950/60">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-violet-400" />
          <span className="text-sm font-semibold">Director</span>
          <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">
            {runningCount}/{agents.length}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowBroadcast(!showBroadcast)}
            className={`rounded p-1 transition-colors ${showBroadcast ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}
            title="Broadcast to all"
          >
            <Megaphone className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`rounded p-1 transition-colors ${showFilters ? 'bg-zinc-800 text-zinc-200' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}
            title="Filters & sort"
          >
            <Filter className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Broadcast bar */}
      {showBroadcast && (
        <div className="border-b border-zinc-800 bg-violet-950/20 p-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-violet-400">
            <Megaphone className="h-3 w-3" />
            Broadcast to {runningCount} running agent{runningCount !== 1 ? 's' : ''}
          </div>
          <div className="flex gap-1.5">
            <input
              value={broadcastMsg}
              onChange={(e) => setBroadcastMsg(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleBroadcast()}
              placeholder="Message all agents..."
              className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              autoFocus
            />
            <button
              onClick={handleBroadcast}
              disabled={broadcastSending || !broadcastMsg.trim()}
              className="rounded-md bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {broadcastSending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
            </button>
            <button
              onClick={() => setShowBroadcast(false)}
              className="rounded-md px-1 py-1 text-zinc-500 hover:text-zinc-300"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Filter/sort bar */}
      {showFilters && (
        <div className="border-b border-zinc-800 bg-zinc-900/40 p-2.5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Filter</span>
            <div className="flex gap-0.5">
              {(['all', 'running', 'stopped', 'offline'] as FilterStatus[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilterStatus(f)}
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium capitalize transition-colors ${
                    filterStatus === f ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Sort</span>
            <div className="flex gap-0.5">
              {([['status', 'Status'], ['name', 'Name'], ['lastActive', 'Last Active'], ['type', 'Type']] as [SortField, string][]).map(([field, label]) => (
                <button
                  key={field}
                  onClick={() => toggleSort(field)}
                  className={`flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                    sortField === field ? 'bg-violet-600/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {label}
                  {sortField === field && (
                    <ArrowUpDown className="h-2.5 w-2.5" />
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Quick actions */}
      {dockerRunning.length > 0 && (
        <div className="flex items-center gap-1 border-b border-zinc-800 px-3 py-1.5">
          <button
            onClick={handleStopAll}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <StopCircle className="h-3 w-3" />
            Stop All
          </button>
          <button
            onClick={handleRestartAll}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <RefreshCw className="h-3 w-3" />
            Restart All
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-zinc-800">
        <button
          onClick={() => setActiveTab('agents')}
          className={`flex-1 py-1.5 text-[11px] font-medium transition-colors ${
            activeTab === 'agents' ? 'border-b-2 border-violet-500 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Agents
        </button>
        <button
          onClick={() => setActiveTab('activity')}
          className={`flex-1 py-1.5 text-[11px] font-medium transition-colors ${
            activeTab === 'activity' ? 'border-b-2 border-violet-500 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Activity Feed
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'agents' ? (
          <div className="divide-y divide-zinc-800/60">
            {sorted.length === 0 && (
              <div className="px-3 py-8 text-center text-xs text-zinc-600">No agents match filter</div>
            )}
            {sorted.map((agent) => (
              <AgentRow
                key={agent.id}
                agent={agent}
                expanded={expandedActivity.has(agent.id)}
                onToggle={() => toggleActivity(agent.id)}
                onConnectChat={() => handleConnectChat(agent)}
                chatSessions={chatSessions}
              />
            ))}
          </div>
        ) : (
          <ActivityFeed items={activityFeed} />
        )}
      </div>
    </div>
  )
}

// ── Agent Row ──

function AgentRow({
  agent,
  expanded,
  onToggle,
  onConnectChat,
  chatSessions,
}: {
  agent: UnifiedAgentRow
  expanded: boolean
  onToggle: () => void
  onConnectChat: () => void
  chatSessions: Map<string, unknown>
}) {
  const isRunning = /running|online/i.test(agent.status)
  const hasChat = chatSessions.has(agent.id) && agent.chatConnected

  return (
    <div className={`group transition-colors ${expanded ? 'bg-zinc-900/40' : 'hover:bg-zinc-900/20'}`}>
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer"
        onClick={onToggle}
      >
        {/* Type icon */}
        <div className="flex-shrink-0">
          {agent.kind === 'docker' ? (
            <Box className="h-3.5 w-3.5 text-blue-400/70" />
          ) : (
            <Cog className="h-3.5 w-3.5 text-amber-400/70" />
          )}
        </div>

        {/* Name + status line */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="truncate text-xs font-medium text-zinc-200">{agent.name}</span>
            {agent.port && (
              <span className="text-[10px] font-mono text-zinc-600">:{agent.port}</span>
            )}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <Circle className={`h-1.5 w-1.5 fill-current ${statusColor(agent.status)}`} />
            <span className={`text-[10px] capitalize ${statusColor(agent.status)}`}>{agent.status}</span>
            {agent.busy && agent.statusText && (
              <>
                <span className="text-[10px] text-zinc-600">·</span>
                <span className="flex items-center gap-0.5 text-[10px] text-violet-400 truncate max-w-[120px]">
                  <Wrench className="h-2.5 w-2.5 shrink-0" />
                  {agent.statusText}
                </span>
              </>
            )}
            {!agent.busy && isRunning && (
              <>
                <span className="text-[10px] text-zinc-600">·</span>
                <span className="text-[10px] text-zinc-500">idle</span>
              </>
            )}
          </div>
        </div>

        {/* Right side: last active + actions */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {agent.lastMessageTime && (
            <div className="flex items-center gap-0.5 text-[10px] text-zinc-600" title={agent.lastMessageTime}>
              <Clock className="h-2.5 w-2.5" />
              {relativeTime(agent.lastMessageTime)}
            </div>
          )}
          {isRunning && !hasChat && (
            <button
              onClick={(e) => { e.stopPropagation(); onConnectChat() }}
              className="rounded p-0.5 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Connect chat"
            >
              <MessageSquare className="h-3 w-3" />
            </button>
          )}
          {hasChat && (
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 flex-shrink-0" title="Chat connected" />
          )}
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-zinc-600" />
          ) : (
            <ChevronRight className="h-3 w-3 text-zinc-600" />
          )}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-2.5 pt-0.5">
          {agent.description && (
            <p className="text-[10px] text-zinc-500 mb-1.5 pl-5">{agent.description}</p>
          )}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 pl-5 text-[10px]">
            <div>
              <span className="text-zinc-600">Type: </span>
              <span className="text-zinc-400">{agent.kind === 'docker' ? 'Docker' : 'Local'}</span>
            </div>
            <div>
              <span className="text-zinc-600">Host: </span>
              <span className="text-zinc-400 font-mono">{agent.host}{agent.port ? `:${agent.port}` : ''}</span>
            </div>
            {agent.created && (
              <div>
                <span className="text-zinc-600">Created: </span>
                <span className="text-zinc-400">{relativeTime(agent.created)}</span>
              </div>
            )}
            <div>
              <span className="text-zinc-600">Messages: </span>
              <span className="text-zinc-400">{agent.messageCount}</span>
            </div>
            <div>
              <span className="text-zinc-600">Chat: </span>
              <span className={agent.chatConnected ? 'text-emerald-400' : 'text-zinc-500'}>
                {agent.chatConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div>
              <span className="text-zinc-600">Activity: </span>
              <span className={agent.busy ? 'text-violet-400' : 'text-zinc-500'}>
                {agent.busy ? 'Working' : 'Idle'}
              </span>
            </div>
          </div>

          {/* Recent messages preview */}
          {agent.chatConnected && agent.messageCount > 0 && (
            <RecentMessages agentId={agent.id} />
          )}
        </div>
      )}
    </div>
  )
}

// ── Recent Messages (for expanded row) ──

function RecentMessages({ agentId }: { agentId: string }) {
  const session = useChatStore((s) => s.sessions.get(agentId))
  if (!session?.messages?.length) return null
  const recent = session.messages.slice(-3)
  return (
    <div className="mt-2 pl-5 space-y-1">
      <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-600">Recent</span>
      {recent.map((m, i) => (
        <div key={i} className="flex items-start gap-1.5 text-[10px]">
          <span className={`shrink-0 font-medium ${
            m.role === 'user' ? 'text-violet-400' : m.role === 'tool' ? 'text-amber-400' : 'text-zinc-400'
          }`}>
            {m.role === 'user' ? 'You' : m.role === 'tool' ? '🔧' : 'AI'}:
          </span>
          <span className="text-zinc-500 truncate">{m.content?.slice(0, 100) || m.tool_name || ''}</span>
        </div>
      ))}
    </div>
  )
}

// ── Activity Feed ──

function ActivityFeed({ items }: { items: { agentId: string; agentName: string; role: string; content: string; timestamp: string; toolName?: string }[] }) {
  if (items.length === 0) {
    return (
      <div className="px-3 py-8 text-center text-xs text-zinc-600">
        No activity yet. Connect to agents via chat to see live activity.
      </div>
    )
  }

  return (
    <div className="divide-y divide-zinc-800/40">
      {items.map((item, i) => (
        <div key={i} className="flex items-start gap-2 px-3 py-1.5 hover:bg-zinc-900/30">
          <div className="mt-0.5 flex-shrink-0">
            {item.role === 'tool' ? (
              <Wrench className="h-3 w-3 text-amber-400/70" />
            ) : item.role === 'user' ? (
              <Send className="h-3 w-3 text-violet-400/70" />
            ) : item.role === 'system' ? (
              <Activity className="h-3 w-3 text-red-400/70" />
            ) : (
              <MessageSquare className="h-3 w-3 text-zinc-500" />
            )}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-medium text-zinc-400">{item.agentName}</span>
              <span className="text-[10px] text-zinc-600">{relativeTime(item.timestamp)}</span>
            </div>
            <p className="text-[10px] text-zinc-500 truncate">
              {item.toolName ? `Used ${item.toolName}` : item.content?.slice(0, 120) || ''}
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}
