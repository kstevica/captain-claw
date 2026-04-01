import { useState } from 'react'
import {
  Monitor,
  GitBranch,
  Plus,
  Radio,
  ChevronLeft,
  Settings,
  Check,
  MessageSquare,
  BarChart3,
  Shield,
} from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { useAgentStore } from '../../stores/agentStore'
import { useAuthStore } from '../../stores/authStore'
import { useConnectionStore } from '../../stores/connectionStore'
import { useChatStore } from '../../stores/chatStore'
import { botportWS } from '../../services/ws'
import { StatusBadge } from '../common/StatusBadge'
import type { ViewMode } from '../../types'

const navItems: { id: ViewMode; icon: typeof Monitor; label: string; adminOnly?: boolean }[] = [
  { id: 'desktop', icon: Monitor, label: 'Agent Desktop' },
  { id: 'operations', icon: BarChart3, label: 'Operations' },
  { id: 'workflow', icon: GitBranch, label: 'Workflows' },
  { id: 'spawner', icon: Plus, label: 'Spawn Agent' },
  { id: 'admin', icon: Shield, label: 'Admin', adminOnly: true },
]

export function Sidebar() {
  const { view, setView, sidebarOpen, toggleSidebar } = useUIStore()
  const { instances, wsConnected, selectInstance, selectedInstanceId, fetchInstances, fetchStats, fetchConcerns } = useAgentStore()
  const { authEnabled, user: authUser } = useAuthStore()
  const { botportUrl, setBotportUrl } = useConnectionStore()
  const { sessions, chatOpen, switchChat } = useChatStore()
  const [showSettings, setShowSettings] = useState(false)
  const [urlDraft, setUrlDraft] = useState(botportUrl)
  const chatSessions = Array.from(sessions.values())

  const connectedInstances = instances.filter((i) => i.status === 'connected')

  const applyConnection = () => {
    setBotportUrl(urlDraft.trim())
    // Reconnect WS and refetch data with new URL
    botportWS.reconnect()
    fetchInstances()
    fetchStats()
    fetchConcerns(true)
    setShowSettings(false)
  }

  return (
    <aside
      className={`flex flex-col border-r border-zinc-800 bg-zinc-900/50 transition-all duration-200 ${
        sidebarOpen ? 'w-[var(--fd-sidebar)]' : 'w-14'
      }`}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-zinc-800 px-3">
        {sidebarOpen && (
          <div className="flex items-center gap-2">
            <Radio className={`h-4 w-4 ${wsConnected ? 'text-emerald-400' : 'text-zinc-600'}`} />
            <span className="text-sm font-semibold tracking-tight">Flight Deck</span>
          </div>
        )}
        <div className="flex items-center gap-0.5">
          {sidebarOpen && (
            <button
              onClick={() => { setUrlDraft(botportUrl); setShowSettings(!showSettings) }}
              className={`rounded p-1 transition-colors ${
                showSettings
                  ? 'bg-zinc-800 text-zinc-200'
                  : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
              }`}
              title="Connection settings"
            >
              <Settings className="h-4 w-4" />
            </button>
          )}
          <button
            onClick={toggleSidebar}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <ChevronLeft className={`h-4 w-4 transition-transform ${sidebarOpen ? '' : 'rotate-180'}`} />
          </button>
        </div>
      </div>

      {/* Connection settings */}
      {showSettings && sidebarOpen && (
        <div className="border-b border-zinc-800 p-3">
          <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-zinc-500">
            BotPort Address
          </label>
          <div className="flex gap-1.5">
            <input
              value={urlDraft}
              onChange={(e) => setUrlDraft(e.target.value)}
              placeholder="http://localhost:23180"
              onKeyDown={(e) => e.key === 'Enter' && applyConnection()}
              className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
            <button
              onClick={applyConnection}
              className="rounded-md bg-violet-600 px-2 py-1.5 text-xs text-white hover:bg-violet-500"
              title="Apply"
            >
              <Check className="h-3.5 w-3.5" />
            </button>
          </div>
          <p className="mt-1.5 text-xs text-zinc-600">
            {botportUrl
              ? <>Connected to <span className="font-mono text-zinc-500">{botportUrl}</span></>
              : 'Empty = use Vite dev proxy (localhost)'}
          </p>
        </div>
      )}

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 p-2">
        {navItems.filter((item) => {
          if (item.adminOnly) return authEnabled && authUser?.role === 'admin'
          return true
        }).map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setView(id)}
            className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors ${
              view === id
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
            }`}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {sidebarOpen && label}
          </button>
        ))}
      </nav>

      {/* Connected agents */}
      {sidebarOpen && (
        <div className="mt-4 flex flex-1 flex-col overflow-hidden border-t border-zinc-800">
          <div className="flex items-center justify-between px-3 py-2">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Agents ({connectedInstances.length})
            </span>
          </div>
          <div className="flex-1 overflow-y-auto px-2 pb-2">
            {connectedInstances.map((inst) => (
              <button
                key={inst.id}
                onClick={() => selectInstance(inst.id)}
                className={`mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                  selectedInstanceId === inst.id
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{inst.name || inst.id.slice(0, 8)}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <StatusBadge status={inst.status} />
                    {inst.active_concerns > 0 && (
                      <span className="text-xs text-zinc-500">{inst.active_concerns} tasks</span>
                    )}
                  </div>
                </div>
              </button>
            ))}
            {connectedInstances.length === 0 && (
              <p className="px-2.5 py-4 text-center text-xs text-zinc-600">
                No agents connected
              </p>
            )}
          </div>
        </div>
      )}

      {/* Active chats */}
      {sidebarOpen && chatSessions.length > 0 && !chatOpen && (
        <div className="border-t border-zinc-800 p-2">
          <div className="px-2 py-1">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Chats ({chatSessions.length})
            </span>
          </div>
          {chatSessions.map((s) => (
            <button
              key={s.containerId}
              onClick={() => switchChat(s.containerId)}
              className="mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
            >
              <MessageSquare className="h-3.5 w-3.5 shrink-0" />
              <span className="min-w-0 flex-1 truncate">{s.containerName}</span>
              <span className={`h-1.5 w-1.5 rounded-full ${s.connected ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
            </button>
          ))}
        </div>
      )}
    </aside>
  )
}
