import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Radio,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  Search,
  Plus,
  Box,
  Cpu,
  Server,
  Play,
  Loader2,
  Sun,
  Moon,
  RefreshCw,
  LogOut,
  Keyboard,
  MessagesSquare,
  FolderOpen,
  Database,
  Settings,
} from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { useAgentStore } from '../../stores/agentStore'
import { useAuthStore, logoutUser } from '../../stores/authStore'
import { useChatStore, parseLaneKey } from '../../stores/chatStore'
import { useNotificationStore } from '../../stores/notificationStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useProcessStore } from '../../stores/processStore'
import { useThemeStore } from '../../stores/themeStore'
import { usePersistedSize } from '../../hooks/usePersistedSize'
import { ChatPanel } from '../agents/ChatPanel'
import { AgentFilesPanel } from '../agents/AgentFilesPanel'
import { AgentDatastorePanel } from '../agents/AgentDatastorePanel'
import { AgentConfigEditor } from '../agents/AgentConfigEditor'
import { NotificationBell } from '../common/NotificationCenter'
import { APP_VERSION, BUILD_DATE } from '../../version'

// ── Simple layout ────────────────────────────────────────────────────
//
// A chat-first arrangement of the same Flight Deck: agents on the left, the
// conversation in the middle, the active agent's files and datastore on the
// right. No nav, no pages, no director — those all live in the full layout,
// one click away. Both side columns collapse to a thin rail and remember
// their state.

type AgentState = 'running' | 'starting' | 'stopped' | 'unknown'

interface SimpleAgent {
  /** Chat id: docker container id, `proc-<slug>`, or the local agent id. */
  id: string
  kind: 'docker' | 'process' | 'local'
  name: string
  description: string
  state: AgentState
  /** Running AND listening on a web port — a chat can open right now. */
  reachable: boolean
  host: string
  port: number
  auth: string
  /** What the Start button needs: the container id or the process slug. */
  startKey: string
}

const STATE_RANK: Record<AgentState, number> = { running: 0, starting: 1, unknown: 2, stopped: 3 }

function dockerState(status: string): AgentState {
  if (/running/i.test(status)) return 'running'
  if (/created|restarting/i.test(status)) return 'starting'
  if (/exited|stopped|dead|paused/i.test(status)) return 'stopped'
  return 'unknown'
}

function stateLabel(a: SimpleAgent): string {
  if (a.state === 'running') return a.reachable ? 'Running' : 'Running — no web port'
  if (a.state === 'starting') return 'Starting…'
  if (a.state === 'stopped') return a.kind === 'local' ? 'Offline' : 'Stopped'
  return a.kind === 'local' ? 'Checking…' : 'Unknown'
}

function initials(name: string): string {
  const words = name.trim().split(/[\s_-]+/).filter(Boolean)
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase()
  return name.trim().slice(0, 2).toUpperCase() || '?'
}

const iconBtn = 'rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300'

export function SimpleLayout({ onToggleShortcuts }: { onToggleShortcuts: () => void }) {
  const { containers, fetchContainers, checkHealth, startContainer, descriptionOverrides: dockerDesc } = useContainerStore()
  const { processes, fetchProcesses, startProcess, descriptionOverrides: procDesc } = useProcessStore()
  const { agents: localAgents, probeAll, probeAgent } = useLocalAgentStore()
  const sessions = useChatStore((s) => s.sessions)
  const activeChatId = useChatStore((s) => s.activeChatId)
  const openChat = useChatStore((s) => s.openChat)
  const setLayoutMode = useUIStore((s) => s.setLayoutMode)
  const setView = useUIStore((s) => s.setView)
  const authEnabled = useAuthStore((s) => s.authEnabled)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)

  const [starting, setStarting] = useState<Set<string>>(() => new Set())
  const [optionsAgent, setOptionsAgent] = useState<SimpleAgent | null>(null)

  // Keep the agent list fresh — the Desktop page normally does this polling,
  // and it isn't mounted here. Same auth guard as there: with auth on and no
  // session yet, every request would only 401.
  useEffect(() => {
    if (authEnabled === true && !isAuthenticated) return
    checkHealth()
    fetchContainers()
    fetchProcesses()
    probeAll()
    const interval = setInterval(() => { fetchContainers(); fetchProcesses() }, 10000)
    return () => clearInterval(interval)
  }, [checkHealth, fetchContainers, fetchProcesses, probeAll, authEnabled, isAuthenticated])

  const agents: SimpleAgent[] = useMemo(() => {
    const list: SimpleAgent[] = []
    for (const c of containers) {
      const state = dockerState(c.status)
      list.push({
        id: c.id, kind: 'docker',
        name: c.agent_name || c.name,
        description: dockerDesc[c.id] || c.description || '',
        state, reachable: state === 'running' && !!c.web_port,
        host: 'localhost', port: c.web_port ?? 0, auth: c.web_auth || '',
        startKey: c.id,
      })
    }
    for (const p of processes) {
      const state: AgentState = p.status === 'running' ? 'running' : 'stopped'
      list.push({
        id: `proc-${p.slug}`, kind: 'process',
        name: p.name || p.slug,
        description: procDesc[p.slug] || p.description || '',
        state, reachable: state === 'running' && !!p.web_port,
        host: 'localhost', port: p.web_port, auth: p.web_auth || '',
        startKey: p.slug,
      })
    }
    for (const a of localAgents) {
      const state: AgentState = a.status === 'online' ? 'running' : a.status === 'offline' ? 'stopped' : 'unknown'
      list.push({
        id: a.id, kind: 'local',
        name: a.name, description: a.description || '',
        state, reachable: state === 'running',
        host: a.host, port: a.port, auth: a.authToken || '',
        startKey: a.id,
      })
    }
    list.sort((x, y) => STATE_RANK[x.state] - STATE_RANK[y.state] || x.name.localeCompare(y.name))
    return list
  }, [containers, processes, localAgents, dockerDesc, procDesc])

  // Per-agent chat state, folded across lanes: busy if any lane is, unread if
  // any lane finished something while the user looked elsewhere.
  const sessionInfo = useMemo(() => {
    const m = new Map<string, { busy: boolean; unread: boolean }>()
    for (const s of sessions.values()) {
      const cur = m.get(s.containerId) ?? { busy: false, unread: false }
      cur.busy = cur.busy || s.busy
      cur.unread = cur.unread || !!s.unread
      m.set(s.containerId, cur)
    }
    return m
  }, [sessions])

  const session = activeChatId ? sessions.get(activeChatId) : undefined
  const activeAgentId = session ? parseLaneKey(activeChatId!).containerId : null
  const activeAgent = activeAgentId ? agents.find((a) => a.id === activeAgentId) ?? null : null

  const handleOpen = (a: SimpleAgent) => {
    // An existing session survives the agent stopping; openChat just
    // re-activates it (stale host/port are ignored for a known key).
    if (!a.reachable && !sessions.has(a.id)) return
    openChat(a.id, a.name, a.host, a.port, a.auth)
  }

  const handleStart = async (a: SimpleAgent) => {
    setStarting((prev) => new Set(prev).add(a.id))
    try {
      if (a.kind === 'docker') await startContainer(a.startKey)
      else if (a.kind === 'process') await startProcess(a.startKey)
      else await probeAgent(a.startKey)
    } catch (e) {
      useNotificationStore.getState().add('error', 'Could not start agent', `${a.name}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setStarting((prev) => { const n = new Set(prev); n.delete(a.id); return n })
    }
  }

  const refresh = () => { fetchContainers(); fetchProcesses(); probeAll() }

  const goSpawn = () => { useChatStore.setState({ chatFullscreen: false }); setLayoutMode('full'); setView('spawner') }

  return (
    <div className="flex h-screen overflow-hidden">
      <AgentsColumn
        agents={agents}
        activeAgentId={activeAgentId}
        sessionInfo={sessionInfo}
        starting={starting}
        onOpen={handleOpen}
        onStart={handleStart}
        onRefresh={refresh}
        onSpawn={goSpawn}
        onOptions={setOptionsAgent}
        onToggleShortcuts={onToggleShortcuts}
      />

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden" style={{ minWidth: 320 }}>
        {session
          ? <ChatPanel variant="simple" />
          : <EmptyChat hasAgents={agents.length > 0} onSpawn={goSpawn} />}
      </main>

      <ContextColumn
        agentId={activeAgentId}
        agentName={session?.containerName ?? ''}
        onOptions={activeAgent ? () => setOptionsAgent(activeAgent) : undefined}
      />

      {optionsAgent && createPortal(
        <AgentConfigEditor
          kind={optionsAgent.kind}
          identifier={optionsAgent.startKey}
          agentName={optionsAgent.name}
          onClose={() => setOptionsAgent(null)}
        />,
        document.body,
      )}
    </div>
  )
}

// ── Left: agents ─────────────────────────────────────────────────────

function AgentsColumn({
  agents, activeAgentId, sessionInfo, starting, onOpen, onStart, onRefresh, onSpawn, onOptions, onToggleShortcuts,
}: {
  agents: SimpleAgent[]
  activeAgentId: string | null
  sessionInfo: Map<string, { busy: boolean; unread: boolean }>
  starting: Set<string>
  onOpen: (a: SimpleAgent) => void
  onStart: (a: SimpleAgent) => void
  onRefresh: () => void
  onSpawn: () => void
  onOptions: (a: SimpleAgent) => void
  onToggleShortcuts: () => void
}) {
  const open = useUIStore((s) => s.simpleLeftOpen)
  const setOpen = useUIStore((s) => s.setSimpleLeftOpen)
  const setLayoutMode = useUIStore((s) => s.setLayoutMode)
  const wsConnected = useAgentStore((s) => s.wsConnected)
  const { authEnabled, user: authUser } = useAuthStore()
  const { theme, toggle: toggleTheme } = useThemeStore()
  const width = usePersistedSize('fd:simple-left-width', 260, 200, 440, 'x')
  const [search, setSearch] = useState('')
  // The search box only renders past a threshold; gate the filter on the SAME
  // condition so a stale query can't strand the list once the box is gone.
  const searchable = agents.length > 6

  const shown = useMemo(() => {
    const q = searchable ? search.trim().toLowerCase() : ''
    if (!q) return agents
    return agents.filter((a) => a.name.toLowerCase().includes(q) || a.description.toLowerCase().includes(q))
  }, [agents, search, searchable])

  // ── Collapsed rail ──
  if (!open) {
    return (
      <aside className="flex w-14 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/50">
        <div className="flex flex-col items-center gap-0.5 border-b border-zinc-800 py-2">
          <button onClick={() => setOpen(true)} className={iconBtn} title="Show agents">
            <ChevronRight className="h-4 w-4" />
          </button>
          <button onClick={() => setLayoutMode('full')} className={iconBtn} title="Full view — all pages and panels">
            <LayoutDashboard className="h-4 w-4" />
          </button>
          {/* Bell lives at the TOP so its downward dropdown isn't clipped by
              the h-screen overflow-hidden root when the rail is collapsed. */}
          <NotificationBell align="left" />
        </div>
        <div className="flex flex-1 flex-col items-center gap-1.5 overflow-y-auto py-2">
          {agents.map((a) => {
            const info = sessionInfo.get(a.id)
            const active = a.id === activeAgentId
            const hasChat = sessionInfo.has(a.id)
            const openable = a.reachable || hasChat
            const canStart = a.kind === 'local' ? a.state !== 'running' : a.state === 'stopped'
            return (
              <button
                key={a.id}
                onClick={() => openable ? onOpen(a) : canStart ? onStart(a) : undefined}
                aria-disabled={!openable && !canStart}
                title={openable
                  ? `Chat with ${a.name}`
                  : canStart
                    ? `${a.name} — ${stateLabel(a)} · click to ${a.kind === 'local' ? 'check again' : 'start'}`
                    : `${a.name} — ${stateLabel(a)}`}
                className={`relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[11px] font-semibold transition-colors ${
                  active
                    ? 'bg-violet-600/20 text-violet-300 ring-1 ring-violet-500/40'
                    : openable
                      ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                      : 'bg-zinc-900 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                {starting.has(a.id) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : initials(a.name)}
                <StateDot agent={a} className="absolute -bottom-0.5 -right-0.5 ring-2 ring-zinc-900" />
                {info?.busy && (
                  <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-violet-400 ring-2 ring-zinc-900" />
                )}
                {info?.unread && !active && !info.busy && (
                  <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-sky-400 ring-2 ring-zinc-900" />
                )}
              </button>
            )
          })}
          <button onClick={onSpawn} className={`${iconBtn} mt-1`} title="Spawn a new agent">
            <Plus className="h-4 w-4" />
          </button>
        </div>
        <div className="flex flex-col items-center gap-0.5 border-t border-zinc-800 py-2">
          <button onClick={toggleTheme} className={iconBtn} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          </button>
          <button onClick={onRefresh} className={iconBtn} title="Refresh agents">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          {authEnabled && authUser && (
            <button onClick={logoutUser} className={iconBtn} title={`Sign out ${authUser.display_name || authUser.email}`}>
              <LogOut className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </aside>
    )
  }

  // ── Expanded column ──
  return (
    <aside
      className="relative flex shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/50"
      style={{ width: width.size }}
    >
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-zinc-800 px-3">
        <div className="flex min-w-0 items-center gap-2">
          <Radio className={`h-4 w-4 shrink-0 ${wsConnected ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-600'}`} />
          <div className="min-w-0">
            <span className="text-sm font-semibold tracking-tight">Flight Deck</span>
            <div className="text-[9px] leading-none text-zinc-600">v{APP_VERSION} &middot; {BUILD_DATE}</div>
          </div>
        </div>
        <div className="flex items-center gap-0.5">
          <NotificationBell align="left" />
          <button onClick={() => setLayoutMode('full')} className={iconBtn} title="Full view — all pages and panels">
            <LayoutDashboard className="h-4 w-4" />
          </button>
          <button onClick={() => setOpen(false)} className={iconBtn} title="Hide agents">
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* List header */}
      <div className="flex items-center justify-between px-3 pt-3 pb-1">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          Agents ({agents.length})
        </span>
        <button onClick={onSpawn} className="rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300" title="Spawn a new agent">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Search (only once the list is long enough to need it) */}
      {searchable && (
        <div className="px-2 pb-1.5">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-600" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search agents…"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-2 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
            />
          </div>
        </div>
      )}

      {/* Agents */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {agents.length === 0 && (
          <p className="px-2.5 py-6 text-center text-xs text-zinc-600">No agents yet.</p>
        )}
        {agents.length > 0 && shown.length === 0 && (
          <p className="px-2.5 py-6 text-center text-xs text-zinc-600">No agents match your search.</p>
        )}
        <ul className="flex flex-col gap-0.5">
          {shown.map((a) => {
            const info = sessionInfo.get(a.id)
            const active = a.id === activeAgentId
            const KindIcon = a.kind === 'docker' ? Box : a.kind === 'process' ? Cpu : Server
            const hasChat = sessionInfo.has(a.id)
            const openable = a.reachable || hasChat
            const canStart = a.kind === 'local' ? a.state !== 'running' : a.state === 'stopped'
            return (
              <li key={a.id} className="group flex items-center gap-0.5">
                <button
                  onClick={() => onOpen(a)}
                  disabled={!openable}
                  title={openable ? `Chat with ${a.name}` : `${a.name} — ${stateLabel(a)}`}
                  className={`flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors ${
                    active
                      ? 'bg-zinc-800 text-zinc-100'
                      : openable
                        ? 'text-zinc-300 hover:bg-zinc-800/50 hover:text-zinc-100'
                        : 'text-zinc-500'
                  }`}
                >
                  <StateDot agent={a} className="shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate font-medium">{a.name}</span>
                      {info?.busy && (
                        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-violet-400" title="Working" />
                      )}
                      {info?.unread && !active && !info.busy && (
                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400" title="New reply" />
                      )}
                    </div>
                    <div className="truncate text-[11px] text-zinc-500">
                      {a.state === 'running' && a.reachable && a.description ? a.description : stateLabel(a)}
                    </div>
                  </div>
                  <KindIcon className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
                </button>
                <button
                  onClick={() => onOptions(a)}
                  title={`Options — ${a.name}`}
                  className="shrink-0 rounded p-1.5 text-zinc-500 opacity-0 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover:opacity-100"
                >
                  <Settings className="h-3.5 w-3.5" />
                </button>
                {canStart && (
                  <button
                    onClick={() => onStart(a)}
                    disabled={starting.has(a.id)}
                    title={a.kind === 'local' ? 'Check again' : 'Start agent'}
                    className="shrink-0 rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-emerald-600 dark:hover:text-emerald-400 disabled:opacity-60"
                  >
                    {starting.has(a.id)
                      ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      : a.kind === 'local' ? <RefreshCw className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between gap-2 border-t border-zinc-800 px-2 py-1.5">
        <div className="flex items-center gap-0.5">
          <button onClick={toggleTheme} className={iconBtn} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
            {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          </button>
          <button onClick={onToggleShortcuts} className={iconBtn} title="Keyboard Shortcuts (Cmd+K)">
            <Keyboard className="h-3.5 w-3.5" />
          </button>
          <button onClick={onRefresh} className={iconBtn} title="Refresh agents">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
        {authEnabled && authUser && (
          <div className="flex min-w-0 items-center gap-1">
            <span className="truncate text-xs text-zinc-500">{authUser.display_name || authUser.email}</span>
            <button onClick={logoutUser} className={iconBtn} title="Sign out">
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Right edge: drag to resize */}
      <div
        onMouseDown={width.onResizeStart}
        title="Drag to resize"
        className="absolute right-0 top-0 z-20 h-full w-1.5 cursor-col-resize transition-colors hover:bg-violet-500/30 active:bg-violet-500/40"
      />
    </aside>
  )
}

function StateDot({ agent, className = '' }: { agent: SimpleAgent; className?: string }) {
  const cls = agent.state === 'running'
    ? (agent.reachable ? 'bg-emerald-600 dark:bg-emerald-400' : 'bg-amber-500 dark:bg-amber-400')
    : agent.state === 'starting'
      ? 'animate-pulse bg-amber-400'
      : agent.state === 'unknown'
        ? 'bg-zinc-500'
        : 'bg-zinc-600'
  return <span className={`h-2 w-2 rounded-full ${cls} ${className}`} />
}

// ── Centre: nothing open yet ─────────────────────────────────────────

function EmptyChat({ hasAgents, onSpawn }: { hasAgents: boolean; onSpawn: () => void }) {
  return (
    <div className="flex h-full items-center justify-center bg-zinc-950/80 p-6">
      <div className="w-full max-w-sm rounded-2xl border border-zinc-800 bg-zinc-900/60 p-8 text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-violet-600/20 text-violet-400">
          <MessagesSquare className="h-6 w-6" />
        </div>
        <h2 className="mb-1.5 text-lg font-semibold text-zinc-100">
          {hasAgents ? 'Pick an agent' : 'No agents yet'}
        </h2>
        <p className="text-sm text-zinc-400">
          {hasAgents
            ? 'Choose an agent on the left to start a conversation. Its files and data show up on the right.'
            : 'Spawn your first agent to start chatting.'}
        </p>
        {!hasAgents && (
          <button
            onClick={onSpawn}
            className="mt-6 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500"
          >
            Spawn an agent
          </button>
        )}
      </div>
    </div>
  )
}

// ── Right: files + datastore of the active agent ─────────────────────

function ContextColumn({ agentId, agentName, onOptions }: { agentId: string | null; agentName: string; onOptions?: () => void }) {
  const open = useUIStore((s) => s.simpleRightOpen)
  const setOpen = useUIStore((s) => s.setSimpleRightOpen)
  // The handle is on the LEFT edge of a right-docked column, so dragging left
  // makes it wider.
  const width = usePersistedSize('fd:simple-right-width', 320, 240, 640, 'x', 'backward')
  // Measure the height available to the files+datastore split so the files
  // pane can never be taller than (region − a datastore minimum).
  const regionRef = useRef<HTMLDivElement>(null)
  const [regionH, setRegionH] = useState(0)
  useEffect(() => {
    const el = regionRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setRegionH(el.clientHeight))
    ro.observe(el)
    setRegionH(el.clientHeight)
    return () => ro.disconnect()
  }, [open])
  const DATASTORE_MIN = 140
  const filesLiveMax = regionH > 0 ? Math.max(120, regionH - DATASTORE_MIN) : undefined
  const filesH = usePersistedSize('fd:simple-right-files-height', 360, 120, 1400, 'y', 'forward', filesLiveMax)

  if (!open) {
    return (
      <aside className="flex w-12 shrink-0 flex-col items-center gap-0.5 border-l border-zinc-800 bg-zinc-900/50 py-2">
        <button onClick={() => setOpen(true)} className={iconBtn} title="Show files and data">
          <ChevronLeft className="h-4 w-4" />
        </button>
        {onOptions && (
          <button onClick={onOptions} className={iconBtn} title="Agent options">
            <Settings className="h-4 w-4" />
          </button>
        )}
        <button onClick={() => setOpen(true)} className={iconBtn} title="Files">
          <FolderOpen className="h-4 w-4 text-violet-400" />
        </button>
        <button onClick={() => setOpen(true)} className={iconBtn} title="Datastore">
          <Database className="h-4 w-4 text-emerald-400" />
        </button>
      </aside>
    )
  }

  return (
    <aside
      className="relative flex shrink-0 flex-col border-l border-zinc-800 bg-zinc-950/40"
      style={{ width: width.size }}
    >
      {/* Left edge: drag to resize */}
      <div
        onMouseDown={width.onResizeStart}
        title="Drag to resize"
        className="absolute left-0 top-0 z-20 h-full w-1.5 cursor-col-resize transition-colors hover:bg-violet-500/30 active:bg-violet-500/40"
      />

      {/* Header */}
      <div className="flex h-10 shrink-0 items-center justify-between border-b border-zinc-800 px-3">
        <span className="min-w-0 truncate text-xs font-medium text-zinc-400" title={agentName || undefined}>
          {agentId ? agentName : 'Files & data'}
        </span>
        <div className="flex items-center gap-0.5">
          {onOptions && (
            <button onClick={onOptions} className={iconBtn} title="Agent options">
              <Settings className="h-3.5 w-3.5" />
            </button>
          )}
          <button onClick={() => setOpen(false)} className={iconBtn} title="Hide files and data">
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {agentId ? (
        <div ref={regionRef} className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {/* Top: files. Shrinkable with a floor so a stale oversized height
              can't push the datastore off-screen. */}
          <div className="min-h-[120px] overflow-hidden" style={{ height: filesH.size }}>
            <AgentFilesPanel key={agentId} containerId={agentId} />
          </div>
          {/* Divider (drag to resize files vs datastore) */}
          <div
            onMouseDown={filesH.onResizeStart}
            title="Drag to resize"
            className="h-1 shrink-0 cursor-row-resize border-y border-zinc-800 bg-zinc-900 transition-colors hover:bg-violet-500/40"
          />
          {/* Bottom: datastore tables — keeps a floor so it never hits 0. */}
          <div className="min-h-[120px] flex-1 overflow-hidden">
            <AgentDatastorePanel key={agentId} containerId={agentId} />
          </div>
        </div>
      ) : (
        <div className="flex flex-1 items-center justify-center px-6 text-center text-xs text-zinc-500">
          Open a chat to see that agent's files and datastore here.
        </div>
      )}
    </aside>
  )
}
