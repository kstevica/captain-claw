import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
import { useAgentStore } from '../stores/agentStore'
import { useContainerStore } from '../stores/containerStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { useProcessStore } from '../stores/processStore'
import { AgentCard } from '../components/agents/AgentCard'
import { AgentDetail } from '../components/agents/AgentDetail'
import { ContainerCard } from '../components/agents/ContainerCard'
import { LocalAgentCard } from '../components/agents/LocalAgentCard'
import { ProcessCard } from '../components/agents/ProcessCard'
import { FileBrowser } from '../components/agents/FileBrowser'
import { Radio, Plus, Server, LayoutGrid, Move } from 'lucide-react'
import type { AgentEndpoint } from '../services/fileTransfer'
import { useGroupStore } from '../stores/groupStore'
import { useAuthStore } from '../stores/authStore'
import { GroupFilter } from '../components/common/AgentGroups'
import { queueSave, registerHydrator } from '../services/settingsSync'

// ── Unified agent item ──

type UnifiedAgent =
  | { kind: 'docker'; id: string; data: ReturnType<typeof useContainerStore.getState>['containers'][number] }
  | { kind: 'local'; id: string; data: ReturnType<typeof useLocalAgentStore.getState>['agents'][number] }
  | { kind: 'process'; id: string; data: ReturnType<typeof useProcessStore.getState>['processes'][number] }

// ── Position persistence ──

interface Position { x: number; y: number }

const POS_KEY = 'fd:agent-positions'
const LAYOUT_KEY = 'fd:agent-layout-mode'
const CARD_W = 480
const CARD_GAP = 24

function _persist(key: string, value: string) {
  localStorage.setItem(key, value)
  if (useAuthStore.getState().authEnabled) queueSave(key, value)
}

function loadPositions(): Record<string, Position> {
  try { return JSON.parse(localStorage.getItem(POS_KEY) || '{}') } catch { return {} }
}
function savePositions(pos: Record<string, Position>) {
  _persist(POS_KEY, JSON.stringify(pos))
}
function loadLayoutMode(): 'grid' | 'free' {
  return (localStorage.getItem(LAYOUT_KEY) as 'grid' | 'free') || 'grid'
}
function saveLayoutMode(mode: 'grid' | 'free') {
  _persist(LAYOUT_KEY, mode)
}

// Hydrate from server settings on login — notify mounted component to re-read
type DesktopHydrateListener = () => void
const _hydrateListeners = new Set<DesktopHydrateListener>()

registerHydrator((settings) => {
  let changed = false
  const posVal = settings[POS_KEY]
  if (posVal) { localStorage.setItem(POS_KEY, posVal); changed = true }
  const layoutVal = settings[LAYOUT_KEY]
  if (layoutVal) { localStorage.setItem(LAYOUT_KEY, layoutVal); changed = true }
  if (changed) {
    for (const fn of _hydrateListeners) fn()
  }
})

// Auto-layout: arrange cards in a grid pattern for initial positions
function autoPosition(index: number, containerWidth: number): Position {
  const cols = Math.max(1, Math.floor((containerWidth + CARD_GAP) / (CARD_W + CARD_GAP)))
  const col = index % cols
  const row = Math.floor(index / cols)
  return { x: col * (CARD_W + CARD_GAP), y: row * 220 }
}

export function DesktopPage() {
  const { instances, concerns, selectedInstanceId, selectInstance } = useAgentStore()
  const { containers, fetchContainers, dockerAvailable, checkHealth } = useContainerStore()
  const { agents: localAgents, addAgent, probeAll } = useLocalAgentStore()
  const { processes, fetchProcesses } = useProcessStore()
  const selectedInstance = instances.find((i) => i.id === selectedInstanceId)
  const [showAddAgent, setShowAddAgent] = useState(false)
  const [browsingAgent, setBrowsingAgent] = useState<AgentEndpoint | null>(null)
  const [positions, setPositions] = useState<Record<string, Position>>(loadPositions)
  const [layoutMode, setLayoutMode] = useState<'grid' | 'free'>(loadLayoutMode)
  const [groupFilter, setGroupFilter] = useState<string | null>(null)

  // Re-read from localStorage when server settings hydrate
  useEffect(() => {
    const onHydrate = () => {
      setPositions(loadPositions())
      setLayoutMode(loadLayoutMode())
    }
    _hydrateListeners.add(onHydrate)
    return () => { _hydrateListeners.delete(onHydrate) }
  }, [])
  const groups = useGroupStore((s) => s.groups)

  // Drag state
  const [dragId, setDragId] = useState<string | null>(null)
  const dragStart = useRef<{ mouseX: number; mouseY: number; startX: number; startY: number } | null>(null)
  const canvasRef = useRef<HTMLDivElement>(null)

  // Build list of all agents with web endpoints for file transfer
  const allAgentEndpoints: AgentEndpoint[] = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port)
      .map((c) => ({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })),
    ...processes
      .filter((p) => p.status === 'running')
      .map((p) => ({ id: `proc-${p.slug}`, name: p.name, host: 'localhost', port: p.web_port, auth: p.web_auth })),
    ...localAgents
      .filter((a) => a.status === 'online')
      .map((a) => ({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })),
  ]

  useEffect(() => {
    checkHealth()
    fetchContainers()
    fetchProcesses()
    probeAll()
    const interval = setInterval(() => { fetchContainers(); fetchProcesses() }, 10000)
    return () => clearInterval(interval)
  }, [checkHealth, fetchContainers, fetchProcesses, probeAll])

  // Build unified agent list (with optional group filter)
  const unifiedAgents: UnifiedAgent[] = useMemo(() => {
    const all: UnifiedAgent[] = [
      ...containers.map((c) => ({ kind: 'docker' as const, id: c.id, data: c })),
      ...processes.map((p) => ({ kind: 'process' as const, id: `proc-${p.slug}`, data: p })),
      ...localAgents.map((a) => ({ kind: 'local' as const, id: a.id, data: a })),
    ]
    if (!groupFilter) return all
    const group = groups.find((g) => g.id === groupFilter)
    if (!group) return all
    return all.filter((a) => group.agentIds.includes(a.id))
  }, [containers, processes, localAgents, groupFilter, groups])

  // Assign initial positions for new agents
  useEffect(() => {
    if (layoutMode !== 'free') return
    const canvasW = canvasRef.current?.clientWidth ?? 960
    let changed = false
    const updated = { ...positions }
    unifiedAgents.forEach((agent, i) => {
      if (!updated[agent.id]) {
        updated[agent.id] = autoPosition(i, canvasW)
        changed = true
      }
    })
    if (changed) {
      savePositions(updated)
      setPositions(updated)
    }
  }, [unifiedAgents, layoutMode]) // eslint-disable-line react-hooks/exhaustive-deps

  // Free drag handlers (pointer events for smooth dragging)
  const handlePointerDown = useCallback((e: React.PointerEvent, id: string) => {
    // Only start drag from the handle
    if (layoutMode !== 'free') return
    e.preventDefault()
    e.stopPropagation()
    const pos = positions[id] || { x: 0, y: 0 }
    dragStart.current = { mouseX: e.clientX, mouseY: e.clientY, startX: pos.x, startY: pos.y }
    setDragId(id)
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }, [layoutMode, positions])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragId || !dragStart.current) return
    const dx = e.clientX - dragStart.current.mouseX
    const dy = e.clientY - dragStart.current.mouseY
    const newX = Math.max(0, dragStart.current.startX + dx)
    const newY = Math.max(0, dragStart.current.startY + dy)
    setPositions((prev) => ({ ...prev, [dragId]: { x: newX, y: newY } }))
  }, [dragId])

  const handlePointerUp = useCallback(() => {
    if (dragId) {
      savePositions({ ...positions, [dragId]: positions[dragId] })
    }
    setDragId(null)
    dragStart.current = null
  }, [dragId, positions])

  const toggleLayout = () => {
    const next = layoutMode === 'grid' ? 'free' : 'grid'
    saveLayoutMode(next)
    setLayoutMode(next)
    // When switching to free, auto-position any agents without positions
    if (next === 'free') {
      const canvasW = canvasRef.current?.clientWidth ?? 960
      const updated = { ...positions }
      unifiedAgents.forEach((agent, i) => {
        if (!updated[agent.id]) {
          updated[agent.id] = autoPosition(i, canvasW)
        }
      })
      savePositions(updated)
      setPositions(updated)
    }
  }

  const resetPositions = () => {
    const canvasW = canvasRef.current?.clientWidth ?? 960
    const updated: Record<string, Position> = {}
    unifiedAgents.forEach((agent, i) => {
      updated[agent.id] = autoPosition(i, canvasW)
    })
    savePositions(updated)
    setPositions(updated)
  }

  const hasContent = instances.length > 0 || containers.length > 0 || processes.length > 0 || localAgents.length > 0
  const agentCount = containers.length + processes.length + localAgents.length

  // Calculate canvas height for free mode
  const canvasHeight = useMemo(() => {
    if (layoutMode !== 'free') return 'auto'
    let maxY = 400
    for (const agent of unifiedAgents) {
      const pos = positions[agent.id]
      if (pos) maxY = Math.max(maxY, pos.y + 260)
    }
    return `${maxY}px`
  }, [layoutMode, unifiedAgents, positions])

  const renderAgentCard = (agent: UnifiedAgent, onDragStart?: (e: React.PointerEvent) => void, isDragging?: boolean) => {
    if (agent.kind === 'docker') {
      return (
        <ContainerCard
          container={agent.data}
          onBrowseFiles={
            agent.data.status === 'running' && agent.data.web_port
              ? () => setBrowsingAgent({ id: agent.data.id, name: agent.data.agent_name || agent.data.name, host: 'localhost', port: agent.data.web_port!, auth: agent.data.web_auth })
              : undefined
          }
          onDragStart={onDragStart}
          isDragging={isDragging}
        />
      )
    }
    if (agent.kind === 'process') {
      return (
        <ProcessCard
          process={agent.data}
          onBrowseFiles={
            agent.data.status === 'running'
              ? () => setBrowsingAgent({ id: `proc-${agent.data.slug}`, name: agent.data.name, host: 'localhost', port: agent.data.web_port, auth: agent.data.web_auth })
              : undefined
          }
          onDragStart={onDragStart}
          isDragging={isDragging}
        />
      )
    }
    return (
      <LocalAgentCard
        agent={agent.data}
        onBrowseFiles={
          agent.data.status === 'online'
            ? () => setBrowsingAgent({ id: agent.data.id, name: agent.data.name, host: agent.data.host, port: agent.data.port, auth: agent.data.authToken })
            : undefined
        }
        onDragStart={onDragStart}
        isDragging={isDragging}
      />
    )
  }

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-auto p-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold">Agent Desktop</h1>
            <p className="text-sm text-zinc-500">Monitor and control your personal assistants</p>
          </div>
          <div className="flex items-center gap-2">
            <GroupFilter selected={groupFilter} onChange={setGroupFilter} />
            {layoutMode === 'free' && (
              <button
                onClick={resetPositions}
                className="rounded-md px-2 py-1 text-xs font-medium text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                title="Reset positions"
              >
                Reset
              </button>
            )}
            <button
              onClick={toggleLayout}
              className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 border border-zinc-800"
              title={layoutMode === 'grid' ? 'Switch to free layout' : 'Switch to grid layout'}
            >
              {layoutMode === 'grid' ? <Move className="h-3.5 w-3.5" /> : <LayoutGrid className="h-3.5 w-3.5" />}
              {layoutMode === 'grid' ? 'Free Layout' : 'Grid Layout'}
            </button>
          </div>
        </div>

        {/* BotPort connected agents */}
        {instances.length > 0 && (
          <div className="mb-8">
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              <Radio className="h-3.5 w-3.5" />
              BotPort Agents ({instances.length})
            </div>
            <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
              {instances.map((inst) => (
                <AgentCard key={inst.id} instance={inst} />
              ))}
            </div>
          </div>
        )}

        {/* Unified Agents (Docker + Local) */}
        <div className="mb-8">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              <Server className="h-3.5 w-3.5" />
              Agents ({agentCount})
            </div>
            <button
              onClick={() => setShowAddAgent(!showAddAgent)}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Plus className="h-3.5 w-3.5" />
              Add Local Agent
            </button>
          </div>

          {showAddAgent && (
            <AddAgentForm
              onAdd={(name, description, host, port, auth) => { addAgent(name, description, host, port, auth); setShowAddAgent(false) }}
              onCancel={() => setShowAddAgent(false)}
            />
          )}

          {agentCount > 0 ? (
            layoutMode === 'grid' ? (
              /* ── Grid layout ── */
              <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
                {unifiedAgents.map((agent) => (
                  <div key={agent.id}>{renderAgentCard(agent)}</div>
                ))}
              </div>
            ) : (
              /* ── Free layout canvas ── */
              <div
                ref={canvasRef}
                className="relative"
                style={{ minHeight: canvasHeight }}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
              >
                {unifiedAgents.map((agent) => {
                  const pos = positions[agent.id] || { x: 0, y: 0 }
                  const isDragging = dragId === agent.id
                  return (
                    <div
                      key={agent.id}
                      className={`absolute transition-shadow ${isDragging ? 'z-50 shadow-2xl shadow-violet-500/20' : 'z-10'}`}
                      style={{
                        left: pos.x,
                        top: pos.y,
                        width: CARD_W,
                        transition: isDragging ? 'none' : 'box-shadow 0.2s',
                      }}
                    >
                      {renderAgentCard(agent, (e: React.PointerEvent) => handlePointerDown(e, agent.id), isDragging)}
                    </div>
                  )
                })}
              </div>
            )
          ) : (
            !showAddAgent && (
              <p className="text-sm text-zinc-600">
                {dockerAvailable
                  ? 'No agents running. Spawn one from the Spawn Agent page, or click "Add Local Agent" to connect to an existing instance.'
                  : 'No agents registered. Click "Add Local Agent" to connect to a Captain Claw instance.'}
              </p>
            )
          )}
        </div>

        {!hasContent && (
          <div className="mt-20 text-center">
            <p className="text-zinc-500">No agents running.</p>
            <p className="mt-1 text-sm text-zinc-600">
              {dockerAvailable
                ? 'Spawn one from the Spawn Agent page, or add a local agent.'
                : 'Start a Captain Claw instance with botport enabled, or add a local agent.'}
            </p>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedInstance && (
        <AgentDetail
          instance={selectedInstance}
          concerns={concerns}
          onClose={() => selectInstance(null)}
        />
      )}

      {/* File browser modal */}
      {browsingAgent && (
        <FileBrowser
          agent={browsingAgent}
          allAgents={allAgentEndpoints}
          onClose={() => setBrowsingAgent(null)}
        />
      )}
    </div>
  )
}

function AddAgentForm({ onAdd, onCancel }: {
  onAdd: (name: string, description: string, host: string, port: number, auth: string) => void
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [host, setHost] = useState('localhost')
  const [port, setPort] = useState('23080')
  const [auth, setAuth] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const p = parseInt(port, 10)
    if (!name.trim() || !host.trim() || isNaN(p)) return
    onAdd(name.trim(), description.trim(), host.trim(), p, auth.trim())
  }

  return (
    <form onSubmit={handleSubmit} className="mb-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-500">Name</label>
          <input
            value={name} onChange={(e) => setName(e.target.value)}
            placeholder="My Agent"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            autoFocus
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-500">Description</label>
          <input
            value={description} onChange={(e) => setDescription(e.target.value)}
            placeholder="What this agent does..."
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-500">Host</label>
          <input
            value={host} onChange={(e) => setHost(e.target.value)}
            placeholder="localhost"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-500">Port</label>
          <input
            value={port} onChange={(e) => setPort(e.target.value)}
            placeholder="24080" type="number"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-500">Auth Token (optional)</label>
          <input
            value={auth} onChange={(e) => setAuth(e.target.value)}
            placeholder="secret"
            className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <button type="submit" className="rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500">
          Add Agent
        </button>
        <button type="button" onClick={onCancel} className="rounded-md px-3 py-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200">
          Cancel
        </button>
      </div>
    </form>
  )
}
