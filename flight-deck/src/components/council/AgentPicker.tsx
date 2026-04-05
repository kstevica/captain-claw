import { useMemo } from 'react'
import { Check, Crown } from 'lucide-react'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import type { CouncilAgentDef } from '../../stores/councilStore'

interface PickableAgent {
  id: string
  name: string
  host: string
  port: number
  auth: string
  source: 'container' | 'process' | 'local'
  isOldMan: boolean
}

interface AgentPickerProps {
  selected: CouncilAgentDef[]
  onChange: (agents: CouncilAgentDef[]) => void
}

function isOldManName(name: string): boolean {
  const n = name.toLowerCase()
  return n.includes('old-man') || n.includes('old man') || n.includes('oldman')
}

export function AgentPicker({ selected, onChange }: AgentPickerProps) {
  const containers = useContainerStore(s => s.containers)
  const processes = useProcessStore(s => s.processes)
  const localAgents = useLocalAgentStore(s => s.agents)

  const available = useMemo<PickableAgent[]>(() => {
    const agents: PickableAgent[] = []

    for (const c of containers) {
      if (!c.web_port || c.status !== 'running') continue
      agents.push({
        id: c.id,
        name: c.agent_name || c.name,
        host: 'localhost',
        port: c.web_port,
        auth: c.web_auth || '',
        source: 'container',
        isOldMan: isOldManName(c.agent_name || c.name),
      })
    }

    for (const p of processes) {
      if (!p.web_port || p.status !== 'running') continue
      agents.push({
        id: p.slug,
        name: p.name,
        host: 'localhost',
        port: p.web_port,
        auth: p.web_auth || '',
        source: 'process',
        isOldMan: isOldManName(p.name),
      })
    }

    for (const la of localAgents) {
      if (la.status !== 'online') continue
      agents.push({
        id: la.id,
        name: la.name,
        host: la.host,
        port: la.port,
        auth: la.authToken || '',
        source: 'local',
        isOldMan: isOldManName(la.name),
      })
    }

    return agents
  }, [containers, processes, localAgents])

  const selectedIds = new Set(selected.map(a => a.id))

  const toggle = (agent: PickableAgent) => {
    if (selectedIds.has(agent.id)) {
      onChange(selected.filter(a => a.id !== agent.id))
    } else {
      onChange([...selected, {
        id: agent.id, name: agent.name, host: agent.host,
        port: agent.port, auth: agent.auth, muted: false,
      }])
    }
  }

  if (available.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-700/50 bg-zinc-800/50 p-4 text-center text-sm text-zinc-500">
        No running agents found. Start agents from the Agent Desktop first.
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {available.map(agent => (
        <button
          key={agent.id}
          onClick={() => toggle(agent)}
          className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
            selectedIds.has(agent.id)
              ? 'bg-violet-500/20 border border-violet-500/40'
              : 'bg-zinc-800/50 border border-zinc-700/30 hover:bg-zinc-700/50'
          }`}
        >
          <div className={`flex h-5 w-5 shrink-0 items-center justify-center rounded ${
            selectedIds.has(agent.id) ? 'bg-violet-500 text-white' : 'border border-zinc-600'
          }`}>
            {selectedIds.has(agent.id) && <Check className="h-3 w-3" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium text-zinc-200 truncate">{agent.name}</span>
              {agent.isOldMan && (
                <span className="flex items-center gap-1 text-xs text-amber-400">
                  <Crown className="h-3 w-3" /> Moderator
                </span>
              )}
              <span className="text-xs text-zinc-500">{agent.source}</span>
            </div>
          </div>
        </button>
      ))}
    </div>
  )
}

export { isOldManName }
