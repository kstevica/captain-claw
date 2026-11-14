import { useMemo } from 'react'
import { Check, Crown, Bot } from 'lucide-react'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import type { CouncilAgentDef } from '../../stores/councilStore'
import { AgentAvatar } from './councilLook'

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
      <div className="flex flex-col items-center rounded-xl border border-dashed border-zinc-800 px-4 py-8 text-center">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-900 text-zinc-500">
          <Bot className="h-5 w-5" />
        </span>
        <p className="mt-3 max-w-xs text-sm leading-relaxed text-zinc-500">
          No running agents found. Start agents from the Agent Desktop first.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-1.5">
      {available.map(agent => {
        const on = selectedIds.has(agent.id)
        return (
          <button
            key={agent.id}
            type="button"
            onClick={() => toggle(agent)}
            aria-pressed={on}
            className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left text-sm transition-colors ${
              on
                ? 'border-violet-500/50 bg-violet-500/[0.08]'
                : 'border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-800/60'
            }`}
          >
            <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md ${
              on ? 'bg-violet-600 text-white' : 'border border-zinc-600'
            }`}>
              {on && <Check className="h-3 w-3" />}
            </span>
            <AgentAvatar name={agent.name} id={agent.id} size="sm" />
            <span className="min-w-0 flex-1">
              <span className="flex min-w-0 items-center gap-2">
                <span className="truncate font-medium text-zinc-100">{agent.name}</span>
                {agent.isOldMan && (
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300">
                    <Crown className="h-3 w-3" /> Moderator
                  </span>
                )}
              </span>
              <span className="block text-xs text-zinc-500">{agent.source}</span>
            </span>
          </button>
        )
      })}
    </div>
  )
}

export { isOldManName }
