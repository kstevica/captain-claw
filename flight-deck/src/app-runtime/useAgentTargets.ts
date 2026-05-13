// Aggregate the three flavours of Captain Claw agents (docker containers,
// local processes, remote local agents) into a single AgentTarget list.
// Mirrors the inline logic in TodayPage / SkillsPage so anywhere that needs
// to dispatch through an agent's LLM has the same picker source.

import { useMemo } from 'react'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useLocalAgentStore } from '../stores/localAgentStore'

export interface AgentTarget {
  id: string
  name: string
  kind: 'docker' | 'process' | 'local'
  host: string
  port: number
  auth: string
  model?: string
}

export function useAgentTargets(): AgentTarget[] {
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const localAgents = useLocalAgentStore((s) => s.agents)

  return useMemo<AgentTarget[]>(() => {
    const out: AgentTarget[] = []
    for (const c of containers) {
      if (c.status !== 'running' || !c.web_port) continue
      out.push({
        id: c.id,
        name: c.agent_name || c.name,
        kind: 'docker',
        host: 'localhost',
        port: c.web_port,
        auth: c.web_auth || '',
      })
    }
    for (const p of processes) {
      if (p.status !== 'running' || !p.web_port) continue
      out.push({
        id: p.slug,
        name: p.name || p.slug,
        kind: 'process',
        host: 'localhost',
        port: p.web_port,
        auth: p.web_auth || '',
        model: p.model || undefined,
      })
    }
    for (const a of localAgents) {
      if (a.status && a.status !== 'online') continue
      out.push({
        id: a.id,
        name: a.name,
        kind: 'local',
        host: a.host,
        port: a.port,
        auth: a.authToken || '',
      })
    }
    return out
  }, [containers, processes, localAgents])
}
