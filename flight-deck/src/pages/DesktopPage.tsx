import { useEffect, useState } from 'react'
import { useAgentStore } from '../stores/agentStore'
import { useContainerStore } from '../stores/containerStore'
import { useLocalAgentStore } from '../stores/localAgentStore'
import { AgentCard } from '../components/agents/AgentCard'
import { AgentDetail } from '../components/agents/AgentDetail'
import { ContainerCard } from '../components/agents/ContainerCard'
import { LocalAgentCard } from '../components/agents/LocalAgentCard'
import { FileBrowser } from '../components/agents/FileBrowser'
import { Box, Radio, Cpu, Plus } from 'lucide-react'
import type { AgentEndpoint } from '../services/fileTransfer'

export function DesktopPage() {
  const { instances, concerns, selectedInstanceId, selectInstance } = useAgentStore()
  const { containers, fetchContainers, dockerAvailable, checkHealth } = useContainerStore()
  const { agents: localAgents, addAgent, probeAll } = useLocalAgentStore()
  const selectedInstance = instances.find((i) => i.id === selectedInstanceId)
  const [showAddAgent, setShowAddAgent] = useState(false)
  const [browsingAgent, setBrowsingAgent] = useState<AgentEndpoint | null>(null)

  // Build list of all agents with web endpoints for file transfer
  const allAgentEndpoints: AgentEndpoint[] = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port)
      .map((c) => ({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })),
    ...localAgents
      .filter((a) => a.status === 'online')
      .map((a) => ({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })),
  ]

  useEffect(() => {
    checkHealth()
    fetchContainers()
    probeAll()
    const interval = setInterval(fetchContainers, 10000)
    return () => clearInterval(interval)
  }, [checkHealth, fetchContainers, probeAll])

  const hasContent = instances.length > 0 || containers.length > 0 || localAgents.length > 0

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="mb-6">
          <h1 className="text-lg font-semibold">Agent Desktop</h1>
          <p className="text-sm text-zinc-500">Monitor and control your personal assistants</p>
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

        {/* Docker containers */}
        {dockerAvailable && (
          <div className="mb-8">
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              <Box className="h-3.5 w-3.5" />
              Docker Containers ({containers.length})
            </div>
            {containers.length > 0 ? (
              <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
                {containers.map((c) => (
                  <ContainerCard key={c.id} container={c} onBrowseFiles={
                    c.status === 'running' && c.web_port
                      ? () => setBrowsingAgent({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })
                      : undefined
                  } />
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-600">No managed containers. Spawn one from the Spawn Agent page.</p>
            )}
          </div>
        )}

        {/* Local agents (pip-installed, remote, etc.) */}
        <div className="mb-8">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
              <Cpu className="h-3.5 w-3.5" />
              Local Agents ({localAgents.length})
            </div>
            <button
              onClick={() => setShowAddAgent(!showAddAgent)}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Plus className="h-3.5 w-3.5" />
              Add Agent
            </button>
          </div>

          {showAddAgent && (
            <AddAgentForm
              onAdd={(name, host, port, auth) => { addAgent(name, host, port, auth); setShowAddAgent(false) }}
              onCancel={() => setShowAddAgent(false)}
            />
          )}

          {localAgents.length > 0 && (
            <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
              {localAgents.map((a) => (
                <LocalAgentCard key={a.id} agent={a} onBrowseFiles={
                  a.status === 'online'
                    ? () => setBrowsingAgent({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })
                    : undefined
                } />
              ))}
            </div>
          )}

          {localAgents.length === 0 && !showAddAgent && (
            <p className="text-sm text-zinc-600">
              No local agents registered. Click "Add Agent" to connect to a Captain Claw instance running on this machine or remotely.
            </p>
          )}
        </div>

        {!hasContent && !dockerAvailable && (
          <div className="mt-20 text-center">
            <p className="text-zinc-500">No agents connected to BotPort.</p>
            <p className="mt-1 text-sm text-zinc-600">
              Start a Captain Claw instance with botport enabled, or spawn one from the Spawn Agent page.
            </p>
            <p className="mt-3 text-xs text-zinc-600">
              Flight Deck backend not reachable — start it to manage Docker containers directly.
            </p>
          </div>
        )}

        {!hasContent && dockerAvailable && (
          <div className="mt-20 text-center">
            <p className="text-zinc-500">No agents running.</p>
            <p className="mt-1 text-sm text-zinc-600">
              Spawn one from the Spawn Agent page.
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
  onAdd: (name: string, host: string, port: number, auth: string) => void
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [host, setHost] = useState('localhost')
  const [port, setPort] = useState('24080')
  const [auth, setAuth] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const p = parseInt(port, 10)
    if (!name.trim() || !host.trim() || isNaN(p)) return
    onAdd(name.trim(), host.trim(), p, auth.trim())
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
