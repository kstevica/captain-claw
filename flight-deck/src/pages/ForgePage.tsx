import { useState, useEffect } from 'react'
import { Wand2, Loader2, Trash2, Plus, Rocket, ChevronDown, ChevronRight, Check, AlertTriangle, Cpu, Server, Crown } from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useGroupStore } from '../stores/groupStore'
import { useUIStore } from '../stores/uiStore'
import { spawnAgent, spawnProcess, type SpawnConfig } from '../services/docker'
import { queueSave, registerHydrator } from '../services/settingsSync'
import { refreshAccessToken } from '../stores/authStore'

// ── LLM config persistence ──

const FORGE_CONFIG_KEY = 'fd:forge-llm-config'

interface ForgeLLMConfig {
  provider: string
  model: string
  api_key: string
}

function loadForgeConfig(): ForgeLLMConfig {
  try {
    return JSON.parse(localStorage.getItem(FORGE_CONFIG_KEY) || '{}')
  } catch {
    return { provider: 'anthropic', model: 'claude-sonnet-4-20250514', api_key: '' }
  }
}

function saveForgeConfig(config: ForgeLLMConfig) {
  const val = JSON.stringify(config)
  if (useAuthStore.getState().authEnabled) queueSave(FORGE_CONFIG_KEY, val)
  else localStorage.setItem(FORGE_CONFIG_KEY, val)
}

registerHydrator((settings) => {
  const raw = settings[FORGE_CONFIG_KEY]
  if (raw) {
    try {
      const parsed = JSON.parse(raw)
      localStorage.setItem(FORGE_CONFIG_KEY, JSON.stringify(parsed))
    } catch { /* ignore */ }
  }
})

// ── Types ──

interface AgentProposal {
  id: string
  name: string
  role: string
  lead: boolean
  description: string
  fleet_instructions: string
  tools: string[]
  type: 'process' | 'docker'
  provider: string
  model: string
  providerApiKey: string
}

type Phase = 'input' | 'review' | 'spawning' | 'done'

const PROVIDERS = ['anthropic', 'openai', 'ollama', 'gemini', 'xai', 'openrouter']

const DEFAULT_TOOLS = [
  'shell', 'read', 'write', 'glob', 'edit', 'web_fetch', 'web_search',
  'personality', 'playbooks', 'scripts',
]

// ── API call ──

async function callForge(prompt: string, provider: string, model: string, apiKey: string) {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  let res = await fetch('/fd/forge', {
    method: 'POST',
    headers,
    credentials: 'include',
    body: JSON.stringify({ prompt, provider, model, api_key: apiKey }),
  })
  if (res.status === 401 && authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2: Record<string, string> = { 'Content-Type': 'application/json' }
      const t2 = useAuthStore.getState().token
      if (t2) h2['Authorization'] = `Bearer ${t2}`
      res = await fetch('/fd/forge', {
        method: 'POST',
        headers: h2,
        credentials: 'include',
        body: JSON.stringify({ prompt, provider, model, api_key: apiKey }),
      })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Component ──

export function ForgePage() {
  const [phase, setPhase] = useState<Phase>('input')
  const [prompt, setPrompt] = useState('')
  const [llmConfig, setLlmConfig] = useState<ForgeLLMConfig>(() => {
    const saved = loadForgeConfig()
    return {
      provider: saved.provider || 'anthropic',
      model: saved.model || 'claude-sonnet-4-20250514',
      api_key: saved.api_key || '',
    }
  })
  const [showSettings, setShowSettings] = useState(false)
  const [envVars, setEnvVars] = useState<{ key: string; value: string }[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('fd:forge-env-vars') || '[]')
      return Array.isArray(saved) ? saved : []
    } catch { return [] }
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [teamName, setTeamName] = useState('')
  const [summary, setSummary] = useState('')
  const [agents, setAgents] = useState<AgentProposal[]>([])
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)

  // Spawning state
  const [spawnProgress, setSpawnProgress] = useState<Record<string, 'pending' | 'spawning' | 'done' | 'error'>>({})
  const [spawnErrors, setSpawnErrors] = useState<Record<string, string>>({})

  const { dockerAvailable, fetchContainers } = useContainerStore()
  const { fetchProcesses } = useProcessStore()
  const { setFleetInstructions: setContainerFleetInst } = useContainerStore()
  const { setFleetInstructions: setProcessFleetInst } = useProcessStore()
  const { setDescription: setContainerDesc, setNameOverride: setContainerName } = useContainerStore()
  const { setDescription: setProcessDesc, setNameOverride: setProcessName } = useProcessStore()
  const { createGroup, addToGroup, groups } = useGroupStore()
  const { setView } = useUIStore()

  useEffect(() => {
    saveForgeConfig(llmConfig)
  }, [llmConfig])

  useEffect(() => {
    const val = JSON.stringify(envVars)
    if (useAuthStore.getState().authEnabled) queueSave('fd:forge-env-vars', val)
    else localStorage.setItem('fd:forge-env-vars', val)
  }, [envVars])

  const handleDecompose = async () => {
    if (!prompt.trim()) return
    setLoading(true)
    setError('')
    try {
      const result = await callForge(prompt, llmConfig.provider, llmConfig.model, llmConfig.api_key)
      setTeamName(result.team_name || 'Agent Team')
      setSummary(result.summary || '')
      const proposals: AgentProposal[] = (result.agents || []).map((a: any, i: number) => ({
        id: `forge-${Date.now()}-${i}`,
        name: a.name || `agent-${i + 1}`,
        role: a.role || 'Agent',
        lead: a.lead === true,
        description: a.description || '',
        fleet_instructions: a.fleet_instructions || '',
        tools: a.tools || DEFAULT_TOOLS,
        type: 'process' as const,
        provider: llmConfig.provider,
        model: llmConfig.model,
        providerApiKey: llmConfig.api_key,
      }))
      setAgents(proposals)
      setPhase('review')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const updateAgent = (id: string, patch: Partial<AgentProposal>) => {
    setAgents((prev) => prev.map((a) => a.id === id ? { ...a, ...patch } : a))
  }

  const removeAgent = (id: string) => {
    setAgents((prev) => prev.filter((a) => a.id !== id))
  }

  const toggleLead = (id: string) => {
    setAgents((prev) => prev.map((a) => ({ ...a, lead: a.id === id })))
  }

  const addAgent = () => {
    setAgents((prev) => [...prev, {
      id: `forge-${Date.now()}-new`,
      name: 'new-agent',
      role: 'Agent',
      lead: false,
      description: '',
      fleet_instructions: '',
      tools: DEFAULT_TOOLS,
      type: 'process',
      provider: llmConfig.provider,
      model: llmConfig.model,
      providerApiKey: llmConfig.api_key,
    }])
  }

  const handleSpawnAll = async () => {
    setPhase('spawning')
    const progress: Record<string, 'pending' | 'spawning' | 'done' | 'error'> = {}
    const errors: Record<string, string> = {}
    for (const a of agents) progress[a.id] = 'pending'
    setSpawnProgress({ ...progress })
    setSpawnErrors({})

    // Create groups: one for the team, one per unique role
    const groupMap: Record<string, string> = {} // group name → group ID

    // Team group
    const teamGroupName = teamName || 'Agent Team'
    const existingTeamGroup = groups.find((g) => g.name === teamGroupName)
    groupMap[teamGroupName] = existingTeamGroup ? existingTeamGroup.id : createGroup(teamGroupName)

    // Role groups
    const uniqueRoles = [...new Set(agents.map((a) => a.role))]
    for (const role of uniqueRoles) {
      const existing = groups.find((g) => g.name === role)
      groupMap[role] = existing ? existing.id : createGroup(role)
    }

    for (const agent of agents) {
      progress[agent.id] = 'spawning'
      setSpawnProgress({ ...progress })

      const payload: SpawnConfig = {
        name: agent.name,
        description: agent.description,
        hostname: 'captain-claw',
        image: 'kstevica/captain-claw:latest',
        provider: agent.provider,
        model: agent.model,
        temperature: 0.7,
        max_tokens: 32768,
        provider_api_key: agent.providerApiKey,
        botport_enabled: false,
        botport_url: '',
        botport_instance_name: '',
        botport_key: '',
        botport_secret: '',
        botport_max_concurrent: 5,
        tools: agent.tools,
        web_enabled: true,
        web_port: 0,
        web_auth_token: '',
        telegram_enabled: false,
        telegram_bot_token: '',
        discord_enabled: false,
        discord_bot_token: '',
        slack_enabled: false,
        slack_bot_token: '',
        network_mode: 'host',
        restart_policy: 'unless-stopped',
        extra_volumes: [],
        env_vars: envVars.filter((ev) => ev.key.trim() && ev.value.trim()),
      }

      try {
        let agentStoreId: string
        if (agent.type === 'process') {
          const result = await spawnProcess(payload)
          if (!result.ok) throw new Error(result.message)
          const slug = agent.name.replace(/[^a-z0-9-]/gi, '-').toLowerCase()
          agentStoreId = `proc-${slug}`
          setProcessFleetInst(slug, agent.fleet_instructions)
          setProcessDesc(slug, agent.description)
          const displayName = agent.lead ? `${agent.name} (${agent.role}) [Lead]` : `${agent.name} (${agent.role})`
          setProcessName(slug, displayName)
        } else {
          if (!dockerAvailable) throw new Error('Docker not available')
          const result = await spawnAgent(payload)
          if (!result.ok) throw new Error(result.message)
          agentStoreId = result.container_id
          setContainerFleetInst(agentStoreId, agent.fleet_instructions)
          setContainerDesc(agentStoreId, agent.description)
          const displayName = agent.lead ? `${agent.name} (${agent.role}) [Lead]` : `${agent.name} (${agent.role})`
          setContainerName(agentStoreId, displayName)
        }

        // Add to team group
        addToGroup(groupMap[teamGroupName], agentStoreId)
        // Add to role group
        if (groupMap[agent.role]) {
          addToGroup(groupMap[agent.role], agentStoreId)
        }

        progress[agent.id] = 'done'
      } catch (e) {
        progress[agent.id] = 'error'
        errors[agent.id] = e instanceof Error ? e.message : String(e)
      }
      setSpawnProgress({ ...progress })
      setSpawnErrors({ ...errors })
    }

    fetchContainers()
    fetchProcesses()
    setPhase('done')
  }

  // ── Render ──

  const leadAgent = agents.find((a) => a.lead)

  return (
    <div className="flex-1 overflow-auto p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Wand2 className="h-5 w-5 text-violet-400" /> Agent Forge
        </h1>
        <p className="text-xs text-zinc-500 sm:text-sm">Describe your objective and let AI design a team of specialized agents</p>
      </div>

      {/* Phase 1: Input */}
      {phase === 'input' && (
        <div className="space-y-4">
          {/* LLM Settings */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-zinc-300 hover:bg-zinc-800/50"
            >
              <span>LLM Settings — {llmConfig.provider}/{llmConfig.model}</span>
              {showSettings ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </button>
            {showSettings && (
              <div className="border-t border-zinc-800 px-4 py-3 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] font-medium text-zinc-500 mb-1">Provider</label>
                    <select
                      value={llmConfig.provider}
                      onChange={(e) => setLlmConfig({ ...llmConfig, provider: e.target.value })}
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    >
                      {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-zinc-500 mb-1">Model</label>
                    <input
                      value={llmConfig.model}
                      onChange={(e) => setLlmConfig({ ...llmConfig, model: e.target.value })}
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                      placeholder="claude-sonnet-4-20250514"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-zinc-500 mb-1">API Key</label>
                  <input
                    type="password"
                    value={llmConfig.api_key}
                    onChange={(e) => setLlmConfig({ ...llmConfig, api_key: e.target.value })}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    placeholder="sk-..."
                  />
                </div>

                {/* Additional API Keys / Environment Variables */}
                <div>
                  <label className="block text-[11px] font-medium text-zinc-500 mb-1">Additional API Keys (passed to all agents)</label>
                  <div className="space-y-2">
                    {envVars.map((ev, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <input
                          value={ev.key}
                          onChange={(e) => { const next = [...envVars]; next[i] = { ...next[i], key: e.target.value }; setEnvVars(next) }}
                          className="w-48 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="BRAVE_API_KEY"
                        />
                        <input
                          type="password"
                          value={ev.value}
                          onChange={(e) => { const next = [...envVars]; next[i] = { ...next[i], value: e.target.value }; setEnvVars(next) }}
                          className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="value"
                        />
                        <button
                          onClick={() => setEnvVars(envVars.filter((_, j) => j !== i))}
                          className="rounded p-1 text-zinc-600 hover:text-red-400 hover:bg-zinc-800"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    ))}
                    <button
                      onClick={() => setEnvVars([...envVars, { key: '', value: '' }])}
                      className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300"
                    >
                      <Plus className="h-3 w-3" /> Add variable
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Prompt Input */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
            <label className="block text-sm font-medium text-zinc-300">Describe your objective</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-4 py-3 text-sm text-zinc-200 leading-relaxed focus:border-violet-500/50 focus:outline-none resize-none placeholder-zinc-600"
              placeholder="Example: I need a team to research competitors in the SaaS market, analyze their pricing strategies, create a comparison report, and draft marketing copy for our product launch."
            />
            <div className="flex items-center gap-3">
              <button
                onClick={handleDecompose}
                disabled={loading || !prompt.trim()}
                className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500 transition-colors disabled:opacity-40"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
                {loading ? 'Decomposing...' : 'Decompose into Agents'}
              </button>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
              <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
            </div>
          )}
        </div>
      )}

      {/* Phase 2: Review & Edit */}
      {phase === 'review' && (
        <div className="space-y-4">
          {/* Team Summary */}
          <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <label className="text-[11px] font-medium text-zinc-500 shrink-0">Team:</label>
                <input
                  value={teamName}
                  onChange={(e) => setTeamName(e.target.value)}
                  className="flex-1 min-w-0 text-base font-semibold bg-transparent border-none focus:outline-none text-zinc-200 placeholder-zinc-600"
                  placeholder="Team Name"
                />
              </div>
              <span className="text-xs text-zinc-500">{agents.length} agents</span>
            </div>
            {summary && <p className="text-xs text-zinc-400 leading-relaxed">{summary}</p>}
            {leadAgent && (
              <p className="text-[11px] text-violet-400 mt-2">
                <Crown className="inline h-3 w-3 mr-1" />
                Lead agent: <strong>{leadAgent.name}</strong> ({leadAgent.role})
              </p>
            )}
          </div>

          {/* Agent Cards */}
          <div className="space-y-3">
            {agents.map((agent) => {
              const isExpanded = expandedAgent === agent.id
              return (
                <div key={agent.id} className={`rounded-xl border overflow-hidden ${agent.lead ? 'border-amber-500/60 bg-amber-500/10 shadow-sm shadow-amber-500/10' : 'border-zinc-800 bg-zinc-900/50'}`}>
                  {/* Card Header */}
                  <div
                    className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-zinc-800/30"
                    onClick={() => setExpandedAgent(isExpanded ? null : agent.id)}
                  >
                    {isExpanded ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        {agent.lead && <Crown className="h-3.5 w-3.5 text-amber-400 shrink-0" />}
                        <span className="text-sm font-medium text-zinc-200 truncate">{agent.name}</span>
                        <span className="rounded-full bg-blue-600/20 border border-blue-500/30 px-2 py-0.5 text-[10px] font-semibold text-blue-400 shrink-0">{teamName || 'Team'}</span>
                        <span className="rounded-full bg-violet-600/20 border border-violet-500/30 px-2 py-0.5 text-[10px] font-semibold text-violet-400 shrink-0">{agent.role}</span>
                        {agent.lead && <span className="rounded-full bg-amber-600/20 border border-amber-500/30 px-2 py-0.5 text-[10px] font-semibold text-amber-400 shrink-0">Lead</span>}
                      </div>
                      <p className="text-[11px] text-zinc-500 truncate mt-0.5">{agent.description}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${agent.type === 'docker' ? 'bg-blue-600/20 border border-blue-500/30 text-blue-400' : 'bg-emerald-600/20 border border-emerald-500/30 text-emerald-400'}`}>
                        {agent.type === 'docker' ? <Server className="h-2.5 w-2.5" /> : <Cpu className="h-2.5 w-2.5" />}
                        {agent.type}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleLead(agent.id) }}
                        className={`rounded p-1 transition-colors ${agent.lead ? 'text-amber-400 bg-amber-500/10' : 'text-zinc-600 hover:text-amber-400 hover:bg-zinc-800'}`}
                        title={agent.lead ? 'Lead agent' : 'Set as lead'}
                      >
                        <Crown className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeAgent(agent.id) }}
                        className="rounded p-1 text-zinc-600 hover:text-red-400 hover:bg-zinc-800"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Expanded Editor */}
                  {isExpanded && (
                    <div className="border-t border-zinc-800 px-4 py-3 space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Name</label>
                          <input
                            value={agent.name}
                            onChange={(e) => updateAgent(agent.id, { name: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Role</label>
                          <input
                            value={agent.role}
                            onChange={(e) => updateAgent(agent.id, { role: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Description</label>
                          <input
                            value={agent.description}
                            onChange={(e) => updateAgent(agent.id, { description: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Type</label>
                          <select
                            value={agent.type}
                            onChange={(e) => updateAgent(agent.id, { type: e.target.value as 'process' | 'docker' })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          >
                            <option value="process">Process</option>
                            {dockerAvailable && <option value="docker">Docker</option>}
                          </select>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Provider</label>
                          <select
                            value={agent.provider}
                            onChange={(e) => updateAgent(agent.id, { provider: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          >
                            {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Model</label>
                          <input
                            value={agent.model}
                            onChange={(e) => updateAgent(agent.id, { model: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-zinc-500 mb-1">API Key (for this agent)</label>
                        <input
                          type="password"
                          value={agent.providerApiKey}
                          onChange={(e) => updateAgent(agent.id, { providerApiKey: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="Inherited from forge settings"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-zinc-500 mb-1">Fleet Instructions</label>
                        <textarea
                          value={agent.fleet_instructions}
                          onChange={(e) => updateAgent(agent.id, { fleet_instructions: e.target.value })}
                          rows={8}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-[13px] font-mono text-zinc-300 leading-relaxed focus:border-violet-500/50 focus:outline-none resize-none"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={addAgent}
              className="flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-300 hover:bg-zinc-800 transition-colors"
            >
              <Plus className="h-4 w-4" /> Add Agent
            </button>
            <div className="flex-1" />
            <button
              onClick={() => { setPhase('input'); setAgents([]); setError('') }}
              className="flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-400 hover:bg-zinc-800 transition-colors"
            >
              Back
            </button>
            <button
              onClick={handleSpawnAll}
              disabled={agents.length === 0}
              className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500 transition-colors disabled:opacity-40"
            >
              <Rocket className="h-4 w-4" /> Spawn {agents.length} Agent{agents.length !== 1 ? 's' : ''}
            </button>
          </div>
        </div>
      )}

      {/* Phase 3: Spawning Progress */}
      {(phase === 'spawning' || phase === 'done') && (
        <div className="max-w-3xl space-y-4">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <h2 className="text-sm font-semibold text-zinc-200 mb-3">
              {phase === 'spawning' ? 'Spawning agents...' : 'Team deployed!'}
            </h2>
            <div className="space-y-2">
              {agents.map((agent) => {
                const status = spawnProgress[agent.id] || 'pending'
                const err = spawnErrors[agent.id]
                return (
                  <div key={agent.id} className="flex items-center gap-3 rounded-lg bg-zinc-950/50 px-3 py-2">
                    {status === 'pending' && <div className="h-4 w-4 rounded-full border-2 border-zinc-700" />}
                    {status === 'spawning' && <Loader2 className="h-4 w-4 animate-spin text-violet-400" />}
                    {status === 'done' && <Check className="h-4 w-4 text-emerald-400" />}
                    {status === 'error' && <AlertTriangle className="h-4 w-4 text-red-400" />}
                    <div className="flex-1 min-w-0">
                      <span className="text-sm text-zinc-300">
                        {agent.lead && <Crown className="inline h-3 w-3 text-amber-400 mr-1" />}
                        {agent.name}
                      </span>
                      <span className="ml-2 text-[10px] text-zinc-500">{agent.role}</span>
                    </div>
                    {err && <span className="text-[11px] text-red-400 truncate max-w-[200px]">{err}</span>}
                  </div>
                )
              })}
            </div>
          </div>

          {phase === 'done' && (
            <div className="flex items-center gap-3">
              <button
                onClick={() => setView('desktop')}
                className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500 transition-colors"
              >
                Go to Agent Desktop
              </button>
              <button
                onClick={() => { setPhase('input'); setAgents([]); setPrompt(''); setError('') }}
                className="flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-400 hover:bg-zinc-800 transition-colors"
              >
                Forge Another Team
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
