import { useState, useEffect, useMemo } from 'react'
import { Wand2, Loader2, Trash2, Plus, Rocket, ChevronDown, ChevronRight, Check, AlertTriangle, Cpu, Server, Crown, FolderKanban, Gauge, Boxes, Sparkles } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { useContainerStore } from '../stores/containerStore'
import { useProcessStore } from '../stores/processStore'
import { useGroupStore } from '../stores/groupStore'
import { useUIStore } from '../stores/uiStore'
import { useOnboardingStore } from '../stores/onboardingStore'
import { HelpHint } from '../components/common/HelpHint'
import { spawnAgent, spawnProcess, type SpawnConfig } from '../services/docker'
import { updateArchetype, type ArchetypeInput } from '../services/archetypes'
import { useTierConfig, PROVIDERS, TIER_ORDER, DEFAULT_TOOLS, type Archetype } from '../services/tierConfig'

// ── Types ──

// A full reusable archetype the decomposition invented for a bespoke agent
// (no catalog entry fit). Saved to the user's Library on spawn (Phase 3).
interface NewArchetypeDef {
  role?: string
  family?: string
  description?: string
  cognitive_mode?: string
  tier?: string
  tools?: string[]
  keywords?: string[]
  fleet_instructions?: string
}

interface AgentProposal {
  id: string
  name: string
  role: string
  lead: boolean
  description: string
  tools: string[]
  cognitive_mode: string
  type: 'process' | 'docker'
  provider: string
  model: string
  providerApiKey: string
  // Model-recommendation tier. When set, the backend resolves provider/model
  // from the central tier table; cleared when the user pins a model explicitly.
  tier?: string
  // ── Archetype-base composition ──
  // The base archetype id this agent is built on ('' = bespoke). When set it
  // resolves against the registry; baseInstructions is its SOP snapshot.
  archetype: string
  // Snapshot of the base archetype's SOP (from the catalog entry or newArchetype).
  baseInstructions: string
  // Task-specific delta layered on top of the base SOP.
  additional_instructions: string
  // Non-empty = the user hand-edited the full instructions; used verbatim and
  // supersedes the base+additional composition.
  fleetOverride: string
  // Inline archetype definition for a bespoke agent; drives auto-mint (Phase 3).
  newArchetype: NewArchetypeDef | null
}

type Phase = 'input' | 'review' | 'spawning' | 'done'

// Final system instructions sent to a spawned agent: a manual full-text override
// wins; otherwise the archetype base SOP with the task-specific delta appended.
function composeFleet(a: AgentProposal): string {
  if (a.fleetOverride.trim()) return a.fleetOverride
  const base = a.baseInstructions.trim()
  const extra = a.additional_instructions.trim()
  if (base && extra) return `${base}\n\n## Task-Specific Instructions\n\n${extra}`
  return base || extra
}

// ── Archetype minting (Phase 3) ──
// The backend archetype validator only accepts these tiers/modes (coding/vision
// are catalog-only), so a new_archetype's values are clamped before upsert.
const _MINT_VALID_TIERS = new Set(['reason', 'balanced', 'fast', 'longctx'])
const _MINT_TIER_FALLBACK: Record<string, string> = { coding: 'reason', vision: 'balanced' }
const _MINT_VALID_MODES = new Set(['ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian', 'neutra'])

// Mirror the backend _slugify so ids we choose match what it would store.
function slugifyId(s: string): string {
  return (s || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}
function clampArchTier(t?: string): string {
  const v = (t || '').trim()
  return _MINT_VALID_TIERS.has(v) ? v : (_MINT_TIER_FALLBACK[v] || 'balanced')
}
function clampArchMode(m?: string): string {
  const v = (m || '').trim()
  return _MINT_VALID_MODES.has(v) ? v : 'neutra'
}

// ── API call ──

async function callForge(prompt: string, provider: string, model: string, apiKey: string, baseUrl: string, maxTokens: number, projectId?: string) {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`

  const payload: Record<string, string | number> = { prompt, provider, model, api_key: apiKey, base_url: baseUrl, max_tokens: maxTokens }
  if (projectId) payload.project_id = projectId

  let res = await fetch('/fd/forge', {
    method: 'POST',
    headers,
    credentials: 'include',
    body: JSON.stringify(payload),
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
        body: JSON.stringify(payload),
      })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

interface SimpleProject {
  id: string
  name: string
  status: string
}

// ── Component ──

export function ForgePage() {
  const [phase, setPhase] = useState<Phase>('input')
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [teamName, setTeamName] = useState('')
  const [summary, setSummary] = useState('')
  const [agents, setAgents] = useState<AgentProposal[]>([])
  // Card state keyed by agent id (Sets) so many cards can be open at once
  // (Expand all) without cross-talk: which are expanded, which have their base-
  // SOP or full-text editors open, and which are selected for bulk actions.
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [baseSopOpen, setBaseSopOpen] = useState<Set<string>>(new Set())
  const [fullEditOpen, setFullEditOpen] = useState<Set<string>>(new Set())
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const toggleInSet = (
    setter: (updater: (prev: Set<string>) => Set<string>) => void, id: string,
  ) => setter((prev) => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n })
  const removeFromSet = (
    setter: (updater: (prev: Set<string>) => Set<string>) => void, id: string,
  ) => setter((prev) => { const n = new Set(prev); n.delete(id); return n })

  // Per-user tier config + archetype registry. Edited on the Library page; Forge
  // consumes them (run the decomposition, resolve models at spawn) read-only.
  const { tiers, forgeTier, envVars, registry, refreshRegistry } = useTierConfig()

  // Project selector
  const [projects, setProjects] = useState<SimpleProject[]>([])
  const forgeProjectId = useUIStore((s) => s.forgeProjectId)
  const setForgeProjectId = useUIStore((s) => s.setForgeProjectId)
  const [selectedProjectId, setSelectedProjectId] = useState(forgeProjectId)

  // Spawning state
  const [spawnProgress, setSpawnProgress] = useState<Record<string, 'pending' | 'spawning' | 'done' | 'error'>>({})
  const [spawnErrors, setSpawnErrors] = useState<Record<string, string>>({})
  // Count of invented archetypes saved to the Library this spawn (Phase 3).
  const [mintedCount, setMintedCount] = useState(0)

  const { dockerAvailable, fetchContainers } = useContainerStore()
  const { fetchProcesses } = useProcessStore()
  const { setFleetInstructions: setContainerFleetInst } = useContainerStore()
  const { setFleetInstructions: setProcessFleetInst } = useProcessStore()
  const { setDescription: setContainerDesc, setNameOverride: setContainerName } = useContainerStore()
  const { setDescription: setProcessDesc, setNameOverride: setProcessName } = useProcessStore()
  const { createGroup, addToGroup, groups } = useGroupStore()
  const { setView } = useUIStore()
  const onboarding = useOnboardingStore()

  useEffect(() => {
    fetch('/fd/projects').then((r) => r.json()).then((list) => {
      setProjects((list || []).filter((p: SimpleProject) => p.status === 'active'))
    }).catch(() => {})
    // Clear the forge-from-project trigger
    if (forgeProjectId) setForgeProjectId('')
  }, [])

  // Editing provider/model pins the agent to a custom model (clears its tier).
  // Carry the tier's API key forward so a manual edit doesn't silently drop auth.
  const pinAgentModel = (agent: AgentProposal, patch: Partial<AgentProposal>) => {
    const next: Partial<AgentProposal> = { tier: '', ...patch }
    const tc = agent.tier ? tiers[agent.tier] : undefined
    if (tc && !agent.providerApiKey) next.providerApiKey = tc.api_key
    updateAgent(agent.id, next)
  }

  const handleDecompose = async () => {
    if (!prompt.trim()) return
    const ft = tiers[forgeTier]
    if (!ft || !ft.model.trim()) {
      setError(`Configure the "${forgeTier}" tier model in the Library page before forging.`)
      return
    }
    setLoading(true)
    setError('')
    setMintedCount(0)
    setSelectedIds(new Set())
    setExpandedIds(new Set())
    try {
      const result = await callForge(prompt, ft.provider, ft.model, ft.api_key, ft.base_url, ft.output_ctx > 0 ? ft.output_ctx : 32768, selectedProjectId || undefined)
      setTeamName(result.team_name || 'Agent Team')
      setSummary(result.summary || '')
      const proposals: AgentProposal[] = (result.agents || []).map((a: any, i: number) => {
        // Each agent is built on an archetype: a catalog entry (resolved from the
        // registry) or an inline new_archetype the model invented. The base SOP,
        // tools, cognitive_mode, and tier are inherited from it unless the model
        // set an explicit per-agent override. provider/model resolve from the
        // user's tier config at spawn.
        const archId = typeof a.archetype === 'string' ? a.archetype : ''
        const arch = archId ? registry?.archetypes.find((x) => x.id === archId) : undefined
        const na = (a.new_archetype && typeof a.new_archetype === 'object') ? a.new_archetype as NewArchetypeDef : null
        const base = arch || na || undefined

        const baseInstructions = arch?.fleet_instructions || na?.fleet_instructions || a.fleet_instructions || ''
        const overrideTools = Array.isArray(a.tools) && a.tools.length ? a.tools : null
        const tools = overrideTools || (base?.tools && base.tools.length ? base.tools : DEFAULT_TOOLS)
        const cognitive_mode = a.cognitive_mode || base?.cognitive_mode || 'neutra'
        const rawTier = a.tier || base?.tier || 'balanced'
        const tier = (typeof rawTier === 'string' && tiers[rawTier]) ? rawTier : 'balanced'
        const tc = tiers[tier]
        return {
          id: `forge-${Date.now()}-${i}`,
          name: a.name || `agent-${i + 1}`,
          role: a.role || 'Agent',
          lead: a.lead === true,
          description: a.description || '',
          tools,
          cognitive_mode,
          type: 'process' as const,
          provider: tc?.provider || '',
          model: tc?.model || '',
          providerApiKey: '',
          tier,
          archetype: arch ? archId : '',
          baseInstructions,
          additional_instructions: a.additional_instructions || '',
          fleetOverride: '',
          newArchetype: arch ? null : na,
        }
      })
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
    const tc = tiers['balanced']
    setAgents((prev) => [...prev, {
      id: `forge-${Date.now()}-new`,
      name: 'new-agent',
      role: 'Agent',
      lead: false,
      description: '',
      tools: DEFAULT_TOOLS,
      cognitive_mode: 'neutra',
      type: 'process',
      provider: tc?.provider || '',
      model: tc?.model || '',
      providerApiKey: '',
      tier: 'balanced',
      archetype: '',
      baseInstructions: '',
      additional_instructions: '',
      fleetOverride: '',
      newArchetype: null,
    }])
  }

  // Archetype catalog grouped by family for the base-archetype dropdown.
  const archetypesByFamily = useMemo(() => {
    const groups: Record<string, Archetype[]> = {}
    for (const a of (registry?.archetypes || [])) {
      const fam = a.family || 'Other'
      ;(groups[fam] ||= []).push(a)
    }
    return groups
  }, [registry])

  // The state patch that rebases an agent onto a catalog archetype (or clears to
  // bespoke). Inherits the archetype's SOP, tools, cognitive_mode, and tier;
  // keeps the task-specific additional_instructions; drops any full-text override.
  const archetypePatch = (agent: AgentProposal, id: string): Partial<AgentProposal> => {
    if (!id) return { archetype: '' }
    const a = registry?.archetypes.find((x) => x.id === id)
    if (!a) return { archetype: '' }
    const tier = tiers[a.tier] ? a.tier : (agent.tier || 'balanced')
    const tc = tiers[tier]
    return {
      archetype: id,
      baseInstructions: a.fleet_instructions || '',
      tools: a.tools && a.tools.length ? a.tools : agent.tools,
      cognitive_mode: a.cognitive_mode || agent.cognitive_mode,
      newArchetype: null,
      fleetOverride: '',
      tier,
      provider: tc?.provider || agent.provider,
      model: tc?.model || agent.model,
    }
  }
  const applyArchetype = (agent: AgentProposal, id: string) => updateAgent(agent.id, archetypePatch(agent, id))

  // ── Bulk actions across selected cards ──
  const allSelected = agents.length > 0 && selectedIds.size === agents.length
  const toggleSelectAll = () => setSelectedIds(allSelected ? new Set() : new Set(agents.map((a) => a.id)))
  const expandAll = () => setExpandedIds(new Set(agents.map((a) => a.id)))
  const collapseAll = () => setExpandedIds(new Set())
  const bulkPatch = (patch: Partial<AgentProposal>) =>
    setAgents((prev) => prev.map((a) => (selectedIds.has(a.id) ? { ...a, ...patch } : a)))
  const bulkSetTier = (t: string) => {
    if (t && tiers[t]) bulkPatch({ tier: t, provider: tiers[t].provider, model: tiers[t].model })
  }
  const bulkSetArchetype = (id: string) =>
    setAgents((prev) => prev.map((a) => (selectedIds.has(a.id) ? { ...a, ...archetypePatch(a, id) } : a)))
  const bulkDelete = () => {
    setAgents((prev) => prev.filter((a) => !selectedIds.has(a.id)))
    setSelectedIds(new Set())
  }

  // Persist any invented archetypes (bespoke agents carrying a newArchetype) to
  // the user's Library via PUT-upsert, so the team is fully archetype-backed and
  // the templates are reusable next time. Best-effort: a failure just leaves the
  // agent bespoke (it still spawns from its composed instructions). Ids are
  // slugged and suffixed to avoid clobbering an existing base/user archetype or
  // a sibling minted in the same run. Returns the number saved.
  const mintArchetypes = async (): Promise<number> => {
    const used = new Set<string>((registry?.archetypes || []).map((a) => a.id))
    let minted = 0
    for (const agent of agents) {
      if (agent.archetype || !agent.newArchetype) continue
      const na = agent.newArchetype
      // The (possibly user-tweaked) base SOP lives in baseInstructions; fall back
      // to the model's fleet_instructions. additional_instructions stays task-
      // specific and is deliberately NOT baked into the reusable archetype.
      const sop = (agent.baseInstructions || na.fleet_instructions || '').trim()
      const baseId = slugifyId(na.role || agent.role || agent.name)
      if (!sop || !baseId) continue
      let id = baseId
      let n = 2
      while (used.has(id)) { id = `${baseId}-${n}`; n++ }
      used.add(id)
      const input: ArchetypeInput = {
        archetype_id: id,
        role: na.role || agent.role || 'Agent',
        family: na.family || 'Forged',
        description: na.description || agent.description || '',
        cognitive_mode: clampArchMode(na.cognitive_mode || agent.cognitive_mode),
        tier: clampArchTier(na.tier || agent.tier),
        tools: (Array.isArray(na.tools) && na.tools.length ? na.tools : agent.tools) || DEFAULT_TOOLS,
        fleet_instructions: sop,
        keywords: Array.isArray(na.keywords) ? na.keywords : [],
        lead: false,
        reliability_seed: 0.7,
      }
      try {
        await updateArchetype(id, input) // PUT = idempotent upsert
        // Rebase the agent onto the saved archetype so it's no longer bespoke.
        updateAgent(agent.id, { archetype: id, newArchetype: null, baseInstructions: sop })
        minted++
      } catch (e) {
        console.warn('Forge: failed to save archetype', id, e)
      }
    }
    return minted
  }

  const handleSpawnAll = async () => {
    setPhase('spawning')
    const pending: Record<string, 'pending' | 'spawning' | 'done' | 'error'> = {}
    for (const a of agents) pending[a.id] = 'pending'
    setSpawnProgress(pending)
    setSpawnErrors({})

    // Persist any invented archetypes to the Library first, so the team is fully
    // archetype-backed. Best-effort — never blocks the spawn.
    let minted = 0
    try {
      minted = await mintArchetypes()
    } catch { /* best-effort */ }
    setMintedCount(minted)
    if (minted > 0) refreshRegistry()

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

    // Spawn a single agent. Progress/errors use functional setState so the
    // bounded-concurrency pool below can't clobber each other's updates.
    const spawnOne = async (agent: AgentProposal) => {
      setSpawnProgress((p) => ({ ...p, [agent.id]: 'spawning' }))

      // Resolve the tier to a concrete model from the user's per-tier config.
      // Done client-side (multi-tenant: each workspace has its own models/keys),
      // so we send concrete provider/model and clear `tier` for the backend.
      const tc = agent.tier ? tiers[agent.tier] : undefined
      const provider = tc ? tc.provider : agent.provider
      const model = tc ? tc.model : agent.model
      const apiKey = tc ? tc.api_key : agent.providerApiKey
      const baseUrl = tc ? tc.base_url : ''
      const maxTokens = tc && tc.output_ctx > 0 ? tc.output_ctx : 32768
      const maxContext = tc && tc.input_ctx > 0 ? tc.input_ctx : 0

      const payload: SpawnConfig = {
        name: agent.name,
        description: agent.description,
        hostname: 'captain-claw',
        image: 'kstevica/captain-claw:latest',
        provider,
        model,
        tier: '',
        temperature: 0.7,
        max_tokens: maxTokens,
        max_context: maxContext,
        provider_api_key: apiKey,
        base_url: baseUrl,
        botport_enabled: false,
        botport_url: '',
        botport_instance_name: '',
        botport_key: '',
        botport_secret: '',
        botport_max_concurrent: 5,
        tools: agent.tools,
        cognitive_mode: agent.cognitive_mode || 'neutra',
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
          setProcessFleetInst(slug, composeFleet(agent))
          setProcessDesc(slug, agent.description)
          const displayName = agent.lead ? `${agent.name} (${agent.role}) [Lead]` : `${agent.name} (${agent.role})`
          setProcessName(slug, displayName)
        } else {
          if (!dockerAvailable) throw new Error('Docker not available')
          const result = await spawnAgent(payload)
          if (!result.ok) throw new Error(result.message)
          agentStoreId = result.container_id
          setContainerFleetInst(agentStoreId, composeFleet(agent))
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

        // Join project if selected
        if (selectedProjectId) {
          try {
            await fetch(`/fd/projects/${selectedProjectId}/members`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                agent_id: agentStoreId,
                agent_name: agent.name,
                role: agent.role,
                expertise_tags: agent.tools.slice(0, 5),
              }),
            })
          } catch { /* best-effort */ }
        }

        setSpawnProgress((p) => ({ ...p, [agent.id]: 'done' }))
      } catch (e) {
        setSpawnProgress((p) => ({ ...p, [agent.id]: 'error' }))
        setSpawnErrors((er) => ({ ...er, [agent.id]: e instanceof Error ? e.message : String(e) }))
      }
    }

    // Bounded-concurrency pool: a 10–15 agent team stands up in seconds instead
    // of sequentially. Workers steal from a shared queue; shift() is atomic
    // between awaits so no agent is spawned twice.
    const CONCURRENCY = 4
    const queue = [...agents]
    const worker = async () => {
      for (;;) {
        const agent = queue.shift()
        if (!agent) return
        await spawnOne(agent)
      }
    }
    await Promise.all(
      Array.from({ length: Math.min(CONCURRENCY, agents.length) }, worker),
    )

    fetchContainers()
    fetchProcesses()
    setPhase('done')
  }

  // ── Render ──

  const leadAgent = agents.find((a) => a.lead)

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Wand2 className="h-5 w-5 text-violet-700 dark:text-violet-400" /> Agent Forge
        </h1>
        <p className="text-xs text-zinc-500 sm:text-sm">Describe your objective and let AI design a team of specialized agents</p>
      </div>

      {/* Phase 1: Input */}
      {phase === 'input' && (
        <div className="space-y-4">
          {/* Models + archetypes are configured on the Library page */}
          <div className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3 text-xs text-zinc-400">
            <Gauge className="h-4 w-4 text-cyan-700 dark:text-cyan-400 shrink-0" />
            <span>
              Designs on tier <strong className="text-zinc-300">{forgeTier}</strong>
              {tiers[forgeTier]?.model ? ` · ${tiers[forgeTier].provider}/${tiers[forgeTier].model}` : ' (unset)'}.
            </span>
            <button
              onClick={() => setView('library')}
              className="ml-auto rounded-lg border border-zinc-700 px-2.5 py-1 font-medium text-zinc-300 hover:bg-zinc-800"
            >
              Configure in Library →
            </button>
          </div>

          {/* Forge intro for new users */}
          <HelpHint id="forge-intro" banner>
            <p className="mb-2 font-medium text-zinc-200">How Agent Forge works</p>
            <p className="mb-2">Describe a business objective in plain English. The AI will design specialized agents with roles, tools, and instructions tailored to your goal. You can review and customize each agent before spawning the team.</p>
            <p>Try one of these examples:</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {[
                'Research competitors in the SaaS market and create a comparison report',
                'Monitor social media mentions and draft weekly summary reports',
                'Review a codebase, identify issues, and create documentation',
              ].map((ex) => (
                <button
                  key={ex}
                  onClick={() => setPrompt(ex)}
                  className="fd-help-chip rounded-full border border-violet-500/30 bg-violet-500/10 px-3 py-1 text-[11px] text-violet-700 dark:text-violet-300 hover:bg-violet-500/20 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </HelpHint>

          {/* Project selector */}
          {projects.length > 0 && (
            <div className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3">
              <FolderKanban className="h-4 w-4 text-zinc-500" />
              <label className="text-sm text-zinc-400">Project:</label>
              <select
                value={selectedProjectId}
                onChange={(e) => setSelectedProjectId(e.target.value)}
                className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              >
                <option value="">None (standalone team)</option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>
          )}

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
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-700 dark:text-red-400">
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
              <p className="text-[11px] text-violet-700 dark:text-violet-400 mt-2">
                <Crown className="inline h-3 w-3 mr-1" />
                Lead agent: <strong>{leadAgent.name}</strong> ({leadAgent.role})
              </p>
            )}
          </div>

          {/* Bulk toolbar — select / expand / collapse across the whole team */}
          <div className="flex flex-wrap items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs">
            <label className="flex items-center gap-2 text-zinc-400 cursor-pointer select-none">
              <input type="checkbox" checked={allSelected} onChange={toggleSelectAll} className="accent-violet-500" />
              {selectedIds.size > 0 ? `${selectedIds.size} selected` : 'Select all'}
            </label>
            <div className="flex-1" />
            <button onClick={expandAll} className="rounded-lg border border-zinc-700 px-2.5 py-1 text-zinc-300 hover:bg-zinc-800">Expand all</button>
            <button onClick={collapseAll} className="rounded-lg border border-zinc-700 px-2.5 py-1 text-zinc-300 hover:bg-zinc-800">Collapse all</button>
          </div>

          {/* Bulk-edit bar — appears when one or more cards are selected */}
          {selectedIds.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 rounded-xl border border-violet-500/30 bg-violet-500/5 px-3 py-2 text-xs">
              <span className="font-medium text-violet-700 dark:text-violet-300">Bulk edit {selectedIds.size}:</span>
              {Object.keys(tiers).length > 0 && (
                <select
                  value=""
                  onChange={(e) => { if (e.target.value) bulkSetTier(e.target.value) }}
                  className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-200 focus:outline-none"
                >
                  <option value="">Set tier…</option>
                  {TIER_ORDER.filter((t) => tiers[t]).map((t) => <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>)}
                </select>
              )}
              <select
                value=""
                onChange={(e) => { if (e.target.value) bulkPatch({ type: e.target.value as 'process' | 'docker' }) }}
                className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-200 focus:outline-none"
              >
                <option value="">Set type…</option>
                <option value="process">Process</option>
                {dockerAvailable && <option value="docker">Docker</option>}
              </select>
              {(registry?.archetypes?.length ?? 0) > 0 && (
                <select
                  value=""
                  onChange={(e) => { if (e.target.value) bulkSetArchetype(e.target.value === '__none__' ? '' : e.target.value) }}
                  className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-200 focus:outline-none"
                >
                  <option value="">Set archetype…</option>
                  <option value="__none__">None (bespoke)</option>
                  {Object.entries(archetypesByFamily).map(([fam, list]) => (
                    <optgroup key={fam} label={fam}>
                      {list.map((a) => <option key={a.id} value={a.id}>{a.role} ({a.id})</option>)}
                    </optgroup>
                  ))}
                </select>
              )}
              <div className="flex-1" />
              <button onClick={bulkDelete} className="flex items-center gap-1 rounded-lg border border-red-500/30 px-2.5 py-1 text-red-700 dark:text-red-400 hover:bg-red-500/10">
                <Trash2 className="h-3 w-3" /> Delete
              </button>
            </div>
          )}

          {/* Agent Cards */}
          <div className="space-y-3">
            {agents.map((agent) => {
              const isExpanded = expandedIds.has(agent.id)
              return (
                <div key={agent.id} className={`rounded-xl border overflow-hidden ${agent.lead ? 'border-amber-500/60 bg-amber-500/10 shadow-sm shadow-amber-500/10' : 'border-zinc-800 bg-zinc-900/50'}`}>
                  {/* Card Header */}
                  <div
                    className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-zinc-800/30"
                    onClick={() => toggleInSet(setExpandedIds, agent.id)}
                  >
                    <input
                      type="checkbox"
                      checked={selectedIds.has(agent.id)}
                      onClick={(e) => e.stopPropagation()}
                      onChange={() => toggleInSet(setSelectedIds, agent.id)}
                      className="accent-violet-500 shrink-0"
                    />
                    {isExpanded ? <ChevronDown className="h-4 w-4 text-zinc-500" /> : <ChevronRight className="h-4 w-4 text-zinc-500" />}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        {agent.lead && <Crown className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400 shrink-0" />}
                        <span className="text-sm font-medium text-zinc-200 truncate">{agent.name}</span>
                        <span className="rounded-full bg-blue-600/20 border border-blue-500/30 px-2 py-0.5 text-[10px] font-semibold text-blue-700 dark:text-blue-400 shrink-0">{teamName || 'Team'}</span>
                        <span className="rounded-full bg-violet-600/20 border border-violet-500/30 px-2 py-0.5 text-[10px] font-semibold text-violet-700 dark:text-violet-400 shrink-0">{agent.role}</span>
                        <span className="inline-flex items-center gap-1 rounded-full bg-teal-600/20 border border-teal-500/30 px-2 py-0.5 text-[10px] font-semibold text-teal-700 dark:text-teal-300 shrink-0">
                          {agent.archetype
                            ? <><Boxes className="h-2.5 w-2.5" />{agent.archetype}</>
                            : agent.newArchetype
                              ? <><Sparkles className="h-2.5 w-2.5" />new archetype</>
                              : <>custom</>}
                        </span>
                        {agent.tier && <span className="inline-flex items-center gap-1 rounded-full bg-cyan-600/20 border border-cyan-500/30 px-2 py-0.5 text-[10px] font-semibold text-cyan-700 dark:text-cyan-400 shrink-0"><Gauge className="h-2.5 w-2.5" />{agent.tier}</span>}
                        {agent.lead && <span className="rounded-full bg-amber-600/20 border border-amber-500/30 px-2 py-0.5 text-[10px] font-semibold text-amber-700 dark:text-amber-400 shrink-0">Lead</span>}
                      </div>
                      <p className="text-[11px] text-zinc-500 truncate mt-0.5">{agent.description}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${agent.type === 'docker' ? 'bg-blue-600/20 border border-blue-500/30 text-blue-700 dark:text-blue-400' : 'bg-emerald-600/20 border border-emerald-500/30 text-emerald-700 dark:text-emerald-400'}`}>
                        {agent.type === 'docker' ? <Server className="h-2.5 w-2.5" /> : <Cpu className="h-2.5 w-2.5" />}
                        {agent.type}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleLead(agent.id) }}
                        className={`rounded p-1 transition-colors ${agent.lead ? 'text-amber-700 dark:text-amber-400 bg-amber-500/10' : 'text-zinc-600 hover:text-amber-600 dark:hover:text-amber-400 hover:bg-zinc-800'}`}
                        title={agent.lead ? 'Lead agent' : 'Set as lead'}
                      >
                        <Crown className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeAgent(agent.id) }}
                        className="rounded p-1 text-zinc-600 hover:text-red-600 dark:hover:text-red-400 hover:bg-zinc-800"
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
                      {(registry?.archetypes?.length ?? 0) > 0 && (
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Base archetype</label>
                          <select
                            value={agent.archetype || ''}
                            onChange={(e) => applyArchetype(agent, e.target.value)}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          >
                            <option value="">None (bespoke{agent.newArchetype ? ' — new archetype' : ''})</option>
                            {Object.entries(archetypesByFamily).map(([fam, list]) => (
                              <optgroup key={fam} label={fam}>
                                {list.map((a) => <option key={a.id} value={a.id}>{a.role} ({a.id})</option>)}
                              </optgroup>
                            ))}
                          </select>
                          <p className="mt-1 text-[11px] text-teal-600 dark:text-teal-500/80">
                            {agent.archetype
                              ? 'Inherits this archetype’s SOP, tools, mode & tier. Add task-specific notes below.'
                              : agent.newArchetype
                                ? 'Bespoke agent with a new archetype (saved to your Library on spawn).'
                                : 'Bespoke agent — write its full instructions below.'}
                          </p>
                        </div>
                      )}
                      {Object.keys(tiers).length > 0 && (
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Model tier</label>
                          <select
                            value={agent.tier || ''}
                            onChange={(e) => {
                              const t = e.target.value
                              // Selecting a tier fills provider/model from that tier's
                              // config so the fields show the effective values.
                              // "Custom" keeps the current provider/model.
                              if (t && tiers[t]) updateAgent(agent.id, { tier: t, provider: tiers[t].provider, model: tiers[t].model })
                              else updateAgent(agent.id, { tier: '' })
                            }}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          >
                            <option value="">Custom (pick provider &amp; model below)</option>
                            {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                              <option key={t} value={t}>{registry?.tiers[t]?.label || t} — {tiers[t].provider}/{tiers[t].model || '(unset)'}</option>
                            ))}
                          </select>
                          {agent.tier && tiers[agent.tier] && (
                            <p className="mt-1 text-[11px] text-cyan-600 dark:text-cyan-500/80">Runs on {tiers[agent.tier].provider}/{tiers[agent.tier].model || '(unset)'} from your tier config. Editing provider or model below switches to Custom.</p>
                          )}
                        </div>
                      )}
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Provider</label>
                          <select
                            value={agent.provider}
                            onChange={(e) => pinAgentModel(agent, { provider: e.target.value })}
                            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          >
                            {!PROVIDERS.includes(agent.provider) && <option value={agent.provider}>{agent.provider || '— select —'}</option>}
                            {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[11px] font-medium text-zinc-500 mb-1">Model</label>
                          <input
                            value={agent.model}
                            onChange={(e) => pinAgentModel(agent, { model: e.target.value })}
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
                      {/* Instructions: read-only archetype base SOP + task-specific
                          delta, with a full-text override escape hatch. */}
                      {(fullEditOpen.has(agent.id) || agent.fleetOverride) ? (
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <label className="block text-[11px] font-medium text-zinc-500">Full instructions (manual override)</label>
                            <button
                              onClick={() => { updateAgent(agent.id, { fleetOverride: '' }); removeFromSet(setFullEditOpen, agent.id) }}
                              className="text-[11px] text-teal-700 dark:text-teal-400 hover:text-teal-600 dark:hover:text-teal-300"
                            >
                              Reset to archetype base
                            </button>
                          </div>
                          <textarea
                            value={agent.fleetOverride || composeFleet(agent)}
                            onChange={(e) => updateAgent(agent.id, { fleetOverride: e.target.value })}
                            rows={12}
                            className="w-full rounded-lg border border-amber-500/40 bg-zinc-950 px-3 py-2 text-[13px] font-mono text-zinc-300 leading-relaxed focus:border-amber-500/60 focus:outline-none resize-none"
                          />
                          <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-500/80">Editing here overrides the archetype base + additional instructions.</p>
                        </div>
                      ) : (
                        <>
                          {agent.baseInstructions.trim() && (
                            <div>
                              <button
                                onClick={() => toggleInSet(setBaseSopOpen, agent.id)}
                                className="flex items-center gap-1 text-[11px] font-medium text-zinc-400 hover:text-zinc-200"
                              >
                                {baseSopOpen.has(agent.id) ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                Base SOP {agent.archetype ? `(from ${agent.archetype})` : '(from new archetype)'}
                              </button>
                              {baseSopOpen.has(agent.id) && (
                                <textarea
                                  readOnly
                                  value={agent.baseInstructions}
                                  rows={8}
                                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-[12px] font-mono text-zinc-500 leading-relaxed focus:outline-none resize-none"
                                />
                              )}
                            </div>
                          )}
                          <div>
                            <label className="block text-[11px] font-medium text-zinc-500 mb-1">
                              {agent.baseInstructions.trim() ? 'Additional instructions (task-specific)' : 'Instructions'}
                            </label>
                            <textarea
                              value={agent.additional_instructions}
                              onChange={(e) => updateAgent(agent.id, { additional_instructions: e.target.value })}
                              rows={8}
                              placeholder={agent.baseInstructions.trim() ? 'What this agent does for THIS objective, who it collaborates with, inputs/outputs…' : 'Full instructions for this bespoke agent…'}
                              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-[13px] font-mono text-zinc-300 leading-relaxed focus:border-violet-500/50 focus:outline-none resize-none"
                            />
                          </div>
                          <button
                            onClick={() => setFullEditOpen((prev) => new Set(prev).add(agent.id))}
                            className="text-[11px] text-zinc-500 hover:text-zinc-300"
                          >
                            Edit full instructions text →
                          </button>
                        </>
                      )}
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
            {phase === 'done' && mintedCount > 0 && (
              <p className="mb-3 flex items-center gap-1.5 text-[11px] text-teal-700 dark:text-teal-400">
                <Sparkles className="h-3 w-3" />
                Saved {mintedCount} new archetype{mintedCount !== 1 ? 's' : ''} to your Library.
              </p>
            )}
            <div className="space-y-2">
              {agents.map((agent) => {
                const status = spawnProgress[agent.id] || 'pending'
                const err = spawnErrors[agent.id]
                return (
                  <div key={agent.id} className="flex items-center gap-3 rounded-lg bg-zinc-950/50 px-3 py-2">
                    {status === 'pending' && <div className="h-4 w-4 rounded-full border-2 border-zinc-700" />}
                    {status === 'spawning' && <Loader2 className="h-4 w-4 animate-spin text-violet-700 dark:text-violet-400" />}
                    {status === 'done' && <Check className="h-4 w-4 text-emerald-700 dark:text-emerald-400" />}
                    {status === 'error' && <AlertTriangle className="h-4 w-4 text-red-700 dark:text-red-400" />}
                    <div className="flex-1 min-w-0">
                      <span className="text-sm text-zinc-300">
                        {agent.lead && <Crown className="inline h-3 w-3 text-amber-700 dark:text-amber-400 mr-1" />}
                        {agent.name}
                      </span>
                      <span className="ml-2 text-[10px] text-zinc-500">{agent.role}</span>
                    </div>
                    {err && <span className="text-[11px] text-red-700 dark:text-red-400 truncate max-w-[200px]">{err}</span>}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Aggregated failure summary — surfaces plan-limit rejections once
              rather than repeating the same error across every failed card. */}
          {phase === 'done' && (() => {
            const failed = agents.filter((a) => spawnProgress[a.id] === 'error')
            if (failed.length === 0) return null
            const limitHit = failed.some((a) => /limit|plan|quota/i.test(spawnErrors[a.id] || ''))
            return (
              <div className="flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-700 dark:text-red-400">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <p>{failed.length} of {agents.length} agent{agents.length !== 1 ? 's' : ''} failed to spawn.</p>
                  {limitHit && (
                    <p className="mt-1 text-[12px] text-red-700/80 dark:text-red-300/80">
                      Your plan's concurrent-agent limit was reached. Remove some agents, stop idle ones on the Agent Desktop, or upgrade your plan, then spawn the rest.
                    </p>
                  )}
                </div>
              </div>
            )
          })()}

          {phase === 'done' && (
            <div className="flex items-center gap-3">
              <button
                onClick={() => { onboarding.completeStep('forge'); setView('desktop') }}
                className="flex items-center gap-2 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500 transition-colors"
              >
                Go to Agent Desktop
              </button>
              {!onboarding.completed.council && (
                <button
                  onClick={() => { onboarding.completeStep('forge'); setView('council') }}
                  className="flex items-center gap-2 rounded-xl border border-violet-500/30 bg-violet-500/10 px-4 py-2.5 text-sm font-medium text-violet-700 dark:text-violet-300 hover:bg-violet-500/20 transition-colors"
                >
                  Next: Run a Council Session
                </button>
              )}
              <button
                onClick={() => { onboarding.completeStep('forge'); setPhase('input'); setAgents([]); setPrompt(''); setError(''); setMintedCount(0) }}
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
