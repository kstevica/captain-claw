import { useState } from 'react'
import {
  Library, Gauge, Trash2, Plus, Crown, Loader2, Check, AlertTriangle, Rocket,
  Layers, Copy,
} from 'lucide-react'
import { useProcessStore } from '../stores/processStore'
import { spawnProcess, type SpawnConfig } from '../services/docker'
import {
  useTierConfig, PROVIDERS, TIER_ORDER, type Archetype,
} from '../services/tierConfig'

type SpawnState = 'spawning' | 'done' | 'error'

export function LibraryPage() {
  const {
    tiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, updateTier,
    sets, activeSetId, setActiveSet, addSet, duplicateSet, renameSet, deleteSet,
  } = useTierConfig()

  const activeSet = sets.find((s) => s.id === activeSetId) || sets[0]

  const { setFleetInstructions, setDescription, setNameOverride, fetchProcesses } = useProcessStore()

  const [spawnState, setSpawnState] = useState<Record<string, SpawnState>>({})
  const [spawnMsg, setSpawnMsg] = useState<Record<string, string>>({})

  // One-click spawn of a library archetype as a process agent, resolving its
  // tier to a concrete model from the saved tier config.
  const spawnArchetype = async (a: Archetype) => {
    const tc = tiers[a.tier]
    if (!tc || !tc.model.trim()) {
      setSpawnState((s) => ({ ...s, [a.id]: 'error' }))
      setSpawnMsg((m) => ({ ...m, [a.id]: `Configure the "${a.tier}" tier first` }))
      return
    }
    setSpawnState((s) => ({ ...s, [a.id]: 'spawning' }))
    setSpawnMsg((m) => ({ ...m, [a.id]: '' }))

    const payload: SpawnConfig = {
      name: a.id,
      description: a.description,
      hostname: 'captain-claw',
      image: 'kstevica/captain-claw:latest',
      provider: tc.provider,
      model: tc.model,
      tier: '',
      temperature: 0.7,
      max_tokens: tc.output_ctx > 0 ? tc.output_ctx : 32768,
      max_context: tc.input_ctx > 0 ? tc.input_ctx : 0,
      provider_api_key: tc.api_key,
      base_url: tc.base_url,
      botport_enabled: false,
      botport_url: '',
      botport_instance_name: '',
      botport_key: '',
      botport_secret: '',
      botport_max_concurrent: 5,
      tools: a.tools,
      cognitive_mode: a.cognitive_mode || 'neutra',
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
      const result = await spawnProcess(payload)
      if (!result.ok) throw new Error(result.message)
      const slug = a.id.replace(/[^a-z0-9-]/gi, '-').toLowerCase()
      setFleetInstructions(slug, a.fleet_instructions)
      setDescription(slug, a.description)
      setNameOverride(slug, a.lead ? `${a.id} (${a.role}) [Lead]` : `${a.id} (${a.role})`)
      setSpawnState((s) => ({ ...s, [a.id]: 'done' }))
      fetchProcesses()
    } catch (e) {
      setSpawnState((s) => ({ ...s, [a.id]: 'error' }))
      setSpawnMsg((m) => ({ ...m, [a.id]: e instanceof Error ? e.message : String(e) }))
    }
  }

  const families = registry ? [...new Set(registry.archetypes.map((a) => a.family))] : []

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Library className="h-5 w-5 text-violet-400" /> Library
        </h1>
        <p className="text-xs text-zinc-500 sm:text-sm">
          Configure the model behind each tier, and spawn agents one-click from the curated archetypes.
        </p>
      </div>

      <div className="space-y-4">
        {/* ── Model Tiers ── */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Gauge className="h-4 w-4 text-cyan-400" />
            <h2 className="text-sm font-medium text-zinc-200">Model Tiers</h2>
          </div>
          <p className="text-[11px] text-zinc-500">
            Each tier is a model definition. Agents (from the gallery or Agent Forge) run on the model of their tier. Settings are saved to your workspace.
          </p>

          {/* ── Tier sets: pick the active profile, or manage them ── */}
          <div className="rounded-lg border border-violet-500/20 bg-violet-500/[0.04] p-3 space-y-2">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-violet-400 shrink-0" />
              <span className="text-sm font-medium text-zinc-200">Tier Sets</span>
              <span className="text-[11px] text-zinc-500 truncate">— the active set drives Forge, Basna &amp; spawns</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={activeSetId}
                onChange={(e) => setActiveSet(e.target.value)}
                className="min-w-[10rem] flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              >
                {sets.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
              <button
                onClick={() => addSet()}
                title="New set (seeded from defaults)"
                className="flex items-center gap-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100"
              >
                <Plus className="h-3.5 w-3.5" /> New
              </button>
              <button
                onClick={() => duplicateSet(activeSetId)}
                title="Duplicate the active set"
                className="flex items-center gap-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100"
              >
                <Copy className="h-3.5 w-3.5" /> Duplicate
              </button>
              <button
                onClick={() => { if (sets.length > 1 && confirm(`Delete tier set "${activeSet?.name}"?`)) deleteSet(activeSetId) }}
                disabled={sets.length <= 1}
                title={sets.length <= 1 ? 'Keep at least one set' : 'Delete the active set'}
                className="flex items-center gap-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-400 hover:border-red-500/40 hover:text-red-400 disabled:opacity-40 disabled:hover:border-zinc-700 disabled:hover:text-zinc-400"
              >
                <Trash2 className="h-3.5 w-3.5" /> Delete
              </button>
            </div>
            {activeSet && (
              <div>
                <label className="block text-[10px] font-medium text-zinc-500 mb-1">Set name</label>
                <input
                  value={activeSet.name}
                  onChange={(e) => renameSet(activeSetId, e.target.value)}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  placeholder="e.g. All Anthropic, Local Ollama"
                />
              </div>
            )}
          </div>

          {/* Which tier designs the team in Forge */}
          <div>
            <label className="block text-[11px] font-medium text-zinc-500 mb-1">Forge using — model that designs the team</label>
            <select
              value={forgeTier}
              onChange={(e) => setForgeTier(e.target.value)}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
            >
              {TIER_ORDER.filter((t) => tiers[t]).map((t) => (
                <option key={t} value={t}>
                  {registry?.tiers[t]?.label || t} — {tiers[t].provider}/{tiers[t].model || '(unset)'}
                </option>
              ))}
            </select>
          </div>

          {/* Per-tier model definitions */}
          {Object.keys(tiers).length === 0 ? (
            <p className="text-[11px] text-zinc-600">Loading model tiers…</p>
          ) : (
            <div className="space-y-2.5">
              {TIER_ORDER.map((t) => {
                const tc = tiers[t]
                if (!tc) return null
                const def = registry?.tiers[t]
                return (
                  <div key={t} className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3 space-y-2">
                    <div className="flex items-center gap-2">
                      <Gauge className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                      <span className="text-sm font-medium text-zinc-200">{def?.label || t}</span>
                      {def?.use && <span className="text-[11px] text-zinc-500 truncate">— {def.use}</span>}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-[10px] font-medium text-zinc-500 mb-1">Provider</label>
                        <select
                          value={tc.provider}
                          onChange={(e) => updateTier(t, { provider: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                        >
                          {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="block text-[10px] font-medium text-zinc-500 mb-1">Model</label>
                        <input
                          value={tc.model}
                          onChange={(e) => updateTier(t, { model: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="model id"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="block text-[10px] font-medium text-zinc-500 mb-1">API Key</label>
                      <input
                        type="password"
                        value={tc.api_key}
                        onChange={(e) => updateTier(t, { api_key: e.target.value })}
                        className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                        placeholder="sk-… (leave blank to use server env key)"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      <div className="sm:col-span-2">
                        <label className="block text-[10px] font-medium text-zinc-500 mb-1">Base URL</label>
                        <input
                          value={tc.base_url}
                          onChange={(e) => updateTier(t, { base_url: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="optional"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-medium text-zinc-500 mb-1">Input ctx</label>
                        <input
                          type="number"
                          value={tc.input_ctx}
                          onChange={(e) => updateTier(t, { input_ctx: Number(e.target.value) || 0 })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] font-medium text-zinc-500 mb-1">Output ctx</label>
                        <input
                          type="number"
                          value={tc.output_ctx}
                          onChange={(e) => updateTier(t, { output_ctx: Number(e.target.value) || 0 })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Additional API Keys / Environment Variables */}
          <div>
            <label className="block text-[11px] font-medium text-zinc-500 mb-1">Additional API Keys — part of this set, passed to agents it spawns</label>
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

        {/* ── Archetype gallery ── */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Library className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-zinc-200">
              Archetypes{registry ? ` — ${registry.archetypes.length}` : ''}
            </h2>
            <span className="text-[11px] text-zinc-500">click to spawn</span>
          </div>
          {!registry ? (
            <p className="text-[11px] text-zinc-600">Loading archetypes…</p>
          ) : (
            <div className="space-y-4">
              {families.map((family) => (
                <div key={family}>
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500 mb-2">{family}</p>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {registry.archetypes.filter((a) => a.family === family).map((a) => {
                      const st = spawnState[a.id]
                      return (
                        <button
                          key={a.id}
                          onClick={() => spawnArchetype(a)}
                          disabled={st === 'spawning'}
                          title={`Spawn ${a.role}`}
                          className="group text-left rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 hover:border-violet-500/40 hover:bg-zinc-900 transition-colors disabled:opacity-60"
                        >
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <span className="flex items-center gap-1 text-sm font-medium text-zinc-200 truncate">
                              {a.lead && <Crown className="h-3 w-3 text-amber-400 shrink-0" />}{a.role}
                            </span>
                            {st === 'spawning' ? <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-400 shrink-0" />
                              : st === 'done' ? <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                              : st === 'error' ? <AlertTriangle className="h-3.5 w-3.5 text-red-400 shrink-0" />
                              : <Rocket className="h-3.5 w-3.5 text-zinc-600 group-hover:text-violet-400 shrink-0" />}
                          </div>
                          <p className="text-[11px] text-zinc-500 leading-snug mb-2">{a.description}</p>
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="inline-flex items-center gap-1 rounded bg-cyan-600/15 border border-cyan-500/25 px-1.5 py-0.5 text-[10px] font-medium text-cyan-400"><Gauge className="h-2.5 w-2.5" />{a.tier}</span>
                            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{a.cognitive_mode}</span>
                            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">{a.tools.length} tools</span>
                          </div>
                          {st === 'error' && spawnMsg[a.id] && (
                            <p className="mt-1.5 text-[10px] text-red-400">{spawnMsg[a.id]}</p>
                          )}
                          {st === 'done' && (
                            <p className="mt-1.5 text-[10px] text-emerald-400">Spawned — see Agent Desktop</p>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
