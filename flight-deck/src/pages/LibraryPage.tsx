import { useEffect, useMemo, useState } from 'react'
import {
  Library, Gauge, Trash2, Plus, Crown, Loader2, Check, AlertTriangle, Rocket,
  Layers, Copy, Pencil, Sparkles, X, KeyRound, Wand2,
} from 'lucide-react'
import { useProcessStore } from '../stores/processStore'
import { useUIStore } from '../stores/uiStore'
import { spawnProcess, type SpawnConfig } from '../services/docker'
import {
  useTierConfig, PROVIDERS, TIER_ORDER, isSetUnconfigured, type Archetype,
} from '../services/tierConfig'
import {
  createArchetype, updateArchetype, deleteArchetype, generateArchetype,
  type ArchetypeInput,
} from '../services/archetypes'
import { ForgeArchetypesModal } from '../components/library/ForgeArchetypesModal'

type SpawnState = 'spawning' | 'done' | 'error'

const COGNITIVE_MODES = [
  'neutra', 'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian',
]

// Auto-open the setup wizard once per browser/user until they complete or dismiss it.
const WIZARD_SEEN_KEY = 'fd:tier-wizard-seen'

// Context-window presets offered by the wizard (tokens, power-of-2 based —
// k = 1024, M = 1024²). Defaults: 256k in, 32k out.
const INPUT_CTX_OPTS = [128, 160, 256, 400, 512, 1024].map((k) => k * 1024)
const OUTPUT_CTX_OPTS = [16, 32, 64, 128, 256].map((k) => k * 1024)

function fmtCtx(n: number): string {
  return n >= 1024 * 1024 ? `${Math.round(n / (1024 * 1024))}M` : `${Math.round(n / 1024)}k`
}

interface WizardValues {
  name: string
  provider: string
  model: string
  apiKey: string
  baseUrl: string
  inputCtx: number
  outputCtx: number
}

function slugify(s: string): string {
  return s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
}

export function LibraryPage() {
  const {
    tiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, refreshRegistry, updateTier, setupSet,
    sets, activeSetId, setActiveSet, addSet, duplicateSet, renameSet, deleteSet, bootstrapped,
  } = useTierConfig()

  // Archetype editor: null = closed; otherwise the draft being edited. `editingId`
  // is the existing archetype_id when editing (PUT), or null for a new one (POST).
  const [editor, setEditor] = useState<ArchetypeInput | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editorErr, setEditorErr] = useState('')
  const [saving, setSaving] = useState(false)
  const [genPrompt, setGenPrompt] = useState('')
  const [generating, setGenerating] = useState(false)
  // Batch "Forge archetypes" modal (instructions + documents → a reusable set).
  const [forgeOpen, setForgeOpen] = useState(false)

  const [tab, setTab] = useState<'tiers' | 'archetypes'>('tiers')

  // ── Setup wizard: one model → all tiers. Auto-opens once on a fresh install. ──
  const setView = useUIStore((s) => s.setView)
  const [wizardOpen, setWizardOpen] = useState(false)
  // True when the wizard opened by itself on first launch (vs the manual button).
  // A first-launch completion sends the user straight on to Quick chat.
  const [wizardFirstLaunch, setWizardFirstLaunch] = useState(false)
  useEffect(() => {
    if (!bootstrapped || !registry) return
    if (localStorage.getItem(WIZARD_SEEN_KEY)) return
    const unconfigured = sets.length === 0 || (sets.length === 1 && isSetUnconfigured(sets[0]))
    if (unconfigured) { setWizardFirstLaunch(true); setWizardOpen(true) }
  }, [bootstrapped, registry, sets])

  const openWizard = () => { setWizardFirstLaunch(false); setWizardOpen(true) }
  const closeWizard = () => { localStorage.setItem(WIZARD_SEEN_KEY, '1'); setWizardOpen(false) }
  const submitWizard = (v: WizardValues) => {
    setupSet(v.name.trim() || 'My models', {
      provider: v.provider, model: v.model.trim(), api_key: v.apiKey,
      base_url: v.baseUrl.trim(), input_ctx: v.inputCtx, output_ctx: v.outputCtx,
    })
    localStorage.setItem(WIZARD_SEEN_KEY, '1')
    setWizardOpen(false)
    // On first launch, saving tiers means setup is done — move on to Quick chat.
    if (wizardFirstLaunch) setView('quick-chat')
    else setTab('tiers')
  }

  const blankDraft = (): ArchetypeInput => ({
    archetype_id: '', role: '', family: 'Custom', description: '',
    cognitive_mode: 'neutra', tier: 'balanced',
    tools: registry?.base_tools ? [...registry.base_tools] : [],
    fleet_instructions: '', keywords: [], lead: false, reliability_seed: 0.7,
  })

  const openNew = () => { setEditingId(null); setEditorErr(''); setGenPrompt(''); setEditor(blankDraft()) }
  const openEdit = (a: Archetype) => {
    setEditingId(a.id); setEditorErr(''); setGenPrompt('')
    setEditor({
      archetype_id: a.id, role: a.role, family: a.family || 'Custom',
      description: a.description || '', cognitive_mode: a.cognitive_mode || 'neutra',
      tier: a.tier || 'balanced', tools: a.tools || [],
      fleet_instructions: a.fleet_instructions || '', keywords: a.keywords || [],
      lead: !!a.lead, reliability_seed: a.reliability_seed ?? 0.7,
    })
  }
  const closeEditor = () => { setEditor(null); setEditingId(null) }

  // Tool palette for the multiselect: every tool the platform's archetypes use
  // (base_tools ∪ all archetypes' tools), so the list stays data-driven.
  const knownTools = useMemo(() => {
    const s = new Set<string>(registry?.base_tools || [])
    registry?.archetypes.forEach((a) => a.tools?.forEach((t) => s.add(t)))
    return [...s].sort()
  }, [registry])
  // Union with the draft's own tools so a generated/custom tool still shows.
  const toolOptions = editor
    ? [...new Set([...knownTools, ...editor.tools])].sort()
    : knownTools
  const toggleTool = (t: string) =>
    setEditor((e) => e ? ({
      ...e,
      tools: e.tools.includes(t) ? e.tools.filter((x) => x !== t) : [...e.tools, t],
    }) : e)

  const generateDraft = async () => {
    if (!genPrompt.trim()) return
    const ft = tiers[forgeTier]
    setGenerating(true); setEditorErr('')
    try {
      const draft = await generateArchetype(
        genPrompt.trim(), ft?.provider || '', ft?.model || '',
        ft?.api_key || '', ft?.base_url || '', ft?.output_ctx || 0,
      )
      // Keep the user's chosen id when editing an existing archetype.
      if (editingId) draft.archetype_id = editingId
      setEditor(draft)
    } catch (e) {
      setEditorErr(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }

  const saveDraft = async () => {
    if (!editor) return
    const body: ArchetypeInput = {
      ...editor,
      archetype_id: slugify(editor.archetype_id || editor.role),
    }
    if (!body.role.trim()) { setEditorErr('Role is required'); return }
    if (!body.archetype_id) { setEditorErr('An id (or role) is required'); return }
    setSaving(true); setEditorErr('')
    try {
      if (editingId) await updateArchetype(editingId, body)
      else await createArchetype(body)
      refreshRegistry()
      closeEditor()
    } catch (e) {
      setEditorErr(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const removeArchetype = async (a: Archetype) => {
    if (!confirm(`Delete your archetype "${a.role}"?${a.overrides ? ' This restores the base version.' : ''}`)) return
    try {
      await deleteArchetype(a.id)
      refreshRegistry()
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e))
    }
  }

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

      {/* ── Tabs ── */}
      <div className="mb-4 flex gap-1 border-b border-zinc-800">
        {([['tiers', 'Model Tiers', Gauge], ['archetypes', 'Archetypes', Library]] as const).map(([k, label, Icon]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={`-mb-px flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
              tab === k
                ? 'border-violet-400 text-zinc-100'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
            {k === 'archetypes' && registry && (
              <span className="text-[11px] font-normal text-zinc-500">({registry.archetypes.length})</span>
            )}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        {/* ── Model Tiers ── */}
        {tab === 'tiers' && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Gauge className="h-4 w-4 text-cyan-400" />
            <h2 className="text-sm font-medium text-zinc-200">Model Tiers</h2>
          </div>
          <p className="text-[11px] text-zinc-500">
            Each tier is a model definition. Agents (from the gallery or Agent Forge) run on the model of their tier. Settings are saved to your workspace.
          </p>

          {/* ── Tier sets: pick the active profile, or manage them ── */}
          <div className="rounded-lg border border-zinc-800 border-l-2 border-l-violet-500/60 bg-violet-500/[0.04] p-4 space-y-3">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-violet-500/10 text-violet-400">
                <Layers className="h-3.5 w-3.5" />
              </span>
              <span className="text-sm font-semibold text-zinc-200">Tier Sets</span>
              <span className="truncate text-[11px] text-zinc-500">the active set drives Forge, Basna &amp; spawns</span>
              <button
                onClick={openWizard}
                title="Set up a tier set from one model"
                className="ml-auto flex shrink-0 items-center gap-1 rounded-lg border border-violet-500/40 bg-violet-500/10 px-2.5 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-500/20 dark:text-violet-200"
              >
                <Wand2 className="h-3.5 w-3.5" /> Setup wizard
              </button>
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
                  <div key={t} className="rounded-lg border border-zinc-800 border-l-2 border-l-cyan-500/60 bg-zinc-950/40 p-4 space-y-3.5">
                    {/* Tier header */}
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-cyan-500/10 text-cyan-400">
                        <Gauge className="h-3.5 w-3.5" />
                      </span>
                      <span className="text-sm font-semibold text-zinc-200">{def?.label || t}</span>
                      {def?.use && <span className="truncate text-[11px] text-zinc-500">{def.use}</span>}
                    </div>

                    {/* Identity — provider + model */}
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-12">
                      <div className="sm:col-span-4">
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">Provider</label>
                        <select
                          value={tc.provider}
                          onChange={(e) => updateTier(t, { provider: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                        >
                          {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      </div>
                      <div className="sm:col-span-8">
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">Model</label>
                        <input
                          value={tc.model}
                          onChange={(e) => updateTier(t, { model: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="model id"
                        />
                      </div>
                    </div>

                    {/* Capacity — context window */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                          Input context <span className="font-normal normal-case text-zinc-600">tokens</span>
                        </label>
                        <input
                          type="number"
                          value={tc.input_ctx}
                          onChange={(e) => updateTier(t, { input_ctx: Number(e.target.value) || 0 })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm tabular-nums text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="0"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                          Output context <span className="font-normal normal-case text-zinc-600">tokens</span>
                        </label>
                        <input
                          type="number"
                          value={tc.output_ctx}
                          onChange={(e) => updateTier(t, { output_ctx: Number(e.target.value) || 0 })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm tabular-nums text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="0"
                        />
                      </div>
                    </div>

                    {/* Connection — credentials (usually server-provided) */}
                    <div className="grid grid-cols-1 gap-3 border-t border-zinc-800/70 pt-3 sm:grid-cols-2">
                      <div>
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                          API Key <span className="font-normal normal-case text-zinc-600">— optional</span>
                        </label>
                        <input
                          type="password"
                          value={tc.api_key}
                          onChange={(e) => updateTier(t, { api_key: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="sk-… (blank = server env key)"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                          Base URL <span className="font-normal normal-case text-zinc-600">— optional</span>
                        </label>
                        <input
                          value={tc.base_url}
                          onChange={(e) => updateTier(t, { base_url: e.target.value })}
                          className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                          placeholder="default endpoint"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Additional API Keys / Environment Variables */}
          <div className="rounded-lg border border-zinc-800 border-l-2 border-l-amber-500/60 bg-zinc-950/40 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-amber-500/10 text-amber-400">
                <KeyRound className="h-3.5 w-3.5" />
              </span>
              <span className="text-sm font-semibold text-zinc-200">Additional API Keys</span>
              <span className="truncate text-[11px] text-zinc-500">passed to every agent this set spawns</span>
            </div>
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

        {/* ── Archetype gallery ── */}
        {tab === 'archetypes' && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Library className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-medium text-zinc-200">
              Archetypes{registry ? ` — ${registry.archetypes.length}` : ''}
            </h2>
            <span className="text-[11px] text-zinc-500">click to spawn</span>
            <button
              onClick={() => setForgeOpen(true)}
              title="Forge a set of archetypes from instructions and documents"
              className="ml-auto flex items-center gap-1 rounded-lg bg-violet-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500"
            >
              <Wand2 className="h-3.5 w-3.5" /> Forge archetypes
            </button>
            <button
              onClick={openNew}
              title="Create your own archetype"
              className="flex items-center gap-1 rounded-lg border border-violet-300 bg-violet-50 px-2.5 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-100 dark:border-violet-500/30 dark:bg-violet-500/10 dark:font-normal dark:text-violet-200 dark:hover:bg-violet-500/20"
            >
              <Plus className="h-3.5 w-3.5" /> New archetype
            </button>
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
                      const isUser = a.source === 'user'
                      return (
                        <div key={a.id} className="group relative rounded-lg border border-zinc-800 bg-zinc-950/50 hover:border-violet-500/40 hover:bg-zinc-900 transition-colors">
                          {isUser && (
                            <div className="absolute top-1.5 right-1.5 z-10 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={(e) => { e.stopPropagation(); openEdit(a) }}
                                title="Edit archetype"
                                className="rounded p-1 text-zinc-500 hover:text-violet-300 hover:bg-zinc-800"
                              >
                                <Pencil className="h-3 w-3" />
                              </button>
                              <button
                                onClick={(e) => { e.stopPropagation(); removeArchetype(a) }}
                                title={a.overrides ? 'Delete (restores base version)' : 'Delete archetype'}
                                className="rounded p-1 text-zinc-500 hover:text-red-400 hover:bg-zinc-800"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                          )}
                          <button
                            onClick={() => spawnArchetype(a)}
                            disabled={st === 'spawning'}
                            title={`Spawn ${a.role}`}
                            className="w-full text-left p-3 disabled:opacity-60"
                          >
                            <div className="flex items-center justify-between gap-2 mb-1">
                              <span className="flex items-center gap-1 text-sm font-medium text-zinc-200 truncate">
                                {a.lead && <Crown className="h-3 w-3 text-amber-400 shrink-0" />}{a.role}
                                {isUser && <span className="rounded bg-violet-500/15 border border-violet-500/25 px-1 py-0.5 text-[9px] font-medium text-violet-700 dark:text-violet-300 shrink-0">{a.overrides ? 'custom·override' : 'custom'}</span>}
                              </span>
                              {/* For custom archetypes the edit/delete overlay occupies this
                                  corner on hover, so fade the spawn/status indicator out to
                                  avoid overlap (no reflow — opacity only). */}
                              <span className={`shrink-0 ${isUser ? 'transition-opacity group-hover:opacity-0' : ''}`}>
                                {st === 'spawning' ? <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-400" />
                                  : st === 'done' ? <Check className="h-3.5 w-3.5 text-emerald-400" />
                                  : st === 'error' ? <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
                                  : <Rocket className="h-3.5 w-3.5 text-zinc-600 group-hover:text-violet-400" />}
                              </span>
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
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        )}
      </div>

      {/* ── Tier setup wizard ── */}
      {wizardOpen && <TierSetupWizard onSubmit={submitWizard} onClose={closeWizard} />}

      {/* ── Forge archetypes (batch, from instructions + documents) ── */}
      {forgeOpen && (
        <ForgeArchetypesModal
          tiers={tiers}
          forgeTier={forgeTier}
          existingIds={registry?.archetypes.map((a) => a.id) || []}
          toolPalette={knownTools}
          onClose={() => setForgeOpen(false)}
          onSaved={() => refreshRegistry()}
        />
      )}

      {/* ── Archetype editor modal ── */}
      {editor && (
        <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/60 p-4" onClick={closeEditor}>
          <div
            className="my-8 w-full max-w-2xl rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
              <Library className="h-4 w-4 text-violet-400" />
              <h3 className="text-sm font-medium text-zinc-200">
                {editingId ? 'Edit archetype' : 'New archetype'}
              </h3>
              <button onClick={closeEditor} className="ml-auto rounded p-1 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-3 p-4">
              {/* Generate from prompt */}
              <div className="rounded-lg border border-violet-500/20 bg-violet-500/[0.04] p-3 space-y-2">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-violet-400" />
                  <span className="text-xs font-medium text-zinc-200">Generate from a prompt</span>
                  <span className="text-[10px] text-zinc-500">— fills the form below; review before saving</span>
                </div>
                <textarea
                  value={genPrompt}
                  onChange={(e) => setGenPrompt(e.target.value)}
                  rows={2}
                  placeholder="e.g. A meticulous contract reviewer that flags risky clauses in vendor agreements"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                />
                <button
                  onClick={generateDraft}
                  disabled={generating || !genPrompt.trim()}
                  className="flex items-center gap-1.5 rounded-lg border border-violet-300 bg-violet-50 px-3 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-100 dark:border-violet-500/30 dark:bg-violet-500/10 dark:font-normal dark:text-violet-200 dark:hover:bg-violet-500/20 disabled:opacity-50"
                >
                  {generating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                  {generating ? 'Generating…' : 'Generate draft'}
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] font-medium text-zinc-500 mb-1">Role *</label>
                  <input
                    value={editor.role}
                    onChange={(e) => setEditor({ ...editor, role: e.target.value })}
                    placeholder="Contract Reviewer"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-zinc-500 mb-1">
                    ID {editingId ? '(fixed)' : '(auto from role)'}
                  </label>
                  <input
                    value={editor.archetype_id}
                    disabled={!!editingId}
                    onChange={(e) => setEditor({ ...editor, archetype_id: e.target.value })}
                    placeholder="contract-reviewer"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none disabled:opacity-60"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[10px] font-medium text-zinc-500 mb-1">Description</label>
                <input
                  value={editor.description}
                  onChange={(e) => setEditor({ ...editor, description: e.target.value })}
                  placeholder="One sentence on what this agent does."
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-[10px] font-medium text-zinc-500 mb-1">Family</label>
                  <input
                    value={editor.family}
                    onChange={(e) => setEditor({ ...editor, family: e.target.value })}
                    placeholder="Custom"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-zinc-500 mb-1">Tier</label>
                  <select
                    value={editor.tier}
                    onChange={(e) => setEditor({ ...editor, tier: e.target.value })}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  >
                    {TIER_ORDER.map((t) => <option key={t} value={t}>{registry?.tiers[t]?.label || t}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-zinc-500 mb-1">Cognitive mode</label>
                  <select
                    value={editor.cognitive_mode}
                    onChange={(e) => setEditor({ ...editor, cognitive_mode: e.target.value })}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                  >
                    {COGNITIVE_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[10px] font-medium text-zinc-500 mb-1">
                  Tools — click to toggle{editor.tools.length ? ` (${editor.tools.length} selected)` : ''}
                </label>
                <div className="flex flex-wrap gap-1.5 rounded-lg border border-zinc-700 bg-zinc-950/40 p-2">
                  {toolOptions.map((t) => {
                    const on = editor.tools.includes(t)
                    return (
                      <button
                        key={t}
                        type="button"
                        onClick={() => toggleTool(t)}
                        className={`rounded px-2 py-0.5 text-[11px] font-mono border transition-colors ${
                          on
                            ? 'border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-200'
                            : 'border-zinc-700 bg-zinc-900/40 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'
                        }`}
                      >
                        {t}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div>
                <label className="block text-[10px] font-medium text-zinc-500 mb-1">Keywords (comma-separated — used by the Basna router)</label>
                <input
                  value={editor.keywords.join(', ')}
                  onChange={(e) => setEditor({ ...editor, keywords: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })}
                  placeholder="contracts, legal, risk, clauses"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-[10px] font-medium text-zinc-500 mb-1">Fleet instructions — the agent's operating manual</label>
                <textarea
                  value={editor.fleet_instructions}
                  onChange={(e) => setEditor({ ...editor, fleet_instructions: e.target.value })}
                  rows={8}
                  placeholder="You are a … Your job is to … Standard operating procedure: 1) …"
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                />
              </div>

              <label className="flex items-center gap-2 text-xs text-zinc-300">
                <input
                  type="checkbox"
                  checked={editor.lead}
                  onChange={(e) => setEditor({ ...editor, lead: e.target.checked })}
                  className="accent-violet-500"
                />
                Lead / coordinator role
              </label>

              {editorErr && <p className="text-[11px] text-red-400">{editorErr}</p>}
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-zinc-800 px-4 py-3">
              <button onClick={closeEditor} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:text-zinc-100">
                Cancel
              </button>
              <button
                onClick={saveDraft}
                disabled={saving}
                className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
              >
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                {editingId ? 'Save changes' : 'Create archetype'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// A row of context-window preset chips; the selected value is highlighted.
function CtxRow({ opts, value, onChange }: { opts: number[]; value: number; onChange: (n: number) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {opts.map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => onChange(n)}
          className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium tabular-nums transition-colors ${
            value === n
              ? 'border-violet-500/50 bg-violet-500/15 text-violet-700 dark:text-violet-200'
              : 'border-zinc-700 bg-zinc-950 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200'
          }`}
        >
          {fmtCtx(n)}
        </button>
      ))}
    </div>
  )
}

// ── Setup wizard: collect one model + context and apply it to every tier. ──
function TierSetupWizard({ onSubmit, onClose }: {
  onSubmit: (v: WizardValues) => void
  onClose: () => void
}) {
  const [name, setName] = useState('My models')
  const [provider, setProvider] = useState(PROVIDERS[0])
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [inputCtx, setInputCtx] = useState(256 * 1024)
  const [outputCtx, setOutputCtx] = useState(32 * 1024)

  const valid = name.trim().length > 0 && model.trim().length > 0
  const submit = () => {
    if (!valid) return
    onSubmit({ name, provider, model, apiKey, baseUrl, inputCtx, outputCtx })
  }

  const inputCls = 'w-full rounded-lg border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none'
  const labelCls = 'mb-1 block text-[10px] font-medium uppercase tracking-wide text-zinc-500'

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/60 p-4" onClick={onClose}>
      <div className="my-8 w-full max-w-lg rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-3">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-violet-500/10 text-violet-400">
            <Wand2 className="h-3.5 w-3.5" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-zinc-200">Set up your models</h3>
            <p className="text-[11px] text-zinc-500">One model powers every tier — fine-tune per tier later.</p>
          </div>
          <button onClick={onClose} className="ml-auto rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 p-4">
          <div>
            <label className={labelCls}>Tier set name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. All Anthropic, Local Ollama" className={inputCls} autoFocus />
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div>
              <label className={labelCls}>Provider</label>
              <select value={provider} onChange={(e) => setProvider(e.target.value)} className={inputCls}>
                {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="sm:col-span-2">
              <label className={labelCls}>Model</label>
              <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" className={inputCls} />
            </div>
          </div>

          <div>
            <label className={labelCls}>API Key <span className="font-normal normal-case text-zinc-600">— optional, blank = server env key</span></label>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-…" className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>Base URL <span className="font-normal normal-case text-zinc-600">— optional</span></label>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="default endpoint" className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>Input context</label>
            <CtxRow opts={INPUT_CTX_OPTS} value={inputCtx} onChange={setInputCtx} />
          </div>

          <div>
            <label className={labelCls}>Output context</label>
            <CtxRow opts={OUTPUT_CTX_OPTS} value={outputCtx} onChange={setOutputCtx} />
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-zinc-800 px-4 py-3">
          <button onClick={onClose} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-300 hover:text-zinc-100">
            Skip
          </button>
          <button
            onClick={submit}
            disabled={!valid}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
          >
            <Check className="h-3.5 w-3.5" /> Create tier set
          </button>
        </div>
      </div>
    </div>
  )
}
