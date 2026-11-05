// Shared model-tier configuration + curated archetype registry.
//
// Both the Library page (the editor + archetype gallery) and Agent Forge (which
// consumes the saved tiers to run decomposition and resolve models at spawn)
// read from here, so there is ONE source of truth for the per-user tier config.

import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuthStore } from '../stores/authStore'
import { queueSave, registerHydrator, fetchSettingsOrNull } from './settingsSync'
import { fetchSharedTierSets } from './sharedTierSets'

// ── Persistence keys (per-user / multi-tenant via /fd/settings) ──────

// Legacy single-model config — read only, to migrate the API key forward.
export const FORGE_CONFIG_KEY = 'fd:forge-llm-config'
// Per-user tier configuration. Now holds MULTIPLE named sets plus the id of the
// active one: `{ sets: TierSet[], activeSetId }`. Older single-set payloads
// (`{ tiers, forgeTier }`) are migrated forward on read. Each set is a
// self-contained profile: its own 4 tiers, Forge tier, and additional API keys.
export const TIERS_KEY = 'fd:forge-tiers'
// Legacy global env vars — read only, folded into the migrated "Default" set.
export const ENV_VARS_KEY = 'fd:forge-env-vars'

// ── Types ────────────────────────────────────────────────────────────

export interface LegacyForgeConfig {
  provider?: string
  model?: string
  api_key?: string
  base_url?: string
}

// One concrete model definition for a tier.
export interface TierConfig {
  provider: string
  model: string
  api_key: string
  base_url: string
  input_ctx: number   // max input context window
  output_ctx: number  // max output (completion) tokens
  // Marks this slot as a fast/weak model so a Vatra run leans on the deterministic
  // backstops (surfaced to the worker as CLAW_MODEL_WEAK). Optional; omitted =
  // not weak. An explicit signal — never inferred from the model name.
  weak?: boolean
  // The model sees images itself (the agent's model.vision: on). Optional;
  // omitted = auto, decided by the agent. Local Qwen builds need it on.
  vision?: boolean
}
export type TierMap = Record<string, TierConfig>

/** The spawn's `vision` for an agent on this tier. */
export function tierVision(tc?: TierConfig): string {
  return tc?.vision ? 'on' : 'auto'
}

export interface TierDef {
  label: string
  use: string
  provider: string
  model: string
  base_url?: string
  input_ctx?: number
  output_ctx?: number
}

// Archetype registry served by GET /fd/archetypes — base set merged with the
// caller's own custom archetypes (the latter tagged source: 'user').
// The shelf above the family: a built-in one, or one an admin published for the deck.
// A person's own archetype may not have one yet.
export type ArchetypeCategory = 'generalist' | 'specialist' | 'personal' | `group-${string}`

export interface Archetype {
  id: string
  family: string
  category?: ArchetypeCategory
  // A key into the registry's `disclaimers`: legal, health or clinical.
  disclaimer?: string
  // A team archetype's own icon, by name.
  icon?: string
  role: string
  lead?: boolean
  cognitive_mode: string
  tier: string
  tools: string[]
  description: string
  fleet_instructions: string
  // '' | 'classic' = full agent loop; 'mrav' = micro small-model runtime (8k cap)
  runtime?: string
  keywords?: string[]
  reliability_seed?: number
  // 'user' archetypes are editable/deletable; 'base' come from the JSON file;
  // 'team' an admin published for the whole deck;
  // 'shared' were shared to you by another user (use-only, tagged with owner).
  source?: 'base' | 'user' | 'shared' | 'team'
  overrides?: boolean
  shared_owner?: string
  shared_owner_email?: string
  shared_owner_name?: string
  shared_permission?: string
}

/** A person looking for an archetype by role, family, what it does or a keyword. */
export function archetypeMatches(a: Archetype, query: string): boolean {
  const q = query.trim().toLowerCase()
  return !q || [a.role, a.id, a.family, a.description, ...(a.keywords || [])].some((t) => (t || '').toLowerCase().includes(q))
}

export interface ArchetypeRegistry {
  tiers: Record<string, TierDef>
  base_tools?: string[]
  disclaimers?: Record<string, string>
  // What this deck's admin added: shelves, and how its new families look.
  groups?: { id: `group-${string}`; label: string; hint: string }[]
  family_looks?: { name: string; icon: string; color: string }[]
  archetypes: Archetype[]
}

export interface EnvVar { key: string; value: string }

// A named, self-contained model profile: the 4 tier definitions, which tier
// drives the Forge decomposition, and its own additional API keys. The user can
// keep several (e.g. "All Anthropic", "Local Ollama") and switch the active one.
export interface TierSet {
  id: string
  name: string
  tiers: TierMap
  forgeTier: string
  envVars: EnvVar[]
}

interface SetsBlob { sets: TierSet[]; activeSetId: string }

// ── Constants ────────────────────────────────────────────────────────

export const PROVIDERS = ['anthropic', 'openai', 'ollama', 'gemini', 'antigravity-cli', 'xai', 'openrouter', 'litert', 'browser']

/** The providers an agent can run on (the agent runtime's create_provider),
 *  named for people — every picker that sets an agent's provider offers these. */
export const LLM_PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Gemini (API)' },
  { value: 'antigravity-cli', label: 'Google (Antigravity subscription, text only)' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'xai', label: 'xAI' },
  { value: 'litert', label: 'LiteRT (local Gemma)' },
  { value: 'browser', label: 'Browser tab (Local inference)' },
] as const

// Fixed context-size menus — every panel that edits input/output context
// uses these (no freeform token counts). Powers of two where natural,
// decimal for the model-family sizes (100k, 160k).
export const INPUT_CTX_OPTIONS = [
  4096, 8192, 16384, 32768, 65536, 100000, 131072, 160000,
  262144, 524288, 786432, 1048576,
]
export const OUTPUT_CTX_OPTIONS = [
  1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 131072, 262144,
]

export function fmtCtxTokens(n: number): string {
  if (n >= 1024 * 1024 && n % (1024 * 1024) === 0) return `${n / (1024 * 1024)}M`
  if (n % 1024 === 0) return `${n / 1024}k`
  if (n % 1000 === 0) return `${n / 1000}k`
  return String(n)
}

// Reasoning effort is carried as a model-name suffix (`gpt-6-luna-none`,
// `gpt-5-high`) — the backend strips it and sends `reasoning_effort`, so every
// spawn path honors it with no extra field. Mirrors _REASONING_EFFORT_VALUES in
// captain_claw/llm/__init__.py. Only OpenAI and DeepSeek models use it; 'none'
// turns reasoning off (required by gpt-6-luna to call tools).
export const REASONING_EFFORTS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const
export const REASONING_EFFORT_PROVIDERS = ['openai', 'deepseek']

export function splitReasoningEffort(model: string): { base: string; effort: string } {
  const m = model.trim()
  const lower = m.toLowerCase()
  for (const sep of ['-', '_']) {
    for (const effort of REASONING_EFFORTS) {
      if (lower.endsWith(`${sep}${effort}`)) {
        return { base: m.slice(0, -(effort.length + 1)), effort }
      }
    }
  }
  return { base: m, effort: '' }
}

// '' effort = the model's own default (no suffix).
export function withReasoningEffort(model: string, effort: string): string {
  const { base } = splitReasoningEffort(model)
  return effort ? `${base}-${effort}` : base
}

export const TIER_ORDER = ['reason', 'balanced', 'fast', 'longctx', 'coding', 'vision', 'micro']

export const DEFAULT_TOOLS = [
  'shell', 'read', 'write', 'glob', 'edit', 'web_fetch', 'web_search',
  'personality', 'playbooks', 'scripts',
]

// ── Load / persist helpers ───────────────────────────────────────────

export function loadLegacyConfig(): LegacyForgeConfig {
  try {
    return JSON.parse(localStorage.getItem(FORGE_CONFIG_KEY) || '{}')
  } catch {
    return {}
  }
}

// A short unique id for a set. crypto.randomUUID where available, else a cheap
// random fallback (collisions are harmless — only used to key local sets).
export function newSetId(): string {
  try { return crypto.randomUUID() } catch { /* older browsers */ }
  return 'set-' + Math.random().toString(36).slice(2, 10)
}

// Parse the persisted tier blob into the multi-set shape, migrating the older
// single-set payload (`{ tiers, forgeTier }` + a separate global env-vars blob)
// forward into one "Default" set. Returns null when there's nothing usable.
export function parseSetsBlob(rawTiers?: string | null, rawEnv?: string | null): SetsBlob | null {
  if (!rawTiers) return null
  try {
    const p = JSON.parse(rawTiers)
    if (Array.isArray(p?.sets) && p.sets.length > 0) {
      const sets = (p.sets as TierSet[]).map((s) => ({
        id: s.id || newSetId(),
        name: s.name || 'Set',
        tiers: s.tiers || {},
        forgeTier: s.forgeTier || 'reason',
        envVars: Array.isArray(s.envVars) ? s.envVars : [],
      }))
      const activeSetId = sets.some((s) => s.id === p.activeSetId) ? p.activeSetId : sets[0].id
      return { sets, activeSetId }
    }
    // Legacy single-set shape — wrap it, folding in the old global env vars.
    if (p?.tiers && typeof p.tiers === 'object') {
      let env: EnvVar[] = []
      try { const a = JSON.parse(rawEnv || '[]'); if (Array.isArray(a)) env = a } catch { /* ignore */ }
      const set: TierSet = {
        id: newSetId(), name: 'Default', tiers: p.tiers as TierMap,
        forgeTier: p.forgeTier || 'reason', envVars: env,
      }
      return { sets: [set], activeSetId: set.id }
    }
  } catch { /* ignore */ }
  return null
}

// Write-through: always update localStorage (fast path / non-auth store) AND,
// when auth is on, queue a debounced save to the per-user server settings.
export function persistSetting(key: string, val: string) {
  localStorage.setItem(key, val)
  if (useAuthStore.getState().authEnabled) queueSave(key, val)
}

// Fired on window whenever the saved tier sets change (hasTierSets readers).
export const TIER_SETS_EVENT = 'fd:tier-sets'

export function saveSets(sets: TierSet[], activeSetId: string) {
  const blob = JSON.stringify({ sets, activeSetId })
  persistSetting(TIERS_KEY, blob)
  // A deck without sign-in keeps settings in the browser only, but Flight Deck
  // itself resolves an archetype's model from this setting when it creates an
  // agent (the kiosk's "New agent"): its one user's sets go to the server too.
  if (useAuthStore.getState().authEnabled === false) queueSave(TIERS_KEY, blob)
  window.dispatchEvent(new Event(TIER_SETS_EVENT))
}

/** Whether this user has saved a tier set of their own (the local mirror of
 *  their settings — current once signed-in settings are hydrated). */
export function hasTierSets(): boolean {
  try {
    const blob = parseSetsBlob(localStorage.getItem(TIERS_KEY), localStorage.getItem(ENV_VARS_KEY))
    return !!blob && blob.sets.length > 0
  } catch { return false }
}

/** How many provider keys this user's own sets carry (a provider on its own
 *  endpoint counts apart: DeepSeek through "openai" is not OpenAI). */
export function tierSetKeyCount(): number {
  try {
    const blob = parseSetsBlob(localStorage.getItem(TIERS_KEY), localStorage.getItem(ENV_VARS_KEY))
    const seen = new Set<string>()
    for (const s of blob?.sets || []) {
      for (const t of Object.values(s.tiers)) {
        if (t.api_key?.trim()) seen.add(`${t.provider}|${(t.base_url || '').trim()}|${t.api_key.trim()}`)
      }
    }
    return seen.size
  } catch { return 0 }
}

// Deep-copy a set (tiers + env vars are nested) so edits to a duplicate don't
// alias the source. A fresh id/name are applied by the caller.
export function cloneSet(s: TierSet): TierSet {
  return {
    id: s.id, name: s.name, forgeTier: s.forgeTier,
    tiers: Object.fromEntries(Object.entries(s.tiers).map(([k, v]) => [k, { ...v }])),
    envVars: s.envVars.map((e) => ({ ...e })),
  }
}

// Seed a tier map from the registry defaults, carrying the user's existing key
// and provider forward. The legacy single model is replicated across tiers so
// spawns work immediately; the user then differentiates each tier's model.
export function seedTiers(registry: ArchetypeRegistry, legacy: LegacyForgeConfig): TierMap {
  const out: TierMap = {}
  for (const t of TIER_ORDER) {
    const def = registry.tiers[t]
    if (!def) continue
    out[t] = {
      provider: legacy.provider || def.provider,
      model: legacy.model || def.model,
      api_key: legacy.api_key || '',
      base_url: legacy.base_url || def.base_url || '',
      input_ctx: def.input_ctx || 200000,
      output_ctx: def.output_ctx || 32768,
    }
  }
  return out
}

// Backfill a set's tier map with any registry tiers it doesn't have yet (e.g.
// `coding`/`vision` added after the set was first seeded). Missing tiers inherit
// the set's existing provider/key/base_url from a representative sibling — so a
// new tier in an all-DeepSeek set is wired to DeepSeek, not empty Anthropic —
// while ctx limits come from the registry def. When the sibling is on the tier's
// canonical provider we use the curated default model; otherwise we carry the
// sibling's model forward as a starting point. Returns the SAME reference when
// nothing is missing, so callers can cheaply skip a re-render/persist.
export function backfillTierMap(tiers: TierMap, registry: ArchetypeRegistry): TierMap {
  // Preferred credential donor per new tier, then sensible fallbacks.
  const donorFor = (t: string): TierConfig | undefined => {
    const pref = t === 'coding' ? 'reason' : t === 'vision' ? 'balanced' : 'balanced'
    return tiers[pref] || tiers.balanced || tiers.reason || Object.values(tiers)[0]
  }
  let changed = false
  const out: TierMap = { ...tiers }
  for (const t of TIER_ORDER) {
    if (out[t]) continue
    const def = registry.tiers[t]
    if (!def) continue
    const sib = donorFor(t)
    const sameProvider = !!sib && sib.provider === def.provider
    out[t] = {
      provider: sib?.provider || def.provider,
      model: sameProvider ? def.model : (sib?.model || def.model),
      api_key: sib?.api_key || '',
      base_url: sib?.base_url || def.base_url || '',
      input_ctx: def.input_ctx || 200000,
      output_ctx: def.output_ctx || 32768,
    }
    changed = true
  }
  return changed ? out : tiers
}

// A set is "unconfigured" if no tier carries user credentials — i.e. it's still
// the untouched registry seed. Used to decide whether to auto-open the setup
// wizard on a fresh install, and whether that wizard replaces the lone seed set
// instead of stacking a second one.
export function isSetUnconfigured(s: TierSet): boolean {
  const tiers = Object.values(s.tiers)
  if (tiers.length === 0) return true
  return tiers.every((t) => !t.api_key?.trim() && !t.base_url?.trim())
}

// Build a brand-new set seeded from the registry defaults (carrying the legacy
// single-model key/provider forward), with empty additional API keys.
export function freshSet(
  registry: ArchetypeRegistry | null, legacy: LegacyForgeConfig, name: string,
): TierSet {
  return {
    id: newSetId(),
    name,
    tiers: registry ? seedTiers(registry, legacy) : {},
    forgeTier: 'reason',
    envVars: [],
  }
}

// Mirror server values into localStorage on login hydration so the fast path is
// warm. Registered once at module load (the module is a singleton).
registerHydrator((settings) => {
  for (const key of [TIERS_KEY, ENV_VARS_KEY]) {
    const raw = settings[key]
    if (raw) {
      try { JSON.parse(raw); localStorage.setItem(key, raw) } catch { /* ignore */ }
    } else {
      // The server is the authority (hydrators run only after a good fetch):
      // a previous user's sets left in this browser are not this user's.
      localStorage.removeItem(key)
    }
  }
  // hasTierSets readers that showed before the settings arrived look again.
  queueMicrotask(() => window.dispatchEvent(new Event(TIER_SETS_EVENT)))
})

// ── Shared state hook ────────────────────────────────────────────────

export interface TierConfigState {
  // Active-set view — what Forge/Basna/Library spawns consume. The setters all
  // mutate the active set, so existing read-only consumers need no changes.
  tiers: TierMap
  setTiers: React.Dispatch<React.SetStateAction<TierMap>>
  forgeTier: string
  setForgeTier: (t: string) => void
  envVars: EnvVar[]
  setEnvVars: React.Dispatch<React.SetStateAction<EnvVar[]>>
  registry: ArchetypeRegistry | null
  refreshRegistry: () => void
  bootstrapped: boolean
  updateTier: (key: string, patch: Partial<TierConfig>) => void
  // Create a fully-configured set where every tier shares one model + context
  // (the setup wizard). Replaces the lone untouched seed set on a fresh install
  // rather than stacking a second one. Returns the resulting set id.
  // `replaceSeed: false` (the console): always a new set.
  setupSet: (name: string, apply: Partial<TierConfig>, opts?: { replaceSeed?: boolean }) => string
  // The same with a model per tier (a quick start's provider preset).
  setupPresetSet: (name: string, perTier: Record<string, Partial<TierConfig>>, opts?: { replaceSeed?: boolean }) => string
  // Multi-set management (Library page).
  sets: TierSet[]
  activeSetId: string
  setActiveSet: (id: string) => void
  addSet: (name?: string) => string
  duplicateSet: (id: string) => string
  renameSet: (id: string, name: string) => void
  deleteSet: (id: string) => void
  // Admin-published team defaults (read-only view). Keys are @system sentinels.
  sharedSets: TierSet[]
  // Copy a team-default set into the user's own editable sets and activate it.
  adoptSharedSet: (id: string) => string
}

/**
 * Load (server-authoritative when auth is on), seed-from-registry, and persist
 * the per-user tier sets + the archetype registry. Used by the Library editor
 * (which manages sets) and by Agent Forge / Basna (which read the active set),
 * so there is ONE persisted source of truth.
 */
/**
 * `seed: false` — the Admin console: never create a starting set for someone
 * who has none (they create their first one there), where everywhere else a
 * user with no set gets the team default or the registry's.
 */
export function useTierConfig({ seed = true }: { seed?: boolean } = {}): TierConfigState {
  // Compute the initial blob once from localStorage (avoids generating mismatched
  // migration ids across two lazy initializers).
  const initRef = useRef<SetsBlob | null>(null)
  if (initRef.current === null) {
    initRef.current = parseSetsBlob(
      localStorage.getItem(TIERS_KEY), localStorage.getItem(ENV_VARS_KEY),
    ) || { sets: [], activeSetId: '' }
  }
  const [sets, setSets] = useState<TierSet[]>(() => initRef.current!.sets)
  const [activeSetId, setActiveSetId] = useState<string>(() => initRef.current!.activeSetId)
  // Gate seeding/persisting until the authoritative server load completes, so a
  // slow fetch can't let the seed/persist effects overwrite real saved data.
  const [bootstrapped, setBootstrapped] = useState(() => !useAuthStore.getState().authEnabled)
  const [registry, setRegistry] = useState<ArchetypeRegistry | null>(null)
  // Admin-published team defaults + which one is the default (read-only view).
  const [sharedSets, setSharedSets] = useState<TierSet[]>([])
  const [sharedDefaultId, setSharedDefaultId] = useState<string | null>(null)

  // Authoritative load when auth is on, then mark bootstrapped.
  useEffect(() => {
    if (!useAuthStore.getState().authEnabled) return
    let cancelled = false
    fetchSettingsOrNull().then((s) => {
      if (cancelled || s === null) return   // unreadable: keep this browser's copy
      const blob = parseSetsBlob(s[TIERS_KEY], s[ENV_VARS_KEY])
      // None on the server: none for this user, whatever this browser held.
      setSets(blob ? blob.sets : [])
      setActiveSetId(blob ? blob.activeSetId : '')
    }).catch(() => {}).finally(() => { if (!cancelled) setBootstrapped(true) })
    return () => { cancelled = true }
  }, [])

  // Load the admin-published team-default tier sets (keys are @system sentinels).
  useEffect(() => {
    let cancelled = false
    fetchSharedTierSets().then((res) => {
      if (cancelled) return
      setSharedSets(res.sets)
      setSharedDefaultId(res.defaultSetId)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  // Load the archetype registry (tiers + merged gallery: base + the user's own).
  // Sent authenticated so the server can include this user's custom archetypes.
  const refreshRegistry = useCallback(() => {
    const { token, authEnabled } = useAuthStore.getState()
    const headers: Record<string, string> = {}
    if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
    fetch('/fd/archetypes', { headers, credentials: 'include' })
      .then((r) => r.json())
      .then((reg) => { if (reg && Array.isArray(reg.archetypes)) setRegistry(reg) })
      .catch(() => {})
  }, [])

  useEffect(() => { refreshRegistry() }, [refreshRegistry])

  // Seed a set when the user has none saved. Prefer the admin-published team
  // default (so a teammate who configured nothing runs on the team's models)
  // over an empty registry seed; fall back to the registry default otherwise.
  useEffect(() => {
    if (!seed || !bootstrapped || sets.length > 0) return
    const teamDefault = sharedSets.find((s) => s.id === sharedDefaultId) || sharedSets[0]
    if (teamDefault) {
      const adopted: TierSet = { ...cloneSet(teamDefault), id: newSetId(), name: `${teamDefault.name} (team)` }
      setSets([adopted]); setActiveSetId(adopted.id)
      return
    }
    if (!registry) return
    const def = freshSet(registry, loadLegacyConfig(), 'Default')
    setSets([def]); setActiveSetId(def.id)
  }, [seed, bootstrapped, registry, sets, sharedSets, sharedDefaultId])

  // Backfill saved sets with any registry tiers added after they were seeded
  // (e.g. `coding`/`vision`), so existing users get the new tier cards without
  // losing their edits. No-op (stable ref) once every set has every tier.
  useEffect(() => {
    if (!bootstrapped || !registry || sets.length === 0) return
    let changed = false
    const next = sets.map((s) => {
      const filled = backfillTierMap(s.tiers, registry)
      if (filled === s.tiers) return s
      changed = true
      return { ...s, tiers: filled }
    })
    if (changed) setSets(next)
  }, [bootstrapped, registry, sets])

  // Keep the active id valid (e.g. after a delete, or a migration id mismatch).
  useEffect(() => {
    if (sets.length > 0 && !sets.some((s) => s.id === activeSetId)) setActiveSetId(sets[0].id)
  }, [sets, activeSetId])

  // Persist whenever the sets or the active selection change (gated on bootstrap).
  useEffect(() => {
    if (!bootstrapped || sets.length === 0) return
    saveSets(sets, activeSetId)
  }, [sets, activeSetId, bootstrapped])

  // ── Active-set view + mutators ──────────────────────────────────────
  const activeSet = sets.find((s) => s.id === activeSetId) || sets[0] || null
  const tiers = activeSet?.tiers || {}
  const forgeTier = activeSet?.forgeTier || 'reason'
  const envVars = activeSet?.envVars || []

  // Apply an updater to the active set, immutably.
  const patchActive = (updater: (s: TierSet) => TierSet) =>
    setSets((prev) => prev.map((s) => (s.id === (activeSet?.id ?? activeSetId) ? updater(s) : s)))

  const setTiers: React.Dispatch<React.SetStateAction<TierMap>> = (action) =>
    patchActive((s) => ({
      ...s,
      tiers: typeof action === 'function'
        ? (action as (p: TierMap) => TierMap)(s.tiers) : action,
    }))
  const setForgeTier = (t: string) => patchActive((s) => ({ ...s, forgeTier: t }))
  const setEnvVars: React.Dispatch<React.SetStateAction<EnvVar[]>> = (action) =>
    patchActive((s) => ({
      ...s,
      envVars: typeof action === 'function'
        ? (action as (p: EnvVar[]) => EnvVar[])(s.envVars) : action,
    }))
  const updateTier = (key: string, patch: Partial<TierConfig>) =>
    patchActive((s) => ({ ...s, tiers: { ...s.tiers, [key]: { ...s.tiers[key], ...patch } } }))

  // ── Set management ──────────────────────────────────────────────────
  const setActiveSet = (id: string) => setActiveSetId(id)
  const addSet = (name?: string): string => {
    const s = freshSet(registry, loadLegacyConfig(), name || `Set ${sets.length + 1}`)
    setSets((prev) => [...prev, s]); setActiveSetId(s.id)
    return s.id
  }
  // Wizard: one model + context applied to every tier. On a fresh install the
  // only set is the untouched auto-seed, so overwrite it in place; otherwise add.
  const setupSet = (name: string, apply: Partial<TierConfig>, opts: { replaceSeed?: boolean } = {}): string =>
    buildSet(name, () => apply, opts)
  // Quick start: each tier its own model. A tier the preset doesn't name runs
  // on its balanced tier, so every tier is on the chosen provider.
  const setupPresetSet = (name: string, perTier: Record<string, Partial<TierConfig>>, opts: { replaceSeed?: boolean } = {}): string =>
    buildSet(name, (k) => perTier[k] || perTier.balanced || Object.values(perTier)[0] || {}, opts, Object.keys(perTier))
  const buildSet = (
    name: string, patchFor: (tier: string) => Partial<TierConfig>,
    { replaceSeed = true }: { replaceSeed?: boolean }, extraTiers: string[] = [],
  ): string => {
    const template = freshSet(registry, loadLegacyConfig(), name)
    const blank: TierConfig = { provider: '', model: '', api_key: '', base_url: '', input_ctx: 200000, output_ctx: 32768 }
    const baseTiers: TierMap = { ...template.tiers }
    for (const t of [...(Object.keys(baseTiers).length === 0 ? TIER_ORDER : []), ...extraTiers]) {
      if (TIER_ORDER.includes(t) && !baseTiers[t]) baseTiers[t] = { ...blank }
    }
    const tiers: TierMap = Object.fromEntries(
      Object.entries(baseTiers).map(([k, v]) => [k, { ...v, ...patchFor(k) }]),
    )
    const replace = replaceSeed && sets.length === 1 && isSetUnconfigured(sets[0])
    const id = replace ? sets[0].id : template.id
    const newSet: TierSet = { ...template, id, name, tiers }
    const next = replace ? sets.map((s) => (s.id === id ? newSet : s)) : [...sets, newSet]
    setSets(next)
    setActiveSetId(id)
    // Saved now, not by the persist effect: the first-launch wizard moves on to
    // Quick chat in the same click, and the page unmounts before effects run.
    if (bootstrapped) saveSets(next, id)
    return id
  }
  const duplicateSet = (id: string): string => {
    const src = sets.find((s) => s.id === id) || activeSet
    if (!src) return ''
    const copy = { ...cloneSet(src), id: newSetId(), name: `${src.name} copy` }
    setSets((prev) => [...prev, copy]); setActiveSetId(copy.id)
    return copy.id
  }
  const renameSet = (id: string, name: string) =>
    setSets((prev) => prev.map((s) => (s.id === id ? { ...s, name } : s)))
  const deleteSet = (id: string) => {
    if (sets.length <= 1) return  // always keep at least one set
    const next = sets.filter((s) => s.id !== id)
    setSets(next)
    if (activeSetId === id) setActiveSetId(next[0].id)
  }
  // Copy a team-default set into the user's own editable sets and activate it.
  const adoptSharedSet = (id: string): string => {
    const src = sharedSets.find((s) => s.id === id)
    if (!src) return ''
    const copy: TierSet = { ...cloneSet(src), id: newSetId(), name: `${src.name} (team)` }
    setSets((prev) => [...prev, copy]); setActiveSetId(copy.id)
    return copy.id
  }

  return {
    tiers, setTiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, refreshRegistry, bootstrapped, updateTier, setupSet,
    setupPresetSet, sets, activeSetId, setActiveSet, addSet, duplicateSet, renameSet, deleteSet,
    sharedSets, adoptSharedSet,
  }
}
