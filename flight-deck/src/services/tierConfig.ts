// Shared model-tier configuration + curated archetype registry.
//
// Both the Library page (the editor + archetype gallery) and Agent Forge (which
// consumes the saved tiers to run decomposition and resolve models at spawn)
// read from here, so there is ONE source of truth for the per-user tier config.

import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuthStore } from '../stores/authStore'
import { queueSave, registerHydrator, fetchSettings } from './settingsSync'

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
}
export type TierMap = Record<string, TierConfig>

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
export interface Archetype {
  id: string
  family: string
  role: string
  lead?: boolean
  cognitive_mode: string
  tier: string
  tools: string[]
  description: string
  fleet_instructions: string
  keywords?: string[]
  reliability_seed?: number
  // 'user' archetypes are editable/deletable; 'base' come from the JSON file.
  source?: 'base' | 'user'
  overrides?: boolean
}

export interface ArchetypeRegistry {
  tiers: Record<string, TierDef>
  base_tools?: string[]
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

export const PROVIDERS = ['anthropic', 'openai', 'ollama', 'gemini', 'xai', 'openrouter', 'litert']

export const TIER_ORDER = ['reason', 'balanced', 'fast', 'longctx']

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

export function saveSets(sets: TierSet[], activeSetId: string) {
  persistSetting(TIERS_KEY, JSON.stringify({ sets, activeSetId }))
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
    }
  }
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
  // Multi-set management (Library page).
  sets: TierSet[]
  activeSetId: string
  setActiveSet: (id: string) => void
  addSet: (name?: string) => string
  duplicateSet: (id: string) => string
  renameSet: (id: string, name: string) => void
  deleteSet: (id: string) => void
}

/**
 * Load (server-authoritative when auth is on), seed-from-registry, and persist
 * the per-user tier sets + the archetype registry. Used by the Library editor
 * (which manages sets) and by Agent Forge / Basna (which read the active set),
 * so there is ONE persisted source of truth.
 */
export function useTierConfig(): TierConfigState {
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

  // Authoritative load when auth is on, then mark bootstrapped.
  useEffect(() => {
    if (!useAuthStore.getState().authEnabled) return
    let cancelled = false
    fetchSettings().then((s) => {
      if (cancelled) return
      const blob = parseSetsBlob(s[TIERS_KEY], s[ENV_VARS_KEY])
      if (blob) { setSets(blob.sets); setActiveSetId(blob.activeSetId) }
    }).catch(() => {}).finally(() => { if (!cancelled) setBootstrapped(true) })
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

  // Seed one "Default" set from registry defaults if the user has none saved.
  useEffect(() => {
    if (!bootstrapped || !registry || sets.length > 0) return
    const def = freshSet(registry, loadLegacyConfig(), 'Default')
    setSets([def]); setActiveSetId(def.id)
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

  return {
    tiers, setTiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, refreshRegistry, bootstrapped, updateTier,
    sets, activeSetId, setActiveSet, addSet, duplicateSet, renameSet, deleteSet,
  }
}
