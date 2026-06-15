// Shared model-tier configuration + curated archetype registry.
//
// Both the Library page (the editor + archetype gallery) and Agent Forge (which
// consumes the saved tiers to run decomposition and resolve models at spawn)
// read from here, so there is ONE source of truth for the per-user tier config.

import { useState, useEffect } from 'react'
import { useAuthStore } from '../stores/authStore'
import { queueSave, registerHydrator, fetchSettings } from './settingsSync'

// ── Persistence keys (per-user / multi-tenant via /fd/settings) ──────

// Legacy single-model config — read only, to migrate the API key forward.
export const FORGE_CONFIG_KEY = 'fd:forge-llm-config'
// Per-user tier configuration: a model definition per tier + which tier drives
// the Forge decomposition call.
export const TIERS_KEY = 'fd:forge-tiers'
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

// Curated archetype registry served by GET /fd/archetypes.
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
}

export interface ArchetypeRegistry {
  tiers: Record<string, TierDef>
  archetypes: Archetype[]
}

export interface EnvVar { key: string; value: string }

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

export function loadTierSettings(): { tiers: TierMap; forgeTier: string } | null {
  try {
    const raw = localStorage.getItem(TIERS_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    if (p && p.tiers && typeof p.tiers === 'object') {
      return { tiers: p.tiers as TierMap, forgeTier: p.forgeTier || 'reason' }
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

export function saveTierSettings(tiers: TierMap, forgeTier: string) {
  persistSetting(TIERS_KEY, JSON.stringify({ tiers, forgeTier }))
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
  tiers: TierMap
  setTiers: React.Dispatch<React.SetStateAction<TierMap>>
  forgeTier: string
  setForgeTier: (t: string) => void
  envVars: EnvVar[]
  setEnvVars: React.Dispatch<React.SetStateAction<EnvVar[]>>
  registry: ArchetypeRegistry | null
  bootstrapped: boolean
  updateTier: (key: string, patch: Partial<TierConfig>) => void
}

/**
 * Load (server-authoritative when auth is on), seed-from-registry, and persist
 * the per-user tier config + env vars + the archetype registry. Used by both the
 * Library editor and Agent Forge so they share one persisted source of truth.
 */
export function useTierConfig(): TierConfigState {
  const [tiers, setTiers] = useState<TierMap>(() => loadTierSettings()?.tiers || {})
  const [forgeTier, setForgeTier] = useState<string>(() => loadTierSettings()?.forgeTier || 'reason')
  // Gate seeding/persisting until the authoritative server load completes, so a
  // slow fetch can't let the seed/persist effects overwrite real saved data.
  const [bootstrapped, setBootstrapped] = useState(() => !useAuthStore.getState().authEnabled)
  const [envVars, setEnvVars] = useState<EnvVar[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(ENV_VARS_KEY) || '[]')
      return Array.isArray(saved) ? saved : []
    } catch { return [] }
  })
  const [registry, setRegistry] = useState<ArchetypeRegistry | null>(null)

  // Authoritative load when auth is on, then mark bootstrapped.
  useEffect(() => {
    if (!useAuthStore.getState().authEnabled) return
    let cancelled = false
    fetchSettings().then((s) => {
      if (cancelled) return
      const rawTiers = s[TIERS_KEY]
      if (rawTiers) {
        try {
          const p = JSON.parse(rawTiers)
          if (p?.tiers && typeof p.tiers === 'object') {
            setTiers(p.tiers)
            setForgeTier(p.forgeTier || 'reason')
          }
        } catch { /* ignore */ }
      }
      const rawEnv = s[ENV_VARS_KEY]
      if (rawEnv) {
        try {
          const a = JSON.parse(rawEnv)
          if (Array.isArray(a)) setEnvVars(a)
        } catch { /* ignore */ }
      }
    }).catch(() => {}).finally(() => { if (!cancelled) setBootstrapped(true) })
    return () => { cancelled = true }
  }, [])

  // Load the curated archetype registry (tiers + gallery).
  useEffect(() => {
    fetch('/fd/archetypes').then((r) => r.json()).then((reg) => {
      if (reg && Array.isArray(reg.archetypes)) setRegistry(reg)
    }).catch(() => {})
  }, [])

  // Seed tiers from registry defaults once, if the user has no saved config.
  useEffect(() => {
    if (!bootstrapped || !registry || Object.keys(tiers).length > 0) return
    setTiers(seedTiers(registry, loadLegacyConfig()))
  }, [bootstrapped, registry, tiers])

  // Persist tier settings whenever they change (gated on bootstrapped).
  useEffect(() => {
    if (!bootstrapped || Object.keys(tiers).length === 0) return
    saveTierSettings(tiers, forgeTier)
  }, [tiers, forgeTier, bootstrapped])

  // Persist env vars, write-through to localStorage.
  useEffect(() => {
    if (!bootstrapped) return
    persistSetting(ENV_VARS_KEY, JSON.stringify(envVars))
  }, [envVars, bootstrapped])

  const updateTier = (key: string, patch: Partial<TierConfig>) =>
    setTiers((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }))

  return { tiers, setTiers, forgeTier, setForgeTier, envVars, setEnvVars, registry, bootstrapped, updateTier }
}
