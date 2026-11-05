// Quick start tier sets (GET /fd/tier-presets): a provider, its key, and that
// provider's current models on every tier. The catalog lives on the server in
// captain_claw/instructions/tier_presets.json, so a model release needs no
// frontend build. The set itself is built here, like every other tier set.

import { fdFetch } from './notifications'
import type { TierConfig } from './tierConfig'

export interface PresetTier {
  model: string
  input_ctx: number
  output_ctx: number
  vision?: boolean
  weak?: boolean
}

export interface TierPreset {
  id: string
  label: string
  set_name: string
  blurb: string
  provider: string
  base_url: string
  key_placeholder: string
  key_url: string
  // How this provider's keys start ([] / missing: they look like others').
  key_prefixes?: string[]
  tiers: Record<string, PresetTier>
}

export interface TierPresetCatalog {
  presets: TierPreset[]
  // When the models were last checked against the providers (YYYY-MM-DD).
  checked: string
}

export type KeyCheck =
  | { ok: true; checked?: boolean; missing: string[] }
  | { ok: false; refused?: boolean; unreachable?: boolean; status?: number }

let cached: Promise<TierPresetCatalog> | null = null

/** The quick starts this deck offers (none when it offers none or can't say). */
export function fetchTierPresets(): Promise<TierPresetCatalog> {
  if (!cached) {
    cached = fdFetch('/fd/tier-presets')
      .then((r) => (r.ok ? r.json() : {}))
      .then((d: { presets?: unknown; checked?: unknown }) => ({
        presets: Array.isArray(d?.presets) ? d.presets as TierPreset[] : [],
        checked: typeof d?.checked === 'string' ? d.checked : '',
      }))
      .catch(() => ({ presets: [], checked: '' }))
      .then((cat) => {
        if (cat.presets.length === 0) cached = null  // try again next time
        return cat
      })
  }
  return cached
}

/** The preset whose keys start the way this one does, if any says so. */
export function presetForKey(presets: TierPreset[], key: string): TierPreset | null {
  const k = key.trim()
  if (!k) return null
  return presets.find((p) => (p.key_prefixes || []).some((pre) => pre && k.startsWith(pre))) || null
}

/** Ask the preset's provider whether it takes this key. */
export async function checkPresetKey(preset: string, apiKey: string): Promise<KeyCheck> {
  try {
    const r = await fdFetch('/fd/tier-presets/check', {
      method: 'POST',
      body: JSON.stringify({ preset, api_key: apiKey }),
    })
    if (!r.ok) return { ok: false, unreachable: true, status: r.status }
    return await r.json()
  } catch {
    return { ok: false, unreachable: true }
  }
}

/** The preset as tier patches, the key on every tier. */
export function presetTiers(p: TierPreset, apiKey: string): Record<string, Partial<TierConfig>> {
  return Object.fromEntries(Object.entries(p.tiers).map(([k, t]) => [k, {
    provider: p.provider,
    model: t.model,
    api_key: apiKey,
    base_url: p.base_url || '',
    input_ctx: t.input_ctx,
    output_ctx: t.output_ctx,
    // Only what the preset says; otherwise the agent decides (auto / not weak).
    vision: t.vision || undefined,
    weak: t.weak || undefined,
  }]))
}

/** The set's name: the provider's, numbered when one by that name exists. */
export function presetSetName(p: TierPreset, existing: string[]): string {
  const taken = new Set(existing.map((n) => n.trim().toLowerCase()))
  const base = p.set_name || p.label
  if (!taken.has(base.toLowerCase())) return base
  for (let i = 2; ; i++) {
    const name = `${base} ${i}`
    if (!taken.has(name.toLowerCase())) return name
  }
}

/** Which models a tier set from this preset runs on, grouped: "Opus · Sonnet · Haiku" → labels. */
export function presetModelGroups(p: TierPreset, labelOf: (tier: string) => string, order: string[]): { model: string; tiers: string[] }[] {
  const groups: { model: string; tiers: string[] }[] = []
  for (const k of [...order, ...Object.keys(p.tiers).filter((t) => !order.includes(t))]) {
    const t = p.tiers[k]
    if (!t) continue
    const g = groups.find((x) => x.model === t.model)
    if (g) g.tiers.push(labelOf(k))
    else groups.push({ model: t.model, tiers: [labelOf(k)] })
  }
  return groups
}
