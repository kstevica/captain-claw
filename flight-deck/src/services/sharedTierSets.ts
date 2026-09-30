// Team-default ("shared") tier sets.
//
// An admin publishes one or more of their own tier sets so teammates who never
// configured models run on the team's providers. Keys are stored server-side as
// the "@system" sentinel and resolved at run time — no secret ever reaches a
// browser. GET is available to any authenticated user; PUT is admin-only.

import { useAuthStore } from '../stores/authStore'
import type { TierSet } from './tierConfig'

export interface SharedTierSets {
  sets: TierSet[]
  defaultSetId: string | null
}

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

/** Team-default tier sets visible to every user (keys already @system-masked). */
export async function fetchSharedTierSets(): Promise<SharedTierSets> {
  try {
    const r = await fetch('/fd/settings/shared-tier-sets', {
      headers: authHeaders(), credentials: 'include',
    })
    if (!r.ok) return { sets: [], defaultSetId: null }
    const data = await r.json()
    return {
      sets: Array.isArray(data?.sets) ? data.sets : [],
      defaultSetId: data?.defaultSetId ?? null,
    }
  } catch {
    return { sets: [], defaultSetId: null }
  }
}

/** What publishing did to the team's provider keys (ids and tier names, never key values). */
export interface PublishResult {
  /** Providers whose key in the set became the team key (none existed). */
  teamKeysAdded: string[]
  /** Providers whose key in the set is not the existing team key (which stays). */
  teamKeysDiffer: string[]
  /** Tiers on a custom endpoint whose key isn't shared with the team. */
  teamKeysUnshared: string[]
  /** Providers the set uses that teammates still have no key for. */
  teamKeysMissing: string[]
}

/**
 * Publish tier sets as team defaults (admin only). The published copy never
 * holds a raw tier key; server-side, a key on a provider's own endpoint becomes
 * the team key for that provider when none is set, so teammates' agents can run
 * on it (it is written into those agents, whose owners can read it).
 */
export async function publishSharedTierSets(
  sets: TierSet[], defaultSetId: string | null,
): Promise<PublishResult> {
  const r = await fetch('/fd/admin/shared-tier-sets', {
    method: 'PUT',
    headers: authHeaders(),
    credentials: 'include',
    body: JSON.stringify({ sets, defaultSetId }),
  })
  if (!r.ok) {
    const detail = await r.text().catch(() => '')
    throw new Error(`${r.status} ${detail || r.statusText}`)
  }
  const data = await r.json().catch(() => ({}))
  const list = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : [])
  return {
    teamKeysAdded: list(data?.team_keys_added),
    teamKeysDiffer: list(data?.team_keys_differ),
    teamKeysUnshared: list(data?.team_keys_unshared),
    teamKeysMissing: list(data?.team_keys_missing),
  }
}
