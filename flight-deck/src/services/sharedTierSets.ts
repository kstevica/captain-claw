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

/** What publishing did to the team's keys (provider ids and hostnames, never key values). */
export interface PublishResult {
  /** Providers whose key in the set became the team key (none existed). */
  teamKeysAdded: string[]
  /** Providers whose team key was replaced by the different one in the set. */
  teamKeysUpdated: string[]
  /** Custom endpoints (hosts) whose key is now shared, for that endpoint only. */
  teamEndpointsShared: string[]
  /** Providers / endpoint hosts the set holds more than one key for (one is used). */
  teamKeysConflict: string[]
  /** Providers the set uses that teammates still have no key for. */
  teamKeysMissing: string[]
  /** Existing agents that had no model key and were given the team's. */
  agentsGivenKey: number
}

/**
 * Publish tier sets as team defaults (admin only). The published copy never
 * holds a raw tier key; server-side each key becomes the team key for its
 * endpoint — the provider's own, or the tier's custom base URL — so teammates'
 * agents run on it (it is written into those agents, whose owners can read it).
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
    teamKeysUpdated: list(data?.team_keys_updated),
    teamEndpointsShared: list(data?.team_endpoints_shared),
    teamKeysConflict: list(data?.team_keys_conflict),
    teamKeysMissing: list(data?.team_keys_missing),
    agentsGivenKey: Number(data?.agents_given_key) || 0,
  }
}
