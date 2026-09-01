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

/** Publish tier sets as team defaults (admin only). API keys are masked server-side. */
export async function publishSharedTierSets(
  sets: TierSet[], defaultSetId: string | null,
): Promise<void> {
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
}
