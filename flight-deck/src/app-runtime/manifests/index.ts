// Fetches agent-app manifests from Captain Claw. The framework owns
// manifest authoring; the renderer just binds to what it's served.

import type { AgentManifest } from '../types'
import { useAuthStore } from '../../stores/authStore'

export interface AppSummary {
  id: string
  name: string
  tagline?: string
}

function authHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function fetchJson(url: string): Promise<unknown> {
  const resp = await fetch(url, { credentials: 'include', headers: authHeaders() })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`)
  }
  return resp.json()
}

export async function fetchAppList(): Promise<AppSummary[]> {
  const data = (await fetchJson('/fd/apps')) as { apps?: AppSummary[] }
  return data.apps ?? []
}

export async function fetchManifest(agentId: string): Promise<AgentManifest | null> {
  try {
    const data = (await fetchJson(`/fd/apps/${encodeURIComponent(agentId)}`)) as {
      manifest?: AgentManifest
    }
    return data.manifest ?? null
  } catch {
    return null
  }
}
