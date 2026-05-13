// Frontend client for the manifest authoring endpoints. The framework
// (Captain Claw) owns the actual generation; this is just a thin RPC layer.

import { useAuthStore } from '../stores/authStore'
import type { AgentManifest } from './types'

function authHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

export interface GenerateRequest {
  description: string
  mcp_server?: string
  base_agent_id?: string
  agent?: { host: string; port: number; auth: string; name?: string }
}

export interface GenerateResult {
  manifest: AgentManifest | null
  errors: string[]
}

export async function generateManifest(req: GenerateRequest): Promise<GenerateResult> {
  const resp = await fetch('/fd/apps/generate', {
    method: 'POST',
    credentials: 'include',
    headers: authHeaders(),
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    return { manifest: null, errors: [`${resp.status}: ${text || resp.statusText}`] }
  }
  return (await resp.json()) as GenerateResult
}

export interface SaveResult {
  ok: boolean
  path: string | null
  errors: string[]
}

export async function saveManifest(manifest: AgentManifest): Promise<SaveResult> {
  const resp = await fetch('/fd/apps/save', {
    method: 'POST',
    credentials: 'include',
    headers: authHeaders(),
    body: JSON.stringify({ manifest }),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    return { ok: false, path: null, errors: [`${resp.status}: ${text || resp.statusText}`] }
  }
  return (await resp.json()) as SaveResult
}

export async function deleteManifest(agentId: string): Promise<{ ok: boolean; error?: string }> {
  const resp = await fetch(`/fd/apps/${encodeURIComponent(agentId)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: authHeaders(),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    return { ok: false, error: `${resp.status}: ${text || resp.statusText}` }
  }
  return { ok: true }
}
