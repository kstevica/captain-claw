// Per-tenant archetype management — talks to /fd/archetypes (CRUD + generate).
// The merged registry itself (base + the user's own) is fetched by useTierConfig.
import { useAuthStore, refreshAccessToken } from '../stores/authStore'

// The editable shape of a user archetype, matching the backend ArchetypeBody.
export interface ArchetypeInput {
  archetype_id: string
  role: string
  family: string
  description: string
  cognitive_mode: string
  tier: string
  tools: string[]
  fleet_instructions: string
  keywords: string[]
  lead: boolean
  reliability_seed: number
}

function authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

// fetch with one transparent token-refresh retry on 401, mirroring callForge.
async function authFetch(path: string, init: RequestInit): Promise<Response> {
  const { authEnabled } = useAuthStore.getState()
  let res = await fetch(path, { ...init, headers: authHeaders(), credentials: 'include' })
  if (res.status === 401 && authEnabled) {
    if (await refreshAccessToken()) {
      res = await fetch(path, { ...init, headers: authHeaders(), credentials: 'include' })
    }
  }
  return res
}

async function jsonOrThrow(res: Response) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

export async function createArchetype(body: ArchetypeInput) {
  return jsonOrThrow(await authFetch('/fd/archetypes', {
    method: 'POST', body: JSON.stringify(body),
  }))
}

export async function updateArchetype(archetypeId: string, body: ArchetypeInput) {
  return jsonOrThrow(await authFetch(`/fd/archetypes/${encodeURIComponent(archetypeId)}`, {
    method: 'PUT', body: JSON.stringify(body),
  }))
}

export async function deleteArchetype(archetypeId: string) {
  return jsonOrThrow(await authFetch(`/fd/archetypes/${encodeURIComponent(archetypeId)}`, {
    method: 'DELETE',
  }))
}

// Draft an archetype from a prompt (not persisted). provider/model come from the
// active tier set, same as Agent Forge's decomposition call.
export async function generateArchetype(
  prompt: string, provider: string, model: string,
  apiKey = '', baseUrl = '', maxTokens = 0,
): Promise<ArchetypeInput> {
  const draft = await jsonOrThrow(await authFetch('/fd/archetypes/generate', {
    method: 'POST',
    body: JSON.stringify({ prompt, provider, model, api_key: apiKey, base_url: baseUrl, max_tokens: maxTokens }),
  }))
  // The backend returns the archetype with `id`; map to the form's archetype_id.
  return {
    archetype_id: draft.id || '',
    role: draft.role || '',
    family: draft.family || 'Custom',
    description: draft.description || '',
    cognitive_mode: draft.cognitive_mode || 'neutra',
    tier: draft.tier || 'balanced',
    tools: draft.tools || [],
    fleet_instructions: draft.fleet_instructions || '',
    keywords: draft.keywords || [],
    lead: !!draft.lead,
    reliability_seed: typeof draft.reliability_seed === 'number' ? draft.reliability_seed : 0.7,
  }
}
