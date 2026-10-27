// The kiosk New agent dialog's "Advanced" setup — POST /fd/agent-preset/prepare
// reads the person's instructions and documents with their own model into a
// preset they review; the spawn (spawnArchetypeProcess's `preset`) seeds it.
import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { ACT_AS_HEADER } from './docker'

export interface PresetSource { doc: string; section: string }

export interface PresetRule { text: string; source: PresetSource }
export interface PresetIntent { name: string; when: string; steps: string[]; examples: string[]; source: PresetSource }
export interface PresetFact { content: string; category: string; importance: number; source: PresetSource }
export interface PresetTemplate { name: string; body: string; source: PresetSource }
export interface PresetTerm { term: string; meaning: string; source: PresetSource }
export interface PresetDoc { doc: string; title: string; summary: string }

export interface AgentPreset {
  rules: PresetRule[]
  intents: PresetIntent[]
  facts: PresetFact[]
  escalation: PresetRule[]
  templates: PresetTemplate[]
  glossary: PresetTerm[]
  knowledge: PresetDoc[]
  conflicts: string[]
  warnings: string[]
}

export interface PreparedPreset {
  id: string
  preset: AgentPreset
  docs: { name: string; chars: number; truncated: boolean }[]
  model: string
}

/** What the spawn takes: the prepared id and the items the person kept. */
export type ReviewedPreset = Omit<AgentPreset, 'conflicts' | 'warnings'> & { id: string }

export async function prepareAgentPreset(opts: {
  archetype: string
  instructions: string
  files: File[]
  asUser?: string
}): Promise<PreparedPreset> {
  const form = new FormData()
  form.append('archetype', opts.archetype)
  form.append('instructions', opts.instructions)
  for (const f of opts.files) form.append('files', f)

  const headers = (): Record<string, string> => {
    // No Content-Type: the browser sets the multipart boundary itself.
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    if (opts.asUser) h[ACT_AS_HEADER] = opts.asUser
    return h
  }
  const send = () => fetch('/fd/agent-preset/prepare', {
    method: 'POST', headers: headers(), credentials: 'include', body: form,
  })
  let res = await send()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await send()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status}`)
  }
  return res.json()
}
