/** The skills an agent has — the procedures it follows when a request fits
 *  (SKILL.md files: from its owner's setup, written by itself, or built in) —
 *  read, switched on or off and corrected through Flight Deck's
 *  /fd/agent-skills/{host}/{port} proxy (the agent's own /api/skills; only its
 *  owner gets through). Not the kiosk's "What can I do?" cards (agentSkills.ts). */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { AgentAddress } from './agentInsights'

export interface OwnSkill {
  name: string
  description: string
  /** Where it was loaded from: captain-managed, captain-workspace, captain-bundled, … */
  source: string
  skill_key: string
  emoji: string | null
  /** null: on (no setting); false: switched off. */
  enabled: boolean | null
  /** Loaded and usable now (its requirements met, not switched off). */
  active: boolean
  user_invocable: boolean
  /** Agents from before these fields: undefined. */
  editable?: boolean
  from_owner?: boolean
}

export interface SkillFile { name: string; source: string; editable: boolean; from_owner: boolean; content: string }

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    return h
  }
  const go = () => fetch(path, { ...init, headers: headers(), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : body.error || `${res.status}`)
  }
  return res
}

const base = (a: AgentAddress) => `/fd/agent-skills/${encodeURIComponent(a.host)}/${a.port}`
const qs = (a: AgentAddress, more: Record<string, string> = {}) => {
  const q = new URLSearchParams({ ...(a.auth ? { token: a.auth } : {}), ...more }).toString()
  return q ? `?${q}` : ''
}

export async function listOwnSkills(a: AgentAddress): Promise<OwnSkill[]> {
  const body = await (await send(`${base(a)}${qs(a)}`)).json()
  if (body.error && !(body.skills || []).length) throw new Error(body.error)
  return body.skills || []
}

export async function readSkill(a: AgentAddress, name: string): Promise<SkillFile> {
  return (await send(`${base(a)}/content${qs(a, { name })}`)).json()
}

export async function saveSkill(a: AgentAddress, name: string, content: string): Promise<void> {
  await send(`${base(a)}/content${qs(a)}`, { method: 'PUT', body: JSON.stringify({ name, content }) })
}

export async function switchSkill(a: AgentAddress, skillKey: string, enabled: boolean): Promise<void> {
  await send(`${base(a)}/toggle${qs(a)}`, { method: 'POST', body: JSON.stringify({ skill_key: skillKey, enabled }) })
}
