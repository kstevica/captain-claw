/** Pass an onboarded agent on: its documents, lessons, skills and last exam's questions packaged with
 *  a new archetype of its owner's (/fd/agent-package/{slug}). An agent made from it starts with all of
 *  it as its base. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { actAsHeaders } from './onboardingAs'

export interface PassOnBody {
  role: string
  description?: string
  docs: boolean
  lessons: boolean
  skills: boolean
  exam: boolean
}

export interface PassedOn {
  archetype: { id: string; role: string; description: string }
  counts: { docs: number; tables: number; skills: number; lessons: number; questions: number }
}

export async function passOn(slug: string, body: PassOnBody): Promise<PassedOn> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    return { 'Content-Type': 'application/json', ...(authEnabled && token ? { Authorization: `Bearer ${token}` } : {}),
      ...actAsHeaders() }
  }
  const go = () => fetch(`/fd/agent-package/${encodeURIComponent(slug)}`, {
    method: 'POST', headers: headers(), body: JSON.stringify(body), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof err.detail === 'string' ? err.detail : `${res.status}`)
  }
  return res.json()
}
