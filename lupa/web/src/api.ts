/** Fetch layer: attaches the access token, silently refreshes once on 401
 * (the refresh cookie is first-party to the BFF's /api/auth path). */

let accessToken: string | null = null
let onSession: ((token: string | null, user: unknown | null) => void) | null = null

export function setAccessToken(t: string | null) { accessToken = t }
export function onSessionChange(cb: typeof onSession) { onSession = cb }

async function tryRefresh(): Promise<boolean> {
  const r = await fetch('/api/auth/refresh', { method: 'POST', credentials: 'same-origin' })
  if (!r.ok) return false
  const data = await r.json()
  accessToken = data.access_token ?? null
  onSession?.(accessToken, data.user ?? null)
  return accessToken !== null
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

export async function api<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const doFetch = () => fetch(path, {
    ...init,
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
      ...(init.headers ?? {}),
    },
  })
  let r = await doFetch()
  if (r.status === 401 && await tryRefresh()) r = await doFetch()
  if (r.status === 401) { onSession?.(null, null); throw new ApiError(401, 'signed out') }
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* not json */ }
    throw new ApiError(r.status, String(detail))
  }
  return r.json() as Promise<T>
}

export const post = <T = unknown>(path: string, body: unknown) =>
  api<T>(path, { method: 'POST', body: JSON.stringify(body) })
