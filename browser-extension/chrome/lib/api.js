// Talking to Flight Deck: the stored connection, JSON calls, the upload.

const CONN_KEY = 'conn'

/** "Chrome on macOS", "Edge on Windows"… — how Flight Deck names this browser. */
export function browserLabel() {
  const ua = navigator.userAgentData
  const brand = (ua?.brands || []).map((b) => b.brand).find((b) => !/not.?a.?brand|chromium/i.test(b)) || 'Chrome'
  const platform = ua?.platform || (/Mac/.test(navigator.platform) ? 'macOS' : /Win/.test(navigator.platform) ? 'Windows' : 'Linux')
  return `${brand.replace(/^Google /, '')} on ${platform}`
}

export class AuthError extends Error {}

export async function loadConn() {
  const got = await chrome.storage.local.get(CONN_KEY)
  const c = got[CONN_KEY]
  return c && c.url && c.token ? c : null
}

export async function saveConn(conn) {
  await chrome.storage.local.set({ [CONN_KEY]: conn })
}

export async function clearConn() {
  await chrome.storage.local.remove(CONN_KEY)
}

/** "glasses.example.com" → "https://glasses.example.com"; a bare host on the
 * local network (localhost, an IP, *.local) gets http. Path and query go. */
export function normalizeDeckUrl(text) {
  let s = String(text || '').trim()
  if (!s) return ''
  if (!/^https?:\/\//i.test(s)) {
    const host = s.split('/')[0].split(':')[0]
    const local = host === 'localhost' || /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.endsWith('.local') || host.endsWith('.localhost')
    s = (local ? 'http://' : 'https://') + s
  }
  try {
    const u = new URL(s)
    return u.origin
  } catch {
    return ''
  }
}

async function readError(r) {
  let detail = ''
  try {
    const body = await r.json()
    detail = typeof body?.detail === 'string' ? body.detail : ''
  } catch { /* not JSON */ }
  return detail || `Flight Deck answered ${r.status}`
}

export async function api(conn, path, opts = {}) {
  let r
  try {
    r = await fetch(conn.url + path, {
      ...opts,
      headers: { ...(opts.body && !(opts.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
                 ...(opts.headers || {}), Authorization: `Bearer ${conn.token}` },
    })
  } catch {
    throw new Error(`Can't reach ${new URL(conn.url).host} — is it up?`)
  }
  if (r.status === 401) throw new AuthError('This browser was disconnected from Flight Deck.')
  if (!r.ok) throw new Error(await readError(r))
  return opts.raw ? r : r.json()
}

/** No sign-in yet: the pairing calls. */
export async function publicCall(deckUrl, path, opts = {}) {
  let r
  try {
    r = await fetch(deckUrl + path, {
      ...opts,
      headers: { ...(opts.body ? { 'Content-Type': 'application/json' } : {}), ...(opts.headers || {}) },
    })
  } catch {
    throw new Error(`Can't reach ${new URL(deckUrl).host}. Check the address.`)
  }
  if (r.status === 404 && path.startsWith('/fd/clip/pair')) {
    throw new Error('That Flight Deck has no browser connections yet — it needs an update.')
  }
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

/** The clip, with upload progress (fetch has none). */
export function upload(conn, path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest()
    x.open('POST', conn.url + path)
    x.setRequestHeader('Authorization', `Bearer ${conn.token}`)
    x.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded, e.total) }
    x.onload = () => {
      let body = null
      try { body = JSON.parse(x.responseText) } catch { /* not JSON */ }
      if (x.status === 401) reject(new AuthError('This browser was disconnected from Flight Deck.'))
      else if (x.status >= 200 && x.status < 300) resolve(body)
      else reject(new Error((body && typeof body.detail === 'string' && body.detail) || `Flight Deck answered ${x.status}`))
    }
    x.onerror = () => reject(new Error(`Can't reach ${new URL(conn.url).host} — is it up?`))
    x.send(form)
  })
}
