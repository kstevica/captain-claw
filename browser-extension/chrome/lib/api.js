// Talking to Flight Deck: the decks this browser is connected to, JSON calls,
// the upload.
//
// A browser can be connected to several Flight Decks at once — a work deck and
// a home one, say. Each gave it a token of its own. Agents on every one of them
// can use the browser (hands.js keeps a socket to each); the panel works with
// the one picked at its top, the deck "in use".

const DECKS_KEY = 'decks'   // [{ url, token, user, agent?, agent_name? }], oldest first
const USE_KEY = 'deck'      // the address of the deck in use
const OLD_KEY = 'conn'      // one deck, kept the way it was before 0.10 — and how Flight Deck seeds an agent's own browser

/** "Chrome on macOS", "Edge on Windows"… — how Flight Deck names this browser. */
export function browserLabel() {
  const ua = navigator.userAgentData
  const brand = (ua?.brands || []).map((b) => b.brand).find((b) => !/not.?a.?brand|chromium/i.test(b)) || 'Chrome'
  const platform = ua?.platform || (/Mac/.test(navigator.platform) ? 'macOS' : /Win/.test(navigator.platform) ? 'Windows' : 'Linux')
  return `${brand.replace(/^Google /, '')} on ${platform}`
}

export class AuthError extends Error {}

const usable = (c) => !!(c && typeof c === 'object' && typeof c.url === 'string' && c.url && c.token)

/** { decks, use }: every deck this browser is connected to, and the one in use (null: none). */
export async function loadDecks() {
  const got = await chrome.storage.local.get([DECKS_KEY, USE_KEY, OLD_KEY])
  let decks = (Array.isArray(got[DECKS_KEY]) ? got[DECKS_KEY] : []).filter(usable)
  let use = typeof got[USE_KEY] === 'string' ? got[USE_KEY] : ''
  const old = got[OLD_KEY]
  if (old !== undefined) {
    // Kept the old way (an update from before, or Flight Deck seeding an
    // agent's own browser): it joins the list, in use, replacing an older
    // token for the same deck.
    if (usable(old)) {
      decks = [...decks.filter((d) => d.url !== old.url), old]
      use = old.url
      await chrome.storage.local.set({ [DECKS_KEY]: decks, [USE_KEY]: use })
    }
    await chrome.storage.local.remove(OLD_KEY)
  }
  return { decks, use: decks.find((d) => d.url === use) || decks[0] || null }
}

/** The deck in use, or null. */
export async function loadConn() {
  return (await loadDecks()).use
}

/** A deck connected, or connected again: kept (its new token replaces the old one), and in use. */
export async function saveDeck(conn) {
  const { decks } = await loadDecks()
  await chrome.storage.local.set({ [DECKS_KEY]: [...decks.filter((d) => d.url !== conn.url), conn], [USE_KEY]: conn.url })
}

export async function useDeck(url) {
  await chrome.storage.local.set({ [USE_KEY]: url })
}

/** A deck disconnected: off the list. Returns the deck in use after it (null: none left). */
export async function forgetDeck(url) {
  const { decks, use } = await loadDecks()
  const left = decks.filter((d) => d.url !== url)
  const next = (use && use.url !== url ? use : left[left.length - 1]) || null
  await chrome.storage.local.set({ [DECKS_KEY]: left, [USE_KEY]: next ? next.url : '' })
  return next
}

/** Is *url* a page of one of these decks? Flight Deck's own pages are off-limits to agents. */
export function onDeck(url, decks) {
  let origin = ''
  try { origin = new URL(url).origin } catch { return false }
  return decks.some((d) => {
    try { return new URL(d.url).origin === origin } catch { return false }
  })
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
