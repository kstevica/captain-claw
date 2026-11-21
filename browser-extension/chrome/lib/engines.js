// The search engines this browser can run the person's agents' web searches
// on (search.js). The person picks one in Flight Deck (Profile → Web searches)
// — their opt-in — and Flight Deck reads the result pages; this list is what
// the extension itself allows: a search only ever opens a result page on the
// engine's own host, and only while Chrome lets Captain read that site (the
// panel's Allow, or "Work on every site" already on). Turning searches off in
// the panel's settings declines them in this browser.

export const SEARCH_ENGINES = {
  google: { name: 'Google', host: 'www.google.com', origins: ['https://www.google.com/*'] },
  bing: { name: 'Bing', host: 'www.bing.com', origins: ['https://www.bing.com/*'] },
  duckduckgo: { name: 'DuckDuckGo', host: 'html.duckduckgo.com', origins: ['https://html.duckduckgo.com/*'] },
  yahoo: { name: 'Yahoo', host: 'search.yahoo.com', origins: ['https://search.yahoo.com/*'] },
  qwant: { name: 'Qwant', host: 'www.qwant.com', origins: ['https://www.qwant.com/*'] },
}

const KEY = 'search'
const PICK_KEY = 'searchPick'   // the engine picked in Flight Deck ('' = off), as it last said

/** { on, engines: [ids the person allowed here], declined: id } */
export async function loadSearchPrefs() {
  const got = await chrome.storage.local.get(KEY).catch(() => ({}))
  const p = got[KEY] && typeof got[KEY] === 'object' ? got[KEY] : {}
  return { on: !!p.on, engines: Array.isArray(p.engines) ? p.engines.filter((e) => SEARCH_ENGINES[e]) : [], declined: String(p.declined || '') }
}

export async function saveSearchPrefs(prefs) {
  await chrome.storage.local.set({ [KEY]: prefs })
}

export async function loadSearchPick() {
  const got = await chrome.storage.local.get(PICK_KEY).catch(() => ({}))
  const id = String(got[PICK_KEY] || '')
  return SEARCH_ENGINES[id] ? id : ''
}

export async function saveSearchPick(id) {
  const pick = SEARCH_ENGINES[id] ? id : ''
  if (pick !== (await loadSearchPick())) await chrome.storage.local.set({ [PICK_KEY]: pick })
}

/** Engines this browser serves now: allowed here (the panel's Allow), or the
 * one picked in Flight Deck unless it was turned off here — and Chrome still
 * lets Captain read the engine's result pages. */
export async function servedEngines() {
  const prefs = await loadSearchPrefs()
  const pick = await loadSearchPick()
  if (!chrome.permissions?.contains) return []
  const wanted = new Set(prefs.on ? prefs.engines : [])
  if (pick && prefs.declined !== pick) wanted.add(pick)
  const out = []
  for (const id of wanted) {
    try {
      if (await chrome.permissions.contains({ origins: SEARCH_ENGINES[id].origins })) out.push(id)
    } catch { /* not allowed */ }
  }
  return out
}
