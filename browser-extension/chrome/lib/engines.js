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
// The engine picked in each Flight Deck ('' = off), as it last said: { deck address: id }.
// Each deck has its own pick; a browser connected to two serves each its own.
const PICK_KEY = 'searchPicks'

/** { on, engines: [ids the person allowed here], declined: id } */
export async function loadSearchPrefs() {
  const got = await chrome.storage.local.get(KEY).catch(() => ({}))
  const p = got[KEY] && typeof got[KEY] === 'object' ? got[KEY] : {}
  return { on: !!p.on, engines: Array.isArray(p.engines) ? p.engines.filter((e) => SEARCH_ENGINES[e]) : [], declined: String(p.declined || '') }
}

export async function saveSearchPrefs(prefs) {
  await chrome.storage.local.set({ [KEY]: prefs })
}

export async function loadSearchPick(deck) {
  const got = await chrome.storage.local.get(PICK_KEY).catch(() => ({}))
  const id = String(got[PICK_KEY]?.[deck] || '')
  return SEARCH_ENGINES[id] ? id : ''
}

export async function saveSearchPick(deck, id) {
  if (!deck) return
  const pick = SEARCH_ENGINES[id] ? id : ''
  const got = await chrome.storage.local.get(PICK_KEY).catch(() => ({}))
  const picks = got[PICK_KEY] && typeof got[PICK_KEY] === 'object' ? got[PICK_KEY] : {}
  if ((picks[deck] || '') === pick) return
  await chrome.storage.local.set({ [PICK_KEY]: { ...picks, [deck]: pick } })
}

/** Engines this browser serves *deck* now: allowed here (the panel's Allow),
 * or the one picked in that Flight Deck unless it was turned off here — and
 * Chrome still lets Captain read the engine's result pages. */
export async function servedEngines(deck) {
  const prefs = await loadSearchPrefs()
  const pick = await loadSearchPick(deck)
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
