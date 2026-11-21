// The search engines this browser can run the person's agents' web searches
// on (search.js). Flight Deck picks one (Profile → Web searches) and reads the
// result pages; this list is what the extension itself allows: a search only
// ever opens a result page on the engine's own host, and only after the person
// let Captain read that site (Chrome's permission prompt, from the panel).

export const SEARCH_ENGINES = {
  google: { name: 'Google', host: 'www.google.com', origins: ['https://www.google.com/*'] },
  bing: { name: 'Bing', host: 'www.bing.com', origins: ['https://www.bing.com/*'] },
  duckduckgo: { name: 'DuckDuckGo', host: 'html.duckduckgo.com', origins: ['https://html.duckduckgo.com/*'] },
  yahoo: { name: 'Yahoo', host: 'search.yahoo.com', origins: ['https://search.yahoo.com/*'] },
  qwant: { name: 'Qwant', host: 'www.qwant.com', origins: ['https://www.qwant.com/*'] },
}

const KEY = 'search'

/** { on, engines: [ids the person allowed here], declined: id } */
export async function loadSearchPrefs() {
  const got = await chrome.storage.local.get(KEY).catch(() => ({}))
  const p = got[KEY] && typeof got[KEY] === 'object' ? got[KEY] : {}
  return { on: !!p.on, engines: Array.isArray(p.engines) ? p.engines.filter((e) => SEARCH_ENGINES[e]) : [], declined: String(p.declined || '') }
}

export async function saveSearchPrefs(prefs) {
  await chrome.storage.local.set({ [KEY]: prefs })
}

/** Engines this browser serves now: turned on here, and Chrome still lets
 * Captain read the engine's result pages. */
export async function servedEngines() {
  const prefs = await loadSearchPrefs()
  if (!prefs.on || !chrome.permissions?.contains) return []
  const out = []
  for (const id of prefs.engines) {
    try {
      if (await chrome.permissions.contains({ origins: SEARCH_ENGINES[id].origins })) out.push(id)
    } catch { /* not allowed */ }
  }
  return out
}
