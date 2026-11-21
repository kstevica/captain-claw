// Captain for Chrome — the person's agents' web searches, in this browser.
//
// Flight Deck sends {type: "search", id, engine, url, wait_for, block_for,
// wait} over the socket hands.js keeps (opened for searches alone when the
// browser isn't lent to agents). The result page opens in a background tab —
// never brought to the front, in a collapsed "Captain · search" group — and
// once its results show (or the engine asks for a robot check), the page goes
// back without scripts, styles or pictures, and the tab closes. Flight Deck
// reads the results out of it, so a changed result page needs a Flight Deck
// update, not a new extension.
//
// Only the engines in lib/engines.js, only on the engine's own host, only
// after the person turned searches on in the panel — which is when Chrome
// asked them to let Captain read that site.

import { SEARCH_ENGINES, loadSearchPrefs } from './lib/engines.js'

const AT_ONCE = 3
const WAIT_MAX_MS = 25000
const POLL_MS = 300
const LOAD_GRACE_MS = 2500     // after the page loaded, how long to wait for the results to show
const HTML_MAX = 1200000
const GROUP_TITLE = 'Captain · search'

let running = 0
const waiting = []
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

class Refusal extends Error {}

/** One result page for Flight Deck: what to send back as the act's result. */
export async function runSearch(msg, isCancelled) {
  const engine = SEARCH_ENGINES[String(msg.engine || '')]
  try {
    if (!engine) throw new Refusal(`“${msg.engine}” isn't a search engine this browser runs.`)
    const prefs = await loadSearchPrefs()
    if (!prefs.on || !prefs.engines.includes(msg.engine)) {
      throw new Refusal(`Searches on ${engine.name} aren't turned on in this browser (Captain → Settings).`)
    }
    let url
    try { url = new URL(String(msg.url || '')) } catch { throw new Refusal('That search address is broken.') }
    if (url.protocol !== 'https:' || url.hostname !== engine.host) {
      throw new Refusal(`This browser only searches on ${engine.host}.`)
    }
    if (!(await chrome.permissions.contains({ origins: engine.origins }).catch(() => false))) {
      throw new Refusal(`Captain may no longer read ${engine.name} in this browser — turn searches on again in its settings.`)
    }
    await slot()
    try {
      return await loadResults(url.href, engine, msg, isCancelled)
    } finally {
      running--
      waiting.shift()?.()
    }
  } catch (e) {
    return { ok: false, error: String(e?.message || e).slice(0, 400) }
  }
}

function slot() {
  if (running < AT_ONCE) { running++; return Promise.resolve() }
  return new Promise((resolve) => waiting.push(() => { running++; resolve() }))
}

async function loadResults(href, engine, msg, isCancelled) {
  const { tabId, windowId, ownWindow } = await openBackground(href)
  const id = String(msg.id || '')
  try {
    const limit = Date.now() + Math.min(Math.max(Number(msg.wait) || 20, 5) * 1000, WAIT_MAX_MS)
    let loadedAt = 0
    let probe = null
    while (Date.now() < limit) {
      if (isCancelled(id)) return { ok: false, error: 'Cancelled.' }
      await sleep(POLL_MS)
      const tab = await chrome.tabs.get(tabId).catch(() => null)
      if (!tab) return { ok: false, error: 'The search tab was closed.' }
      if (tab.status === 'complete' && !loadedAt) loadedAt = Date.now()
      probe = await look(tabId, msg.wait_for, msg.block_for)
      if (!probe) {
        // Off the engine's site (a cookie question on consent.google.com …):
        // nothing to read there. Flight Deck tells the person from the address.
        if (loadedAt) return { ok: true, content: '', data: { url: tab.url || '', title: tab.title || '', ready: false, blocked: false } }
        continue
      }
      if (probe.ready || probe.blocked) break
      if (loadedAt && Date.now() - loadedAt > LOAD_GRACE_MS) break
    }
    const page = await snapshot(tabId)
    if (!page) {
      const tab = await chrome.tabs.get(tabId).catch(() => null)
      return { ok: true, content: '', data: { url: tab?.url || '', title: tab?.title || '', ready: false, blocked: false } }
    }
    return { ok: true, content: page.html.slice(0, HTML_MAX),
             data: { url: page.url, title: page.title, ready: !!probe?.ready, blocked: !!probe?.blocked } }
  } finally {
    if (ownWindow) chrome.windows.remove(windowId).catch(() => {})
    else chrome.tabs.remove(tabId).catch(() => {})
  }
}

// ── the background tab ──

async function openBackground(href) {
  let win = null
  try { win = await chrome.windows.getLastFocused({ windowTypes: ['normal'] }) } catch { /* none open */ }
  if (!win || win.incognito) {
    const all = await chrome.windows.getAll({ windowTypes: ['normal'] }).catch(() => [])
    win = all.find((w) => !w.incognito) || null
  }
  if (!win) {
    // No window open (Chrome running in the background on a Mac): a minimized
    // one of its own, closed afterwards.
    const made = await chrome.windows.create({ url: href, focused: false, state: 'minimized', type: 'normal' })
    const tabId = made.tabs?.[0]?.id
    if (tabId === undefined) throw new Error("Chrome didn't open the search window.")
    chrome.tabs.update(tabId, { muted: true }).catch(() => {})
    return { tabId, windowId: made.id, ownWindow: true }
  }
  const tab = await chrome.tabs.create({ windowId: win.id, url: href, active: false })
  chrome.tabs.update(tab.id, { muted: true }).catch(() => {})
  await tuck(tab.id, win.id)
  return { tabId: tab.id, windowId: win.id, ownWindow: false }
}

// Into the window's collapsed "Captain · search" group, out of the person's way.
async function tuck(tabId, windowId) {
  if (!chrome.tabGroups?.query || !chrome.tabs.group) return
  try {
    const found = await chrome.tabGroups.query({ windowId, title: GROUP_TITLE })
    const groupId = await chrome.tabs.group(found[0] ? { tabIds: [tabId], groupId: found[0].id } : { tabIds: [tabId], createProperties: { windowId } })
    await chrome.tabGroups.update(groupId, { title: GROUP_TITLE, color: 'grey', collapsed: true })
  } catch { /* a window that takes no groups: the tab stays where it is, in the background */ }
}

// ── reading the page ──

async function look(tabId, waitFor, blockFor) {
  try {
    const [res] = await chrome.scripting.executeScript({
      target: { tabId },
      args: [String(waitFor || ''), String(blockFor || '')],
      func: (w, b) => {
        const has = (sel) => { try { return !!(sel && document.querySelector(sel)) } catch { return false } }
        return { ready: has(w), blocked: has(b) || /\/sorry\//.test(location.pathname), url: location.href }
      },
    })
    return res?.result || null
  } catch {
    return null   // not loaded yet, or a page on another site this extension may not read
  }
}

async function snapshot(tabId) {
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId }, func: pageWithoutExtras })
    return res?.result || null
  } catch {
    return null
  }
}

// Runs in the result page: its HTML without what reading results doesn't need
// — scripts, styles, pictures, form fields, event handlers, inline styles —
// with every link made absolute.
function pageWithoutExtras() {
  const KEEP = new Set(['href', 'class', 'id', 'role', 'datetime', 'aria-label', 'data-testid', 'data-type', 'data-pos', 'data-sncf'])
  const DROP = 'script,style,noscript,template,svg,iframe,link,meta,img,video,audio,canvas,picture,source,input,textarea,select,button,object,embed'
  const links = [...document.querySelectorAll('a[href]')]
  for (const a of links) a.setAttribute('data-cc-href', a.href)
  const doc = document.documentElement.cloneNode(true)
  for (const a of links) a.removeAttribute('data-cc-href')
  doc.querySelectorAll(DROP).forEach((n) => n.remove())
  const walk = document.createTreeWalker(doc, NodeFilter.SHOW_COMMENT)
  const comments = []
  while (walk.nextNode()) comments.push(walk.currentNode)
  comments.forEach((c) => c.remove())
  for (const el of [doc, ...doc.querySelectorAll('*')]) {
    const abs = el.getAttribute('data-cc-href')
    for (const at of [...el.attributes]) if (!KEEP.has(at.name)) el.removeAttribute(at.name)
    if (abs) el.setAttribute('href', abs)
  }
  return { html: doc.outerHTML.replace(/\s{2,}/g, ' '), url: location.href, title: document.title }
}
