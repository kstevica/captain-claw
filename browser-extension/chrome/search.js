// Captain for Chrome — the person's agents' web searches, in this browser.
//
// Flight Deck sends {type: "search", id, engine, url, wait_for, block_for,
// wait, hand_over} over the socket hands.js keeps (opened for searches alone
// when the browser isn't lent to agents). The result page opens in a
// background tab — never brought to the front, in a collapsed "Captain ·
// search" group — and once its results show, the page goes back without
// scripts, styles or pictures, and the tab closes. Flight Deck reads the
// results out of it, so a changed result page needs a Flight Deck update, not
// a new extension.
//
// A robot check instead of results (the engine's own, or any other —
// lib/robot-check.js): with hand_over (seconds the agent waits for the
// person), the tab comes to the front for the person to answer, and the
// search goes on in it once they did; not answered, it stays open for them.
// Without, the page goes back marked blocked, as before.
//
// Only the engines in lib/engines.js, only on the engine's own host, only
// one the person picked in Flight Deck or allowed in the panel, and only while
// Chrome lets Captain read that site.

import { SEARCH_ENGINES, servedEngines } from './lib/engines.js'
import { giveBack, handOver, letGo, passesByItself, robotCheck, shownTab } from './lib/robot-check.js'

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

/** One result page for Flight Deck: what to send back as the act's result.
 * *tell* sends Flight Deck a word about this act meanwhile (a check handed
 * to the person). */
export async function runSearch(msg, isCancelled, tell = () => {}) {
  const engine = SEARCH_ENGINES[String(msg.engine || '')]
  try {
    if (!engine) throw new Refusal(`“${msg.engine}” isn't a search engine this browser runs.`)
    if (!(await servedEngines()).includes(msg.engine)) {
      throw new Refusal(`Searches on ${engine.name} aren't on in this browser (Captain → Settings → Web searches).`)
    }
    let url
    try { url = new URL(String(msg.url || '')) } catch { throw new Refusal('That search address is broken.') }
    if (url.protocol !== 'https:' || url.hostname !== engine.host) {
      throw new Refusal(`This browser only searches on ${engine.host}.`)
    }
    if (!(await chrome.permissions.contains({ origins: engine.origins }).catch(() => false))) {
      throw new Refusal(`Captain may no longer read ${engine.name} in this browser — turn searches on again in its settings.`)
    }
    return await inBackgroundTab(url.href, GROUP_TITLE, (tabId) => loadResults(tabId, msg, engine, isCancelled, tell))
  } catch (e) {
    return { ok: false, error: String(e?.message || e).slice(0, 400) }
  }
}

function slot() {
  if (running < AT_ONCE) { running++; return Promise.resolve() }
  return new Promise((resolve) => waiting.push(() => { running++; resolve() }))
}

async function loadResults(tabId, msg, engine, isCancelled, tell) {
  const id = String(msg.id || '')
  const cancelled = () => isCancelled(id)
  const handFor = Math.max(0, Number(msg.hand_over) || 0) * 1000
  let robot = null
  for (let round = 0; ; round++) {
    const got = await waitForResults(tabId, msg, cancelled)
    if (got.error) return { ok: false, error: got.error, data: robot ? { robot } : undefined }
    const { probe, tab } = got
    if (!probe) {
      // Off the engine's site (a cookie question on consent.google.com …):
      // nothing to read there. Flight Deck tells the person from the address.
      return { ok: true, content: '', data: { url: tab?.url || '', title: tab?.title || '', ready: false, blocked: false, robot } }
    }
    let blocked = !!probe.blocked
    if (!probe.ready) {
      // Any robot check, not only the engine's own form.
      const check = await robotCheck(tabId)
      if (check?.check) blocked = true
      if (blocked && round < 3) {
        if (check?.passes && (await passesByItself(tabId, cancelled))) continue
        if (handFor) {
          const vendor = check?.vendor || engine.name
          const outcome = await handOver(tabId, {
            site: engine.host, forMs: handFor, isCancelled: cancelled, tell, vendor,
            why: `${engine.name} asks to confirm a person is searching`,
            // Answered: back on the engine's own result page.
            cleared: async (_t, now) => {
              try { const u = new URL(now.url); return u.hostname === engine.host && !u.pathname.startsWith('/sorry/') } catch { return false }
            },
          })
          robot = { outcome, vendor, site: engine.host }
          if (outcome === 'solved' || outcome === 'again') continue
          const left = await chrome.tabs.get(tabId).catch(() => null)
          return { ok: true, content: '', data: { url: left?.url || '', title: left?.title || '', ready: false, blocked: true, robot } }
        }
      }
    }
    const page = await snapshot(tabId)
    if (!page) {
      const left = await chrome.tabs.get(tabId).catch(() => null)
      return { ok: true, content: '', data: { url: left?.url || '', title: left?.title || '', ready: false, blocked, robot } }
    }
    return { ok: true, content: page.html.slice(0, HTML_MAX),
             data: { url: page.url, title: page.title, ready: !!probe.ready, blocked, robot } }
  }
}

// Until the results show, the engine asks for a robot check, or the page
// loaded and still shows neither: ``{probe, tab}`` or ``{error}``.
async function waitForResults(tabId, msg, cancelled) {
  const limit = Date.now() + Math.min(Math.max(Number(msg.wait) || 20, 5) * 1000, WAIT_MAX_MS)
  let loadedAt = 0
  let probe = null
  let tab = null
  while (Date.now() < limit) {
    if (cancelled()) return { error: 'Cancelled.' }
    await sleep(POLL_MS)
    tab = await chrome.tabs.get(tabId).catch(() => null)
    if (!tab) return { error: 'The search tab was closed.' }
    if (tab.status === 'complete' && !loadedAt) loadedAt = Date.now()
    probe = await look(tabId, msg.wait_for, msg.block_for)
    if (!probe) {
      if (loadedAt) return { probe: null, tab }
      continue
    }
    if (probe.ready || probe.blocked) break
    if (loadedAt && Date.now() - loadedAt > LOAD_GRACE_MS) break
  }
  return { probe: probe || { ready: false, blocked: false }, tab }
}

// ── the background tab (searches here, pages read for web_fetch in pages.js) ──

/** Run *work(tabId)* on *href* opened in a background tab, then close it —
 * at most AT_ONCE at a time across searches and pages. */
export async function inBackgroundTab(href, groupTitle, work) {
  await slot()
  let opened = null
  try {
    opened = await openBackground(href, groupTitle)
    return await work(opened.tabId)
  } finally {
    // A robot check the person didn't answer stays open for them; one they
    // answered closes with their own tab back in front.
    const shown = opened ? shownTab(opened.tabId) : null
    if (shown?.keep) letGo(opened.tabId)
    else {
      if (shown) await giveBack(opened.tabId).catch(() => {})
      if (opened?.ownWindow) chrome.windows.remove(opened.windowId).catch(() => {})
      else if (opened) chrome.tabs.remove(opened.tabId).catch(() => {})
    }
    running--
    waiting.shift()?.()
  }
}

async function openBackground(href, groupTitle) {
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
  await tuck(tab.id, win.id, groupTitle)
  return { tabId: tab.id, windowId: win.id, ownWindow: false }
}

// Into the window's collapsed group ("Captain · search" …), out of the person's way.
async function tuck(tabId, windowId, title) {
  if (!chrome.tabGroups?.query || !chrome.tabs.group) return
  try {
    const found = await chrome.tabGroups.query({ windowId, title })
    const groupId = await chrome.tabs.group(found[0] ? { tabIds: [tabId], groupId: found[0].id } : { tabIds: [tabId], createProperties: { windowId } })
    await chrome.tabGroups.update(groupId, { title, color: 'grey', collapsed: true })
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
