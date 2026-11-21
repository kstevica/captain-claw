// Captain for Chrome — web pages read here for the person's agents' web_fetch.
//
// Flight Deck sends {type: "fetch", id, url, wait} over the socket hands.js
// keeps. The page opens in a background tab — never in front, in a collapsed
// "Captain · reading" group — signed in as the person, so sites that turn
// Flight Deck away (a sign-in wall, a bot check) open as they do for them.
// Once the page has settled (no change to its DOM for a moment; scrolled
// through a screen at a time for what's drawn only when scrolled to), its HTML goes back without scripts, styles or form fields,
// with every link and picture address made absolute, and the tab closes. The
// agent's web_fetch reads it like any other page.
//
// Only while the person turned it on in Flight Deck (Profile → Web pages),
// Chrome lets Captain read every site ("Work on every site"), and it isn't
// turned off here (Settings → Web pages). Never Flight Deck's own pages,
// addresses on a private network or this computer, or anything but http(s).

import { loadConn } from './lib/api.js'
import { scrollThrough, settleInPage } from './lib/settle.js'
import { inBackgroundTab } from './search.js'

const GROUP_TITLE = 'Captain · reading'
const PICK_KEY = 'fetchMode'       // Flight Deck's mode ('' = off), as it last said
const PREFS_KEY = 'fetchPrefs'     // { declined: true } = turned off in this browser
const ALL = ['<all_urls>']
const WAIT_MAX_MS = 36000
const POLL_MS = 400
const SETTLE_MS = 10000            // after load: longest wait for the page to stop changing
const SCROLLS = 8                  // screens scrolled down for what's drawn only when scrolled to
const HTML_MAX = 2900000

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
class Refusal extends Error {}

export async function loadFetchMode() {
  const got = await chrome.storage.local.get([PICK_KEY, PREFS_KEY]).catch(() => ({}))
  const mode = ['blocked', 'always'].includes(got[PICK_KEY]) ? got[PICK_KEY] : ''
  return { mode, declined: !!got[PREFS_KEY]?.declined }
}

export async function saveFetchMode(mode) {
  const m = ['blocked', 'always'].includes(mode) ? mode : ''
  if (m !== (await loadFetchMode()).mode) await chrome.storage.local.set({ [PICK_KEY]: m })
}

export async function setFetchDeclined(declined) {
  await chrome.storage.local.set({ [PREFS_KEY]: { declined: !!declined } })
}

export async function readsPages() {
  const { mode, declined } = await loadFetchMode()
  if (!mode || declined) return false
  try { return await chrome.permissions.contains({ origins: ALL }) } catch { return false }
}

const LOCAL_SUFFIXES = ['.local', '.localhost', '.internal', '.lan', '.home', '.home.arpa', '.intranet', '.corp']

function privateHost(host) {
  const h = host.toLowerCase().replace(/\.$/, '').replace(/^\[|\]$/g, '')
  if (h === 'localhost' || LOCAL_SUFFIXES.some((s) => h.endsWith(s))) return true
  const v4 = h.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/)
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])]
    return a === 0 || a === 10 || a === 127 || (a === 100 && b >= 64 && b <= 127) || (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || a >= 224
  }
  if (h.includes(':')) return h === '::1' || h === '::' || /^f[cd]/.test(h) || /^fe[89ab]/.test(h) || h.startsWith('::ffff:')
  return !h.includes('.')   // a single name: something on the local network
}

/** Why this browser won't open *href* for an agent — '' when it may. */
export async function refusal(href) {
  let u
  try { u = new URL(href) } catch { return 'That address is broken.' }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') return 'Only web pages (http or https) are read here.'
  if (privateHost(u.hostname)) return 'Addresses on a private network or this computer are never opened for an agent.'
  const conn = await loadConn().catch(() => null)
  try {
    if (conn && new URL(conn.url).host === u.host) return "Flight Deck's own pages are never opened for an agent."
  } catch { /* no deck address */ }
  return ''
}

/** One page for Flight Deck: what to send back as the act's result. */
export async function runFetch(msg, isCancelled) {
  try {
    if (!(await readsPages())) {
      throw new Refusal("This browser doesn't read pages for agents now (Flight Deck → Profile → Web pages, and “Work on every site” in Captain's settings).")
    }
    const href = String(msg.url || '')
    const why = await refusal(href)
    if (why) throw new Refusal(why)
    return await inBackgroundTab(href, GROUP_TITLE, (tabId) => settleAndRead(tabId, msg, isCancelled))
  } catch (e) {
    return { ok: false, error: String(e?.message || e).slice(0, 400) }
  }
}

async function settleAndRead(tabId, msg, isCancelled) {
  const id = String(msg.id || '')
  const limit = Date.now() + Math.min(Math.max(Number(msg.wait) || 20, 5) * 1000, WAIT_MAX_MS)
  const left = () => limit - Date.now()
  // Loaded first …
  while (Date.now() < limit) {
    if (isCancelled(id)) return { ok: false, error: 'Cancelled.' }
    const tab = await chrome.tabs.get(tabId).catch(() => null)
    if (!tab) return { ok: false, error: 'The tab was closed.' }
    if (tab.status === 'complete') break
    await sleep(POLL_MS)
  }
  const type = await run(tabId, () => document.contentType || '')
  if (type && !/html|xml/i.test(type)) return { ok: false, error: `Not a web page (${type}) — fetched the usual way instead.` }
  // … then drawn: "loaded" isn't "there" on pages that draw their sections as
  // their data arrives (lib/settle.js) …
  await run(tabId, settleInPage, [{ maxMs: Math.max(1000, Math.min(SETTLE_MS, left() - 6000)), quietMs: 900, minMs: 1500 }])
  if (isCancelled(id)) return { ok: false, error: 'Cancelled.' }
  // … and scrolled through, for what's drawn only when scrolled to (feeds,
  // comments, profile sections), then drawn again.
  if (left() > 4000) {
    const went = await run(tabId, scrollThrough, [{ steps: SCROLLS, quietMs: 600, stepMaxMs: 2000, totalMs: Math.min(10000, left() - 3000) }])
    if (went?.steps) await run(tabId, settleInPage, [{ maxMs: Math.max(500, Math.min(2500, left() - 1500)), quietMs: 500 }])
  }
  const tab = await chrome.tabs.get(tabId).catch(() => null)
  const where = await refusal(tab?.url || '')
  if (where) return { ok: false, error: where }   // it ended up somewhere it mustn't be read
  const page = await run(tabId, pageForReading)
  if (!page) return { ok: false, error: "The page couldn't be read in this browser." }
  return { ok: true, content: page.html.slice(0, HTML_MAX), data: { url: page.url, title: page.title, type: page.type } }
}

async function run(tabId, func, args = []) {
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId }, func, args })
    return res?.result ?? null
  } catch {
    return null
  }
}

// Runs in the page: its HTML without scripts, styles, form fields and event
// handlers — pictures kept (absolute src, alt) and links made absolute — the
// way web_fetch reads a page it fetched itself.
function pageForReading() {
  const KEEP = new Set(['href', 'src', 'alt', 'title', 'class', 'id', 'role', 'datetime', 'aria-label', 'colspan',
    'rowspan', 'lang', 'data-testid'])
  const DROP = 'script,style,noscript,template,svg,iframe,link,meta,video,audio,canvas,source,input,textarea,select,button,object,embed,form[role=search]'
  const links = [...document.querySelectorAll('a[href]')]
  const imgs = [...document.images]
  for (const a of links) a.setAttribute('data-cc-href', a.href)
  for (const img of imgs) img.setAttribute('data-cc-src', img.currentSrc || img.src || '')
  const doc = document.documentElement.cloneNode(true)
  for (const a of links) a.removeAttribute('data-cc-href')
  for (const img of imgs) img.removeAttribute('data-cc-src')
  doc.querySelectorAll(DROP).forEach((n) => n.remove())
  const walk = document.createTreeWalker(doc, NodeFilter.SHOW_COMMENT)
  const comments = []
  while (walk.nextNode()) comments.push(walk.currentNode)
  comments.forEach((c) => c.remove())
  for (const el of [doc, ...doc.querySelectorAll('*')]) {
    const href = el.getAttribute('data-cc-href')
    const src = el.getAttribute('data-cc-src')
    for (const at of [...el.attributes]) if (!KEEP.has(at.name)) el.removeAttribute(at.name)
    if (href) el.setAttribute('href', href)
    if (src !== null) {
      if (/^https?:/i.test(src)) el.setAttribute('src', src)
      else el.removeAttribute('src')   // data: and blob: pictures stay out
    }
  }
  return { html: doc.outerHTML.replace(/\s{2,}/g, ' '), url: location.href, title: document.title, type: document.contentType || '' }
}
