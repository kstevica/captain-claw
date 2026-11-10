// Captain for Chrome — lending this browser to the person's agents.
//
// With "Agents can use this browser" on (Settings), the service worker keeps
// a socket to Flight Deck (/fd/clip/hands/ws) and runs what an agent's
// browser tool asks: open a tab, read the page, click, type, scroll, take a
// screenshot, fetch files with the person's session. Each agent works in a
// tab group of its own, named after it; it can't touch any other tab (the
// person can drag one into its group to share it). Input, screenshots and
// the console/network logs go through the debugger, so Chrome shows that
// Captain is debugging the browser while an agent works; the page helpers
// (hands-page.js) run in an isolated world the page can't see. Flight Deck's
// own pages and anything that isn't a web page are off-limits, and the
// person can stop any agent from the panel.

import { browserLabel, loadConn } from './lib/api.js'

const VERSION = chrome.runtime.getManifest().version
const PING_MS = 20000
const IDLE_DETACH_MS = 3 * 60 * 1000
const LOG_MAX = 80
const BUFFER_MAX = 300
const NAV_WAIT_MS = 30000
const SETTLE_WAIT_MS = 10000
const SHOT_MAX_W = 1600
const ZOOM_W = 1200
const FILE_MAX = 25 * 1024 * 1024
const FILES_MAX = 20
const TEXT_DEFAULT = 30000
const GROUP_COLORS = ['blue', 'purple', 'cyan', 'green', 'orange', 'pink', 'yellow', 'red', 'grey']
const IS_MAC = /mac/i.test(navigator.userAgentData?.platform || navigator.userAgent || '')

class Refusal extends Error {}

// ── state ──

let ws = null
let wsConn = null       // the deck connection the socket belongs to
let retry = 0
let retryTimer = 0
let pingTimer = 0
let prefs = { on: false, front: false }
let stopped = new Set()
let groups = {}         // agent slug → { groupId, windowId, name, current }
let log = []
let loaded = null
const attached = new Map()   // tabId → { lastUse, world, console: [], net: Map, shotScale, since }
const cancelled = new Set()
const openedFrom = new Map() // tabId → [new tab ids opened from it]

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const oneLine = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim()
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
const q = (s) => JSON.stringify(String(s))

async function loadState() {
  if (!loaded) {
    loaded = (async () => {
      const local = await chrome.storage.local.get(['hands', 'handsStopped']).catch(() => ({}))
      const session = await chrome.storage.session.get(['handsGroups', 'handsLog']).catch(() => ({}))
      prefs = { on: false, front: false, ...(local.hands || {}) }
      stopped = new Set(Array.isArray(local.handsStopped) ? local.handsStopped : [])
      groups = session.handsGroups && typeof session.handsGroups === 'object' ? session.handsGroups : {}
      log = Array.isArray(session.handsLog) ? session.handsLog : []
    })()
  }
  return loaded
}

function saveGroups() {
  chrome.storage.session.set({ handsGroups: groups }).catch(() => {})
}

function saveLog() {
  chrome.storage.session.set({ handsLog: log.slice(-LOG_MAX) }).catch(() => {})
}

function setStatus(state, error = '') {
  chrome.storage.session.set({ handsStatus: { state, error, at: Date.now() } }).catch(() => {})
}

// ── the socket ──

export async function startHands() {
  await loadState()
  chrome.alarms?.create('captain-hands', { periodInMinutes: 0.5 })
  ensureConnected()
}

// Both are in the manifest's permissions (Chrome won't have debugger as an
// optional one); an install that hasn't been reloaded since may lack them.
function permitted() {
  return !!(chrome.debugger?.attach && chrome.tabGroups?.update)
}

async function ensureConnected() {
  await loadState()
  // Read fresh: a change can land while the first read is still on its way.
  const got = await chrome.storage.local.get(['hands', 'handsStopped']).catch(() => ({}))
  prefs = { on: false, front: false, ...(got.hands || {}) }
  stopped = new Set(Array.isArray(got.handsStopped) ? got.handsStopped : [])
  if (!prefs.on) { disconnect('off'); return }
  if (!permitted()) { disconnect('permission'); return }
  const conn = await loadConn()
  if (!conn) { disconnect('off'); return }
  if (ws && wsConn && wsConn.url === conn.url && wsConn.token === conn.token &&
      (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
  disconnect(null)
  connect(conn)
}

function disconnect(state) {
  clearTimeout(retryTimer)
  clearInterval(pingTimer)
  if (ws) {
    const old = ws
    ws = null
    old.onclose = null
    try { old.close(1000) } catch { /* already gone */ }
  }
  wsConn = null
  if (state) {
    setStatus(state)
    if (state === 'off' || state === 'permission') detachAll()
  }
}

function connect(conn) {
  let sock
  try {
    sock = new WebSocket(conn.url.replace(/^http/i, 'ws') + '/fd/clip/hands/ws')
  } catch (e) {
    setStatus('error', String(e.message || e))
    return
  }
  ws = sock
  wsConn = conn
  setStatus('connecting')
  sock.onopen = () => {
    sock.send(JSON.stringify({ type: 'hello', token: conn.token, browser: browserLabel(), version: VERSION, stopped: [...stopped] }))
    clearInterval(pingTimer)
    pingTimer = setInterval(() => send({ type: 'ping' }), PING_MS)
  }
  sock.onmessage = (ev) => {
    let msg
    try { msg = JSON.parse(ev.data) } catch { return }
    if (msg.type === 'welcome') { retry = 0; setStatus('on') }
    else if (msg.type === 'denied') setStatus('denied', msg.error || '')
    else if (msg.type === 'act') runAct(msg)
    else if (msg.type === 'cancel') cancelled.add(String(msg.id || ''))
  }
  sock.onclose = (ev) => {
    if (ws !== sock) return
    ws = null
    clearInterval(pingTimer)
    if (ev.code === 4401) { setStatus('denied', 'This browser was disconnected from Flight Deck.'); return }
    setStatus('connecting', "Can't reach Flight Deck — trying again.")
    const wait = [1000, 2000, 5000, 10000, 20000, 30000][Math.min(retry++, 5)]
    clearTimeout(retryTimer)
    retryTimer = setTimeout(ensureConnected, wait)
  }
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify(msg)) } catch { /* closing */ }
  }
}

chrome.alarms?.onAlarm.addListener((a) => { if (a.name === 'captain-hands') ensureConnected() })
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.hands) {
    prefs = { on: false, front: false, ...(changes.hands.newValue || {}) }
    ensureConnected()
  }
  if (area === 'local' && changes.handsStopped) {
    stopped = new Set(changes.handsStopped.newValue || [])
    send({ type: 'stopped', agents: [...stopped] })
  }
  if (area === 'local' && changes.conn) ensureConnected()
})
chrome.permissions?.onAdded.addListener(() => ensureConnected())
chrome.permissions?.onRemoved.addListener(() => ensureConnected())
chrome.windows.onFocusChanged.addListener((id) => { if (id !== chrome.windows.WINDOW_ID_NONE) send({ type: 'focus' }) })

// ── one act ──

function describeAct(action, a) {
  const at = a.ref ? ` [${a.ref}]` : a.coordinate ? ` at (${a.coordinate.join(', ')})` : a.selector ? ` ${a.selector}` : ''
  switch (action) {
    case 'navigate': return a.url === 'back' ? 'Went back' : a.url === 'forward' ? 'Went forward' : `Opened ${hostOf(a.url) || a.url}`
    case 'tab_new': return a.url ? `New tab: ${hostOf(a.url) || a.url}` : 'New tab'
    case 'click': return `Clicked${at || (a.text ? ` “${clip(a.text, 40)}”` : '')}`
    case 'double_click': return `Double-clicked${at}`
    case 'right_click': return `Right-clicked${at}`
    case 'triple_click': return `Triple-clicked${at}`
    case 'hover': return `Pointed at${at}`
    case 'drag': return 'Dragged'
    case 'type': return `Typed ${a.text ? `“${clip(a.text.length > 40 ? a.text.slice(0, 40) + '…' : a.text, 44)}”` : ''}`
    case 'press_key': return `Pressed ${a.key || ''}`
    case 'form_input': return `Filled${at}`
    case 'scroll': return `Scrolled ${a.scroll_direction || 'down'}`
    case 'scroll_to': return `Scrolled to${at}`
    case 'screenshot': return 'Took a screenshot'
    case 'zoom': return 'Looked closer'
    case 'read_page': return 'Read the page'
    case 'find': return `Looked for “${clip(a.query || a.text || '', 40)}”`
    case 'page_text': return 'Read the text'
    case 'page_files': return 'Listed the files'
    case 'download': return 'Downloaded files'
    case 'save_page': return 'Saved the page'
    case 'javascript': return 'Ran a script'
    case 'console': return 'Read the console'
    case 'network_requests': return 'Read the network log'
    case 'upload_file': return `Attached ${a.file?.name || 'a file'}`
    case 'tab_close': return `Closed tab ${a.tab_id}`
    default: return action.replace(/_/g, ' ')
  }
}

function hostOf(url) {
  try { return new URL(String(url).includes('://') ? url : 'https://' + url).host.replace(/^www\./, '') } catch { return '' }
}

async function runAct(msg) {
  await loadState()
  const id = String(msg.id || '')
  const agent = { slug: String(msg.agent?.slug || ''), name: oneLine(msg.agent?.name || msg.agent?.slug || 'Agent').slice(0, 40) }
  const action = String(msg.action || '')
  const args = msg.args && typeof msg.args === 'object' ? msg.args : {}
  const entry = { id, at: Date.now(), slug: agent.slug, name: agent.name, action, text: describeAct(action, args), state: 'working' }
  const quiet = action === 'status' || action === 'release'
  if (!quiet) { log.push(entry); if (log.length > LOG_MAX) log = log.slice(-LOG_MAX); saveLog() }
  let result
  try {
    if (stopped.has(agent.slug)) throw new Refusal("The person stopped you using their browser. Don't try again unless they ask you to.")
    const fn = ACTIONS[action]
    if (!fn) throw new Refusal(`“${action}” isn't something this browser does (Captain for Chrome ${VERSION}).`)
    const conn = wsConn || (await loadConn())
    result = await fn({ id, agent, args, conn })
    result = { ok: true, ...result }
    entry.state = 'done'
  } catch (e) {
    result = { ok: false, error: cleanError(e) }
    entry.state = 'failed'
    entry.error = clip(result.error, 160)
  }
  cancelled.delete(id)
  if (!quiet) saveLog()
  send({ type: 'result', id, ...result })
}

function cleanError(e) {
  let s = String(e?.message || e || 'Something went wrong.')
  s = s.replace(/^(Uncaught )?(Error|TypeError): /, '').split('\n    at ')[0]
  if (/Cannot access a chrome|cannot be scripted|Cannot attach to this target|chrome-extension:\/\//i.test(s)) {
    return "Chrome doesn't let extensions work on this page (a chrome:// or extension page, or the Web Store)."
  }
  if (/Another debugger is already attached/i.test(s)) {
    return 'DevTools (or another debugger) is open on that tab — the person has to close it first.'
  }
  if (/No tab with id/i.test(s)) return 'That tab was closed.'
  return clip(s, 2000)
}

// ── the agent's tabs ──

function colorFor(slug) {
  let h = 0
  for (const ch of slug) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return GROUP_COLORS[h % GROUP_COLORS.length]
}

async function agentGroup(agent) {
  const g = groups[agent.slug]
  if (!g || g.groupId == null) return null
  try {
    const tg = await chrome.tabGroups.get(g.groupId)
    g.windowId = tg.windowId
    if (tg.title !== agent.name) chrome.tabGroups.update(g.groupId, { title: agent.name }).catch(() => {})
    return g
  } catch {
    delete groups[agent.slug]
    saveGroups()
    return null
  }
}

async function groupTabs(g) {
  return g ? chrome.tabs.query({ groupId: g.groupId }) : []
}

async function newTab(agent, { front = prefs.front } = {}) {
  let g = await agentGroup(agent)
  let windowId = g?.windowId
  if (windowId == null) {
    const w = await chrome.windows.getLastFocused({ windowTypes: ['normal'] }).catch(() => null)
    windowId = w?.id
  }
  let tab
  if (windowId == null) {
    const w = await chrome.windows.create({ focused: false, url: 'about:blank' })
    windowId = w.id
    tab = w.tabs[0]
  } else {
    tab = await chrome.tabs.create({ windowId, url: 'about:blank', active: !!front })
  }
  if (g) {
    await chrome.tabs.group({ groupId: g.groupId, tabIds: [tab.id] })
  } else {
    const groupId = await chrome.tabs.group({ tabIds: [tab.id], createProperties: { windowId } })
    await chrome.tabGroups.update(groupId, { title: agent.name, color: colorFor(agent.slug) }).catch(() => {})
    g = { groupId, windowId, name: agent.name }
    groups[agent.slug] = g
  }
  g.current = tab.id
  saveGroups()
  return chrome.tabs.get(tab.id)
}

async function tabFor(agent, args, { create = false } = {}) {
  const g = await agentGroup(agent)
  if (args.tab_id !== undefined && args.tab_id !== null && args.tab_id !== '') {
    const id = Number(args.tab_id)
    const tab = await chrome.tabs.get(id).catch(() => null)
    if (!tab) throw new Refusal(`There's no tab ${id} (it was closed?). Call tabs to see yours.`)
    if (!g || tab.groupId !== g.groupId) {
      throw new Refusal(`Tab ${id} isn't one of yours. You can only use the tabs in your own tab group “${agent.name}” — the person can drag a tab into it to share it with you.`)
    }
    g.current = id
    saveGroups()
    return tab
  }
  if (g) {
    const tabs = await groupTabs(g)
    const cur = tabs.find((t) => t.id === g.current) || tabs.find((t) => t.active) || tabs[0]
    if (cur) {
      if (g.current !== cur.id) { g.current = cur.id; saveGroups() }
      return cur
    }
  }
  if (create) return newTab(agent)
  throw new Refusal("You have no tab open in the person's Chrome yet: navigate to a page first (it opens one in your own tab group).")
}

function guardTab(tab, conn) {
  const url = tab.pendingUrl || tab.url || ''
  if (!url || url === 'about:blank' || /^https?:/i.test(url)) {
    if (conn && url && sameOrigin(url, conn.url)) throw new Refusal("Flight Deck's own pages are off-limits in the person's browser.")
    return
  }
  throw new Refusal(`That tab shows ${clip(url, 80)} — agents work on web pages only. Navigate it to a web address.`)
}

function sameOrigin(a, b) {
  try { return new URL(a).origin === new URL(b).origin } catch { return false }
}

function checkUrl(raw, conn) {
  const text = String(raw || '').trim()
  if (!text) throw new Refusal('Give a url.')
  let u
  try { u = new URL(/^[a-z][a-z0-9+.-]*:/i.test(text) ? text : 'https://' + text) } catch {
    throw new Refusal(`“${clip(text, 80)}” isn't a web address.`)
  }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Refusal(`Only web pages (http/https) — not ${u.protocol} addresses.`)
  if (conn && sameOrigin(u.href, conn.url)) throw new Refusal("Flight Deck's own pages are off-limits in the person's browser.")
  return u.href
}

async function useTab(ctx, args, opts = {}) {
  const tab = await tabFor(ctx.agent, args, opts)
  guardTab(tab, ctx.conn)
  await attach(tab.id)
  if (prefs.front && !tab.active) await chrome.tabs.update(tab.id, { active: true }).catch(() => {})
  return chrome.tabs.get(tab.id)
}

chrome.tabs.onCreated.addListener((tab) => {
  if (tab.openerTabId !== undefined && attached.has(tab.openerTabId)) {
    const list = openedFrom.get(tab.openerTabId) || []
    list.push(tab.id)
    openedFrom.set(tab.openerTabId, list.slice(-10))
  }
})
chrome.tabs.onRemoved.addListener((tabId) => {
  attached.delete(tabId)
  openedFrom.delete(tabId)
})

// ── the debugger ──

function cdp(tabId, method, params = {}) {
  return chrome.debugger.sendCommand({ tabId }, method, params)
}

async function attach(tabId) {
  const t = attached.get(tabId)
  if (t) { t.lastUse = Date.now(); return t }
  await chrome.debugger.attach({ tabId }, '1.3')
  const state = { lastUse: Date.now(), since: Date.now(), world: null, console: [], net: new Map(), shotScale: 1, loads: [] }
  attached.set(tabId, state)
  await Promise.all([
    cdp(tabId, 'Page.enable'),
    cdp(tabId, 'Runtime.enable'),
    cdp(tabId, 'Log.enable').catch(() => {}),
    cdp(tabId, 'Network.enable', { maxPostDataSize: 0 }).catch(() => {}),
  ])
  return state
}

async function detach(tabId) {
  attached.delete(tabId)
  try { await chrome.debugger.detach({ tabId }) } catch { /* gone already */ }
}

function detachAll() {
  for (const id of [...attached.keys()]) detach(id)
}

setInterval(() => {
  const now = Date.now()
  for (const [id, t] of attached) if (now - t.lastUse > IDLE_DETACH_MS) detach(id)
}, 30000)

chrome.debugger?.onDetach.addListener((src, reason) => {
  if (src.tabId === undefined) return
  const had = attached.delete(src.tabId)
  if (had && reason === 'canceled_by_user') {
    // The person pressed Cancel on Chrome's "is debugging this browser" bar:
    // every agent that was working here stops until they let it again.
    for (const [slug, g] of Object.entries(groups)) {
      if (g.recentAt && Date.now() - g.recentAt < 10 * 60 * 1000) stopped.add(slug)
    }
    chrome.storage.local.set({ handsStopped: [...stopped] }).catch(() => {})
    detachAll()
  }
})

function remoteText(a) {
  if (!a) return ''
  if (a.value !== undefined) return typeof a.value === 'string' ? a.value : JSON.stringify(a.value)
  if (a.unserializableValue) return a.unserializableValue
  return a.description || a.type || ''
}

function pushCapped(list, item) {
  list.push(item)
  if (list.length > BUFFER_MAX) list.splice(0, list.length - BUFFER_MAX)
}

chrome.debugger?.onEvent.addListener((src, method, p) => {
  const t = attached.get(src.tabId)
  if (!t) return
  switch (method) {
    case 'Runtime.consoleAPICalled': {
      const top = p.stackTrace?.callFrames?.[0]
      pushCapped(t.console, { at: Date.now(), level: p.type === 'warning' ? 'warn' : p.type, text: clip((p.args || []).map(remoteText).join(' '), 2000),
        where: top ? `${top.url}:${top.lineNumber + 1}` : '' })
      break
    }
    case 'Runtime.exceptionThrown': {
      const d = p.exceptionDetails || {}
      pushCapped(t.console, { at: Date.now(), level: 'error', text: clip(d.exception?.description || d.text || 'Uncaught error', 2000),
        where: d.url ? `${d.url}:${(d.lineNumber || 0) + 1}` : '' })
      break
    }
    case 'Log.entryAdded': {
      const e = p.entry || {}
      pushCapped(t.console, { at: Date.now(), level: e.level === 'warning' ? 'warn' : e.level, text: clip(e.text || '', 2000), where: e.url || '' })
      break
    }
    case 'Network.requestWillBeSent': {
      if (t.net.size >= BUFFER_MAX) t.net.delete(t.net.keys().next().value)
      t.net.set(p.requestId, { at: Date.now(), method: p.request?.method || 'GET', url: p.request?.url || '', type: p.type || '', status: 0 })
      break
    }
    case 'Network.responseReceived': {
      const r = t.net.get(p.requestId)
      if (r) { r.status = p.response?.status || 0; r.mime = p.response?.mimeType || '' }
      break
    }
    case 'Network.loadingFinished': {
      const r = t.net.get(p.requestId)
      if (r) r.size = p.encodedDataLength || 0
      break
    }
    case 'Network.loadingFailed': {
      const r = t.net.get(p.requestId)
      if (r) r.failed = p.canceled ? 'canceled' : (p.errorText || 'failed')
      break
    }
    case 'Page.loadEventFired': {
      const waiting = t.loads
      t.loads = []
      for (const done of waiting) done(true)
      break
    }
    case 'Page.frameNavigated':
      if (!p.frame?.parentId) t.world = null
      break
    case 'Runtime.executionContextsCleared':
      t.world = null
      break
    case 'Runtime.executionContextDestroyed':
      if (t.world === p.executionContextId) t.world = null
      break
  }
})

let helperSrc = ''
let captureSrc = ''
async function source(file) {
  const r = await fetch(chrome.runtime.getURL(file))
  return r.text()
}

async function world(tabId) {
  const t = attached.get(tabId) || (await attach(tabId))
  if (t.world) return t.world
  const { frameTree } = await cdp(tabId, 'Page.getFrameTree')
  const { executionContextId } = await cdp(tabId, 'Page.createIsolatedWorld', { frameId: frameTree.frame.id, worldName: 'captain-hands' })
  t.world = executionContextId
  return t.world
}

async function evaluate(tabId, expression, { main = false, timeout = 20000, repl = false } = {}) {
  const params = { expression, returnByValue: true, awaitPromise: true, userGesture: true }
  if (repl) params.replMode = true
  if (!main) params.contextId = await world(tabId)
  let r
  try {
    r = await withTimeout(cdp(tabId, 'Runtime.evaluate', params), timeout, 'The page took too long to answer.')
  } catch (e) {
    if (main || !/context/i.test(String(e.message || e))) throw e
    const t = attached.get(tabId)
    if (t) t.world = null
    params.contextId = await world(tabId)
    r = await withTimeout(cdp(tabId, 'Runtime.evaluate', params), timeout, 'The page took too long to answer.')
  }
  if (r.exceptionDetails) {
    const d = r.exceptionDetails
    throw new Error(d.exception?.description || d.exception?.value || d.text || 'The page script failed.')
  }
  return r.result
}

async function helper(tabId, fn, arg = {}) {
  if (!helperSrc) helperSrc = await source('hands-page.js')
  const res = await evaluate(tabId, `${helperSrc}\n;globalThis.__captainHands.${fn}(${JSON.stringify(arg)})`)
  return res.value
}

async function capturePage(tabId) {
  if (!captureSrc) captureSrc = await source('capture.js')
  const res = await evaluate(tabId, captureSrc, { timeout: 30000 })
  const v = res.value
  if (!v || !v.ok) throw new Error('Could not read the page.')
  return v
}

function withTimeout(p, ms, why) {
  let t
  return Promise.race([p, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(why || 'Timed out.')), ms) })]).finally(() => clearTimeout(t))
}

// ── waiting for pages ──

async function navigateTab(tabId, url) {
  const t = attached.get(tabId) || (await attach(tabId))
  const loaded = new Promise((resolve) => t.loads.push(resolve))
  const res = await cdp(tabId, 'Page.navigate', { url })
  if (res.errorText && res.errorText !== 'net::ERR_ABORTED') throw new Refusal(`Couldn't open ${url}: ${res.errorText}`)
  if (!res.loaderId) return true   // same document (#hash)
  return withTimeout(loaded, NAV_WAIT_MS, 'slow').then(() => true, () => false)
}

async function waitComplete(tabId, ms) {
  const end = Date.now() + ms
  while (Date.now() < end) {
    const tab = await chrome.tabs.get(tabId).catch(() => null)
    if (!tab) return false
    if (tab.status === 'complete') return true
    await sleep(200)
  }
  return false
}

async function groupTabIds(agent) {
  const g = await agentGroup(agent)
  return new Set((await groupTabs(g)).map((t) => t.id))
}

// After a click or a key: a new tab it opened becomes the agent's, a page it
// started loading is waited for; what changed is said. *before* = the
// agent's tab ids before it acted (a tab opened from its tab joins its group).
async function settle(ctx, tab, beforeUrl, before = null) {
  let newId = null
  for (let waited = 0; waited < 800; waited += 160) {
    await sleep(160)
    const opened = (openedFrom.get(tab.id) || []).splice(0)
    if (opened.length) { newId = opened[opened.length - 1]; break }
    if (before) {
      const now = await groupTabIds(ctx.agent)
      const fresh = [...now].filter((id) => !before.has(id))
      if (fresh.length) { newId = fresh[fresh.length - 1]; break }
    }
    const cur = await chrome.tabs.get(tab.id).catch(() => null)
    if (!cur || cur.status === 'loading' || cur.url !== beforeUrl) break
    if (waited >= 320 && !before) break
  }
  if (newId !== null) {
    const g = await agentGroup(ctx.agent)
    const nt = await chrome.tabs.get(newId).catch(() => null)
    if (nt && g) {
      if (nt.groupId !== g.groupId) await chrome.tabs.group({ groupId: g.groupId, tabIds: [nt.id] }).catch(() => {})
      g.current = nt.id
      saveGroups()
      await attach(nt.id).catch(() => {})
      await waitComplete(nt.id, SETTLE_WAIT_MS)
      const fresh = await chrome.tabs.get(nt.id)
      return `It opened a new tab ${fresh.id} — now your current tab: ${clip(fresh.title || '', 120)} — ${fresh.url || fresh.pendingUrl}`
    }
  }
  const now = await chrome.tabs.get(tab.id).catch(() => null)
  if (!now) return 'The tab closed.'
  if (now.status === 'loading') await waitComplete(tab.id, SETTLE_WAIT_MS)
  const after = await chrome.tabs.get(tab.id).catch(() => null)
  if (after && after.url !== beforeUrl) return `Now at: ${clip(after.title || '', 120)} — ${after.url}`
  return ''
}

// ── input ──

const MOD = { alt: 1, option: 1, opt: 1, ctrl: 2, control: 2, meta: 4, cmd: 4, command: 4, super: 4, win: 4, shift: 8 }

function modsOf(text) {
  let m = 0
  for (const part of String(text || '').toLowerCase().split(/[+\s,]+/)) {
    if (!part) continue
    let bit = MOD[part] || 0
    if ((part === 'cmd' || part === 'command') && !IS_MAC) bit = 2
    m |= bit
  }
  return m
}

async function mouseClick(tabId, x, y, { button = 'left', count = 1, mods = 0 } = {}) {
  const buttons = button === 'right' ? 2 : button === 'middle' ? 4 : 1
  await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, modifiers: mods })
  for (let i = 1; i <= count; i++) {
    await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button, buttons, clickCount: i, modifiers: mods })
    await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button, buttons: 0, clickCount: i, modifiers: mods })
  }
}

const NAMED_KEYS = {
  enter: ['Enter', 'Enter', 13, '\r'], return: ['Enter', 'Enter', 13, '\r'], tab: ['Tab', 'Tab', 9],
  escape: ['Escape', 'Escape', 27], esc: ['Escape', 'Escape', 27], backspace: ['Backspace', 'Backspace', 8],
  delete: ['Delete', 'Delete', 46], del: ['Delete', 'Delete', 46], space: [' ', 'Space', 32, ' '],
  arrowup: ['ArrowUp', 'ArrowUp', 38], up: ['ArrowUp', 'ArrowUp', 38], arrowdown: ['ArrowDown', 'ArrowDown', 40], down: ['ArrowDown', 'ArrowDown', 40],
  arrowleft: ['ArrowLeft', 'ArrowLeft', 37], left: ['ArrowLeft', 'ArrowLeft', 37], arrowright: ['ArrowRight', 'ArrowRight', 39], right: ['ArrowRight', 'ArrowRight', 39],
  home: ['Home', 'Home', 36], end: ['End', 'End', 35], pageup: ['PageUp', 'PageUp', 33], pagedown: ['PageDown', 'PageDown', 34],
  insert: ['Insert', 'Insert', 45],
}
for (let i = 1; i <= 12; i++) NAMED_KEYS[`f${i}`] = [`F${i}`, `F${i}`, 111 + i]

const MAC_COMMANDS = { a: 'selectAll', c: 'copy', x: 'cut', v: 'paste', z: 'undo' }

function keySpec(name, mods) {
  const lower = name.toLowerCase()
  if (NAMED_KEYS[lower]) {
    const [key, code, keyCode, text] = NAMED_KEYS[lower]
    return { key, code, keyCode, text: mods & (2 | 4) ? '' : (text || '') }
  }
  if ([...name].length === 1) {
    const ch = name
    const up = ch.toUpperCase()
    let code = ''
    let keyCode = up.charCodeAt(0)
    if (/[a-z]/i.test(ch)) code = `Key${up}`
    else if (/[0-9]/.test(ch)) code = `Digit${ch}`
    else keyCode = 0
    const shifted = mods & 8 && /[a-z]/.test(ch) ? up : ch
    return { key: shifted, code, keyCode, text: mods & (2 | 4) ? '' : shifted }
  }
  throw new Refusal(`“${name}” isn't a key I know (try Enter, Tab, Escape, ArrowDown, a letter, or a combination like cmd+a).`)
}

async function pressCombo(tabId, combo) {
  const parts = combo.split('+').filter(Boolean)
  if (!parts.length) return
  const keyName = parts.pop()
  const mods = modsOf(parts.join('+'))
  const spec = keySpec(keyName, mods)
  const base = { key: spec.key, code: spec.code, windowsVirtualKeyCode: spec.keyCode, nativeVirtualKeyCode: spec.keyCode, modifiers: mods }
  const down = { ...base, type: spec.text ? 'keyDown' : 'rawKeyDown' }
  if (spec.text) { down.text = spec.text; down.unmodifiedText = spec.text }
  const cmdKey = IS_MAC ? mods & 4 : mods & 2
  if (IS_MAC && cmdKey && MAC_COMMANDS[spec.key.toLowerCase()]) {
    down.commands = [spec.key.toLowerCase() === 'z' && mods & 8 ? 'redo' : MAC_COMMANDS[spec.key.toLowerCase()]]
  }
  await cdp(tabId, 'Input.dispatchKeyEvent', down)
  await cdp(tabId, 'Input.dispatchKeyEvent', { ...base, type: 'keyUp' })
}

async function typeText(tabId, text) {
  if (text.length > 300) {
    await cdp(tabId, 'Input.insertText', { text })
    return
  }
  for (const ch of text) {
    if (ch === '\n') { await pressCombo(tabId, 'Enter'); continue }
    if (ch === '\t') { await pressCombo(tabId, 'Tab'); continue }
    if (ch.length > 1 || ch.charCodeAt(0) > 0x7e) { await cdp(tabId, 'Input.insertText', { text: ch }); continue }
    const spec = /[a-z0-9]/i.test(ch) ? keySpec(ch, 0) : { key: ch, code: '', keyCode: 0, text: ch }
    const base = { key: ch, code: spec.code, windowsVirtualKeyCode: spec.keyCode || ch.toUpperCase().charCodeAt(0), nativeVirtualKeyCode: spec.keyCode }
    await cdp(tabId, 'Input.dispatchKeyEvent', { ...base, type: 'keyDown', text: ch, unmodifiedText: ch })
    await cdp(tabId, 'Input.dispatchKeyEvent', { ...base, type: 'keyUp' })
  }
}

// ── screenshots ──

function b64ToBlob(b64, type) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type })
}

async function blobToB64(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer())
  let s = ''
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000))
  return btoa(s)
}

async function rawShot(tabId) {
  const params = { format: 'png', fromSurface: true }
  try {
    return await withTimeout(cdp(tabId, 'Page.captureScreenshot', params), 5000, 'slow')
  } catch {
    // A tab in the background may not draw: bring it to the front once.
    await chrome.tabs.update(tabId, { active: true }).catch(() => {})
    await sleep(300)
    return withTimeout(cdp(tabId, 'Page.captureScreenshot', params), 10000, 'The tab did not draw a picture in time.')
  }
}

async function shoot(tabId, region = null) {
  const inf = await helper(tabId, 'info')
  const shot = await rawShot(tabId)
  const bmp = await createImageBitmap(b64ToBlob(shot.data, 'image/png'))
  const k = bmp.width / Math.max(1, inf.vw)   // device pixels per CSS pixel
  const t = attached.get(tabId)
  let sx = 0
  let sy = 0
  let sw = bmp.width
  let sh = bmp.height
  let outW
  let outH
  if (region) {
    const scale = t?.shotScale || 1
    const [x0, y0, x1, y1] = region.map((v) => Number(v) / scale)
    if (![x0, y0, x1, y1].every(Number.isFinite) || x1 <= x0 || y1 <= y0) throw new Refusal('region is [x0, y0, x1, y1] from a screenshot, with x1 > x0 and y1 > y0.')
    sx = Math.max(0, x0 * k); sy = Math.max(0, y0 * k)
    sw = Math.min(bmp.width - sx, (x1 - x0) * k); sh = Math.min(bmp.height - sy, (y1 - y0) * k)
    const z = Math.min(4, ZOOM_W / Math.max(1, x1 - x0))
    outW = Math.round((x1 - x0) * z)
    outH = Math.round((y1 - y0) * z)
  } else {
    outW = Math.min(inf.vw, SHOT_MAX_W)
    outH = Math.round(inf.vh * (outW / inf.vw))
    if (t) t.shotScale = outW / inf.vw
  }
  const canvas = new OffscreenCanvas(Math.max(1, outW), Math.max(1, outH))
  canvas.getContext('2d').drawImage(bmp, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height)
  bmp.close?.()
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 })
  return { image: await blobToB64(blob), width: canvas.width, height: canvas.height, info: inf, scale: t?.shotScale || 1 }
}

// ── pointing ──

async function pointFor(tab, args) {
  if (Array.isArray(args.coordinate) && args.coordinate.length >= 2) {
    const scale = attached.get(tab.id)?.shotScale || 1
    const [x, y] = args.coordinate.map(Number)
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Refusal('coordinate is [x, y] from a screenshot.')
    return { x: x / scale, y: y / scale, what: `(${x}, ${y})` }
  }
  if (args.ref) return helper(tab.id, 'point', { ref: args.ref })
  if (args.selector || args.text || args.role) {
    return helper(tab.id, 'pointBy', { selector: args.selector || '', text: args.text || '', role: args.role || '', nth: args.nth ?? null })
  }
  throw new Refusal('Point at something: ref (from read_page or find — best), coordinate (from a screenshot), selector, or text.')
}

function pointNotes(p) {
  return [p.covered ? `Careful: ${p.covered} was on top of it — the click may have landed there instead.` : '',
    p.visible === false ? 'It has no size on the page (hidden?) — check with a screenshot.' : '', p.note || ''].filter(Boolean)
}

async function clickLike(ctx, args, { button = 'left', count = 1, verb = 'Clicked' } = {}) {
  const tab = await useTab(ctx, args)
  const p = await pointFor(tab, args)
  const before = await groupTabIds(ctx.agent)
  await mouseClick(tab.id, p.x, p.y, { button, count, mods: modsOf(args.modifiers) })
  const after = await settle(ctx, tab, tab.url, before)
  return { content: [`${verb} ${p.what}${p.ref ? ` [${p.ref}]` : ''}.`, ...pointNotes(p), after].filter(Boolean).join('\n') }
}

// ── files ──

async function fetchInPage(tabId, url) {
  const expr = `(async () => {
    const r = await fetch(${JSON.stringify(url)}, { credentials: 'include' })
    if (!r.ok) return { ok: false, status: r.status }
    const blob = await r.blob()
    if (blob.size > ${FILE_MAX}) return { ok: false, big: blob.size }
    const data = await new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(String(fr.result).split(',')[1] || ''); fr.onerror = rej; fr.readAsDataURL(blob) })
    return { ok: true, type: blob.type, size: blob.size, data, disposition: r.headers.get('content-disposition') || '', final: r.url }
  })()`
  try {
    const res = await evaluate(tabId, expr, { timeout: 45000 })
    return res.value
  } catch (e) {
    return { ok: false, error: String(e.message || e) }
  }
}

async function fetchHere(url) {
  try {
    const r = await fetch(url, { credentials: 'include' })
    if (!r.ok) return { ok: false, status: r.status }
    const blob = await r.blob()
    if (blob.size > FILE_MAX) return { ok: false, big: blob.size }
    return { ok: true, type: blob.type, size: blob.size, blob, disposition: r.headers.get('content-disposition') || '', final: r.url }
  } catch (e) {
    return { ok: false, error: String(e.message || e) }
  }
}

const TYPE_EXT = { 'application/pdf': 'pdf', 'text/csv': 'csv', 'application/zip': 'zip', 'image/png': 'png', 'image/jpeg': 'jpg',
  'image/gif': 'gif', 'image/webp': 'webp', 'application/json': 'json', 'text/plain': 'txt',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx' }

function fileNameOf(url, got, hint) {
  const m = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(got.disposition || '')
  let name = m ? decodeURIComponent(m[1].replace(/"$/, '')) : ''
  if (!name) name = hint || ''
  if (!name) { try { name = decodeURIComponent(new URL(got.final || url).pathname.split('/').pop() || '') } catch { name = '' } }
  name = name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').trim() || 'file'
  const ext = TYPE_EXT[(got.type || '').split(';')[0]]
  if (ext && !/\.[a-z0-9]{2,5}$/i.test(name)) name += '.' + ext
  return name.slice(0, 120)
}

async function uploadFiles(ctx, files, page) {
  const form = new FormData()
  form.append('meta', JSON.stringify({ page_url: page.url || '', page_title: page.title || '' }))
  for (const f of files) form.append('file', f.blob, f.name)
  const r = await fetch(`${ctx.conn.url}/fd/clip/hands/files/${encodeURIComponent(ctx.id)}`, {
    method: 'POST', body: form, headers: { Authorization: `Bearer ${ctx.conn.token}` },
  })
  let body = null
  try { body = await r.json() } catch { /* not JSON */ }
  if (!r.ok) throw new Error((body && typeof body.detail === 'string' && body.detail) || `Flight Deck answered ${r.status}`)
  return body
}

// ── the actions ──

function tabLine(t, current) {
  return `- tab ${t.id}${t.id === current ? ' (current)' : ''}: ${clip(oneLine(t.title || '(loading)'), 100)} — ${t.url || t.pendingUrl || ''}`
}

async function listTabs(ctx) {
  const g = await agentGroup(ctx.agent)
  const tabs = await groupTabs(g)
  if (!tabs.length) return 'You have no tabs open yet. navigate opens a page in a tab of your own (your tab group in the person\'s Chrome).'
  return [`Your tabs (group “${ctx.agent.name}”):`, ...tabs.map((t) => tabLine(t, g.current))].join('\n')
}

const ACTIONS = {
  async status(ctx) {
    const lines = [`Connected: the person's own Chrome (${browserLabel()}, Captain for Chrome ${VERSION}), signed in as them.`, await listTabs(ctx)]
    return { content: lines.join('\n') }
  },

  async tabs(ctx) {
    return { content: await listTabs(ctx) }
  },

  async tab_new(ctx, args = ctx.args) {
    const url = args.url ? checkUrl(args.url, ctx.conn) : ''
    const tab = await newTab(ctx.agent)
    await attach(tab.id)
    if (!url) return { content: `Opened a new empty tab ${tab.id} — now your current tab.` }
    const done = await navigateTab(tab.id, url)
    const t = await chrome.tabs.get(tab.id)
    return { content: `Opened tab ${t.id} (now current): ${clip(t.title || '', 120)} — ${t.url}${done ? '' : '\n(It was still loading after 30 s.)'}` }
  },

  async tab_close(ctx, args = ctx.args) {
    if (args.tab_id === undefined || args.tab_id === null || args.tab_id === '') throw new Refusal('Say which tab: tab_id (from tabs).')
    const tab = await tabFor(ctx.agent, args)
    await detach(tab.id)
    await chrome.tabs.remove(tab.id)
    const g = await agentGroup(ctx.agent)
    if (g && g.current === tab.id) { g.current = null; saveGroups() }
    return { content: `Closed tab ${tab.id}.\n${await listTabs(ctx)}` }
  },

  async tab_select(ctx, args = ctx.args) {
    if (args.tab_id === undefined || args.tab_id === null || args.tab_id === '') throw new Refusal('Say which tab: tab_id (from tabs).')
    const tab = await tabFor(ctx.agent, args)
    return { content: `Tab ${tab.id} is now your current tab: ${clip(tab.title || '', 120)} — ${tab.url}` }
  },

  async navigate(ctx, args = ctx.args) {
    const where = String(args.url || '').trim()
    if (where === 'back' || where === 'forward') {
      const tab = await useTab(ctx, args)
      await (where === 'back' ? chrome.tabs.goBack(tab.id) : chrome.tabs.goForward(tab.id)).catch(() => {
        throw new Refusal(`There's no page to go ${where} to.`)
      })
      await sleep(300)
      await waitComplete(tab.id, 15000)
      const t = await chrome.tabs.get(tab.id)
      guardTab(t, ctx.conn)
      return { content: `Went ${where}: ${clip(t.title || '', 120)} — ${t.url}` }
    }
    const url = checkUrl(where, ctx.conn)
    const tab = await useTab(ctx, args, { create: true })
    const done = await navigateTab(tab.id, url)
    const t = await chrome.tabs.get(tab.id)
    return { content: `Opened in tab ${t.id}: ${clip(t.title || '', 120)} — ${t.url}${done ? '' : '\n(It was still loading after 30 s — take a screenshot or read the page to see.)'}` }
  },

  async back(ctx) { return ACTIONS.navigate({ ...ctx, args: { ...ctx.args, url: 'back' } }) },
  async forward(ctx) { return ACTIONS.navigate({ ...ctx, args: { ...ctx.args, url: 'forward' } }) },

  async read_page(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const text = await helper(tab.id, 'tree', { filter: args.filter === 'all' ? 'all' : 'interactive', refId: args.ref || args.ref_id || '',
      maxChars: Math.min(Math.max(Number(args.max_chars) || 40000, 2000), 200000) })
    return { content: text }
  },

  async find(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    return { content: await helper(tab.id, 'find', { query: args.query || args.text || '' }) }
  },

  async page_text(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const cap = await capturePage(tab.id)
    const facts = [cap.tables?.length ? `${cap.tables.length} tables` : '', cap.files?.length ? `${cap.files.length} files (page_files lists them)` : '',
      cap.selection ? 'a selection' : ''].filter(Boolean)
    const head = `# ${cap.meta?.title || tab.title || ''}\n${cap.meta?.url || tab.url}${facts.length ? `\nThe page has ${facts.join(', ')}.` : ''}\n\n`
    const body = (cap.selection ? `Selected on the page:\n> ${cap.selection.replace(/\n/g, '\n> ')}\n\n` : '') + (cap.markdown || '(no text found)')
    const offset = Math.max(0, Number(args.offset) || 0)
    const max = Math.min(Math.max(Number(args.max_chars) || TEXT_DEFAULT, 1000), 200000)
    const part = body.slice(offset, offset + max)
    const rest = body.length - offset - part.length
    return { content: head + part + (rest > 0 ? `\n\n… (${rest.toLocaleString()} more characters — page_text with offset=${offset + part.length})` : '') }
  },

  async page_files(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const cap = await capturePage(tab.id)
    const all = [...(cap.files || []), ...(cap.images || [])]
    if (!all.length) return { content: 'No files are linked from this page.' }
    const lines = all.slice(0, 150).map((f, i) => `${i + 1}. ${f.name} — ${f.kind}${f.context ? ` (“${clip(f.context, 60)}”)` : ''} — ${f.url}`)
    return { content: `${all.length} file${all.length === 1 ? '' : 's'} on ${tab.url}:\n${lines.join('\n')}${all.length > 150 ? `\n… and ${all.length - 150} more` : ''}\nTo keep some: download with urls=[…] (or all=true for the documents).`,
      data: { files: all.slice(0, 150).map((f) => ({ name: f.name, url: f.url, kind: f.kind })) } }
  },

  async download(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    let wanted = Array.isArray(args.urls) ? args.urls.map(String) : args.url ? [String(args.url)] : []
    const names = {}
    if (!wanted.length && args.all) {
      const cap = await capturePage(tab.id)
      const docs = (cap.files || []).filter((f) => ['document', 'archive', 'data', 'design'].includes(f.kind))
      for (const f of docs) names[f.url] = f.name
      wanted = docs.map((f) => f.url)
    }
    if (!wanted.length) throw new Refusal('Say which files: urls=[…] (from page_files), or all=true for the documents on the page.')
    wanted = [...new Set(wanted)].slice(0, FILES_MAX)
    const got = []
    const failed = []
    for (const raw of wanted) {
      if (cancelled.has(ctx.id)) break
      let url
      try { url = new URL(raw, tab.url).href } catch { failed.push(`${raw}: not an address`); continue }
      if (!/^https?:/i.test(url)) { failed.push(`${url}: not a web address`); continue }
      if (sameOrigin(url, ctx.conn.url)) { failed.push(`${url}: Flight Deck's own address`); continue }
      let res = sameOrigin(url, tab.url) ? await fetchInPage(tab.id, url) : { ok: false }
      if (!res.ok) {
        const here = await fetchHere(url)
        if (here.ok) res = here
        else if (!sameOrigin(url, tab.url)) res = await fetchInPage(tab.id, url)
        if (!res.ok && !here.ok && here.status) res = here
      }
      if (!res.ok) {
        failed.push(`${url}: ${res.big ? 'over 25 MB' : res.status ? `the site answered ${res.status}` : 'the browser could not fetch it (another site that does not allow it?)'}`)
        continue
      }
      const type = (res.type || '').split(';')[0]
      const looksFile = /\.[a-z0-9]{2,5}($|[?#])/i.test(new URL(url).pathname) && !/\.(html?|php|aspx?)$/i.test(new URL(url).pathname)
      if (type === 'text/html' && looksFile) { failed.push(`${url}: got a web page instead of the file (a sign-in page?)`); continue }
      const blob = res.blob || b64ToBlob(res.data || '', type || 'application/octet-stream')
      got.push({ blob, name: fileNameOf(url, res, names[url]), url })
    }
    if (!got.length) throw new Refusal(`Nothing could be downloaded:\n${failed.map((f) => `- ${f}`).join('\n')}`)
    const up = await uploadFiles(ctx, got, { url: tab.url, title: tab.title })
    const lines = [`Downloaded ${got.length} file${got.length === 1 ? '' : 's'} with the person's browser session into ${up.folder}:`,
      ...(up.saved || []).map((f) => `- ${f.path} (${(f.size / 1024).toFixed(0)} KB)`)]
    if (failed.length) lines.push('Not downloaded:', ...failed.map((f) => `- ${f}`))
    return { content: lines.join('\n') }
  },

  async save_page(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const { data } = await withTimeout(cdp(tab.id, 'Page.captureSnapshot', { format: 'mhtml' }), 45000, 'Saving the page took too long.')
    const base = (oneLine(tab.title) || hostOf(tab.url) || 'page').replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 80)
    const up = await uploadFiles(ctx, [{ blob: new Blob([data], { type: 'multipart/related' }), name: `${base}.mhtml` }], { url: tab.url, title: tab.title })
    return { content: `Saved the whole page (${tab.url}) as ${(up.saved || [])[0]?.path || up.folder} — it opens offline in a browser.` }
  },

  async form_input(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    if (!args.ref) throw new Refusal('form_input needs a ref (from read_page or find) and a value.')
    return { content: await helper(tab.id, 'formInput', { ref: args.ref, value: args.value ?? args.text ?? '' }) }
  },

  click: (ctx) => clickLike(ctx, ctx.args),
  double_click: (ctx) => clickLike(ctx, ctx.args, { count: 2, verb: 'Double-clicked' }),
  triple_click: (ctx) => clickLike(ctx, ctx.args, { count: 3, verb: 'Triple-clicked' }),
  right_click: (ctx) => clickLike(ctx, ctx.args, { button: 'right', verb: 'Right-clicked' }),

  async hover(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const p = await pointFor(tab, args)
    await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x: p.x, y: p.y })
    await sleep(250)
    return { content: [`The pointer is on ${p.what}${p.ref ? ` [${p.ref}]` : ''}.`, ...pointNotes(p)].join('\n') }
  },

  async drag(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    if (!Array.isArray(args.start_coordinate)) throw new Refusal('drag needs start_coordinate and coordinate (or ref) for where to drop.')
    const scale = attached.get(tab.id)?.shotScale || 1
    const [sx, sy] = args.start_coordinate.map((v) => Number(v) / scale)
    const end = await pointFor(tab, args)
    await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x: sx, y: sy })
    await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mousePressed', x: sx, y: sy, button: 'left', buttons: 1, clickCount: 1 })
    for (let i = 1; i <= 12; i++) {
      await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x: sx + ((end.x - sx) * i) / 12, y: sy + ((end.y - sy) * i) / 12, button: 'left', buttons: 1 })
      await sleep(16)
    }
    await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x: end.x, y: end.y, button: 'left', buttons: 0, clickCount: 1 })
    return { content: `Dragged from (${args.start_coordinate.join(', ')}) to ${end.what}.` }
  },

  async type(ctx, args = ctx.args) {
    const text = String(args.text ?? '')
    if (!text) throw new Refusal('type needs the text to type.')
    const tab = await useTab(ctx, args)
    let focused = ''
    if (args.ref || args.selector || Array.isArray(args.coordinate)) {
      const p = await pointFor(tab, { ref: args.ref, selector: args.selector, coordinate: args.coordinate, nth: args.nth })
      await mouseClick(tab.id, p.x, p.y)
      focused = ` into ${p.what}${p.ref ? ` [${p.ref}]` : ''}`
    }
    await typeText(tab.id, text)
    const after = await settle(ctx, tab, tab.url)
    return { content: [`Typed ${text.length} character${text.length === 1 ? '' : 's'}${focused}.`, after].filter(Boolean).join('\n') }
  },

  async press_key(ctx, args = ctx.args) {
    const keys = String(args.key || args.text || '').trim()
    if (!keys) throw new Refusal('press_key needs key (Enter, Tab, Escape, cmd+a, … — several separated by spaces).')
    const tab = await useTab(ctx, args)
    const before = await groupTabIds(ctx.agent)
    for (const combo of keys.split(/\s+/)) await pressCombo(tab.id, combo)
    const after = await settle(ctx, tab, tab.url, before)
    return { content: [`Pressed ${keys}.`, after].filter(Boolean).join('\n') }
  },

  async scroll(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const dir = String(args.scroll_direction || 'down')
    const amount = Math.min(Math.max(Number(args.scroll_amount) || 500, 10), 20000)
    let x
    let y
    if (Array.isArray(args.coordinate)) {
      const scale = attached.get(tab.id)?.shotScale || 1
      x = Number(args.coordinate[0]) / scale
      y = Number(args.coordinate[1]) / scale
    } else if (args.ref) {
      const p = await helper(tab.id, 'point', { ref: args.ref })
      x = p.x; y = p.y
    } else {
      const inf = await helper(tab.id, 'info')
      x = inf.vw / 2; y = inf.vh / 2
    }
    const dx = dir === 'left' ? -amount : dir === 'right' ? amount : 0
    const dy = dir === 'up' ? -amount : dir === 'down' ? amount : 0
    await cdp(tab.id, 'Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: dx, deltaY: dy })
    await sleep(400)
    const inf = await helper(tab.id, 'info')
    return { content: `Scrolled ${dir} ${amount} px. The page is at ${inf.sy} of ${inf.sh} px${inf.sx ? `, x ${inf.sx}` : ''}.` }
  },

  async scroll_to(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    return { content: await helper(tab.id, 'scrollTo', { ref: args.ref }) }
  },

  async screenshot(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const s = await shoot(tab.id)
    const t = await chrome.tabs.get(tab.id)
    return { image: s.image, image_type: 'image/jpeg',
      content: `Screenshot of tab ${t.id}: ${clip(t.title || '', 120)} — ${t.url}\n${s.width}×${s.height} px; a coordinate you give (click, scroll…) is a point in this picture. Scrolled to ${s.info.sy} of ${s.info.sh} px.`,
      data: { width: s.width, height: s.height, url: t.url } }
  },

  async zoom(ctx, args = ctx.args) {
    if (!Array.isArray(args.region) || args.region.length !== 4) throw new Refusal('zoom needs region: [x0, y0, x1, y1] in a screenshot.')
    const tab = await useTab(ctx, args)
    const s = await shoot(tab.id, args.region)
    return { image: s.image, image_type: 'image/jpeg',
      content: `Closer look at [${args.region.join(', ')}] (${s.width}×${s.height} px). Coordinates for clicks still come from the full screenshot.` }
  },

  async javascript(ctx, args = ctx.args) {
    const code = String(args.code || '').trim()
    if (!code) throw new Refusal('javascript needs code (its last expression is the answer; await works).')
    const tab = await useTab(ctx, args)
    let res
    try {
      res = await evaluate(tab.id, code, { main: true, repl: true, timeout: 30000 })
    } catch (e) {
      throw new Error(`The script failed: ${cleanError(e)}`)
    }
    let out
    if (res.type === 'undefined') out = 'undefined'
    else if (res.value !== undefined) out = typeof res.value === 'string' ? res.value : JSON.stringify(res.value, null, 1)
    else out = res.description || res.type
    return { content: clip(String(out), 30000) }
  },

  async console(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const t = attached.get(tab.id)
    let items = t.console.slice()
    if (args.only_errors) items = items.filter((m) => m.level === 'error')
    if (args.pattern) {
      let re = null
      try { re = new RegExp(args.pattern, 'i') } catch { re = null }
      items = items.filter((m) => (re ? re.test(m.text) : m.text.toLowerCase().includes(String(args.pattern).toLowerCase())))
    }
    if (args.clear) t.console = []
    const since = new Date(t.since).toLocaleTimeString()
    if (!items.length) return { content: `No console messages${args.only_errors ? ' that are errors' : ''}${args.pattern ? ` matching ${q(args.pattern)}` : ''} since ${since} (when you started on this tab — reload it to catch what the page logs while loading).` }
    return { content: `Console since ${since}:\n${items.slice(-100).map((m) => `[${m.level}] ${m.text}${m.where ? `  (${clip(m.where, 120)})` : ''}`).join('\n')}` }
  },

  async network_requests(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const t = attached.get(tab.id)
    let items = [...t.net.values()]
    if (args.url_pattern) items = items.filter((r) => r.url.includes(String(args.url_pattern)))
    if (args.clear) t.net.clear()
    const since = new Date(t.since).toLocaleTimeString()
    if (!items.length) return { content: `No requests${args.url_pattern ? ` with ${q(args.url_pattern)} in the address` : ''} since ${since} (when you started on this tab).` }
    const lines = items.slice(-150).map((r) => `${r.method} ${r.failed ? r.failed.toUpperCase() : r.status || '…'} ${clip(r.url, 200)}${r.type ? ` (${r.type.toLowerCase()}${r.size ? `, ${(r.size / 1024).toFixed(1)} KB` : ''})` : ''}`)
    return { content: `Requests since ${since}${items.length > 150 ? ` (last 150 of ${items.length})` : ''}:\n${lines.join('\n')}` }
  },

  async resize(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args)
    const width = Math.round(Number(args.width))
    const height = Math.round(Number(args.height))
    if (!(width >= 300 && height >= 200)) throw new Refusal('resize needs width and height (at least 300×200).')
    await chrome.windows.update(tab.windowId, { state: 'normal', width, height })
    const inf = await helper(tab.id, 'info')
    return { content: `The window is now ${width}×${height}; the page sees ${inf.vw}×${inf.vh} CSS px.` }
  },

  async upload_file(ctx, args = ctx.args) {
    const f = args.file || {}
    if (!args.ref || !f.data) throw new Refusal('upload_file needs ref (the file field) and the file.')
    const tab = await useTab(ctx, args)
    return { content: await helper(tab.id, 'upload', { ref: args.ref, name: String(f.name || 'file'), type: String(f.type || 'application/octet-stream'), data: String(f.data) }) }
  },

  async release(ctx) {
    const g = await agentGroup(ctx.agent)
    const tabs = await groupTabs(g)
    for (const t of tabs) await detach(t.id)
    if (g) { g.current = null; saveGroups() }
    return { content: `Let go of the person's Chrome.${tabs.length ? ` Your ${tabs.length} tab${tabs.length === 1 ? ' stays' : 's stay'} open in the “${ctx.agent.name}” group for them.` : ''}` }
  },
}

// Mark the agent as recently at work (a Cancel on Chrome's debugging bar stops it).
for (const [name, fn] of Object.entries(ACTIONS)) {
  ACTIONS[name] = async (ctx) => {
    const g = groups[ctx.agent.slug]
    const out = await fn(ctx)
    const after = groups[ctx.agent.slug] || g
    if (after) after.recentAt = Date.now()
    return out
  }
}
