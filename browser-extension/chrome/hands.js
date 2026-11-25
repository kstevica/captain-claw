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
//
// A browser connected to several Flight Decks keeps a socket to each, and
// agents on every one of them can use it. An agent is known by its deck and
// slug together (two decks may each have a "researcher"): its tab group, a
// Stop in the panel and the tab to go back to are kept that way.
//
// The same socket carries the agents' web searches (search.js) and the pages
// read here for their web_fetch (pages.js) when the person turned those on —
// then it's open even with "Agents can use this browser" off, and Flight Deck
// sends no acts (the hello says what's lent).
//
// Chrome keeps the debugger off a tab that holds a frame of another extension
// (a helper button some extension put into the page). There the acts that can
// go without it run plain — the page helpers through chrome.scripting (which
// needs "Work on every site"), navigation through chrome.tabs, screenshots of
// the tab in front — and the rest say what's in the way (lib/hands-errors.js).

import { browserLabel, loadDecks, onDeck } from './lib/api.js'
import { saveSearchPick, servedEngines } from './lib/engines.js'
import { cleanError, LOST_DEBUGGER, needsDebugger, noScripting, OTHER_EXTENSION, plainNote } from './lib/hands-errors.js'
import { scrollThrough, settleInPage } from './lib/settle.js'
import { readsPages, runFetch, saveFetchMode } from './pages.js'
import { runSearch } from './search.js'

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

// One link per Flight Deck this browser is connected to: { conn, ws, lent,
// retry, retryTimer, waiting, pingTimer } — lent is what the socket told that
// deck this browser lends ({ hands, search: [engine ids], fetch }).
const links = new Map()  // deck address → link
let decks = []          // every deck this browser is connected to (lib/api.js)
let statuses = {}       // deck address → { state, error, at }, for the panel
let prefs = { on: false, front: false }
let stopped = new Set() // agents the person stopped: "<deck>|<slug>" (a bare slug, kept from before: on every deck)
let groups = {}         // agent key ("<deck>|<slug>") → { groupId, windowId, name, title, current }
let log = []
let loaded = null
const attached = new Map()   // tabId → { lastUse, world, console: [], net: Map, shotScale, since }
// Tabs Chrome keeps the debugger off (another extension's frame is in the
// page) → { ext: [its extension ids], scriptable: chrome.scripting may run there }.
const plainTabs = new Map()
const cancelled = new Set()
// Where to go back when an agent's run is over: window id → the tab the person
// was on when one of that agent's tabs took the front ({ tabId, key }).
let returnTo = new Map()
let actsRunning = 0
const openedFrom = new Map() // tabId → [new tab ids opened from it]

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const oneLine = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim()
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
const q = (s) => JSON.stringify(String(s))

async function loadState() {
  if (!loaded) {
    loaded = (async () => {
      const local = await chrome.storage.local.get(['hands', 'handsStopped']).catch(() => ({}))
      const session = await chrome.storage.session.get(['handsGroups', 'handsLog', 'handsReturn']).catch(() => ({}))
      prefs = { on: false, front: false, ...(local.hands || {}) }
      stopped = new Set(Array.isArray(local.handsStopped) ? local.handsStopped : [])
      groups = session.handsGroups && typeof session.handsGroups === 'object' ? session.handsGroups : {}
      returnTo = new Map(Array.isArray(session.handsReturn) ? session.handsReturn : [])
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

function setStatus(deck, state, error = '') {
  statuses[deck] = { state, error, at: Date.now() }
  saveStatuses()
}

function saveStatuses() {
  chrome.storage.session.set({ handsStatus: statuses }).catch(() => {})
}

// ── agents, by deck ──

const agentKey = (deck, slug) => `${deck}|${slug}`

/** "glasses" for glasses.example.com; a deck on this computer or an address keeps its host. */
function deckShort(deck) {
  try {
    const u = new URL(deck)
    return u.hostname === 'localhost' || /^[\d.]+$/.test(u.hostname) || !u.hostname.includes('.') ? u.host : u.hostname.split('.')[0]
  } catch { return deck }
}

// With more than one deck connected, an agent's tab group says which deck it's from.
const groupTitle = (name, deck) => (decks.length > 1 ? `${name} · ${deckShort(deck)}` : name)

function agentOf(msg, link) {
  const deck = link.conn.url
  const slug = String(msg.agent?.slug || '')
  const name = oneLine(msg.agent?.name || slug || 'Agent').slice(0, 40)
  return { slug, name, deck, key: agentKey(deck, slug), title: groupTitle(name, deck) }
}

const isStopped = (agent) => stopped.has(agent.key) || stopped.has(agent.slug)

/** The slugs of *deck*'s agents the person stopped, as that deck knows them. */
function stoppedOn(deck) {
  const out = new Set()
  for (const s of stopped) {
    const bar = s.lastIndexOf('|')
    if (bar < 0) out.add(s)
    else if (s.slice(0, bar) === deck) out.add(s.slice(bar + 1))
  }
  return [...out]
}

// Flight Deck's own pages are off-limits — every connected deck's, and the
// one an act came from (an agent's own headless browser has just that one).
const offLimits = (url, conn) => onDeck(url, conn ? [...decks, conn] : decks)

// ── the sockets, one per deck ──

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

const live = (ws) => !!ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
const sameLent = (a, b) => a.hands === b.hands && a.search.join() === b.search.join() && a.fetch === b.fetch

async function ensureConnected() {
  await loadState()
  // Read fresh: a change can land while the first read is still on its way.
  const got = await chrome.storage.local.get(['hands', 'handsStopped']).catch(() => ({}))
  prefs = { on: false, front: false, ...(got.hands || {}) }
  stopped = new Set(Array.isArray(got.handsStopped) ? got.handsStopped : [])
  decks = (await loadDecks().catch(() => null))?.decks || []
  const hands = prefs.on && permitted()
  if ((!hands || !decks.length) && attached.size) detachAll()
  // A deck disconnected, or connected again with a new token: its socket goes.
  for (const [url, link] of links) {
    if (!decks.some((d) => d.url === url && d.token === link.conn.token)) dropLink(url)
  }
  const before = Object.keys(statuses).length
  statuses = Object.fromEntries(Object.entries(statuses).filter(([url]) => decks.some((d) => d.url === url)))
  if (Object.keys(statuses).length !== before) saveStatuses()
  for (const conn of decks) {
    const want = { hands, search: await servedEngines(conn.url), fetch: await readsPages(conn.url) }
    const link = links.get(conn.url)
    if (!want.hands && !want.search.length && !want.fetch) {
      if (link) dropLink(conn.url)
      if (statuses[conn.url]?.state !== (prefs.on ? 'permission' : 'off')) setStatus(conn.url, prefs.on ? 'permission' : 'off')
      continue
    }
    if (link && live(link.ws)) {
      if (!sameLent(want, link.lent)) {
        link.lent = want
        send(link, { type: 'prefs', ...want })
        setStatus(conn.url, link.ws.readyState === WebSocket.OPEN ? (want.hands ? 'on' : 'search') : 'connecting')
      }
      continue
    }
    // Waiting to try again after the deck went away: its own timer does that.
    if (link?.waiting) { link.lent = want; continue }
    connect(conn, want)
  }
}

function dropLink(url) {
  const link = links.get(url)
  if (!link) return
  links.delete(url)
  clearTimeout(link.retryTimer)
  clearInterval(link.pingTimer)
  if (link.ws) {
    const old = link.ws
    link.ws = null
    old.onclose = null
    try { old.close(1000) } catch { /* already gone */ }
  }
}

function connect(conn, lent) {
  const retry = links.get(conn.url)?.retry || 0
  dropLink(conn.url)
  const link = { conn, ws: null, lent, retry, retryTimer: 0, waiting: false, pingTimer: 0 }
  links.set(conn.url, link)
  let sock
  try {
    sock = new WebSocket(conn.url.replace(/^http/i, 'ws') + '/fd/clip/hands/ws')
  } catch (e) {
    setStatus(conn.url, 'error', String(e.message || e))
    return
  }
  link.ws = sock
  setStatus(conn.url, 'connecting')
  sock.onopen = () => {
    // An agent's own (headless) browser is named after the agent by Flight Deck.
    sock.send(JSON.stringify({ type: 'hello', token: conn.token, browser: conn.agent ? `Headless · ${conn.agent}` : browserLabel(),
                               version: VERSION, stopped: stoppedOn(conn.url), hands: link.lent.hands, search: link.lent.search,
                               fetch: link.lent.fetch }))
    clearInterval(link.pingTimer)
    link.pingTimer = setInterval(() => send(link, { type: 'ping' }), PING_MS)
  }
  sock.onmessage = (ev) => {
    let msg
    try { msg = JSON.parse(ev.data) } catch { return }
    if (msg.type === 'welcome') {
      link.retry = 0
      setStatus(conn.url, link.lent.hands ? 'on' : 'search')
      if ('search_pick' in msg) saveSearchPick(conn.url, String(msg.search_pick || ''))
      if ('fetch_mode' in msg) saveFetchMode(conn.url, String(msg.fetch_mode || ''))
    }
    else if (msg.type === 'search_pick') saveSearchPick(conn.url, String(msg.engine || ''))
    else if (msg.type === 'fetch_mode') saveFetchMode(conn.url, String(msg.mode || ''))
    else if (msg.type === 'fetch') fetchFor(msg, link)
    else if (msg.type === 'denied') setStatus(conn.url, 'denied', msg.error || '')
    else if (msg.type === 'act') runAct(msg, link)
    else if (msg.type === 'search') searchFor(msg, link)
    else if (msg.type === 'cancel') cancelled.add(String(msg.id || ''))
    else if (msg.type === 'turn_over') turnOver(agentKey(conn.url, String(msg.agent?.slug || '')))
  }
  sock.onclose = (ev) => {
    if (link.ws !== sock) return
    link.ws = null
    clearInterval(link.pingTimer)
    if (links.get(conn.url) !== link) return
    if (ev.code === 4401) { setStatus(conn.url, 'denied', 'This browser was disconnected from Flight Deck.'); return }
    setStatus(conn.url, 'connecting', "Can't reach Flight Deck — trying again.")
    const wait = [1000, 2000, 5000, 10000, 20000, 30000][Math.min(link.retry++, 5)]
    clearTimeout(link.retryTimer)
    link.waiting = true
    link.retryTimer = setTimeout(() => { link.waiting = false; ensureConnected() }, wait)
  }
}

function send(link, msg) {
  if (link?.ws && link.ws.readyState === WebSocket.OPEN) {
    try { link.ws.send(JSON.stringify(msg)) } catch { /* closing */ }
  }
}

chrome.alarms?.onAlarm.addListener((a) => { if (a.name === 'captain-hands') ensureConnected() })
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return
  if (changes.search || changes.searchPicks || changes.fetchModes || changes.fetchPrefs) ensureConnected()
  if (changes.hands) {
    prefs = { on: false, front: false, ...(changes.hands.newValue || {}) }
    ensureConnected()
  }
  if (changes.handsStopped) {
    stopped = new Set(changes.handsStopped.newValue || [])
    for (const link of links.values()) send(link, { type: 'stopped', agents: stoppedOn(link.conn.url) })
  }
  // A deck connected or disconnected (decks), or seeded the old way (conn).
  if (changes.decks || changes.conn) ensureConnected()
})
chrome.permissions?.onAdded.addListener(() => ensureConnected())
chrome.permissions?.onRemoved.addListener(() => ensureConnected())
chrome.windows.onFocusChanged.addListener((id) => {
  if (id !== chrome.windows.WINDOW_ID_NONE) for (const link of links.values()) send(link, { type: 'focus' })
})

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

// What the panel's log keeps of an act: by deck and agent.
const logEntry = (id, agent, action, text) => ({ id, at: Date.now(), deck: agent.deck, key: agent.key, slug: agent.slug, name: agent.name,
                                                 action, text, state: 'working' })

async function runAct(msg, link) {
  await loadState()
  const id = String(msg.id || '')
  const agent = agentOf(msg, link)
  const action = String(msg.action || '')
  const args = msg.args && typeof msg.args === 'object' ? msg.args : {}
  const entry = logEntry(id, agent, action, describeAct(action, args))
  const quiet = action === 'status' || action === 'release'
  if (!quiet) { log.push(entry); if (log.length > LOG_MAX) log = log.slice(-LOG_MAX); saveLog() }
  let result
  const before = await activeTabs()
  actsRunning++
  try {
    if (!prefs.on) throw new Refusal("This browser isn't lent to agents (“Agents can use this browser” is off in Captain's settings).")
    if (isStopped(agent)) throw new Refusal("The person stopped you using their browser (Stop in the Captain panel). They let you go on with “Let it again” there — ask them to, and don't try again until they say so.")
    const fn = ACTIONS[action]
    if (!fn) throw new Refusal(`“${action}” isn't something this browser does (Captain for Chrome ${VERSION}).`)
    const ctx = { id, agent, args, conn: link.conn, action }
    try {
      result = await fn(ctx)
    } catch (e) {
      // Chrome cut the debugger off mid-way (another extension's frame just
      // appeared): an act that only reads goes again — plain, if it must.
      if (e instanceof Refusal || !AGAIN_OK.has(action) || !lostDebugger(e)) throw e
      result = await fn(ctx)
    }
    result = { ok: true, ...result }
    entry.state = 'done'
  } catch (e) {
    result = { ok: false, error: cleanError(e) }
    entry.state = 'failed'
    entry.error = clip(result.error, 160)
  }
  cancelled.delete(id)
  try { await noteFront(agent, before) } catch { /* a tab went away meanwhile */ }
  actsRunning--
  if (!quiet) saveLog()
  send(link, { type: 'result', id, ...result })
}

// ── a robot check handed to the person (lib/robot-check.js) ──

// Flight Deck hears it (the agent waits longer, its chat says why), and the
// panel's log shows it while it lasts.
function tellerFor(id, agent, link) {
  let entry = null
  return (m) => {
    send(link, { ...m, id })
    if (m.joined) return
    if (!('over' in m)) {
      entry = logEntry(`${id}:check`, agent, 'robot_check', `Waiting for you: answer ${m.site || 'the site'}'s robot check in the tab in front`)
      log.push(entry); if (log.length > LOG_MAX) log = log.slice(-LOG_MAX); saveLog()
    } else if (entry) {
      const solved = m.over === 'solved'
      entry.state = solved ? 'done' : 'failed'
      entry.text = solved ? `You answered ${m.site || 'the site'}'s robot check — going on` : `${m.site || 'The site'}'s robot check wasn't answered`
      if (!solved) entry.error = { timeout: 'not in time — the tab is still open', closed: 'the tab was closed', cancelled: 'the agent stopped waiting — the tab is still open' }[m.over] || m.over
      saveLog()
    }
  }
}

// ── a web search (search.js) ──

async function searchFor(msg, link) {
  const id = String(msg.id || '')
  const result = await runSearch(msg, (x) => cancelled.has(x), tellerFor(id, agentOf(msg, link), link), link.conn.url)
  cancelled.delete(id)
  send(link, { type: 'result', id, ...result })
}

// ── a page read for an agent's web_fetch (pages.js) ──

async function fetchFor(msg, link) {
  const id = String(msg.id || '')
  const agent = agentOf(msg, link)
  const entry = logEntry(id, agent, 'fetch', `Read ${hostOf(String(msg.url || '')) || 'a page'} in the background`)
  log.push(entry); if (log.length > LOG_MAX) log = log.slice(-LOG_MAX); saveLog()
  const result = await runFetch(msg, (x) => cancelled.has(x), tellerFor(id, agent, link), link.conn.url)
  cancelled.delete(id)
  entry.state = result.ok ? 'done' : 'failed'
  if (!result.ok) entry.error = clip(String(result.error || ''), 160)
  saveLog()
  send(link, { type: 'result', id, ...result })
}

// ── going back to the person's tab ──

async function activeTabs() {
  try {
    const tabs = await chrome.tabs.query({ active: true })
    return new Map(tabs.map((t) => [t.windowId, t]))
  } catch { return new Map() }
}

function saveReturn() {
  chrome.storage.session.set({ handsReturn: [...returnTo] }).catch(() => {})
}

// After an act: one of the agent's tabs took the front where the person was on
// a tab of their own → remember that tab (once per run).
async function noteFront(agent, before) {
  const g = groups[agent.key]
  if (!g) return
  const after = await activeTabs()
  for (const [windowId, tab] of after) {
    const was = before.get(windowId)
    if (tab.groupId !== g.groupId || !was || was.id === tab.id || was.groupId === g.groupId) continue
    if (returnTo.has(windowId)) continue
    returnTo.set(windowId, { tabId: was.id, key: agent.key })
    saveReturn()
  }
}

// The agent's run is over: where its tab is still in front, the person's comes back.
async function turnOver(key) {
  await loadState()
  const g = await agentGroup({ key })
  let changed = false
  for (const [windowId, r] of [...returnTo]) {
    if (r.key !== key) continue
    returnTo.delete(windowId)
    changed = true
    const [cur] = await chrome.tabs.query({ active: true, windowId }).catch(() => [])
    if (!cur || !g || cur.groupId !== g.groupId) continue   // they moved on: leave it
    const back = await chrome.tabs.get(r.tabId).catch(() => null)
    if (back && back.windowId === windowId) await chrome.tabs.update(back.id, { active: true }).catch(() => {})
  }
  if (changed) saveReturn()
}

// A tab the person picks themselves (no act running) is their choice: no going back.
chrome.tabs.onActivated.addListener(({ windowId }) => {
  if (actsRunning > 0 || !returnTo.has(windowId)) return
  returnTo.delete(windowId)
  saveReturn()
})

// Acts safe to run twice: when Chrome cuts the debugger off mid-way they go again.
const AGAIN_OK = new Set(['navigate', 'read_page', 'find', 'page_text', 'page_files', 'screenshot', 'zoom', 'save_page', 'scroll_to'])

function lostDebugger(e) {
  const s = String(e?.message || e || '')
  return LOST_DEBUGGER.test(s) || OTHER_EXTENSION.test(s)
}

// ── the agent's tabs ──

function colorFor(slug) {
  let h = 0
  for (const ch of slug) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return GROUP_COLORS[h % GROUP_COLORS.length]
}

async function agentGroup(agent) {
  const g = groups[agent.key]
  if (!g || g.groupId == null) return null
  try {
    const tg = await chrome.tabGroups.get(g.groupId)
    g.windowId = tg.windowId
    if (agent.title && tg.title !== agent.title) {
      chrome.tabGroups.update(g.groupId, { title: agent.title }).catch(() => {})
      g.title = agent.title
    }
    return g
  } catch {
    delete groups[agent.key]
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
    await chrome.tabGroups.update(groupId, { title: agent.title, color: colorFor(agent.key) }).catch(() => {})
    g = { groupId, windowId, name: agent.name, title: agent.title }
    groups[agent.key] = g
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
      throw new Refusal(`Tab ${id} isn't one of yours. You can only use the tabs in your own tab group “${agent.title}” — the person can drag a tab into it to share it with you.`)
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
    if (url && offLimits(url, conn)) throw new Refusal("Flight Deck's own pages are off-limits in the person's browser.")
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
  if (offLimits(u.href, conn)) throw new Refusal("Flight Deck's own pages are off-limits in the person's browser.")
  return u.href
}

// The agent's tab with the debugger on it. Where Chrome keeps the debugger off
// (another extension's frame is in the page) the tab comes back marked plain
// for an act that can go without it — opts.plain: 'script' (needs
// chrome.scripting there) or 'tabs' (chrome.tabs is enough) — and any other
// act is refused, saying why (opts.need names it, else the act).
async function useTab(ctx, args, opts = {}) {
  const tab = await tabFor(ctx.agent, args, opts)
  guardTab(tab, ctx.conn)
  let plain = null
  try {
    await attach(tab.id)
    plainTabs.delete(tab.id)
  } catch (e) {
    if (!OTHER_EXTENSION.test(String(e?.message || e))) throw e
    plain = await goPlain(tab.id)
    if (!opts.plain) throw new Refusal(needsDebugger(opts.need || ctx.action || 'that', plain))
    if (opts.plain === 'script' && !plain.scriptable) throw new Refusal(noScripting(plain))
  }
  if (prefs.front && !tab.active) await chrome.tabs.update(tab.id, { active: true }).catch(() => {})
  const now = await chrome.tabs.get(tab.id)
  if (plain) now.plain = plain
  return now
}

// The tab holds another extension's frame: from now on its page helpers run
// through chrome.scripting (while the person lets Captain work on every site).
async function goPlain(tabId) {
  attached.delete(tabId)
  const page = { ext: [], scriptable: true, shotScale: plainTabs.get(tabId)?.shotScale || 1 }
  try {
    const [r] = await chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, func: extensionFrames })
    page.ext = (Array.isArray(r?.result) ? r.result : []).filter((id) => id !== chrome.runtime.id)
  } catch {
    page.scriptable = false
  }
  plainTabs.set(tabId, page)
  return page
}

// Runs in the page: the ids of extensions whose frames are in it (in shadow roots too).
function extensionFrames() {
  const ids = new Set()
  const look = (root, depth) => {
    for (const f of root.querySelectorAll('iframe, frame, embed, object')) {
      const m = /^chrome-extension:\/\/([a-p]{32})\//.exec(f.src || f.data || '')
      if (m) ids.add(m[1])
    }
    if (depth >= 3) return
    for (const el of root.querySelectorAll('*')) {
      const shadow = globalThis.chrome?.dom?.openOrClosedShadowRoot?.(el) || el.shadowRoot
      if (shadow) look(shadow, depth + 1)
    }
  }
  look(document, 0)
  return [...ids].slice(0, 3)
}

// After a page loaded in the agent's tab: '' while the debugger works there,
// else a note on what works without it.
async function debuggerNote(tabId) {
  try {
    await attach(tabId)
    plainTabs.delete(tabId)
    return ''
  } catch (e) {
    if (!OTHER_EXTENSION.test(String(e?.message || e))) return ''
  }
  return `\n${plainNote(await goPlain(tabId))}`
}

chrome.tabs.onCreated.addListener((tab) => {
  if (tab.openerTabId !== undefined && (attached.has(tab.openerTabId) || plainTabs.has(tab.openerTabId))) {
    const list = openedFrom.get(tab.openerTabId) || []
    list.push(tab.id)
    openedFrom.set(tab.openerTabId, list.slice(-10))
  }
})
chrome.tabs.onRemoved.addListener((tabId) => {
  attached.delete(tabId)
  plainTabs.delete(tabId)
  openedFrom.delete(tabId)
})

// ── the debugger ──

function cdp(tabId, method, params = {}) {
  return chrome.debugger.sendCommand({ tabId }, method, params)
}

// A command to one of the tab's frames from another site (its own debugger
// session, Chrome 125+), or to the tab itself when *sid* is ''.
function cdpS(tabId, sid, method, params = {}) {
  return chrome.debugger.sendCommand(sid ? { tabId, sessionId: sid } : { tabId }, method, params)
}

async function attach(tabId) {
  const t = attached.get(tabId)
  if (t) { t.lastUse = Date.now(); return t }
  await chrome.debugger.attach({ tabId }, '1.3')
  const state = { lastUse: Date.now(), since: Date.now(), world: null, console: [], net: new Map(), shotScale: 1, loads: [],
    frames: new Map(), frameIx: 0 }
  attached.set(tabId, state)
  await Promise.all([
    cdp(tabId, 'Page.enable'),
    cdp(tabId, 'Runtime.enable'),
    cdp(tabId, 'Log.enable').catch(() => {}),
    cdp(tabId, 'Network.enable', { maxPostDataSize: 0 }).catch(() => {}),
    cdp(tabId, 'DOM.enable').catch(() => {}),
    // Frames from other sites (a sign-up's username box, a card field) run in
    // processes of their own: each gets a session, so its fields can be read too.
    cdp(tabId, 'Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }).catch(() => {}),
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

chrome.debugger?.onDetach.addListener((src) => {
  // Cancel on Chrome's "is debugging this browser" bar only hides it: the next
  // act attaches again. Stopping an agent is the panel's Stop (people press
  // Cancel to get rid of the bar, not to stop their agent). A page that's
  // loading for an act is waited for without the debugger from here.
  if (src.tabId === undefined) return
  const t = attached.get(src.tabId)
  attached.delete(src.tabId)
  if (t) for (const done of t.loads.splice(0)) done('detached')
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
  if (src.sessionId) {
    // An event of a frame from another site.
    const f = t.frames.get(src.sessionId)
    if (!f) return
    if ((method === 'Page.frameNavigated' && !p.frame?.parentId) || method === 'Runtime.executionContextsCleared' ||
        (method === 'Runtime.executionContextDestroyed' && f.world === p.executionContextId)) f.world = null
    if (method === 'Runtime.consoleAPICalled' || method === 'Runtime.exceptionThrown') {
      const text = method === 'Runtime.exceptionThrown'
        ? (p.exceptionDetails?.exception?.description || p.exceptionDetails?.text || 'Uncaught error')
        : (p.args || []).map(remoteText).join(' ')
      pushCapped(t.console, { at: Date.now(), level: method === 'Runtime.exceptionThrown' ? 'error' : p.type, text: clip(text, 2000), where: f.url })
    }
    return
  }
  switch (method) {
    case 'Target.attachedToTarget': {
      const info = p.targetInfo || {}
      if (info.type === 'iframe' && p.sessionId) {
        t.frames.set(p.sessionId, { index: ++t.frameIx, targetId: info.targetId, url: info.url || '', world: null })
        cdpS(src.tabId, p.sessionId, 'Page.enable').catch(() => {})
        cdpS(src.tabId, p.sessionId, 'Runtime.enable').catch(() => {})
      }
      break
    }
    case 'Target.detachedFromTarget':
      t.frames.delete(p.sessionId)
      break
    case 'Target.targetInfoChanged':
      for (const f of t.frames.values()) if (f.targetId === p.targetInfo?.targetId) f.url = p.targetInfo.url || f.url
      break
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
      if (r) { r.size = p.encodedDataLength || 0; r.done = true }
      break
    }
    case 'Network.loadingFailed': {
      const r = t.net.get(p.requestId)
      if (r) { r.failed = p.canceled ? 'canceled' : (p.errorText || 'failed'); r.done = true }
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

function holderOf(tabId, sid) {
  const t = attached.get(tabId)
  const h = sid ? t?.frames.get(sid) : t
  if (!h) throw new Refusal("That frame is gone (the page changed) — read the page again.")
  return h
}

async function world(tabId, sid = '') {
  if (!attached.get(tabId)) await attach(tabId)
  const h = holderOf(tabId, sid)
  if (h.world) return h.world
  const { frameTree } = await cdpS(tabId, sid, 'Page.getFrameTree')
  const { executionContextId } = await cdpS(tabId, sid, 'Page.createIsolatedWorld', { frameId: frameTree.frame.id, worldName: 'captain-hands' })
  h.world = executionContextId
  return h.world
}

async function evaluate(tabId, expression, { main = false, timeout = 20000, repl = false, sid = '' } = {}) {
  const params = { expression, returnByValue: true, awaitPromise: true, userGesture: true }
  if (repl) params.replMode = true
  if (!main) params.contextId = await world(tabId, sid)
  let r
  try {
    r = await withTimeout(cdpS(tabId, sid, 'Runtime.evaluate', params), timeout, 'The page took too long to answer.')
  } catch (e) {
    if (main || !/context/i.test(String(e.message || e))) throw e
    holderOf(tabId, sid).world = null
    params.contextId = await world(tabId, sid)
    r = await withTimeout(cdpS(tabId, sid, 'Runtime.evaluate', params), timeout, 'The page took too long to answer.')
  }
  if (r.exceptionDetails) {
    const d = r.exceptionDetails
    throw new Error(d.exception?.description || d.exception?.value || d.text || 'The page script failed.')
  }
  return r.result
}

async function helper(tabId, fn, arg = {}, sid = '') {
  if (plainTabs.has(tabId) && !sid) return plainHelper(tabId, fn, arg)
  if (!helperSrc) helperSrc = await source('hands-page.js')
  const prefix = sid ? `f${holderOf(tabId, sid).index}_` : ''
  const res = await evaluate(tabId, `${helperSrc}\n;globalThis.__captainRefPrefix = ${JSON.stringify(prefix)};globalThis.__captainHands.${fn}(${JSON.stringify(arg)})`, { sid })
  return res.value
}

// A page helper run through chrome.scripting, in Captain's own isolated world
// of the page (the debugger can't go there). Its refs are ref_p12 — never
// mistaken for refs read with the debugger, which live in another world.
async function plainHelper(tabId, fn, arg = {}) {
  const target = { tabId, frameIds: [0] }
  const call = async () => {
    const [res] = await chrome.scripting.executeScript({ target, args: [fn, arg], func: async (name, a) => {
      if (!globalThis.__captainHands) return { missing: true }
      globalThis.__captainRefPrefix = 'p'
      try { return { ok: true, value: await globalThis.__captainHands[name](a) } } catch (e) { return { ok: false, error: String(e?.message || e) } }
    } })
    return res?.result
  }
  let r = await call()
  if (r?.missing) {
    await chrome.scripting.executeScript({ target, files: ['hands-page.js'] })
    r = await call()
  }
  if (!r || r.missing) throw new Error('The page did not answer (it is loading or changing?) — try again.')
  if (!r.ok) throw new Error(r.error)
  return r.value
}

// The frame a ref lives in ('' = the page itself): ref_f2_5 is frame 2's.
function sidOfRef(tabId, ref) {
  const m = /^ref_f(\d+)_/.exec(String(ref || ''))
  if (!m) return ''
  const t = attached.get(tabId)
  for (const [sid, f] of t?.frames || []) if (f.index === Number(m[1])) return sid
  throw new Refusal(`${ref} was in a frame that's gone (the page changed) — read the page again.`)
}

async function refHelper(tabId, fn, arg) {
  return helper(tabId, fn, arg, sidOfRef(tabId, arg.ref))
}

// Where a frame from another site sits in the page (its top-left inside the
// border), scrolled into view first when asked.
async function frameBox(tabId, sid, { reveal = false } = {}) {
  const f = holderOf(tabId, sid)
  const { backendNodeId } = await cdp(tabId, 'DOM.getFrameOwner', { frameId: f.targetId })
  const { object } = await cdp(tabId, 'DOM.resolveNode', { backendNodeId, executionContextId: await world(tabId) })
  const r = await cdp(tabId, 'Runtime.callFunctionOn', { objectId: object.objectId, returnByValue: true,
    functionDeclaration: `function (reveal) {
      let b = this.getBoundingClientRect()
      if (reveal && (b.top < 0 || b.bottom > innerHeight)) { this.scrollIntoView({ block: 'center', behavior: 'instant' }); b = this.getBoundingClientRect() }
      return { x: b.left + (this.clientLeft || 0), y: b.top + (this.clientTop || 0), w: b.width, h: b.height }
    }`, arguments: [{ value: reveal }] })
  cdp(tabId, 'Runtime.releaseObject', { objectId: object.objectId }).catch(() => {})
  return r.result.value
}

function hostOfUrl(url) {
  try { return new URL(url).host } catch { return 'another site' }
}

// The tab's frames from other sites worth reading: shown, not a pixel.
async function liveFrames(tabId) {
  const t = attached.get(tabId)
  const out = []
  for (const [sid, f] of [...(t?.frames || [])].sort((a, b) => a[1].index - b[1].index)) {
    try {
      const box = await frameBox(tabId, sid)
      if (box.w >= 30 && box.h >= 20) out.push({ sid, f, box })
    } catch { /* not in this page's tree (nested deeper) */ }
    if (out.length >= 6) break
  }
  return out
}

// A helper run in the page and in each of its frames from other sites, joined.
async function inAllFrames(tabId, fn, arg, heading) {
  const parts = []
  for (const { sid, f, box } of await liveFrames(tabId)) {
    let text = ''
    try { text = await helper(tabId, fn, arg, sid) } catch { continue }
    if (typeof text !== 'string' || !text.includes('[ref_')) continue
    const body = text.split('\n').filter((l) => !/^(Page:|Viewport )/.test(l)).join('\n')
    parts.push(`\n${heading} ${hostOfUrl(f.url)} at (${Math.round(box.x)}, ${Math.round(box.y)}), ${Math.round(box.w)}×${Math.round(box.h)} — its refs work like the page's:\n${body}`)
  }
  return parts.join('\n')
}

// What the page and its frames say is wrong now.
async function problemsAll(tabId) {
  const out = []
  try { out.push(...(await helper(tabId, 'problems'))) } catch { /* the page is changing */ }
  for (const { sid } of await liveFrames(tabId).catch(() => [])) {
    try { out.push(...(await helper(tabId, 'problems', {}, sid))) } catch { /* gone */ }
  }
  return out
}

function newProblems(before, after) {
  const seen = new Set(before)
  const fresh = after.filter((p) => !seen.has(p))
  return fresh.length ? `On the page now: ${fresh.slice(0, 6).join(' · ')}` : ''
}

// The field that has the keyboard: in the page, or in one of its frames.
async function focusedField(tabId) {
  let f = null
  try { f = await helper(tabId, 'focused') } catch { f = null }
  if (f && !f.frame) return { ...f, sid: '' }
  for (const { sid } of await liveFrames(tabId).catch(() => [])) {
    try {
      const inner = await helper(tabId, 'focused', {}, sid)
      if (inner && !inner.frame) return { ...inner, sid }
    } catch { /* gone */ }
  }
  return null
}

function fieldSays(info) {
  if (!info) return ''
  if (info.length !== undefined) return `${info.what} now holds ${info.length} character${info.length === 1 ? '' : 's'} (a password: not shown).`
  if (info.checked !== undefined) return `${info.what} is ${info.checked ? 'checked' : 'unchecked'}.`
  return `${info.what} now reads ${JSON.stringify(info.value ?? '')}.`
}

async function capturePage(tabId) {
  let v
  if (plainTabs.has(tabId)) {
    const [res] = await withTimeout(chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, files: ['capture.js'] }),
      30000, 'The page took too long to answer.')
    v = res?.result
  } else {
    if (!captureSrc) captureSrc = await source('capture.js')
    v = (await evaluate(tabId, captureSrc, { timeout: 30000 })).value
  }
  if (!v || !v.ok) throw new Error('Could not read the page.')
  return v
}

function withTimeout(p, ms, why) {
  let t
  return Promise.race([p, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(why || 'Timed out.')), ms) })]).finally(() => clearTimeout(t))
}

// ── waiting for pages ──

// "Loaded" isn't "drawn": pages like LinkedIn draw their sections after the
// load, as their data arrives. Reads wait until the page stops changing
// (lib/settle.js) and no data request of the tab's is still on its way (the
// debugger sees those); never longer than *maxMs*, and never an error.
const DATA_TYPES = new Set(['XHR', 'Fetch', 'Script', 'Document'])
const LONG_POLL_MS = 15000    // a request open longer is a live connection, not data on its way

function dataPending(tabId) {
  const t = attached.get(tabId)
  if (!t) return 0
  const now = Date.now()
  let n = 0
  for (const r of t.net.values()) if (!r.done && DATA_TYPES.has(r.type) && now - r.at < LONG_POLL_MS) n++
  return n
}

async function waitDrawn(tabId, opts) {
  const end = Date.now() + (opts.maxMs || 8000)
  let r = null
  for (let round = 0; round < 8; round++) {
    const left = end - Date.now()
    if (left <= 0) break
    try {
      r = await inPage(tabId, settleInPage, { ...opts, maxMs: left }, left + 4000)
    } catch { return r }
    if (!dataPending(tabId)) return r
    // Data still on its way: wait for it, then for what it draws.
    while (dataPending(tabId) && Date.now() < end) await sleep(100)
    opts = { ...opts, minMs: 0 }
  }
  return r ? { ...r, quiet: false } : null
}

// Sections drawn only when scrolled to: a screen at a time down, then back.
async function scrollToLoad(tabId, opts) {
  try {
    return await inPage(tabId, scrollThrough, opts, (opts.totalMs || 15000) + 5000)
  } catch { return null }
}

// One of lib/settle.js's functions run in the page: through the debugger, or
// through chrome.scripting where the debugger can't go.
async function inPage(tabId, func, arg, timeout) {
  if (plainTabs.has(tabId)) {
    const [res] = await withTimeout(chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, func, args: [arg] }),
      timeout, 'The page took too long to answer.')
    return res?.result ?? null
  }
  return (await evaluate(tabId, `(${func})(${JSON.stringify(arg)})`, { timeout })).value || null
}

async function navigateTab(tabId, url) {
  const t = attached.get(tabId) || (await attach(tabId))
  const loaded = new Promise((resolve) => t.loads.push(resolve))
  let res
  try {
    res = await cdp(tabId, 'Page.navigate', { url })
  } catch (e) {
    if (!lostDebugger(e)) throw e
    // Cut off as the page came (another extension's frame in it): it loads on.
    const now = await chrome.tabs.get(tabId).catch(() => null)
    if (!now) throw new Refusal('The tab closed.')
    if (now.status !== 'loading' && now.url !== url && now.pendingUrl !== url) return plainNavigate(tabId, url)
    return waitComplete(tabId, NAV_WAIT_MS)
  }
  if (res.errorText && res.errorText !== 'net::ERR_ABORTED') throw new Refusal(`Couldn't open ${url}: ${res.errorText}`)
  if (!res.loaderId) return true   // same document (#hash)
  const how = await withTimeout(loaded, NAV_WAIT_MS, 'slow').catch(() => false)
  // The debugger went (the page has another extension's frame): wait the plain way.
  if (how === 'detached') return waitComplete(tabId, NAV_WAIT_MS)
  return how === true
}

// Navigation without the debugger: the tab is sent there and waited for.
async function plainNavigate(tabId, url) {
  await chrome.tabs.update(tabId, { url })
  const start = Date.now()
  let seen = false
  while (Date.now() - start < NAV_WAIT_MS) {
    await sleep(200)
    const t = await chrome.tabs.get(tabId).catch(() => null)
    if (!t) throw new Refusal('The tab closed.')
    if (t.status === 'loading') seen = true
    else if (seen || Date.now() - start > 1500) return true   // loaded — or never had to (#hash)
  }
  return false
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
      await waitDrawn(nt.id, { maxMs: 8000, quietMs: 700 })
      const fresh = await chrome.tabs.get(nt.id)
      return `It opened a new tab ${fresh.id} — now your current tab: ${clip(fresh.title || '', 120)} — ${fresh.url || fresh.pendingUrl}`
    }
  }
  const now = await chrome.tabs.get(tab.id).catch(() => null)
  if (!now) return 'The tab closed.'
  if (now.status === 'loading') await waitComplete(tab.id, SETTLE_WAIT_MS)
  // What the click started (a new route in an app, a panel opening) draws first.
  const moved = (await chrome.tabs.get(tab.id).catch(() => null))?.url !== beforeUrl
  await waitDrawn(tab.id, moved ? { maxMs: 8000, quietMs: 700 } : { maxMs: 3000, quietMs: 400 })
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

// Returns 'cut' when Chrome took the debugger off as the last release went in:
// the click landed and what it did brought another extension's frame (a
// composer opening with that extension's button in it).
async function mouseClick(tabId, x, y, { button = 'left', count = 1, mods = 0 } = {}) {
  const buttons = button === 'right' ? 2 : button === 'middle' ? 4 : 1
  await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, modifiers: mods })
  for (let i = 1; i <= count; i++) {
    await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button, buttons, clickCount: i, modifiers: mods })
    try {
      await cdp(tabId, 'Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button, buttons: 0, clickCount: i, modifiers: mods })
    } catch (e) {
      if (i === count && LOST_DEBUGGER.test(String(e?.message || e))) return 'cut'
      throw e
    }
  }
  return ''
}

const CUT_SAYS = "As it landed Chrome took Captain's debugger off this tab — another extension put a frame of its own into the " +
  'page just then. Check it took with read_page, which also says what works on this page now.'

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

// Each printable character as a US keyboard types it: [code, key code, shift].
// (A character's own code is not its key's: "." is 46, which is Delete.)
const US_KEYS = (() => {
  const m = {}
  for (const c of 'abcdefghijklmnopqrstuvwxyz') {
    const up = c.toUpperCase()
    m[c] = [`Key${up}`, up.charCodeAt(0), false]
    m[up] = [`Key${up}`, up.charCodeAt(0), true]
  }
  for (let d = 0; d <= 9; d++) m[String(d)] = [`Digit${d}`, 48 + d, false]
  const digitShift = { '!': 1, '@': 2, '#': 3, $: 4, '%': 5, '^': 6, '&': 7, '*': 8, '(': 9, ')': 0 }
  for (const [c, d] of Object.entries(digitShift)) m[c] = [`Digit${d}`, 48 + d, true]
  const oem = [[' ', 'Space', 32], ['-', 'Minus', 189], ['=', 'Equal', 187], ['[', 'BracketLeft', 219], [']', 'BracketRight', 221],
    ['\\', 'Backslash', 220], [';', 'Semicolon', 186], ["'", 'Quote', 222], [',', 'Comma', 188], ['.', 'Period', 190],
    ['/', 'Slash', 191], ['`', 'Backquote', 192]]
  for (const [c, code, kc] of oem) m[c] = [code, kc, false]
  const oemShift = { _: '-', '+': '=', '{': '[', '}': ']', '|': '\\', ':': ';', '"': "'", '<': ',', '>': '.', '?': '/', '~': '`' }
  for (const [c, base] of Object.entries(oemShift)) m[c] = [m[base][0], m[base][1], true]
  return m
})()

function keySpec(name, mods) {
  const lower = name.toLowerCase()
  if (NAMED_KEYS[lower]) {
    const [key, code, keyCode, text] = NAMED_KEYS[lower]
    return { key, code, keyCode, text: mods & (2 | 4) ? '' : (text || '') }
  }
  if ([...name].length === 1) {
    const ch = name
    const shifted = mods & 8 && /[a-z]/.test(ch) ? ch.toUpperCase() : ch
    const [code, keyCode] = US_KEYS[shifted] || US_KEYS[ch] || ['', 0]
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
    const k = US_KEYS[ch]
    if (!k) { await cdp(tabId, 'Input.insertText', { text: ch }); continue }   // not on a US keyboard: é, ü, emoji…
    const [code, keyCode, shift] = k
    const base = { key: ch, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode, modifiers: shift ? 8 : 0 }
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
  if (plainTabs.has(tabId)) {
    // Without the debugger Chrome pictures only the tab in front of its window.
    const tab = await chrome.tabs.get(tabId)
    if (!tab.active) { await chrome.tabs.update(tabId, { active: true }); await sleep(300) }
    const url = await withTimeout(chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' }), 10000, 'The tab did not draw a picture in time.')
    return { data: String(url).split(',')[1] || '' }
  }
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
  const t = attached.get(tabId) || plainTabs.get(tabId)
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
  if (args.ref) {
    const sid = sidOfRef(tab.id, args.ref)
    if (!sid) return helper(tab.id, 'point', { ref: args.ref })
    await frameBox(tab.id, sid, { reveal: true })
    const p = await helper(tab.id, 'point', { ref: args.ref }, sid)
    const box = await frameBox(tab.id, sid)
    return { ...p, x: p.x + box.x, y: p.y + box.y }
  }
  if (args.selector || args.text || args.role) {
    return helper(tab.id, 'pointBy', { selector: args.selector || '', text: args.text || '', role: args.role || '', nth: args.nth ?? null })
  }
  throw new Refusal('Point at something: ref (from read_page or find — best), coordinate (from a screenshot), selector, or text.')
}

function pointNotes(p) {
  return [p.covered ? `Careful: ${p.covered} was on top of it — the click may have landed there instead.` : '',
    p.visible === false ? 'It has no size on the page (hidden?) — check with a screenshot.' : '', p.note || ''].filter(Boolean)
}

const PLAIN_SAYS = '(made of page events — no debugger on this page; check it took)'

async function clickLike(ctx, args, { button = 'left', count = 1, verb = 'Clicked' } = {}) {
  const atPoint = Array.isArray(args.coordinate) && args.coordinate.length >= 2
  const simple = button === 'left' && count === 1 && !args.modifiers && !atPoint
  const tab = await useTab(ctx, args, simple ? { plain: 'script' }
    : { need: atPoint ? 'a click at a coordinate (click a ref instead)' : args.modifiers ? 'a click with keys held' : ctx.action })
  const pre = await problemsAll(tab.id)
  if (tab.plain) {
    if (!args.ref && !args.selector && !args.text && !args.role) throw new Refusal('Point at something: ref (from read_page or find — best), selector, or text.')
    const before = await groupTabIds(ctx.agent)
    const p = await helper(tab.id, 'press', { ref: args.ref || '', selector: args.selector || '', text: args.text || '', role: args.role || '', nth: args.nth ?? null })
    const after = await settle(ctx, tab, tab.url, before)
    const said = newProblems(pre, await problemsAll(tab.id))
    return { content: [`${verb} ${p.what}${p.ref ? ` [${p.ref}]` : ''} ${PLAIN_SAYS}.`, ...pointNotes(p), after, said].filter(Boolean).join('\n') }
  }
  const p = await pointFor(tab, args)
  const before = await groupTabIds(ctx.agent)
  const cut = await mouseClick(tab.id, p.x, p.y, { button, count, mods: modsOf(args.modifiers) })
  const after = await settle(ctx, tab, tab.url, before)
  const said = newProblems(pre, await problemsAll(tab.id))
  return { content: [`${verb} ${p.what}${p.ref ? ` [${p.ref}]` : ''}.`, cut ? CUT_SAYS : '', ...pointNotes(p), after, said].filter(Boolean).join('\n') }
}

// Typing into a field (a ref, a selector, a point) replaces what's in it unless
// append is set; the answer says what the field holds afterwards.
async function typeInto(ctx, tab, args, text) {
  const pre = await problemsAll(tab.id)
  if (tab.plain) {
    if (Array.isArray(args.coordinate)) throw new Refusal(needsDebugger('typing at a coordinate (give the field\'s ref instead)', tab.plain))
    const before = await groupTabIds(ctx.agent)
    const r = await helper(tab.id, 'typeIn', { ref: args.ref || '', selector: args.selector || '', nth: args.nth ?? null, value: text, append: !!args.append })
    const after = await settle(ctx, tab, tab.url, before)
    const said = newProblems(pre, await problemsAll(tab.id))
    return [`Typed ${text.length} character${text.length === 1 ? '' : 's'} into ${r.what} [${r.ref}]${args.append ? '' : ', replacing what was there'} ${PLAIN_SAYS}.`,
      /\n/.test(text) ? 'A line break went in as text: it pressed no Enter (nothing was sent or submitted) — click the button for that.' : '',
      r.note, fieldSays(r.field), after, said].filter(Boolean).join('\n')
  }
  let into = ''
  if (args.ref || args.selector || Array.isArray(args.coordinate)) {
    const p = await pointFor(tab, { ref: args.ref, selector: args.selector, coordinate: args.coordinate, nth: args.nth })
    await mouseClick(tab.id, p.x, p.y)
    into = ` into ${p.what}${p.ref ? ` [${p.ref}]` : ''}`
  }
  const field = await focusedField(tab.id)
  if (field && into && !args.append) {
    try { await helper(tab.id, 'selectField', { ref: field.ref }, field.sid) } catch { await pressCombo(tab.id, 'cmd+a') }
  }
  if (!field && into) into += ' (the click put no field in focus — check where the text went)'
  const before = await groupTabIds(ctx.agent)
  await typeText(tab.id, text)
  const after = await settle(ctx, tab, tab.url, before)
  let now = ''
  if (field) {
    try { now = fieldSays(await helper(tab.id, 'field', { ref: field.ref }, field.sid)) } catch { now = '' }
  }
  const said = newProblems(pre, await problemsAll(tab.id))
  return [`Typed ${text.length} character${text.length === 1 ? '' : 's'}${into}${field && into && !args.append ? ', replacing what was there' : ''}.`, now, after, said]
    .filter(Boolean).join('\n')
}

// ── files ──

// Runs in the page: a file fetched with the page's own session, as base64.
async function pageFetch(url, max) {
  const r = await fetch(url, { credentials: 'include' })
  if (!r.ok) return { ok: false, status: r.status }
  const blob = await r.blob()
  if (blob.size > max) return { ok: false, big: blob.size }
  const data = await new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(String(fr.result).split(',')[1] || ''); fr.onerror = rej; fr.readAsDataURL(blob) })
  return { ok: true, type: blob.type, size: blob.size, data, disposition: r.headers.get('content-disposition') || '', final: r.url }
}

async function fetchInPage(tabId, url) {
  try {
    if (plainTabs.has(tabId)) {
      const [res] = await withTimeout(chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, func: pageFetch, args: [url, FILE_MAX] }),
        45000, 'The page took too long to answer.')
      return res?.result || { ok: false }
    }
    const res = await evaluate(tabId, `(${pageFetch})(${JSON.stringify(url)}, ${FILE_MAX})`, { timeout: 45000 })
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
  return [`Your tabs (group “${ctx.agent.title}”):`, ...tabs.map((t) => tabLine(t, g.current))].join('\n')
}

const ACTIONS = {
  async status(ctx) {
    const own = ctx.conn?.agent
      ? `Connected: your own headless browser on the server (Captain for Chrome ${VERSION}) — not the person's Chrome; their sign-ins aren't here.`
      : `Connected: the person's own Chrome (${browserLabel()}, Captain for Chrome ${VERSION}), signed in as them.`
    const lines = [own, await listTabs(ctx)]
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
    return { content: `Opened tab ${t.id} (now current): ${clip(t.title || '', 120)} — ${t.url}${done ? '' : '\n(It was still loading after 30 s.)'}${await debuggerNote(t.id)}` }
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
      const tab = await useTab(ctx, args, { plain: 'tabs' })
      await (where === 'back' ? chrome.tabs.goBack(tab.id) : chrome.tabs.goForward(tab.id)).catch(() => {
        throw new Refusal(`There's no page to go ${where} to.`)
      })
      await sleep(300)
      await waitComplete(tab.id, 15000)
      const plain = await debuggerNote(tab.id)
      await waitDrawn(tab.id, { maxMs: 8000, quietMs: 700 })
      const t = await chrome.tabs.get(tab.id)
      guardTab(t, ctx.conn)
      return { content: `Went ${where}: ${clip(t.title || '', 120)} — ${t.url}${plain}` }
    }
    const url = checkUrl(where, ctx.conn)
    const tab = await useTab(ctx, args, { create: true, plain: 'tabs' })
    const done = tab.plain ? await plainNavigate(tab.id, url) : await navigateTab(tab.id, url)
    const plain = await debuggerNote(tab.id)
    const drawn = done ? await waitDrawn(tab.id, { maxMs: 10000, quietMs: 800, minMs: 1500 }) : null
    const t = await chrome.tabs.get(tab.id)
    const note = !done ? '\n(It was still loading after 30 s — take a screenshot or read the page to see.)'
      : drawn && !drawn.quiet ? '\n(It was still drawing after 10 s — read it again in a moment if something is missing.)' : ''
    return { content: `Opened in tab ${t.id}: ${clip(t.title || '', 120)} — ${t.url}${note}${plain}` }
  },

  async back(ctx) { return ACTIONS.navigate({ ...ctx, args: { ...ctx.args, url: 'back' } }) },
  async forward(ctx) { return ACTIONS.navigate({ ...ctx, args: { ...ctx.args, url: 'forward' } }) },

  async read_page(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    const filter = args.filter === 'all' ? 'all' : 'interactive'
    const maxChars = Math.min(Math.max(Number(args.max_chars) || 40000, 2000), 200000)
    const inside = args.ref || args.ref_id || ''
    await waitDrawn(tab.id, { maxMs: 4000, quietMs: 500 })
    if (inside) return { content: await refHelper(tab.id, 'tree', { filter, refId: inside, maxChars, ref: inside }) }
    const text = await helper(tab.id, 'tree', { filter, refId: '', maxChars })
    const frames = await inAllFrames(tab.id, 'tree', { filter, refId: '', maxChars: Math.min(maxChars, 12000) }, 'Inside a frame from')
    return { content: text + frames }
  },

  async find(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    const query = args.query || args.text || ''
    await waitDrawn(tab.id, { maxMs: 4000, quietMs: 500 })
    const here = await helper(tab.id, 'find', { query })
    const frames = await inAllFrames(tab.id, 'find', { query }, 'In a frame from')
    if (frames && /^Nothing on the page matches/.test(here)) return { content: `Nothing in the page itself matches ${JSON.stringify(query)}.${frames}` }
    return { content: here + frames }
  },

  async page_text(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    const offset = Math.max(0, Number(args.offset) || 0)
    await waitDrawn(tab.id, { maxMs: 6000, quietMs: 700 })
    // The first read goes through the whole page, so sections drawn only when
    // scrolled to are there too (the next parts, with offset, read it as is).
    if (!offset && args.scroll !== false) {
      const went = await scrollToLoad(tab.id, { steps: 10, quietMs: 600, stepMaxMs: 2500, totalMs: 12000 })
      if (went?.steps) await waitDrawn(tab.id, { maxMs: 3000, quietMs: 500 })
    }
    const cap = await capturePage(tab.id)
    const facts = [cap.tables?.length ? `${cap.tables.length} tables` : '', cap.files?.length ? `${cap.files.length} files (page_files lists them)` : '',
      cap.selection ? 'a selection' : ''].filter(Boolean)
    const head = `# ${cap.meta?.title || tab.title || ''}\n${cap.meta?.url || tab.url}${facts.length ? `\nThe page has ${facts.join(', ')}.` : ''}\n\n`
    const body = (cap.selection ? `Selected on the page:\n> ${cap.selection.replace(/\n/g, '\n> ')}\n\n` : '') + (cap.markdown || '(no text found)')
    const max = Math.min(Math.max(Number(args.max_chars) || TEXT_DEFAULT, 1000), 200000)
    const part = body.slice(offset, offset + max)
    const rest = body.length - offset - part.length
    return { content: head + part + (rest > 0 ? `\n\n… (${rest.toLocaleString()} more characters — page_text with offset=${offset + part.length})` : '') }
  },

  async page_files(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    const cap = await capturePage(tab.id)
    const all = [...(cap.files || []), ...(cap.images || [])]
    if (!all.length) return { content: 'No files are linked from this page.' }
    const lines = all.slice(0, 150).map((f, i) => `${i + 1}. ${f.name} — ${f.kind}${f.context ? ` (“${clip(f.context, 60)}”)` : ''} — ${f.url}`)
    return { content: `${all.length} file${all.length === 1 ? '' : 's'} on ${tab.url}:\n${lines.join('\n')}${all.length > 150 ? `\n… and ${all.length - 150} more` : ''}\nTo keep some: download with urls=[…] (or all=true for the documents).`,
      data: { files: all.slice(0, 150).map((f) => ({ name: f.name, url: f.url, kind: f.kind })) } }
  },

  async download(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
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
      if (offLimits(url, ctx.conn)) { failed.push(`${url}: Flight Deck's own address`); continue }
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
    const tab = await useTab(ctx, args, { plain: 'tabs' })
    const blob = tab.plain
      ? await withTimeout(chrome.pageCapture.saveAsMHTML({ tabId: tab.id }), 45000, 'Saving the page took too long.')
      : new Blob([(await withTimeout(cdp(tab.id, 'Page.captureSnapshot', { format: 'mhtml' }), 45000, 'Saving the page took too long.')).data], { type: 'multipart/related' })
    const base = (oneLine(tab.title) || hostOf(tab.url) || 'page').replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 80)
    const up = await uploadFiles(ctx, [{ blob, name: `${base}.mhtml` }], { url: tab.url, title: tab.title })
    return { content: `Saved the whole page (${tab.url}) as ${(up.saved || [])[0]?.path || up.folder} — it opens offline in a browser.` }
  },

  async form_input(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    if (!args.ref) throw new Refusal('form_input needs a ref (from read_page or find) and a value.')
    const value = args.value ?? args.text ?? ''
    const sid = tab.plain ? '' : sidOfRef(tab.id, args.ref)
    const pre = await problemsAll(tab.id)
    let said
    if (tab.plain && await helper(tab.id, 'fieldKind', { ref: args.ref }) === 'text') {
      // No debugger: the text goes in as an editing command, which pages see as input.
      said = fieldSays((await helper(tab.id, 'typeIn', { ref: args.ref, value: String(value) })).field)
    } else if (await helper(tab.id, 'fieldKind', { ref: args.ref }, sid) === 'text') {
      // A text field is set the way a person does it — select what's there, then
      // the new text as real input — so React, Vue and the rest all see it.
      await helper(tab.id, 'selectField', { ref: args.ref }, sid)
      const text = String(value)
      if (text) await cdp(tab.id, 'Input.insertText', { text })
      else await pressCombo(tab.id, 'Backspace')
      await sleep(80)
      said = fieldSays(await helper(tab.id, 'field', { ref: args.ref }, sid))
    } else {
      said = await helper(tab.id, 'formInput', { ref: args.ref, value }, sid)
    }
    return { content: [said, newProblems(pre, await problemsAll(tab.id))].filter(Boolean).join('\n') }
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
    const tab = await useTab(ctx, args, { plain: 'script' })
    return { content: await typeInto(ctx, tab, args, text) }
  },

  async press_key(ctx, args = ctx.args) {
    const keys = String(args.key || args.text || '').trim()
    if (!keys) throw new Refusal('press_key needs key (Enter, Tab, Escape, cmd+a, … — several separated by spaces).')
    const tab = await useTab(ctx, args)
    const pre = await problemsAll(tab.id)
    const before = await groupTabIds(ctx.agent)
    for (const combo of keys.split(/\s+/)) await pressCombo(tab.id, combo)
    const after = await settle(ctx, tab, tab.url, before)
    return { content: [`Pressed ${keys}.`, after, newProblems(pre, await problemsAll(tab.id))].filter(Boolean).join('\n') }
  },

  async scroll(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, Array.isArray(args.coordinate) ? { need: 'scrolling at a coordinate (give a ref, or none for the page)' } : { plain: 'script' })
    const dir = String(args.scroll_direction || 'down')
    const amount = Math.min(Math.max(Number(args.scroll_amount) || 500, 10), 20000)
    if (tab.plain) {
      const inf = await helper(tab.id, 'wheel', { ref: args.ref || '', dx: dir === 'left' ? -amount : dir === 'right' ? amount : 0,
        dy: dir === 'up' ? -amount : dir === 'down' ? amount : 0 })
      return { content: `Scrolled ${dir} ${amount} px. The page is at ${inf.sy} of ${inf.sh} px${inf.sx ? `, x ${inf.sx}` : ''}.` }
    }
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
    const tab = await useTab(ctx, args, { plain: 'script' })
    return { content: await refHelper(tab.id, 'scrollTo', { ref: args.ref }) }
  },

  async screenshot(ctx, args = ctx.args) {
    const tab = await useTab(ctx, args, { plain: 'script' })
    await waitDrawn(tab.id, { maxMs: 3000, quietMs: 400 })
    const s = await shoot(tab.id)
    const t = await chrome.tabs.get(tab.id)
    return { image: s.image, image_type: 'image/jpeg',
      content: `Screenshot of tab ${t.id}: ${clip(t.title || '', 120)} — ${t.url}\n${s.width}×${s.height} px; a coordinate you give (click, scroll…) is a point in this picture. Scrolled to ${s.info.sy} of ${s.info.sh} px.`,
      data: { width: s.width, height: s.height, url: t.url } }
  },

  async zoom(ctx, args = ctx.args) {
    if (!Array.isArray(args.region) || args.region.length !== 4) throw new Refusal('zoom needs region: [x0, y0, x1, y1] in a screenshot.')
    const tab = await useTab(ctx, args, { plain: 'script' })
    const s = await shoot(tab.id, args.region)
    return { image: s.image, image_type: 'image/jpeg',
      content: `Closer look at [${args.region.join(', ')}] (${s.width}×${s.height} px). Coordinates for clicks still come from the full screenshot.` }
  },

  async javascript(ctx, args = ctx.args) {
    const code = String(args.code || '').trim()
    if (!code) throw new Refusal('javascript needs code (its last expression is the answer; await works).')
    const tab = await useTab(ctx, args)
    await waitDrawn(tab.id, { maxMs: 3000, quietMs: 500 })
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
    const tab = await useTab(ctx, args, { plain: 'script' })
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
    const tab = await useTab(ctx, args, { plain: 'script' })
    return { content: await refHelper(tab.id, 'upload', { ref: args.ref, name: String(f.name || 'file'), type: String(f.type || 'application/octet-stream'), data: String(f.data) }) }
  },

  async release(ctx) {
    const g = await agentGroup(ctx.agent)
    const tabs = await groupTabs(g)
    for (const t of tabs) { await detach(t.id); plainTabs.delete(t.id) }
    if (g) { g.current = null; saveGroups() }
    return { content: `Let go of the person's Chrome.${tabs.length ? ` Your ${tabs.length} tab${tabs.length === 1 ? ' stays' : 's stay'} open in the “${ctx.agent.title}” group for them.` : ''}` }
  },
}

// Mark the agent as recently at work (a Cancel on Chrome's debugging bar stops it).
for (const [name, fn] of Object.entries(ACTIONS)) {
  ACTIONS[name] = async (ctx) => {
    const g = groups[ctx.agent.key]
    const out = await fn(ctx)
    const after = groups[ctx.agent.key] || g
    if (after) after.recentAt = Date.now()
    return out
  }
}
