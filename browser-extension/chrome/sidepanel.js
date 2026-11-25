// Captain for Chrome — the side panel.
//
// Reads the page (capture.js), fetches the files it links to with the
// person's own browser session, and sends it all to Flight Deck
// (/fd/clip/send), which saves it in the Web clips shared folder and hands
// it to the chosen agent. Then it follows the agent's progress and answer.
// "Anonymize" sends the page or the selection to Flight Deck instead
// (/fd/clip/anonymize): names, numbers and addresses come back as tokens,
// the real values stay in a vault named after the site.
//
// The browser can be connected to several Flight Decks (lib/api.js). The panel
// works with the one in use, picked from the menu at its top; agents on every
// one of them can use the browser all the same (hands.js).

import { icon } from './lib/icons.js'
import { renderMarkdown, escapeHtml as esc } from './lib/md.js'
import { AuthError, api, browserLabel, forgetDeck, loadDecks, normalizeDeckUrl, publicCall, saveDeck, upload, useDeck } from './lib/api.js'
import { SEARCH_ENGINES, loadSearchPick, loadSearchPrefs, saveSearchPick, saveSearchPrefs, servedEngines } from './lib/engines.js'
import { loadFetchMode, readsPages, saveFetchMode, setFetchDeclined } from './pages.js'

const VERSION = chrome.runtime.getManifest().version
const HISTORY_MAX = 30
const DOC_KINDS = new Set(['document', 'archive', 'data', 'design'])
const AUTO_FILES_MAX = 20
const FETCH_PARALLEL = 3

const $view = document.getElementById('view')
const $deckName = document.getElementById('deckName')
const $btnHistory = document.getElementById('btnHistory')
const $btnPlaybooks = document.getElementById('btnPlaybooks')
const $btnSettings = document.getElementById('btnSettings')
const $toast = document.getElementById('toast')
$btnHistory.innerHTML = icon('history')
$btnPlaybooks.innerHTML = icon('bolt')
$btnSettings.innerHTML = icon('settings')

const S = {
  view: 'loading',
  conn: null,          // the deck in use
  decks: [],           // every deck this browser is connected to
  kept: {},            // deck address → what the panel keeps for it: { agent, lenses, me }
  adding: false,       // connecting another deck (the connect view has a way back)
  me: null,
  meError: '',
  windowId: null,
  tab: null,
  scan: null,
  scanning: false,
  scanError: '',
  needsPermission: '',
  agent: '',
  action: 'auto',
  perspective: '',
  perspectiveText: '',
  note: '',
  include: { selection: true, snapshot: false, screenshot: false, tables: true },
  checked: new Set(),
  touchedFiles: false,
  extraFiles: [],
  focus: null,
  sending: null,
  sendError: '',
  run: null,
  followUp: '',
  pairing: null,
  connectError: '',
  deckInput: '',
  prefs: { snapshot: false, screenshot: false, textSize: 'small', watchEvery: 'daily' },
  allSites: false,
  history: [],
  follow: true,        // the run view keeps the newest activity in sight
  fileQuery: '',
  fileType: '',
  lenses: {},          // agent slug → {state, lenses, general, agent, at}
  lensMore: false,     // the general lenses shown beside the agent's own
  newTopic: false,     // a new topic being started
  hands: { on: false, front: false },   // agents may use this browser (hands.js)
  handsStatus: null,   // {state: on|connecting|permission|denied|off|error, error}
  handsLog: [],        // what agents did here, newest last
  handsStopped: [],    // agents the person stopped
  handsSeen: 0,        // the newest act the panel has followed
  search: { on: false, engines: [], declined: '' },   // agents' web searches here (lib/engines.js)
  searchServed: [],    // engines this browser runs searches on now (allowed + Chrome's permission)
  searchPickId: '',    // the engine picked in Flight Deck, as last heard (Flight Deck or the socket)
  fetch: { mode: '', declined: false },   // pages read here for agents' web_fetch (pages.js)
  fetchServed: false,  // this browser reads them now
  anon: null,          // the page or selection anonymized (/fd/clip/anonymize)
  rec: null,           // the recording going on (the worker's, session storage "rec")
  recNotes: {},        // the person's instructions on its steps, by step id
  recNoteOpen: new Set(),
  draft: null,         // a recording being reviewed or edited, not saved yet
  draftError: '',
  draftSaving: false,
  recs: null,          // the saved recordings (Flight Deck's list)
  recsError: '',
  recRun: null,        // a recording about to be run: {rec, note, inputs}
  asks: [],            // agents' questions open right now (/fd/clip/asks)
}

// ── small helpers ──

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const originOf = (url) => { try { return new URL(url).origin } catch { return '' } }
const hostOf = (url) => { try { return new URL(url).host.replace(/^www\./, '') } catch { return '' } }
const human = (n) => {
  if (!n && n !== 0) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

function toast(text) {
  $toast.textContent = text
  $toast.hidden = false
  clearTimeout(toast.t)
  toast.t = setTimeout(() => { $toast.hidden = true }, 2600)
}

function actionDef(id = S.action) {
  return (S.me?.actions || []).find((a) => a.id === id) || { id, label: id, files: 'none' }
}

function agentDef(slug = S.agent) {
  return (S.me?.agents || []).find((a) => a.slug === slug) || null
}

async function getStore(keys) {
  try { return await chrome.storage.local.get(keys) } catch { return {} }
}
async function setStore(obj) {
  try { await chrome.storage.local.set(obj) } catch { /* storage full or gone: just don't remember */ }
}

// ── rendering ──

function keepFocus(fn) {
  const el = document.activeElement
  const id = el && el.id
  const start = el && 'selectionStart' in el ? el.selectionStart : null
  const end = el && 'selectionEnd' in el ? el.selectionEnd : null
  fn()
  if (id) {
    const again = document.getElementById(id)
    if (again && again !== document.activeElement) {
      again.focus()
      if (start !== null && 'setSelectionRange' in again) {
        try { again.setSelectionRange(start, end) } catch { /* not a text field */ }
      }
    }
  }
}

// Lists that scroll on their own keep their place across a redraw; the steps
// list stays at its newest line unless the person scrolled it up.
function listScroll() {
  const out = {}
  for (const id of ['steps', 'fileList', 'recSteps']) {
    const el = document.getElementById(id)
    if (el) out[id] = { top: el.scrollTop, atEnd: el.scrollTop + el.clientHeight >= el.scrollHeight - 8 }
  }
  return out
}

function restoreScroll(before) {
  const steps = document.getElementById('steps')
  if (steps) steps.scrollTop = !before.steps || before.steps.atEnd ? steps.scrollHeight : before.steps.top
  const files = document.getElementById('fileList')
  if (files && before.fileList) files.scrollTop = before.fileList.top
  // A recording going on shows its newest step; a list being edited stays put.
  const rec = document.getElementById('recSteps')
  if (rec) rec.scrollTop = S.view === 'rec' && (!before.recSteps || before.recSteps.atEnd) ? rec.scrollHeight : before.recSteps?.top || 0
}

function render() {
  const connected = !!S.conn
  $btnHistory.hidden = !connected || S.view === 'connect' || S.view === 'agent' || S.view === 'decks'
  $btnPlaybooks.hidden = $btnHistory.hidden
  $btnPlaybooks.classList.toggle('on', S.view === 'pbs' || S.view === 'pb')
  $btnSettings.hidden = !connected || S.view === 'connect' || S.view === 'agent'
  $deckName.textContent = connected ? deckTitle(S.conn) : 'Not connected'
  // The deck menu: with a deck in use (not an agent's own browser).
  $deckName.disabled = !connected || !!S.conn.agent
  $deckName.classList.toggle('on', S.view === 'decks')
  $deckName.title = connected && !S.conn.agent ? (S.decks.length > 1 ? 'Switch Flight Deck' : 'Your Flight Decks') : ''
  const before = listScroll()
  renderView()
  restoreScroll(before)
  if (S.view === 'run') followRun()
}

function viewAgentOwned() {
  return `
    <div class="card">
      <div class="label">Connected</div>
      <div><b>${esc(S.conn.agent_name || S.conn.agent)}</b>'s own browser</div>
      <div class="dim">Flight Deck connected this browser for that agent's browser tool (${esc(hostOf(S.conn.url))}). It isn't a person's Chrome: sending pages from here is off.</div>
    </div>
    <div class="foot">Captain for Chrome ${esc(VERSION)}</div>`
}

// ── this extension's version against the one Flight Deck hands out ──
//
// Flight Deck answers /fd/clip/me with its own copy's version and how ours
// stands against it: "update" (the deck has a newer one), "ahead" (ours is
// newer than the deck's — the deck wasn't updated), "same".

function versionGap() {
  const ext = S.me?.extension
  // An answer kept from before this copy was put in says nothing about it.
  if (!ext || ext.yours !== VERSION || !ext.version) return null
  return ext.state === 'update' || ext.state === 'ahead' ? ext : null
}

/** The note about it; `always` = Settings, where "Later" doesn't hide it. */
function viewVersion(always = false) {
  const ext = versionGap()
  if (!ext || (!always && S.verSkip === `${ext.state}:${ext.version}`)) return ''
  if (ext.state === 'ahead') {
    return `<div class="notice ver"><div class="grow">This Captain for Chrome (${esc(VERSION)}) is newer than the one your Flight Deck has (${esc(ext.version)}). Some things may not work until Flight Deck is updated.</div>
      ${always ? '' : '<button class="btn" data-act="ver-later" style="padding:3px 9px">OK</button>'}</div>`
  }
  return `<div class="notice ver update">${icon('refresh')}<div class="grow"><b>Captain for Chrome ${esc(ext.version)} is ready</b> — this one is ${esc(VERSION)}.
    <div class="dim" style="margin-top:3px">Download it, unzip it over the folder you loaded this from, then press Reload.</div>
    <div class="row" style="margin-top:7px;gap:6px;flex-wrap:wrap">
      <button class="btn primary" data-act="ver-get" style="padding:3px 10px">Download</button>
      <button class="btn" data-act="ver-reload" style="padding:3px 10px">Reload</button>
      ${always ? '' : '<button class="link-btn" data-act="ver-later">Later</button>'}
    </div></div></div>`
}

// Recording, or a recording not saved yet, while the panel shows something else.
function viewRecBanner() {
  if (S.rec?.on) {
    return `<div class="notice rec-on"><span class="status-pill rec"><span class="dot"></span>Recording</span>
      <div class="grow">${plural(S.rec.steps.length, 'step')} so far</div><button class="btn" data-act="rec-show" style="padding:3px 9px">Show</button></div>`
  }
  if (S.draft && !S.draft.id) {
    return `<div class="notice"><div class="grow">Your recording of ${plural(S.draft.steps.length, 'step')} isn't saved yet.</div>
      <button class="btn" data-act="rec-review" style="padding:3px 9px">Review</button></div>`
  }
  return ''
}

function renderView() {
  keepFocus(() => {
    if (S.view === 'agent') $view.innerHTML = viewAgentOwned()
    else if (S.view === 'connect') $view.innerHTML = viewConnect()
    else if (S.view === 'main') $view.innerHTML = viewVersion() + viewRecBanner() + viewHands() + viewSearchAsk() + viewFetchAsk() + viewMain()
    else if (S.view === 'rec') $view.innerHTML = viewRec()
    else if (S.view === 'rec-edit') $view.innerHTML = viewRecEdit()
    else if (S.view === 'recs') $view.innerHTML = viewRecs()
    else if (S.view === 'rec-run') $view.innerHTML = viewRecRun()
    else if (S.view === 'run') $view.innerHTML = viewRun()
    else if (S.view === 'pbs') $view.innerHTML = viewStore()
    else if (S.view === 'pb') $view.innerHTML = viewPb()
    else if (S.view === 'anon') $view.innerHTML = viewAnon()
    else if (S.view === 'history') $view.innerHTML = viewHistory()
    else if (S.view === 'settings') $view.innerHTML = viewSettings()
    else if (S.view === 'decks') $view.innerHTML = viewDecks()
    else $view.innerHTML = '<div class="row dim"><div class="spinner"></div> Loading…</div>'
  })
}

// ── connect ──

function viewConnect() {
  const p = S.pairing
  if (p) {
    return `
      <div class="card">
        <div class="label">Approve in Flight Deck</div>
        <p class="muted" style="margin:0 0 6px">A tab with <b>${esc(hostOf(p.deck))}</b> opened. Sign in there if asked, check the code matches, and approve.</p>
        <div class="code">${esc(p.code)}</div>
        <div class="row dim" style="justify-content:center"><div class="spinner"></div> Waiting for your approval…</div>
      </div>
      <div class="row">
        <button class="btn grow" data-act="reopen">${icon('external', 'sm')} Open the approval page again</button>
        <button class="btn" data-act="cancel-pair">Cancel</button>
      </div>`
  }
  if (S.adding && S.decks.length) {
    return `
    <div class="row"><button class="icon-btn" data-act="decks">${icon('back')}</button><div class="grow" style="font-weight:600">Connect another Flight Deck</div></div>
    <div class="card">
      <p class="muted" style="margin:0">This browser stays connected to ${S.decks.length === 1 ? esc(hostOf(S.decks[0].url)) : `your ${S.decks.length} Flight Decks`} too. Agents on each of them can use it; the panel works with the one you pick at the top.</p>
    </div>
    ${viewDeckInput()}`
  }
  return `
    <div class="card">
      <div style="font-weight:600;font-size:calc(14px * var(--fs));margin-bottom:4px">Send any page to your agents</div>
      <p class="muted" style="margin:0">Summarize it, keep it with its files, pull out its data or ask about it — your agent works on it in Flight Deck, and the page lands in your <b>Web clips</b> shared folder.</p>
    </div>
    ${viewDeckInput()}`
}

function viewDeckInput() {
  const again = S.decks.find((d) => d.url === normalizeDeckUrl(S.deckInput))
  return `
    <div class="card">
      <label class="label" for="deckUrl">Your Flight Deck address</label>
      <input type="url" id="deckUrl" placeholder="captain.example.com" value="${esc(S.deckInput)}" autocomplete="url">
      ${S.connectError ? `<div class="notice err" style="margin-top:8px">${esc(S.connectError)}</div>` : ''}
      ${again && S.adding ? `<div class="dim" style="margin-top:6px">This browser is already connected to ${esc(hostOf(again.url))} — connecting again gives it a new token.</div>` : ''}
      <button class="btn primary block" data-act="connect" style="margin-top:10px">Connect this browser</button>
      <p class="dim" style="margin:8px 0 0">You approve it once in Flight Deck; no password is stored here. Disconnect any time in Connections → Browser.</p>
    </div>`
}

async function startPairing() {
  const deck = normalizeDeckUrl(document.getElementById('deckUrl')?.value || S.deckInput)
  S.deckInput = document.getElementById('deckUrl')?.value || S.deckInput
  S.connectError = ''
  if (!deck) {
    S.connectError = 'Type the address you open Flight Deck at.'
    render()
    return
  }
  try {
    const res = await publicCall(deck, '/fd/clip/pair', { method: 'POST', body: JSON.stringify({ browser: browserLabel() }) })
    S.pairing = { deck, code: res.code, id: res.pair_id, url: deck + res.approve_path, interval: (res.interval || 2) * 1000,
                  until: Date.now() + (res.expires_in || 600) * 1000 }
    await setStore({ lastDeck: deck })
    chrome.tabs.create({ url: S.pairing.url })
    render()
    pollPairing(S.pairing)
  } catch (e) {
    S.connectError = e.message
    render()
  }
}

async function pollPairing(p) {
  while (S.pairing === p) {
    await sleep(p.interval)
    if (S.pairing !== p) return
    if (Date.now() > p.until) {
      S.pairing = null
      S.connectError = 'The code expired — connect again.'
      render()
      return
    }
    try {
      const r = await publicCall(p.deck, `/fd/clip/pair/${encodeURIComponent(p.id)}`)
      if (r.status === 'approved' && r.token) {
        S.pairing = null
        const conn = { url: p.deck, token: r.token, user: r.user || {} }
        await saveDeck(conn)
        S.decks = (await loadDecks()).decks
        S.adding = false
        S.deckInput = ''
        toast(S.decks.length > 1 ? `Connected — ${hostOf(conn.url)} is in use` : 'Connected')
        await enterDeck(conn)
        return
      }
      if (r.status === 'denied' || r.status === 'expired') {
        S.pairing = null
        S.connectError = r.status === 'denied' ? 'The connection was turned down in Flight Deck.' : 'The code expired — connect again.'
        render()
        return
      }
    } catch { /* the deck blinked: keep waiting */ }
  }
}

// ── me, agents ──

async function loadMe() {
  const conn = S.conn
  if (!conn) return
  try {
    const me = await api(conn, `/fd/clip/me?v=${encodeURIComponent(VERSION)}`)
    if (S.conn !== conn) return   // another deck was picked meanwhile
    S.me = me
    S.meAt = Date.now()
    S.meError = ''
    keep({ me: S.me })
    if ('search' in S.me) {   // a deck from before web searches says nothing
      await saveSearchPick(conn.url, S.me.search?.engine || '')
      await refreshSearch()
    }
    if ('fetch' in S.me) {
      await saveFetchMode(conn.url, S.me.fetch?.mode || '')
      await refreshFetch()
    }
    if (!agentDef(S.agent)) {
      const firstUp = S.me.agents.find((a) => a.status === 'running') || S.me.agents.find((a) => a.status === 'asleep')
      S.agent = firstUp ? firstUp.slug : ''
    }
  } catch (e) {
    if (S.conn !== conn) return
    if (e instanceof AuthError) return disconnected(e.message, conn.url)
    S.meError = e.message
  }
}

// ── decks: several Flight Decks, one in use ──

/** "Glasses · glasses.example.com" once the deck said its name, else its host. */
function deckTitle(conn) {
  const name = S.kept[conn.url]?.me?.deck?.name
  return name ? `${name} · ${hostOf(conn.url)}` : hostOf(conn.url)
}

/** A deck in a few letters, beside an agent's name: its own name once it said
 * it ("Home") — unless another connected deck has the same one (most keep
 * the default "Captain") — else its host's first part. */
function deckShort(url) {
  const name = S.kept[url]?.me?.deck?.name
  if (name && !S.decks.some((d) => d.url !== url && S.kept[d.url]?.me?.deck?.name === name)) return name
  try {
    const u = new URL(url)
    return u.hostname === 'localhost' || /^[\d.]+$/.test(u.hostname) || !u.hostname.includes('.') ? u.host : u.hostname.split('.')[0]
  } catch { return hostOf(url) }
}

/** What the panel keeps for the deck in use: the agent picked, its lenses, the last /fd/clip/me. */
function keep(patch, url = S.conn?.url) {
  if (!url) return
  S.kept = { ...S.kept, [url]: { ...(S.kept[url] || {}), ...patch } }
  setStore({ decksKept: S.kept })
}

// The panel switches to a deck: what it kept for that one comes back, the
// rest is asked anew.
async function enterDeck(conn) {
  const kept = S.kept[conn.url] || {}
  S.conn = conn
  S.me = kept.me || null
  S.meError = ''
  S.meAt = 0
  S.agent = kept.agent || ''
  S.lenses = kept.lenses && typeof kept.lenses === 'object' ? kept.lenses : {}
  S.lensMore = false
  S.recs = null
  S.recsError = ''
  S.recRun = null
  PB.site = ''; PB.url = ''; PB.chips = []; PB.recsHere = []; PB.more = 0; PB.at = 0; PB.asked = ''
  PB.store = null; PB.storeError = ''; PB.items = null; PB.open = null
  S.view = S.rec ? 'rec' : 'main'
  if (conn.agent) S.view = 'agent'
  await refreshSearch()
  await refreshFetch()
  render()
  if (conn.agent) return
  await loadMe()
  if (S.conn !== conn) return
  applyDefaults()
  render()
  loadLenses()
  loadRecs()
  await scan()
}

async function switchDeck(url) {
  const d = S.decks.find((x) => x.url === url)
  if (!d || d === S.conn) { S.view = 'main'; render(); return }
  await useDeck(url)
  toast(`Now on ${deckTitle(d)}`)
  await enterDeck(d)
}

// A deck turned this browser away (its token was revoked), or the person
// disconnected it: off the list. The panel goes on with another deck, if any.
async function disconnected(why, url = S.conn?.url) {
  if (!url) return
  const next = await forgetDeck(url)
  S.decks = (await loadDecks()).decks
  const { [url]: _gone, ...rest } = S.kept
  S.kept = rest
  setStore({ decksKept: S.kept })
  if (S.conn?.url !== url) {
    if (why) toast(`${hostOf(url)}: ${why}`)
    render()
    return
  }
  if (next) {
    toast(why ? `${hostOf(url)}: ${why}` : `Disconnected from ${hostOf(url)}`)
    await enterDeck(next)
    return
  }
  S.conn = null
  S.me = null
  S.lenses = {}
  S.view = 'connect'
  S.adding = false
  S.connectError = why || ''
  render()
}

function deckState(url) {
  const st = S.handsStatus?.[url]
  if (!st || st.state === 'off') return null
  if (st.state === 'on') return { tone: 'done', text: 'Agents can use this browser' }
  if (st.state === 'search') return { tone: 'done', text: 'Runs agents\' searches and page reads' }
  if (st.state === 'connecting') return { tone: 'working', text: st.error || 'Connecting…' }
  if (st.state === 'permission') return { tone: 'error', text: 'Needs a reload of Captain for Chrome' }
  if (st.state === 'denied') return { tone: 'error', text: 'It disconnected this browser' }
  return { tone: 'error', text: st.error || 'Not connected' }
}

function viewDecks() {
  const rows = S.decks.map((d) => {
    const inUse = d.url === S.conn?.url
    const u = d.user || {}
    const st = deckState(d.url)
    const denied = S.handsStatus?.[d.url]?.state === 'denied'
    const pick = inUse ? '<span class="status-pill done"><span class="dot"></span>In use</span>'
      : denied ? `<button class="btn" data-act="deck-again" data-deck="${esc(d.url)}" style="padding:3px 10px">Connect again</button>`
      : `<button class="btn" data-act="deck-use" data-deck="${esc(d.url)}" style="padding:3px 10px">Use</button>`
    return `
    <div class="card deck-row ${inUse ? 'in-use' : ''}">
      <div class="row">
        <div class="grow" style="min-width:0">
          <div class="ellipsis" style="font-weight:600">${esc(S.kept[d.url]?.me?.deck?.name || hostOf(d.url))}</div>
          <div class="dim ellipsis">${esc(hostOf(d.url))}${u.name || u.email ? ` · ${esc(u.name || u.email)}` : ''}</div>
        </div>
        ${pick}
      </div>
      ${st ? `<div class="deck-state ${st.tone}"><span class="dot"></span><span class="dim ellipsis" title="${esc(st.text)}">${esc(st.text)}</span></div>` : ''}
      <div class="row" style="margin-top:6px;gap:10px">
        <button class="link-btn" data-act="deck-open" data-deck="${esc(d.url)}">Open Flight Deck</button>
        <button class="link-btn danger" data-act="deck-disconnect" data-deck="${esc(d.url)}">${denied ? 'Remove' : 'Disconnect'}</button>
      </div>
    </div>`
  }).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back">${icon('back')}</button><div class="grow" style="font-weight:600">Your Flight Decks</div></div>
    ${rows}
    <button class="btn block" data-act="deck-add">${icon('plus', 'sm')} Connect another Flight Deck</button>
    <p class="dim" style="margin:8px 2px 0">Agents on every deck here can use this browser, run their searches and read pages in it — each in tabs of its own. Pages you send go to the deck in use.</p>`
}

// ── reading the page ──

// sidepanel.html?tab=<id>: the panel opened in a tab of its own, working on
// that one tab (and what tests drive) instead of the window's active tab.
const PINNED_TAB = Number(new URLSearchParams(location.search).get('tab')) || null

async function currentTab() {
  if (PINNED_TAB) {
    try { return await chrome.tabs.get(PINNED_TAB) } catch { return null }
  }
  const [t] = await chrome.tabs.query({ active: true, windowId: S.windowId ?? chrome.windows.WINDOW_ID_CURRENT })
  return t || null
}

let scanTimer = 0
function scheduleScan(delay = 350) {
  clearTimeout(scanTimer)
  scanTimer = setTimeout(() => { if (S.view === 'main' && !S.sending) scan() }, delay)
}

function readable(url) {
  return /^(https?|file):/i.test(url || '')
}

async function capture(tabId) {
  const [res] = await chrome.scripting.executeScript({ target: { tabId }, files: ['capture.js'] })
  return res?.result || null
}

async function scan() {
  const tab = await currentTab()
  const changed = !S.tab || !tab || S.tab.id !== tab.id || S.tab.url !== tab.url
  S.tab = tab
  pbForTab(tab)
  S.needsPermission = ''
  S.scanError = ''
  if (changed) {
    S.scan = null
    S.touchedFiles = false
    S.extraFiles = []
    S.fileQuery = ''
    S.fileType = ''
    if (S.focus && (!tab || S.focus.tabId !== tab.id)) S.focus = null
  }
  if (!tab || !readable(tab.url)) {
    S.scan = null
    S.scanError = "Captain can't read this kind of page (browser pages, the Web Store, new tabs)."
    render()
    return
  }
  S.scanning = true
  render()
  try {
    const got = await capture(tab.id)
    if (got && got.ok) S.scan = got
    else S.scanError = "Couldn't read this page."
  } catch (e) {
    const msg = String(e?.message || e)
    if (/^file:/i.test(tab.url)) {
      S.scanError = 'To read files on this computer, turn on “Allow access to file URLs” for Captain in chrome://extensions.'
    } else if (/\.pdf($|[?#])/i.test(tab.url)) {
      S.scan = pdfScan(tab)
    } else if (/permission|cannot access|host|not allowed/i.test(msg)) {
      S.needsPermission = originOf(tab.url)
    } else {
      S.scanError = msg
    }
  }
  S.scanning = false
  applyDefaults()
  render()
}

function pdfScan(tab) {
  let name = 'document.pdf'
  try { name = decodeURIComponent(new URL(tab.url).pathname.split('/').pop()) || name } catch { /* keep */ }
  return {
    ok: true, pdf: true, markdown: '', words: 0, selection: null, tables: [], images: [], links: [], structured: '',
    structured_types: [],
    meta: { url: tab.url, title: tab.title || name, description: '', site_name: '', author: '', published: '', lang: '' },
    files: [{ url: tab.url, name, ext: 'pdf', kind: 'document', context: 'This page is a PDF', same_origin: true, source: 'page' }],
  }
}

function allFiles(scanData = S.scan) {
  const seen = new Set()
  const out = []
  for (const f of [...S.extraFiles, ...(scanData?.files || []), ...(scanData?.images || [])]) {
    if (seen.has(f.url)) continue
    seen.add(f.url)
    out.push(f)
  }
  return out
}

function applyDefaults() {
  const a = actionDef()
  S.include.snapshot = !!a.snapshot || S.prefs.snapshot
  S.include.screenshot = S.prefs.screenshot || (!!S.scan && !S.scan.pdf && (S.scan.words || 0) < 80)
  S.include.tables = true
  if (S.touchedFiles) return
  S.checked = new Set(S.extraFiles.map((f) => f.url))
  const files = S.scan?.files || []
  const primary = files.filter((f) => f.source === 'page')
  for (const f of primary) S.checked.add(f.url)
  if (a.files === 'documents' || a.files === 'all') {
    const pool = files.filter((f) => f.source !== 'page' && (a.files === 'all' ? f.kind !== 'image' || f.source !== 'img' : DOC_KINDS.has(f.kind)))
    for (const f of pool.slice(0, a.files === 'all' ? (S.me?.limits?.files || 60) : AUTO_FILES_MAX)) S.checked.add(f.url)
  }
}

// ── files: their kinds, and the list's search and filter ──

const FILE_GROUPS = [
  ['pdf', 'PDF', (f) => f.ext === 'pdf'],
  ['docs', 'Docs', (f) => /^(doc|docx|dot|dotx|odt|rtf|pages|epub|mobi|txt|md)$/.test(f.ext)],
  ['sheets', 'Sheets', (f) => /^(xls|xlsx|xlsm|xlsb|ods|numbers|csv|tsv)$/.test(f.ext)],
  ['slides', 'Slides', (f) => /^(ppt|pptx|pps|ppsx|odp|key)$/.test(f.ext)],
  ['archives', 'Archives', (f) => f.kind === 'archive'],
  ['data', 'Data', (f) => f.kind === 'data'],
  ['images', 'Images', (f) => f.kind === 'image'],
  ['media', 'Audio & video', (f) => f.kind === 'media'],
  ['design', 'Design', (f) => f.kind === 'design'],
]

function fileGroup(f) {
  const g = { ...f, ext: String(f.ext || '').toLowerCase() }
  for (const [id, , test] of FILE_GROUPS) if (test(g)) return id
  return 'other'
}

function matchesQuery(f) {
  const words = S.fileQuery.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return true
  const text = `${f.name} ${f.ext || ''} ${f.context || ''} ${hostOf(f.url)}`.toLowerCase()
  return words.every((w) => text.includes(w))
}

// The kinds on this page with how many match the search: [{id, label, count}].
function fileGroups(files) {
  const found = files.filter(matchesQuery)
  const out = []
  for (const [id, label] of [...FILE_GROUPS, ['other', 'Other']]) {
    const all = files.filter((f) => fileGroup(f) === id).length
    if (all) out.push({ id, label, count: found.filter((f) => fileGroup(f) === id).length })
  }
  return out
}

function shownFiles(files) {
  return files.filter((f) => (!S.fileType || fileGroup(f) === S.fileType) && matchesQuery(f))
}

const filesFiltered = () => !!(S.fileType || S.fileQuery.trim())

// ── lenses: the agent's own ──

let lensTimer = 0

// Asks the deck for this agent's lenses; while the agent is still thinking
// them up the general ones show, and the deck is asked again shortly.
async function loadLenses(slug = S.agent, { fresh = false } = {}) {
  const conn = S.conn
  if (!slug || !conn) return
  clearTimeout(lensTimer)
  let tries = 0
  const ask = async () => {
    if (S.agent !== slug || S.conn !== conn) return
    try {
      const r = await api(conn, `/fd/clip/agents/${encodeURIComponent(slug)}/lenses${fresh && tries === 0 ? '?fresh=1' : ''}`)
      if (S.conn !== conn) return
      S.lenses[slug] = { ...r, at: Date.now() }
      keep({ lenses: S.lenses })
    } catch (e) {
      if (S.conn !== conn) return
      if (e instanceof AuthError) return disconnected(e.message, conn.url)
      // A deck without the route (older) or a hiccup: the general ones, no remark.
      S.lenses[slug] = { ...(S.lenses[slug] || {}), state: 'unavailable', at: Date.now() }
    }
    tries += 1
    keepLensChoice()
    if (S.view === 'main') render()
    if (S.lenses[slug]?.state === 'making' && tries < 40) lensTimer = setTimeout(ask, 2500)
  }
  await ask()
}

function agentLenses(slug = S.agent) {
  const own = S.lenses[slug]
  return own?.source === 'agent' && Array.isArray(own.lenses) ? own.lenses : []
}

function lensDef(id = S.perspective) {
  return [...agentLenses(), ...(S.me?.perspectives || [])].find((x) => x.id === id) || null
}

// A lens of another agent's (or one made again since) isn't offered: back to Plain.
function keepLensChoice() {
  if (S.perspective && S.perspective !== 'custom' && !lensDef()) S.perspective = ''
}

function viewLenses() {
  const own = S.lenses[S.agent]
  const mine = agentLenses()
  const general = S.me?.perspectives || []
  const name = agentDef()?.name || own?.agent || 'the agent'
  const list = mine.length ? [...mine, ...(S.lensMore ? general.map((g) => ({ ...g, general: true })) : [])] : general
  const chip = (p) => `<button class="chip ${p.id === S.perspective ? 'on' : ''} ${p.general ? 'general' : ''}" data-lens="${esc(p.id)}" aria-pressed="${p.id === S.perspective}"${p.hint ? ` title="${esc(p.hint)}"` : ''}>${esc(p.label)}</button>`
  const chips = [{ id: '', label: 'Plain' }, ...list, { id: 'custom', label: 'Your own…' }].map(chip).join('')
  const more = mine.length ? `<button class="chip ghost" data-act="lens-more">${S.lensMore ? 'Fewer' : 'More…'}</button>` : ''
  let note = ''
  if (own?.state === 'making') note = `<span class="lens-note"><span class="spinner" style="width:10px;height:10px"></span> asking ${esc(name)} for its own…</span>`
  else if (mine.length) note = `<span class="lens-note">${esc(name)}'s own</span>`
  else if (own?.state === 'asleep') note = `<span class="lens-note">${esc(name)}'s own once it's running</span>`
  else if (own?.state === 'fallback') note = `<span class="lens-note">general — ${esc(name)} couldn't suggest its own</span>`
  const again = S.agent && own && !['making', 'asleep', 'unavailable'].includes(own.state)
    ? `<button class="icon-btn" data-act="lens-refresh" title="Ask ${esc(name)} for new lenses" style="width:20px;height:20px">${icon('refresh', 'sm')}</button>` : ''
  return `
    <div class="label-row"><span class="label" style="margin:0">Read it as</span>${note}<span class="grow"></span>${again}</div>
    <div class="chips">${chips}${more}</div>
    ${S.perspective === 'custom' ? `<input type="text" id="lensText" style="margin-top:6px" placeholder="e.g. a hotel owner in Split" value="${esc(S.perspectiveText)}">` : ''}`
}

// ── a new topic: the agent's chat starts a fresh session (/new) ──

function newTopicButton(slug = S.agent) {
  const ag = agentDef(slug)
  if (!ag) return ''
  const working = S.run && !S.run.done && S.run.agentSlug === slug
  const why = working ? `Wait for ${ag.name}'s answer first` : `Start a new topic with ${ag.name}: a fresh session in its chat (the last one is kept in its history)`
  return `<button class="link-btn" data-act="new-topic" data-slug="${esc(slug)}" title="${esc(why)}" ${S.newTopic || working ? 'disabled' : ''}>
    ${S.newTopic ? '<span class="spinner" style="width:10px;height:10px"></span>' : icon('newtopic', 'sm')} New topic</button>`
}

async function newTopic(slug) {
  const ag = agentDef(slug)
  if (!ag || S.newTopic) return
  S.newTopic = true
  render()
  try {
    await api(S.conn, `/fd/clip/agents/${encodeURIComponent(slug)}/new-topic`, { method: 'POST' })
    toast(`New topic with ${ag.name} — the last one is kept in its history.`)
  } catch (e) {
    if (e instanceof AuthError) { S.newTopic = false; return disconnected(e.message) }
    toast(e.message)
  }
  S.newTopic = false
  render()
}

// ── "Watch it": how often the deck's Scheduled task checks ──

const WATCH_EVERY = [['hourly', 'Every hour'], ['6h', 'Every 6 hours'], ['daily', 'Daily'], ['weekly', 'Weekly']]

function watchEvery() {
  return WATCH_EVERY.some(([id]) => id === S.prefs.watchEvery) ? S.prefs.watchEvery : 'daily'
}

function browserZone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || '' } catch { return '' }
}

function viewWatchTask(w) {
  if (!w) return ''
  let next = ''
  if (w.next_run_at) {
    try {
      next = new Date(w.next_run_at * 1000).toLocaleString([], { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
    } catch { /* no date */ }
  }
  return `
    <div class="card watch-card">
      <div class="row" style="align-items:flex-start">${icon('eye')}
        <div class="grow">
          <div class="label" style="margin-bottom:2px">${w.made === 'updated' ? 'Updated in Scheduled tasks' : 'Added to Scheduled tasks'}</div>
          <div style="font-weight:600" class="ellipsis">${esc(w.name)}</div>
          <div class="dim">${esc(w.when)}${next ? ` · next ${esc(next)}` : ''}</div>
        </div>
      </div>
      <button class="btn block" data-act="open-task" style="margin-top:8px">${icon('external', 'sm')} Change or stop it in Flight Deck</button>
    </div>`
}

// ── the main view ──

function viewMain() {
  if (S.meError && !S.me) {
    return `<div class="notice err">${esc(S.meError)}</div>
      <button class="btn block" data-act="retry-me">${icon('refresh', 'sm')} Try again</button>`
  }
  if (!S.me) return '<div class="row dim"><div class="spinner"></div> Loading your agents…</div>'
  const agents = S.me.agents || []
  const a = actionDef()
  const t = S.tab
  const scan = S.scan
  const files = allFiles()
  const nChecked = files.filter((f) => S.checked.has(f.url)).length
  const facts = []
  if (S.scanning) facts.push(`<span class="fact"><span class="spinner" style="width:10px;height:10px"></span> Reading…</span>`)
  if (scan) {
    if (scan.pdf) facts.push(`<span class="fact">${icon('file', 'sm')} PDF</span>`)
    else facts.push(`<span class="fact">${icon('text', 'sm')} ${(scan.words || 0).toLocaleString()} words</span>`)
    if (files.length) facts.push(`<span class="fact">${icon('download', 'sm')} ${plural(files.length, 'file')}</span>`)
    if (scan.tables?.length) facts.push(`<span class="fact">${icon('table', 'sm')} ${plural(scan.tables.length, 'table')}</span>`)
    if (scan.selection) facts.push(`<span class="fact">${icon('text', 'sm')} selection</span>`)
    if (scan.structured_types?.length) facts.push(`<span class="fact">${esc(scan.structured_types.slice(0, 2).join(', '))}</span>`)
  }

  const agentOptions = agents.length
    ? agents.map((ag) => {
      const mark = ag.status === 'running' ? '●' : ag.status === 'asleep' ? '◐' : '○'
      const extra = ag.status === 'running' ? '' : ` (${ag.status})`
      return `<option value="${esc(ag.slug)}" ${ag.slug === S.agent ? 'selected' : ''} ${ag.status === 'running' || ag.status === 'asleep' ? '' : 'disabled'}>${mark} ${esc(ag.name)}${esc(extra)}</option>`
    }).join('')
    : ''

  const actions = (S.me.actions || []).map((x) => `
    <button class="action ${x.id === S.action ? 'on' : ''}" data-action="${esc(x.id)}" title="${esc(x.hint || '')}" aria-pressed="${x.id === S.action}">
      ${icon(x.icon, 'lg')}<span>${esc(x.label)}</span>
    </button>`).join('')

  const focus = S.focus ? `
    <div class="focus">${icon(S.focus.kind === 'selection' ? 'text' : S.focus.kind === 'image' ? 'image' : 'link', 'sm')}
      <div class="grow">${esc(S.focus.label)}</div>
      <button class="icon-btn" data-act="clear-focus" title="Forget it" style="width:20px;height:20px">${icon('x', 'sm')}</button>
    </div>` : ''

  let pageBlock
  if (!t) pageBlock = '<div class="dim">No tab.</div>'
  else {
    const fav = t.favIconUrl && /^https?:|^data:image\//.test(t.favIconUrl) ? `<img class="favicon" src="${esc(t.favIconUrl)}" alt="">` : `<span class="favicon"></span>`
    pageBlock = `
      <div class="page">${fav}
        <div class="grow">
          <div class="page-title">${esc(scan?.meta?.title || t.title || t.url)}</div>
          <div class="page-host ellipsis">${esc(hostOf(t.url) || t.url)}</div>
          ${facts.length ? `<div class="facts">${facts.join('')}</div>` : ''}
        </div>
        <button class="icon-btn" data-act="rescan" title="Read the page again">${icon('refresh', 'sm')}</button>
      </div>`
  }
  let pageNotice = ''
  if (S.needsPermission) {
    pageNotice = `<div class="notice">${icon('lock')}<div class="grow">Captain needs your OK to read <b>${esc(hostOf(S.needsPermission))}</b>.
      <div style="margin-top:6px"><button class="btn" data-act="grant">Allow on this site</button></div></div></div>`
  } else if (S.scanError) {
    pageNotice = `<div class="notice">${esc(S.scanError)}</div>`
  }

  const shown = shownFiles(files)
  const groups = files.length > 1 ? fileGroups(files) : []
  const hiddenTicked = files.filter((f) => S.checked.has(f.url) && !shown.includes(f)).length
  const fileRows = shown.map((f) => `
    <label class="file">
      <input type="checkbox" data-file="${esc(f.url)}" ${S.checked.has(f.url) ? 'checked' : ''}>
      <div class="grow">
        <div class="file-name">${esc(f.name)} <span class="tag">${esc(f.ext || f.kind)}</span></div>
        <div class="file-meta ellipsis">${esc([f.context, f.same_origin ? '' : hostOf(f.url)].filter(Boolean).join(' · ') || hostOf(f.url))}</div>
      </div>
    </label>`).join('')

  const sel = scan?.selection
  const includeSummary = ['page text', sel && S.include.selection ? 'selection' : '', S.include.snapshot ? 'whole page' : '',
    S.include.screenshot ? 'screenshot' : '', scan?.tables?.length && S.include.tables ? plural(scan.tables.length, 'table') : '',
    nChecked ? plural(nChecked, 'file') : ''].filter(Boolean).join(', ')

  const ag = agentDef()
  const canSend = !!(ag && t && scan && !S.scanning && !S.needsPermission && !S.sending)
  const needsText = a.needs_text
  const placeholder = needsText ? 'Your question about the page…' : 'Anything to add? (optional)'

  const sending = S.sending ? `
    <div class="card">
      <div class="row"><div class="spinner"></div><div class="grow">${esc(S.sending.text)}</div></div>
      ${S.sending.progress != null ? `<div class="progress"><div style="width:${Math.round(S.sending.progress * 100)}%"></div></div>` : ''}
    </div>` : ''

  return `
    ${agents.length ? `
      <div>
        <div class="label-row"><label class="label" for="agent" style="margin:0">Agent</label><span class="grow"></span>${newTopicButton()}</div>
        <select class="select" id="agent">${agentOptions}</select>
      </div>` : `<div class="notice">No agents yet — create or start one in Flight Deck, then <button class="btn" data-act="retry-me" style="padding:2px 8px">reload</button></div>`}
    <div class="card">${pageBlock}</div>
    ${viewPbStrip()}
    ${pageNotice}
    ${focus}
    <div>
      <div class="label">What should it do?</div>
      <div class="actions">${actions}</div>
      <div class="hint">${esc(a.hint || '')}</div>
      ${a.id === 'watch' ? `
        <div class="label-row" style="margin-top:8px"><span class="label" style="margin:0">How often</span></div>
        <div class="chips">${WATCH_EVERY.map(([id, label]) => `
          <button class="chip ${watchEvery() === id ? 'on' : ''}" data-watch="${id}" aria-pressed="${watchEvery() === id}">${esc(label)}</button>`).join('')}</div>
        <div class="hint">Flight Deck runs it in Scheduled tasks — change or stop it there.</div>` : ''}
    </div>
    <div>${viewLenses()}</div>
    <div>
      <textarea id="note" placeholder="${esc(placeholder)}" rows="${needsText ? 3 : 2}">${esc(S.note)}</textarea>
    </div>
    <details class="include card" ${S.includeOpen ? 'open' : ''}>
      <summary><span class="grow"><b>What goes with it</b> <span class="dim">— ${esc(includeSummary)}</span></span>${icon('chevron', 'sm chev')}</summary>
      <div style="margin-top:6px">
        <label class="opt"><input type="checkbox" checked disabled><div>Page text<div class="sub">The page as Markdown, with its files and links listed</div></div></label>
        ${sel ? `<label class="opt"><input type="checkbox" data-inc="selection" ${S.include.selection ? 'checked' : ''}><div>Selected text<div class="sub">${esc(sel.text.slice(0, 80))}${sel.text.length > 80 ? '…' : ''}</div></div></label>` : ''}
        <label class="opt"><input type="checkbox" data-inc="snapshot" ${S.include.snapshot ? 'checked' : ''}><div>Whole page<div class="sub">One .mhtml file with its pictures and styles — opens offline</div></div></label>
        <label class="opt"><input type="checkbox" data-inc="screenshot" ${S.include.screenshot ? 'checked' : ''}><div>Screenshot<div class="sub">What's on screen now — the agent sees it</div></div></label>
        ${scan?.tables?.length ? `<label class="opt"><input type="checkbox" data-inc="tables" ${S.include.tables ? 'checked' : ''}><div>Tables as CSV<div class="sub">${plural(scan.tables.length, 'table')}</div></div></label>` : ''}
        ${files.length ? `
          <div class="row" style="margin-top:6px"><div class="grow"><b>Files on the page</b> <span class="dim">${nChecked} of ${files.length} ticked</span></div></div>
          ${files.length >= 5 ? `
            <div class="search">${icon('search', 'sm')}<input type="search" id="fileSearch" placeholder="Search ${files.length} files" value="${esc(S.fileQuery)}" autocomplete="off" spellcheck="false"></div>` : ''}
          ${groups.length > 1 ? `
            <div class="chips types">${[{ id: '', label: 'All', count: files.filter(matchesQuery).length }, ...groups].map((g) => `
              <button class="chip sm ${S.fileType === g.id ? 'on' : ''}" data-ftype="${esc(g.id)}" aria-pressed="${S.fileType === g.id}" ${g.count || g.id === S.fileType ? '' : 'disabled'}>${esc(g.label)} <span class="n">${g.count}</span></button>`).join('')}</div>` : ''}
          <div class="links-row">${filesFiltered()
            ? `<button data-act="files-shown-on" ${shown.length ? '' : 'disabled'}>Tick these ${shown.length}</button><button data-act="files-shown-off" ${shown.length ? '' : 'disabled'}>Untick these</button><button data-act="files-unfilter">Show all</button>`
            : '<button data-act="files-docs">Documents</button><button data-act="files-all">All</button><button data-act="files-none">None</button>'}</div>
          <div class="files" id="fileList">${fileRows || `<div class="dim" style="padding:8px">No file matches${S.fileQuery.trim() ? ` “${esc(S.fileQuery.trim())}”` : ''}.</div>`}</div>
          ${hiddenTicked ? `<div class="dim" style="margin-top:4px">${plural(hiddenTicked, 'ticked file')} not shown here</div>` : ''}` : '<div class="dim" style="margin-top:4px">No files linked on this page.</div>'}
      </div>
    </details>
    ${S.sendError ? `<div class="notice err">${esc(S.sendError)}</div>` : ''}
    ${sending}
    <button class="btn primary block" data-act="send" ${canSend ? '' : 'disabled'}>${icon('send', 'sm')} ${ag ? `Send to ${esc(ag.name)}` : 'Pick an agent'}</button>
    <button class="btn block" data-act="anonymize" ${t && scan && !S.scanning && !S.needsPermission ? '' : 'disabled'}
      title="Names, numbers and addresses out — the real ones kept in Flight Deck, in your vault “${esc(vaultOf(t?.url))}”">
      ${icon('shield', 'sm')} ${anonSource() === 'selection' ? 'Anonymize the selection' : 'Anonymize this page'}</button>
    <div class="row">
      <button class="btn grow" data-act="rec-start" ${recordable(t) && !S.rec?.on ? '' : 'disabled'}
        title="Do something on this site once — an agent does it for you after that">${icon('record', 'sm')} Record what I do</button>
      <button class="btn grow" data-act="recs">${icon('play', 'sm')} Recordings${S.recs?.length ? ` <span class="dim">${S.recs.length}</span>` : ''}</button>
    </div>
    <div class="foot">${icon('folder', 'sm')} Saved in Shared folders → ${esc(S.me.folder || 'web-clips')}</div>`
}

// ── anonymize (the page or the selection) ──

// The vault: the site as the browser names it ("www.example.com", www kept).
const vaultOf = (url) => { try { return new URL(url).hostname.toLowerCase() } catch { return 'web' } }

function anonSource() {
  const sel = S.scan?.selection
  return sel?.text && (S.focus?.kind === 'selection' || S.include.selection) ? 'selection' : 'page'
}

function anonContent(source) {
  if (source === 'selection') return S.scan?.selection?.markdown || S.scan?.selection?.text || ''
  return S.scan?.markdown || ''
}

// Text with every token as a chip (the text stays exactly as it is copied).
const TOKEN_RE = /⟦[A-Z][A-Z0-9_]*?_\d{1,6}(?::[a-z]{1,3})?⟧/g
function tokenHtml(text) {
  let out = ''
  let last = 0
  for (const m of String(text || '').matchAll(TOKEN_RE)) {
    out += esc(text.slice(last, m.index)) + `<span class="tok">${esc(m[0])}</span>`
    last = m.index + m[0].length
  }
  return out + esc(String(text || '').slice(last))
}

const KIND_NAMES = { PERSON: 'person', ORG: 'company', EMAIL: 'email', PHONE: 'phone', ADDRESS: 'address', DOB: 'date of birth',
  IBAN: 'IBAN', CARD: 'card', OIB: 'OIB', JMBG: 'JMBG', SSN: 'SSN', PASSPORT: 'passport', IP: 'IP', URL: 'link', PLATE: 'plate' }
const kindName = (k) => KIND_NAMES[k] || k.toLowerCase()

async function anonymize({ content, source, save = false } = {}) {
  const t = S.tab
  if (!t || !S.conn) return
  const src = source || anonSource()
  const text = content ?? anonContent(src)
  if (!text.trim()) { toast("There's no text to anonymize here."); return }
  const title = (S.scan?.meta?.url === t.url && S.scan.meta.title) || t.title || ''
  S.anon = { state: 'working', source: src, content: text, title, url: t.url,
    vault: vaultOf(t.url), output: '', summary: {}, warnings: [], saved: null, saving: false, back: '', restored: null, restoring: false }
  S.view = 'anon'
  render()
  try {
    const r = await api(S.conn, '/fd/clip/anonymize', { method: 'POST', body: JSON.stringify({ content: text, url: t.url, title: S.anon.title, save }) })
    Object.assign(S.anon, { state: 'done', output: r.output || '', summary: r.summary || {}, vault: r.vault?.name || S.anon.vault,
      warnings: r.warnings || [], saved: r.saved || null, lane: r.lane, lanes: r.lanes || {} })
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    Object.assign(S.anon, { state: 'error', error: e.message })
  }
  render()
}

// Keep the result in Flight Deck (Web clips): the same vault gives the same tokens.
async function anonSave() {
  const a = S.anon
  if (!a || a.saving || a.saved) return
  a.saving = true
  render()
  try {
    const r = await api(S.conn, '/fd/clip/anonymize', { method: 'POST', body: JSON.stringify({ content: a.content, url: a.url, title: a.title, save: true }) })
    Object.assign(a, { output: r.output || a.output, saved: r.saved || null })
    toast('Saved in Web clips')
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
  }
  a.saving = false
  render()
}

async function anonRestore() {
  const a = S.anon
  if (!a || a.restoring || !a.back.trim()) return
  a.restoring = true
  render()
  try {
    const r = await api(S.conn, '/fd/clip/anonymize/restore', { method: 'POST', body: JSON.stringify({ content: a.back, vault: a.vault }) })
    a.restored = { output: r.output || '', count: r.restored || 0, unknown: r.unknown || [] }
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
  }
  a.restoring = false
  render()
}

function viewAnon() {
  const a = S.anon
  if (!a) return ''
  const kinds = Object.entries(a.summary || {}).filter(([, n]) => n > 0)
  const found = kinds.length
    ? `<div class="facts">${kinds.map(([k, n]) => `<span class="fact">${esc(kindName(k))} · ${n}</span>`).join('')}</div>`
    : '<div class="dim">Nothing private was found.</div>'
  const who = a.lanes?.llm === 'failed' ? 'The trusted model failed, so code alone looked — check for missed names.'
    : a.lane === 'deterministic' ? 'Code looked (patterns, labels, names the vault knows) — a name with nothing around it can be missed.'
      : 'Code and the trusted model looked.'
  let body
  if (a.state === 'working') body = '<div class="card row"><div class="spinner"></div><div class="grow">Taking names, numbers and addresses out…</div></div>'
  else if (a.state === 'error') body = `<div class="notice err">${esc(a.error)}</div>`
  else {
    const r = a.restored
    body = `
      <div class="card"><div class="label">Found</div>${found}<div class="hint" style="margin-top:6px">${esc(who)}</div></div>
      ${(a.warnings || []).map((w) => `<div class="notice">${esc(w)}</div>`).join('')}
      <div class="card"><div class="anon-out" id="anonOut">${tokenHtml(a.output)}</div></div>
      <div class="row">
        <button class="btn grow" data-act="anon-copy">${icon('copy', 'sm')} Copy</button>
        <button class="btn grow" data-act="anon-save" ${a.saved || a.saving ? 'disabled' : ''}>${a.saving ? '<div class="spinner" style="width:12px;height:12px"></div>' : icon(a.saved ? 'check' : 'folder', 'sm')} ${a.saved ? 'Saved' : 'Save in Flight Deck'}</button>
      </div>
      ${a.saved ? `<div class="dim ellipsis" title="${esc(a.saved.vfs)}">${icon('folder', 'sm')} ${esc(a.saved.vfs)}</div>` : ''}
      <div class="dim">The real values stay in Flight Deck, in your vault <b>${esc(a.vault)}</b> — the same person gets the same token on every page of this site.</div>
      <details class="card anon-back" ${a.back || r ? 'open' : ''}>
        <summary><span class="grow"><b>Put the real names back</b></span>${icon('chevron', 'sm chev')}</summary>
        <div style="margin-top:6px">
          <textarea id="anonBack" rows="4" placeholder="Paste text with tokens like ⟦PERSON_1⟧ — an answer you got for the anonymized text">${esc(a.back)}</textarea>
          <button class="btn block" data-act="anon-restore" style="margin-top:6px" ${a.restoring || !a.back.trim() ? 'disabled' : ''}>${a.restoring ? '<div class="spinner" style="width:12px;height:12px"></div>' : icon('refresh', 'sm')} Put them back</button>
          ${r ? `
            <div class="anon-out" style="margin-top:8px">${esc(r.output)}</div>
            <div class="row" style="margin-top:6px"><span class="dim grow">${plural(r.count, 'value')} put back${r.unknown.length ? ` · not in this vault: ${esc(r.unknown.slice(0, 6).join(', '))}` : ''}</span>
              <button class="btn" data-act="anon-copy-back">${icon('copy', 'sm')} Copy</button></div>` : ''}
        </div>
      </details>`
  }
  return `
    <div class="row">
      <button class="icon-btn" data-act="back" title="Back to the page">${icon('back')}</button>
      <div class="grow">
        <div style="font-weight:600" class="ellipsis">Anonymized: ${esc(a.title || hostOf(a.url))}</div>
        <div class="dim ellipsis">${a.source === 'selection' ? 'The selection' : 'The page'} · vault ${esc(a.vault)}</div>
      </div>
    </div>
    ${body}
    <div class="foot">Captain for Chrome ${esc(VERSION)}</div>`
}

function selectedFiles() {
  return allFiles().filter((f) => S.checked.has(f.url))
}

// ── agents using this browser (hands.js) ──

const HANDS_RECENT_MS = 2 * 60 * 1000

function agoText(at) {
  const s = Math.max(0, Math.round((Date.now() - at) / 1000))
  return s < 10 ? 'now' : s < 60 ? `${s}s ago` : `${Math.round(s / 60)} min ago`
}

// An agent at work here is known by its deck and slug ("<deck>|<slug>", as
// hands.js keeps it); a stop kept from before decks is a bare slug.
const handsKey = (e) => e.key || e.slug
const keyDeck = (key) => (key.includes('|') ? key.slice(0, key.lastIndexOf('|')) : '')
const keySlug = (key) => key.slice(key.lastIndexOf('|') + 1)

/** "Researcher", or "Researcher · Home" while more than one deck is connected. */
function handsName(key, name = '') {
  const deck = keyDeck(key)
  const slug = keySlug(key)
  const own = !deck || deck === S.conn?.url
  name = name || (own && agentDef(slug)?.name) || [...S.handsLog].reverse().find((e) => handsKey(e) === key)?.name || slug
  return S.decks.length > 1 && deck ? `${name} · ${deckShort(deck)}` : name
}

const isHandsStopped = (e) => S.handsStopped.includes(handsKey(e)) || S.handsStopped.includes(e.slug)

/** What's in the way of the agents' socket to a deck ('' = nothing). */
function handsStatusFor(st) {
  const state = st?.state
  if (state === 'on' || state === 'search' || state === 'off') return ''
  if (state === 'permission') return 'Reload Captain for Chrome in chrome://extensions to finish its update (it needs the debugger permission).'
  if (state === 'denied') return st.error || 'Flight Deck refused this browser — connect it again.'
  if (state === 'connecting') return st.error || 'Connecting to Flight Deck…'
  if (state === 'error') return st.error || 'Not connected.'
  return 'Starting…'
}

/** The same for every connected deck, each named once there's more than one. */
function handsStatusText() {
  const lines = S.decks.map((d) => {
    const text = handsStatusFor(S.handsStatus?.[d.url])
    return text && (S.decks.length > 1 ? `${deckShort(d.url)}: ${text}` : text)
  }).filter(Boolean)
  return lines.join(' · ')
}

function viewHands() {
  if (!S.hands.on) return ''
  const now = Date.now()
  const latest = new Map()
  for (const e of S.handsLog) if (now - e.at < HANDS_RECENT_MS) latest.set(handsKey(e), e)
  const working = [...latest.values()].filter((e) => !isHandsStopped(e)).reverse()
  const status = handsStatusText()
  const rows = working.map((e) => `
    <div class="hands-agent">
      <span class="status-pill ${e.state === 'working' ? 'working' : e.state === 'failed' ? 'error' : 'done'}"><span class="dot"></span>${esc(handsName(handsKey(e), e.name))}</span>
      <div class="grow ellipsis" title="${esc(e.error || e.text)}">${esc(e.text)}${e.state === 'failed' ? ' — didn\'t work' : ''} <span class="dim">· ${esc(agoText(e.at))}</span></div>
      <button class="link-btn" data-act="hands-show" data-slug="${esc(handsKey(e))}">Show</button>
      <button class="link-btn danger" data-act="hands-stop" data-slug="${esc(handsKey(e))}">Stop</button>
    </div>`).join('')
  const stoppedRows = S.handsStopped.map((key) => `
    <div class="hands-agent"><div class="grow dim">${esc(handsName(key))} is stopped — it can't use this browser.</div>
      <button class="link-btn" data-act="hands-let" data-slug="${esc(key)}">Let it again</button></div>`).join('')
  const past = S.handsLog.slice(-15).reverse()
  const history = past.length ? `
    <details class="hands-log">
      <summary class="dim">What agents did here</summary>
      ${past.map((e) => `<div class="step ${e.state === 'failed' ? 'err' : ''}"><b>${esc(handsName(handsKey(e), e.name))}</b>&nbsp;${esc(e.text)}${e.state === 'failed' ? ` — ${esc(e.error || 'failed')}` : ''} <span class="dim">· ${esc(agoText(e.at))}</span></div>`).join('')}
    </details>` : ''
  return `
    <div class="card hands ${working.length ? 'active' : ''}">
      <div class="row">${icon('globe', 'sm')}<div class="grow"><b>${working.length ? 'An agent is using this browser' : 'Agents can use this browser'}</b>
        ${status ? `<div class="dim">${esc(status)}</div>` : ''}</div>
        ${Object.values(S.handsStatus || {}).some((st) => st?.state === 'permission') ? '<button class="btn" data-act="hands-extensions" style="padding:3px 9px">Open extensions</button>' : ''}</div>
      ${rows}${stoppedRows}${history}
    </div>`
}

async function handsShow(key) {
  const got = await chrome.storage.session.get('handsGroups').catch(() => ({}))
  const g = got.handsGroups?.[key]
  if (!g) { toast('Its tabs are gone.'); return }
  let tabId = g.current
  if (!tabId) {
    const tabs = await chrome.tabs.query({ groupId: g.groupId }).catch(() => [])
    tabId = tabs[0]?.id
  }
  if (!tabId) { toast('Its tabs are gone.'); return }
  const tab = await chrome.tabs.update(tabId, { active: true }).catch(() => null)
  if (tab) chrome.windows.update(tab.windowId, { focused: true }).catch(() => {})
}

async function setHandsStopped(list) {
  S.handsStopped = [...new Set(list)]
  await setStore({ handsStopped: S.handsStopped })
  render()
}

async function turnHandsOn(want) {
  if (want && !(chrome.debugger && chrome.tabGroups)) {
    toast('Reload Captain for Chrome in chrome://extensions first — this update needs it.')
    render()
    return
  }
  S.hands = { ...S.hands, on: want }
  await setStore({ hands: S.hands })
  render()
}

// ── the agents' web searches, run here (search.js) ──
//
// Flight Deck says which engine the person picked (/fd/clip/me → search); the
// panel asks Chrome, once, to let Captain read that engine's result pages.

function searchPick() {
  const id = S.me && 'search' in S.me ? (S.me.search?.engine || '') : S.searchPickId
  return SEARCH_ENGINES[id] ? { id, ...SEARCH_ENGINES[id] } : null
}

async function refreshSearch() {
  S.search = await loadSearchPrefs()
  S.searchPickId = await loadSearchPick(S.conn?.url)
  S.searchServed = await servedEngines(S.conn?.url)
}

function searchWhat(eng) {
  return `Your agents' web searches open ${esc(eng.name)} in a background tab here — never in front of you — and the tab closes once the results are in. ` +
    `They show in this browser's history${eng.id === 'google' || eng.id === 'bing' ? `, and in your ${esc(eng.name)} activity if you're signed in` : ''}. ` +
    "When Chrome is closed, Flight Deck's own search engine answers instead."
}

function viewSearchAsk() {
  const eng = searchPick()
  if (!eng || S.searchServed.includes(eng.id) || S.search.declined === eng.id) return ''
  return `
    <div class="card">
      <div class="row">${icon('globe', 'sm')}<div class="grow"><b>Run your agents' searches on ${esc(eng.name)} here?</b>
        <div class="dim">${searchWhat(eng)} Chrome asks you once to let Captain read ${esc(eng.host)}.</div></div></div>
      <div class="row" style="margin-top:8px;gap:8px">
        <button class="btn primary" data-act="search-allow" data-engine="${esc(eng.id)}">Allow</button>
        <button class="btn" data-act="search-decline" data-engine="${esc(eng.id)}">Not in this browser</button>
      </div>
    </div>`
}

function viewSearchSettings() {
  const eng = searchPick()
  if (!eng) {
    return `<div class="card"><div class="label">Web searches</div>
      <div class="dim">Your agents can search the web here instead of through a paid search API. Turn it on in Flight Deck → Profile → Web searches, then come back here to allow it.</div></div>`
  }
  const on = S.searchServed.includes(eng.id)
  return `<div class="card"><div class="label">Web searches</div>
    <label class="switch"><input type="checkbox" id="searchOn" data-engine="${esc(eng.id)}" ${on ? 'checked' : ''}><div>Run my agents' web searches here, on ${esc(eng.name)}<div class="dim">${searchWhat(eng)}</div></div></label>
    ${on && handsStatusFor(S.handsStatus?.[S.conn?.url]) ? `<div class="dim" style="margin-top:4px">${esc(handsStatusFor(S.handsStatus[S.conn.url]))}</div>` : ''}
  </div>`
}

// Chrome's prompt needs the click itself: nothing is awaited before it.
function allowSearch(id) {
  const eng = SEARCH_ENGINES[id]
  if (!eng) return
  chrome.permissions.request({ origins: eng.origins }).catch(() => false).then(async (ok) => {
    if (!ok) {
      toast(`Not allowed — your agents keep searching with Flight Deck's own engine.`)
      render()
      return
    }
    const p = await loadSearchPrefs()
    await saveSearchPrefs({ on: true, engines: [...new Set([...p.engines, id])], declined: '' })
    await refreshSearch()
    toast(`Your agents' searches run on ${eng.name} here now`)
    render()
  })
}

// Off here (the card's "Not in this browser" or the switch): declined in this
// browser — even with Chrome letting Captain read the site anyway.
async function stopSearch(id) {
  const p = await loadSearchPrefs()
  const engines = p.engines.filter((e) => e !== id)
  await saveSearchPrefs({ on: engines.length > 0, engines, declined: id })
  await refreshSearch()
  render()
}

// ── pages read here for the agents' web_fetch (pages.js) ──
//
// Flight Deck says whether the person turned it on (/fd/clip/me → fetch); this
// browser reads pages while Chrome lets Captain read every site.

async function refreshFetch() {
  S.fetch = await loadFetchMode(S.conn?.url)
  S.fetchServed = await readsPages(S.conn?.url)
}

const FETCH_WHAT = "Your agents' web_fetch opens pages here, signed in as you, in a background tab that closes once the page is read — " +
  'so sites that turn Flight Deck away (LinkedIn, Facebook, sign-in walls, bot checks) open for them. ' +
  "Never Flight Deck itself or addresses on your own network. When Chrome is closed, they fetch as before."

function viewFetchAsk() {
  const f = S.fetch
  if (!f.mode || S.fetchServed || f.declined) return ''
  return `
    <div class="card">
      <div class="row">${icon('globe', 'sm')}<div class="grow"><b>Let your agents read web pages here?</b>
        <div class="dim">${esc(FETCH_WHAT)} It needs Captain to work on every site — Chrome asks you once.</div></div></div>
      <div class="row" style="margin-top:8px;gap:8px">
        <button class="btn primary" data-act="fetch-allow">Allow</button>
        <button class="btn" data-act="fetch-decline">Not in this browser</button>
      </div>
    </div>`
}

function viewFetchSettings() {
  const f = S.fetch
  if (!f.mode) {
    return `<div class="card"><div class="label">Web pages</div>
      <div class="dim">Your agents can read pages here, signed in as you — for sites that turn Flight Deck away. Turn it on in Flight Deck → Profile → Web pages.</div></div>`
  }
  const when = f.mode === 'always' ? 'every page they read' : 'pages that turn Flight Deck away'
  return `<div class="card"><div class="label">Web pages</div>
    <label class="switch"><input type="checkbox" id="fetchOn" ${S.fetchServed ? 'checked' : ''}><div>Read web pages for my agents here — ${esc(when)}<div class="dim">${esc(FETCH_WHAT)}${S.allSites ? '' : ' Turning it on lets Captain work on every site.'}</div></div></label>
  </div>`
}

// Chrome's prompt needs the click itself: nothing is awaited before it.
function allowFetch() {
  chrome.permissions.request({ origins: ['<all_urls>'] }).catch(() => false).then(async (ok) => {
    if (!ok) { toast('Not allowed — your agents fetch pages as before.'); render(); return }
    await setFetchDeclined(false)
    await refreshAllSites()
    await refreshFetch()
    toast('Your agents can read pages here now')
    render()
  })
}

async function stopFetch() {
  await setFetchDeclined(true)
  await refreshFetch()
  render()
}

// A new agent at work here becomes the panel's agent: talk to it, follow it.
function followHands() {
  const last = S.handsLog[S.handsLog.length - 1]
  if (!last || last.at <= S.handsSeen) return
  // A page read in the background for web_fetch isn't an agent at work here.
  if (last.action === 'fetch') { S.handsSeen = last.at; return }
  // Only the first step of a stretch of work: picking another agent meanwhile sticks.
  const fresh = !S.handsLog.some((e) => e !== last && handsKey(e) === handsKey(last) && last.at - e.at < HANDS_RECENT_MS)
  S.handsSeen = last.at
  // An agent of another deck works here all the same: the panel stays on the deck in use.
  if (last.deck && last.deck !== S.conn?.url) return
  if (!fresh || last.slug === S.agent || S.sending || S.view === 'run' || !agentDef(last.slug)) return
  S.agent = last.slug
  S.lensMore = false
  keep({ agent: S.agent })
  keepLensChoice()
  loadLenses(S.agent)
  toast(`${last.name} is using your browser`)
}

setInterval(() => {
  if (S.view === 'main' && S.hands.on && S.handsLog.some((e) => Date.now() - e.at < HANDS_RECENT_MS + 20000)) render()
}, 15000)

// ── sending ──

function onSend() {
  // Asked first, in the click itself: the sites the ticked files come from.
  const files = selectedFiles()
  const pageOrigin = originOf(S.tab?.url)
  const origins = [...new Set(files.map((f) => originOf(f.url)).filter((o) => o && o !== pageOrigin))].map((o) => `${o}/*`)
  const asked = origins.length && !S.allSites
    ? chrome.permissions.request({ origins }).catch(() => false)
    : Promise.resolve(true)
  send(asked)
}

async function inPageFetch(tabId, url, max) {
  try {
    const [res] = await chrome.scripting.executeScript({
      target: { tabId },
      func: async (u, limit) => {
        try {
          const r = await fetch(u, { credentials: 'include' })
          if (!r.ok) return { ok: false, error: `HTTP ${r.status}` }
          const len = Number(r.headers.get('content-length') || 0)
          if (len > limit) return { ok: false, error: 'too big', tooBig: true, size: len }
          const blob = await r.blob()
          if (blob.size > limit) return { ok: false, error: 'too big', tooBig: true, size: blob.size }
          const b64 = await new Promise((resolve, reject) => {
            const fr = new FileReader()
            fr.onload = () => resolve(String(fr.result).split(',')[1] || '')
            fr.onerror = () => reject(fr.error)
            fr.readAsDataURL(blob)
          })
          return { ok: true, b64, type: blob.type || r.headers.get('content-type') || '',
                   disposition: r.headers.get('content-disposition') || '', size: blob.size }
        } catch (e) {
          return { ok: false, error: String((e && e.message) || e) }
        }
      },
      args: [url, max],
    })
    const r = res?.result
    if (!r || !r.ok) return r || { ok: false, error: 'no answer from the page' }
    const blob = await (await fetch(`data:${r.type || 'application/octet-stream'};base64,${r.b64}`)).blob()
    return { ok: true, blob, type: r.type, disposition: r.disposition }
  } catch (e) {
    return { ok: false, error: String(e?.message || e) }
  }
}

async function directFetch(url, max) {
  try {
    const r = await fetch(url, { credentials: 'include' })
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` }
    const len = Number(r.headers.get('content-length') || 0)
    if (len > max) return { ok: false, error: 'too big', tooBig: true, size: len }
    const blob = await r.blob()
    if (blob.size > max) return { ok: false, error: 'too big', tooBig: true, size: blob.size }
    return { ok: true, blob, type: blob.type || r.headers.get('content-type') || '', disposition: r.headers.get('content-disposition') || '' }
  } catch (e) {
    return { ok: false, error: String(e?.message || e) }
  }
}

const TYPE_EXT = {
  'application/pdf': 'pdf', 'text/csv': 'csv', 'application/zip': 'zip', 'application/json': 'json',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
  'application/msword': 'doc', 'application/vnd.ms-excel': 'xls', 'application/vnd.ms-powerpoint': 'ppt',
  'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif', 'text/plain': 'txt',
}

function fileName(f, got) {
  let name = ''
  const disp = got.disposition || ''
  const star = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(disp)
  const plain = /filename\s*=\s*"?([^";]+)"?/.exec(disp)
  try { name = star ? decodeURIComponent(star[1]) : plain ? plain[1] : '' } catch { name = plain ? plain[1] : '' }
  name = (name || f.name || 'file').replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_').trim() || 'file'
  const type = String(got.type || '').split(';')[0].trim().toLowerCase()
  if (!/\.[a-z0-9]{1,6}$/i.test(name) && TYPE_EXT[type]) name += `.${TYPE_EXT[type]}`
  return name.slice(0, 120)
}

async function fetchFile(f, tab, max) {
  const same = originOf(f.url) === originOf(tab.url)
  const tries = same ? ['page', 'direct'] : ['direct', 'page']
  let why = ''
  for (const how of tries) {
    const got = how === 'page' ? await inPageFetch(tab.id, f.url, max) : await directFetch(f.url, max)
    if (got.ok) {
      const type = String(got.type || '').toLowerCase()
      if (type.includes('text/html') && !/\.html?$/i.test(f.name)) {
        why = 'got a web page instead of the file (a sign-in or download page?)'
        continue
      }
      return { ok: true, blob: got.blob, name: fileName(f, got) }
    }
    why = got.tooBig ? `over ${human(max)}` : got.error
    if (got.tooBig) break
  }
  return { ok: false, why: why || 'could not download it' }
}

function uniqueName(name, taken) {
  const fold = (s) => s.toLowerCase()
  if (!taken.has(fold(name))) { taken.add(fold(name)); return name }
  const m = /^(.*?)(\.[^.]+)?$/.exec(name)
  for (let n = 2; ; n++) {
    const cand = `${m[1]} (${n})${m[2] || ''}`
    if (!taken.has(fold(cand))) { taken.add(fold(cand)); return cand }
  }
}

const csvCell = (v) => {
  const s = String(v ?? '')
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
const yamlStr = (s) => JSON.stringify(String(s || ''))

function pageMarkdown(cap, saved, unfetched, unticked, tablesSent) {
  const m = cap.meta || {}
  const fm = ['---', `title: ${yamlStr(m.title)}`, `url: ${m.url}`]
  if (m.canonical && m.canonical !== m.url) fm.push(`canonical: ${m.canonical}`)
  if (m.site_name) fm.push(`site: ${yamlStr(m.site_name)}`)
  if (m.author) fm.push(`author: ${yamlStr(m.author)}`)
  if (m.published) fm.push(`published: ${yamlStr(m.published)}`)
  if (m.lang) fm.push(`language: ${m.lang}`)
  if (cap.structured_types?.length) fm.push(`types: ${yamlStr(cap.structured_types.join(', '))}`)
  fm.push(`captured: ${new Date().toISOString()}`, `captured_with: Captain for Chrome ${VERSION}`, '---')
  const out = [fm.join('\n'), '']
  // The page's own heading, when it opens with one, is the title.
  let body = cap.markdown || ''
  const own = /^#\s[^\n]*/.exec(body)
  out.push(own ? own[0] : `# ${m.title || m.url}`, '')
  if (own) body = body.slice(own[0].length).replace(/^\n+/, '')
  if (m.description) out.push(`> ${m.description}`, '')
  if (cap.pdf) out.push('This page is a PDF document — it is in files/.', '')
  out.push(body || '(No readable text on this page — look at screenshot.png or page.mhtml when they were sent.)')
  const fileLines = [
    ...saved.map((f) => `- [${f.label}](${f.url}) — saved as files/${f.name}`),
    ...unfetched.map((f) => `- [${f.name}](${f.url}) — not downloaded: ${f.why}`),
    ...unticked.map((f) => `- [${f.name}](${f.url}) — ${f.kind}, not downloaded (not picked)`),
  ]
  if (fileLines.length) out.push('', '## Files on this page', '', ...fileLines)
  if (tablesSent.length) {
    out.push('', '## Tables', '', ...tablesSent.map((t) => `- tables/${t.name} — ${t.caption || 'untitled'} (${t.rows} rows × ${t.cols} columns)`))
  }
  if (cap.structured) out.push('', '## Structured data (JSON-LD)', '', '```json', cap.structured, '```')
  if (cap.links?.length) out.push('', '## Links on this page', '', ...cap.links.map((l) => `- [${(l.text || l.url).replace(/[[\]]/g, '')}](${l.url})`))
  return out.join('\n') + '\n'
}

async function send(asked) {
  const a = actionDef()
  const ag = agentDef()
  S.sendError = ''
  if (!ag) { S.sendError = 'Pick an agent first.'; render(); return }
  const note = (document.getElementById('note')?.value ?? S.note).trim()
  S.note = note
  if (a.needs_text && !note) { S.sendError = 'Write your question first.'; render(); return }
  const tab = S.tab
  S.sending = { text: 'Reading the page…', progress: null }
  render()
  const granted = await asked
  try {
    let cap = S.scan
    try {
      const fresh = await capture(tab.id)
      if (fresh && fresh.ok) cap = fresh
    } catch { /* keep what was read when the panel opened */ }
    if (!cap) throw new Error("Captain couldn't read this page.")
    const limits = S.me?.limits || {}
    const maxFile = limits.file_bytes || 50 * 1024 * 1024
    let budget = (limits.clip_bytes || 150 * 1024 * 1024) - 4 * 1024 * 1024
    const wanted = allFiles(cap).filter((f) => S.checked.has(f.url))
    const unticked = allFiles(cap).filter((f) => !S.checked.has(f.url) && f.source !== 'img')
    const saved = []
    const unfetched = []
    const taken = new Set()
    let done = 0
    const queue = wanted.slice(0, limits.files || 60)
    for (const f of wanted.slice(limits.files || 60)) unfetched.push({ name: f.name, url: f.url, why: 'over the number of files one clip takes' })
    const workers = Array.from({ length: Math.min(FETCH_PARALLEL, queue.length) }, async () => {
      while (queue.length) {
        const f = queue.shift()
        S.sending = { text: `Downloading ${f.name}… (${done + 1} of ${wanted.length})`, progress: wanted.length ? done / wanted.length : null }
        render()
        const got = await fetchFile(f, tab, Math.min(maxFile, Math.max(0, budget)))
        done++
        if (!got.ok) {
          unfetched.push({ name: f.name, url: f.url, why: got.why + (granted === false && originOf(f.url) !== originOf(tab.url) ? ' (that site was not allowed)' : '') })
          continue
        }
        if (got.blob.size > budget) {
          unfetched.push({ name: f.name, url: f.url, why: "over the clip's size limit" })
          continue
        }
        budget -= got.blob.size
        saved.push({ name: uniqueName(got.name, taken), blob: got.blob, url: f.url, label: f.name })
      }
    })
    await Promise.all(workers)

    let mhtml = null
    const notes = []
    if (S.include.snapshot) {
      S.sending = { text: 'Saving the whole page…', progress: null }
      render()
      try {
        mhtml = await chrome.pageCapture.saveAsMHTML({ tabId: tab.id })
        if (mhtml && mhtml.size > Math.min(maxFile, budget)) { mhtml = null; notes.push('the whole page was too big to send') }
        else if (mhtml) budget -= mhtml.size
      } catch (e) {
        notes.push('the whole page could not be saved')
      }
    }
    let shot = null
    if (S.include.screenshot) {
      try {
        const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' })
        shot = await (await fetch(dataUrl)).blob()
      } catch {
        toast('No screenshot: turn on “Work on every site” in settings, or open Captain from its toolbar button on this page.')
      }
    }

    const tablesSent = []
    const form = new FormData()
    if (S.include.tables && cap.tables?.length) {
      cap.tables.forEach((t, i) => {
        const name = `table-${i + 1}.csv`
        const csv = t.rows.map((r) => r.map(csvCell).join(',')).join('\r\n') + '\r\n'
        form.append('table', new Blob([csv], { type: 'text/csv' }), name)
        tablesSent.push({ name, caption: t.caption, rows: t.rows.length, cols: t.width })
      })
    }
    const pageMd = pageMarkdown(cap, saved, unfetched, unticked, tablesSent)
    const m = cap.meta || {}
    const meta = {
      ...(a.id === 'watch' ? { watch: { every: watchEvery(), tz: browserZone() } } : {}),
      agent: ag.slug, action: a.id, perspective: S.perspective, perspective_label: lensDef()?.label || '', perspective_text: S.perspectiveText, note,
      selection_only: !!(S.focus?.kind === 'selection' && cap.selection && S.include.selection),
      browser: browserLabel(), version: VERSION, captured_at: new Date().toISOString(),
      page: { url: m.url || tab.url, title: m.title || tab.title || '', site_name: m.site_name || '', author: m.author || '',
              published: m.published || '', description: m.description || '', lang: m.lang || '', words: cap.words || 0,
              pdf: !!cap.pdf, canonical: m.canonical || '' },
      unfetched,
      notes,
    }
    form.append('meta', new Blob([JSON.stringify(meta)], { type: 'application/json' }), 'meta.json')
    form.append('page', new Blob([pageMd], { type: 'text/markdown' }), 'page.md')
    if (cap.selection && S.include.selection) {
      form.append('selection', new Blob([`# Selected on ${m.title || tab.url}\n\n${cap.selection.markdown}\n`], { type: 'text/markdown' }), 'selection.md')
    }
    if (mhtml) form.append('mhtml', mhtml, 'page.mhtml')
    if (shot) form.append('screenshot', shot, 'screenshot.png')
    for (const f of saved) form.append('file', f.blob, f.name)

    S.sending = { text: `Sending to ${ag.name}…`, progress: 0 }
    render()
    const res = await upload(S.conn, '/fd/clip/send', form, (loaded, total) => {
      S.sending = { text: `Sending to ${ag.name}… ${human(loaded)} of ${human(total)}`, progress: total ? loaded / total : null }
      render()
    })
    startRun({ clip_id: res.clip_id, folder: res.folder, vfs: res.vfs, agent: res.agent || ag.name, agentSlug: ag.slug,
               title: m.title || tab.title || tab.url, url: m.url || tab.url, action: a.id, actionLabel: a.label,
               sentFiles: saved.length, unfetched: unfetched.length, watch: res.watch || null })
    S.sending = null
    S.note = ''
    S.focus = null
  } catch (e) {
    S.sending = null
    if (e instanceof AuthError) return disconnected(e.message)
    S.sendError = e.message || String(e)
    render()
  }
}

// ── a run ──

// A run belongs to the deck it was sent to: it's followed there even after
// the panel switched to another deck.
function runConn(run) {
  return S.decks.find((d) => d.url === run?.deck) || (run?.deck ? null : S.conn)
}

function startRun(run) {
  S.run = { deck: S.conn?.url, ...run, status: 'starting', events: [], next: 0, result: '', live: '', error: '', files: [], done: false, at: Date.now() }
  watchAsks()
  S.followUp = ''
  S.follow = true
  S.view = 'run'
  remember(S.run)
  render()
  pollRun(S.run)
}

async function pollRun(run) {
  if (run.polling) return
  run.polling = true
  let misses = 0
  let drawn = ''
  while (S.run === run && !run.done) {
    const conn = runConn(run)
    if (!conn) {   // its deck was disconnected
      run.done = true
      run.lost = true
      run.status = run.status === 'done' ? 'done' : 'lost'
      break
    }
    try {
      const r = await api(conn, `/fd/clip/runs/${encodeURIComponent(run.clip_id)}?since=${run.next}`)
      misses = 0
      run.status = r.status
      run.events.push(...(r.events || []))
      run.next = r.next ?? run.next
      // What it did last, for the working bar: a step, or writing the answer.
      if ((r.events || []).length) run.doing = 'step'
      else if ((r.live || '').length > (run.live || '').length) run.doing = 'writing'
      run.live = r.live || ''
      if (r.result) run.result = r.result
      if (r.error) run.error = r.error
      if (r.done) {
        run.done = true
        run.files = r.files || []
        remember(run)
      }
    } catch (e) {
      if (e instanceof AuthError) { run.polling = false; return disconnected(e.message, conn.url) }
      if (/over or unknown/i.test(e.message)) {
        run.done = true
        run.lost = true
        run.status = run.status === 'done' ? 'done' : 'lost'
        remember(run)
      } else if (++misses > 20) {
        run.error = e.message
        run.done = true
        run.status = 'error'
      }
    }
    // Drawn again only when something changed: a redraw every poll would
    // drop a selection in the answer and jump the lists.
    const now = [run.status, run.events.length, run.live.length, run.result.length, run.error, run.done].join('|')
    if (now !== drawn && S.view === 'run' && S.run === run) { drawn = now; render() }
    if (!run.done) await sleep(1200)
  }
  run.polling = false
  if (S.view === 'run' && S.run === run) render()
}

function elapsed(at) {
  const s = Math.max(0, Math.round((Date.now() - (at || Date.now())) / 1000))
  return s < 3600 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

// While the agent works the view follows it: the newest step and the answer
// as it's written stay in sight, the working bar at the bottom. Scrolling up
// stops that (the bar offers the way back); scrolling to the end starts it again.
function followRun() {
  const run = S.run
  if (!run || !S.follow) return
  if (!run.done) {
    window.scrollTo({ top: document.documentElement.scrollHeight })
    return
  }
  if (run.followedEnd) return
  run.followedEnd = true
  // Done: a long answer from its start, a short one with what comes after it.
  const card = document.getElementById('answerCard')
  if (card && card.offsetHeight > window.innerHeight - 140) card.scrollIntoView({ block: 'start' })
  else window.scrollTo({ top: document.documentElement.scrollHeight })
}

let userScrollAt = 0
for (const type of ['wheel', 'touchmove', 'mousedown', 'keydown']) {
  window.addEventListener(type, () => { userScrollAt = Date.now() }, { passive: true, capture: true })
}
window.addEventListener('scroll', () => {
  if (S.view !== 'run' || !S.run || S.run.done) return
  const atEnd = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 40
  const was = S.follow
  if (atEnd) S.follow = true
  else if (Date.now() - userScrollAt < 1000) S.follow = false
  if (was !== S.follow) render()
}, { passive: true })

setInterval(() => {
  if (S.view !== 'run' || !S.run || S.run.done) return
  const el = document.getElementById('elapsed')
  if (el) el.textContent = elapsed(S.run.at)
}, 1000)

function statusPill(run) {
  const map = {
    starting: ['working', 'Sending…'], thinking: ['working', 'Working…'], done: ['done', 'Done'],
    stopped: ['', 'Stopped'], error: ['error', 'Something went wrong'], lost: ['', 'Finished — answer in Flight Deck'],
  }
  const [cls, text] = map[run.status] || ['', run.status]
  return `<span class="status-pill ${cls}"><span class="dot"></span>${esc(text)}</span>`
}

function viewRun() {
  const run = S.run
  if (!run) return ''
  const steps = run.events.slice(-40).map((e) => `<div class="step">${esc(e.text)}</div>`).join('')
  const working = !run.done
  const answer = run.result
    ? `<div class="answer">${renderMarkdown(run.result)}</div>`
    : run.live ? `<div class="answer live">${esc(run.live)}</div>`
      : working ? '<div class="dim">The answer shows here as soon as the agent writes it.</div>' : ''
  const files = (run.files || []).map((f) => `
    <button class="out-file" data-file-path="${esc(f.path)}" title="Save a copy">
      ${icon(/^files\//.test(f.path) ? 'download' : 'file', 'sm')}
      <span class="grow"><span class="file-name">${esc(f.path)}</span></span>
      <span class="dim">${esc(human(f.size))}</span>
    </button>`).join('')
  return `
    <div class="row">
      <button class="icon-btn" data-act="back" title="Back to the page">${icon('back')}</button>
      <div class="grow">
        <div style="font-weight:600" class="ellipsis">${esc(run.actionLabel || 'Clip')}: ${esc(run.title)}</div>
        <div class="dim ellipsis">${esc([run.agent, hostOf(run.url)].filter(Boolean).join(' · '))}</div>
      </div>
    </div>
    ${viewWatchTask(run.watch)}
    <div class="row">${statusPill(run)}<span class="grow"></span>
      ${working ? `<button class="btn" data-act="stop" style="padding:4px 10px">${icon('stop', 'sm')} Stop</button>` : ''}
    </div>
    ${steps ? `<div class="card"><div class="label">Steps</div><div class="steps" id="steps">${steps}</div></div>` : ''}
    ${run.error ? `<div class="notice err">${esc(run.error)}</div>` : ''}
    <div class="card" id="answerCard">${answer || '<div class="dim">No answer.</div>'}</div>
    ${working ? `
      <div class="working-bar" id="workingBar">
        <div class="spinner"></div>
        <div class="grow ellipsis">${esc(run.doing === 'writing' ? `${run.agent} is writing the answer…`
          : run.events.length ? run.events[run.events.length - 1].text
            : run.status === 'starting' ? `Sending to ${run.agent}…` : `${run.agent} is working…`)}</div>
        <span class="elapsed" id="elapsed">${esc(elapsed(run.at))}</span>
        ${S.follow ? '' : `<button class="icon-btn" data-act="follow" title="Show the latest">${icon('chevron', 'sm')}</button>`}
      </div>` : ''}
    ${files ? `<div class="card"><div class="label">${run.action === 'recording' || run.action === 'playbook' ? 'In the run’s folder' : 'In the clip folder'}</div><div class="out-files">${files}</div>
      <div class="dim" style="margin-top:6px">${esc(run.vfs || '')}</div></div>` : ''}
    ${run.done ? `
      <div class="row">
        ${run.result ? `<button class="btn grow" data-act="copy">${icon('copy', 'sm')} Copy answer</button>` : ''}
        <button class="btn grow" data-act="open-deck">${icon('external', 'sm')} Open Flight Deck</button>
      </div>
      <div class="row" style="justify-content:center">${newTopicButton(run.agentSlug)}</div>
      <div>
        <textarea id="followUp" rows="2" placeholder="${run.action === 'recording' || run.action === 'playbook' ? 'Tell it what to do next, or what went wrong…' : 'Ask a follow-up about this page…'}">${esc(S.followUp)}</textarea>
        <button class="btn primary block" data-act="follow-up" style="margin-top:6px">${icon('send', 'sm')} Ask ${esc(run.agent)}</button>
      </div>` : ''}`
}

async function sendFollowUp() {
  const run = S.run
  const text = (document.getElementById('followUp')?.value || '').trim()
  if (!run || !text) return
  S.followUp = text
  const form = new FormData()
  const meta = {
    agent: run.agentSlug, action: 'ask', note: text, follow_up: { folder: run.folder },
    browser: browserLabel(), version: VERSION, captured_at: new Date().toISOString(),
    page: { url: run.url, title: run.title },
  }
  form.append('meta', new Blob([JSON.stringify(meta)], { type: 'application/json' }), 'meta.json')
  const conn = runConn(run)
  if (!conn) { toast('That Flight Deck isn\'t connected any more.'); return }
  try {
    const res = await upload(conn, '/fd/clip/send', form, () => {})
    startRun({ deck: conn.url, clip_id: res.clip_id, folder: res.folder, vfs: res.vfs, agent: res.agent || run.agent, agentSlug: run.agentSlug,
               title: run.title, url: run.url, action: run.action === 'recording' || run.action === 'playbook' ? run.action : 'ask', actionLabel: 'Follow-up' })
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message, conn.url)
    toast(e.message)
  }
}

async function saveRunFile(path) {
  const run = S.run
  try {
    const conn = runConn(run)
    if (!conn) throw new Error('That Flight Deck isn\'t connected any more.')
    const r = await api(conn, `/fd/clip/runs/${encodeURIComponent(run.clip_id)}/file?path=${encodeURIComponent(path)}`, { raw: true })
    const blob = await r.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = path.split('/').pop()
    document.body.appendChild(a)
    a.click()
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove() }, 5000)
  } catch (e) {
    toast(/over or unknown/i.test(e.message) ? 'Open it in Flight Deck → Shared folders → web-clips.' : e.message)
  }
}

// ── recordings: show a process once, an agent does it again ──
//
// Record → the worker (recorder.js) and the page script (recorder-page.js)
// list what the person does; here they add an instruction to any step, steps
// in their own words, and mark values that differ each time. Saved on Flight
// Deck (/fd/clip/recordings); Run hands it to an agent, which does it in this
// browser with its browser tool.

const ROLE_WORDS = {
  button: 'button', link: 'link', textbox: 'field', searchbox: 'search field', combobox: 'list', listbox: 'list',
  checkbox: 'checkbox', radio: 'option', switch: 'switch', tab: 'tab', menuitem: 'menu item', menuitemcheckbox: 'menu item',
  menuitemradio: 'menu item', option: 'option', treeitem: 'item', file: 'file field', img: 'picture', slider: 'slider',
  spinbutton: 'number field',
}

function targetText(t) {
  t = t || {}
  const word = ROLE_WORDS[t.role] || ''
  if (t.name) return word ? `the ${word} “${t.name}”` : `“${t.name}”`
  if (word) return `the ${word}${t.selector ? ` (${t.selector})` : ''}`
  return t.selector ? `the element ${t.selector}` : 'the element'
}

// The same sentences Flight Deck writes for the agent (browser_recordings.step_text).
function stepText(s) {
  const what = targetText(s.target)
  switch (s.kind) {
    case 'do': return s.text || ''
    case 'open': return `Open ${s.url}${s.new_tab ? ' in a new tab' : ''}`
    case 'switch': return `Switch to the tab “${s.title || s.url}”`
    case 'click': return `${s.double ? 'Double-click' : 'Click'} ${what}`
    case 'type':
      if (s.secret) return `Type the secret value into ${what} (not recorded)`
      if (askMode(s) === 'during') return `Type what you answer then into ${what}`
      if (askMode(s) === 'agent') return `Type what the agent judges right into ${what}`
      return s.value ? `Type “${s.value}” into ${what}` : `Clear ${what}`
    case 'select':
      if (askMode(s) === 'during') return `Choose what you answer then in ${what}`
      if (askMode(s) === 'agent') return `Choose what the agent judges right in ${what}`
      return `Choose “${s.value}” in ${what}`
    case 'check': return s.target?.role === 'radio' ? `Choose ${what}` : `${s.checked ? 'Tick' : 'Untick'} ${what}`
    case 'upload': return `Pick the file “${s.value}” for ${what}`
    case 'key': return `Press ${s.key}${s.target ? ` in ${what}` : ''}`
    default: return s.kind
  }
}

// Where a typed or chosen value comes from on a run.
const ASK_MODES = [['', 'Always the same'], ['before', 'Ask me before it starts'], ['during', 'Ask me when it gets there'], ['agent', 'The agent decides']]
const ASK_HINTS = {
  before: 'You fill it in on the Run form.',
  during: 'The agent stops at this step and asks you: a box opens here (and in Flight Deck) and it waits. Cancel stops the run there.',
  agent: 'The agent picks what fits the run and tells you afterwards what it used.',
}
const askMode = (s) => (s.ask === true ? 'before' : ASK_MODES.some(([id]) => id && id === s.ask) ? s.ask : '')

const shortUrl = (url) => { try { const u = new URL(url); return (u.host.replace(/^www\./, '') + (u.pathname === '/' ? '' : u.pathname)).slice(0, 60) } catch { return '' } }

const recSend = (msg) => chrome.runtime.sendMessage(msg).catch((e) => ({ ok: false, error: String(e?.message || e) }))

function recordable(tab = S.tab) {
  return !!tab && /^https?:/i.test(tab.url || '')
}

async function recStart() {
  const tab = (await currentTab()) || S.tab
  if (!recordable(tab)) { toast('Open the site you want to record first.'); return }
  if (S.draft && !S.draft.id && !confirm('You have a recording you haven\'t saved. Record a new one and drop it?')) return
  // Asked in the click itself: a recording keeps seeing this site, page after page.
  const asked = S.allSites ? Promise.resolve(true) : chrome.permissions.request({ origins: [`${originOf(tab.url)}/*`] }).catch(() => false)
  asked.then(async (ok) => {
    if (!ok) { toast(`Captain needs your OK to see what you do on ${hostOf(tab.url)}.`); return }
    const res = await recSend({ type: 'captain-rec-start', tabId: tab.id })
    if (!res?.ok) { toast(res?.error || "Couldn't start recording."); return }
    S.recNotes = {}
    S.recNoteOpen = new Set()
    setDraft(null)
    S.view = 'rec'
    render()
  })
}

let notesTimer = 0
function saveRecNotes() {
  clearTimeout(notesTimer)
  notesTimer = setTimeout(() => chrome.storage.session.set({ recNotes: S.recNotes }).catch(() => {}), 300)
}

let draftTimer = 0
function setDraft(draft, { now = false } = {}) {
  S.draft = draft
  clearTimeout(draftTimer)
  const write = () => (draft ? setStore({ recDraft: draft }) : chrome.storage.local.remove('recDraft').catch(() => {}))
  if (now || !draft) write()
  else draftTimer = setTimeout(write, 400)
}

async function recStop(discard) {
  const rec = S.rec
  if (discard && rec?.steps?.length > 1 && !confirm('Drop this recording?')) return
  if (!discard && rec) {
    const first = rec.steps.find((s) => s.url)
    const steps = rec.steps.map(({ at, tabId, ...s }) => ({ ...s, note: (S.recNotes[s.id] || '').trim() }))
    setDraft({ name: '', about: '', site: hostOf(first?.url || ''), steps }, { now: true })
  }
  await recSend({ type: 'captain-rec-stop', discard: true })
  S.rec = null
  S.recNotes = {}
  S.recNoteOpen = new Set()
  if (!discard && S.draft?.steps.length) {
    S.view = 'rec-edit'
    render()
    document.getElementById('recName')?.focus()
  } else {
    if (!discard) { setDraft(null); toast('Nothing was recorded.') }
    S.view = 'main'
    render()
    scheduleScan(0)
  }
}

function recStepRow(s, i, n, edit) {
  const note = edit ? (s.note || '') : (S.recNotes[s.id] || '')
  const open = !!note || S.recNoteOpen.has(s.id)
  const where = [s.kind === 'open' || s.kind === 'switch' || s.kind === 'do' ? '' : shortUrl(s.url), s.target?.near || '',
    s.frame ? 'in a frame' : '', s.leads_to ? `→ ${shortUrl(s.leads_to)}${s.new_tab ? ' (new tab)' : ''}` : ''].filter(Boolean).join(' · ')
  const canAsk = edit && (s.kind === 'type' || s.kind === 'select') && !s.secret
  const mode = askMode(s)
  const body = s.kind === 'do' && edit
    ? `<textarea id="do-${esc(s.id)}" data-step-do="${esc(s.id)}" rows="2" placeholder="What should happen here, in your own words…">${esc(s.text || '')}</textarea>`
    : `<div class="rstep-text">${esc(stepText(s))}</div>`
  return `
    <div class="rstep ${s.kind === 'do' ? 'own' : ''}">
      <span class="rstep-n" title="${esc(s.kind)}">${i + 1}</span>
      <div class="grow">
        ${body}
        ${where ? `<div class="dim ellipsis" title="${esc(where)}">${esc(where)}</div>` : ''}
        ${canAsk && s.kind === 'type' && !mode ? `<input type="text" id="val-${esc(s.id)}" data-step-val="${esc(s.id)}" value="${esc(s.value || '')}" placeholder="What to type" style="margin-top:4px">` : ''}
        ${canAsk ? `<label class="rstep-ask"><span>This value</span>
          <select class="select" data-step-ask="${esc(s.id)}">${ASK_MODES.map(([id, label]) => `<option value="${id}" ${mode === id ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select></label>
          ${mode ? `<div class="dim">${esc(ASK_HINTS[mode])}</div>` : ''}` : ''}
        ${s.kind === 'do' ? '' : open
          ? `<textarea id="note-${esc(s.id)}" class="rstep-note" data-step-note="${esc(s.id)}" rows="2" placeholder="Your instruction for this step — what to pick, when to skip it, what to check…">${esc(note)}</textarea>`
          : `<button class="link-btn" data-step-open="${esc(s.id)}">${icon('pencil', 'sm')} Add an instruction</button>`}
      </div>
      <div class="rstep-tools">
        ${edit ? `<button class="icon-btn" data-step-move="${esc(s.id)}" data-dir="-1" title="Move up" ${i ? '' : 'disabled'}>${icon('up', 'sm')}</button>
          <button class="icon-btn" data-step-move="${esc(s.id)}" data-dir="1" title="Move down" ${i < n - 1 ? '' : 'disabled'}>${icon('down', 'sm')}</button>` : ''}
        <button class="icon-btn" data-step-del="${esc(s.id)}" title="Remove this step">${icon('x', 'sm')}</button>
      </div>
    </div>`
}

function viewRec() {
  const rec = S.rec
  if (!rec) return '<div class="row dim"><div class="spinner"></div> Stopping…</div>'
  const rows = rec.steps.map((s, i) => recStepRow(s, i, rec.steps.length, false)).join('')
  return `
    <div class="row"><span class="status-pill rec"><span class="dot"></span>Recording</span><span class="grow"></span>
      <span class="dim" id="recElapsed">${esc(elapsed(rec.startedAt))}</span></div>
    <div class="dim">Do it on the site the way you always do. Every step shows up here — add an instruction to any of them, now or after you stop.</div>
    ${rec.blocked ? `<div class="notice">${icon('lock')}<div class="grow">Captain can't see what you do on <b>${esc(hostOf(rec.blocked))}</b> yet.
      <div style="margin-top:6px"><button class="btn" data-act="rec-allow">Allow on this site</button></div></div></div>` : ''}
    ${rec.full ? '<div class="notice">That\'s the most steps one recording holds — stop here and record the rest as a second one.</div>' : ''}
    <div class="rsteps" id="recSteps">${rows || '<div class="dim" style="padding:10px">Nothing yet — click something on the page.</div>'}</div>
    <div class="rec-bar">
      <button class="btn primary grow" data-act="rec-stop">${icon('stop', 'sm')} Stop and review</button>
      <button class="btn danger" data-act="rec-discard">Discard</button>
    </div>`
}

function viewRecEdit() {
  const d = S.draft
  if (!d) return '<div class="dim">Nothing to review.</div><button class="btn block" data-act="back">Back</button>'
  const rows = d.steps.map((s, i) => recStepRow(s, i, d.steps.length, true)).join('')
  const secrets = d.steps.filter((s) => s.secret).length
  return `
    <div class="row"><button class="icon-btn" data-act="rec-edit-back" title="Back">${icon('back')}</button>
      <div class="grow" style="font-weight:600">${d.id ? 'Edit the recording' : 'Review the recording'}</div></div>
    <div>
      <label class="label" for="recName">Name</label>
      <input type="text" id="recName" maxlength="120" placeholder="${esc(d.site ? `What you did on ${d.site}` : 'What this does')}" value="${esc(d.name || '')}">
    </div>
    <div>
      <label class="label" for="recAbout">What it's for</label>
      <textarea id="recAbout" rows="2" placeholder="Anything the agent should know for the whole thing — the goal, what a good result looks like, when to stop and ask (optional)">${esc(d.about || '')}</textarea>
    </div>
    <div class="label-row" style="margin:0"><span class="label" style="margin:0">Steps</span><span class="grow"></span><span class="dim">${plural(d.steps.length, 'step')}</span></div>
    <div class="rsteps edit" id="recSteps">${rows || '<div class="dim" style="padding:10px">No steps left.</div>'}</div>
    <button class="btn block" data-act="rec-add-step">${icon('plus', 'sm')} Add a step in your own words</button>
    ${secrets ? `<div class="dim">${icon('lock', 'sm')} ${secrets === 1 ? "A password or code you typed wasn't recorded" : "The passwords or codes you typed weren't recorded"} — the agent asks you when it gets there.</div>` : ''}
    ${S.draftError ? `<div class="notice err">${esc(S.draftError)}</div>` : ''}
    <div class="rec-bar">
      <button class="btn primary grow" data-act="rec-save" ${d.steps.length && !S.draftSaving ? '' : 'disabled'}>${icon('check', 'sm')} ${S.draftSaving ? 'Saving…' : 'Save'}</button>
      <button class="btn danger" data-act="rec-drop">${d.id ? 'Cancel' : 'Discard'}</button>
    </div>`
}

async function loadRecs() {
  if (!S.conn || S.conn.agent) return
  try {
    const r = await api(S.conn, '/fd/clip/recordings')
    S.recs = r.recordings || []
    S.recsError = ''
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    S.recsError = /404|answered 404/.test(e.message) ? 'This Flight Deck can\'t keep recordings yet — it needs an update.' : e.message
  }
  if (['recs', 'main'].includes(S.view)) render()
}

async function recSave() {
  const d = S.draft
  if (!d || S.draftSaving) return
  const name = (d.name || '').trim()
  if (!name) { S.draftError = 'Give it a name first.'; render(); document.getElementById('recName')?.focus(); return }
  S.draftSaving = true
  S.draftError = ''
  render()
  try {
    const steps = d.steps.filter((s) => s.kind !== 'do' || (s.text || '').trim())
    await api(S.conn, '/fd/clip/recordings', { method: 'POST', body: JSON.stringify({ id: d.id || undefined, name, about: d.about || '', steps }) })
    setDraft(null)
    S.draftSaving = false
    S.view = 'recs'
    render()
    toast('Saved')
    loadRecs()
  } catch (e) {
    S.draftSaving = false
    if (e instanceof AuthError) return disconnected(e.message)
    S.draftError = e.message
    render()
  }
}

async function recEdit(id) {
  try {
    const r = await api(S.conn, `/fd/clip/recordings/${encodeURIComponent(id)}`)
    setDraft({ id: r.id, name: r.name, about: r.about || '', site: r.site || '', steps: r.steps || [] }, { now: true })
    S.draftError = ''
    S.view = 'rec-edit'
    render()
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
    loadRecs()
  }
}

async function recDelete(id) {
  const rec = (S.recs || []).find((r) => r.id === id)
  if (!rec || !confirm(`Delete “${rec.name}”?`)) return
  try {
    await api(S.conn, `/fd/clip/recordings/${encodeURIComponent(id)}/delete`, { method: 'POST' })
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
  }
  loadRecs()
}

function viewRecs() {
  const recs = S.recs
  const items = (recs || []).map((r) => `
    <div class="card rec-item">
      <div style="font-weight:600" class="ellipsis" title="${esc(r.name)}">${esc(r.name)}</div>
      <div class="dim ellipsis">${esc([r.site, plural(r.steps, 'step'), r.runs ? `ran ${r.runs}×` : ''].filter(Boolean).join(' · '))}</div>
      ${r.about ? `<div class="dim rec-about">${esc(r.about)}</div>` : ''}
      <div class="row" style="margin-top:8px">
        <button class="btn primary grow" data-rec-run="${esc(r.id)}">${icon('play', 'sm')} Run</button>
        <button class="btn" data-rec-edit="${esc(r.id)}">Edit</button>
        <button class="btn danger" data-rec-del="${esc(r.id)}" title="Delete">${icon('trash', 'sm')}</button>
      </div>
    </div>`).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back" title="Back">${icon('back')}</button>
      <div class="grow" style="font-weight:600">Recordings</div>
      <button class="btn" data-act="rec-start" style="padding:4px 10px" title="Record what you do in the tab in front">${icon('record', 'sm')} Record</button></div>
    ${S.draft && !S.draft.id ? `<div class="notice"><div class="grow">A recording of ${plural(S.draft.steps.length, 'step')} isn't saved yet.</div><button class="btn" data-act="rec-review" style="padding:3px 9px">Review</button></div>` : ''}
    ${S.recsError ? `<div class="notice err">${esc(S.recsError)}</div>` : ''}
    ${recs === null && !S.recsError ? '<div class="row dim"><div class="spinner"></div> Loading…</div>' : ''}
    ${items || (recs && !recs.length ? `<div class="card"><div style="font-weight:600;margin-bottom:4px">Show it once</div>
      <div class="dim">Press Record, do the thing on the site the way you always do, and say what matters at each step. After that one of your agents does it for you — in this browser, signed in as you.</div></div>` : '')}`
}

function openRecRun(id) {
  const rec = (S.recs || []).find((r) => r.id === id)
  if (!rec) return
  S.recRun = { rec, note: '', inputs: Object.fromEntries((rec.asks || []).map((a) => [a.id, a.was || ''])), error: '', sending: false,
               data: null, files: [], runs: null, busy: false }
  S.view = 'rec-run'
  render()
  loadRecExtras(S.recRun)
}

// The recording's extra data and how its earlier runs went.
async function loadRecExtras(run) {
  const base = `/fd/clip/recordings/${encodeURIComponent(run.rec.id)}`
  const [data, runs] = await Promise.all([
    api(S.conn, `${base}/data`).catch(() => null),      // an older Flight Deck has neither
    api(S.conn, `${base}/runs`).catch(() => null),
  ])
  if (S.recRun !== run) return
  run.data = data ? data.items || [] : false
  run.runs = runs ? runs.runs || [] : false
  if (S.view === 'rec-run') render()
}

function pickFiles(multiple = true) {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = multiple
    input.onchange = () => resolve([...input.files])
    input.click()
  })
}

async function recAddData() {
  const run = S.recRun
  if (!run || run.busy) return
  const files = await pickFiles()
  if (!files.length || S.recRun !== run) return
  run.busy = true
  run.error = ''
  render()
  try {
    const form = new FormData()
    for (const f of files) form.append('file', f, f.name)
    const r = await upload(S.conn, `/fd/clip/recordings/${encodeURIComponent(run.rec.id)}/data`, form, () => {})
    run.data = r.items || []
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    run.error = e.message
  }
  run.busy = false
  if (S.recRun === run) render()
}

async function recDropData(itemId) {
  const run = S.recRun
  if (!run) return
  try {
    const r = await api(S.conn, `/fd/clip/recordings/${encodeURIComponent(run.rec.id)}/data/${encodeURIComponent(itemId)}/delete`, { method: 'POST' })
    run.data = r.items || []
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
  }
  if (S.recRun === run) render()
}

async function recRunFiles() {
  const run = S.recRun
  if (!run) return
  const files = await pickFiles()
  if (S.recRun !== run) return
  run.files = [...run.files, ...files].slice(0, 20)
  render()
}

const RUN_STATUS = { done: 'Done', error: 'Went wrong', stopped: 'Stopped', thinking: 'Working…', starting: 'Starting…', lost: 'Ended — answer in Flight Deck' }

// An earlier run, opened the way a finished clip is.
function openPastRun(id) {
  const r = (S.recRun?.runs || []).find((x) => x.id === id)
  if (!r) return
  const live = r.status === 'thinking' || r.status === 'starting'
  S.run = { deck: S.conn?.url, clip_id: r.id, folder: r.folder, vfs: r.vfs, agent: r.agent, agentSlug: r.agent_slug, title: r.name, url: S.recRun.rec.start_url || '',
            action: 'recording', actionLabel: 'Recording', status: r.status, events: (r.steps_done || []).map((text) => ({ text })), next: 0,
            result: r.result || '', live: '', error: r.error || '', files: [], done: !live, at: Date.parse(r.started_at) || Date.now(), followedEnd: true }
  S.follow = true
  S.view = 'run'
  render()
  if (live) { pollRun(S.run); watchAsks() }
}

function viewRecRun() {
  const run = S.recRun
  if (!run) return ''
  const rec = run.rec
  const agents = (S.me?.agents || []).map((ag) => {
    const up = ag.status === 'running' || ag.status === 'asleep'
    return `<option value="${esc(ag.slug)}" ${ag.slug === S.agent ? 'selected' : ''} ${up ? '' : 'disabled'}>${ag.status === 'running' ? '●' : ag.status === 'asleep' ? '◐' : '○'} ${esc(ag.name)}${up ? '' : ` (${esc(ag.status)})`}</option>`
  }).join('')
  const ag = agentDef()
  const asks = (rec.asks || []).map((a) => `
    <div>
      <label class="label" for="ask-${esc(a.id)}">${esc(a.label)}</label>
      ${a.options?.length
        ? `<select class="select" id="ask-${esc(a.id)}" data-ask-input="${esc(a.id)}">${[...new Set([...(a.options.includes(run.inputs[a.id]) || !run.inputs[a.id] ? [] : [run.inputs[a.id]]), ...a.options])].map((o) => `<option ${o === run.inputs[a.id] ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`
        : `<input type="text" id="ask-${esc(a.id)}" data-ask-input="${esc(a.id)}" value="${esc(run.inputs[a.id] ?? '')}" placeholder="${esc(a.was ? `When recorded: ${a.was}` : '')}">`}
    </div>`).join('')
  const dataRows = (run.data || []).map((d) => `
    <div class="data-row">${icon('file', 'sm')}<span class="grow ellipsis" title="${esc(`${d.name} · ${d.origin || ''}`)}">${esc(d.name)}</span>
      <span class="dim">${esc(human(d.size))}</span>
      <button class="icon-btn" data-rec-data-del="${esc(d.id)}" title="Take it off">${icon('x', 'sm')}</button></div>`).join('')
  const fileRows = run.files.map((f, i) => `
    <div class="data-row">${icon('file', 'sm')}<span class="grow ellipsis">${esc(f.name)}</span><span class="dim">${esc(human(f.size))}</span>
      <button class="icon-btn" data-rec-file-del="${i}" title="Take it off">${icon('x', 'sm')}</button></div>`).join('')
  const extras = run.data === false ? '' : `
    <details class="include card" ${run.dataOpen || run.files.length ? 'open' : ''} data-rec-extras>
      <summary><span class="grow"><b>Extra data</b> <span class="dim">— ${run.data === null ? '…' : [run.data.length ? plural(run.data.length, 'file') + ' every run' : '', run.files.length ? plural(run.files.length, 'file') + ' this run' : ''].filter(Boolean).join(', ') || 'none'}</span></span>${icon('chevron', 'sm chev')}</summary>
      <div style="margin-top:6px">
        <div class="dim">What ${esc(ag?.name || 'the agent')} reads before it starts: how you want it done, lists, rules, examples.</div>
        <div class="label" style="margin-top:8px">Every run</div>
        ${dataRows || '<div class="dim">Nothing yet.</div>'}
        <div class="links-row"><button data-act="rec-data-add" ${run.busy ? 'disabled' : ''}>${run.busy ? 'Adding…' : 'Add files'}</button></div>
        <div class="dim">Agents' files, Flight Deck files, pasted text and links: add them in Flight Deck → Connections → Browser.</div>
        <div class="label" style="margin-top:8px">This run only</div>
        ${fileRows || '<div class="dim">Nothing.</div>'}
        <div class="links-row"><button data-act="rec-run-files">Add files</button></div>
      </div>
    </details>`
  const past = (run.runs || []).slice(0, 6).map((r) => `
    <button class="history-item" data-rec-past="${esc(r.id)}">
      <span class="ellipsis"><b>${esc(RUN_STATUS[r.status] || r.status)}</b> <span class="dim">· ${esc(r.agent || '')} · ${esc(new Date(r.started_at).toLocaleString())}</span></span>
      ${r.inputs?.length || r.note ? `<span class="dim ellipsis">${esc([...(r.inputs || []).map((i) => `${i.label}: ${i.value}`), r.note].filter(Boolean).join(' · '))}</span>` : ''}
    </button>`).join('')
  const handsOk = S.hands.on && S.handsStatus?.[S.conn?.url]?.state !== 'denied'
  return `
    <div class="row"><button class="icon-btn" data-act="recs" title="Back">${icon('back')}</button>
      <div class="grow"><div style="font-weight:600" class="ellipsis">${esc(rec.name)}</div>
        <div class="dim ellipsis">${esc([rec.site, plural(rec.steps, 'step')].filter(Boolean).join(' · '))}</div></div></div>
    ${rec.about ? `<div class="dim">${esc(rec.about)}</div>` : ''}
    <div><label class="label" for="agent">Who does it</label><select class="select" id="agent">${agents}</select></div>
    ${asks}
    <div><textarea id="recRunNote" rows="2" placeholder="Anything different this time? (optional)">${esc(run.note)}</textarea></div>
    ${extras}
    ${rec.asks_during?.length ? `<div class="dim">${icon('ask', 'sm')} It will stop and ask you for ${esc(rec.asks_during.map((l) => `“${l}”`).join(', '))} when it gets there — keep this panel open.</div>` : ''}
    ${rec.agent_decides?.length ? `<div class="dim">${icon('sparkles', 'sm')} It decides ${esc(rec.agent_decides.map((l) => `“${l}”`).join(', '))} itself.</div>` : ''}
    ${handsOk ? '' : `<div class="notice">${icon('globe')}<div class="grow">Your agents can't use this browser yet — without it the agent has only a browser of its own, where you aren't signed in.
      <div style="margin-top:6px"><button class="btn" data-act="rec-hands-on">Let agents use this browser</button></div></div></div>`}
    ${run.error ? `<div class="notice err">${esc(run.error)}</div>` : ''}
    <button class="btn primary block" data-act="rec-run-go" ${ag && !run.sending ? '' : 'disabled'}>${icon('play', 'sm')} ${run.sending ? 'Starting…' : ag ? `Run with ${esc(ag.name)}` : 'Pick an agent'}</button>
    <div class="dim">${esc(ag?.name || 'The agent')} works in tabs of its own, grouped under its name — you can keep using yours. It stops and asks when a step doesn't fit what it sees.</div>
    ${past ? `<div class="label" style="margin-top:6px">Earlier runs</div>${past}` : ''}`
}

async function recRunGo() {
  const run = S.recRun
  const ag = agentDef()
  if (!run || !ag || run.sending) return
  run.sending = true
  run.error = ''
  render()
  try {
    const meta = { agent: ag.slug, note: run.note, inputs: run.inputs, browser: browserLabel(), hands: !!S.hands.on }
    const path = `/fd/clip/recordings/${encodeURIComponent(run.rec.id)}/run`
    let res
    if (run.files.length) {
      const form = new FormData()
      form.append('meta', new Blob([JSON.stringify(meta)], { type: 'application/json' }), 'meta.json')
      for (const f of run.files) form.append('file', f, f.name)
      res = await upload(S.conn, path, form, () => {})
    } else {
      res = await api(S.conn, path, { method: 'POST', body: JSON.stringify(meta) })
    }
    S.recRun = null
    startRun({ clip_id: res.clip_id, folder: res.folder, vfs: res.vfs, agent: res.agent || ag.name, agentSlug: ag.slug,
               title: run.rec.name, url: run.rec.start_url || '', action: 'recording', actionLabel: 'Recording' })
    loadRecs()
  } catch (e) {
    run.sending = false
    if (e instanceof AuthError) return disconnected(e.message)
    run.error = e.message
    render()
  }
}

function draftStep(id) {
  return S.draft?.steps.find((s) => s.id === id) || null
}

$view.addEventListener('click', (ev) => {
  const extras = ev.target.closest('details[data-rec-extras] > summary')
  if (extras && S.recRun) { S.recRun.dataOpen = !extras.parentElement.open; return }
  const el = ev.target.closest('[data-act^="rec"],[data-step-open],[data-step-del],[data-step-move],[data-rec-run],[data-rec-edit],[data-rec-del],[data-rec-data-del],[data-rec-file-del],[data-rec-past]')
  if (!el) return
  const d = el.dataset
  if (d.recDataDel) return void recDropData(d.recDataDel)
  if (d.recFileDel !== undefined && S.recRun) { S.recRun.files.splice(Number(d.recFileDel), 1); render(); return }
  if (d.recPast) return openPastRun(d.recPast)
  if (d.act === 'rec-data-add') return void recAddData()
  if (d.act === 'rec-run-files') return void recRunFiles()
  if (d.stepOpen) {
    S.recNoteOpen.add(d.stepOpen)
    render()
    document.getElementById(`note-${d.stepOpen}`)?.focus()
  } else if (d.stepDel) {
    if (S.view === 'rec') recSend({ type: 'captain-rec-remove', id: d.stepDel })
    else if (S.draft) { setDraft({ ...S.draft, steps: S.draft.steps.filter((s) => s.id !== d.stepDel) }); render() }
  } else if (d.stepMove && S.draft) {
    const steps = [...S.draft.steps]
    const i = steps.findIndex((s) => s.id === d.stepMove)
    const j = i + Number(d.dir)
    if (i < 0 || j < 0 || j >= steps.length) return
    ;[steps[i], steps[j]] = [steps[j], steps[i]]
    setDraft({ ...S.draft, steps })
    render()
  } else if (d.recRun) openRecRun(d.recRun)
  else if (d.recEdit) recEdit(d.recEdit)
  else if (d.recDel) recDelete(d.recDel)
  else if (d.act === 'rec-start') recStart()
  else if (d.act === 'recs') { S.view = 'recs'; render(); loadRecs() }
  else if (d.act === 'rec-show') { S.view = 'rec'; render() }
  else if (d.act === 'rec-stop') recStop(false)
  else if (d.act === 'rec-discard') recStop(true)
  else if (d.act === 'rec-allow' && S.rec?.blocked) {
    chrome.permissions.request({ origins: [`${S.rec.blocked}/*`] }).then((ok) => { if (ok) recSend({ type: 'captain-rec-retry' }) }).catch(() => {})
  } else if (d.act === 'rec-review' && S.draft) { S.view = 'rec-edit'; render() }
  else if (d.act === 'rec-edit-back') { S.view = S.draft?.id ? 'recs' : 'main'; if (S.draft?.id) setDraft(null); render(); if (S.view === 'main') scheduleScan(0) }
  else if (d.act === 'rec-add-step' && S.draft) {
    const id = `d${Date.now().toString(36)}`
    setDraft({ ...S.draft, steps: [...S.draft.steps, { id, kind: 'do', text: '' }] })
    render()
    document.getElementById(`do-${id}`)?.focus()
  } else if (d.act === 'rec-save') recSave()
  else if (d.act === 'rec-drop' && S.draft) {
    const saved = !!S.draft.id
    if (!saved && !confirm('Drop this recording?')) return
    setDraft(null)
    S.draftError = ''
    S.view = saved ? 'recs' : 'main'
    render()
    if (!saved) scheduleScan(0)
  } else if (d.act === 'rec-hands-on') turnHandsOn(true)
  else if (d.act === 'rec-run-go') recRunGo()
})

$view.addEventListener('input', (ev) => {
  const el = ev.target
  const d = el.dataset
  if (d.stepNote) {
    if (S.view === 'rec') { S.recNotes[d.stepNote] = el.value; saveRecNotes() }
    else { const s = draftStep(d.stepNote); if (s) { s.note = el.value; setDraft(S.draft) } }
  } else if (d.stepDo) { const s = draftStep(d.stepDo); if (s) { s.text = el.value; setDraft(S.draft) } }
  else if (d.stepVal) { const s = draftStep(d.stepVal); if (s) { s.value = el.value; setDraft(S.draft) } }
  else if (el.id === 'recName' && S.draft) { S.draft.name = el.value; setDraft(S.draft) }
  else if (el.id === 'recAbout' && S.draft) { S.draft.about = el.value; setDraft(S.draft) }
  else if (el.id === 'recRunNote' && S.recRun) S.recRun.note = el.value
  else if (d.askInput && S.recRun) S.recRun.inputs[d.askInput] = el.value
})

$view.addEventListener('change', (ev) => {
  const el = ev.target
  if (el.dataset.askInput && S.recRun) S.recRun.inputs[el.dataset.askInput] = el.value
  else if (el.dataset.stepAsk) {
    const s = draftStep(el.dataset.stepAsk)
    if (s) {
      if (el.value) s.ask = el.value
      else delete s.ask
      setDraft(S.draft)
      render()
    }
  } else if (el.dataset.stepVal) render()   // the sentence above it says the new words
})

setInterval(() => {
  if (S.view !== 'rec' || !S.rec) return
  const el = document.getElementById('recElapsed')
  if (el) el.textContent = elapsed(S.rec.startedAt)
}, 1000)

// ── an agent's question, answered here ──
//
// While a run is followed the agent may ask (ask_user): a value a recording
// leaves to the moment, a choice the page forces. Flight Deck holds the
// question (/fd/clip/asks) and shows it in its own tabs too; here it opens as
// a box over the panel and the agent waits. Cancel = "don't answer".

const $askLayer = document.getElementById('askLayer')
let asksWatching = false

async function watchAsks() {
  if (asksWatching) return
  asksWatching = true
  let misses = 0
  while (S.run && !S.run.done && runConn(S.run)) {
    try {
      const known = (S.asks || []).map((a) => a.id).join(',')
      const r = await api(runConn(S.run), `/fd/clip/asks?wait=20&known=${encodeURIComponent(known)}`)
      misses = 0
      showAsks(r.asks || [])
    } catch (e) {
      if (e instanceof AuthError || /answered 404/.test(e.message) || ++misses > 5) break   // an older Flight Deck asks in its own tabs only
      await sleep(3000)
    }
  }
  asksWatching = false
  showAsks([])
}

// The agent's suggested answer (ask_user's `suggested`, from the chat, its
// memory or the person's profile): a choice starts on it, marked; a text box
// shows it greyed and Tab writes it in, as in Flight Deck's dialog.
const hasSuggestion = (q) => q.suggested !== undefined && q.suggested !== null && q.kind !== 'secret'
const suggestedTag = '<span class="tag">suggested</span>'

function askField(q) {
  const id = `askq-${esc(q.id)}`
  const hint = hasSuggestion(q)
  if (q.kind === 'choice') {
    const start = q.default ?? (hint ? q.suggested : undefined)
    return q.choices.map((c, i) => `<label class="opt"><input type="radio" name="${id}" value="${esc(c)}" ${start === c || (start === undefined && i === 0 && !q.optional) ? 'checked' : ''}><div>${esc(c)}${hint && q.suggested === c ? ` ${suggestedTag}` : ''}</div></label>`).join('')
  }
  if (q.kind === 'multiple') {
    const marked = hint && Array.isArray(q.suggested) ? q.suggested : []
    return q.choices.map((c) => `<label class="opt"><input type="checkbox" name="${id}" value="${esc(c)}" ${(q.default || []).includes(c) ? 'checked' : ''}><div>${esc(c)}${marked.includes(c) ? ` ${suggestedTag}` : ''}</div></label>`).join('')
  }
  if (q.kind === 'confirm') {
    const start = q.default ?? (hint ? q.suggested : undefined)
    return `<label class="opt"><input type="radio" name="${id}" value="yes" ${start === true ? 'checked' : ''}><div>Yes${hint && q.suggested === true ? ` ${suggestedTag}` : ''}</div></label>
      <label class="opt"><input type="radio" name="${id}" value="no" ${start === false ? 'checked' : ''}><div>No${hint && q.suggested === false ? ` ${suggestedTag}` : ''}</div></label>`
  }
  if (q.kind === 'secret') return `<input type="password" name="${id}" autocomplete="off">`
  const ghost = hint ? ` placeholder="${esc(String(q.suggested))}" data-suggest="${esc(String(q.suggested))}"` : ''
  return q.lines > 1
    ? `<textarea name="${id}" rows="${Math.min(q.lines, 8)}"${ghost}>${esc(q.default || '')}</textarea>`
    : `<input type="text" name="${id}" value="${esc(q.default || '')}" autocomplete="off"${ghost}>`
}

function suggestionNote(q) {
  if (!hasSuggestion(q)) return ''
  const typed = !['choice', 'multiple', 'confirm'].includes(q.kind)
  const from = q.suggested_from ? ` — ${esc(q.suggested_from)}` : ''
  return `<div class="dim">${typed ? '<kbd>Tab</kbd> takes the suggestion' : 'Suggested'}${from}</div>`
}

// Drawn only when the questions change: a redraw would drop what's being typed.
function showAsks(asks) {
  const was = (S.asks || []).map((a) => a.id).join(',')
  S.asks = asks
  const a = asks[0]
  if (!a) { $askLayer.hidden = true; $askLayer.innerHTML = ''; return }
  if (was === asks.map((x) => x.id).join(',') && !$askLayer.hidden) return
  $askLayer.hidden = false
  $askLayer.innerHTML = `
    <form class="ask-box card" data-ask="${esc(a.id)}" role="dialog" aria-modal="true" aria-label="${esc(a.title || 'Your agent')} asks">
      <div class="label">${esc(a.agent_name || a.title || 'Your agent')} asks</div>
      ${a.questions.map((q) => `
        <div class="ask-q">
          <div class="ask-text">${esc(q.question)}${q.optional ? ' <span class="dim">(optional)</span>' : ''}</div>
          ${q.help ? `<div class="dim">${esc(q.help)}</div>` : ''}
          ${askField(q)}
          ${suggestionNote(q)}
        </div>`).join('')}
      <div class="notice err" id="askError" hidden></div>
      <div class="row">
        <button type="submit" class="btn primary grow">${icon('check', 'sm')} Answer</button>
        <button type="button" class="btn" data-ask-cancel>Cancel</button>
      </div>
      ${asks.length > 1 ? `<div class="dim">${plural(asks.length - 1, 'more question')} after this one</div>` : ''}
    </form>`
  $askLayer.querySelector('input:not([type=radio]):not([type=checkbox]), textarea, input')?.focus()
}

function askAnswers(form, a) {
  const out = {}
  for (const q of a.questions) {
    const name = `askq-${q.id}`
    if (q.kind === 'multiple') out[q.id] = [...form.querySelectorAll(`[name="${name}"]:checked`)].map((el) => el.value)
    else if (q.kind === 'choice' || q.kind === 'confirm') {
      const picked = form.querySelector(`[name="${name}"]:checked`)?.value
      if (picked !== undefined) out[q.id] = q.kind === 'confirm' ? picked === 'yes' : picked
    } else {
      const value = form.querySelector(`[name="${name}"]`)?.value ?? ''
      if (value !== '' || !q.optional) out[q.id] = value
    }
  }
  return out
}

async function sendAnswer(form, declined) {
  const a = (S.asks || []).find((x) => x.id === form.dataset.ask)
  if (!a || form.dataset.busy) return
  form.dataset.busy = '1'
  // The questions come from the deck of the run being followed.
  const conn = runConn(S.run) || S.conn
  try {
    await api(conn, `/fd/clip/asks/${encodeURIComponent(a.id)}`, {
      method: 'POST', body: JSON.stringify(declined ? { declined: true } : { answers: askAnswers(form, a) }) })
    showAsks(S.asks.filter((x) => x.id !== a.id))
  } catch (e) {
    delete form.dataset.busy
    if (e instanceof AuthError) return disconnected(e.message, conn.url)
    if (/isn't open any more/.test(e.message)) return showAsks(S.asks.filter((x) => x.id !== a.id))
    const box = form.querySelector('#askError')
    box.textContent = e.message
    box.hidden = false
  }
}

$askLayer.addEventListener('submit', (ev) => { ev.preventDefault(); sendAnswer(ev.target, false) })
$askLayer.addEventListener('click', (ev) => {
  if (ev.target.closest('[data-ask-cancel]')) sendAnswer(ev.target.closest('form'), true)
})
$askLayer.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) { ev.preventDefault(); sendAnswer(ev.target.closest('form'), false) }
  // Tab in an empty box with a suggestion: write it in, and on to the next question.
  const el = ev.target
  if (ev.key === 'Tab' && !ev.shiftKey && !ev.metaKey && !ev.ctrlKey && !ev.altKey && !ev.isComposing
      && el.dataset?.suggest && !el.value.trim()) {
    ev.preventDefault()
    el.value = el.dataset.suggest
    const qs = [...el.closest('form').querySelectorAll('.ask-q')]
    const next = qs.slice(qs.indexOf(el.closest('.ask-q')) + 1)
      .map((box) => box.querySelector('input:not([type=radio]):not([type=checkbox]), textarea, input')).find(Boolean)
    ;(next || el.closest('form').querySelector('button[type=submit]'))?.focus()
  }
})

// ── web playbooks: chips for the site in front, the store, a playbook ──
//
// Flight Deck serves them (/fd/clip/playbooks, browser_playbooks.py): the
// ones for the page's site, the store (categories, search, the person's
// shelf) and one playbook's form. Run hands it to the chosen agent, which
// runs it with its web_playbook tool; the panel follows that like any run.

const PB_HUE = { travel: 205, local: 152, shopping: 330, accounts: 35, work: 250, research: 188, money: 140, home: 18, media: 280, regional: 0 }
const PB_ICON = { travel: 'plane', local: 'pin', shopping: 'bag', accounts: 'receipt', work: 'briefcase', research: 'search', money: 'coins', home: 'home', media: 'film', regional: 'flag' }
const PB_WHERE = { any: 'Any browser', prefer_chrome: 'Best in your Chrome', chrome: 'Your Chrome, signed in' }
const PB_FOR_KEEP_MS = 60 * 1000
const PB_SEARCH_WAIT_MS = 180

const PB = {
  site: '', url: '', chips: [], recsHere: [], more: 0, at: 0, asked: '',   // the page in front
  store: null, storeError: '', cat: '', q: '', items: null, searching: false, listError: '',
  open: null,           // one playbook: {book, values, note, error, sending, loading}
  back: 'main',         // where its Back goes
  region: '',
}

const pbRegion = () => {
  if (PB.region) return PB.region
  const m = /^[a-z]{2,3}-([A-Za-z]{2})\b/.exec(navigator.language || '')
  PB.region = m ? m[1].toLowerCase() : ''
  return PB.region
}
const flagOf = (cc) => (/^[a-z]{2}$/i.test(cc || '') ? String.fromCodePoint(...[...cc.toUpperCase()].map((c) => 0x1F1E6 + c.charCodeAt(0) - 65)) : '')
const pbSiteName = (site) => String(site || '').replace(/^www\./, '').replace(/\/+$/, '')
const pbBrand = (host) => String(host || '').replace(/^www\./, '').split('.')[0]
// A link can't carry where the person is: they give the town or address.
const PB_HERE = /^\s*(here|me|my location|current location|where i am|home)\s*[.!]?\s*$/i

function pbTile(p, cls = '') {
  const lead = String(p.id || '').split('.')[0]
  const letter = lead && lead !== 'web' ? lead.replace(/^\d+/, '')[0] || lead[0] : ''
  const inner = letter ? esc(letter.toUpperCase()) : icon(PB_ICON[p.category] || 'bolt')
  return `<span class="pb-tile ${cls}" style="--h:${PB_HUE[p.category] ?? 250}" aria-hidden="true">${inner}</span>`
}

function pbMeta(p) {
  const bits = [p.site ? esc(pbSiteName(p.site)) : 'a site you name']
  bits.push(p.kind === 'act' ? `<span class="pb-kind act">Fills it in</span>` : 'Looks it up')
  if (p.where === 'chrome') bits.push('your Chrome')
  if (p.region && p.region.length === 2) bits.push(`${flagOf(p.region)} ${esc(p.region.toUpperCase())}`)
  return bits.join(' · ')
}

function pbRow(p) {
  return `
    <div class="pb-row" role="button" tabindex="0" data-pb="${esc(p.id)}" title="${esc(p.hint || p.title)}">
      ${pbTile(p)}
      <div class="grow">
        <div class="pb-title">${esc(p.title)}</div>
        <div class="pb-meta">${pbMeta(p)}</div>
      </div>
      <button class="pb-add ${p.mine ? 'on' : ''}" data-pb-shelf="${esc(p.id)}" aria-pressed="${!!p.mine}"
        title="${p.mine ? 'On My playbooks — take it off' : 'Add to My playbooks'}">${icon(p.mine ? 'check' : 'plus')}</button>
    </div>`
}

const pbList = (items) => `<div class="pb-list">${items.map(pbRow).join('')}</div>`
const pbSkeleton = (n = 4) => `<div class="pb-list">${'<div class="pb-skel"></div>'.repeat(n)}</div>`

// ── the chips under the page card ──

async function pbForTab(tab = S.tab) {
  const url = tab?.url || ''
  if (!S.conn || S.conn.agent || !/^https?:/i.test(url)) {
    if (PB.url) { PB.url = ''; PB.site = ''; PB.chips = []; PB.recsHere = []; PB.more = 0 }
    return
  }
  let host = ''
  try { host = new URL(url).host } catch { return }
  const key = host + new URL(url).pathname.split('/').slice(0, 2).join('/')   // a site's part: google.com/maps
  if (PB.asked === key && Date.now() - PB.at < PB_FOR_KEEP_MS) return
  PB.asked = key
  try {
    const got = await api(S.conn, `/fd/clip/playbooks/for?url=${encodeURIComponent(url)}`)
    if (PB.asked !== key) return
    PB.url = url
    PB.site = got.site || ''
    PB.chips = got.items || []
    PB.recsHere = got.recordings || []
    PB.more = got.more || 0
    PB.at = Date.now()
    if (S.view === 'main') render()
  } catch (e) {
    PB.asked = ''
    if (e instanceof AuthError) disconnected(e.message)
    // An older Flight Deck has no playbooks: no chips, nothing to say.
  }
}

function viewPbStrip() {
  if (!PB.chips.length && !PB.recsHere.length) return ''
  const fav = S.tab?.favIconUrl && /^https?:|^data:image\//.test(S.tab.favIconUrl) ? S.tab.favIconUrl : ''
  const chip = (p) => `<button class="pb-chip" data-pb="${esc(p.id)}" title="${esc(p.hint || p.title)}">
      ${fav ? `<img class="fav" src="${esc(fav)}" alt="">` : pbTile(p, 'sm')}<span>${esc(p.title)}</span></button>`
  const recs = PB.recsHere.map((r) => `<button class="pb-chip own" data-rec-run="${esc(r.id)}" title="Your recording · ${esc(plural(r.steps, 'step'))}">
      <span class="rec-dot"></span><span>${esc(r.name)}</span></button>`).join('')
  return `
    <div class="pb-strip">
      <div class="label-row"><span class="label" style="margin:0">Do it here</span><span class="grow"></span>
        <button class="link-btn" data-act="pb-store-site">${PB.chips.length + PB.more > PB.chips.length ? `All ${PB.chips.length + PB.more}` : 'More'} ${icon('arrowright', 'sm')}</button></div>
      <div class="pb-chips">${recs}${PB.chips.map(chip).join('')}</div>
    </div>`
}

// ── the store ──

async function openStore({ cat = PB.cat, q = PB.q, site = false } = {}) {
  PB.cat = cat
  PB.q = q
  if (site && PB.site) { PB.q = ''; PB.cat = '' }
  S.view = 'pbs'
  render()
  window.scrollTo(0, 0)
  if (!PB.store) loadStore()
  if (PB.q || PB.cat) pbSearch(true)
}

async function loadStore() {
  PB.storeError = ''
  try {
    PB.store = await api(S.conn, `/fd/clip/playbooks?region=${encodeURIComponent(pbRegion())}`)
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    PB.storeError = /404/.test(e.message) ? 'This Flight Deck has no playbooks yet — it needs an update.' : e.message
  }
  if (S.view === 'pbs') render()
}

let pbSearchTimer = 0
let pbSearchSeq = 0
function pbSearch(now = false) {
  clearTimeout(pbSearchTimer)
  if (!PB.q.trim() && !PB.cat) { PB.items = null; PB.searching = false; render(); return }
  PB.searching = true
  const go = async () => {
    const seq = ++pbSearchSeq
    const qs = new URLSearchParams({ q: PB.q.trim(), category: PB.cat, region: pbRegion(), limit: '40' })
    try {
      const got = await api(S.conn, `/fd/clip/playbooks/search?${qs}`)
      if (seq !== pbSearchSeq) return
      PB.items = got.items || []
      PB.listError = ''
    } catch (e) {
      if (e instanceof AuthError) return disconnected(e.message)
      if (seq !== pbSearchSeq) return
      PB.listError = e.message
    }
    PB.searching = false
    if (S.view === 'pbs') render()
  }
  if (now) go()
  else pbSearchTimer = setTimeout(go, PB_SEARCH_WAIT_MS)
}

function viewStore() {
  const st = PB.store
  const cats = st?.categories || []
  const mineN = st?.shelf?.length || 0
  const catBtn = (id, label, ic, n) => `<button class="pb-cat ${PB.cat === id ? 'on' : ''}" data-pb-cat="${esc(id)}" aria-pressed="${PB.cat === id}">
      ${ic ? icon(ic) : ''}${esc(label)}${n != null ? ` <span class="n">${n}</span>` : ''}</button>`
  const head = `
    <div class="pb-head">
      <button class="icon-btn" data-act="pb-back" title="Back">${icon('back')}</button>
      <div class="grow"><div class="pb-h1">Playbooks</div>
        <div class="dim">${st ? `${st.total} ready-made ways to get things done` : 'Ready-made ways to get things done'}</div></div>
    </div>
    <div class="pb-sticky" id="pbSticky">
      <div class="search pb-search">${icon('search', 'sm')}<input type="search" id="pbSearch" placeholder="What do you want done? Try “track a parcel”" value="${esc(PB.q)}" autocomplete="off" spellcheck="false"></div>
      <div class="pb-cats">
        ${catBtn('', 'All', 'grid')}${mineN ? catBtn('mine', 'Mine', 'star', mineN) : ''}
        ${cats.map((c) => catBtn(c.id, c.label, c.icon)).join('')}
      </div>
    </div>`
  if (PB.storeError) return head + `<div class="notice err">${esc(PB.storeError)}</div>`
  if (!st) return head + pbSkeleton(6)

  if (PB.q.trim() || PB.cat) {
    const catName = PB.cat === 'mine' ? 'My playbooks' : cats.find((c) => c.id === PB.cat)?.label || ''
    let body
    if (PB.listError) body = `<div class="notice err">${esc(PB.listError)}</div>`
    else if (PB.items === null) body = pbSkeleton()
    else if (!PB.items.length) {
      body = `<div class="pb-empty">${icon('search')}<div>${PB.q.trim() ? `Nothing for “${esc(PB.q.trim())}”${catName ? ` in ${esc(catName)}` : ''}.` : 'Nothing here yet.'}</div>
        <div class="dim" style="margin-top:4px">Ask your agent anyway — it works sites step by step, or record it once.</div></div>`
    } else if (PB.cat === 'regional' && !PB.q.trim()) {
      const by = {}
      for (const p of PB.items) (by[p.region] = by[p.region] || []).push(p)
      body = Object.entries(by).map(([cc, items]) => `
        <div class="pb-section"><div class="pb-section-head"><span class="t">${flagOf(cc)} ${esc(cc.toUpperCase())}</span><span class="n">${items.length}</span></div>${pbList(items)}</div>`).join('')
    } else {
      body = `<div class="pb-section">
        <div class="pb-section-head"><span class="t">${PB.q.trim() ? `Best for “${esc(PB.q.trim())}”` : esc(catName)}</span>
          <span class="n">${PB.items.length}${PB.searching ? ' · …' : ''}</span></div>${pbList(PB.items)}</div>`
    }
    return head + body
  }

  const sections = []
  if (PB.chips.length) {
    sections.push(`<div class="pb-section"><div class="pb-section-head">${S.tab?.favIconUrl ? `<img class="fav" src="${esc(S.tab.favIconUrl)}" alt="" style="width:14px;height:14px;border-radius:3px">` : icon('globe', 'sm')}
      <span class="t">On ${esc(PB.site)}</span><span class="n">${PB.chips.length + PB.more}</span>
      ${PB.more ? `<button class="link-btn pb-see" data-pb-q="${esc(pbBrand(PB.site))}">See all ${icon('arrowright', 'sm')}</button>` : ''}</div>${pbList(PB.chips)}</div>`)
  }
  if (mineN) {
    sections.push(`<div class="pb-section"><div class="pb-section-head">${icon('star', 'sm')}<span class="t">My playbooks</span><span class="n">${mineN}</span>
      ${mineN > 4 ? `<button class="link-btn pb-see" data-pb-cat="mine">See all ${icon('arrowright', 'sm')}</button>` : ''}</div>${pbList(st.shelf.slice(0, 4))}</div>`)
  }
  sections.push(`<div class="pb-section"><div class="pb-section-head">${icon('sparkles', 'sm')}<span class="t">Popular</span></div>${pbList(st.popular)}</div>`)
  for (const c of cats) {
    const top = st.top?.[c.id] || []
    if (!top.length) continue
    sections.push(`<div class="pb-section"><div class="pb-section-head">${icon(c.icon, 'sm')}<span class="t">${esc(c.label)}</span><span class="n">${c.count}</span>
      <button class="link-btn pb-see" data-pb-cat="${esc(c.id)}">See all ${icon('arrowright', 'sm')}</button></div>${pbList(top)}</div>`)
  }
  return head + sections.join('')
}

// ── one playbook: what it does, its form, Run ──

async function openPb(id, back = S.view) {
  PB.back = back === 'pb' ? PB.back : back
  PB.open = { id, book: null, values: {}, note: '', error: '', sending: false, loading: true }
  S.view = 'pb'
  render()
  window.scrollTo(0, 0)
  try {
    const book = await api(S.conn, `/fd/clip/playbooks/${encodeURIComponent(id)}`)
    if (PB.open?.id !== id) return
    PB.open.book = book
    PB.open.values = pbPrefill(book)
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    if (PB.open?.id === id) PB.open.error = e.message
  }
  if (PB.open?.id === id) { PB.open.loading = false; render() }
}

/** What the page already says: the site in front for one the person names,
 * a short selection for its one text box. */
function pbPrefill(book) {
  const values = {}
  const tabUrl = S.tab?.url || ''
  const urlInput = book.inputs.find((i) => i.kind === 'url' && i.required)
  if (urlInput && /^https:\/\//i.test(tabUrl)) values[urlInput.name] = new URL(tabUrl).origin
  const sel = String(S.scan?.selection?.text || '').trim()
  const texts = book.inputs.filter((i) => i.required && !('default' in i) && i.kind === 'text')
  if (sel && sel.length <= 120 && !sel.includes('\n') && texts.length === 1) values[texts[0].name] = sel
  return values
}

// What's wrong with a number typed for a playbook ("at least 0"), or '': Flight
// Deck and the agent's tool check it again.
function pbRange(i, raw) {
  const text = String(raw ?? '').trim()
  if (i.kind !== 'number' || !text) return ''
  const n = Number(text.replace(',', '.'))
  if (!Number.isFinite(n)) return 'a number'
  if (!((i.min !== undefined && n < i.min) || (i.max !== undefined && n > i.max))) return ''
  return i.min !== undefined && i.max !== undefined ? `${i.min} to ${i.max}` : i.min !== undefined ? `at least ${i.min}` : `at most ${i.max}`
}

function pbMissing(o) {
  return o.book.inputs.filter((i) => (i.required && !('default' in i) && i.ask === 'before' && !String(o.values[i.name] ?? '').trim())
    || (i.kind === 'place' && PB_HERE.test(String(o.values[i.name] ?? 'x')))
    || pbRange(i, o.values[i.name]))
}

function pbField(i, o) {
  const id = `pbIn-${i.name}`
  const v = o.values[i.name]
  const optional = !i.required || 'default' in i
  const label = `<label class="label" for="${id}">${esc(i.label)}${optional ? '<span class="opt-mark">optional</span>' : ''}</label>`
  const hint = i.hint ? `<div class="help">${esc(i.hint)}</div>` : ''
  const dflt = 'default' in i && i.default !== '' && i.default !== false ? String(i.default) : ''
  let control
  if (i.kind === 'choice') {
    const cur = v ?? i.default ?? ''
    control = i.options.length <= 5
      ? `<div class="pb-seg" role="group" aria-label="${esc(i.label)}">${i.options.map((opt) => `
          <button type="button" class="${opt === cur ? 'on' : ''}" data-pb-choice="${esc(i.name)}" data-val="${esc(opt)}" aria-pressed="${opt === cur}">${esc(opt.replace(/_/g, ' '))}</button>`).join('')}</div>`
      : `<select class="select" id="${id}" data-pb-input="${esc(i.name)}">${i.options.map((opt) => `<option ${opt === cur ? 'selected' : ''}>${esc(opt)}</option>`).join('')}</select>`
    return `<div class="pb-field"><div class="label">${esc(i.label)}${optional ? '<span class="opt-mark">optional</span>' : ''}</div>${control}${hint}</div>`
  }
  if (i.kind === 'flag') {
    const on = v ?? i.default ?? false
    return `<div class="pb-field"><label class="pb-toggle"><input type="checkbox" id="${id}" data-pb-flag="${esc(i.name)}" ${on ? 'checked' : ''}><span class="track"></span><span>${esc(i.label)}</span></label>${hint}</div>`
  }
  if (i.kind === 'file') {
    return `<div class="pb-field">${label}<div class="dim">${icon('file', 'sm')} ${esc(dflt || 'Your agent picks the file')}</div></div>`
  }
  const type = { date: 'date', time: 'time', number: 'number', url: 'url' }[i.kind] || 'text'
  const value = v ?? (type === 'date' && /^\d{4}-\d{2}-\d{2}$/.test(dflt) ? dflt : type === 'time' && /^\d{2}:\d{2}$/.test(dflt) ? dflt : '')
  const ph = i.kind === 'place' ? 'Town or address' : i.kind === 'url' ? 'https://…' : dflt ? `Default: ${dflt}` : ''
  const range = type === 'number' ? `${i.min !== undefined ? ` min="${esc(i.min)}"` : ''}${i.max !== undefined ? ` max="${esc(i.max)}"` : ''}` : ''
  const input = `<input type="${type}" id="${id}" data-pb-input="${esc(i.name)}" value="${esc(value)}" placeholder="${esc(ph)}" ${type === 'number' ? `inputmode="decimal" step="any"${range}` : ''} autocomplete="off">`
  control = i.kind === 'place' ? `<div class="with-ic">${icon('pin', 'sm')}${input}</div>`
    : i.kind === 'url' ? `<div class="with-ic">${icon('globe', 'sm')}${input}</div>` : input
  const bad = pbRange(i, v)
  const here = bad ? `<div class="warn">${esc(i.label)} is ${esc(bad)}.</div>`
    : i.kind === 'place' && PB_HERE.test(String(v ?? 'x')) ? '<div class="warn">Type the town or address — a link can\'t carry where you are.</div>' : ''
  return `<div class="pb-field">${label}${control}${here || hint || (type === 'date' && dflt && !value ? `<div class="help">Leave it for ${esc(dflt)}</div>` : '')}</div>`
}

function viewPb() {
  const o = PB.open
  const back = `<button class="icon-btn" data-act="pb-close" title="Back">${icon('back')}</button>`
  if (!o) return ''
  if (o.loading || !o.book) {
    return `<div class="pb-head">${back}<div class="grow"></div></div>
      ${o.error ? `<div class="notice err">${esc(o.error)}</div>` : `<div class="pb-hero"><span class="pb-tile lg"></span><div class="grow">${pbSkeleton(1)}</div></div>`}`
  }
  const b = o.book
  const main = b.inputs.filter((i) => i.required && !('default' in i))
  const more = b.inputs.filter((i) => !main.includes(i))
  const missing = pbMissing(o)
  const ag = agentDef()
  const agents = (S.me?.agents || []).map((a) => {
    const up = a.status === 'running' || a.status === 'asleep'
    return `<option value="${esc(a.slug)}" ${a.slug === S.agent ? 'selected' : ''} ${up ? '' : 'disabled'}>${a.status === 'running' ? '●' : a.status === 'asleep' ? '◐' : '○'} ${esc(a.name)}${up ? '' : ` (${esc(a.status)})`}</option>`
  }).join('')
  const needsChrome = b.where === 'chrome' || b.where === 'prefer_chrome'
  const handsOk = S.hands.on && S.handsStatus?.[S.conn?.url]?.state !== 'denied'
  const cat = (PB.store?.categories || []).find((c) => c.id === b.category)
  const wrong = missing.find((i) => pbRange(i, o.values[i.name]))
  const runLabel = o.sending ? 'Starting…' : !ag ? 'Pick an agent' : wrong ? `${wrong.label} is ${pbRange(wrong, o.values[wrong.name])}`
    : missing.length ? `Fill in ${missing.map((i) => i.label).slice(0, 2).join(' and ')}` : `Run with ${ag.name}`
  return `
    <div class="pb-head">${back}<div class="grow"></div>
      <button class="pb-star ${b.mine ? 'on' : ''}" data-pb-shelf="${esc(b.id)}" aria-pressed="${!!b.mine}">${icon('star', 'sm')} ${b.mine ? 'In My playbooks' : 'Add to mine'}</button></div>
    <div class="pb-hero">${pbTile(b, 'lg')}
      <div class="grow"><div class="pb-h1">${esc(b.title)}</div>
        <div class="dim">${esc([b.sites?.length ? b.sites.map(pbSiteName).filter((x, k, all) => all.indexOf(x) === k).slice(0, 3).join(', ') : 'Any site you name', cat?.label].filter(Boolean).join(' · '))}</div></div></div>
    <div class="pb-badges">
      ${b.kind === 'act' ? `<span class="pb-badge act">${icon('pencil')} Fills it in for you</span>` : `<span class="pb-badge look">${icon('eye')} Looks it up</span>`}
      <span class="pb-badge">${icon(needsChrome ? 'lock' : 'globe')} ${esc(PB_WHERE[b.where] || b.where)}</span>
      ${b.region?.length === 2 ? `<span class="pb-badge">${flagOf(b.region)} ${esc(b.region.toUpperCase())}</span>` : ''}
      ${b.runs ? `<span class="pb-badge">${icon('history')} ran ${b.runs}×</span>` : ''}
    </div>
    ${b.says?.length ? `<div class="pb-says">${b.says.slice(0, 2).map((s) => `<div class="pb-say">“${esc(s)}”</div>`).join('')}</div>` : ''}
    ${b.kind === 'act' ? `<div class="pb-stop">${icon('stop', 'sm')}<div>It fills everything in and stops before <b>“${esc(b.stop)}”</b> — you look it over and press it yourself.</div></div>` : ''}
    <div class="pb-form">
      ${main.map((i) => pbField(i, o)).join('')}
      ${more.length ? (main.length
        ? `<details class="pb-more" ${o.moreOpen ? 'open' : ''} data-pb-more><summary>${icon('chevron', 'sm chev')} More options <span class="dim">— ${more.length}, or let ${esc(ag?.name || 'the agent')} decide</span></summary><div class="pb-form">${more.map((i) => pbField(i, o)).join('')}</div></details>`
        : more.map((i) => pbField(i, o)).join('')) : ''}
      <div><label class="label" for="agent">Who does it</label><select class="select" id="agent">${agents}</select></div>
      <textarea id="pbNote" rows="2" placeholder="Anything else it should know? (optional)">${esc(o.note)}</textarea>
    </div>
    ${needsChrome && !handsOk ? `<div class="notice">${icon('globe')}<div class="grow">${b.where === 'chrome' ? 'This one needs you signed in, in this browser' : 'This site often turns robots away — it goes best in this browser'}, and your agents can't use it yet.
      <div style="margin-top:6px"><button class="btn" data-act="rec-hands-on">Let agents use this browser</button></div></div></div>` : ''}
    <details class="pb-fold card" ${o.foldOpen ? 'open' : ''} data-pb-fold>
      <summary>${icon('chevron', 'sm chev')}<span class="grow"><b>How it goes</b> <span class="dim">— ${b.steps.length ? plural(b.steps.length, 'step') + ', then ' : ''}it reads ${plural(b.read.length, 'thing')}</span></span></summary>
      ${b.steps.length ? `<ol class="pb-steps">${b.steps.map((s) => `<li><span>${esc(s.replace(/^\d+\.\s*/, ''))}</span></li>`).join('')}</ol>` : ''}
      ${b.read.length ? `<div class="pb-reads">${b.read.map((r) => `<div class="pb-read"><b>${esc(r.name.replace(/_/g, ' '))}</b> — ${esc(r.about)}</div>`).join('')}</div>` : ''}
    </details>
    ${o.error ? `<div class="notice err">${esc(o.error)}</div>` : ''}
    <div class="pb-runbar">
      <button class="btn primary block" data-act="pb-run" ${ag && !o.sending && !missing.length ? '' : 'disabled'}>${icon('play', 'sm')} ${esc(runLabel)}</button>
    </div>`
}

async function pbShelf(id, on) {
  const flip = (p) => { if (p && p.id === id) p.mine = on }
  try {
    await api(S.conn, `/fd/clip/playbooks/${encodeURIComponent(id)}/shelf`, { method: 'POST', body: JSON.stringify({ on }) })
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
    return
  }
  for (const list of [PB.chips, PB.items || [], PB.store?.popular || [], ...Object.values(PB.store?.top || {})]) list.forEach(flip)
  flip(PB.open?.book)
  if (PB.store) {
    const item = [...(PB.items || []), ...PB.chips, ...(PB.store.popular || []), ...Object.values(PB.store.top || {}).flat(), PB.open?.book]
      .find((p) => p && p.id === id)
    PB.store.shelf = on ? [{ ...(item || { id, title: id }), mine: true }, ...PB.store.shelf.filter((p) => p.id !== id)]
      : PB.store.shelf.filter((p) => p.id !== id)
  }
  toast(on ? 'Added to My playbooks' : 'Taken off My playbooks')
  render()
}

async function pbRunGo() {
  const o = PB.open
  const ag = agentDef()
  if (!o?.book || !ag || o.sending || pbMissing(o).length) return
  o.sending = true
  o.error = ''
  render()
  const inputs = {}
  for (const [k, v] of Object.entries(o.values)) if (v !== '' && v != null) inputs[k] = v
  try {
    const res = await api(S.conn, `/fd/clip/playbooks/${encodeURIComponent(o.book.id)}/run`, {
      method: 'POST', body: JSON.stringify({ agent: ag.slug, inputs, note: o.note, browser: browserLabel(), hands: !!S.hands.on }) })
    PB.open = null
    startRun({ clip_id: res.clip_id, folder: res.folder, vfs: res.vfs, agent: res.agent || ag.name, agentSlug: ag.slug,
               title: o.book.title, url: o.book.sites?.[0] ? `https://${o.book.sites[0]}` : '', action: 'playbook', actionLabel: 'Playbook' })
  } catch (e) {
    o.sending = false
    if (e instanceof AuthError) return disconnected(e.message)
    o.error = e.message
    render()
  }
}

$view.addEventListener('click', (ev) => {
  const el = ev.target.closest('[data-pb-shelf],[data-pb],[data-pb-cat],[data-pb-q],[data-pb-choice],[data-act^="pb-"]')
  if (!el) return
  const d = el.dataset
  if (d.pbShelf) {
    ev.stopPropagation()
    const cur = [PB.open?.book, ...PB.chips, ...(PB.items || []), ...(PB.store?.shelf || []), ...(PB.store?.popular || []), ...Object.values(PB.store?.top || {}).flat()]
      .find((p) => p && p.id === d.pbShelf)
    return void pbShelf(d.pbShelf, !cur?.mine)
  }
  if (d.pbChoice && PB.open) {
    PB.open.values[d.pbChoice] = d.val
    return void render()
  }
  if (d.pbCat !== undefined) {
    PB.cat = PB.cat === d.pbCat && d.pbCat ? '' : d.pbCat
    if (S.view !== 'pbs') return void openStore()
    window.scrollTo(0, 0)
    return void pbSearch(true)
  }
  if (d.pbQ !== undefined) { PB.q = d.pbQ; PB.cat = ''; window.scrollTo(0, 0); return void pbSearch(true) }
  if (d.pb) return void openPb(d.pb)
  if (d.act === 'pb-store' || d.act === 'pb-store-site') return void openStore({ site: d.act === 'pb-store-site' })
  if (d.act === 'pb-back') { S.view = 'main'; render(); scheduleScan(0); return }
  if (d.act === 'pb-close') {
    PB.open = null
    S.view = PB.back === 'pbs' ? 'pbs' : 'main'
    if (S.view === 'pbs' && !PB.store) loadStore()
    render()
    return
  }
  if (d.act === 'pb-run') pbRunGo()
})

$view.addEventListener('keydown', (ev) => {
  const row = ev.target.closest?.('.pb-row[data-pb]')
  if (row && (ev.key === 'Enter' || ev.key === ' ') && ev.target === row) { ev.preventDefault(); openPb(row.dataset.pb) }
  if (ev.target.id === 'pbSearch' && ev.key === 'Escape' && PB.q) { ev.preventDefault(); PB.q = ''; pbSearch(true) }
})

$view.addEventListener('input', (ev) => {
  const el = ev.target
  if (el.id === 'pbSearch') {
    PB.q = el.value
    // Typing is a new search, over everything (a category picked after it narrows it; Mine stays Mine).
    if (PB.cat && PB.cat !== 'mine' && PB.q.trim()) PB.cat = ''
    if (!PB.q.trim() && !PB.cat) { PB.items = null; render(); return }
    if (PB.items === null) render()
    pbSearch()
  } else if (el.dataset.pbInput && PB.open) {
    const was = pbMissing(PB.open).length
    PB.open.values[el.dataset.pbInput] = el.value
    if (was !== pbMissing(PB.open).length || PB.open.error) { PB.open.error = ''; render() }
  } else if (el.id === 'pbNote' && PB.open) PB.open.note = el.value
})

// A focused number box counts up or down under the wheel (scrolling past one
// turned 7 nights into -51 in Flight Deck's store): let go of it instead.
$view.addEventListener('wheel', (ev) => {
  if (ev.target instanceof HTMLInputElement && ev.target.type === 'number' && document.activeElement === ev.target) ev.target.blur()
}, { passive: true })

$view.addEventListener('change', (ev) => {
  const el = ev.target
  if (el.dataset.pbFlag && PB.open) PB.open.values[el.dataset.pbFlag] = el.checked
  else if (el.dataset.pbInput && PB.open && el.tagName === 'SELECT') PB.open.values[el.dataset.pbInput] = el.value
})

$view.addEventListener('toggle', (ev) => {
  if (!PB.open) return
  if (ev.target.matches('details[data-pb-more]')) PB.open.moreOpen = ev.target.open
  if (ev.target.matches('details[data-pb-fold]')) PB.open.foldOpen = ev.target.open
}, true)

window.addEventListener('scroll', () => {
  document.getElementById('pbSticky')?.classList.toggle('scrolled', window.scrollY > 40)
}, { passive: true })

$btnPlaybooks.addEventListener('click', () => {
  if (S.view === 'pbs') { S.view = 'main'; render(); return }
  openStore()
})

// ── history ──

async function remember(run) {
  const item = {
    deck: run.deck || S.conn?.url || '',
    clip_id: run.clip_id, title: run.title, url: run.url, agent: run.agent, agentSlug: run.agentSlug, action: run.action,
    actionLabel: run.actionLabel, folder: run.folder, vfs: run.vfs, status: run.status, at: run.at || Date.now(), watch: run.watch || null,
    result: String(run.result || '').slice(0, 8000), error: run.error || '', files: run.files || [],
  }
  S.history = [item, ...S.history.filter((h) => h.clip_id !== item.clip_id)].slice(0, HISTORY_MAX)
  await setStore({ history: S.history })
}

function viewHistory() {
  const items = S.history.map((h, i) => `
    <button class="history-item" data-history="${i}">
      <span class="ellipsis" style="font-weight:600">${esc(h.actionLabel || h.action)}: ${esc(h.title)}</span>
      <span class="dim ellipsis">${esc(h.agent)}${S.decks.length > 1 && h.deck ? ` on ${esc(deckShort(h.deck))}` : ''} · ${esc(hostOf(h.url))} · ${esc(new Date(h.at).toLocaleString())}</span>
    </button>`).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back">${icon('back')}</button><div class="grow" style="font-weight:600">Recent</div>
      ${S.history.length ? '<button class="btn" data-act="clear-history" style="padding:3px 8px">Clear</button>' : ''}</div>
    ${items || '<div class="dim">Nothing sent yet.</div>'}`
}

function openHistory(i) {
  const h = S.history[i]
  if (!h) return
  // Kept from before decks were remembered: the deck in use then was the only one.
  S.run = { deck: S.decks[0]?.url, ...h, events: [], next: 0, live: '', done: !['starting', 'thinking'].includes(h.status), followedEnd: true }
  S.follow = true
  S.view = 'run'
  render()
  if (!S.run.done) { pollRun(S.run); watchAsks() }
}

// ── settings ──

function viewSettings() {
  const u = S.conn?.user || {}
  const agents = (S.me?.agents || []).map((a) => `<option value="${esc(a.slug)}" ${a.slug === S.agent ? 'selected' : ''}>${esc(a.name)}</option>`).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back">${icon('back')}</button><div class="grow" style="font-weight:600">Settings</div></div>
    ${viewVersion(true)}
    <div class="card">
      <div class="label">${S.decks.length > 1 ? 'In use' : 'Connected'}</div>
      <div>${esc(deckTitle(S.conn))}</div>
      <div class="dim">${esc(u.name || '')}${u.email && u.email !== u.name ? ` · ${esc(u.email)}` : ''}</div>
      <button class="link-btn" data-act="decks" style="margin-top:6px">${S.decks.length > 1 ? `Your ${S.decks.length} Flight Decks` : 'Connect another Flight Deck'}</button>
    </div>
    ${agents ? `<div class="card"><label class="label" for="setAgent">Agent</label><select class="select" id="setAgent">${agents}</select></div>` : ''}
    <div class="card">
      <div class="label" id="textSizeLabel">Text size</div>
      <div class="seg" role="radiogroup" aria-labelledby="textSizeLabel">${TEXT_SIZES.map(([id, label, scale]) => `
        <button class="seg-btn ${textSize() === id ? 'on' : ''}" data-text-size="${id}" role="radio" aria-checked="${textSize() === id}">
          <span class="seg-a" style="font-size:${Math.round(13 * scale)}px">A</span>${esc(label)}
        </button>`).join('')}</div>
    </div>
    <div class="card">
      <div class="label">Agents</div>
      <label class="switch"><input type="checkbox" id="handsOn" ${S.hands.on ? 'checked' : ''}><div>Agents can use this browser<div class="dim">Your agents' browser tool works here, signed in as you — in tabs of their own, grouped under the agent's name. Chrome shows that Captain is debugging the browser while one works. Flight Deck's own pages stay off-limits.${S.decks.length > 1 ? ` Agents on each of your ${S.decks.length} Flight Decks can.` : ''}</div></div></label>
      <label class="switch"><input type="checkbox" id="handsFront" ${S.hands.front ? 'checked' : ''} ${S.hands.on ? '' : 'disabled'}><div>Bring an agent's tab to the front while it works</div></label>
      ${S.hands.on && handsStatusText() ? `<div class="dim" style="margin-top:4px">${esc(handsStatusText())}</div>` : ''}
    </div>
    ${viewSearchSettings()}
    ${viewFetchSettings()}
    <div class="card">
      <div class="label">Every time</div>
      <label class="switch"><input type="checkbox" id="prefAll" ${S.allSites ? 'checked' : ''}><div>Work on every site without asking<div class="dim">Reads pages and fetches their files anywhere; also lets screenshots work.</div></div></label>
      <label class="switch"><input type="checkbox" id="prefSnapshot" ${S.prefs.snapshot ? 'checked' : ''}><div>Always send the whole page (.mhtml)</div></label>
      <label class="switch"><input type="checkbox" id="prefShot" ${S.prefs.screenshot ? 'checked' : ''}><div>Always send a screenshot</div></label>
    </div>
    <button class="btn block" data-act="open-deck">${icon('external', 'sm')} Open Flight Deck</button>
    <button class="btn block danger" data-act="disconnect">${S.decks.length > 1 ? `Disconnect from ${esc(hostOf(S.conn?.url || ''))}` : 'Disconnect this browser'}</button>
    <div class="foot">Captain for Chrome ${esc(VERSION)}${S.me?.extension?.version && S.me.extension.version !== VERSION ? ` · Flight Deck has ${esc(S.me.extension.version)}` : ''}</div>`
}

// ── text size ──

const TEXT_SIZES = [['small', 'Small', 1], ['medium', 'Medium', 1.15], ['large', 'Large', 1.3]]

function textSize() {
  return TEXT_SIZES.some(([id]) => id === S.prefs.textSize) ? S.prefs.textSize : 'small'
}

// Small is the panel as it always was; the stylesheet scales from data-size.
function applyTextSize() {
  const size = textSize()
  if (size === 'small') delete document.documentElement.dataset.size
  else document.documentElement.dataset.size = size
}

async function refreshAllSites() {
  try { S.allSites = await chrome.permissions.contains({ origins: ['<all_urls>'] }) } catch { S.allSites = false }
}

// ── events ──

$view.addEventListener('click', (ev) => {
  // Opened or closed now, not when the (later) toggle event comes: a redraw
  // in between would draw it the old way.
  const summary = ev.target.closest('details.include:not([data-rec-extras]) > summary')
  if (summary) { S.includeOpen = !summary.parentElement.open; return }
  const el = ev.target.closest('[data-act],[data-action],[data-lens],[data-history],[data-file-path],[data-ftype],[data-text-size],[data-watch]')
  if (!el) return
  if (el.dataset.action) {
    S.action = el.dataset.action
    applyDefaults()
    render()
    if (actionDef().needs_text) document.getElementById('note')?.focus()
    return
  }
  if (el.dataset.lens !== undefined) {
    S.perspective = el.dataset.lens
    render()
    if (S.perspective === 'custom') document.getElementById('lensText')?.focus()
    return
  }
  if (el.dataset.watch) {
    S.prefs = { ...S.prefs, watchEvery: el.dataset.watch }
    setStore({ prefs: S.prefs })
    render()
    return
  }
  if (el.dataset.textSize) {
    S.prefs = { ...S.prefs, textSize: el.dataset.textSize }
    applyTextSize()
    setStore({ prefs: S.prefs })
    render()
    return
  }
  if (el.dataset.ftype !== undefined) {
    S.fileType = S.fileType === el.dataset.ftype ? '' : el.dataset.ftype
    render()
    return
  }
  if (el.dataset.history !== undefined) return openHistory(Number(el.dataset.history))
  if (el.dataset.filePath) return saveRunFile(el.dataset.filePath)
  const act = el.dataset.act
  if (act === 'connect') startPairing()
  else if (act === 'reopen' && S.pairing) chrome.tabs.create({ url: S.pairing.url })
  else if (act === 'cancel-pair') { S.pairing = null; render() }
  else if (act === 'retry-me') loadMe().then(() => { applyDefaults(); render(); loadLenses() })
  else if (act === 'rescan') scan()
  else if (act === 'grant') {
    chrome.permissions.request({ origins: [`${S.needsPermission}/*`] }).then((ok) => { if (ok) scan() }).catch(() => {})
  } else if (act === 'send') onSend()
  else if (act === 'anonymize') anonymize()
  else if (act === 'anon-save') anonSave()
  else if (act === 'anon-restore') anonRestore()
  else if (act === 'anon-copy' && S.anon?.output) {
    navigator.clipboard.writeText(S.anon.output).then(() => toast('Copied'), () => toast("Couldn't copy"))
  } else if (act === 'anon-copy-back' && S.anon?.restored) {
    navigator.clipboard.writeText(S.anon.restored.output).then(() => toast('Copied'), () => toast("Couldn't copy"))
  }
  else if (act === 'files-docs' || act === 'files-all' || act === 'files-none') {
    const files = allFiles()
    S.checked = new Set(act === 'files-none' ? [] : files.filter((f) => act === 'files-all' || DOC_KINDS.has(f.kind)).map((f) => f.url))
    S.touchedFiles = true
    render()
  } else if (act === 'files-shown-on' || act === 'files-shown-off') {
    for (const f of shownFiles(allFiles())) {
      if (act === 'files-shown-on') S.checked.add(f.url)
      else S.checked.delete(f.url)
    }
    S.touchedFiles = true
    render()
  } else if (act === 'files-unfilter') { S.fileQuery = ''; S.fileType = ''; render() }
  else if (act === 'lens-more') { S.lensMore = !S.lensMore; render() }
  else if (act === 'lens-refresh') loadLenses(S.agent, { fresh: true })
  else if (act === 'clear-focus') { S.focus = null; render() }
  else if (act === 'follow') { S.follow = true; render() }
  else if (act === 'new-topic') newTopic(el.dataset.slug || S.agent)
  else if (act === 'open-task' && runConn(S.run) && S.run?.watch?.id) chrome.tabs.create({ url: `${runConn(S.run).url}/#task=${encodeURIComponent(S.run.watch.id)}` })
  else if (act === 'back') { S.view = 'main'; render(); scheduleScan(0) }
  else if (act === 'stop' && S.run && runConn(S.run)) {
    api(runConn(S.run), `/fd/clip/runs/${encodeURIComponent(S.run.clip_id)}/stop`, { method: 'POST' }).catch((e) => toast(e.message))
  } else if (act === 'copy' && S.run?.result) {
    navigator.clipboard.writeText(S.run.result).then(() => toast('Copied'), () => toast("Couldn't copy"))
  } else if (act === 'open-deck' && S.conn) chrome.tabs.create({ url: `${S.conn.url}/` })
  else if (act === 'ver-get' && S.conn) chrome.tabs.create({ url: `${S.conn.url}/fd/clip/extension.zip` })
  else if (act === 'ver-reload') chrome.runtime.reload()
  else if (act === 'ver-later') {
    const ext = versionGap()
    S.verSkip = ext ? `${ext.state}:${ext.version}` : ''
    setStore({ verSkip: S.verSkip })
    render()
  }
  else if (act === 'follow-up') sendFollowUp()
  else if (act === 'hands-show') handsShow(el.dataset.slug)
  else if (act === 'hands-stop') setHandsStopped([...S.handsStopped, el.dataset.slug]).then(() => toast(`${handsName(el.dataset.slug)} can't use this browser now`))
  else if (act === 'hands-let') setHandsStopped(S.handsStopped.filter((s) => s !== el.dataset.slug))
  else if (act === 'hands-extensions') chrome.tabs.create({ url: 'chrome://extensions/?id=' + chrome.runtime.id })
  else if (act === 'search-allow') allowSearch(el.dataset.engine)
  else if (act === 'search-decline') stopSearch(el.dataset.engine)
  else if (act === 'fetch-allow') allowFetch()
  else if (act === 'fetch-decline') stopFetch()
  else if (act === 'clear-history') { S.history = []; setStore({ history: [] }); render() }
  else if (act === 'disconnect' && S.conn) disconnectDeck(S.conn.url)
  else if (act === 'decks') { S.adding = false; S.connectError = ''; S.pairing = null; S.view = 'decks'; render() }
  else if (act === 'deck-use') switchDeck(el.dataset.deck)
  else if (act === 'deck-open') chrome.tabs.create({ url: `${el.dataset.deck}/` })
  else if (act === 'deck-disconnect') disconnectDeck(el.dataset.deck)
  else if (act === 'deck-add') { S.adding = true; S.deckInput = ''; S.connectError = ''; S.pairing = null; S.view = 'connect'; render() }
  else if (act === 'deck-again') { S.adding = true; S.deckInput = el.dataset.deck; S.connectError = ''; S.pairing = null; S.view = 'connect'; render() }
})

function disconnectDeck(url) {
  const d = S.decks.find((x) => x.url === url)
  if (!d) return
  if (S.handsStatus?.[url]?.state === 'denied') { disconnected('', url); return }   // it said no already: just off the list
  if (!confirm(S.decks.length > 1 ? `Disconnect this browser from ${hostOf(url)}? It stays connected to your other Flight Decks.` : 'Disconnect this browser from Flight Deck?')) return
  api(d, '/fd/clip/disconnect', { method: 'POST' }).catch(() => {}).finally(() => disconnected('', url))
}

// The deck name in the header opens the list of decks.
$deckName.addEventListener('click', () => {
  if (!S.conn || S.conn.agent) return
  if (S.view === 'decks') { S.view = 'main'; render(); scheduleScan(0); return }
  S.adding = false
  S.view = 'decks'
  render()
})

$view.addEventListener('change', (ev) => {
  const el = ev.target
  if (el.id === 'agent' || el.id === 'setAgent') {
    S.agent = el.value
    S.lensMore = false
    keep({ agent: S.agent })
    keepLensChoice()
    if (el.id === 'agent') render()
    loadLenses(S.agent)
  } else if (el.dataset.file) {
    if (el.checked) S.checked.add(el.dataset.file)
    else S.checked.delete(el.dataset.file)
    S.touchedFiles = true
    render()
  } else if (el.dataset.inc) {
    S.include[el.dataset.inc] = el.checked
    render()
  } else if (el.id === 'prefSnapshot' || el.id === 'prefShot') {
    S.prefs = { ...S.prefs, snapshot: document.getElementById('prefSnapshot').checked, screenshot: document.getElementById('prefShot').checked }
    setStore({ prefs: S.prefs })
    applyDefaults()
  } else if (el.id === 'handsOn') {
    turnHandsOn(el.checked)
  } else if (el.id === 'fetchOn') {
    if (el.checked) allowFetch()
    else stopFetch().then(() => toast("Your agents don't read pages in this browser now"))
  } else if (el.id === 'searchOn') {
    if (el.checked) allowSearch(el.dataset.engine)
    else stopSearch(el.dataset.engine).then(() => toast("Your agents' searches don't run in this browser now"))
  } else if (el.id === 'handsFront') {
    S.hands = { ...S.hands, front: el.checked }
    setStore({ hands: S.hands })
  } else if (el.id === 'prefAll') {
    const want = el.checked
    const p = want ? chrome.permissions.request({ origins: ['<all_urls>'] }) : chrome.permissions.remove({ origins: ['<all_urls>'] })
    p.catch(() => false).then(refreshAllSites).then(render)
  }
})

$view.addEventListener('input', (ev) => {
  const el = ev.target
  if (el.id === 'note') S.note = el.value
  else if (el.id === 'lensText') S.perspectiveText = el.value
  else if (el.id === 'followUp') S.followUp = el.value
  else if (el.id === 'anonBack' && S.anon) {
    const was = !!S.anon.back.trim()
    S.anon.back = el.value
    if (was !== !!el.value.trim()) render()   // the button turns on/off
  }
  else if (el.id === 'deckUrl') S.deckInput = el.value
  else if (el.id === 'fileSearch') { S.fileQuery = el.value; render() }
})

$view.addEventListener('keydown', (ev) => {
  const el = ev.target
  if (ev.key === 'Escape' && el.id === 'fileSearch' && S.fileQuery) { ev.preventDefault(); S.fileQuery = ''; render(); return }
  if (ev.key !== 'Enter') return
  if (el.id === 'deckUrl') { ev.preventDefault(); startPairing() }
  else if ((ev.metaKey || ev.ctrlKey) && el.id === 'note') { ev.preventDefault(); onSend() }
  else if ((ev.metaKey || ev.ctrlKey) && el.id === 'followUp') { ev.preventDefault(); sendFollowUp() }
})

$view.addEventListener('toggle', (ev) => {
  if (ev.target.matches('details.include:not([data-rec-extras])')) S.includeOpen = ev.target.open
}, true)

$btnHistory.addEventListener('click', () => { S.view = S.view === 'history' ? 'main' : 'history'; render() })
$btnSettings.addEventListener('click', async () => {
  if (S.view === 'settings') { S.view = 'main'; render(); return }
  await refreshAllSites()
  S.view = 'settings'
  render()
  // What was set in Flight Deck since (the search engine picked …) shows here.
  if (S.conn && !S.conn.agent && Date.now() - (S.meAt || 0) > 10000) loadMe().then(render)
})

chrome.tabs.onActivated.addListener((info) => { if (!PINNED_TAB && info.windowId === S.windowId) scheduleScan() })
chrome.tabs.onUpdated.addListener((id, info, tab) => {
  if (PINNED_TAB ? id !== PINNED_TAB : (!tab.active || tab.windowId !== S.windowId)) return
  if (info.status === 'complete' || info.url) scheduleScan(info.status === 'complete' ? 200 : 600)
})
chrome.permissions.onAdded.addListener(() => { refreshAllSites(); refreshSearch(); refreshFetch(); scheduleScan(0) })
chrome.permissions.onRemoved.addListener(() => {
  refreshAllSites()
  Promise.all([refreshSearch(), refreshFetch()]).then(() => { if (S.view === 'main' || S.view === 'settings') render() })
})

// A right-click on a selection, a link or an image.
async function takeFocus() {
  let got
  try { got = await chrome.storage.session.get('focus') } catch { return }
  const f = got?.focus
  if (!f || Date.now() - f.at > 60000) return
  await chrome.storage.session.remove('focus').catch(() => {})
  const tab = await currentTab()
  if (!tab || tab.id !== f.tabId) return
  if (f.kind === 'anonymize') {
    // Right-click "Anonymize with Captain": the selection as it is, or the
    // page once it's read.
    if (S.conn && S.view !== 'run') {
      if (f.selection) {
        S.tab = tab
        return anonymize({ content: f.selection, source: 'selection' })
      }
      if (!S.scan || S.scanning || S.tab?.id !== tab.id) await scan()
      if (S.scan) return anonymize({ source: 'page' })
    }
    render()
    return
  }
  if (f.kind === 'selection' && f.selection) {
    S.focus = { kind: 'selection', tabId: f.tabId, label: `Selected: “${f.selection.slice(0, 140)}${f.selection.length > 140 ? '…' : ''}”` }
    S.include.selection = true
  } else if (f.kind === 'link' && f.link) {
    const name = decodeURIComponent((f.link.split(/[?#]/)[0].split('/').pop() || '')).slice(0, 120)
    if (/\.[a-z0-9]{2,5}$/i.test(name) && !/\.(html?|php|aspx?)$/i.test(name)) {
      S.extraFiles = [{ url: f.link, name, ext: name.split('.').pop().toLowerCase(), kind: 'document', context: 'The link you picked', same_origin: originOf(f.link) === originOf(tab.url), source: 'picked' }]
      S.checked.add(f.link)
      S.touchedFiles = true
    } else if (!S.note) {
      S.note = `About this link: ${f.link}`
    }
    S.focus = { kind: 'link', tabId: f.tabId, label: `Link: ${f.link}` }
  } else if (f.kind === 'image' && f.src && /^https?:/i.test(f.src)) {
    let name = decodeURIComponent((f.src.split(/[?#]/)[0].split('/').pop() || 'image')).slice(0, 120) || 'image'
    if (!/\.[a-z0-9]{2,5}$/i.test(name)) name += '.jpg'
    S.extraFiles = [{ url: f.src, name, ext: name.split('.').pop().toLowerCase(), kind: 'image', context: 'The picture you picked', same_origin: originOf(f.src) === originOf(tab.url), source: 'picked' }]
    S.checked.add(f.src)
    S.touchedFiles = true
    if (!S.note) S.note = 'About this picture:'
    S.focus = { kind: 'image', tabId: f.tabId, label: 'Picture you picked' }
  }
  if (S.view !== 'run' && S.conn) S.view = 'main'
  render()
}
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'session' && changes.focus?.newValue) takeFocus()
  if (area === 'session' && changes.rec) {
    const was = S.rec
    S.rec = changes.rec.newValue?.on ? changes.rec.newValue : null
    // Started from another window's panel: this one shows it too.
    if (S.view === 'rec' || (S.view === 'main' && (!!was !== !!S.rec || was?.steps.length !== S.rec?.steps.length))) render()
  }
  if (area === 'session' && (changes.handsLog || changes.handsStatus)) {
    if (changes.handsLog) S.handsLog = changes.handsLog.newValue || []
    if (changes.handsStatus) S.handsStatus = changes.handsStatus.newValue || null
    followHands()
    if (S.view === 'main' || S.view === 'settings' || S.view === 'decks') render()
  }
  if (area === 'local' && (changes.fetchModes || changes.fetchPrefs)) {
    refreshFetch().then(() => { if (S.view === 'main' || S.view === 'settings') render() })
  }
  if (area === 'local' && (changes.search || changes.searchPicks)) {
    refreshSearch().then(() => { if (S.view === 'main' || S.view === 'settings') render() })
  }
  // A deck connected or disconnected in another window's panel.
  if (area === 'local' && changes.decks) decksChanged()
  if (area === 'local' && (changes.hands || changes.handsStopped)) {
    if (changes.hands) S.hands = { on: false, front: false, ...(changes.hands.newValue || {}) }
    if (changes.handsStopped) S.handsStopped = changes.handsStopped.newValue || []
    if (S.view === 'main' || S.view === 'settings' || S.view === 'rec-run' || S.view === 'pb') render()
  }
  // A size picked in another window's panel applies here too.
  if (area === 'local' && changes.prefs?.newValue && changes.prefs.newValue.textSize !== S.prefs.textSize) {
    S.prefs = { ...S.prefs, textSize: changes.prefs.newValue.textSize }
    applyTextSize()
    if (S.view === 'settings') render()
  }
})

async function decksChanged() {
  const { decks } = await loadDecks().catch(() => ({ decks: S.decks }))
  S.decks = decks
  if (!S.conn) {
    // Connected in another window while this one showed the connect form.
    if (decks.length && S.view === 'connect' && !S.pairing) await enterDeck(decks[decks.length - 1])
    return
  }
  const same = decks.find((d) => d.url === S.conn.url)
  if (!same) return disconnected('', S.conn.url)
  if (same.token !== S.conn.token) S.conn = same   // connected again: the new token
  if (['main', 'settings', 'decks', 'history'].includes(S.view)) render()
}

// A panel left open: when it's looked at again (and a couple of minutes went
// by), ask Flight Deck anew — a newer version, or a choice made there (the
// search engine picked) shows up then.
const ME_STALE_MS = 2 * 60 * 1000
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' || !S.conn || S.conn.agent || !S.meAt) return
  if (Date.now() - S.meAt < ME_STALE_MS) return
  S.meAt = Date.now()
  loadMe().then(render)
})

// ── start ──

async function start() {
  try { S.windowId = (await chrome.windows.getCurrent()).id } catch { S.windowId = null }
  const stored = await getStore(['agent', 'prefs', 'history', 'lastDeck', 'me', 'lenses', 'decksKept', 'hands', 'handsStopped', 'recDraft', 'verSkip'])
  S.draft = stored.recDraft && Array.isArray(stored.recDraft.steps) ? stored.recDraft : null
  S.verSkip = stored.verSkip || ''
  S.prefs = { snapshot: false, screenshot: false, textSize: 'small', watchEvery: 'daily', ...(stored.prefs || {}) }
  applyTextSize()
  S.history = Array.isArray(stored.history) ? stored.history : []
  S.deckInput = stored.lastDeck || ''
  S.hands = { on: false, front: false, ...(stored.hands || {}) }
  S.handsStopped = Array.isArray(stored.handsStopped) ? stored.handsStopped : []
  try {
    const sess = await chrome.storage.session.get(['handsLog', 'handsStatus', 'rec', 'recNotes'])
    S.rec = sess.rec?.on ? sess.rec : null
    S.recNotes = sess.recNotes && typeof sess.recNotes === 'object' ? sess.recNotes : {}
    S.handsLog = Array.isArray(sess.handsLog) ? sess.handsLog : []
    S.handsStatus = sess.handsStatus || null
    S.handsSeen = S.handsLog.length ? S.handsLog[S.handsLog.length - 1].at : 0
  } catch { /* no session storage */ }
  await refreshAllSites()
  const { decks, use } = await loadDecks().catch(() => ({ decks: [], use: null }))
  S.decks = decks
  S.kept = stored.decksKept && typeof stored.decksKept === 'object' ? stored.decksKept : {}
  if (use && !stored.decksKept && (stored.agent || stored.me || stored.lenses)) {
    // Kept from before decks: it was all about the one deck there was.
    keep({ agent: stored.agent || '', me: stored.me || null, lenses: stored.lenses || {} }, use.url)
    chrome.storage.local.remove(['agent', 'me', 'lenses']).catch(() => {})
  }
  if (!use) {
    await refreshSearch()
    await refreshFetch()
    S.view = 'connect'
    render()
    return
  }
  // An agent's own browser (its token is for the agent's browser tool only)
  // shows just that; a recording going on is what the panel is for right now.
  await enterDeck(use)
  takeFocus()
}

start()
