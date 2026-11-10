// Captain for Chrome — the side panel.
//
// Reads the page (capture.js), fetches the files it links to with the
// person's own browser session, and sends it all to Flight Deck
// (/fd/clip/send), which saves it in the Web clips shared folder and hands
// it to the chosen agent. Then it follows the agent's progress and answer.

import { icon } from './lib/icons.js'
import { renderMarkdown, escapeHtml as esc } from './lib/md.js'
import { AuthError, api, browserLabel, clearConn, loadConn, normalizeDeckUrl, publicCall, saveConn, upload } from './lib/api.js'

const VERSION = chrome.runtime.getManifest().version
const HISTORY_MAX = 30
const DOC_KINDS = new Set(['document', 'archive', 'data', 'design'])
const AUTO_FILES_MAX = 20
const FETCH_PARALLEL = 3

const $view = document.getElementById('view')
const $deckName = document.getElementById('deckName')
const $btnHistory = document.getElementById('btnHistory')
const $btnSettings = document.getElementById('btnSettings')
const $toast = document.getElementById('toast')
$btnHistory.innerHTML = icon('history')
$btnSettings.innerHTML = icon('settings')

const S = {
  view: 'loading',
  conn: null,
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
  for (const id of ['steps', 'fileList']) {
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
}

function render() {
  const connected = !!S.conn
  $btnHistory.hidden = !connected || S.view === 'connect'
  $btnSettings.hidden = !connected || S.view === 'connect'
  $deckName.textContent = connected ? (S.me?.deck?.name ? `${S.me.deck.name} · ${hostOf(S.conn.url)}` : hostOf(S.conn.url)) : 'Not connected'
  const before = listScroll()
  renderView()
  restoreScroll(before)
  if (S.view === 'run') followRun()
}

function renderView() {
  keepFocus(() => {
    if (S.view === 'connect') $view.innerHTML = viewConnect()
    else if (S.view === 'main') $view.innerHTML = viewHands() + viewMain()
    else if (S.view === 'run') $view.innerHTML = viewRun()
    else if (S.view === 'history') $view.innerHTML = viewHistory()
    else if (S.view === 'settings') $view.innerHTML = viewSettings()
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
  return `
    <div class="card">
      <div style="font-weight:600;font-size:calc(14px * var(--fs));margin-bottom:4px">Send any page to your agents</div>
      <p class="muted" style="margin:0">Summarize it, keep it with its files, pull out its data or ask about it — your agent works on it in Flight Deck, and the page lands in your <b>Web clips</b> shared folder.</p>
    </div>
    <div class="card">
      <label class="label" for="deckUrl">Your Flight Deck address</label>
      <input type="url" id="deckUrl" placeholder="captain.example.com" value="${esc(S.deckInput)}" autocomplete="url">
      ${S.connectError ? `<div class="notice err" style="margin-top:8px">${esc(S.connectError)}</div>` : ''}
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
        S.conn = { url: p.deck, token: r.token, user: r.user || {} }
        await saveConn(S.conn)
        toast('Connected')
        await afterConnect()
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
  try {
    S.me = await api(S.conn, '/fd/clip/me')
    S.meError = ''
    await setStore({ me: S.me })
    if (!agentDef(S.agent)) {
      const firstUp = S.me.agents.find((a) => a.status === 'running') || S.me.agents.find((a) => a.status === 'asleep')
      S.agent = firstUp ? firstUp.slug : ''
    }
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    S.meError = e.message
  }
}

async function disconnected(why) {
  await clearConn()
  S.conn = null
  S.me = null
  S.lenses = {}
  setStore({ lenses: {} })
  S.view = 'connect'
  S.connectError = why || ''
  render()
}

async function afterConnect() {
  S.view = 'main'
  render()
  await loadMe()
  applyDefaults()
  render()
  scan()
  loadLenses()
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
  if (!slug || !S.conn) return
  clearTimeout(lensTimer)
  let tries = 0
  const ask = async () => {
    if (S.agent !== slug || !S.conn) return
    try {
      const r = await api(S.conn, `/fd/clip/agents/${encodeURIComponent(slug)}/lenses${fresh && tries === 0 ? '?fresh=1' : ''}`)
      S.lenses[slug] = { ...r, at: Date.now() }
      setStore({ lenses: S.lenses })
    } catch (e) {
      if (e instanceof AuthError) return disconnected(e.message)
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
    <div class="foot">${icon('folder', 'sm')} Saved in Shared folders → ${esc(S.me.folder || 'web-clips')}</div>`
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

function handsName(slug) {
  return agentDef(slug)?.name || [...S.handsLog].reverse().find((e) => e.slug === slug)?.name || slug
}

function handsStatusText() {
  const st = S.handsStatus?.state
  if (st === 'on') return ''
  if (st === 'permission') return 'Chrome hasn\'t given Captain the permission yet.'
  if (st === 'denied') return S.handsStatus.error || 'Flight Deck refused this browser — connect it again.'
  if (st === 'connecting') return S.handsStatus.error || 'Connecting to Flight Deck…'
  if (st === 'error') return S.handsStatus.error || 'Not connected.'
  return 'Starting…'
}

function viewHands() {
  if (!S.hands.on) return ''
  const now = Date.now()
  const latest = new Map()
  for (const e of S.handsLog) if (now - e.at < HANDS_RECENT_MS) latest.set(e.slug, e)
  const working = [...latest.values()].filter((e) => !S.handsStopped.includes(e.slug)).reverse()
  const status = handsStatusText()
  const rows = working.map((e) => `
    <div class="hands-agent">
      <span class="status-pill ${e.state === 'working' ? 'working' : e.state === 'failed' ? 'error' : 'done'}"><span class="dot"></span>${esc(e.name)}</span>
      <div class="grow ellipsis" title="${esc(e.error || e.text)}">${esc(e.text)}${e.state === 'failed' ? ' — didn\'t work' : ''} <span class="dim">· ${esc(agoText(e.at))}</span></div>
      <button class="link-btn" data-act="hands-show" data-slug="${esc(e.slug)}">Show</button>
      <button class="link-btn danger" data-act="hands-stop" data-slug="${esc(e.slug)}">Stop</button>
    </div>`).join('')
  const stoppedRows = S.handsStopped.map((slug) => `
    <div class="hands-agent"><div class="grow dim">${esc(handsName(slug))} is stopped — it can't use this browser.</div>
      <button class="link-btn" data-act="hands-let" data-slug="${esc(slug)}">Let it again</button></div>`).join('')
  const past = S.handsLog.slice(-15).reverse()
  const history = past.length ? `
    <details class="hands-log">
      <summary class="dim">What agents did here</summary>
      ${past.map((e) => `<div class="step ${e.state === 'failed' ? 'err' : ''}"><b>${esc(e.name)}</b>&nbsp;${esc(e.text)}${e.state === 'failed' ? ` — ${esc(e.error || 'failed')}` : ''} <span class="dim">· ${esc(agoText(e.at))}</span></div>`).join('')}
    </details>` : ''
  return `
    <div class="card hands ${working.length ? 'active' : ''}">
      <div class="row">${icon('globe', 'sm')}<div class="grow"><b>${working.length ? 'An agent is using this browser' : 'Agents can use this browser'}</b>
        ${status ? `<div class="dim">${esc(status)}</div>` : ''}</div>
        ${S.handsStatus?.state === 'permission' ? '<button class="btn" data-act="hands-allow" style="padding:3px 9px">Allow</button>' : ''}</div>
      ${rows}${stoppedRows}${history}
    </div>`
}

async function handsShow(slug) {
  const got = await chrome.storage.session.get('handsGroups').catch(() => ({}))
  const g = got.handsGroups?.[slug]
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
  if (want) {
    let ok = false
    try { ok = await chrome.permissions.request({ permissions: ['debugger', 'tabGroups'] }) } catch { ok = false }
    if (!ok) { toast('Chrome needs your OK for Captain to work in tabs for your agents.'); render(); return }
  }
  S.hands = { ...S.hands, on: want }
  await setStore({ hands: S.hands })
  render()
}

// A new agent at work here becomes the panel's agent: talk to it, follow it.
function followHands() {
  const last = S.handsLog[S.handsLog.length - 1]
  if (!last || last.at <= S.handsSeen) return
  // Only the first step of a stretch of work: picking another agent meanwhile sticks.
  const fresh = !S.handsLog.some((e) => e !== last && e.slug === last.slug && last.at - e.at < HANDS_RECENT_MS)
  S.handsSeen = last.at
  if (!fresh || last.slug === S.agent || S.sending || S.view === 'run' || !agentDef(last.slug)) return
  S.agent = last.slug
  S.lensMore = false
  setStore({ agent: S.agent })
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

function startRun(run) {
  S.run = { ...run, status: 'starting', events: [], next: 0, result: '', live: '', error: '', files: [], done: false, at: Date.now() }
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
    try {
      const r = await api(S.conn, `/fd/clip/runs/${encodeURIComponent(run.clip_id)}?since=${run.next}`)
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
      if (e instanceof AuthError) { run.polling = false; return disconnected(e.message) }
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
        <div class="dim ellipsis">${esc(run.agent)} · ${esc(hostOf(run.url))}</div>
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
    ${files ? `<div class="card"><div class="label">In the clip folder</div><div class="out-files">${files}</div>
      <div class="dim" style="margin-top:6px">${esc(run.vfs || '')}</div></div>` : ''}
    ${run.done ? `
      <div class="row">
        ${run.result ? `<button class="btn grow" data-act="copy">${icon('copy', 'sm')} Copy answer</button>` : ''}
        <button class="btn grow" data-act="open-deck">${icon('external', 'sm')} Open Flight Deck</button>
      </div>
      <div class="row" style="justify-content:center">${newTopicButton(run.agentSlug)}</div>
      <div>
        <textarea id="followUp" rows="2" placeholder="Ask a follow-up about this page…">${esc(S.followUp)}</textarea>
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
  try {
    const res = await upload(S.conn, '/fd/clip/send', form, () => {})
    startRun({ clip_id: res.clip_id, folder: res.folder, vfs: res.vfs, agent: res.agent || run.agent, agentSlug: run.agentSlug,
               title: run.title, url: run.url, action: 'ask', actionLabel: 'Follow-up' })
  } catch (e) {
    if (e instanceof AuthError) return disconnected(e.message)
    toast(e.message)
  }
}

async function saveRunFile(path) {
  const run = S.run
  try {
    const r = await api(S.conn, `/fd/clip/runs/${encodeURIComponent(run.clip_id)}/file?path=${encodeURIComponent(path)}`, { raw: true })
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

// ── history ──

async function remember(run) {
  const item = {
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
      <span class="dim ellipsis">${esc(h.agent)} · ${esc(hostOf(h.url))} · ${esc(new Date(h.at).toLocaleString())}</span>
    </button>`).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back">${icon('back')}</button><div class="grow" style="font-weight:600">Recent</div>
      ${S.history.length ? '<button class="btn" data-act="clear-history" style="padding:3px 8px">Clear</button>' : ''}</div>
    ${items || '<div class="dim">Nothing sent yet.</div>'}`
}

function openHistory(i) {
  const h = S.history[i]
  if (!h) return
  S.run = { ...h, events: [], next: 0, live: '', done: !['starting', 'thinking'].includes(h.status), followedEnd: true }
  S.follow = true
  S.view = 'run'
  render()
  if (!S.run.done) pollRun(S.run)
}

// ── settings ──

function viewSettings() {
  const u = S.conn?.user || {}
  const agents = (S.me?.agents || []).map((a) => `<option value="${esc(a.slug)}" ${a.slug === S.agent ? 'selected' : ''}>${esc(a.name)}</option>`).join('')
  return `
    <div class="row"><button class="icon-btn" data-act="back">${icon('back')}</button><div class="grow" style="font-weight:600">Settings</div></div>
    <div class="card">
      <div class="label">Connected</div>
      <div>${esc(hostOf(S.conn?.url || ''))}</div>
      <div class="dim">${esc(u.name || '')}${u.email && u.email !== u.name ? ` · ${esc(u.email)}` : ''}</div>
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
      <label class="switch"><input type="checkbox" id="handsOn" ${S.hands.on ? 'checked' : ''}><div>Agents can use this browser<div class="dim">Your agents' browser tool works here, signed in as you — in tabs of their own, grouped under the agent's name. Chrome shows that Captain is debugging the browser while one works. Flight Deck's own pages stay off-limits.</div></div></label>
      <label class="switch"><input type="checkbox" id="handsFront" ${S.hands.front ? 'checked' : ''} ${S.hands.on ? '' : 'disabled'}><div>Bring an agent's tab to the front while it works</div></label>
      ${S.hands.on && handsStatusText() ? `<div class="dim" style="margin-top:4px">${esc(handsStatusText())}</div>` : ''}
    </div>
    <div class="card">
      <div class="label">Every time</div>
      <label class="switch"><input type="checkbox" id="prefAll" ${S.allSites ? 'checked' : ''}><div>Work on every site without asking<div class="dim">Reads pages and fetches their files anywhere; also lets screenshots work.</div></div></label>
      <label class="switch"><input type="checkbox" id="prefSnapshot" ${S.prefs.snapshot ? 'checked' : ''}><div>Always send the whole page (.mhtml)</div></label>
      <label class="switch"><input type="checkbox" id="prefShot" ${S.prefs.screenshot ? 'checked' : ''}><div>Always send a screenshot</div></label>
    </div>
    <button class="btn block" data-act="open-deck">${icon('external', 'sm')} Open Flight Deck</button>
    <button class="btn block danger" data-act="disconnect">Disconnect this browser</button>
    <div class="foot">Captain for Chrome ${esc(VERSION)}</div>`
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
  const summary = ev.target.closest('details.include > summary')
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
  else if (act === 'open-task' && S.conn && S.run?.watch?.id) chrome.tabs.create({ url: `${S.conn.url}/#task=${encodeURIComponent(S.run.watch.id)}` })
  else if (act === 'back') { S.view = 'main'; render(); scheduleScan(0) }
  else if (act === 'stop' && S.run) {
    api(S.conn, `/fd/clip/runs/${encodeURIComponent(S.run.clip_id)}/stop`, { method: 'POST' }).catch((e) => toast(e.message))
  } else if (act === 'copy' && S.run?.result) {
    navigator.clipboard.writeText(S.run.result).then(() => toast('Copied'), () => toast("Couldn't copy"))
  } else if (act === 'open-deck' && S.conn) chrome.tabs.create({ url: `${S.conn.url}/` })
  else if (act === 'follow-up') sendFollowUp()
  else if (act === 'hands-show') handsShow(el.dataset.slug)
  else if (act === 'hands-stop') setHandsStopped([...S.handsStopped, el.dataset.slug]).then(() => toast(`${handsName(el.dataset.slug)} can't use this browser now`))
  else if (act === 'hands-let') setHandsStopped(S.handsStopped.filter((s) => s !== el.dataset.slug))
  else if (act === 'hands-allow') turnHandsOn(true)
  else if (act === 'clear-history') { S.history = []; setStore({ history: [] }); render() }
  else if (act === 'disconnect') {
    if (!confirm('Disconnect this browser from Flight Deck?')) return
    api(S.conn, '/fd/clip/disconnect', { method: 'POST' }).catch(() => {}).finally(() => disconnected(''))
  }
})

$view.addEventListener('change', (ev) => {
  const el = ev.target
  if (el.id === 'agent' || el.id === 'setAgent') {
    S.agent = el.value
    S.lensMore = false
    setStore({ agent: S.agent })
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
  if (ev.target.matches('details.include')) S.includeOpen = ev.target.open
}, true)

$btnHistory.addEventListener('click', () => { S.view = S.view === 'history' ? 'main' : 'history'; render() })
$btnSettings.addEventListener('click', async () => {
  if (S.view === 'settings') { S.view = 'main'; render(); return }
  await refreshAllSites()
  S.view = 'settings'
  render()
})

chrome.tabs.onActivated.addListener((info) => { if (!PINNED_TAB && info.windowId === S.windowId) scheduleScan() })
chrome.tabs.onUpdated.addListener((id, info, tab) => {
  if (PINNED_TAB ? id !== PINNED_TAB : (!tab.active || tab.windowId !== S.windowId)) return
  if (info.status === 'complete' || info.url) scheduleScan(info.status === 'complete' ? 200 : 600)
})
chrome.permissions.onAdded.addListener(() => { refreshAllSites(); scheduleScan(0) })
chrome.permissions.onRemoved.addListener(() => refreshAllSites())

// A right-click on a selection, a link or an image.
async function takeFocus() {
  let got
  try { got = await chrome.storage.session.get('focus') } catch { return }
  const f = got?.focus
  if (!f || Date.now() - f.at > 60000) return
  await chrome.storage.session.remove('focus').catch(() => {})
  const tab = await currentTab()
  if (!tab || tab.id !== f.tabId) return
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
  if (area === 'session' && (changes.handsLog || changes.handsStatus)) {
    if (changes.handsLog) S.handsLog = changes.handsLog.newValue || []
    if (changes.handsStatus) S.handsStatus = changes.handsStatus.newValue || null
    followHands()
    if (S.view === 'main' || S.view === 'settings') render()
  }
  if (area === 'local' && (changes.hands || changes.handsStopped)) {
    if (changes.hands) S.hands = { on: false, front: false, ...(changes.hands.newValue || {}) }
    if (changes.handsStopped) S.handsStopped = changes.handsStopped.newValue || []
    if (S.view === 'main' || S.view === 'settings') render()
  }
  // A size picked in another window's panel applies here too.
  if (area === 'local' && changes.prefs?.newValue && changes.prefs.newValue.textSize !== S.prefs.textSize) {
    S.prefs = { ...S.prefs, textSize: changes.prefs.newValue.textSize }
    applyTextSize()
    if (S.view === 'settings') render()
  }
})

// ── start ──

async function start() {
  try { S.windowId = (await chrome.windows.getCurrent()).id } catch { S.windowId = null }
  const stored = await getStore(['agent', 'prefs', 'history', 'lastDeck', 'me', 'lenses', 'hands', 'handsStopped'])
  S.agent = stored.agent || ''
  S.prefs = { snapshot: false, screenshot: false, textSize: 'small', watchEvery: 'daily', ...(stored.prefs || {}) }
  applyTextSize()
  S.history = Array.isArray(stored.history) ? stored.history : []
  S.deckInput = stored.lastDeck || ''
  S.lenses = stored.lenses && typeof stored.lenses === 'object' ? stored.lenses : {}
  S.hands = { on: false, front: false, ...(stored.hands || {}) }
  S.handsStopped = Array.isArray(stored.handsStopped) ? stored.handsStopped : []
  try {
    const sess = await chrome.storage.session.get(['handsLog', 'handsStatus'])
    S.handsLog = Array.isArray(sess.handsLog) ? sess.handsLog : []
    S.handsStatus = sess.handsStatus || null
    S.handsSeen = S.handsLog.length ? S.handsLog[S.handsLog.length - 1].at : 0
  } catch { /* no session storage */ }
  await refreshAllSites()
  S.conn = await loadConn()
  if (!S.conn) {
    S.view = 'connect'
    render()
    return
  }
  if (stored.me) S.me = stored.me   // drawn at once; refreshed below
  S.view = 'main'
  render()
  await loadMe()
  applyDefaults()
  render()
  loadLenses()
  await scan()
  takeFocus()
}

start()
