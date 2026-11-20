// Captain for Chrome — recording what the person does (the worker's side).
//
// The side panel starts and stops a recording; the page script
// (recorder-page.js) tells each thing done; this keeps the list. The
// recording follows the window it was started in: the tab in front gets the
// page script (where the person let Captain read the site), an address typed
// or a tab switched to becomes a step, and a page a click led to is noted on
// that click.
//
// The recording lives in session storage ("rec"), so the panel can close and
// the worker can sleep without losing it. Only this file writes it — in one
// queue, since steps arrive from several frames at once. The person's notes
// on steps are the panel's own ("recNotes").

const KEY = 'rec'
const MAX_STEPS = 300
const CAUSED_MS = 5000     // a page change this soon after a step was that step's doing
const DOUBLE_MS = 700

let chain = Promise.resolve()
/** One at a time: read the recording, change it, write it. */
function locked(fn) {
  const run = chain.then(fn, fn)
  chain = run.catch(() => {})
  return run
}

async function load() {
  try { return (await chrome.storage.session.get(KEY))[KEY] || null } catch { return null }
}

async function save(rec) {
  await chrome.storage.session.set({ [KEY]: rec })
}

const isWeb = (url) => /^https?:/i.test(url || '')
const originOf = (url) => { try { return new URL(url).origin } catch { return '' } }
const sameSpot = (a, b) => String(a || '').split('#')[0] === String(b || '').split('#')[0]

function badge(on) {
  try {
    chrome.action.setBadgeBackgroundColor({ color: '#dc2626' })
    chrome.action.setBadgeText({ text: on ? 'REC' : '' })
  } catch { /* no toolbar button */ }
}

/** The page script into every frame of the tab Captain may reach:
 * 'ok', 'blocked' (the person hasn't let Captain read this site) or
 * 'unreadable' (a browser page). */
export async function inject(tabId) {
  let tab
  try { tab = await chrome.tabs.get(tabId) } catch { return 'unreadable' }
  if (!isWeb(tab.url) && !/^file:/i.test(tab.url || '')) return 'unreadable'
  try {
    await chrome.scripting.executeScript({ target: { tabId, allFrames: true }, files: ['recorder-page.js'] })
    return 'ok'
  } catch {
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ['recorder-page.js'] })
      return 'ok'
    } catch {
      return 'blocked'
    }
  }
}

function push(rec, step) {
  if (rec.steps.length >= MAX_STEPS) { rec.full = true; return null }
  rec.seq += 1
  const full = { id: `s${rec.seq}`, at: Date.now(), ...step }
  rec.steps.push(full)
  return full
}

const last = (rec) => rec.steps[rec.steps.length - 1] || null

async function injectFor(rec, tab) {
  const got = await inject(tab.id)
  rec.blocked = got === 'blocked' ? originOf(tab.url) : ''
}

// ── from the panel ──

async function start({ tabId }) {
  return locked(async () => {
    let tab
    try { tab = await chrome.tabs.get(tabId) } catch { return { ok: false, error: 'That tab is gone.' } }
    const rec = { on: true, id: `r${Date.now().toString(36)}`, windowId: tab.windowId, tab: tab.id, startedAt: Date.now(),
                  steps: [], seq: 0, blocked: '', full: false, actAt: 0, newTab: false }
    if (isWeb(tab.url)) push(rec, { kind: 'open', url: tab.url, title: tab.title || '', tabId: tab.id })
    await injectFor(rec, tab)
    await save(rec)
    await chrome.storage.session.remove('recNotes').catch(() => {})
    badge(true)
    return { ok: true }
  })
}

async function stop({ discard = false } = {}) {
  return locked(async () => {
    const rec = await load()
    if (!rec) return { ok: true }
    rec.on = false
    rec.stoppedAt = Date.now()
    if (discard) await chrome.storage.session.remove([KEY, 'recNotes']).catch(() => {})
    else await save(rec)
    badge(false)
    const tabs = await chrome.tabs.query({ windowId: rec.windowId }).catch(() => [])
    for (const t of tabs) chrome.tabs.sendMessage(t.id, { type: 'captain-rec', on: false }).catch(() => {})
    return { ok: true }
  })
}

async function removeStep({ id }) {
  return locked(async () => {
    const rec = await load()
    if (!rec) return { ok: true }
    rec.steps = rec.steps.filter((s) => s.id !== id)
    await save(rec)
    return { ok: true }
  })
}

/** The person just let Captain read the site in front: start seeing it. */
async function retry() {
  return locked(async () => {
    const rec = await load()
    if (!rec?.on) return { ok: true }
    const tab = await chrome.tabs.get(rec.tab).catch(() => null)
    if (tab) await injectFor(rec, tab)
    await save(rec)
    return { ok: true }
  })
}

// ── from the page ──

async function onStep(step, sender) {
  return locked(async () => {
    const rec = await load()
    const tab = sender.tab
    if (!rec?.on || !tab || tab.windowId !== rec.windowId) return { off: true }
    if (!step || typeof step.kind !== 'string') return {}
    const prev = last(rec)
    // The second click of a double click: the click before it, doubled.
    if (step.kind === 'click' && step.double && prev && prev.kind === 'click' && prev.tabId === tab.id &&
        prev.target?.selector === step.target?.selector && Date.now() - prev.at < DOUBLE_MS) {
      prev.double = true
      await save(rec)
      return {}
    }
    delete step.double
    const full = { ...step, tabId: tab.id, url: tab.url || '', title: tab.title || '' }
    if (sender.frameId) full.frame = String(sender.url || '').slice(0, 500)
    if (push(rec, full)) {
      rec.tab = tab.id
      rec.actAt = Date.now()
    }
    await save(rec)
    return rec.full ? { off: true } : {}
  })
}

// ── from the browser: pages and tabs ──

/** The tab in front shows another address. */
function moved(rec, tab, url) {
  const prev = last(rec)
  const caused = prev && prev.kind !== 'open' && prev.kind !== 'switch' && Date.now() - rec.actAt < CAUSED_MS
  if (caused) {
    if (sameSpot(prev.url, url) && !rec.newTab) return
    prev.leads_to = url
    if (rec.newTab) prev.new_tab = true
  } else if (prev && prev.kind === 'open' && prev.tabId === tab.id && Date.now() - prev.at < CAUSED_MS) {
    prev.url = url          // a redirect on the way in: where it ended
    prev.title = tab.title || ''
  } else if (!prev || !sameSpot(prev.leads_to || prev.url, url) || prev.tabId !== tab.id) {
    const step = { kind: 'open', url, title: tab.title || '', tabId: tab.id }
    if (rec.newTab) step.new_tab = true
    push(rec, step)
  }
  rec.newTab = false
}

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (!info.url && info.status !== 'complete' && !info.title) return
  locked(async () => {
    const rec = await load()
    if (!rec?.on || tab.windowId !== rec.windowId || !tab.active) return
    if (info.url && isWeb(info.url)) {
      rec.tab = tabId
      moved(rec, tab, info.url)
    }
    if (info.title) {
      const prev = last(rec)
      if (prev && prev.kind === 'open' && prev.tabId === tabId && sameSpot(prev.url, tab.url)) prev.title = info.title
    }
    if (info.status === 'complete') await injectFor(rec, tab)
    await save(rec)
  })
})

chrome.tabs.onActivated.addListener(({ tabId, windowId }) => {
  locked(async () => {
    const rec = await load()
    if (!rec?.on || windowId !== rec.windowId || tabId === rec.tab) return
    const tab = await chrome.tabs.get(tabId).catch(() => null)
    if (!tab) return
    const known = rec.steps.some((s) => s.tabId === tabId)
    rec.tab = tabId
    if (!isWeb(tab.url)) {
      rec.newTab = !known        // a new, empty tab: its first address says what it's for
    } else if (known) {
      rec.newTab = false
      // Back in the tab the last step was in (after a look elsewhere): nothing to tell.
      if (last(rec)?.tabId !== tabId) push(rec, { kind: 'switch', url: tab.url, title: tab.title || '', tabId })
    } else {
      rec.newTab = true          // a tab a click opened, or one of theirs brought into it
      moved(rec, tab, tab.url)
    }
    await injectFor(rec, tab)
    await save(rec)
  })
})

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (!msg || typeof msg.type !== 'string' || !msg.type.startsWith('captain-rec')) return false
  if (sender.id !== chrome.runtime.id) return false
  // The panel is one of the extension's own pages; a page script's sender is the web page.
  const fromPanel = String(sender.url || '').startsWith(chrome.runtime.getURL(''))
  let work
  if (msg.type === 'captain-rec-step') work = onStep(msg.step, sender)
  else if (msg.type === 'captain-rec-frames') work = sender.tab ? inject(sender.tab.id).then(() => ({})) : Promise.resolve({})
  else if (!fromPanel) return false          // the rest is the panel's to ask, never a page's
  else if (msg.type === 'captain-rec-start') work = start(msg)
  else if (msg.type === 'captain-rec-stop') work = stop(msg)
  else if (msg.type === 'captain-rec-remove') work = removeStep(msg)
  else if (msg.type === 'captain-rec-retry') work = retry()
  else return false
  work.then((r) => respond(r || {}), (e) => respond({ ok: false, error: String(e?.message || e) }))
  return true
})

/** The worker woke up: the toolbar says whether a recording is still on. */
export function startRecorder() {
  load().then((rec) => badge(!!rec?.on))
}
