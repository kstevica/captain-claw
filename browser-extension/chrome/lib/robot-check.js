// Captain for Chrome — telling a robot check from a page, and handing one to
// the person.
//
// A page an agent's web search or web_fetch opened in a background tab
// (search.js, pages.js) can be a site asking to prove a person is there:
// Google's "unusual traffic" page, Cloudflare's "Just a moment…", a reCAPTCHA,
// hCaptcha or Turnstile box, DataDome, PerimeterX's "Press & hold", Akamai,
// Imperva, Amazon's "Enter the characters you see", DDoS-Guard … Then the tab
// isn't closed: it comes out of its collapsed group to the front of its window,
// the window comes forward, and Flight Deck hears it (the agent's chat says
// it's waiting for the person). Once the check is gone — the person answered
// it — the agent's search or read goes on in the same tab, and the tab closes
// with the person's own tab back in front. Not answered in time, or the agent
// stopped waiting: the tab stays open for the person, and the agent goes on
// without it. Closed by the person: the agent goes on without it, and that
// site isn't handed over again for a while.
//
// pageFacts runs in the page (self-contained: chrome.scripting sends only its
// source); judge reads the facts here and is plain, so tests run it in node.

const SHORT_TEXT = 2500          // a check's page says little; a long page with a captcha box is a form
const POLL_MS = 1000
const GRACE_MS = 7000            // checks that pass by themselves (Cloudflare's "Just a moment…")
const CLEAR_READS = 2            // the check gone this many polls in a row: answered
const DECLINED_MS = 10 * 60 * 1000   // the person closed the tab: not again for that site for a while
const FOR_MAX_MS = 10 * 60 * 1000

// Titles robot checks use, whole …
const WALL_TITLES = new Set([
  'just a moment', 'just a moment...', 'just a moment…', 'one moment, please...', 'please wait...', 'attention required!',
  'attention required! | cloudflare', 'robot check', 'security check', 'human verification', 'are you a robot?',
  'are you a robot', 'bot verification', 'captcha', 'captcha challenge', 'pardon our interruption', 'checking your browser',
  'verify you are human', 'please verify you are a human', 'client challenge', 'ddos-guard', 'verifying...',
  'verifying you are human', 'security verification', 'are you human?', 'human check',
])
// … bits of titles distinctive enough anywhere in one …
const WALL_TITLE_BITS = [
  'just a moment', 'checking your browser', 'verify you are human', 'are you a robot', 'ddos-guard',
  'pardon our interruption', 'checking if the site connection is secure',
]
// … and bits that count only on a short page ("How CAPTCHA works" is an article).
const WEAK_TITLE_BITS = ['captcha', 'robot check', 'human verification', 'security check', 'are you human']
// Text of a check's page (only on a short page: an article may quote these).
const WALL_TEXT = [
  ['enable javascript and cookies to continue', 'Cloudflare'],
  ['checking if the site connection is secure', 'Cloudflare'],
  ['needs to review the security of your connection', 'Cloudflare'],
  ['verify you are human by completing the action below', 'Cloudflare'],
  ['press & hold', 'PerimeterX'],
  ['press and hold', 'PerimeterX'],
  ['pardon our interruption', 'Imperva'],
  ['checking your browser before accessing', ''],
  ['verify you are human', ''],
  ['verifying you are human', ''],
  ['please verify you are a human', ''],
  ['please complete the security check', ''],
  ['unusual traffic from your computer', 'Google'],
  ['our systems have detected unusual traffic', 'Google'],
  ['enter the characters you see below', 'Amazon'],
  ['type the characters you see in this image', ''],
  ['are you a robot', ''],
  ["i'm not a robot", ''],
  ['not a robot', ''],
  ['confirm you are human', ''],
  ['prove you are human', ''],
  ['complete the captcha', ''],
  ['solve the captcha', ''],
]
// Pages that only say no: nothing for a person to answer there.
const BLOCK_TEXT = ['sorry, you have been blocked', 'you have been blocked', 'access denied', 'the requested url was rejected',
  'you don\'t have permission to access', 'access to this page has been denied']
// A check's own frames (an invisible reCAPTCHA on an ordinary page isn't one).
const WALL_FRAMES = [
  ['challenges.cloudflare.com', 'Cloudflare'],
  ['/cdn-cgi/challenge-platform', 'Cloudflare'],
  ['captcha-delivery.com', 'DataDome'],
  ['hcaptcha.com', 'hCaptcha'],
  ['/recaptcha/', 'reCAPTCHA'],
  ['recaptcha.net', 'reCAPTCHA'],
  ['arkoselabs.com', 'Arkose'],
  ['funcaptcha.com', 'Arkose'],
  ['geetest.com', 'GeeTest'],
  ['perimeterx.net', 'PerimeterX'],
  ['px-cloud.net', 'PerimeterX'],
]
// A check's own boxes, by selector (looked up in the page by pageFacts).
export const WALL_SELECTORS = {
  '#captcha-form': 'Google',
  'form[action*="/sorry/"]': 'Google',
  '#challenge-form': 'Cloudflare',
  '#challenge-stage': 'Cloudflare',
  '#challenge-running': 'Cloudflare',
  '#cf-challenge-running': 'Cloudflare',
  '.cf-turnstile': 'Cloudflare',
  '#turnstile-wrapper': 'Cloudflare',
  '.g-recaptcha': 'reCAPTCHA',
  '.h-captcha': 'hCaptcha',
  '#px-captcha': 'PerimeterX',
  '#sec-if-cpt-container': 'Akamai',
  '#sec-cpt-if': 'Akamai',
  'form[action*="validateCaptcha"]': 'Amazon',
  '#captcha-container': '',
  '#captcha': '',
  '.captcha': '',
  '#b_captcha': 'Bing',
  '.anomaly-modal__modal': 'DuckDuckGo',
}
// Checks that usually pass on their own in a real browser within seconds.
const PASS_BY_THEMSELVES = new Set(['Cloudflare', 'DDoS-Guard', 'Imperva'])

/** Runs in the page: what judge needs, nothing more. */
export function pageFacts(selectors) {
  const text = (document.body?.innerText || '').replace(/\s+/g, ' ').trim()
  const frames = [...document.querySelectorAll('iframe')]
    .map((f) => f.src || f.getAttribute('data-src') || '')
    .filter((s) => s && !/[?&]size=invisible\b/.test(s))
    .slice(0, 40)
  const boxes = []
  for (const sel of selectors || []) {
    try {
      const el = document.querySelector(sel)
      // A box that's there but hidden (display:none in a page's footer form) isn't asking anything.
      if (el && el.getClientRects().length) boxes.push(sel)
    } catch { /* a selector this page can't take */ }
  }
  return { url: location.href, title: document.title || '', text: text.slice(0, 4000), length: text.length, frames, boxes,
           ready: document.readyState }
}

/** ``{check, block, vendor, why, passes}`` — check: a person could answer it;
 * block: the page only says no; passes: it usually goes by itself. */
export function judge(facts) {
  const none = { check: false, block: false, vendor: '', why: '', passes: false }
  if (!facts || typeof facts !== 'object') return none
  let url
  try { url = new URL(String(facts.url || '')) } catch { url = null }
  const title = String(facts.title || '').replace(/\s+/g, ' ').trim().toLowerCase()
  const text = String(facts.text || '').toLowerCase()
  const short = Number(facts.length ?? text.length) < SHORT_TEXT
  const frames = Array.isArray(facts.frames) ? facts.frames.map((f) => String(f).toLowerCase()) : []
  const boxes = Array.isArray(facts.boxes) ? facts.boxes.map(String) : []
  // "Just a moment…", "Checking your browser": an interstitial that usually goes by itself.
  const passing = /^(just a moment|one moment|please wait|checking your browser|verifying)/.test(title)
  const out = (vendor, why) => ({ check: true, block: false, vendor, why, passes: passing || PASS_BY_THEMSELVES.has(vendor) })

  // Google's "unusual traffic" page lives at /sorry/ on any of its hosts.
  if (url && /(^|\.)google\.[a-z.]+$/.test(url.hostname) && url.pathname.startsWith('/sorry/')) {
    return out('Google', 'Google asks to confirm a person is searching')
  }
  if (url && url.pathname.startsWith('/cdn-cgi/challenge-platform')) return out('Cloudflare', 'Cloudflare checks the browser')
  const titled = WALL_TITLES.has(title) || WALL_TITLE_BITS.some((b) => title.includes(b))
  const weakTitle = WEAK_TITLE_BITS.some((b) => title.includes(b))
  const frame = WALL_FRAMES.find(([mark]) => frames.some((f) => f.includes(mark)))
  const box = boxes.find((sel) => sel in WALL_SELECTORS)
  const said = WALL_TEXT.find(([mark]) => text.includes(mark))
  const vendor = (frame && frame[1]) || (box && WALL_SELECTORS[box]) || (said && said[1]) ||
    (title.includes('cloudflare') || text.includes('performance & security by cloudflare') ? 'Cloudflare' : '') ||
    (title.includes('ddos-guard') || text.includes('ddos-guard') ? 'DDoS-Guard' : '')
  const named = vendor ? ` (${vendor})` : ''
  if (titled && (short || frame || box)) return out(vendor, `the page is a robot check${named}`)
  if (short && weakTitle) return out(vendor, `the page is a robot check${named}`)
  if (short && (frame || box || said)) {
    return out(vendor, frame || box ? `the page shows a robot check${named}` : `the page asks to prove a person is there${named}`)
  }
  if (short && BLOCK_TEXT.some((b) => text.includes(b) || title.includes(b))) {
    return { check: false, block: true, vendor, why: `the site refuses this browser${named}`, passes: false }
  }
  return none
}

/** The check on *tabId*'s page now — ``null`` when the page can't be read yet
 * (loading, or on a site Chrome keeps this extension off). */
export async function robotCheck(tabId) {
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId }, func: pageFacts, args: [Object.keys(WALL_SELECTORS)] })
    if (!res?.result) return null
    return { ...judge(res.result), url: res.result.url, title: res.result.title, ready: res.result.ready }
  } catch {
    return null
  }
}

export function siteOf(href) {
  try { return new URL(href).hostname.replace(/^www\./, '') } catch { return '' }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** A check that goes by itself (Cloudflare's "Just a moment…") gets a few
 * seconds first: true when the page is clear by then. */
export async function passesByItself(tabId, isCancelled) {
  const until = Date.now() + GRACE_MS
  let clear = 0
  while (Date.now() < until) {
    if (isCancelled()) return false
    await sleep(POLL_MS)
    const now = await robotCheck(tabId)
    clear = now && !now.check && now.ready === 'complete' ? clear + 1 : 0
    if (clear >= CLEAR_READS) return true
  }
  return false
}

// ── handing a check to the person ──

const handing = new Map()     // site → { tabId, done: Promise<outcome> }
const declined = new Map()    // site → until (the person closed the tab)
const shown = new Map()       // tabId → { windowId, before, ownWindow }

/** Whether the person turned this site's check away a moment ago. */
export function declinedNow(site) {
  return (declined.get(site) || 0) > Date.now()
}

/**
 * Hand the check on *tabId* to the person and wait until it's gone.
 * Outcome: 'solved' | 'timeout' | 'closed' | 'cancelled' | 'declined' |
 * 'again' (another tab's check on this site was answered: look at this one again).
 * opts: { site, forMs, isCancelled, cleared(tabId) → Promise<bool>, tell(msg) }
 */
export async function handOver(tabId, opts) {
  const site = opts.site || ''
  if (declinedNow(site)) return 'declined'
  const other = handing.get(site)
  if (other && other.tabId !== tabId) {
    // One check per site at a time in front of the person: wait for that one
    // (Flight Deck hears this read waits too, so it doesn't give up on it).
    opts.tell?.({ type: 'waiting', site, vendor: opts.vendor || '', why: opts.why || '', joined: true,
                  for_s: Math.round(Math.min(opts.forMs, FOR_MAX_MS) / 1000) })
    const outcome = await settledOrCancelled(other.done, opts.isCancelled)
    opts.tell?.({ type: 'waiting', site, over: outcome, joined: true })
    if (outcome !== 'solved') return outcome
    await chrome.tabs.reload(tabId).catch(() => {})
    return 'again'
  }
  let finish
  const done = new Promise((r) => { finish = r })
  handing.set(site, { tabId, done })
  let outcome = 'timeout'
  const started = Date.now()
  try {
    await bringForward(tabId)
    opts.tell?.({ type: 'waiting', site, vendor: opts.vendor || '', why: opts.why || '',
                  for_s: Math.round(Math.min(opts.forMs, FOR_MAX_MS) / 1000) })
    outcome = await watch(tabId, Math.min(opts.forMs, FOR_MAX_MS), opts)
  } catch {
    outcome = 'closed'
  } finally {
    if (outcome === 'closed') declined.set(site, Date.now() + DECLINED_MS)
    // Not answered: the tab stays open for the person (the caller doesn't close it).
    const was = shown.get(tabId)
    if (was) was.keep = outcome === 'timeout' || outcome === 'cancelled'
    handing.delete(site)
    finish(outcome)
    opts.tell?.({ type: 'waiting', site, over: outcome, waited_s: Math.round((Date.now() - started) / 1000) })
  }
  return outcome
}

async function watch(tabId, forMs, opts) {
  const until = Date.now() + forMs
  let clear = 0
  while (Date.now() < until) {
    if (opts.isCancelled()) return 'cancelled'
    await sleep(POLL_MS)
    const tab = await chrome.tabs.get(tabId).catch(() => null)
    if (!tab) return 'closed'
    if (tab.status !== 'complete') { clear = 0; continue }
    const now = await robotCheck(tabId)
    clear = now && !now.check && (!opts.cleared || (await opts.cleared(tabId, now))) ? clear + 1 : 0
    if (clear >= CLEAR_READS) return 'solved'
  }
  return 'timeout'
}

async function settledOrCancelled(done, isCancelled) {
  let timer = 0
  const cancelled = new Promise((resolve) => {
    timer = setInterval(() => { if (isCancelled()) resolve('cancelled') }, POLL_MS)
  })
  try {
    return await Promise.race([done, cancelled])
  } finally {
    clearInterval(timer)
  }
}

// Out of its collapsed group, in front in its window, the window forward.
async function bringForward(tabId) {
  const tab = await chrome.tabs.get(tabId)
  const [before] = await chrome.tabs.query({ active: true, windowId: tab.windowId }).catch(() => [])
  shown.set(tabId, { windowId: tab.windowId, before: before && before.id !== tabId ? before.id : null })
  if (tab.groupId !== undefined && tab.groupId >= 0) await chrome.tabs.ungroup(tabId).catch(() => {})
  await chrome.tabs.update(tabId, { active: true, muted: false })
  const win = await chrome.windows.get(tab.windowId).catch(() => null)
  if (win?.state === 'minimized') await chrome.windows.update(tab.windowId, { state: 'normal' }).catch(() => {})
  await chrome.windows.update(tab.windowId, { focused: true, drawAttention: true }).catch(() => {})
}

/** Was *tabId* handed to the person? Then the tab it took the front from. */
export function shownTab(tabId) {
  return shown.get(tabId) || null
}

/** Done with a tab that was handed over and answered: the person's own tab
 * goes back in front (if they're still on this one). */
export async function giveBack(tabId) {
  const was = shown.get(tabId)
  shown.delete(tabId)
  if (!was?.before) return
  const tab = await chrome.tabs.get(tabId).catch(() => null)
  if (tab?.active) await chrome.tabs.update(was.before, { active: true }).catch(() => {})
}

/** Forget a tab left open for the person (not answered in time). */
export function letGo(tabId) {
  shown.delete(tabId)
}
