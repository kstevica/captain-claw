// End-to-end: the HUD on a 600x600 "Meta Ray-Ban Display" against a real
// Flight Deck (auth on) + mock agent. Drives it with arrow keys / Enter like
// the Neural Band. Writes screenshots + a JSON report next to this file.
// Playwright is not a Flight Deck dependency: use a global install
// (NODE_PATH) or point PLAYWRIGHT_MODULE at one.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
const fs = require('fs')
const path = require('path')

const BASE = process.env.BASE || 'http://127.0.0.1:25190'
const RUN = process.env.E2E_RUN_DIR || path.join(__dirname, '.run')
const OUT = path.join(RUN, 'shots')
const UA = 'Mozilla/5.0 (Linux; Android 14; Greatwhite Build/UKQ1.250303.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/146.0.7680.177 Safari/537.36'
const EMAIL = 'pilot@example.com', PASSWORD = 'correct-horse-battery'
fs.mkdirSync(OUT, { recursive: true })

const results = []
function check(name, ok, detail) {
  results.push({ name, ok: !!ok, detail })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== undefined ? '  — ' + JSON.stringify(detail) : ''}`)
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// A signed-in browser session: Bearer token + the fd_refresh cookie (approve
// and lookup require both since the pairing hardening).
async function apiLogin() {
  const r = await fetch(`${BASE}/fd/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: EMAIL, password: PASSWORD }) })
  const cookie = (r.headers.getSetCookie() || []).map((c) => c.split(';')[0]).join('; ')
  return { token: (await r.json()).access_token, cookie }
}

async function active(page) {
  return page.evaluate(() => {
    const el = document.activeElement
    if (!el || el === document.body) return { tag: 'body' }
    return { tag: el.tagName, cls: el.className, fk: el.closest('[data-fk]')?.getAttribute('data-fk') || null, text: (el.textContent || el.value || '').trim().slice(0, 60) }
  })
}

async function press(page, key, n = 1, gap = 120) {
  for (let i = 0; i < n; i++) { await page.keyboard.press(key); await sleep(gap) }
}

async function shot(page, name) { await page.screenshot({ path: path.join(OUT, name + '.png') }) }

async function overflow(page) {
  return page.evaluate(() => {
    const s = document.querySelector('.hud-scroll')
    return { doc: document.documentElement.scrollWidth - document.documentElement.clientWidth, scroll: s ? s.scrollWidth - s.clientWidth : null }
  })
}

// Simulate Meta's composer committing text: one value set, then input + change.
async function composerCommit(page, selector, text) {
  await page.evaluate(({ selector, text }) => {
    const el = document.querySelector(selector)
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, text)
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }, { selector, text })
}

async function focusFk(page, fk) {
  await page.evaluate((fk) => { const el = document.querySelector(`[data-fk="${fk}"]`); el && el.focus() }, fk)
}

;(async () => {
  const browser = await chromium.launch()
  const consoleErrors = []

  // ── Glasses ──
  const ctx = await browser.newContext({ viewport: { width: 600, height: 600 }, deviceScaleFactor: 1, userAgent: UA })
  const page = await ctx.newPage()
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()) })
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message))
  const firstLoad = []
  let firstLoadOpen = true
  const externalRequests = []
  page.on('request', (req) => { if (!req.url().startsWith(BASE)) externalRequests.push(req.url()) })
  page.on('response', async (res) => {
    if (!firstLoadOpen) return
    try {
      const h = await res.allHeaders()
      firstLoad.push({ url: res.url().replace(BASE, ''), status: res.status(), len: Number(h['content-length'] || 0) || (await res.body().catch(() => Buffer.alloc(0))).length })
    } catch { /* ignore */ }
  })

  await page.goto(BASE + '/')
  check('root redirects Display UA to /hud/', page.url().startsWith(BASE + '/hud/'), page.url())
  await page.waitForSelector('.hud-pair-code', { timeout: 15000 })
  firstLoadOpen = false
  const assetBytes = firstLoad.filter((r) => r.url.startsWith('/assets/') || r.url.startsWith('/hud')).reduce((a, r) => a + r.len, 0)
  check('first load: requests < 15 (excluding the / → /hud/ redirect)', firstLoad.filter((r) => r.url !== '/').length < 15, { count: firstLoad.length, urls: firstLoad.map((r) => r.url) })
  check('first load: html+assets bytes (uncompressed)', true, assetBytes)
  const head = await page.evaluate(() => ({
    capable: document.querySelector('meta[name="mrbd-web-app-capable"]')?.content,
    desc: !!document.querySelector('meta[name="description"]'),
    manifest: document.querySelector('link[rel="manifest"]')?.href,
  }))
  check('head has mrbd-web-app-capable + description', head.capable === 'yes' && head.desc, head)
  await sleep(300)
  await shot(page, '01-pair')
  const code = (await page.textContent('.hud-pair-code')).trim()
  check('pair code shown as XXXX-XXXX', /^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$/.test(code), code)
  check('pair screen initial focus on a focusable', (await active(page)).tag !== 'body', await active(page))

  // Approve from "the phone" via API.
  const { token, cookie } = await apiLogin()
  const bearerOnly = await fetch(`${BASE}/fd/auth/pair/approve`, { method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ user_code: code, approve: true }) })
  check('approve with only a Bearer token is refused (needs the browser session)', bearerOnly.status === 403, bearerOnly.status)
  const lk = await fetch(`${BASE}/fd/auth/pair/lookup?code=${encodeURIComponent(code.toLowerCase().replace('-', ''))}`, { headers: { Authorization: `Bearer ${token}`, Cookie: cookie } })
  const info = await lk.json()
  check('lookup (lowercase, no dash) returns device info', lk.status === 200 && /Meta Ray-Ban Display/.test(info.label), info)
  const ap = await fetch(`${BASE}/fd/auth/pair/approve`, { method: 'POST', headers: { Authorization: `Bearer ${token}`, Cookie: cookie, 'Content-Type': 'application/json' }, body: JSON.stringify({ user_code: code, approve: true }) })
  check('approve ok', ap.status === 200, await ap.json())

  await page.waitForSelector('.hud-row', { timeout: 15000 })
  await sleep(600)
  await shot(page, '02-agents')
  const cookies = await ctx.cookies()
  check('refresh cookie set on glasses after pairing', cookies.some((c) => c.name === 'fd_refresh'), cookies.map((c) => c.name))
  check('agents: focus lands on Scout row', /Scout/.test((await active(page)).text || ''), await active(page))

  // ── Chat ──
  await press(page, 'Enter')
  await page.waitForSelector('.hud-composer-field', { timeout: 10000 })
  await sleep(800)
  await shot(page, '03-chat-empty')
  check('chat: composer textarea has initial focus', (await active(page)).tag === 'TEXTAREA', await active(page))
  await composerCommit(page, '.hud-composer-field', 'What did we ship this week?')
  await sleep(150)
  const afterCommit = await active(page)
  check('chat: composer commit moves focus to Send', /hud-composer-send/.test(afterCommit.cls || ''), afterCommit)
  await press(page, 'Enter') // the composer's late pinch — must be swallowed
  await sleep(100)
  const sentEarly = await page.$$eval('.hud-msg--user', (els) => els.length)
  check('chat: late pinch within quiet window does not send', sentEarly === 0, sentEarly)
  await sleep(800)
  await press(page, 'Enter')
  await page.waitForSelector('.hud-msg--user', { timeout: 5000 })
  await sleep(200)
  await shot(page, '04-chat-thinking')
  await page.waitForSelector('.hud-msg--assistant', { timeout: 10000 })
  await sleep(800)
  await shot(page, '05-chat-reply')
  const frames = fs.readFileSync(path.join(RUN, 'frames.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  const firstChat = frames.find((f) => f.type === 'chat')
  check('agent got surface=glasses + rules block on first message', firstChat && firstChat.surface === 'glasses' && firstChat.content.startsWith('[SYSTEM CONTEXT') && firstChat.content.endsWith('What did we ship this week?'), firstChat && { surface: firstChat.surface, head: firstChat.content.slice(0, 40), tail: firstChat.content.slice(-30) })
  const userText = await page.textContent('.hud-msg--user')
  check('chat: user bubble does not show the rules block', !/SYSTEM CONTEXT/.test(userText), userText.slice(0, 80))
  const cards = await page.$$eval('.hud-msg--assistant .hud-md-card', (e) => e.length)
  check('chat: 4-column table in reply rendered as cards', cards >= 2, cards)
  check('chat: no horizontal overflow', (await overflow(page)).scroll <= 0 && (await overflow(page)).doc <= 0, await overflow(page))
  const chips = await page.$$eval('.hud-chat-file', (e) => e.map((x) => x.textContent))
  check('chat: file chip for saved/report.md', chips.some((t) => /report\.md/.test(t)), chips)
  const steps = await page.$$eval('.hud-chat-steps .hud-btn', (e) => e.map((x) => x.textContent))
  check('chat: next-step chips shown', steps.length >= 1, steps)
  // Read the reply upward from the composer.
  await page.evaluate(() => document.querySelector('.hud-composer-field').focus())
  const before = await page.evaluate(() => document.querySelector('.hud-scroll').scrollTop)
  await press(page, 'ArrowUp', 6, 150)
  const after = await page.evaluate(() => document.querySelector('.hud-scroll').scrollTop)
  await shot(page, '06-chat-reading')
  check('chat: arrows move through the transcript (scroll changes)', before !== after, { before, after, active: await active(page) })

  // File chip → file screen → Back.
  await page.evaluate(() => { const c = [...document.querySelectorAll('.hud-chat-file')].find((x) => /report\.md/.test(x.textContent)); c && c.focus() })
  await press(page, 'Enter')
  await page.waitForSelector('.hud-md', { timeout: 10000 })
  await sleep(500)
  check('file chip opens report.md', /report\.md/.test(await page.textContent('.hud-title-main')), await page.textContent('.hud-title-main'))
  await page.goBack()
  await page.waitForSelector('.hud-composer-field', { timeout: 10000 })
  await sleep(700)
  check('Back from file returns to chat (transcript kept)', (await page.$$('.hud-msg--assistant')).length >= 1)

  // ── Files tab ──
  await sleep(800)
  await focusFk(page, 'tab-files')
  await press(page, 'Enter')
  await page.waitForSelector('.hud-row', { timeout: 10000 })
  await sleep(700)
  await shot(page, '07-files')
  const fileRows = await page.$$eval('.hud-row .hud-row-title', (e) => e.map((x) => x.textContent))
  check('files: only markdown, newest first', fileRows[0] === 'report.md' && fileRows.includes('notes.md') && !fileRows.some((t) => /csv/.test(t)), fileRows)
  const notesMeta = await page.$$eval('.hud-row', (rows) => rows.map((r) => r.textContent).find((t) => /notes\.md/.test(t)))
  check('files: member file labelled with creator', /Dana/.test(notesMeta || ''), notesMeta)
  check('files: first row focused', /report\.md/.test((await active(page)).text || ''), await active(page))
  await press(page, 'Enter')
  await page.waitForSelector('.hud-md .hud-block', { timeout: 10000 })
  await sleep(700)
  await shot(page, '08-file-top')
  const md = await page.evaluate(() => ({
    cards: document.querySelectorAll('.hud-md-card').length,
    narrow: document.querySelectorAll('.hud-md-table table').length,
    imgs: document.querySelectorAll('.hud-md img').length,
    links: document.querySelectorAll('.hud-md a').length,
    blocks: document.querySelectorAll('.hud-md .hud-block').length,
  }))
  check('file: wide table → cards, narrow table kept, no <img>/<a>', md.cards === 3 && md.narrow === 1 && md.imgs === 0 && md.links === 0, md)
  check('file: no horizontal overflow', (await overflow(page)).scroll <= 0 && (await overflow(page)).doc <= 0, await overflow(page))
  await press(page, 'ArrowDown', 5, 150)
  await shot(page, '09-file-down')
  const st1 = await page.evaluate(() => document.querySelector('.hud-scroll').scrollTop)
  await press(page, 'ArrowRight', 1, 400)
  const st2 = await page.evaluate(() => document.querySelector('.hud-scroll').scrollTop)
  await shot(page, '10-file-pageturn')
  const act2 = await active(page)
  const focVisible = await page.evaluate(() => {
    const s = document.querySelector('.hud-scroll').getBoundingClientRect(); const r = document.activeElement.getBoundingClientRect()
    return r.bottom > s.top && r.top < s.bottom
  })
  check('file: ArrowRight turns the page and focus stays on screen', st2 > st1 && focVisible, { st1, st2, act2, focVisible })
  await page.goBack()
  await page.waitForSelector('.hud-row', { timeout: 10000 })
  await sleep(900)
  check('Back to files list restores focus on report.md', /report\.md/.test((await active(page)).text || ''), await active(page))

  // ── Data tab ──
  await sleep(400)
  await focusFk(page, 'tab-data')
  await press(page, 'Enter')
  await page.waitForSelector('.hud-row', { timeout: 10000 })
  await sleep(700)
  await shot(page, '11-tables')
  const tables = await page.$$eval('.hud-row .hud-row-title', (e) => e.map((x) => x.textContent))
  check('tables: listed, most recently updated first', tables[0] === 'contacts' && tables.includes('empty_table'), tables)
  await press(page, 'Enter')
  await page.waitForFunction(() => document.querySelectorAll('.hud-row').length >= 8, null, { timeout: 10000 })
  await sleep(700)
  await shot(page, '12-rows-p1')
  const p1 = await page.$$eval('.hud-row .hud-row-title', (e) => e.map((x) => x.textContent))
  check('rows: page 1 = 8 newest records', p1.length === 8 && p1[0] === 'Person 23', p1)
  await press(page, 'ArrowRight', 1, 900)
  const p2 = await page.$$eval('.hud-row .hud-row-title', (e) => e.map((x) => x.textContent))
  await shot(page, '13-rows-p2')
  check('rows: ArrowRight → next page', p2[0] === 'Person 15', p2)
  await press(page, 'ArrowDown', 2, 150)
  const target = await active(page)
  await press(page, 'Enter')
  await page.waitForSelector('.hud-block', { timeout: 10000 })
  await sleep(800)
  await shot(page, '14-record')
  const sub1 = await page.textContent('.hud-title-sub')
  const recText = await page.textContent('.hud-scroll')
  check('record: opens with formatted fields', /Person 1[0-9]/.test(recText) && /(Yes|No)/.test(recText), { sub1, target })
  await press(page, 'ArrowRight', 1, 900)
  const sub2 = await page.textContent('.hud-title-sub')
  await shot(page, '15-record-next')
  check('record: ArrowRight → next record', sub1 !== sub2, { sub1, sub2 })
  check('record: no horizontal overflow', (await overflow(page)).scroll <= 0, await overflow(page))
  await page.goBack()
  await page.waitForFunction(() => document.querySelectorAll('.hud-row').length >= 1, null, { timeout: 10000 })
  await sleep(900)
  check('Back from record → rows with a row focused', /hud-row/.test((await active(page)).cls || ''), await active(page))
  await shot(page, '16-rows-back')

  // ── Approval round-trip ──
  await page.goBack() // rows → agent (data tab)
  await sleep(1100) // past the post-Back focus pin (a real swipe would clear it)
  await focusFk(page, 'tab-chat')
  await press(page, 'Enter')
  await page.waitForSelector('.hud-composer-field', { timeout: 10000 })
  await sleep(600)
  await page.evaluate(() => document.querySelector('.hud-composer-field').focus())
  await composerCommit(page, '.hud-composer-field', 'please approve the ls')
  await sleep(1000) // past the composer's 800 ms late-pinch guard
  await press(page, 'Enter')
  await page.waitForSelector('.hud-chat-approval', { timeout: 10000 })
  await sleep(400)
  await shot(page, '17-approval')
  const approveBtn = await page.$('.hud-chat-approval .hud-btn--primary')
  await approveBtn.focus()
  await press(page, 'Enter')
  await page.waitForFunction(() => /Approved: ran/.test(document.querySelector('.hud-scroll').textContent), null, { timeout: 10000 })
  check('approval: Approve sends approval_response and reply arrives', true)
  await sleep(400)
  await shot(page, '18-after-approval')

  // Agents list via Back.
  await page.goBack()
  await sleep(800)
  check('Back from agent → agents list', /Agents/.test(await page.textContent('.hud-title-main')), await page.textContent('.hud-title-main'))
  check('no external (non-deck) requests from the HUD', externalRequests.length === 0, externalRequests)
  check('no console errors on glasses page (the pre-pairing refresh 401 is expected)', consoleErrors.filter((t) => !/status of 401/.test(t)).length === 0, consoleErrors)

  // ── Phone: /hud/pair approve page ──
  const st = await (await fetch(`${BASE}/fd/auth/pair/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label: 'Test glasses' }) })).json()
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true })
  const pp = await phone.newPage()
  const ppErrors = []
  pp.on('pageerror', (e) => ppErrors.push(e.message))
  await pp.goto(`${BASE}/hud/pair?code=${st.user_code}`)
  await pp.waitForSelector('input[type="password"]', { timeout: 15000 })
  await pp.screenshot({ path: path.join(OUT, '20-phone-login.png') })
  await pp.fill('input[type="email"]', EMAIL)
  await pp.fill('input[type="password"]', PASSWORD)
  await pp.click('button[type="submit"], .hud-pp-btn--primary')
  await pp.waitForSelector('.hud-pp-code-input, .hud-pp-code-show', { timeout: 15000 })
  await sleep(500)
  await pp.screenshot({ path: path.join(OUT, '21-phone-code.png') })
  // The page never prefills ?code= (phishing hardening): the wearer types it.
  check('approve page does not prefill the code from the URL', (await pp.inputValue('.hud-pp-code-input')) === '', await pp.inputValue('.hud-pp-code-input'))
  await pp.fill('.hud-pp-code-input', st.user_code)
  await pp.click('.hud-pp-btn--primary')
  await pp.waitForFunction(() => /Test glasses/.test(document.body.textContent), null, { timeout: 15000 })
  await sleep(300)
  await pp.screenshot({ path: path.join(OUT, '22-phone-review.png') })
  await pp.click('.hud-pp-btn--primary')
  await pp.waitForFunction(() => /signed in/i.test(document.body.textContent), null, { timeout: 15000 })
  await pp.screenshot({ path: path.join(OUT, '23-phone-done.png') })
  const poll = await (await fetch(`${BASE}/fd/auth/pair/poll`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device_code: st.device_code }) })).json()
  check('phone approve page → device poll approved', poll.status === 'approved' && !!poll.access_token, poll.status)
  check('no page errors on phone page', ppErrors.length === 0, ppErrors)

  await browser.close()
  fs.writeFileSync(path.join(RUN, 'report.json'), JSON.stringify({ results, consoleErrors }, null, 2))
  const failed = results.filter((r) => !r.ok)
  console.log(`\n${results.length - failed.length}/${results.length} passed`)
  process.exit(failed.length ? 1 : 0)
})().catch((e) => { console.error('E2E CRASH', e); fs.writeFileSync(path.join(RUN, 'report.json'), JSON.stringify({ results, crash: String(e && e.stack || e) }, null, 2)); process.exit(2) })
