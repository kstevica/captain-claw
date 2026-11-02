// Screenshots of the kiosk for its Getting started guide — or, with
// MODE=guide, of the guide itself plus a check of how it behaves. Run by
// run.sh against a throwaway Flight Deck it has set up. Playwright is not a
// Flight Deck dependency: use a global install (NODE_PATH) or point
// PLAYWRIGHT_MODULE at one.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright')
// The chat's message box (the kiosk's says "Ask anything…").
const CHAT_INPUT = 'textarea[placeholder^="Ask anything"], textarea[placeholder^="Message"]'
const fs = require('fs')
const path = require('path')

const BASE = process.env.BASE || 'http://127.0.0.1:25191'
const RUN = process.env.E2E_RUN_DIR || path.join(__dirname, '.run')
const MODE = process.env.MODE || 'assets'
// Which layouts to shoot (assets mode): the desktop one, the phone one, or both.
const FORMS = (process.env.FORMS || 'wide,phone').split(',')
const THEMES = (process.env.THEMES || 'dark,light').split(',')
const OUT = path.join(RUN, MODE === 'guide' ? 'guide' : 'png')
const EMAIL = 'maya@example.com', PASSWORD = 'correct-horse-battery'
fs.mkdirSync(OUT, { recursive: true })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

let token = ''
let current = null  // the page in use: shot on a failure
async function api(method, url, body) {
  const r = await fetch(BASE + url, {
    method,
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${method} ${url}: HTTP ${r.status} ${await r.text()}`)
  return r.json()
}

async function settings(theme, guideOptedOut) {
  await api('PUT', '/fd/settings', { settings: {
    'fd:theme': theme,
    // The panels as a newcomer finds them, except the queue: shown where it's the subject.
    'fd:simple-left-open': 'true', 'fd:simple-right-open': 'true', 'fd:simple-queue-open': 'false',
    'fd:onboarding': JSON.stringify({ completed: { desktop: false, forge: false, council: false, kiosk: guideOptedOut }, dismissedHints: [] }),
  } })
}

async function signIn(page) {
  await page.goto(BASE + '/')
  await page.fill('input[type=email]', EMAIL)
  await page.fill('input[type=password]', PASSWORD)
  await page.click('button[type=submit]')
  await page.getByText('Research Assistant').first().waitFor()
}

// A dialog cut off below `bottom` (the last thing that matters in it).
async function shootDialog(page, dialog, name, bottom, pad = 20) {
  const box = await dialog.boundingBox()
  let height = box.height
  if (bottom) {
    const b = await bottom.boundingBox()
    height = Math.min(box.height, b.y + b.height - box.y + pad)
  }
  await page.screenshot({ path: path.join(OUT, name + '.png'), clip: { x: box.x, y: box.y, width: box.width, height } })
}

// A context for one pass. Flight Deck's copy of each chat stays out of it: the
// queued task the dark pass sends would otherwise come back in the light one.
async function passContext(browser, theme, phone) {
  const context = await browser.newContext(phone
    ? { viewport: { width: 390, height: 700 }, deviceScaleFactor: 2, colorScheme: theme, isMobile: true, hasTouch: true }
    : { viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2, colorScheme: theme })
  await context.route('**/fd/chat/sessions**', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: route.request().method() === 'GET' ? '[]' : '{}' }))
  return context
}

// The archetype picker with Deep Researcher picked (its name filled in, not
// selected: a caret and a highlight read as noise in a picture).
async function pickArchetype(page, theme) {
  const spawn = page.locator('[role=dialog][aria-label="New agent from an archetype"]')
  await spawn.locator('ul button').first().waitFor()
  if (theme === 'dark') console.log('archetypes:', (await spawn.locator('ul button span.font-semibold').allTextContents()).join(' | '))
  const pick = spawn.locator('ul button', { hasText: /research/i }).first()
  const target = (await pick.count()) ? pick : spawn.locator('ul button').first()
  await target.click()
  // On a phone the detail has replaced the list by now.
  if (await target.count()) await target.scrollIntoViewIfNeeded()
  await spawn.locator('input[aria-label="Agent name"]').evaluate((el) => { el.setSelectionRange(0, 0); el.blur() })
  await sleep(300)
  return spawn
}

async function assets(browser) {
  for (const theme of THEMES) {
    await settings(theme, true)
    if (FORMS.includes('phone')) await phoneAssets(browser, theme)
    if (!FORMS.includes('wide')) continue
    const context = await passContext(browser, theme, false)
    const page = current = await context.newPage()
    await signIn(page)

    // The deck with a conversation open.
    await page.locator('aside button', { hasText: 'Research Assistant' }).first().click()
    await page.getByText('position against').waitFor()
    await page.getByText('ebike-subscriptions.md').first().waitFor()
    await sleep(800)
    await page.screenshot({ path: path.join(OUT, `overview-${theme}.png`) })

    // At work: the first queued task under way (the demo agent never finishes
    // it), the next one waiting, a question typed, files on the right.
    await page.locator('button[data-fd-queue-toggle]').click()
    const queue = page.locator('textarea[aria-label="New task"]')
    for (const task of ['Draft a one-page pricing memo for the board from the comparison',
      'Check Dance and Rebike prices again next Monday and tell me what changed']) {
      await queue.fill(task)
      await queue.press('Enter')
    }
    await queue.evaluate((el) => el.blur())
    await page.locator(CHAT_INPUT).fill('Which of the three has the best repair service in Munich?')
    await sleep(500)
    const main = await page.locator('main').first().boundingBox()
    await page.screenshot({ path: path.join(OUT, `chat-${theme}.png`), clip: { x: main.x, y: main.y, width: 1280 - main.x, height: main.height } })
    await page.locator(CHAT_INPUT).fill('')

    // New agent: an archetype picked, its name filled in.
    await page.locator('button[title="New agent from an archetype"]').first().click()
    await shootDialog(page, await pickArchetype(page, theme), `new-agent-${theme}`)
    await page.keyboard.press('Escape')

    // Settings.
    await page.locator('aside button', { hasText: 'Settings' }).click()
    const set = page.locator('[role=dialog][aria-label="Settings"]')
    await set.waitFor()
    await sleep(300)
    await shootDialog(page, set, `settings-${theme}`)

    // Profile, filled in: all three fields, so a taller window for it.
    await page.setViewportSize({ width: 1280, height: 1150 })
    await set.getByRole('button', { name: /Profile/ }).click()
    const prof = page.locator('[role=dialog][aria-label="Profile"]')
    await prof.getByText('My standing preferences').waitFor()
    await sleep(600)
    await shootDialog(page, prof, `profile-${theme}`, prof.locator('textarea').nth(2), 40)
    await page.keyboard.press('Escape')
    await page.setViewportSize({ width: 1280, height: 800 })

    // Connections: Google, then Telegram.
    await page.locator('aside button', { hasText: 'Settings' }).click()
    await set.getByRole('button', { name: /Connections/ }).click()
    const conn = page.locator('[role=dialog][aria-label="Connections"]')
    await conn.locator('#connections-tab-google').click()
    const google = conn.locator('#connections-panel-google')
    await google.getByRole('button', { name: 'Connect Google' }).waitFor()
    await sleep(500)
    await shootDialog(page, conn, `google-${theme}`, google.locator('> div').first())

    await conn.locator('#connections-tab-telegram').click()
    const tg = conn.locator('#connections-panel-telegram')
    // Before Connect: the bot's one-time link needs a live bot, which this deck hasn't got.
    await tg.getByRole('button', { name: 'Connect', exact: true }).waitFor()
    await sleep(500)
    await shootDialog(page, conn, `telegram-${theme}`, tg.locator('> div').first())
    await context.close()
  }
}

// The same screens in the phone layout: each one full screen, as a phone shows it.
async function phoneAssets(browser, theme) {
  const context = await passContext(browser, theme, true)
  const page = current = await context.newPage()
  const shoot = async (name) => { await sleep(500); await page.screenshot({ path: path.join(OUT, `${name}-phone-${theme}.png`) }) }
  await signIn(page)
  await shoot('overview')

  await page.getByRole('button', { name: 'New agent', exact: true }).click()
  await pickArchetype(page, theme)
  await shoot('new-agent')
  await page.locator('[role=dialog][aria-label="New agent from an archetype"]').getByRole('button', { name: 'Close' }).click()

  await page.locator('button', { hasText: 'Research Assistant' }).first().click()
  await page.getByText('position against').waitFor()
  await page.locator(CHAT_INPUT).fill('Which of the three has the best repair service in Munich?')
  await page.locator(CHAT_INPUT).evaluate((el) => el.blur())
  await shoot('chat')
  await page.getByRole('button', { name: 'Back to agents' }).click()

  const openSettings = async () => {
    await page.getByRole('button', { name: 'Settings', exact: true }).click()
    const set = page.locator('[role=dialog][aria-label="Settings"]')
    await set.waitFor()
    return set
  }
  await openSettings()
  await shoot('settings')
  await page.locator('[role=dialog][aria-label="Settings"]').getByRole('button', { name: /Profile/ }).click()
  const prof = page.locator('[role=dialog][aria-label="Profile"]')
  await prof.getByText('My company').waitFor()
  await shoot('profile')
  await prof.getByRole('button', { name: 'Close' }).first().click()

  await (await openSettings()).getByRole('button', { name: /Connections/ }).click()
  const conn = page.locator('[role=dialog][aria-label="Connections"]')
  await conn.locator('#connections-tab-google').click()
  await conn.locator('#connections-panel-google').getByRole('button', { name: 'Connect Google' }).waitFor()
  await shoot('google')
  await conn.locator('#connections-tab-telegram').click()
  await conn.locator('#connections-panel-telegram').getByRole('button', { name: 'Connect', exact: true }).waitFor()
  await shoot('telegram')
  await context.close()
}

const results = []
function check(name, ok, detail) {
  results.push({ name, ok: !!ok })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== undefined ? '  — ' + JSON.stringify(detail) : ''}`)
}

// How one step of the guide sits on a phone: nothing wider than the screen, the
// footer on one row, the title in view without scrolling, tap targets of 44 px.
async function phoneLayout(g) {
  return g.evaluate((dialog) => {
    const vw = window.innerWidth
    const vh = window.innerHeight
    const page = dialog.querySelector('.overflow-y-auto:not(nav)')
    const footer = dialog.lastElementChild
    const title = dialog.querySelector('h3')
    const img = page.querySelector('img')
    const buttons = [...footer.querySelectorAll('button')].map((b) => Math.round(b.getBoundingClientRect().height))
    const wide = [...dialog.querySelectorAll('*')].some((el) => el.getBoundingClientRect().right > vw + 1)
    return {
      overflow: wide || page.scrollWidth > page.clientWidth || document.documentElement.scrollWidth > vw,
      footerRows: footer.getBoundingClientRect().height > 80 ? 2 : 1,
      titleInView: title.getBoundingClientRect().bottom <= footer.getBoundingClientRect().top,
      phoneShot: img ? img.currentSrc.includes('-phone-') : null,
      tap: Math.min(...buttons),
      vh,
    }
  })
}

// A finger dragged sideways across the page.
async function swipe(page, g, dx) {
  await g.evaluate((dialog, dx) => {
    const el = dialog.querySelector('.overflow-y-auto:not(nav)')
    const r = el.getBoundingClientRect()
    const at = (x) => new Touch({ identifier: 1, target: el, clientX: x, clientY: r.top + r.height / 2 })
    const x0 = r.left + r.width / 2 - dx / 2
    el.dispatchEvent(new TouchEvent('touchstart', { bubbles: true, touches: [at(x0)], changedTouches: [at(x0)] }))
    el.dispatchEvent(new TouchEvent('touchend', { bubbles: true, touches: [], changedTouches: [at(x0 + dx)] }))
  }, dx)
  await sleep(300)
}

async function guide(browser) {
  const guideDialog = (page) => page.locator('[role=dialog][aria-label="Getting started"]')
  const runs = [
    ['wide', { width: 1280, height: 800 }, 'dark'],
    ['wide', { width: 1280, height: 800 }, 'light'],
    ['phone', { width: 390, height: 844 }, 'dark'],
    ['phone', { width: 390, height: 844 }, 'light'],
    ['phone-small', { width: 360, height: 640 }, 'dark'],
    ['phone-se', { width: 375, height: 667 }, 'light'],
    ['phone-landscape', { width: 844, height: 390 }, 'dark'],
  ]
  for (const [form, viewport, theme] of runs) {
    const phone = form.startsWith('phone')
    await settings(theme, false)
    const context = await browser.newContext({ viewport, deviceScaleFactor: 2, colorScheme: theme, hasTouch: phone, isMobile: phone })
    const page = current = await context.newPage()
    await signIn(page)
    const g = guideDialog(page)
    await g.waitFor({ timeout: 10000 }).catch(() => {})
    check(`${form}/${theme}: opens by itself on first visit`, await g.isVisible())
    await sleep(1200)  // /fd/info: steps for what the deck offers
    const total = Number((await g.locator('h2 + span').textContent()).split(' of ')[1])
    check(`${form}/${theme}: all seven topics on a deck with Google, Microsoft and Telegram`, total === 7, total)
    const bad = []
    for (let i = 1; i <= total; i++) {
      await sleep(400)
      await page.screenshot({ path: path.join(OUT, `${form}-${theme}-${i}.png`) })
      const img = g.locator('img').first()
      if (await img.count()) {
        const loaded = await img.evaluate((el) => el.complete && el.naturalWidth > 100)
        if (!loaded) bad.push(`step ${i}: screenshot didn't load`)
      }
      if (phone) {
        const l = await phoneLayout(g)
        if (l.overflow) bad.push(`step ${i}: wider than the screen`)
        if (l.footerRows !== 1) bad.push(`step ${i}: footer wraps`)
        if (l.phoneShot === false) bad.push(`step ${i}: desktop screenshot on a phone`)
        if (l.tap < 44) bad.push(`step ${i}: footer button ${l.tap}px tall`)
        if (!l.titleInView) bad.push(`step ${i}: title below the fold`)
      }
      if (i < total) await g.getByRole('button', { name: 'Next', exact: true }).click()
    }
    check(`${form}/${theme}: every step${phone ? ' fits the phone and' : ''} loads its screenshot`, bad.length === 0, bad.length ? bad : undefined)
    if (form === 'phone' && theme === 'dark') {
      // Swipe back and forth; enlarge shows the phone screenshot across the screen.
      await swipe(page, g, 160)
      check('swipe right goes back a step', (await g.locator('h2 + span').textContent()).startsWith('6 '))
      await swipe(page, g, -160)
      check('swipe left goes forward a step', (await g.locator('h2 + span').textContent()).startsWith('7 '))
      await g.locator('.overflow-y-auto').first().evaluate((el) => el.scrollBy(0, 40))
      await swipe(page, g, 30)
      check('a small drag does not turn the page', (await g.locator('h2 + span').textContent()).startsWith('7 '))
      await g.getByRole('button', { name: 'Enlarge screenshot' }).click()
      await sleep(400)
      const zoomed = await page.locator('img.shadow-2xl').evaluate((el) => el.getBoundingClientRect().width / window.innerWidth)
      check('enlarge shows the phone screenshot across the screen', zoomed > 0.9, Math.round(zoomed * 100) + '%')
      await page.screenshot({ path: path.join(OUT, 'phone-dark-zoom.png') })
      await page.getByRole('button', { name: 'Close screenshot' }).click()
      check('closing the enlarged screenshot keeps the guide', await g.isVisible())
    }
    if (form === 'wide' && theme === 'dark') {
      // Enlarge, then a step's button: the guide gives way to Connections on Google.
      await g.getByRole('button', { name: 'Back', exact: true }).click()
      await page.locator('nav[aria-label=Topics] button', { hasText: 'Google' }).click()
      await g.getByRole('button', { name: 'Enlarge screenshot' }).click()
      await sleep(300)
      await page.screenshot({ path: path.join(OUT, 'wide-dark-zoom.png') })
      await page.keyboard.press('Escape')
      check('zoom closes on Escape, guide stays', await g.isVisible())
      await g.getByRole('button', { name: 'Connect Google', exact: true }).click()
      const conn = page.locator('[role=dialog][aria-label="Connections"]')
      await conn.waitFor()
      check('"Connect Google" opens Connections on the Google tab',
        (await conn.locator('#connections-tab-google').getAttribute('aria-selected')) === 'true' && !(await g.isVisible()))
      await page.keyboard.press('Escape')

      // Reopen from Info; tick "Do not show again"; it stays away after a reload.
      await page.locator('aside button[aria-label=Info]').click()
      await page.locator('[role=dialog][aria-label=Info]').getByRole('button', { name: /Getting started/ }).click()
      check('Info → Getting started reopens it', await g.isVisible())
      await g.getByLabel('Do not show again').check()
      await sleep(800)  // the settings write is debounced
      const saved = await api('GET', '/fd/settings')
      check('"Do not show again" is saved to the user\'s settings', JSON.parse(saved['fd:onboarding'] || '{}').completed?.kiosk === true)
      await page.getByRole('button', { name: 'Close' }).first().click()
      await page.reload()
      await page.getByText('Research Assistant').first().waitFor()
      await sleep(2000)
      check('stays closed after a reload once opted out', !(await g.isVisible()))
      await page.locator('aside button', { hasText: 'Settings' }).click()
      await page.locator('[role=dialog][aria-label="Settings"]').getByRole('button', { name: /Getting started/ }).click()
      check('Settings → Getting started reopens it, box still ticked', await g.isVisible() && await g.getByLabel('Do not show again').isChecked())
    }
    await context.close()
  }
  fs.writeFileSync(path.join(RUN, 'report.json'), JSON.stringify(results, null, 2))
  if (results.some((r) => !r.ok)) process.exitCode = 1
}

;(async () => {
  token = (await api('POST', '/fd/auth/login', { email: EMAIL, password: PASSWORD })).access_token
  await api('PUT', '/fd/profile', {
    about_me: 'Pricing lead at Velora, a Berlin e-bike subscription start-up. I prepare the monthly pricing review.',
    company: 'Velora rents e-bikes by the month to city commuters in Berlin, Hamburg and Munich.',
    instructions: 'Answer in English. Keep it short: a table first, then a one-paragraph recommendation.',
  })
  const browser = await chromium.launch()
  try {
    if (MODE === 'guide') await guide(browser)
    else await assets(browser)
  } catch (e) {
    if (current) await current.screenshot({ path: path.join(RUN, 'failure.png') }).catch(() => {})
    throw e
  } finally {
    await browser.close()
  }
})().catch((e) => { console.error(e); process.exit(1) })
