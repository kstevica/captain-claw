// The Captain Claw desktop app (desktop/, Electron): its preload script
// exposes `window.captainClawDesktop` to the page. Absent in a browser.
//
// To try the desktop app's behaviour in a browser without building the app,
// open any deck address with `?app=desktop` — the tab then acts as the desktop
// app until it closes (`?app=web` turns it back off). Only the layout follows
// it: sign-in popups still go by the real shell.
//
// A browser tab works like the apps too — the Admin console or the users'
// kiosk, nothing else (isKioskOnly) — unless it asks for the classic Flight
// Deck with `?app=classic`: the sidebar, top bar and pages, as before. That
// lasts until the tab closes, or another `?app=` (web, desktop) turns it off.

interface DesktopBridge {
  isDesktop?: boolean
  platform?: string
  version?: string
  // Hands a Google / Microsoft sign-in to the system browser (main.js checks
  // it is this deck's /fd/<provider>/login). Absent in older builds.
  openSignIn?: (url: string) => Promise<boolean>
}

function bridge(): DesktopBridge | undefined {
  return (window as Window & { captainClawDesktop?: DesktopBridge }).captainClawDesktop
}

const SIMULATE_KEY = 'fd:simulate-desktop'

function simulated(): boolean {
  try {
    const ask = new URLSearchParams(window.location.search).get('app')
    if (ask === 'desktop') sessionStorage.setItem(SIMULATE_KEY, '1')
    else if (ask === 'web' || ask === 'classic') sessionStorage.removeItem(SIMULATE_KEY)
    return sessionStorage.getItem(SIMULATE_KEY) === '1'
  } catch { return false }
}

const CLASSIC_KEY = 'fd:classic-layout'

function classicAsked(): boolean {
  try {
    const ask = new URLSearchParams(window.location.search).get('app')
    if (ask === 'classic') sessionStorage.setItem(CLASSIC_KEY, '1')
    else if (ask) sessionStorage.removeItem(CLASSIC_KEY)
    return sessionStorage.getItem(CLASSIC_KEY) === '1'
  } catch {
    // Storage blocked: this page load only.
    try { return new URLSearchParams(window.location.search).get('app') === 'classic' } catch { return false }
  }
}

// Read once, at load: the app rewrites the address (e.g. to /console/…) later.
const SIMULATED = simulated()
const CLASSIC = classicAsked()

/** Running inside the desktop app (or a tab simulating it) rather than a browser tab. */
export function isDesktopApp(): boolean {
  return SIMULATED || !!bridge()?.isDesktop
}

// The Captain Android app (kept out of this repo until it is published): a
// WebView shell around a deck's kiosk that appends ` CaptainClawHUD/<version>`
// to its user agent. The token is older than the kiosk — the app first showed
// the HUD (hud/device.ts reads it too).
const ANDROID_APP = /\bCaptainClawHUD\/\S+/.test(navigator.userAgent || '')

/** The Captain Claw Android app. */
export function isAndroidApp(): boolean {
  return ANDROID_APP
}

/** Running inside one of the Captain Claw apps — the desktop app (or a tab
 * simulating it) or the Android app — rather than a browser tab. Either app is
 * the Admin console or the users' kiosk, nothing else: anyone but an admin
 * gets the kiosk, and an admin starts in it when they have agents. */
export function isAppShell(): boolean {
  return isDesktopApp() || ANDROID_APP
}

/** The Admin console or the users' kiosk, nothing else: the apps, and every
 * browser tab that didn't ask for the classic Flight Deck (`?app=classic`).
 * Anyone but an admin gets the kiosk; an admin switches between the two. */
export function isKioskOnly(): boolean {
  return isAppShell() || !CLASSIC
}

// The Android app's native side (MainActivity), injected for the deck's origin
// only. Absent everywhere else.
function androidBridge(): { postMessage(message: string): void } | null {
  if (!ANDROID_APP) return null
  const b = (window as Window & { CaptainClawHUD?: { postMessage?: unknown } }).CaptainClawHUD
  return b && typeof b.postMessage === 'function' ? (b as { postMessage(message: string): void }) : null
}

function tellAndroidApp(message: Record<string, unknown>): void {
  try { androidBridge()?.postMessage(JSON.stringify(message)) } catch { /* bridge gone */ }
}

/** The Android app has a start screen: its saved decks and accounts. */
export function hasAndroidAppSettings(): boolean {
  return androidBridge() !== null
}

/** Back to the Android app's start screen, to switch deck or account. */
export function openAndroidAppSettings(): void {
  tellAndroidApp({ t: 'settings' })
}

/** Who signed in: the Android app lists the account (a sign-in on this page included). */
export function reportAndroidSession(user: { email: string; display_name?: string; role?: string } | null): void {
  if (user?.email) tellAndroidApp({ t: 'session', email: user.email, name: user.display_name || '', role: user.role || '' })
}

/** Signed out, or the session ended: the Android app forgets it and shows its start screen. */
export function reportAndroidSignedOut(): void {
  tellAndroidApp({ t: 'signed-out' })
}

/** The real desktop app (its preload bridge), never a tab simulating it: only
 * the shell sends a sign-in to the system browser — a browser tab keeps its
 * popup, which holds the ticket's cookie. */
export function isDesktopShell(): boolean {
  return !!bridge()?.isDesktop
}

/** Open a Google / Microsoft sign-in (`/fd/<provider>/login?ticket=…`) in the
 * system browser — Google refuses sign-in inside the app's own window. A shell
 * without the bridge call gets window.open, which it sends there too. */
export async function openInSystemBrowser(url: string): Promise<void> {
  const open = bridge()?.openSignIn
  if (open) {
    await open(url)
    return
  }
  window.open(url, '_blank', 'noopener')
}

const BROWSER_POLL_MS = 2000
const BROWSER_WATCH_MAX_MS = 10 * 60_000  // Flight Deck forgets the pending sign-in by then

/** Follow a sign-in that finishes in the system browser. There is no popup to
 * watch and no message from the callback page, so `done` is asked every couple
 * of seconds until it says yes, the sign-in has expired, or the returned
 * function cancels it. `onEnd` runs once, however it ends. */
export function watchBrowserSignIn(
  done: () => Promise<boolean>,
  onEnd: (connected: boolean) => void,
): () => void {
  const started = Date.now()
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | null = null
  const stop = (connected: boolean) => {
    if (stopped) return
    stopped = true
    if (timer) clearTimeout(timer)
    onEnd(connected)
  }
  const tick = async () => {
    timer = null
    let ok = false
    try { ok = await done() } catch { /* asked again next time */ }
    if (stopped) return
    if (ok) return stop(true)
    if (Date.now() - started > BROWSER_WATCH_MAX_MS) return stop(false)
    timer = setTimeout(tick, BROWSER_POLL_MS)
  }
  timer = setTimeout(tick, BROWSER_POLL_MS)
  return () => stop(false)
}
