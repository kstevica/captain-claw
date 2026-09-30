import { create } from 'zustand'
import { useNotificationStore } from './notificationStore'

export interface User {
  id: string
  email: string
  display_name: string
  role: string
}

export interface AuthStore {
  user: User | null
  token: string
  isAuthenticated: boolean
  authEnabled: boolean | null  // null = not yet checked
  dockerSpawnEnabled: boolean
  internalFdUrl: string  // Internal URL for agent-to-FD calls (e.g. http://localhost:25080)
  // Kiosk lock (server `--simple-chat` / FD_SIMPLE_CHAT): force the locked,
  // chat-only Simple layout and hide every escape hatch.
  simpleChatOnly: boolean
  // A sign-out is in flight (logoutUser) — the sign-out buttons show it.
  signingOut: boolean

  setAuth: (user: User, token: string) => void
  clearAuth: () => void
  setAuthEnabled: (enabled: boolean) => void
  setDockerSpawnEnabled: (enabled: boolean) => void
  setInternalFdUrl: (url: string) => void
  setSimpleChatOnly: (v: boolean) => void
  setToken: (token: string) => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  token: '',
  isAuthenticated: false,
  authEnabled: null,
  dockerSpawnEnabled: true,
  internalFdUrl: '',
  simpleChatOnly: false,
  signingOut: false,

  setAuth: (user, token) => set({ user, token, isAuthenticated: true }),
  clearAuth: () => set({ user: null, token: '', isAuthenticated: false }),
  setAuthEnabled: (enabled) => set({ authEnabled: enabled }),
  setDockerSpawnEnabled: (enabled) => set({ dockerSpawnEnabled: enabled }),
  setInternalFdUrl: (url) => set({ internalFdUrl: url }),
  setSimpleChatOnly: (v) => set({ simpleChatOnly: v }),
  setToken: (token) => set({ token }),
}))

/**
 * Whether the kiosk lock (server `--simple-chat` / FD_SIMPLE_CHAT) is in effect
 * for the CURRENT viewer. Admins are exempt — an admin keeps full access even
 * when the deck runs in simple-chat mode, so they can spawn agents, open config
 * and administer users. Everyone else (the shared kiosk account) stays locked to
 * the chat-only Simple layout, keeping only sign-out, the archetype picker and
 * their own connections (Google, MCP).
 *
 * Use this instead of reading `simpleChatOnly` directly wherever the lock is
 * enforced, so the admin exemption stays consistent across the app. Reactive:
 * it re-derives when the user logs in (role becomes known) or the flag changes.
 */
export const selectKioskLocked = (s: AuthStore): boolean =>
  s.simpleChatOnly && s.user?.role !== 'admin'

// ── Auth API calls ──

const FD = '/fd'

// Throws when Flight Deck can't be reached or doesn't give a real answer
// (network error, a proxy's 5xx mid-restart, an HTML error page). That is NOT
// "auth is off": failing open to auth-disabled would render the full, unlocked
// UI — kiosk lock dropped — on a deck that enforces auth. Only an actual
// { auth_enabled: false } means auth is off; see awaitAuthStatus for the retry.
export async function checkAuthStatus(): Promise<boolean> {
  const res = await fetch(`${FD}/auth/status`)
  if (!res.ok) throw new Error(`auth status: HTTP ${res.status}`)
  const data = await res.json()
  if (!data || typeof data.auth_enabled !== 'boolean') throw new Error('auth status: unexpected response')
  const enabled = data.auth_enabled
  useAuthStore.getState().setAuthEnabled(enabled)
  useAuthStore.getState().setDockerSpawnEnabled(data.docker_spawn_enabled !== false)
  if (data.internal_fd_url) useAuthStore.getState().setInternalFdUrl(data.internal_fd_url)
  useAuthStore.getState().setSimpleChatOnly(data.simple_chat_only === true)
  return enabled
}

/**
 * checkAuthStatus() until Flight Deck answers, backing off 1s → 10s.
 * `onUnreachable` fires after each failed attempt (App shows "Can't reach
 * Flight Deck — retrying…"). Resolves with the deck's auth_enabled, or null if
 * `isCancelled()` turns true first. The store's authEnabled stays null (not
 * yet checked) the whole time — never a guessed `false`.
 */
export async function awaitAuthStatus(
  onUnreachable: () => void,
  isCancelled: () => boolean = () => false,
  baseDelayMs = 1_000,
): Promise<boolean | null> {
  for (let attempt = 0; ; attempt++) {
    if (isCancelled()) return null
    try {
      return await checkAuthStatus()
    } catch {
      if (isCancelled()) return null
      onUnreachable()
      await new Promise((resolve) => setTimeout(resolve, Math.min(baseDelayMs * 2 ** attempt, 10 * baseDelayMs)))
    }
  }
}

export async function loginUser(email: string, password: string): Promise<User> {
  const res = await fetch(`${FD}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Login failed' }))
    throw new Error(body.detail || 'Login failed')
  }
  const data = await res.json()
  useAuthStore.getState().setAuth(data.user, data.access_token)
  return data.user
}

export async function registerUser(email: string, password: string, displayName: string): Promise<User> {
  const res = await fetch(`${FD}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password, display_name: displayName }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Registration failed' }))
    throw new Error(body.detail || 'Registration failed')
  }
  const data = await res.json()
  useAuthStore.getState().setAuth(data.user, data.access_token)
  return data.user
}

// ── Session hand-off (sign-out / expiry) ──
//
// Signing out has to hand the whole TAB over, not just drop the token: agent
// chat sockets (authorized once, at handshake, as the outgoing user), queue
// autopilot timers and ~every zustand store would otherwise carry straight
// into the next person's session on a shared kiosk — their transcripts, and
// live sockets that drive their agents (and so their Google account). A page
// reload tears all of that down in one step, so a confirmed sign-out reloads
// (this tab and every other tab of the deck — _broadcastSignedOut), and so
// does a session that ends on its own (App's auth effect →
// reloadAfterSessionEnd).

// Every /auth/refresh rotates the refresh cookie, so concurrent refreshes race
// each other — and a sign-out: a logout POST carrying the pre-rotation cookie
// deletes nothing and the rotated one keeps the session alive. Single-flight.
let _refreshInFlight: Promise<boolean> | null = null
let _refreshAbort: AbortController | null = null
// Bumped on a confirmed sign-out: a refresh that started before it must not
// setAuth() the outgoing user back in.
let _authGeneration = 0
// Pending/finished sign-out; resolves true once the server confirmed it.
let _signOut: Promise<boolean> | null = null

// Debounced writers (settings sync, chat persistence) register a flush here so
// a sign-out can drain them while the outgoing user's token is still valid —
// after clearAuth() they'd 401, and the reload would drop them outright.
const _signOutFlushes: Array<() => Promise<void>> = []

export function registerSignOutFlush(fn: () => Promise<void>): void {
  _signOutFlushes.push(fn)
}

// Best effort, and bounded: a hung write must not keep a tab from being handed
// over (refreshes wait on the sign-out meanwhile).
async function _drainSignOutFlushes(maxMs: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined
  await Promise.race([
    Promise.allSettled(_signOutFlushes.map((flush) => flush())),
    new Promise((resolve) => { timer = setTimeout(resolve, maxMs) }),
  ])
  clearTimeout(timer)
}

// Synchronous teardown of things a reload doesn't reach — windows this tab
// opened (the Google OAuth popup) outlive it. Run once the session is over:
// a confirmed sign-out here or in another tab, or a session that ended on
// its own.
const _signOutTeardowns: Array<() => void> = []

export function registerSignOutTeardown(fn: () => void): void {
  _signOutTeardowns.push(fn)
}

function _runSignOutTeardowns(): void {
  for (const teardown of _signOutTeardowns) {
    try { teardown() } catch { /* one failure must not block the rest */ }
  }
}

export function refreshAccessToken(): Promise<boolean> {
  // A refresh now would rotate the cookie out from under the logout POST, so
  // wait for the sign-out's outcome: done → the session is over; failed →
  // still signed in, refresh as usual.
  if (_signOut) return _signOut.then((signedOut) => (signedOut ? false : refreshAccessToken()))
  if (_refreshInFlight) return _refreshInFlight
  const generation = _authGeneration
  const abort = new AbortController()
  _refreshAbort = abort
  _refreshInFlight = (async () => {
    try {
      const res = await fetch(`${FD}/auth/refresh`, {
        method: 'POST',
        credentials: 'include',
        signal: abort.signal,
      })
      if (!res.ok) return false
      const data = await res.json()
      if (generation !== _authGeneration) return false  // signed out meanwhile
      useAuthStore.getState().setAuth(data.user, data.access_token)
      return true
    } catch {
      return false
    } finally {
      _refreshInFlight = null
      _refreshAbort = null
    }
  })()
  return _refreshInFlight
}

// Browser-local (not per-user) chat state: the queue / plan slices chatStore
// mirrors to localStorage by container id. On a kiosk deck (simpleChatOnly)
// everyone shares one account, so they'd reload for the next person who opens
// the same agent — and a queue with auto-mode on would auto-dispatch the
// previous person's items as theirs — so a kiosk sign-out drops them.
// Anywhere else they're the user's own work in progress (the queue is meant
// to survive a reload) on agents only they own, and are kept.
// `fd.queue.plan.*` shares the prefix but is QueuePlannerModal's own
// "continue from row N" record, not a queue slice — kept.
const _BROWSER_LOCAL_PREFIXES = ['fd.queue.', 'fd.plan.']
const _BROWSER_LOCAL_KEEP_PREFIXES = ['fd.queue.plan.']

function _purgeBrowserLocalState(): void {
  try {
    const doomed: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (!key || _BROWSER_LOCAL_KEEP_PREFIXES.some((p) => key.startsWith(p))) continue
      if (_BROWSER_LOCAL_PREFIXES.some((p) => key.startsWith(p))) doomed.push(key)
    }
    for (const key of doomed) localStorage.removeItem(key)
  } catch { /* storage blocked — nothing persisted to purge */ }
}

// The session is over (confirmed here or in another tab): hand this tab over.
function _handOverTab(): void {
  _authGeneration++
  _runSignOutTeardowns()
  if (useAuthStore.getState().simpleChatOnly) {
    _purgeBrowserLocalState()
    // Chat sockets can still deliver a message (and re-save a queue slice)
    // between here and the unload — purge once more on the way out.
    window.addEventListener('pagehide', _purgeBrowserLocalState, { once: true })
  }
  useAuthStore.getState().clearAuth()
  window.location.reload()
}

// Other tabs of this deck hold the outgoing user's access JWT (valid for up
// to 15 min) and agent sockets authorized as them, so a sign-out tells them
// to hand over too. BroadcastChannel where there is one, else a localStorage
// 'storage' event — both are per origin, so other decks on the host (other
// ports) never hear it. One channel object sends and listens: a channel
// doesn't hear its own messages, nor a tab its own storage writes.
const _SIGNED_OUT_CHANNEL = 'fd-auth'
const _SIGNED_OUT_MESSAGE = 'signed-out'
const _SIGNED_OUT_KEY = 'fd.auth.signedOutAt'

const _authChannel: BroadcastChannel | null = (() => {
  try {
    return typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(_SIGNED_OUT_CHANNEL)
  } catch {
    return null
  }
})()

function _broadcastSignedOut(): void {
  try {
    if (_authChannel) _authChannel.postMessage(_SIGNED_OUT_MESSAGE)
    else localStorage.setItem(_SIGNED_OUT_KEY, String(Date.now()))
  } catch { /* can't reach the other tabs — their next refresh fails instead */ }
}

// Short: the session is already over — only this tab's last debounced writes
// (chat messages still streaming in, a settings change) are worth the wait.
const _ELSEWHERE_FLUSH_MS = 1_500

function _onSignedOutElsewhere(): void {
  const s = useAuthStore.getState()
  if (!s.authEnabled || !s.isAuthenticated) return  // nothing of the session here
  if (_signOut) return  // this tab is already signing out
  // Settled sign-out: refreshes here now fail fast (the cookie is gone) and
  // App's session-end reload stands down — _handOverTab reloads.
  _signOut = Promise.resolve(true)
  useAuthStore.setState({ signingOut: true })
  // This tab's access JWT is still valid (stateless, up to 15 min), so drain
  // its pending writes first, as the signing-out tab did — the reload would
  // drop them.
  void _drainSignOutFlushes(_ELSEWHERE_FLUSH_MS).then(_handOverTab)
}

if (_authChannel) {
  _authChannel.onmessage = (event) => {
    if (event.data === _SIGNED_OUT_MESSAGE) _onSignedOutElsewhere()
  }
} else if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key === _SIGNED_OUT_KEY && event.newValue) _onSignedOutElsewhere()
  })
}

export function logoutUser(): Promise<boolean> {
  if (_signOut) return _signOut  // double-click
  useAuthStore.setState({ signingOut: true })
  _signOut = (async () => {
    await _drainSignOutFlushes(3_000)
    // Let a refresh that's already in flight land first, so the logout POST
    // carries the current refresh cookie (see _refreshInFlight). Bounded too:
    // one that's still stalled after 5s is aborted, so a late answer can't
    // rotate the cookie under the POST and resurrect the session.
    if (_refreshInFlight) {
      let timer: ReturnType<typeof setTimeout> | undefined
      const landed = await Promise.race([
        _refreshInFlight.then(() => true),
        new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(false), 5_000) }),
      ])
      clearTimeout(timer)
      if (!landed) _refreshAbort?.abort()
    }

    let ok = false
    const abort = new AbortController()
    const timer = setTimeout(() => abort.abort(), 10_000)
    try {
      const res = await fetch(`${FD}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
        signal: abort.signal,
      })
      ok = res.ok
    } catch {
      /* network error / timeout — handled below */
    } finally {
      clearTimeout(timer)
    }

    if (!ok && !useAuthStore.getState().isAuthenticated) {
      // This tab's session already ended while the sign-out was pending: a
      // refresh that failed (or that the wait above aborted) → clearAuth(),
      // and App's session-end reload stood down for this sign-out. "Still
      // signed in" would be false here, and the login page would sit on top
      // of the old session's agent sockets and stores — hand the tab over.
      // No broadcast: the server may still hold the session, and other tabs
      // keep theirs. Refreshes until the reload fail fast.
      _signOut = Promise.resolve(true)
      _handOverTab()
      return false
    }

    if (!ok) {
      // The refresh cookie may still be valid server-side: "signing out" the
      // UI now would only look signed out — the next reload or background
      // refresh would log this user straight back in. Stay signed in and say so.
      _signOut = null
      useAuthStore.setState({ signingOut: false })
      useNotificationStore.getState().add(
        'error', 'Sign out failed',
        'Could not reach Flight Deck to end your session — you are still signed in. Try again.',
      )
      return false
    }

    _broadcastSignedOut()
    _handOverTab()
    return true
  })()
  return _signOut
}

// A session that ended on its own (a 401 whose refresh failed → clearAuth())
// lands on the login page with the same residue as a sign-out, so App reloads
// on every signed-in → signed-out transition. Startup never goes true → false,
// and clearAuth() only follows a FAILED refresh (the fdFetch helpers), so the
// reload lands on the login page — unless the refresh flaps (two tabs racing
// to rotate the same cookie, a proxy 5xx mid-restart) and the reloaded tab
// signs straight back in. Hence at most one such reload per 30s window; past
// that, fall back to the plain login page instead of risking a loop.
const _SESSION_END_RELOAD_KEY = 'fd.auth.sessionEndReloadAt'
const _SESSION_END_RELOAD_WINDOW_MS = 30_000

export function reloadAfterSessionEnd(): void {
  if (!useAuthStore.getState().authEnabled) return
  if (_signOut) return  // logoutUser() reloads (or stays signed in) itself
  // Whether or not the loop guard lets this reload through, the session is
  // over — don't leave e.g. its Google OAuth popup for the next person.
  _runSignOutTeardowns()
  try {
    const last = Number(sessionStorage.getItem(_SESSION_END_RELOAD_KEY)) || 0
    if (Date.now() - last < _SESSION_END_RELOAD_WINDOW_MS) return
    sessionStorage.setItem(_SESSION_END_RELOAD_KEY, String(Date.now()))
  } catch {
    return  // can't record the loop guard — don't risk a reload loop
  }
  window.location.reload()
}

export async function updateProfile(data: { display_name?: string; password?: string; current_password?: string }): Promise<User> {
  const { token } = useAuthStore.getState()
  const res = await fetch(`${FD}/auth/me`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Update failed' }))
    throw new Error(body.detail || 'Update failed')
  }
  const user = await res.json()
  useAuthStore.getState().setAuth(user, token)
  return user
}
