import { create } from 'zustand'
import { useAuthStore, refreshAccessToken, registerSignOutTeardown } from './authStore'

export interface GrantedScope {
  scope: string
  label: string
}

export interface GoogleUserInfo {
  email?: string
  name?: string
  picture?: string
  [key: string]: unknown
}

export type GoogleAuthMode = 'custom'

export interface GoogleAuthStatus {
  configured: boolean
  mode: GoogleAuthMode
  supports_vertex: boolean
  connected: boolean
  user: GoogleUserInfo | null
  granted_scopes: GrantedScope[]
  redirect_uri: string
}

export interface ScopeCatalogEntry {
  scope: string
  label: string
  description: string
  sensitivity: 'none' | 'sensitive' | 'restricted'
  group: string
}

export interface GoogleAuthConfig {
  mode: GoogleAuthMode
  client_id: string
  client_id_set: boolean
  client_secret_set: boolean
  project_id: string
  location: string
  scopes: string[]
  default_scopes: string[]
  scope_catalog: ScopeCatalogEntry[]
  redirect_uri: string
}

// The signed-in user's Gmail sending policy (GET/PUT /fd/google/gmail-send).
// Off until the user opts in: their agents can only draft.
export interface GmailSendPolicy {
  enabled: boolean
  // Exact addresses, or whole domains written '@b.c'. Empty = anyone.
  allowed_recipients: string[]
  daily_limit: number
  // The deck's FD_GMAIL_SEND=off: no agent sends, whatever the policy says.
  deck_disabled: boolean
  sent_last_24h: number
}

export type GmailSendPatch = Partial<Pick<GmailSendPolicy, 'enabled' | 'allowed_recipients' | 'daily_limit'>>

// While the deck switch is off, the opt-in box refuses only turning sending
// ON (it would do nothing). A user who opted in can still opt out, or their
// agents start sending again the moment the admin turns the deck back on.
// Keyed on the SAVED value, so unticking and re-ticking to undo still works.
export function gmailSendOptInLocked(policy: GmailSendPolicy | null): boolean {
  return !!policy?.deck_disabled && !policy.enabled
}

// One email an agent sent from the user's account (GET /fd/google/gmail-sends).
export interface GmailSendRecord {
  id: string | number
  // 'sent', or 'unknown': Gmail never confirmed the send — it may have gone out.
  status?: string
  agent: string
  to: string
  cc: string
  bcc: string
  subject: string
  gmail_message_id: string
  thread_id: string
  draft_id: string
  created_at: string  // ISO 8601, UTC
}

interface GoogleAuthStore {
  status: GoogleAuthStatus | null
  config: GoogleAuthConfig | null
  loading: boolean
  error: string | null
  lastPopupMessage: string | null
  gmailSend: GmailSendPolicy | null
  gmailSends: GmailSendRecord[]
  gmailSendError: string | null

  refresh: () => Promise<void>
  syncStatus: () => Promise<void>
  saveConfig: (patch: Partial<{
    client_id: string
    client_secret: string
    project_id: string
    location: string
    scopes: string[]
  }>) => Promise<boolean>
  clearCredentials: () => Promise<boolean>
  connect: () => Promise<void>
  disconnect: () => Promise<void>
  startMessageListener: () => () => void
  fetchGmailSend: () => Promise<void>
  saveGmailSend: (patch: GmailSendPatch) => Promise<boolean>
  fetchGmailSends: (limit?: number) => Promise<void>
}

const emptyStatus: GoogleAuthStatus = {
  configured: false,
  mode: 'custom',
  supports_vertex: false,
  connected: false,
  user: null,
  granted_scopes: [],
  redirect_uri: '',
}

function authHeaders(): Record<string, string> {
  const { token } = useAuthStore.getState()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

// A non-2xx answer. `detail` is FastAPI's error detail when it's a string.
class HttpError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, statusText: string, text: string) {
    super(`${status} ${statusText}: ${text}`)
    this.status = status
    let detail = ''
    try {
      const parsed = JSON.parse(text)
      if (typeof parsed?.detail === 'string') detail = parsed.detail
    } catch { /* not JSON */ }
    this.detail = detail
  }
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const doFetch = () => fetch(url, {
    credentials: 'include',
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers || {}),
    },
  })
  let resp = await doFetch()
  // The access JWT lives 15 min; a tab left open on Connections outlives it.
  if (resp.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    resp = await doFetch()
  }
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new HttpError(resp.status, resp.statusText, text)
  }
  return resp.json()
}

// /fd/google/login is a 302 to Google, so it runs in a popup — which can't
// carry an Authorization header. A single-use, short-lived ticket minted by an
// authenticated POST (fetchJson: Bearer, and a refresh + retry on a 401) names
// the connecting user instead, so the JWT never goes in a URL (history, logs,
// Referer). /login binds the flow to the ticket's user — and only in the
// browser holding the cookie this POST sets (credentials: 'include').
async function _mintConnectTicket(): Promise<string> {
  const data = await fetchJson('/fd/google/connect-ticket', { method: 'POST', body: '{}' })
  const ticket = typeof data?.ticket === 'string' ? data.ticket : ''
  if (!ticket) throw new Error('Flight Deck returned no sign-in ticket.')
  return ticket
}

// The OAuth popup this tab opened. Kept so a sign-out can close it: left open,
// the next person at the screen could finish the consent — and the pending
// login is bound to whoever clicked Connect, so THEIR Google account would be
// linked to the outgoing user. Also polled so the card updates once it closes.
let _popup: Window | null = null
let _popupWatch: ReturnType<typeof setInterval> | null = null
const _POPUP_POLL_MS = 500
const _POPUP_WATCH_MAX_MS = 10 * 60_000  // the pending login expires by then

function _forgetPopup(close: boolean): void {
  if (_popupWatch) clearInterval(_popupWatch)
  _popupWatch = null
  if (close && _popup && !_popup.closed) {
    try { _popup.close() } catch { /* already gone */ }
  }
  _popup = null
}

function _watchPopup(popup: Window, onClosed: () => void): void {
  _forgetPopup(false)  // a re-used named window is the same one — don't close it
  _popup = popup
  const started = Date.now()
  _popupWatch = setInterval(() => {
    if (popup.closed) {
      _forgetPopup(false)
      onClosed()
    } else if (Date.now() - started > _POPUP_WATCH_MAX_MS) {
      // Stop polling, but keep the handle so a sign-out can still close it.
      if (_popupWatch) clearInterval(_popupWatch)
      _popupWatch = null
    }
  }, _POPUP_POLL_MS)
}

registerSignOutTeardown(() => _forgetPopup(true))

// The Electron shell (desktop/preload.js): it hands every http(s)
// window.open to the system browser and denies the in-app window.
function _isDesktopShell(): boolean {
  return !!(window as Window & { captainClawDesktop?: { isDesktop?: boolean } }).captainClawDesktop?.isDesktop
}

const _POPUP_BLOCKED = 'The browser blocked the Google sign-in popup — allow popups for this site and try again.'
const _SESSION_EXPIRED = 'Your Flight Deck session has expired — sign in again, then connect Google.'
const _AUTH_OFF = "Google isn't available on this deck: Flight Deck sign-in is turned off."
const _DESKTOP_SHELL = "The desktop app can't run the Google sign-in — open Flight Deck in a web browser to connect Google."

function _connectError(exc: unknown): string {
  if (exc instanceof HttpError && exc.status === 401) return _SESSION_EXPIRED
  const why = exc instanceof HttpError && exc.detail
    ? exc.detail
    : exc instanceof Error ? exc.message : String(exc)
  return `Couldn't start Google sign-in: ${why}`
}

// The backend's own words when it gave some (e.g. which recipient entry it
// couldn't read), else the status line.
function _gmailSendError(exc: unknown): string {
  if (exc instanceof HttpError) {
    if (exc.status === 401) return 'Your Flight Deck session has expired — sign in again.'
    // None of these routes answers 404 or 405 itself, so either means a deck
    // that predates them: FastAPI's bare 404, the SPA catch-all's "No route
    // for /fd/…" (any deck serving this UI), or 405 to the PUT (that
    // catch-all is GET-only).
    if (exc.status === 404 || exc.status === 405) {
      return "This Flight Deck doesn't offer email sending yet — it needs an update and a restart."
    }
    if (exc.detail) return exc.detail
  }
  return exc instanceof Error ? exc.message : String(exc)
}

export const useGoogleAuthStore = create<GoogleAuthStore>((set, get) => ({
  status: null,
  config: null,
  loading: false,
  error: null,
  lastPopupMessage: null,
  gmailSend: null,
  gmailSends: [],
  gmailSendError: null,

  refresh: async () => {
    // No Google via FD with sign-in off (see connect()) — nothing to ask.
    if (useAuthStore.getState().authEnabled === false) {
      set({ status: emptyStatus, config: null, loading: false, error: null })
      return
    }
    set({ loading: true, error: null })
    try {
      const [status, config] = await Promise.all([
        fetchJson('/fd/google/status'),
        fetchJson('/fd/google/config'),
      ])
      set({
        status: status as GoogleAuthStatus,
        config: config as GoogleAuthConfig,
        loading: false,
      })
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
        status: get().status || emptyStatus,
      })
    }
  },

  // Quiet status re-check for the background triggers (popup closed, window
  // focus): no spinner, and a failure keeps whatever the card shows — it must
  // not wipe e.g. a "session expired" error that connect() just set. When it
  // sees the connection appear, it says so in place of the message the
  // callback couldn't deliver (and of the "finish in your browser" hint).
  syncStatus: async () => {
    try {
      const status = (await fetchJson('/fd/google/status')) as GoogleAuthStatus
      const justConnected = !!status?.connected && !get().status?.connected
      set({
        status,
        ...(justConnected
          ? { lastPopupMessage: `Connected${status.user?.email ? ` as ${status.user.email}` : ''}` }
          : {}),
      })
    } catch { /* the next explicit refresh() reports it */ }
  },

  saveConfig: async (patch) => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/config', {
        method: 'POST',
        body: JSON.stringify(patch),
      })
      await get().refresh()
      return true
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
      return false
    }
  },

  clearCredentials: async () => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/config', {
        method: 'POST',
        body: JSON.stringify({ clear: true }),
      })
      await get().refresh()
      return true
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
      return false
    }
  },

  connect: async () => {
    // The login endpoint is a 302 to Google, so it opens as a popup, at
    // /login?ticket=<single-use ticket> (see _mintConnectTicket).
    //
    // The result comes back by postMessage from the callback page when it
    // can; the popup-closed watch and the focus re-check (startMessageListener)
    // cover the flows where it can't.
    set({ error: null })
    // A Google connection belongs to a Flight Deck user, so it needs sign-in:
    // with auth off FD offers no Google (the card says so instead of Connect).
    if (!useAuthStore.getState().authEnabled) {
      set({ error: _AUTH_OFF })
      return
    }
    // The Electron shell denies in-app windows and hands URLs to the system
    // browser — which doesn't hold the ticket's cookie, so /login refuses it
    // there. (The desktop app runs FD with sign-in off anyway: no Google.)
    if (_isDesktopShell()) {
      set({ error: _DESKTOP_SHELL })
      return
    }
    const w = 520
    const h = 640
    const left = window.screenX + (window.outerWidth - w) / 2
    const top = window.screenY + (window.outerHeight - h) / 2
    const name = 'captain-claw-google-oauth'
    const features = `width=${w},height=${h},left=${left},top=${top}`
    // Open the popup synchronously — still inside the click, so blockers allow
    // it — and point it at /login once the ticket is in hand.
    const popup = window.open('about:blank', name, features)
    if (!popup) {
      set({ error: _POPUP_BLOCKED })
      return
    }

    let ticket: string
    try {
      ticket = await _mintConnectTicket()
    } catch (exc) {
      popup.close()
      set({ error: _connectError(exc) })
      return
    }
    // Closed meanwhile — by the user, or by a sign-out's teardown: the ticket
    // is for a flow nobody is waiting on any more.
    if (popup.closed || !useAuthStore.getState().isAuthenticated) {
      if (!popup.closed) popup.close()
      return
    }
    popup.location.href = `${window.location.origin}/fd/google/login?ticket=${encodeURIComponent(ticket)}`
    _watchPopup(popup, () => { get().syncStatus() })
  },

  disconnect: async () => {
    set({ loading: true, error: null })
    try {
      await fetchJson('/fd/google/logout', { method: 'POST' })
      await get().refresh()
    } catch (exc) {
      set({
        loading: false,
        error: exc instanceof Error ? exc.message : String(exc),
      })
    }
  },

  startMessageListener: () => {
    const handler = (event: MessageEvent) => {
      // Only our own /fd/google/callback page (same origin as the SPA) may
      // report a result — any other window could post a fake "Connected as …".
      if (event.origin !== window.location.origin) return
      const data = event.data
      if (!data || typeof data !== 'object') return
      if (data.type !== 'captain-claw-google-oauth') return
      set({
        lastPopupMessage:
          data.status === 'success'
            ? `Connected${data.email ? ` as ${data.email}` : ''}`
            : `Error: ${data.title || 'OAuth failed'}${data.detail ? ` — ${data.detail}` : ''}`,
      })
      setTimeout(() => { get().refresh() }, 300)
    }
    // No message comes when the callback can't reach this window: the Electron
    // flow finishes in the system browser, an FD_PUBLIC_URL deck lands the
    // callback on another origin, and Google's opener policy can cut
    // window.opener. Coming back to this window is the signal then.
    const onFocus = () => { get().syncStatus() }
    window.addEventListener('message', handler)
    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('message', handler)
      window.removeEventListener('focus', onFocus)
    }
  },

  // Per-user, like the Google account itself: the policy and the send log
  // are the caller's own, so no admin is needed (and none can see others').
  fetchGmailSend: async () => {
    set({ gmailSendError: null })
    try {
      const policy = (await fetchJson('/fd/google/gmail-send')) as GmailSendPolicy
      set({ gmailSend: policy })
    } catch (exc) {
      set({ gmailSendError: _gmailSendError(exc) })
    }
  },

  saveGmailSend: async (patch) => {
    set({ gmailSendError: null })
    try {
      const policy = (await fetchJson('/fd/google/gmail-send', {
        method: 'PUT',
        body: JSON.stringify(patch),
      })) as GmailSendPolicy
      set({ gmailSend: policy })
      return true
    } catch (exc) {
      set({ gmailSendError: _gmailSendError(exc) })
      return false
    }
  },

  fetchGmailSends: async (limit = 10) => {
    set({ gmailSendError: null })
    try {
      const data = await fetchJson(`/fd/google/gmail-sends?limit=${limit}`)
      set({ gmailSends: Array.isArray(data?.sends) ? (data.sends as GmailSendRecord[]) : [] })
    } catch (exc) {
      set({ gmailSendError: _gmailSendError(exc) })
    }
  },
}))

// The next person signing in at this screen must not see the outgoing user's
// sending policy or what their agents sent, even for the moment before the
// Connections card refetches.
registerSignOutTeardown(() => {
  useGoogleAuthStore.setState({ gmailSend: null, gmailSends: [], gmailSendError: null })
})
