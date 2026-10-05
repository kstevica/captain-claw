/**
 * WebSocket client for direct chat with a Captain Claw agent.
 * Connects to CC's /ws endpoint on the agent's web port.
 */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { sharedWsUrl } from '../utils/sharedAgent'

export interface TokenUsage {
  prompt_tokens?: number
  completion_tokens?: number
  cache_read_input_tokens?: number
  cache_creation_input_tokens?: number
  total_tokens?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  timestamp: string
  replay?: boolean
  tool_name?: string
  tool_arguments?: Record<string, unknown>
  tool_output?: string
  model?: string
  approval_request_id?: string
  approval_category?: string
  approval_resolved?: boolean
  peer_name?: string
  /** Live between-step narration blurb (rendered as a subtle system line). */
  narration?: boolean
  /** Frozen cumulative turn token usage, stamped onto the last tool message
   *  of an activity group when the turn ends. */
  usage?: TokenUsage
}

type EventHandler = (data: Record<string, unknown>) => void

export class AgentChatWS {
  private ws: WebSocket | null = null
  private handlers = new Map<string, Set<EventHandler>>()
  private _connected = false
  // Auto-reconnect state. The proxy + agent both have keepalives, but transient
  // network blips (laptop sleep, wifi handoff) still close the socket — we want
  // FD to silently re-establish so the user doesn't have to re-click "Chat".
  private _shouldReconnect = false
  private _reconnectAttempt = 0
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null
  readonly agentId: string
  readonly host: string
  readonly port: number
  readonly auth: string
  readonly lane: string
  /** Shared mode: an agent another deck user shared with us, reached only
   *  through Flight Deck's member route by its agent_ref — no host, port or
   *  access token ever leaves the browser. Empty for our own agents. */
  readonly sharedRef: string
  // Shared mode: the `fd_close` frame Flight Deck sends right before it
  // closes the socket — its reason is the one to show.
  private _lastFdClose: { code: number; reason: string } | null = null
  // Shared mode: a 4001 (FD token expired) gets ONE refresh-and-reopen per
  // welcome; a second one is final.
  private _authRetried = false
  private _unsubAuth: (() => void) | null = null

  constructor(agentId: string, host: string, port: number, auth: string, lane: string = '', opts?: { sharedRef?: string }) {
    this.agentId = agentId
    this.host = host
    this.port = port
    this.auth = auth
    // Which parallel context on the agent this socket talks to. Empty (or 'A')
    // means the agent's main context, so nothing about existing callers
    // changes — see docs/queue-lanes-plan.md.
    this.lane = lane
    this.sharedRef = opts?.sharedRef || ''
  }

  get connected() { return this._connected }

  /** A socket is open (or opening) but not yet usable. Shared mode only
   *  becomes usable on the agent's `welcome`, after Flight Deck's checks. */
  get connecting() { return !!this.ws && !this._connected }

  connect() {
    if (this.ws) this._teardownSocket()
    this._shouldReconnect = true
    // An explicit (re)connect is a fresh start for the one-refresh budget.
    this._authRetried = false
    if (this.sharedRef) this._watchToken()
    this._openSocket()
  }

  // Shared mode: Flight Deck re-checks the FD token's expiry on live member
  // sockets, so hand it every rotated token while the socket is open.
  private _watchToken() {
    if (this._unsubAuth) return
    this._unsubAuth = useAuthStore.subscribe((state, prev) => {
      if (!state.token || state.token === prev.token) return
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'fd_auth', fd_token: state.token }))
      }
    })
  }

  private _openSocket() {
    if (this.sharedRef) {
      this._openSharedSocket()
      return
    }
    // Route through FD backend proxy to avoid CORS
    const params = new URLSearchParams()
    if (this.auth) params.set('token', this.auth)
    // The caller's FD JWT — the backend refuses the socket unless the user
    // owns the target agent (HTTP middleware can't guard WebSockets).
    const fdToken = useAuthStore.getState().token
    if (fdToken) params.set('fd_token', fdToken)
    // Lane A is the agent's main context — send nothing, so the URL is
    // byte-identical to what every pre-lane client produced.
    if (this.lane && this.lane !== 'A') params.set('lane', this.lane)
    const qs = params.toString() ? `?${params}` : ''
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${wsProto}//${window.location.host}/fd/agent-ws/${encodeURIComponent(this.host)}/${this.port}${qs}`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this._connected = true
      this._reconnectAttempt = 0
      this.emit('_connected', {})
    }

    this.ws.onclose = () => {
      const wasConnected = this._connected
      this._connected = false
      this.ws = null
      this.emit('_disconnected', { wasConnected })
      if (this._shouldReconnect) this._scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.emit('_error', { message: 'WebSocket connection failed' })
    }

    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const type = data.type || 'unknown'
        this.emit(type, data)
        this.emit('_any', data)
      } catch {
        // ignore non-JSON messages
      }
    }
  }

  private _openSharedSocket() {
    const fdToken = useAuthStore.getState().token || ''
    this._lastFdClose = null
    const ws = new WebSocket(sharedWsUrl(this.sharedRef, this.lane || 'A', fdToken, window.location))
    this.ws = ws

    // Flight Deck accepts first and only then checks membership, the agent and
    // its handshake — so the socket counts as connected once the agent's
    // welcome arrives, not on open.
    ws.onclose = (ev) => {
      const wasConnected = this._connected
      this._connected = false
      this.ws = null
      this.emit('_disconnected', { wasConnected })
      if (ev.code < 4000 || ev.code > 4999) {
        // Network blip, laptop sleep: reconnect as for our own agents.
        if (this._shouldReconnect) this._scheduleReconnect()
        return
      }
      const fdClose = this._lastFdClose
      this._lastFdClose = null
      if (ev.code === 4001 && !this._authRetried && this._shouldReconnect) {
        // FD token expired: refresh it and reopen — once.
        this._authRetried = true
        void refreshAccessToken().then((ok) => {
          if (!this._shouldReconnect || this.ws) return  // closed or reopened meanwhile
          if (ok) { this._openSocket(); return }
          this._shouldReconnect = false
          this.emit('_closed', { code: ev.code, reason: fdClose?.reason || ev.reason || '' })
        })
        return
      }
      // Anything else Flight Deck decided is final — no backoff loop.
      this._shouldReconnect = false
      this.emit('_closed', { code: ev.code, reason: fdClose?.reason || ev.reason || '' })
    }

    ws.onerror = () => {
      this.emit('_error', { message: 'WebSocket connection failed' })
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const type = data.type || 'unknown'
        if (type === 'fd_close') {
          this._lastFdClose = { code: Number(data.code) || 0, reason: String(data.reason || '') }
        } else if (type === 'welcome' && !this._connected) {
          this._connected = true
          this._reconnectAttempt = 0
          this._authRetried = false
          this.emit('_connected', {})
        }
        this.emit(type, data)
        this.emit('_any', data)
      } catch {
        // ignore non-JSON messages
      }
    }
  }

  private _scheduleReconnect() {
    if (this._reconnectTimer) return
    // Exponential backoff capped at 15s. The first retry fires fast (~500ms)
    // so quick blips feel instant; subsequent retries back off so we don't
    // hammer a dead agent.
    const delay = Math.min(15000, 500 * 2 ** this._reconnectAttempt)
    this._reconnectAttempt++
    this.emit('_reconnecting', { attempt: this._reconnectAttempt, delayMs: delay })
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null
      if (!this._shouldReconnect) return
      this._openSocket()
    }, delay)
  }

  private _teardownSocket() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer)
      this._reconnectTimer = null
    }
    if (this.ws) {
      try { this.ws.onclose = null } catch { /* ignore */ }
      try { this.ws.close() } catch { /* ignore */ }
      this.ws = null
    }
  }

  disconnect() {
    this._shouldReconnect = false
    this._reconnectAttempt = 0
    this._teardownSocket()
    this._connected = false
    if (this._unsubAuth) { this._unsubAuth(); this._unsubAuth = null }
  }

  send(content: string, opts?: { noNextSteps?: boolean }) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({
      type: 'chat',
      content,
      // Queue-dispatched turns don't want the post-turn "suggested next
      // steps" round-trip — the next message is already written — nor a
      // task-rephrase, which would rewrite instructions the user chose
      // word by word.
      ...(opts?.noNextSteps ? { no_next_steps: true, no_rephrase: true } : {}),
    }))
  }

  sendBtw(content: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({ type: 'btw', content }))
  }

  cancel() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({ type: 'cancel' }))
  }

  sendJSON(data: Record<string, unknown>) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify(data))
  }

  on(event: string, handler: EventHandler): () => void {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set())
    this.handlers.get(event)!.add(handler)
    return () => { this.handlers.get(event)?.delete(handler) }
  }

  private emit(event: string, data: Record<string, unknown>) {
    this.handlers.get(event)?.forEach((h) => h(data))
  }
}
