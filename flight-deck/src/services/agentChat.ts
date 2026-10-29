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
  /** `rephrase`: the agent's restatement of your request (task_rephrase),
   *  shown as a small preview like an automated result. `image` / `html_file`:
   *  a picture or a page the turn made — `content` is its path on the agent. */
  role: 'user' | 'assistant' | 'system' | 'tool' | 'rephrase' | 'image' | 'html_file'
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
  /** Flight Deck chat-log row id — set when the row came from the log or
   *  once it was persisted to it. Topics jump to a divider by this. */
  rowId?: number
  /** A user turn that came from another surface ("whatsapp", "device",
   *  "glasses"); absent for turns typed here. Shown as "via WhatsApp". */
  via?: string
  /** Where it was sent from, as a person names it ("M5Stack StickS3 (stick)",
   *  "Captain Claw app (Android)") — the agent's turn_source label. */
  sourceLabel?: string
  /** A topic boundary in the continuous per-lane transcript: the agent's
   *  context moved to another session (`new`, `resume`) or was reset in
   *  place (`clear`). Rendered as a divider; the Topics list is built from
   *  these rows. */
  topic?: TopicMark
  /** The result of a scheduled task (Flight Deck's Scheduled tasks), posted
   *  into this chat: labelled, and links back to the task. */
  task?: TaskMark
  /** An automated turn's result the automation lane mirrored into this chat
   *  (Autonomous Work, a plan, a scheduled job): what ran it, as a person
   *  reads it. Shown as a small preview that opens in a dialog. */
  automation?: string
}

export interface TaskMark {
  id: string
  name: string
  /** The agent session the task's runs live in. */
  session_id?: string
}

export interface TopicMark {
  action: 'new' | 'resume' | 'clear'
  /** The agent session the lane's context is on from this row down. */
  agentSessionId: string
  /** Optional title (phase 2: agent-generated); the Topics list otherwise
   *  names a topic after its first user message. */
  title?: string
  /** `web` — this client sent the command; `agent` — the agent reported a
   *  session the transcript had no divider for (another channel, a restart). */
  source: 'web' | 'agent'
}

type EventHandler = (data: Record<string, unknown>) => void

// A picture the chat composers attach is written as an `[Attached image: <path>]`
// line — what the user's bubble shows. On the wire it travels as `image_paths`:
// the agent then words the turn for its model — a model that sees images looks
// at it itself, otherwise a vision model describes it before the turn.
const IMAGE_REF_LINE_RE = /^\[Attached image:\s*(.+?)\s*\]$/
// The hint earlier builds appended after image refs (queued items may carry it).
const LEGACY_IMAGE_HINT_RE = /^\(To view\/understand the image\(s\), call image_vision.*\)$/

export function imageRef(path: string): string {
  return `[Attached image: ${path}]`
}

/** The text and the `image_paths` of an outgoing message (see imageRef). */
export function splitImageRefs(content: string): { text: string; imagePaths: string[] } {
  const imagePaths: string[] = []
  const kept: string[] = []
  for (const line of content.split('\n')) {
    const ref = line.trim().match(IMAGE_REF_LINE_RE)
    if (ref) {
      if (!imagePaths.includes(ref[1])) imagePaths.push(ref[1])
      continue
    }
    if (LEGACY_IMAGE_HINT_RE.test(line.trim())) continue
    kept.push(line)
  }
  return imagePaths.length ? { text: kept.join('\n').trim(), imagePaths } : { text: content, imagePaths }
}

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
  // Where the agent is. Not fixed: an agent Flight Deck restarted can come
  // back on another port (see retarget and the proxy's fd_agent_endpoint).
  host: string
  port: number
  auth: string
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
  // Own agents: when Flight Deck last refused our FD token (4001) and we
  // refreshed it. One refresh-and-reopen per 30s, however long the socket
  // has been away.
  private _lastAuthRefresh = 0
  private _unsubAuth: (() => void) | null = null
  // Asleep (Flight Deck's agent controller stopped it for being idle): the
  // proxy holds the socket until we say something. A chat the user opens
  // (connect) wakes it at once; one that only reconnected leaves it asleep.
  private _wakeOnAsleep = false
  private _asleep = false

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

  /** The agent is asleep and this socket is held for it (see wake). */
  get asleep() { return this._asleep }

  /** A socket is open (or opening) but not yet usable: it becomes usable on
   *  the agent's `welcome`. Flight Deck's proxy opens the socket first and
   *  holds it while the agent is still starting. */
  get connecting() { return !!this.ws && !this._connected }

  /** Point the next connect at a new address (the agent moved). */
  retarget(host: string, port: number, auth?: string) {
    this.host = host
    this.port = port
    if (auth) this.auth = auth
  }

  connect() {
    if (this.ws) this._teardownSocket()
    this._shouldReconnect = true
    this._wakeOnAsleep = true
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
    // Which agent this is: Flight Deck follows a process agent (`proc-<slug>`)
    // to its current port when it came back on a different one.
    if (this.agentId) params.set('agent', this.agentId)
    const qs = params.toString() ? `?${params}` : ''
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${wsProto}//${window.location.host}/fd/agent-ws/${encodeURIComponent(this.host)}/${this.port}${qs}`

    this.ws = new WebSocket(url)

    // Connected means the agent's welcome, not the socket's open: the proxy
    // opens the socket first and holds it while the agent is still starting.
    this.ws.onclose = (ev) => {
      const wasConnected = this._connected
      this._connected = false
      this._asleep = false
      this.ws = null
      this.emit('_disconnected', { wasConnected, code: ev.code })
      if (!this._shouldReconnect) return
      // Flight Deck refused our FD token: it expired, or Flight Deck restarted
      // and no longer accepts tokens it signed before. Retrying it would be
      // refused forever — get a fresh one and reopen right away.
      if (ev.code === 4001 && Date.now() - this._lastAuthRefresh > 30_000) {
        this._lastAuthRefresh = Date.now()
        void refreshAccessToken().then((ok) => {
          if (!this._shouldReconnect || this.ws || this._reconnectTimer) return
          if (ok) this._openSocket()
          else this._scheduleReconnect()
        })
        return
      }
      this._scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.emit('_error', { message: 'WebSocket connection failed' })
    }

    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        const type = data.type || 'unknown'
        if (type === 'welcome' && !this._connected) {
          this._connected = true
          this._asleep = false
          this._reconnectAttempt = 0
          this.emit('_connected', {})
        } else if (type === 'fd_agent_endpoint' && Number(data.port) > 0) {
          // The agent came back on another port: reconnect there next time.
          this.port = Number(data.port)
        } else if (type === 'fd_agent_asleep') {
          this._asleep = true
          if (this._wakeOnAsleep) {
            this.wake()
            data.waking = true
          }
        }
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
    // 0.5s, 1s, then every 2s for about a minute: a Flight Deck restart is
    // over in seconds, and the chat should be back as soon as it is. Then
    // every 10s for an agent that stays away. While Flight Deck is up, its
    // proxy holds each attempt until the agent answers, so these are rare.
    const n = this._reconnectAttempt
    const delay = n < 2 ? 500 * 2 ** n : n < 30 ? 2000 : 10000
    this._reconnectAttempt++
    this.emit('_reconnecting', { attempt: this._reconnectAttempt, delayMs: delay })
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null
      if (!this._shouldReconnect) return
      this._wakeOnAsleep = false
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

  /** Ids of the turns this socket sent (the agent echoes them back on the
   *  user frame, so our own echo is told apart from another surface's turn). */
  private sentIds: string[] = []

  isOwnMessage(clientMsgId: string): boolean {
    return this.sentIds.includes(clientMsgId)
  }

  /** Returns the frame's client_msg_id (the agent echoes it when it takes or
   *  refuses the turn), or undefined when the socket wasn't open. */
  send(content: string, opts?: { noNextSteps?: boolean; imagePaths?: string[]; reasoningEffort?: string }): string | undefined {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return undefined
    const clientMsgId = `web_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
    this.sentIds.push(clientMsgId)
    if (this.sentIds.length > 200) this.sentIds.splice(0, this.sentIds.length - 200)
    this.ws.send(JSON.stringify({
      type: 'chat',
      content,
      client_msg_id: clientMsgId,
      // Queue-dispatched turns don't want the post-turn "suggested next
      // steps" round-trip — the next message is already written — nor a
      // task-rephrase, which would rewrite instructions the user chose
      // word by word.
      ...(opts?.noNextSteps ? { no_next_steps: true, no_rephrase: true } : {}),
      ...(opts?.imagePaths?.length ? { image_paths: opts.imagePaths } : {}),
      // The chat box's reasoning-effort pick for this turn (absent: the
      // model's default).
      ...(opts?.reasoningEffort ? { reasoning_effort: opts.reasoningEffort } : {}),
    }))
    return clientMsgId
  }

  /** Ask Flight Deck to wake the sleeping agent this socket is held for. */
  wake() {
    this._wakeOnAsleep = false
    if (!this._asleep || !this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this._asleep = false
    this.ws.send(JSON.stringify({ type: 'fd_wake' }))
  }

  /** A note for the agent's running turn. `runningOnly`: for that turn
   *  only — with none on, the agent refuses it (`busy: false` on the /btw
   *  command_result) instead of keeping it for the next turn. */
  sendBtw(content: string, opts?: { runningOnly?: boolean }) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({ type: 'btw', content, ...(opts?.runningOnly ? { running_only: true } : {}) }))
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
