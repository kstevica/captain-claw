/**
 * WebSocket client for direct chat with a Captain Claw agent.
 * Connects to CC's /ws endpoint on the agent's web port.
 */

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

  constructor(agentId: string, host: string, port: number, auth: string) {
    this.agentId = agentId
    this.host = host
    this.port = port
    this.auth = auth
  }

  get connected() { return this._connected }

  connect() {
    if (this.ws) this._teardownSocket()
    this._shouldReconnect = true
    this._openSocket()
  }

  private _openSocket() {
    // Route through FD backend proxy to avoid CORS
    const tokenParam = this.auth ? `?token=${encodeURIComponent(this.auth)}` : ''
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${wsProto}//${window.location.host}/fd/agent-ws/${encodeURIComponent(this.host)}/${this.port}${tokenParam}`

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
  }

  send(content: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify({ type: 'chat', content }))
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
