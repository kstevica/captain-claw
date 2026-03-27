/**
 * WebSocket client for direct chat with a Captain Claw agent.
 * Connects to CC's /ws endpoint on the agent's web port.
 */

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
}

type EventHandler = (data: Record<string, unknown>) => void

export class AgentChatWS {
  private ws: WebSocket | null = null
  private handlers = new Map<string, Set<EventHandler>>()
  private _connected = false
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
    if (this.ws) this.disconnect()

    // Route through FD backend proxy to avoid CORS
    const tokenParam = this.auth ? `?token=${encodeURIComponent(this.auth)}` : ''
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${wsProto}//${window.location.host}/fd/agent-ws/${encodeURIComponent(this.host)}/${this.port}${tokenParam}`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this._connected = true
      this.emit('_connected', {})
    }

    this.ws.onclose = () => {
      this._connected = false
      this.emit('_disconnected', {})
      this.ws = null
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

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
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

  on(event: string, handler: EventHandler): () => void {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set())
    this.handlers.get(event)!.add(handler)
    return () => { this.handlers.get(event)?.delete(handler) }
  }

  private emit(event: string, data: Record<string, unknown>) {
    this.handlers.get(event)?.forEach((h) => h(data))
  }
}
