// WebSocket connection to BotPort for real-time updates.
// Listens on the BotPort dashboard WS endpoint and dispatches events to stores.

import { getWsUrl } from '../stores/connectionStore'

type MessageHandler = (data: Record<string, unknown>) => void

class BotPortSocket {
  private ws: WebSocket | null = null
  private currentUrl: string = ''
  private handlers = new Map<string, Set<MessageHandler>>()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectDelay = 1000

  connect() {
    const url = getWsUrl()

    // If URL changed, close old connection first
    if (this.ws && this.currentUrl !== url) {
      this.disconnect()
    }

    if (this.ws?.readyState === WebSocket.OPEN) return

    this.currentUrl = url

    try {
      this.ws = new WebSocket(url)
    } catch {
      this.scheduleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.reconnectDelay = 1000
      this.emit('_connected', {})
    }

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as Record<string, unknown>
        const type = msg.type as string
        if (type) this.emit(type, msg)
        this.emit('*', msg)
      } catch { /* ignore malformed */ }
    }

    this.ws.onclose = () => {
      this.emit('_disconnected', {})
      this.scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = null
    this.ws?.close()
    this.ws = null
  }

  /** Disconnect and reconnect (picks up new URL from connectionStore). */
  reconnect() {
    this.disconnect()
    this.reconnectDelay = 1000
    this.connect()
  }

  send(msg: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg))
    }
  }

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set())
    this.handlers.get(type)!.add(handler)
    return () => this.handlers.get(type)?.delete(handler)
  }

  private emit(type: string, data: Record<string, unknown>) {
    this.handlers.get(type)?.forEach((h) => h(data))
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000)
      this.connect()
    }, this.reconnectDelay)
  }
}

export const botportWS = new BotPortSocket()
