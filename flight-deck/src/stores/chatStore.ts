import { create } from 'zustand'
import { AgentChatWS, type ChatMessage } from '../services/agentChat'

interface ChatSession {
  containerId: string
  containerName: string
  host: string
  port: number
  auth: string
  ws: AgentChatWS
  messages: ChatMessage[]
  connected: boolean
  busy: boolean // agent is processing
  statusText: string
}

interface ChatStore {
  sessions: Map<string, ChatSession>
  activeChatId: string | null
  chatOpen: boolean

  openChat: (id: string, name: string, host: string, port: number, auth: string) => void
  closeChat: () => void
  switchChat: (containerId: string) => void
  disconnectChat: (containerId: string) => void
  sendMessage: (containerId: string, content: string) => void
  sendBtw: (containerId: string, content: string) => void
  cancelTask: (containerId: string) => void
}

let msgCounter = 0
function nextId() { return `msg-${Date.now()}-${++msgCounter}` }

export const useChatStore = create<ChatStore>((set, get) => ({
  sessions: new Map(),
  activeChatId: null,
  chatOpen: false,

  openChat: (containerId, containerName, host, port, auth) => {
    const existing = get().sessions.get(containerId)
    if (existing) {
      // Already have a session, just activate it
      if (!existing.connected) {
        existing.ws.connect()
      }
      set({ activeChatId: containerId, chatOpen: true })
      return
    }

    const ws = new AgentChatWS(containerId, host, port, auth)
    const session: ChatSession = {
      containerId,
      containerName,
      host,
      port,
      auth,
      ws,
      messages: [],
      connected: false,
      busy: false,
      statusText: '',
    }

    const sessions = new Map(get().sessions)
    sessions.set(containerId, session)
    set({ sessions, activeChatId: containerId, chatOpen: true })

    // Wire up event handlers
    ws.on('_connected', () => {
      updateSession(containerId, { connected: true })
    })

    ws.on('_disconnected', () => {
      updateSession(containerId, { connected: false, busy: false, statusText: '' })
    })

    ws.on('welcome', (data) => {
      const sessionInfo = data.session as Record<string, unknown> | undefined
      const name = sessionInfo?.name as string || ''
      if (name) {
        updateSession(containerId, { statusText: `Session: ${name}` })
      }
    })

    ws.on('chat_message', (data) => {
      // Skip echoed user messages — we already add them locally in sendMessage
      if (data.role === 'user' && !data.replay) return
      const msg: ChatMessage = {
        id: nextId(),
        role: data.role as 'user' | 'assistant',
        content: data.content as string || '',
        timestamp: data.timestamp as string || new Date().toISOString(),
        replay: data.replay as boolean || false,
        model: data.model as string || '',
      }
      addMessage(containerId, msg)
      if (data.role === 'assistant' && !data.replay) {
        updateSession(containerId, { busy: false, statusText: '' })
      }
    })

    ws.on('replay_done', () => {
      updateSession(containerId, { busy: false, statusText: '' })
    })

    ws.on('status', (data) => {
      const text = data.text as string || data.status as string || ''
      // "ready", "idle", or empty status means agent is done
      const idle = !text || /^(ready|idle|done|completed)$/i.test(text)
      updateSession(containerId, { busy: !idle, statusText: idle ? '' : text })
    })

    ws.on('monitor', (data) => {
      const toolName = data.tool_name as string || ''
      const output = data.output as string || ''
      if (toolName && !data.replay) {
        const msg: ChatMessage = {
          id: nextId(),
          role: 'tool',
          content: output,
          timestamp: new Date().toISOString(),
          tool_name: toolName,
          tool_arguments: data.arguments as Record<string, unknown>,
          tool_output: output,
        }
        addMessage(containerId, msg)
      }
      updateSession(containerId, { busy: true, statusText: `Using ${toolName}...` })
    })

    ws.on('error', (data) => {
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.message as string || 'Unknown error',
        timestamp: new Date().toISOString(),
      }
      addMessage(containerId, msg)
      updateSession(containerId, { busy: false })
    })

    ws.on('command_result', (data) => {
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.content as string || '',
        timestamp: new Date().toISOString(),
      }
      addMessage(containerId, msg)
    })

    ws.connect()
  },

  closeChat: () => {
    set({ chatOpen: false })
  },

  switchChat: (containerId) => {
    set({ activeChatId: containerId, chatOpen: true })
  },

  disconnectChat: (containerId) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.disconnect()
      const sessions = new Map(get().sessions)
      sessions.delete(containerId)
      set((s) => ({
        sessions,
        activeChatId: s.activeChatId === containerId ? null : s.activeChatId,
        chatOpen: s.activeChatId === containerId ? false : s.chatOpen,
      }))
    }
  },

  sendMessage: (containerId, content) => {
    const session = get().sessions.get(containerId)
    if (!session) return
    // Add user message to local state
    const msg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }
    addMessage(containerId, msg)
    updateSession(containerId, { busy: true, statusText: 'Thinking...' })
    session.ws.send(content)
  },

  sendBtw: (containerId, content) => {
    const session = get().sessions.get(containerId)
    if (session) session.ws.sendBtw(content)
  },

  cancelTask: (containerId) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.cancel()
      updateSession(containerId, { busy: false, statusText: 'Cancelled' })
    }
  },
}))

// Helpers that update a session inside the map
function updateSession(containerId: string, patch: Partial<ChatSession>) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, ...patch }
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    return { sessions }
  })
}

function addMessage(containerId: string, msg: ChatMessage) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, messages: [...session.messages, msg] }
    const sessions = new Map(state.sessions)
    sessions.set(containerId, updated)
    return { sessions }
  })
}
