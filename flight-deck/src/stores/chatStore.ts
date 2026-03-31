import { create } from 'zustand'
import { AgentChatWS, type ChatMessage } from '../services/agentChat'
import { useContainerStore } from './containerStore'
import { useLocalAgentStore } from './localAgentStore'
import { useProcessStore } from './processStore'

interface AgentModelInfo {
  id: string
  label: string
  selector: string
}

interface AgentPersonalityInfo {
  id: string
  name: string
  description?: string
}

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
  models: AgentModelInfo[]
  personalities: AgentPersonalityInfo[]
  activeModel: string
  activePersonality: string
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
  setModel: (containerId: string, selector: string) => void
  setPersonality: (containerId: string, personalityId: string) => void
  respondToApproval: (containerId: string, requestId: string, approved: boolean) => void
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
      models: [],
      personalities: [],
      activeModel: '',
      activePersonality: '',
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
      const models = (data.models as AgentModelInfo[] || [])
      const personalities = (data.personalities as AgentPersonalityInfo[] || [])
      const patch: Partial<ChatSession> = { models, personalities }
      if (name) patch.statusText = `Session: ${name}`
      updateSession(containerId, patch)

      // Send peer agent awareness to the connected agent
      const containers = useContainerStore.getState().containers
      const localAgents = useLocalAgentStore.getState().agents
      const { getForwardingTask: getFwd, getConsultApproval } = useContainerStore.getState()
      const peers: { name: string; description: string; forwardingTask: string; host: string; port: number; auth: string; requireApproval: boolean }[] = []
      for (const c of containers) {
        if (c.id === containerId || c.status !== 'running' || !c.web_port) continue
        peers.push({
          name: c.agent_name || c.name,
          description: c.description || '',
          forwardingTask: getFwd(c.id),
          host: 'localhost',
          port: c.web_port,
          auth: c.web_auth || '',
          requireApproval: getConsultApproval(c.id),
        })
      }
      for (const a of localAgents) {
        if (a.id === containerId || a.status !== 'online') continue
        peers.push({
          name: a.name,
          description: a.description || '',
          forwardingTask: a.forwardingTask || '',
          host: a.host,
          port: a.port,
          auth: a.authToken || '',
          requireApproval: a.consultApproval ?? false,
        })
      }
      const processes = useProcessStore.getState().processes
      const { getForwardingTask: getProcFwd, getConsultApproval: getProcApproval } = useProcessStore.getState()
      for (const p of processes) {
        if (p.slug === containerId || p.status !== 'running' || !p.web_port) continue
        peers.push({
          name: p.name || p.slug,
          description: p.description || '',
          forwardingTask: getProcFwd(p.slug),
          host: 'localhost',
          port: p.web_port,
          auth: '',
          requireApproval: getProcApproval(p.slug),
        })
      }
      if (peers.length > 0) {
        // Derive the FD base URL so the agent can call back to consult peers
        const fdUrl = `${window.location.protocol}//${window.location.host}`
        ws.sendJSON({ type: 'peer_agents', agents: peers, fd_url: fdUrl })
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

    ws.on('approval_request', (data) => {
      const msg: ChatMessage = {
        id: nextId(),
        role: 'system',
        content: data.message as string || 'Approval requested',
        timestamp: new Date().toISOString(),
        approval_request_id: data.id as string,
        approval_category: data.category as string || '',
      }
      addMessage(containerId, msg)
    })

    ws.on('peer_activity', (data) => {
      const peerName = (data.peer_name as string) || 'Peer agent'
      const activityType = (data.activity_type as string) || ''
      const detail = (data.detail as string) || ''

      // Only show tool usage — skip status, connecting, done, errors
      if (activityType !== 'tool' && activityType !== 'thinking') return
      // For thinking, only show if it mentions a tool
      if (activityType === 'thinking' && !detail.startsWith('Using ')) return

      const toolName = activityType === 'tool' ? detail : detail.replace('Using ', '')
      if (!toolName) return

      const msg: ChatMessage = {
        id: nextId(),
        role: 'tool',
        content: '',
        timestamp: new Date().toISOString(),
        tool_name: toolName,
        peer_name: peerName,
      }
      addMessage(containerId, msg)
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
      const command = (data.command as string || '').trim().toLowerCase()
      // If /clear was executed, wipe local messages first
      if (command === '/clear') {
        clearMessages(containerId)
      }
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

  setModel: (containerId, selector) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.sendJSON({ type: 'set_model', selector })
      updateSession(containerId, { activeModel: selector })
    }
  },

  setPersonality: (containerId, personalityId) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.sendJSON({ type: 'set_personality', personality_id: personalityId })
      updateSession(containerId, { activePersonality: personalityId })
    }
  },

  respondToApproval: (containerId, requestId, approved) => {
    const session = get().sessions.get(containerId)
    if (session) {
      session.ws.sendJSON({ type: 'approval_response', id: requestId, approved })
      // Mark the approval message as resolved
      const messages = session.messages.map((m) =>
        m.approval_request_id === requestId ? { ...m, approval_resolved: true } : m
      )
      updateSession(containerId, { messages })
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

function clearMessages(containerId: string) {
  useChatStore.setState((state) => {
    const session = state.sessions.get(containerId)
    if (!session) return state
    const updated = { ...session, messages: [] }
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
