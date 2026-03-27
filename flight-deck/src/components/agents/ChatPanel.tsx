import { useState, useRef, useEffect } from 'react'
import {
  X,
  Send,
  Loader2,
  StopCircle,
  MessageSquare,
  Wrench,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Minus,
  Forward,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChatStore } from '../../stores/chatStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useContainerStore } from '../../stores/containerStore'
import { SendContextModal } from './SendContextModal'
import type { ChatMessage } from '../../services/agentChat'

export function ChatPanel() {
  const { sessions, activeChatId, chatOpen, closeChat, switchChat, disconnectChat, sendMessage, cancelTask } =
    useChatStore()
  const localAgents = useLocalAgentStore((s) => s.agents)
  const containers = useContainerStore((s) => s.containers)
  const [showSendContext, setShowSendContext] = useState(false)

  const session = activeChatId ? sessions.get(activeChatId) : null

  if (!chatOpen || !session) return null

  const chatTabs = Array.from(sessions.values())

  // Build target list for context transfer (all reachable agents except the current one)
  const targets = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port && c.id !== activeChatId)
      .map((c) => ({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })),
    ...localAgents
      .filter((a) => a.status === 'online' && a.id !== activeChatId)
      .map((a) => ({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })),
  ]

  return (
    <div className="flex w-[480px] flex-col border-l border-zinc-800 bg-zinc-950/80">
      {/* Tab bar */}
      <div className="flex items-center border-b border-zinc-800 bg-zinc-900/60">
        <div className="flex flex-1 items-center gap-0.5 overflow-x-auto px-1 py-1">
          {chatTabs.map((s) => (
            <button
              key={s.containerId}
              onClick={() => switchChat(s.containerId)}
              className={`group flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                s.containerId === activeChatId
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${s.connected ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
              <span className="max-w-[100px] truncate">{s.containerName}</span>
              <button
                onClick={(e) => { e.stopPropagation(); disconnectChat(s.containerId) }}
                className="ml-0.5 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-zinc-300 group-hover:opacity-100"
              >
                <X className="h-3 w-3" />
              </button>
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowSendContext(true)}
          title="Send context to another agent"
          className="mr-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <Forward className="h-4 w-4" />
        </button>
        <button onClick={closeChat} className="mr-2 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <Minus className="h-4 w-4" />
        </button>
      </div>

      {/* Chat content */}
      <ChatContent
        session={session}
        onSend={(content) => sendMessage(session.containerId, content)}
        onCancel={() => cancelTask(session.containerId)}
      />

      {/* Send context modal */}
      {showSendContext && (
        <SendContextModal
          sourceId={session.containerId}
          sourceName={session.containerName}
          messages={session.messages}
          targets={targets}
          onClose={() => setShowSendContext(false)}
        />
      )}
    </div>
  )
}

function ChatContent({
  session,
  onSend,
  onCancel,
}: {
  session: { containerId: string; messages: ChatMessage[]; connected: boolean; busy: boolean; statusText: string }
  onSend: (content: string) => void
  onCancel: () => void
}) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages.length, session.statusText])

  // Focus input when panel opens
  useEffect(() => {
    inputRef.current?.focus()
  }, [session.containerId])

  const handleSend = () => {
    const text = input.trim()
    if (!text || !session.connected) return
    onSend(text)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Build visible messages: keep user + assistant, only last 3 tool/system
  const visibleMessages = (() => {
    const all = session.messages.filter((m) => m.role !== 'tool' || m.tool_name)
    const result: ChatMessage[] = []
    // Walk backwards to count tool/system messages per "turn" (between user messages)
    let toolCount = 0
    for (let i = all.length - 1; i >= 0; i--) {
      const m = all[i]
      if (m.role === 'user' || m.role === 'assistant') {
        toolCount = 0 // reset on each user/assistant message
        result.unshift(m)
      } else {
        // tool or system
        toolCount++
        if (toolCount <= 3) {
          result.unshift(m)
        }
      }
    }
    return result
  })()

  return (
    <>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {visibleMessages.length === 0 && session.connected && (
          <div className="mt-12 text-center">
            <MessageSquare className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">Send a message to start chatting with this agent.</p>
          </div>
        )}

        {!session.connected && (
          <div className="mt-12 text-center">
            <AlertCircle className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">Connecting to agent...</p>
          </div>
        )}

        {visibleMessages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Busy indicator */}
        {session.busy && (
          <div className="mb-3 flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{session.statusText || 'Working...'}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-zinc-800 p-3">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={session.connected ? 'Send a message...' : 'Connecting...'}
            disabled={!session.connected}
            rows={1}
            className="max-h-32 min-h-[38px] flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none disabled:opacity-40"
            style={{ height: 'auto', overflow: 'hidden' }}
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = Math.min(el.scrollHeight, 128) + 'px'
            }}
          />
          {session.busy ? (
            <button
              onClick={onCancel}
              className="flex h-[38px] items-center gap-1.5 rounded-lg bg-red-600/20 px-3 text-xs font-medium text-red-400 hover:bg-red-600/30"
            >
              <StopCircle className="h-4 w-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim() || !session.connected}
              className="flex h-[38px] items-center gap-1.5 rounded-lg bg-violet-600 px-3 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              <Send className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const [toolExpanded, setToolExpanded] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="mb-3 flex justify-end">
        <div className="max-w-[85%] rounded-xl rounded-br-sm bg-violet-600/20 px-3.5 py-2.5">
          <div className="fd-markdown text-sm text-zinc-200">
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </div>
          <span className="mt-1 block text-right text-[10px] text-zinc-500">
            {formatTime(message.timestamp)}
          </span>
        </div>
      </div>
    )
  }

  if (message.role === 'tool') {
    return (
      <div className="mb-2">
        <button
          onClick={() => setToolExpanded(!toolExpanded)}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-400"
        >
          <Wrench className="h-3 w-3" />
          <span className="font-medium">{message.tool_name}</span>
          {toolExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </button>
        {toolExpanded && message.content && (
          <pre className="ml-6 mt-1 max-h-40 overflow-auto rounded-md bg-zinc-900/80 p-2 text-xs text-zinc-400 font-mono">
            {message.content.slice(0, 2000)}
            {message.content.length > 2000 && '\n...truncated'}
          </pre>
        )}
      </div>
    )
  }

  if (message.role === 'system') {
    return (
      <div className="mb-3 flex justify-center">
        <div className="max-w-[85%] rounded-lg bg-zinc-800/40 px-3 py-2">
          <p className="whitespace-pre-wrap text-xs text-zinc-500">{message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant — strip trailing suggestions/rating prompts
  const cleanContent = stripSuggestions(message.content)

  return (
    <div className="mb-3">
      <div className={`max-w-[85%] rounded-xl rounded-bl-sm bg-zinc-800/60 px-3.5 py-2.5 ${message.replay ? 'opacity-60' : ''}`}>
        <div className="fd-markdown text-sm text-zinc-300">
          <Markdown remarkPlugins={[remarkGfm]}>{cleanContent}</Markdown>
        </div>
        <div className="mt-1 flex items-center gap-2 text-[10px] text-zinc-600">
          <span>{formatTime(message.timestamp)}</span>
          {message.model && <span className="font-mono">{message.model}</span>}
          {message.replay && <span className="text-zinc-700">(history)</span>}
        </div>
      </div>
    </div>
  )
}

/** Strip trailing suggestion prompts and rating requests from CC responses. */
function stripSuggestions(text: string): string {
  // Remove "---\n💡 If this worked well..." rating block
  let cleaned = text.replace(/\n---\n[💡🔔].*(?:rate good|rate bad).*$/is, '')
  // Remove "SUGGESTED NEXT STEPS" block
  cleaned = cleaned.replace(/\nSUGGESTED NEXT STEPS[\s\S]*$/i, '')
  return cleaned.trimEnd()
}

function formatTime(ts: string): string {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}
