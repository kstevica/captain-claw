import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Send,
  Loader2,
  StopCircle,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  MessageSquare,
  Wrench,
  Paperclip,
  FileIcon,
  ImageIcon,
  XCircle,
  Zap,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { useChatStore } from '../../stores/chatStore'
import { uploadFileToAgent } from '../../services/fileTransfer'
import type { ChatMessage } from '../../services/agentChat'

interface Attachment {
  id: string
  file: File
  name: string
  size: number
  type: string
  preview?: string
  status: 'pending' | 'uploading' | 'uploaded' | 'error'
  uploadedPath?: string
  error?: string
}

/** Convert bare image file paths in message content to markdown images proxied through Flight Deck. */
function processImagePaths(content: string, agentHost?: string, agentPort?: number, agentAuth?: string): string {
  if (!agentPort) return content
  const imagePathRe = /^([`*]*)(\/?(?:\/[\w.@: -]+)+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))([`*]*)$/gm
  return content.replace(imagePathRe, (_match, _pre, filePath, _post) => {
    const params = new URLSearchParams({ path: filePath })
    if (agentAuth) params.set('token', agentAuth)
    const url = `/fd/agent-file-view/${encodeURIComponent(agentHost || 'localhost')}/${agentPort}?${params}`
    return `![](${url})`
  })
}

let attachId = 0
function nextAttachId() { return `emb-attach-${Date.now()}-${++attachId}` }

interface EmbeddedChatProps {
  containerId: string
  containerName: string
  host: string
  port: number
  auth: string
}

export function EmbeddedChat({ containerId, containerName, host, port, auth }: EmbeddedChatProps) {
  const [expanded, setExpanded] = useState(false)
  const openChat = useChatStore((s) => s.openChat)
  const session = useChatStore((s) => s.sessions.get(containerId))
  const sendMessage = useChatStore((s) => s.sendMessage)
  const cancelTask = useChatStore((s) => s.cancelTask)

  // Auto-connect when expanded for the first time
  useEffect(() => {
    if (expanded && !session) {
      openChat(containerId, containerName, host, port, auth)
    }
  }, [expanded, session, containerId, containerName, host, port, auth, openChat])

  const messageCount = session?.messages.filter((m) => m.role === 'user' || m.role === 'assistant').length ?? 0
  const lastTokPerSec = session?.lastTokPerSec ?? 0
  const avgTokPerSec = session?.avgTokPerSec ?? 0

  return (
    <div className="border-t border-zinc-800">
      {/* Toggle header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-4 py-2 text-xs font-medium text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200 transition-colors"
      >
        <MessageSquare className="h-3.5 w-3.5" />
        <span>Chat</span>
        {messageCount > 0 && (
          <span className="rounded-full bg-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-300">{messageCount}</span>
        )}
        {session?.busy && <Loader2 className="h-3 w-3 animate-spin text-violet-400" />}
        <div className="flex-1" />
        {lastTokPerSec > 0 && (
          <span className="flex items-center gap-1 text-xs text-zinc-500 font-mono" title={`avg: ${avgTokPerSec.toFixed(1)} tok/s`}>
            <Zap className="h-3 w-3 text-amber-500/70" />
            {lastTokPerSec.toFixed(1)} tok/s
            {avgTokPerSec > 0 && <span className="text-zinc-600">/ {avgTokPerSec.toFixed(1)} avg</span>}
          </span>
        )}
        {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
      </button>

      {/* Chat body */}
      {expanded && (
        <EmbeddedChatBody
          containerId={containerId}
          session={session}
          host={host}
          port={port}
          auth={auth}
          onSend={(content) => sendMessage(containerId, content)}
          onCancel={() => cancelTask(containerId)}
        />
      )}
    </div>
  )
}

function EmbeddedChatBody({
  containerId,
  session,
  host,
  port,
  auth,
  onSend,
  onCancel,
}: {
  containerId: string
  session: { messages: ChatMessage[]; connected: boolean; busy: boolean; statusText: string } | undefined
  host: string
  port: number
  auth: string
  onSend: (content: string) => void
  onCancel: () => void
}) {
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const connected = session?.connected ?? false
  const busy = session?.busy ?? false
  const messages = session?.messages ?? []

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, session?.statusText])

  useEffect(() => {
    inputRef.current?.focus()
  }, [containerId])

  const uploadAttachment = useCallback(async (att: Attachment) => {
    setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'uploading' } : a))
    try {
      const result = await uploadFileToAgent(host, port, auth, att.file)
      setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'uploaded', uploadedPath: result.path } : a))
    } catch (err) {
      setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'error', error: String(err) } : a))
    }
  }, [host, port, auth])

  const addFiles = useCallback((files: FileList | File[]) => {
    const newAtts: Attachment[] = Array.from(files).map((file) => {
      const att: Attachment = {
        id: nextAttachId(), file, name: file.name, size: file.size, type: file.type, status: 'pending',
      }
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e) => {
          setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, preview: e.target?.result as string } : a))
        }
        reader.readAsDataURL(file)
      }
      return att
    })
    setAttachments((prev) => [...prev, ...newAtts])
    newAtts.forEach((att) => uploadAttachment(att))
  }, [uploadAttachment])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const files: File[] = []
    for (let i = 0; i < e.clipboardData.items.length; i++) {
      const item = e.clipboardData.items[i]
      if (item.kind === 'file') {
        const file = item.getAsFile()
        if (file) files.push(file)
      }
    }
    if (files.length > 0) {
      e.preventDefault()
      addFiles(files)
    }
  }, [addFiles])

  const [dragOver, setDragOver] = useState(false)

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id))
  }, [])

  const handleSend = () => {
    const text = input.trim()
    const uploadedFiles = attachments.filter((a) => a.status === 'uploaded' && a.uploadedPath)
    if ((!text && uploadedFiles.length === 0) || !connected) return

    let content = text
    if (uploadedFiles.length > 0) {
      const refs = uploadedFiles.map((a) => `[Attached file: ${a.name} → ${a.uploadedPath}]`).join('\n')
      content = content ? `${content}\n\n${refs}` : refs
    }
    onSend(content)
    setInput('')
    setAttachments([])
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Build visible messages
  const visibleMessages = (() => {
    const all = messages.filter((m) => m.role !== 'tool' || m.tool_name)
    const result: ChatMessage[] = []
    let toolCount = 0
    for (let i = all.length - 1; i >= 0; i--) {
      const m = all[i]
      if (m.role === 'user' || m.role === 'assistant') {
        toolCount = 0
        result.unshift(m)
      } else {
        toolCount++
        if (toolCount <= 3) result.unshift(m)
      }
    }
    return result
  })()

  const pendingUploads = attachments.filter((a) => a.status === 'uploading').length

  return (
    <div
      className={`flex flex-col ${dragOver ? 'ring-1 ring-inset ring-violet-500/40' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={(e) => { e.preventDefault(); setDragOver(false) }}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files) }}
    >
      {/* Messages area */}
      <div className="max-h-72 overflow-y-auto px-4 py-2">
        {visibleMessages.length === 0 && connected && (
          <p className="py-4 text-center text-xs text-zinc-600">Send a message to start chatting.</p>
        )}
        {!connected && (
          <p className="py-4 text-center text-xs text-zinc-600">Connecting...</p>
        )}

        {visibleMessages.map((msg) => (
          <EmbeddedMessage key={msg.id} message={msg} agentHost={host} agentPort={port} agentAuth={auth} />
        ))}

        {busy && (
          <div className="mb-2 flex items-center gap-2 text-[11px] text-zinc-500">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>{session?.statusText || 'Working...'}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Attachments */}
      {attachments.length > 0 && (
        <div className="border-t border-zinc-800/50 px-4 py-1.5">
          <div className="flex flex-wrap gap-1.5">
            {attachments.map((att) => (
              <div
                key={att.id}
                className={`flex items-center gap-1 rounded-md border px-1.5 py-1 text-[10px] ${
                  att.status === 'error' ? 'border-red-800 bg-red-950/30 text-red-400'
                  : att.status === 'uploading' ? 'border-violet-800/50 bg-violet-950/20 text-violet-300'
                  : 'border-zinc-700 bg-zinc-800/50 text-zinc-400'
                }`}
              >
                {att.preview ? (
                  <img src={att.preview} alt="" className="h-4 w-4 rounded object-cover" />
                ) : att.type.startsWith('image/') ? (
                  <ImageIcon className="h-3 w-3" />
                ) : (
                  <FileIcon className="h-3 w-3" />
                )}
                <span className="max-w-[80px] truncate">{att.name}</span>
                {att.status === 'uploading' && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
                <button onClick={() => removeAttachment(att.id)} className="text-zinc-600 hover:text-zinc-300">
                  <XCircle className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-zinc-800/50 px-3 py-2">
        <div className="flex items-end gap-1.5">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!connected}
            className="flex h-[32px] items-center rounded px-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
            title="Attach file"
          >
            <Paperclip className="h-3.5 w-3.5" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                addFiles(e.target.files)
                e.target.value = ''
              }
            }}
          />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={connected ? 'Message...' : 'Connecting...'}
            disabled={!connected}
            rows={1}
            className="max-h-20 min-h-[32px] flex-1 resize-none rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none disabled:opacity-40"
            style={{ height: 'auto', overflow: 'hidden' }}
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = Math.min(el.scrollHeight, 80) + 'px'
            }}
          />
          {busy ? (
            <button
              onClick={onCancel}
              className="flex h-[32px] items-center gap-1 rounded-md bg-red-600/20 px-2 text-[11px] font-medium text-red-400 hover:bg-red-600/30"
            >
              <StopCircle className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={(!input.trim() && attachments.filter((a) => a.status === 'uploaded').length === 0) || !connected || pendingUploads > 0}
              className="flex h-[32px] items-center rounded-md bg-violet-600 px-2 text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {pendingUploads > 0 ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function EmbeddedMessage({ message, agentHost, agentPort, agentAuth }: { message: ChatMessage; agentHost?: string; agentPort?: number; agentAuth?: string }) {
  const [toolExpanded, setToolExpanded] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="mb-2 flex justify-end">
        <div className="max-w-[90%] rounded-lg rounded-br-sm bg-violet-600/20 px-2.5 py-1.5">
          <div className="fd-markdown text-xs text-zinc-200">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{processImagePaths(message.content, agentHost, agentPort, agentAuth)}</Markdown>
          </div>
          <span className="mt-0.5 block text-right text-[9px] text-zinc-600">{formatTime(message.timestamp)}</span>
        </div>
      </div>
    )
  }

  if (message.role === 'tool') {
    return (
      <div className="mb-1">
        <button
          onClick={() => setToolExpanded(!toolExpanded)}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800/50"
        >
          <Wrench className="h-2.5 w-2.5" />
          <span className="font-medium">{message.tool_name}</span>
          {toolExpanded ? <ChevronDown className="h-2.5 w-2.5" /> : <ChevronRight className="h-2.5 w-2.5" />}
        </button>
        {toolExpanded && message.content && (
          <pre className="ml-4 mt-0.5 max-h-24 overflow-auto rounded bg-zinc-900/80 p-1.5 text-[10px] text-zinc-400 font-mono">
            {message.content.slice(0, 1000)}
            {message.content.length > 1000 && '\n...truncated'}
          </pre>
        )}
      </div>
    )
  }

  if (message.role === 'system') {
    return (
      <div className="mb-2 flex justify-center">
        <div className="rounded bg-zinc-800/40 px-2 py-1">
          <p className="text-[10px] text-zinc-500">{message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant
  const cleanContent = stripSuggestions(message.content)
  return (
    <div className="mb-2">
      <div className={`max-w-[90%] rounded-lg rounded-bl-sm bg-zinc-800/60 px-2.5 py-1.5 ${message.replay ? 'opacity-60' : ''}`}>
        <div className="fd-markdown text-xs text-zinc-300">
          <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{processImagePaths(cleanContent, agentHost, agentPort, agentAuth)}</Markdown>
        </div>
        <div className="mt-0.5 flex items-center gap-1.5 text-[9px] text-zinc-600">
          <span>{formatTime(message.timestamp)}</span>
          {message.model && <span className="font-mono">{message.model}</span>}
          {message.replay && <span>(history)</span>}
        </div>
      </div>
    </div>
  )
}

function stripSuggestions(text: string): string {
  let cleaned = text.replace(/\n---\n[💡🔔].*(?:rate good|rate bad).*$/is, '')
  cleaned = cleaned.replace(/\nSUGGESTED NEXT STEPS[\s\S]*$/i, '')
  return cleaned.trimEnd()
}

function formatTime(ts: string): string {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch { return '' }
}
