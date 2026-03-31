import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
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
  Paperclip,
  FileIcon,
  ImageIcon,
  Clipboard,
  XCircle,
  Copy,
  Check,
  Pin,
  ClipboardList,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChatStore } from '../../stores/chatStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { usePinnedStore } from '../../stores/pinnedStore'
import { useClipboardStore } from '../../stores/clipboardStore'
import { SendContextModal } from './SendContextModal'
import { uploadFileToAgent, formatSize } from '../../services/fileTransfer'
import type { ChatMessage } from '../../services/agentChat'

interface Attachment {
  id: string
  file: File
  name: string
  size: number
  type: string
  preview?: string // data URL for images
  status: 'pending' | 'uploading' | 'uploaded' | 'error'
  uploadedPath?: string
  error?: string
}

let attachId = 0
function nextAttachId() { return `attach-${Date.now()}-${++attachId}` }

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
    <div className="flex h-full flex-col border-l border-zinc-800 bg-zinc-950/80">
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

// Resolve the agent connection info for the current session
function useAgentConnection(containerId: string) {
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const processes = useProcessStore((s) => s.processes)

  const container = containers.find((c) => c.id === containerId)
  if (container && container.web_port) {
    return { host: 'localhost', port: container.web_port, auth: container.web_auth }
  }
  const local = localAgents.find((a) => a.id === containerId)
  if (local) {
    return { host: local.host, port: local.port, auth: local.authToken }
  }
  // Process agents use chatId = `proc-${slug}`
  const proc = processes.find((p) => `proc-${p.slug}` === containerId)
  if (proc && proc.web_port) {
    return { host: 'localhost', port: proc.web_port, auth: proc.web_auth }
  }
  return null
}

function ChatContent({
  session,
  onSend,
  onCancel,
}: {
  session: { containerId: string; containerName: string; messages: ChatMessage[]; connected: boolean; busy: boolean; statusText: string }
  onSend: (content: string) => void
  onCancel: () => void
}) {
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const conn = useAgentConnection(session.containerId)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages.length, session.statusText])

  // Focus input when panel opens
  useEffect(() => {
    inputRef.current?.focus()
  }, [session.containerId])

  // Upload a file attachment to the agent
  const uploadAttachment = useCallback(async (att: Attachment) => {
    if (!conn) return
    setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'uploading' } : a))
    try {
      const result = await uploadFileToAgent(conn.host, conn.port, conn.auth, att.file)
      setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'uploaded', uploadedPath: result.path } : a))
    } catch (err) {
      setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, status: 'error', error: String(err) } : a))
    }
  }, [conn])

  // Add files as attachments
  const addFiles = useCallback((files: FileList | File[]) => {
    const newAttachments: Attachment[] = Array.from(files).map((file) => {
      const att: Attachment = {
        id: nextAttachId(),
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'pending',
      }
      // Generate preview for images
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e) => {
          setAttachments((prev) => prev.map((a) => a.id === att.id ? { ...a, preview: e.target?.result as string } : a))
        }
        reader.readAsDataURL(file)
      }
      return att
    })
    setAttachments((prev) => [...prev, ...newAttachments])
    // Auto-upload each
    newAttachments.forEach((att) => uploadAttachment(att))
  }, [uploadAttachment])

  // Handle paste (clipboard images/files and text)
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items
    const files: File[] = []
    let hasFiles = false

    for (let i = 0; i < items.length; i++) {
      const item = items[i]
      if (item.kind === 'file') {
        const file = item.getAsFile()
        if (file) {
          files.push(file)
          hasFiles = true
        }
      }
    }

    if (hasFiles) {
      e.preventDefault()
      addFiles(files)
    }
    // If no files, let the default paste handle text
  }, [addFiles])

  // Handle drag-and-drop on the chat area
  const [dragOver, setDragOver] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files)
    }
  }, [addFiles])

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id))
  }, [])

  const handleSend = () => {
    const text = input.trim()
    const uploadedFiles = attachments.filter((a) => a.status === 'uploaded' && a.uploadedPath)
    const hasContent = text || uploadedFiles.length > 0
    if (!hasContent || !session.connected) return

    // Build message with file references
    let content = text
    if (uploadedFiles.length > 0) {
      const fileRefs = uploadedFiles.map((a) => `[Attached file: ${a.name} → ${a.uploadedPath}]`).join('\n')
      content = content ? `${content}\n\n${fileRefs}` : fileRefs
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

  const pasteFromClipboard = useCallback(async () => {
    try {
      const items = await navigator.clipboard.read()
      const files: File[] = []
      for (const item of items) {
        for (const type of item.types) {
          if (type.startsWith('image/') || type === 'application/octet-stream') {
            const blob = await item.getType(type)
            const ext = type.split('/')[1] || 'png'
            const file = new File([blob], `clipboard-${Date.now()}.${ext}`, { type })
            files.push(file)
          }
        }
      }
      if (files.length > 0) {
        addFiles(files)
      } else {
        // Fallback: paste as text
        const text = await navigator.clipboard.readText()
        if (text) {
          setInput((prev) => prev + text)
          inputRef.current?.focus()
        }
      }
    } catch {
      // Fallback to text
      try {
        const text = await navigator.clipboard.readText()
        if (text) {
          setInput((prev) => prev + text)
          inputRef.current?.focus()
        }
      } catch { /* clipboard not available */ }
    }
  }, [addFiles])

  // Build visible messages: keep user + assistant, only last 3 tool/system
  const visibleMessages = (() => {
    const all = session.messages.filter((m) => m.role !== 'tool' || m.tool_name)
    const result: ChatMessage[] = []
    let toolCount = 0
    for (let i = all.length - 1; i >= 0; i--) {
      const m = all[i]
      if (m.role === 'user' || m.role === 'assistant') {
        toolCount = 0
        result.unshift(m)
      } else {
        toolCount++
        if (toolCount <= 3) {
          result.unshift(m)
        }
      }
    }
    return result
  })()

  const pendingUploads = attachments.filter((a) => a.status === 'uploading').length
  const hasErrors = attachments.some((a) => a.status === 'error')

  return (
    <div
      className={`flex flex-1 flex-col overflow-hidden ${dragOver ? 'ring-2 ring-inset ring-violet-500/50' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {dragOver && (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center bg-zinc-950/60">
          <div className="rounded-xl border-2 border-dashed border-violet-500/50 bg-zinc-900/90 px-8 py-6 text-center">
            <Paperclip className="mx-auto h-8 w-8 text-violet-400" />
            <p className="mt-2 text-sm font-medium text-violet-300">Drop files to attach</p>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {visibleMessages.length === 0 && session.connected && (
          <div className="mt-12 text-center">
            <MessageSquare className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">Send a message to start chatting with this agent.</p>
            <p className="mt-1 text-xs text-zinc-600">You can drag & drop files or paste images from clipboard.</p>
          </div>
        )}

        {!session.connected && (
          <div className="mt-12 text-center">
            <AlertCircle className="mx-auto h-8 w-8 text-zinc-700" />
            <p className="mt-2 text-sm text-zinc-500">Connecting to agent...</p>
          </div>
        )}

        {visibleMessages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} sourceName={session.containerName} agentId={session.containerId} />
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

      {/* Attachments strip */}
      {attachments.length > 0 && (
        <div className="border-t border-zinc-800/50 px-3 py-2">
          <div className="flex flex-wrap gap-2">
            {attachments.map((att) => (
              <AttachmentChip key={att.id} attachment={att} onRemove={removeAttachment} />
            ))}
          </div>
          {hasErrors && (
            <p className="mt-1 text-[10px] text-red-400">Some files failed to upload. Remove and retry.</p>
          )}
        </div>
      )}

      {/* Input */}
      <div className="border-t border-zinc-800 p-3">
        <div className="flex items-end gap-2">
          {/* Attach button */}
          <div className="flex gap-0.5">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!session.connected}
              className="flex h-[38px] items-center rounded-lg px-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Attach file"
            >
              <Paperclip className="h-4 w-4" />
            </button>
            <button
              onClick={pasteFromClipboard}
              disabled={!session.connected}
              className="flex h-[38px] items-center rounded-lg px-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Paste from clipboard"
            >
              <Clipboard className="h-4 w-4" />
            </button>
          </div>
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
            placeholder={session.connected ? 'Message, paste image, or drop files...' : 'Connecting...'}
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
              disabled={(!input.trim() && attachments.filter((a) => a.status === 'uploaded').length === 0) || !session.connected || pendingUploads > 0}
              className="flex h-[38px] items-center gap-1.5 rounded-lg bg-violet-600 px-3 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
              title={pendingUploads > 0 ? 'Waiting for uploads...' : 'Send'}
            >
              {pendingUploads > 0 ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function AttachmentChip({ attachment, onRemove }: { attachment: Attachment; onRemove: (id: string) => void }) {
  const isImage = attachment.type.startsWith('image/')

  return (
    <div
      className={`group relative flex items-center gap-1.5 rounded-lg border px-2 py-1.5 text-xs ${
        attachment.status === 'error'
          ? 'border-red-800 bg-red-950/30 text-red-400'
          : attachment.status === 'uploading'
          ? 'border-violet-800/50 bg-violet-950/20 text-violet-300'
          : attachment.status === 'uploaded'
          ? 'border-zinc-700 bg-zinc-800/50 text-zinc-300'
          : 'border-zinc-700 bg-zinc-800/50 text-zinc-400'
      }`}
    >
      {/* Preview or icon */}
      {attachment.preview ? (
        <img src={attachment.preview} alt="" className="h-6 w-6 rounded object-cover" />
      ) : isImage ? (
        <ImageIcon className="h-3.5 w-3.5 shrink-0" />
      ) : (
        <FileIcon className="h-3.5 w-3.5 shrink-0" />
      )}

      <span className="max-w-[120px] truncate">{attachment.name}</span>
      <span className="text-zinc-600">{formatSize(attachment.size)}</span>

      {attachment.status === 'uploading' && <Loader2 className="h-3 w-3 animate-spin text-violet-400" />}

      {/* Remove button */}
      <button
        onClick={() => onRemove(attachment.id)}
        className="ml-0.5 rounded p-0.5 text-zinc-600 hover:text-zinc-300"
      >
        <XCircle className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

function MessageBubble({ message, sourceName, agentId }: { message: ChatMessage; sourceName?: string; agentId?: string }) {
  const [toolExpanded, setToolExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showForward, setShowForward] = useState(false)

  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const { pin, isPinned } = usePinnedStore()
  const addClipEntry = useClipboardStore((s) => s.addEntry)
  const pinned = agentId ? isPinned(agentId, message.content) : false
  const openChat = useChatStore((s) => s.openChat)
  const sendMessageToAgent = useChatStore((s) => s.sendMessage)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [message.content])

  const forwardTargets = useMemo(() => {
    const targets: { id: string; name: string; host: string; port: number; auth: string }[] = []
    for (const c of containers) {
      if (c.status === 'running' && c.web_port) {
        targets.push({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port, auth: c.web_auth || '' })
      }
    }
    for (const a of localAgents) {
      if (a.status === 'online') {
        targets.push({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken || '' })
      }
    }
    return targets
  }, [containers, localAgents])

  const handleForward = useCallback(async (target: typeof forwardTargets[0]) => {
    const src = sourceName || 'another agent'
    const role = message.role === 'user' ? 'User' : 'Assistant'
    // Look up forwarding task from target agent's card config
    const container = containers.find((c) => c.id === target.id)
    const localAgent = localAgents.find((a) => a.id === target.id)
    const fwdTask = (container ? useContainerStore.getState().getForwardingTask(target.id) : localAgent?.forwardingTask) || 'Review the context above and provide your analysis.'
    const composed = `--- Context from "${src}" ---\n\n**${role}:**\n${message.content}\n\n--- End of context ---\n\n${fwdTask}`
    openChat(target.id, target.name, target.host, target.port, target.auth)
    await new Promise((r) => setTimeout(r, 1000))
    sendMessageToAgent(target.id, composed)
    setShowForward(false)
  }, [message, sourceName, containers, localAgents, openChat, sendMessageToAgent])

  const handlePin = useCallback(() => {
    if (!agentId || pinned) return
    pin({ agentId, agentName: sourceName || 'Agent', content: message.content, role: message.role, model: message.model })
  }, [agentId, pinned, pin, sourceName, message])

  const handleClip = useCallback(() => {
    addClipEntry(message.content || '', sourceName || 'Agent')
  }, [addClipEntry, message.content, sourceName])

  // Action buttons (copy + pin + clipboard + forward)
  const actionButtons = (align: 'left' | 'right') => (
    <div className={`flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity ${
      align === 'right' ? 'justify-end' : ''
    }`}>
      <button
        onClick={handleCopy}
        className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-zinc-400"
        title="Copy"
      >
        {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
      </button>
      {agentId && !pinned && (
        <button
          onClick={handlePin}
          className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-amber-400"
          title="Pin message"
        >
          <Pin className="h-3 w-3" />
        </button>
      )}
      {agentId && pinned && (
        <Pin className="h-3 w-3 text-amber-400 mx-0.5" />
      )}
      <button
        onClick={handleClip}
        className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-cyan-400"
        title="Add to shared clipboard"
      >
        <ClipboardList className="h-3 w-3" />
      </button>
      {forwardTargets.length > 0 && (
        <div className="relative">
          <button
            onClick={() => setShowForward(!showForward)}
            className="rounded p-0.5 text-zinc-600 hover:bg-zinc-700/50 hover:text-zinc-400"
            title="Send to another agent"
          >
            <Forward className="h-3 w-3" />
          </button>
          {showForward && (
            <div className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1 z-30 min-w-[160px] rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-xl`}>
              <div className="px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Send to
              </div>
              {forwardTargets.map((t) => (
                <button
                  key={t.id}
                  onClick={() => handleForward(t)}
                  className="flex w-full items-center gap-2 px-2 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  <Send className="h-3 w-3 text-zinc-500" />
                  {t.name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )

  if (message.role === 'user') {
    return (
      <div className="group mb-3 flex flex-col items-end gap-0.5">
        <div className="max-w-[85%] rounded-xl rounded-br-sm bg-violet-600/20 px-3.5 py-2.5">
          <div className="fd-markdown text-sm text-zinc-200">
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </div>
          <span className="mt-1 block text-right text-[10px] text-zinc-500">
            {formatTime(message.timestamp)}
          </span>
        </div>
        {actionButtons('right')}
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
          {message.peer_name && (
            <span className="font-medium text-sky-600 dark:text-sky-400">{message.peer_name}</span>
          )}
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
    // Approval request with approve/deny buttons
    if (message.approval_request_id && !message.approval_resolved) {
      const respondToApproval = useChatStore.getState().respondToApproval
      return (
        <div className="mb-3 flex justify-center">
          <div className="max-w-[85%] rounded-lg border border-amber-600/30 dark:border-amber-500/30 bg-amber-50 dark:bg-amber-500/10 px-4 py-3 shadow-sm">
            <p className="whitespace-pre-wrap text-xs text-amber-900 dark:text-amber-200 mb-2.5">{message.content}</p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => agentId && respondToApproval(agentId, message.approval_request_id!, true)}
                className="rounded-md bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-700 dark:bg-emerald-600/80 dark:hover:bg-emerald-600"
              >
                Approve
              </button>
              <button
                onClick={() => agentId && respondToApproval(agentId, message.approval_request_id!, false)}
                className="rounded-md bg-zinc-200 px-3 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-600"
              >
                Deny
              </button>
            </div>
          </div>
        </div>
      )
    }
    // Resolved approval
    if (message.approval_request_id && message.approval_resolved) {
      return (
        <div className="mb-3 flex justify-center">
          <div className="max-w-[85%] rounded-lg bg-zinc-100 dark:bg-zinc-800/40 px-3 py-2">
            <p className="whitespace-pre-wrap text-xs text-zinc-500 dark:text-zinc-500">{message.content}</p>
            <span className="text-[10px] text-zinc-400 dark:text-zinc-600 italic">Responded</span>
          </div>
        </div>
      )
    }
    return (
      <div className="mb-3 flex justify-center">
        <div className="max-w-[85%] rounded-lg bg-zinc-100 dark:bg-zinc-800/40 px-3 py-2">
          <p className="whitespace-pre-wrap text-xs text-zinc-600 dark:text-zinc-500">{message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant — strip trailing suggestions/rating prompts
  const cleanContent = stripSuggestions(message.content)

  return (
    <div className="group mb-3 flex flex-col items-start gap-0.5">
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
      {actionButtons('left')}
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
