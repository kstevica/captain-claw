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
  Activity,
  ListChecks,
  Maximize2,
  Minimize2,
  Play,
  Trash2,
  AlertTriangle,
  CircleDot,
  HelpCircle,
  Wand2,
  Workflow,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { useChatStore, PLAN_LEVELS, LANE_MAIN, LANES, laneKey, type PlanLevel, type NextStepOption, type QueuedMessage } from '../../stores/chatStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { usePinnedStore } from '../../stores/pinnedStore'
import { useClipboardStore } from '../../stores/clipboardStore'
import { useTraceStore, selectSpanCount } from '../../stores/traceStore'
import { SendContextModal } from './SendContextModal'
import { FlowSelectorModal } from './FlowSelectorModal'
import { PlanCard } from './PlanCard'
import TraceTimeline from '../observability/TraceTimeline'
import { uploadFileToAgent, formatSize } from '../../services/fileTransfer'
import { AgentFilesPanel } from './AgentFilesPanel'
import { AgentDatastorePanel } from './AgentDatastorePanel'
import { QueuePlannerModal } from './QueuePlannerModal'
import type { ChatMessage, TokenUsage } from '../../services/agentChat'

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

// Build the outgoing message text, appending references to any uploaded files
// so the agent can locate them. Images use the "[Attached image: …]" marker
// that routes them to the image_vision tool (describe) / image_ocr (read text)
// instead of `read` — and NOT the `cv` (OpenCV pixel-ops) tool.
function appendAttachmentRefs(text: string, uploadedFiles: Attachment[]): string {
  if (uploadedFiles.length === 0) return text
  const hasImage = uploadedFiles.some((a) => a.type.startsWith('image/'))
  const fileRefs = uploadedFiles.map((a) =>
    a.type.startsWith('image/')
      ? `[Attached image: ${a.uploadedPath}]`
      : `[Attached file: ${a.name} → ${a.uploadedPath}]`,
  ).join('\n')
  const hint = hasImage
    ? '\n(To view/understand the image(s), call image_vision (to describe or answer questions) or image_ocr (to read text) with the path. Do NOT use read on an image, and do NOT use the "cv" tool for this — "cv" does pixel ops only (measure/diff/detect regions) and cannot read text or say what an image shows.)'
    : ''
  return text ? `${text}\n\n${fileRefs}${hint}` : `${fileRefs}${hint}`
}

// Shared file-attachment state + handlers for the chat and queue inputs.
// Files upload immediately on add; callers read `attachments` (filtering to
// status==='uploaded') and bake refs into the outgoing text via
// appendAttachmentRefs.
function useFileAttachments(conn: ReturnType<typeof useAgentConnection>) {
  const [attachments, setAttachments] = useState<Attachment[]>([])

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
    newAttachments.forEach((att) => uploadAttachment(att))
  }, [uploadAttachment])

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id))
  }, [])

  const clearAttachments = useCallback(() => setAttachments([]), [])

  // Paste event on a textarea. Returns true if clipboard files were extracted
  // (caller should NOT also handle text); false → let the default text paste run.
  const handlePasteEvent = useCallback((e: React.ClipboardEvent): boolean => {
    const items = e.clipboardData.items
    const files: File[] = []
    for (let i = 0; i < items.length; i++) {
      const item = items[i]
      if (item.kind === 'file') {
        const file = item.getAsFile()
        if (file) files.push(file)
      }
    }
    if (files.length > 0) {
      e.preventDefault()
      addFiles(files)
      return true
    }
    return false
  }, [addFiles])

  // Clipboard-button paste via the async Clipboard API. Files → addFiles;
  // otherwise the text is handed to onText for the caller to insert.
  const pasteFromClipboard = useCallback(async (onText: (text: string) => void) => {
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
        const text = await navigator.clipboard.readText()
        if (text) onText(text)
      }
    } catch {
      // Fallback to text
      try {
        const text = await navigator.clipboard.readText()
        if (text) onText(text)
      } catch { /* clipboard not available */ }
    }
  }, [addFiles])

  return { attachments, addFiles, removeAttachment, clearAttachments, handlePasteEvent, pasteFromClipboard }
}

export function ChatPanel() {
  const {
    sessions,
    activeChatId,
    chatOpen,
    chatFullscreen,
    closeChat,
    switchChat,
    disconnectChat,
    sendMessage,
    cancelTask,
    setPlanningEnabled,
    setPlanLevel,
    toggleChatFullscreen,
  } = useChatStore()
  const localAgents = useLocalAgentStore((s) => s.agents)
  const containers = useContainerStore((s) => s.containers)
  const processes = useProcessStore((s) => s.processes)
  const [showSendContext, setShowSendContext] = useState(false)
  const [showFlows, setShowFlows] = useState(false)
  const [showTracePanel, setShowTracePanel] = useState(false)
  const [planLevelMenuOpen, setPlanLevelMenuOpen] = useState(false)
  const planLevelMenuRef = useRef<HTMLDivElement>(null)

  // Click-away handler for the plan-level dropdown.
  useEffect(() => {
    if (!planLevelMenuOpen) return
    function onDocClick(e: MouseEvent) {
      if (planLevelMenuRef.current && !planLevelMenuRef.current.contains(e.target as Node)) {
        setPlanLevelMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [planLevelMenuOpen])
  const activeId = activeChatId || ''
  const traceSpanCount = useTraceStore((s) => selectSpanCount(s, activeId))

  const session = activeChatId ? sessions.get(activeChatId) : null

  if (!chatOpen || !session) return null

  // One tab per AGENT. A lane is a context inside an agent, not another
  // agent, so lane sessions don't get their own top-level tab.
  const chatTabs = Array.from(sessions.values()).filter((s) => s.lane === LANE_MAIN)

  // Build target list for context transfer (all reachable agents except the
  // current one). Compare against the AGENT id — `activeChatId` is a lane key,
  // so on lane B it would otherwise fail to exclude this very agent.
  const activeAgentId = session.containerId
  const targets = [
    ...containers
      .filter((c) => c.status === 'running' && c.web_port && c.id !== activeAgentId)
      .map((c) => ({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port!, auth: c.web_auth })),
    ...processes
      .filter((p) => p.status === 'running' && p.web_port && `proc-${p.slug}` !== activeAgentId)
      .map((p) => ({ id: `proc-${p.slug}`, name: p.name, host: 'localhost', port: p.web_port, auth: p.web_auth })),
    ...localAgents
      .filter((a) => a.status === 'online' && a.id !== activeAgentId)
      .map((a) => ({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken })),
  ]

  return (
    <div className="flex h-full flex-col border-l border-zinc-800 bg-zinc-950/80">
      {/* Tab bar */}
      <div className="flex items-center border-b border-zinc-800 bg-zinc-900/60">
        <div className="flex flex-1 items-center gap-0.5 overflow-x-auto px-1 py-1">
          {chatTabs.map((s) => (
            <button
              key={s.key}
              onClick={() => switchChat(s.key)}
              className={`group flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                s.key === activeChatId
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${s.connected ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
              <span className="max-w-[100px] truncate">{s.containerName}</span>
              <button
                onClick={(e) => { e.stopPropagation(); disconnectChat(s.key) }}
                className="ml-0.5 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-zinc-300 group-hover:opacity-100"
              >
                <X className="h-3 w-3" />
              </button>
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowFlows(true)}
          title="Flows — enable/disable and start a flow"
          className="mr-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <Workflow className="h-4 w-4" />
        </button>
        <button
          onClick={() => setShowSendContext(true)}
          title="Send context to another agent"
          className="mr-1 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
        >
          <Forward className="h-4 w-4" />
        </button>
        {/* Combined plan toggle + level selector. The button shows the current
            state ("OFF" when planning auto-routing is disabled, otherwise the
            level abbreviation) and opens a 5-row dropdown: Off + the four
            enrichment levels. Picking a level enables planning AND sets the
            level in one click; picking Off disables auto-routing while
            preserving the previously-chosen level. */}
        <div className="relative mr-1" ref={planLevelMenuRef}>
          <button
            onClick={() => setPlanLevelMenuOpen((o) => !o)}
            title={
              session.planningEnabled
                ? `Planning ON (${session.planLevel}). Click to change level or turn off.`
                : 'Planning OFF. Click to choose an enrichment level and turn on.'
            }
            className={`relative flex items-center gap-1 rounded px-1.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-colors ${
              session.planningEnabled
                ? 'bg-violet-500 text-white shadow-sm shadow-violet-500/40 ring-1 ring-violet-300/60 hover:bg-violet-400'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            <ListChecks className="h-4 w-4" />
            <span className="leading-none">
              {session.planningEnabled
                ? (session.planLevel || 'plain').slice(0, 3)
                : 'OFF'}
            </span>
            <ChevronDown className="h-3 w-3 opacity-70" />
            {session.planState && session.planState.status === 'running' && (
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulse rounded-full bg-emerald-400 ring-2 ring-zinc-900" />
            )}
          </button>
          {planLevelMenuOpen && (
            <div className="absolute right-0 top-full z-50 mt-1 w-60 overflow-hidden rounded-md border border-zinc-700 bg-zinc-900 shadow-lg">
              <button
                onClick={() => {
                  // Always send /planning off, even if FD thinks it's already
                  // off — same staleness concern as the level picker below.
                  setPlanningEnabled(session.key, false)
                  setPlanLevelMenuOpen(false)
                }}
                className={`flex w-full flex-col items-start gap-0.5 border-b border-zinc-800 px-3 py-2 text-left text-xs transition-colors ${
                  !session.planningEnabled
                    ? 'bg-zinc-800/80 text-zinc-200'
                    : 'text-zinc-300 hover:bg-zinc-800'
                }`}
              >
                <span className="font-semibold uppercase tracking-wide">Off</span>
                <span className="text-[10px] text-zinc-400">Disable plan auto-routing</span>
              </button>
              {PLAN_LEVELS.map((lvl) => {
                const desc: Record<PlanLevel, string> = {
                  plain: 'No enrichment (default)',
                  enriched: '+ latest reflection',
                  insightful: '+ top user insights',
                  complete: '+ persona-aware planner',
                }
                const active = session.planningEnabled && session.planLevel === lvl
                return (
                  <button
                    key={lvl}
                    onClick={() => {
                      // Always send both commands unconditionally. Skipping
                      // them when the FD mirror "looks right" lets a stale
                      // mirror (e.g., after a ws drop or a missed command_result)
                      // hide the fact that the agent's plan_mode_auto is still
                      // off — the user picks a level, the UI shows it active,
                      // but auto-routing never engages and they have to type
                      // /plan manually. Both commands are idempotent so the
                      // double-send when state already matches is harmless.
                      // Set level first so the /planning on confirmation
                      // already reflects the new level.
                      setPlanLevel(session.key, lvl)
                      setPlanningEnabled(session.key, true)
                      setPlanLevelMenuOpen(false)
                    }}
                    className={`flex w-full flex-col items-start gap-0.5 border-b border-zinc-800 px-3 py-2 text-left text-xs last:border-b-0 transition-colors ${
                      active
                        ? 'bg-violet-500/20 text-violet-200'
                        : 'text-zinc-300 hover:bg-zinc-800'
                    }`}
                  >
                    <span className="font-semibold uppercase tracking-wide">{lvl}</span>
                    <span className="text-[10px] text-zinc-400">{desc[lvl]}</span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
        <button
          onClick={() => setShowTracePanel(!showTracePanel)}
          title={showTracePanel ? 'Hide traces' : 'Show orchestrator traces'}
          className={`relative mr-1 rounded p-1 transition-colors ${
            showTracePanel
              ? 'bg-violet-600/20 text-violet-400'
              : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
          }`}
        >
          <Activity className="h-4 w-4" />
          {traceSpanCount > 0 && (
            <span className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-[14px] items-center justify-center rounded-full bg-violet-600 px-0.5 text-[8px] font-bold text-white">
              {traceSpanCount > 99 ? '99+' : traceSpanCount}
            </span>
          )}
        </button>
        <button
          onClick={toggleChatFullscreen}
          title={chatFullscreen ? 'Exit fullscreen' : 'Fullscreen chat (hide other panels)'}
          className={`mr-1 rounded p-1 transition-colors ${
            chatFullscreen
              ? 'bg-violet-600/20 text-violet-400'
              : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
          }`}
        >
          {chatFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
        </button>
        <button onClick={closeChat} className="mr-2 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <Minus className="h-4 w-4" />
        </button>
      </div>

      {/* Trace panel (replaces chat when active) */}
      {/* Lanes: three parallel contexts on this one agent. */}
      <LaneStrip
        agentId={session.containerId}
        activeLane={session.lane}
        agentName={session.containerName}
      />

      {showTracePanel ? (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex items-center gap-2 border-b border-zinc-800/50 bg-zinc-900/30 px-3 py-2">
            <Activity className="h-3.5 w-3.5 text-violet-400" />
            <span className="text-xs font-medium text-zinc-300">Orchestrator Traces</span>
            {traceSpanCount > 0 && (
              <span className="rounded-full bg-violet-600/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-400">
                {traceSpanCount} spans
              </span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto">
            <TraceTimeline containerId={activeId} />
          </div>
        </div>
      ) : (
        /* Chat content (optionally with queue panel on the left in fullscreen) */
        <div className="flex flex-1 overflow-hidden">
          {chatFullscreen && (
            <QueueSidebar sessionKey={session.key} agentId={session.containerId} />
          )}
          <ChatContent
            session={session}
            sessionKey={session.key}
            onSend={(content) => sendMessage(session.key, content)}
            onCancel={() => cancelTask(session.key)}
          />
        </div>
      )}

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

      {/* Flow selector modal */}
      {showFlows && (
        <FlowSelectorModal
          containerId={session.containerId}
          onClose={() => setShowFlows(false)}
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
  sessionKey,
  onSend,
  onCancel,
}: {
  session: { containerId: string; containerName: string; messages: ChatMessage[]; connected: boolean; busy: boolean; statusText: string; nextStepOptions?: NextStepOption[]; liveTurnUsage?: TokenUsage | null }
  /** This lane's store key — for anything that reads or writes session state.
   *  The agent id lives on `session.containerId`. */
  sessionKey: string
  onSend: (content: string) => void
  onCancel: () => void
}) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const conn = useAgentConnection(session.containerId)
  const { attachments, addFiles, removeAttachment, clearAttachments, handlePasteEvent, pasteFromClipboard } = useFileAttachments(conn)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages.length, session.statusText])

  // Focus input when panel opens
  useEffect(() => {
    inputRef.current?.focus()
  }, [session.containerId])

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

  const handleSend = () => {
    const text = input.trim()
    const uploadedFiles = attachments.filter((a) => a.status === 'uploaded' && a.uploadedPath)
    const hasContent = text || uploadedFiles.length > 0
    if (!hasContent || !session.connected) return

    const content = appendAttachmentRefs(text, uploadedFiles)
    onSend(content)
    setInput('')
    clearAttachments()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handlePasteText = useCallback((text: string) => {
    setInput((prev) => prev + text)
    inputRef.current?.focus()
  }, [])

  // Keep the full transcript — every tool call is shown. The intermediary
  // activity (tools + narration) is grouped into a collapsible panel that
  // streams while the turn runs and collapses to a tools-history summary once
  // it finishes, so showing the whole process doesn't clutter the conversation.
  const visibleMessages = session.messages.filter((m) => m.role !== 'tool' || m.tool_name)

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

      {/* Plan monitor — pinned above the scroll area so it stays visible
          while messages flow underneath during /plan-execute. */}
      <PlanCard containerId={sessionKey} />

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-3">
        <div className="px-4">
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

        {(() => {
          const groups = groupActivity(visibleMessages)
          return groups.map((grp, gi) =>
            grp.kind === 'activity' ? (
              <ActivityGroup
                key={`act-${grp.items[0].id}`}
                items={grp.items}
                live={gi === groups.length - 1 && session.busy}
                liveTurnUsage={gi === groups.length - 1 && session.busy ? session.liveTurnUsage ?? null : null}
              />
            ) : (
              <MessageBubble key={grp.items[0].id} message={grp.items[0]} sourceName={session.containerName} agentId={session.containerId} />
            )
          )
        })()}

        {/* Busy indicator */}
        {session.busy && (
          <div className="mb-3 flex items-center gap-2 text-xs text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{session.statusText || 'Working...'}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Suggested next steps */}
      {session.nextStepOptions && session.nextStepOptions.length > 0 && (
        <NextStepsBar
          options={session.nextStepOptions}
          disabled={!session.connected || session.busy}
          onPick={(action) => onSend(action)}
        />
      )}

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
              onClick={() => pasteFromClipboard(handlePasteText)}
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
            onPaste={handlePasteEvent}
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

function NextStepsBar({
  options,
  disabled,
  onPick,
}: {
  options: NextStepOption[]
  disabled: boolean
  onPick: (action: string) => void
}) {
  return (
    <div className="border-t border-zinc-200 px-4 py-2 dark:border-zinc-800/50">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-500">Suggested next steps</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt, i) => (
          <button
            key={`${i}-${opt.label}`}
            type="button"
            disabled={disabled}
            title={opt.description || opt.action}
            onClick={() => onPick(opt.action)}
            className="rounded-full border border-violet-300 bg-violet-50 px-3 py-1 text-xs font-medium text-violet-700 hover:border-violet-400 hover:bg-violet-100 hover:text-violet-800 dark:border-violet-500/30 dark:bg-violet-500/10 dark:font-normal dark:text-violet-200 dark:hover:bg-violet-500/20 dark:hover:text-violet-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// Drag-to-resize with localStorage persistence. axis 'x' → width (col-resize),
// axis 'y' → height (row-resize). The handle is positioned so a positive drag
// (right / down) grows the tracked size.
function usePersistedSize(key: string, def: number, min: number, max: number, axis: 'x' | 'y') {
  const [size, setSize] = useState<number>(() => {
    const v = Number(localStorage.getItem(key))
    return v >= min && v <= max ? v : def
  })
  const onResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    const startPos = axis === 'x' ? e.clientX : e.clientY
    const startSize = size
    document.body.style.cursor = axis === 'x' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: MouseEvent) => {
      const pos = axis === 'x' ? ev.clientX : ev.clientY
      setSize(Math.min(max, Math.max(min, startSize + (pos - startPos))))
    }
    const onUp = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      setSize((s) => { localStorage.setItem(key, String(Math.round(s))); return s })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }
  return { size, onResizeStart }
}

// The chat's left sidebar: an agent files view on top, the run queue on the
// bottom, with a draggable divider between them and a draggable right edge for
// the whole column. Both sizes persist to localStorage.

// ── Lanes ────────────────────────────────────────────────────────────
//
// Three parallel contexts on ONE agent (docs/queue-lanes-plan.md). Lane A is
// the agent's existing context, so switching to A shows the conversation that
// was always there; B and C are additional rooms that run at the same time.
function LaneStrip({ agentId, activeLane, agentName }: {
  agentId: string
  activeLane: string
  agentName: string
}) {
  const sessions = useChatStore((s) => s.sessions)
  const setActiveLane = useChatStore((s) => s.setActiveLane)

  return (
    <div className="flex items-center gap-1 border-b border-zinc-800 bg-zinc-900/40 px-2 py-1">
      {LANES.map((lane) => {
        const s = sessions.get(laneKey(agentId, lane))
        const isActive = lane === activeLane
        const pending = s ? s.queue.filter((q) => q.status === 'pending').length : 0
        const awaiting = s ? s.queue.some((q) => q.awaitingAnswer) : false
        return (
          <button
            key={lane}
            onClick={() => setActiveLane(agentId, lane)}
            title={
              !s ? `Lane ${lane} — not started`
                : awaiting ? `Lane ${lane} — waiting on an answer`
                : s.busy ? `Lane ${lane} — running`
                : `Lane ${lane} — idle`
            }
            className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
              isActive
                ? 'bg-violet-600/20 text-violet-700 dark:text-violet-200'
                : 'text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300'
            }`}
          >
            {/* State: running (pulsing) · waiting · idle · never opened */}
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                !s ? 'bg-zinc-700'
                  : awaiting ? 'bg-amber-400'
                  : s.busy ? 'animate-pulse bg-violet-400'
                  : 'bg-emerald-400'
              }`}
            />
            <span>{lane} - {agentName}</span>
            {pending > 0 && (
              <span className="rounded-full bg-zinc-800 px-1 text-[9px] text-zinc-400">{pending}</span>
            )}
            {/* Something finished here while you were looking elsewhere. */}
            {s?.unread && !isActive && (
              <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
            )}
          </button>
        )
      })}
    </div>
  )
}

function QueueSidebar({ sessionKey, agentId }: { sessionKey: string; agentId: string }) {
  const width = usePersistedSize('fd:queue-sidebar-width', 288, 220, 640, 'x')
  const filesH = usePersistedSize('fd:queue-sidebar-files-height', 260, 96, 900, 'y')
  const dsH = usePersistedSize('fd:queue-sidebar-datastore-height', 200, 96, 900, 'y')

  return (
    <aside
      className="relative flex shrink-0 flex-col border-r border-zinc-800 bg-zinc-950/40"
      style={{ width: width.size }}
    >
      {/* Top: files */}
      <div className="min-h-0 shrink-0 overflow-hidden" style={{ height: filesH.size }}>
        <AgentFilesPanel containerId={agentId} />
      </div>
      {/* Vertical divider (drag to resize files vs datastore) */}
      <div
        onMouseDown={filesH.onResizeStart}
        title="Drag to resize"
        className="h-1 shrink-0 cursor-row-resize border-y border-zinc-800 bg-zinc-900 transition-colors hover:bg-violet-500/40"
      />
      {/* Middle: datastore tables */}
      <div className="min-h-0 shrink-0 overflow-hidden" style={{ height: dsH.size }}>
        <AgentDatastorePanel containerId={agentId} />
      </div>
      {/* Vertical divider (drag to resize datastore vs queue) */}
      <div
        onMouseDown={dsH.onResizeStart}
        title="Drag to resize"
        className="h-1 shrink-0 cursor-row-resize border-y border-zinc-800 bg-zinc-900 transition-colors hover:bg-violet-500/40"
      />
      {/* Bottom: queue */}
      <div className="min-h-0 flex-1">
        <QueuePanel sessionKey={sessionKey} agentId={agentId} />
      </div>
      {/* Right edge (drag to resize the whole column) */}
      <div
        onMouseDown={width.onResizeStart}
        title="Drag to resize"
        className="absolute right-0 top-0 z-20 h-full w-1.5 cursor-col-resize transition-colors hover:bg-violet-500/30 active:bg-violet-500/40"
      />
    </aside>
  )
}

function QueuePanel({ sessionKey, agentId }: { sessionKey: string; agentId: string }) {
  // Queue state is per lane; uploads go to the agent that owns every lane.
  const containerId = sessionKey
  const session = useChatStore((s) => s.sessions.get(containerId))
  const enqueue = useChatStore((s) => s.enqueueQueueMessage)
  const removeItem = useChatStore((s) => s.removeQueueItem)
  const markDone = useChatStore((s) => s.markQueueItemDone)
  const toggleAuto = useChatStore((s) => s.toggleQueueAutoMode)
  const clearQueue = useChatStore((s) => s.clearQueue)
  const clearFinished = useChatStore((s) => s.clearQueueFinished)
  const [draft, setDraft] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [showPlanner, setShowPlanner] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const conn = useAgentConnection(agentId)
  const { attachments, addFiles, removeAttachment, clearAttachments, handlePasteEvent, pasteFromClipboard } = useFileAttachments(conn)

  const pasteText = useCallback((text: string) => {
    setDraft((prev) => prev + text)
    taRef.current?.focus()
  }, [])

  // Queued items are long instructions, so the box starts at 5 rows and grows
  // with the draft up to ~40% of the panel before handing over to its own
  // scrollbar — the list above keeps the rest of the height.
  const QUEUE_INPUT_MIN_PX = 92
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const cap = Math.max(QUEUE_INPUT_MIN_PX, Math.round((ta.closest('.flex.h-full')?.clientHeight ?? 600) * 0.4))
    ta.style.height = `${Math.min(Math.max(ta.scrollHeight, QUEUE_INPUT_MIN_PX), cap)}px`
  }, [draft])

  if (!session) return null
  const queue = session.queue
  const auto = session.queueAutoMode
  const pendingCount = queue.filter((q) => q.status === 'pending').length
  const dispatchedCount = queue.filter((q) => q.status === 'dispatched').length
  const doneCount = queue.filter((q) => q.status === 'done').length
  const pendingUploads = attachments.filter((a) => a.status === 'uploading').length
  const uploadedFiles = attachments.filter((a) => a.status === 'uploaded' && a.uploadedPath)
  const canAdd = (draft.trim() || uploadedFiles.length > 0) && pendingUploads === 0

  const handleAdd = () => {
    const text = draft.trim()
    if (!text && uploadedFiles.length === 0) return
    if (pendingUploads > 0) return
    enqueue(containerId, appendAttachmentRefs(text, uploadedFiles))
    setDraft('')
    clearAttachments()
    taRef.current?.focus()
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAdd()
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <ListChecks className="h-3.5 w-3.5 text-violet-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Queue</span>
          {queue.length > 0 && (
            <span className="rounded-full bg-violet-500/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-300">
              {pendingCount + dispatchedCount}/{queue.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
        <button
          onClick={() => setShowPlanner(true)}
          title="Plan tasks — turn one description into a batched queue"
          className="flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400 transition-colors hover:border-violet-500/40 hover:text-violet-300"
        >
          <Wand2 className="h-3 w-3" />
          Plan
        </button>
        <button
          onClick={() => toggleAuto(containerId)}
          title={auto ? 'Auto-progress: ON — items marked done when agent replies' : 'Auto-progress: OFF — mark items done manually'}
          className={`rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider transition-colors ${
            auto
              ? 'border-emerald-500/40 bg-emerald-500/20 text-emerald-700 dark:text-emerald-200'
              : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600'
          }`}
        >
          {auto ? 'Auto' : 'Manual'}
        </button>
        </div>
      </div>

      {showPlanner && conn && (
        <QueuePlannerModal
          agentId={agentId}
          agentName={session.containerName}
          host={conn.host}
          port={conn.port}
          auth={conn.auth}
          onClose={() => setShowPlanner(false)}
        />
      )}

      {/* Items — min-h-0 so this is the column's scroll region: cards keep
          their full height and the list scrolls past them. */}
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {queue.length === 0 ? (
          <p className="px-2 py-6 text-center text-[11px] text-zinc-500">
            Add a message below to queue it. Items dispatch one at a time after the previous finishes.
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {queue.map((item) => (
              <QueueItemRow
                key={item.id}
                item={item}
                onRemove={() => removeItem(containerId, item.id)}
                onMarkDone={() => markDone(containerId, item.id)}
                autoMode={auto}
              />
            ))}
          </ul>
        )}
      </div>

      {/* Status + clear */}
      {(doneCount > 0 || queue.length > 0) && (
        <div className="flex items-center justify-between border-t border-zinc-800 px-3 py-1.5 text-[10px] text-zinc-500">
          <span>{doneCount} done · {pendingCount} pending</span>
          <div className="flex items-center gap-0.5">
            {doneCount > 0 && (
              <button
                onClick={() => clearFinished(containerId)}
                className="rounded px-1.5 py-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                title={`Remove ${doneCount} finished item${doneCount === 1 ? '' : 's'} — pending and in-flight stay`}
              >
                Clear finished
              </button>
            )}
            {pendingCount > 0 && (
              <button
                onClick={() => clearQueue(containerId)}
                className="rounded px-1.5 py-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                title="Clear pending items"
              >
                Clear pending
              </button>
            )}
          </div>
        </div>
      )}

      {/* Input */}
      <div
        className={`border-t border-zinc-800 p-2 ${dragOver ? 'bg-violet-500/10 ring-1 ring-inset ring-violet-500/50' : ''}`}
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true) }}
        onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(false) }}
        onDrop={handleDrop}
      >
        {/* Attachment chips */}
        {attachments.length > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1.5">
            {attachments.map((att) => (
              <AttachmentChip key={att.id} attachment={att} onRemove={removeAttachment} />
            ))}
          </div>
        )}
        <textarea
          ref={taRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKey}
          onPaste={handlePasteEvent}
          rows={5}
          placeholder="Queue a message, paste image, or drop files…"
          className="w-full resize-y overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          style={{ minHeight: QUEUE_INPUT_MIN_PX }}
        />
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
        <div className="mt-1.5 flex items-center justify-between">
          <div className="flex items-center gap-0.5">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!conn}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Attach file"
            >
              <Paperclip className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => pasteFromClipboard(pasteText)}
              disabled={!conn}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-40"
              title="Paste from clipboard"
            >
              <Clipboard className="h-3.5 w-3.5" />
            </button>
          </div>
          <button
            onClick={handleAdd}
            disabled={!canAdd}
            className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            title={pendingUploads > 0 ? 'Waiting for uploads…' : 'Add to queue'}
          >
            {pendingUploads > 0 && <Loader2 className="h-3 w-3 animate-spin" />}
            Add
          </button>
        </div>
      </div>
    </div>
  )
}


// ── Queue item run stats ──

/** "14:32:07" on the same day, "Jul 21 14:32:07" otherwise. */
function fmtClock(ms?: number): string {
  if (!ms) return ''
  const d = new Date(ms)
  const time = d.toLocaleTimeString(undefined, { hour12: false })
  const sameDay = new Date().toDateString() === d.toDateString()
  return sameDay
    ? time
    : `${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${time}`
}

/** Elapsed as 42s / 3m 07s / 1h 04m. */
function fmtElapsed(from?: number, to?: number): string {
  if (!from || !to || to < from) return ''
  const secs = Math.round((to - from) / 1000)
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const rest = secs % 60
  if (m < 60) return `${m}m ${String(rest).padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}

function QueueItemRow({
  item,
  onRemove,
  onMarkDone,
  autoMode,
}: {
  item: QueuedMessage
  onRemove: () => void
  onMarkDone: () => void
  autoMode: boolean
}) {
  const isPending = item.status === 'pending'
  const isDispatched = item.status === 'dispatched'
  const isDone = item.status === 'done'

  return (
    <li
      className={`group rounded-md border px-2 py-1.5 text-xs transition-colors ${
        isDispatched
          ? 'border-violet-500/40 bg-violet-500/10'
          : isDone
            ? 'border-zinc-800 bg-zinc-900/40 text-zinc-600'
            : 'border-zinc-800 bg-zinc-900/60 hover:border-zinc-700'
      }`}
    >
      <div className="flex items-start gap-1.5">
        <span
          className="mt-0.5 shrink-0"
          title={item.stuck
            ? `The agent gave up ${item.attempts ?? 0} times — the queue is holding here`
            : item.awaitingAnswer
              ? 'The agent asked a question — answer it in chat, or the queue moves on in 3 min'
              : item.attempts && item.attempts > 1
                ? `Attempt ${item.attempts} — the agent got stuck on the previous try`
                : item.status}
        >
          {isDispatched && item.stuck && <AlertTriangle className="h-3 w-3 text-red-400" />}
          {isDispatched && !item.stuck && item.awaitingAnswer && <HelpCircle className="h-3 w-3 text-amber-400" />}
          {isDispatched && !item.stuck && !item.awaitingAnswer && <Loader2 className="h-3 w-3 animate-spin text-violet-300" />}
          {isPending && <CircleDot className="h-3 w-3 text-zinc-500" />}
          {isDone && <Check className="h-3 w-3 text-emerald-400" />}
        </span>
        <span
          className={`flex-1 whitespace-pre-wrap break-words ${
            isDone ? 'line-through' : ''
          } ${isDispatched ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-300'}`}
        >
          {/* Full text: a queued task is an instruction the user needs to read
              back before it runs. The card grows to fit; the list scrolls. */}
          {item.content}
          {(item.attempts ?? 0) > 1 && (
            <span className={`ml-1.5 rounded px-1 py-0.5 text-[9px] font-medium ${
              item.stuck ? 'bg-red-500/20 text-red-300' : 'bg-amber-500/20 text-amber-300'
            }`}>
              {item.stuck ? `gave up ×${item.attempts}` : `attempt ${item.attempts}`}
            </span>
          )}
        </span>
        <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
          {isDispatched && !autoMode && (
            <button
              onClick={onMarkDone}
              title="Mark done & dispatch next"
              className="rounded p-0.5 text-zinc-500 hover:bg-emerald-500/20 hover:text-emerald-300"
            >
              <Check className="h-3 w-3" />
            </button>
          )}
          {isPending && (
            <button
              onClick={onMarkDone}
              title="Skip (mark done without dispatching)"
              className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Play className="h-3 w-3" />
            </button>
          )}
          <button
            onClick={onRemove}
            title="Remove"
            className="rounded p-0.5 text-zinc-500 hover:bg-red-500/20 hover:text-red-300"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* What the run cost. Only once it has actually started. */}
      {item.dispatchedAt && <QueueItemStats item={item} />}
    </li>
  )
}

function QueueItemStats({ item }: { item: QueuedMessage }) {
  const started = fmtClock(item.dispatchedAt)
  // A finished item is fixed in time; a running one ticks against now.
  const [, force] = useState(0)
  const running = item.status === 'dispatched'
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => force((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [running])

  const ended = item.completedAt
  const elapsed = fmtElapsed(item.dispatchedAt, ended ?? Date.now())
  const u = item.usage
  const cached = (u?.cache_read_input_tokens || 0) + (u?.cache_creation_input_tokens || 0)

  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 border-t border-zinc-800/60 pt-1 pl-[18px] text-[9px] text-zinc-500">
      <span title={`Started ${new Date(item.dispatchedAt!).toLocaleString()}`}>▶ {started}</span>
      {ended ? (
        <span title={`Finished ${new Date(ended).toLocaleString()}`}>■ {fmtClock(ended)}</span>
      ) : null}
      {elapsed && (
        <span className={running ? 'text-violet-400' : 'text-zinc-400'}>{elapsed}</span>
      )}
      {!!item.toolCount && <span title="Tool calls">{item.toolCount} tools</span>}
      {u && (
        <span title={`In ${u.prompt_tokens || 0} · cached ${cached} · out ${u.completion_tokens || 0} tokens`}>
          {fmtTokens(u.prompt_tokens)}↑ {fmtTokens(cached)}⚡ {fmtTokens(u.completion_tokens)}↓
        </span>
      )}
    </div>
  )
}

/** Convert agent-local image paths in message content to URLs proxied through Flight Deck. */
function processImagePaths(content: string, agentHost?: string, agentPort?: number, agentAuth?: string): string {
  if (!agentPort) return content

  const proxyUrl = (filePath: string): string => {
    const params = new URLSearchParams({ path: filePath })
    if (agentAuth) params.set('token', agentAuth)
    return `/fd/agent-file-view/${encodeURIComponent(agentHost || 'localhost')}/${agentPort}?${params}`
  }

  // Rewrite agent-local paths inside markdown image syntax: ![alt](path)
  // Mirrors captain-claw/computer.js: handles file:///, saved/, output/, and absolute /paths.
  // Leaves http(s)://, data:, blob:, and already-proxied URLs alone.
  let out = content.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (match, alt, src) => {
    if (/^(https?:|data:|blob:)/i.test(src)) return match
    if (src.startsWith('/fd/')) return match
    let filePath = src
    if (/^file:\/\/\//i.test(src)) {
      filePath = '/' + src.replace(/^file:\/\/\//i, '')
    } else if (/^(saved\/|output\/|\/)/.test(src)) {
      filePath = src
    } else {
      return match
    }
    return `![${alt}](${proxyUrl(filePath)})`
  })

  // Also handle bare image paths on their own line (existing behavior).
  out = out.replace(
    /^([`*]*)(\/?(?:\/[\w.@: -]+)+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))([`*]*)$/gm,
    (_m, _pre, filePath) => `![](${proxyUrl(filePath)})`,
  )

  return out
}

function MessageBubble({ message, sourceName, agentId }: { message: ChatMessage; sourceName?: string; agentId?: string }) {
  const [toolExpanded, setToolExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showForward, setShowForward] = useState(false)

  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)
  const processes = useProcessStore((s) => s.processes)
  const { pin, isPinned } = usePinnedStore()
  const addClipEntry = useClipboardStore((s) => s.addEntry)
  const pinned = agentId ? isPinned(agentId, message.content) : false
  const openChat = useChatStore((s) => s.openChat)
  const sendMessageToAgent = useChatStore((s) => s.sendMessage)

  // Resolve agent connection for image proxying
  const agentConn = useMemo(() => {
    if (!agentId) return null
    const c = containers.find((x) => x.id === agentId)
    if (c?.web_port) return { host: 'localhost', port: c.web_port, auth: c.web_auth || '' }
    const l = localAgents.find((x) => x.id === agentId)
    if (l) return { host: l.host, port: l.port, auth: l.authToken || '' }
    const p = processes.find((x) => `proc-${x.slug}` === agentId)
    if (p?.web_port) return { host: 'localhost', port: p.web_port, auth: p.web_auth || '' }
    return null
  }, [agentId, containers, localAgents, processes])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }, [message.content])

  const forwardTargets = useMemo(() => {
    const targets: { id: string; name: string; host: string; port: number; auth: string }[] = []
    for (const c of containers) {
      if (c.status === 'running' && c.web_port && c.id !== agentId) {
        targets.push({ id: c.id, name: c.agent_name || c.name, host: 'localhost', port: c.web_port, auth: c.web_auth || '' })
      }
    }
    for (const p of processes) {
      if (p.status === 'running' && p.web_port && `proc-${p.slug}` !== agentId) {
        targets.push({ id: `proc-${p.slug}`, name: p.name, host: 'localhost', port: p.web_port, auth: p.web_auth || '' })
      }
    }
    for (const a of localAgents) {
      if (a.status === 'online' && a.id !== agentId) {
        targets.push({ id: a.id, name: a.name, host: a.host, port: a.port, auth: a.authToken || '' })
      }
    }
    return targets
  }, [containers, processes, localAgents, agentId])

  const handleForward = useCallback(async (target: typeof forwardTargets[0]) => {
    const src = sourceName || 'another agent'
    const role = message.role === 'user' ? 'User' : 'Assistant'
    // Look up forwarding task from target agent's card config
    const container = containers.find((c) => c.id === target.id)
    const process = target.id.startsWith('proc-') ? processes.find((p) => `proc-${p.slug}` === target.id) : undefined
    const localAgent = localAgents.find((a) => a.id === target.id)
    const fwdTask = (container
      ? useContainerStore.getState().getForwardingTask(target.id)
      : process
        ? useProcessStore.getState().getForwardingTask(target.id)
        : localAgent?.forwardingTask
    ) || 'Review the context above and provide your analysis.'
    const composed = `--- Context from "${src}" ---\n\n**${role}:**\n${message.content}\n\n--- End of context ---\n\n${fwdTask}`
    openChat(target.id, target.name, target.host, target.port, target.auth)
    await new Promise((r) => setTimeout(r, 1000))
    sendMessageToAgent(target.id, composed)
    setShowForward(false)
  }, [message, sourceName, containers, processes, localAgents, openChat, sendMessageToAgent])

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
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{processImagePaths(message.content, agentConn?.host, agentConn?.port, agentConn?.auth)}</Markdown>
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
    // Live between-step narration blurb — subtle, in the flow.
    if (message.narration) {
      return (
        <div className="mb-1.5 flex items-start gap-1.5 pl-1 pr-6 text-xs italic text-zinc-500">
          <CircleDot className="mt-0.5 h-3 w-3 shrink-0 text-zinc-600" />
          <div className="fd-markdown min-w-0 flex-1">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
          </div>
        </div>
      )
    }
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
          <div className="max-w-[85%] rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700/50 dark:bg-zinc-800/40">
            <div className="fd-markdown text-xs">
              <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
            </div>
            <span className="text-[10px] text-zinc-400 dark:text-zinc-600 italic">Responded</span>
          </div>
        </div>
      )
    }
    return (
      <div className="mb-3 flex justify-center">
        <div className="max-w-[95%] rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-700/50 dark:bg-zinc-800/40">
          <div className="fd-markdown text-xs">
            <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
          </div>
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
          <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{processImagePaths(cleanContent, agentConn?.host, agentConn?.port, agentConn?.auth)}</Markdown>
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

// ── Intermediary "activity" stream (tool calls + between-step narration) ──
//
// During a turn the agent emits tool calls and short narration blurbs. These
// are grouped into one tinted section that stacks until the turn finishes, so
// they read as a connected activity log rather than loose grey lines. Each
// type gets its own colour: tools (sky), narration (violet).

function isActivityMessage(m: ChatMessage): boolean {
  return m.role === 'tool' || (m.role === 'system' && !!m.narration)
}

function groupActivity(msgs: ChatMessage[]): { kind: 'activity' | 'single'; items: ChatMessage[] }[] {
  const groups: { kind: 'activity' | 'single'; items: ChatMessage[] }[] = []
  for (const m of msgs) {
    if (isActivityMessage(m)) {
      const last = groups[groups.length - 1]
      if (last && last.kind === 'activity') last.items.push(m)
      else groups.push({ kind: 'activity', items: [m] })
    } else {
      groups.push({ kind: 'single', items: [m] })
    }
  }
  return groups
}

function fmtTokens(n?: number): string {
  const v = n || 0
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`
  return String(v)
}

function ActivityGroup({ items, live, liveTurnUsage }: { items: ChatMessage[]; live: boolean; liveTurnUsage?: TokenUsage | null }) {
  // Expanded & streaming while the turn runs; auto-collapses to a tools-history
  // summary once it finishes. The user can still toggle it open afterwards.
  const [collapsed, setCollapsed] = useState(!live)
  useEffect(() => { setCollapsed(!live) }, [live])
  const toolCount = items.filter((m) => m.role === 'tool').length
  // Token usage: live cumulative counter while running, else the frozen value
  // stamped onto one of this group's tool messages when the turn ended.
  const usage: TokenUsage | null = live
    ? (liveTurnUsage ?? null)
    : (items.find((m) => m.usage)?.usage ?? null)
  const hasUsage = !!usage && ((usage.prompt_tokens || 0) > 0 || (usage.completion_tokens || 0) > 0)
  return (
    <div className="mb-3 rounded-lg border border-zinc-800 bg-zinc-900/40 px-2.5 py-2">
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex w-full items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-400"
      >
        <Wrench className="h-3 w-3" />
        Activity{toolCount > 0 && <span className="text-zinc-600">· {toolCount} tool{toolCount !== 1 ? 's' : ''}</span>}
        {hasUsage && usage && (
          <span className="text-zinc-600 normal-case" title="Tokens this turn — input / output / cached">
            · {fmtTokens(usage.prompt_tokens)}↑ {fmtTokens(usage.completion_tokens)}↓
            {(usage.cache_read_input_tokens || 0) > 0 ? ` · ${fmtTokens(usage.cache_read_input_tokens)} cached` : ''}
          </span>
        )}
        <span className="ml-auto">{collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}</span>
      </button>
      {!collapsed && (
        <div className="mt-1.5 space-y-0.5">
          {items.map((m) =>
            m.role === 'tool'
              ? <ActivityToolRow key={m.id} message={m} />
              : <ActivityNarration key={m.id} message={m} />
          )}
        </div>
      )}
    </div>
  )
}

function ActivityToolRow({ message }: { message: ChatMessage }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-xs text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-400"
      >
        <Wrench className="h-3 w-3" />
        {message.peer_name && <span className="font-medium text-sky-600 dark:text-sky-400">{message.peer_name}</span>}
        <span className="font-medium">{message.tool_name}</span>
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>
      {expanded && message.content && (
        <pre className="ml-6 mt-1 max-h-40 overflow-auto rounded-md bg-zinc-900/80 p-2 text-xs text-zinc-400 font-mono">
          {message.content.slice(0, 2000)}
          {message.content.length > 2000 && '\n...truncated'}
        </pre>
      )}
    </div>
  )
}

function ActivityNarration({ message }: { message: ChatMessage }) {
  return (
    <div className="flex items-start gap-1.5 px-1.5 py-0.5 text-xs italic text-zinc-500">
      <CircleDot className="mt-0.5 h-3 w-3 shrink-0 text-zinc-600" />
      <div className="fd-markdown min-w-0 flex-1">
        <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{message.content}</Markdown>
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
