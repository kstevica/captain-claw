import { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { Cpu, Play, Square, RotateCcw, Trash2, ScrollText, ChevronUp, MessageSquare, Loader2, FolderOpen, Database, Pencil, Check, X, Copy, MoreVertical, Minimize2, Maximize2, Settings } from 'lucide-react'
import type { ProcessInfo } from '../../services/docker'
import { getProcessLogs } from '../../services/docker'
import { useProcessStore } from '../../stores/processStore'
import { useChatStore } from '../../stores/chatStore'
import { EmbeddedChat } from './EmbeddedChat'
import { AgentGroupBadges } from '../common/AgentGroups'
import { AgentConfigEditor } from './AgentConfigEditor'
import { DatastoreBrowser } from './DatastoreBrowser'
import { OpenDropdown } from '../common/OpenDropdown'
import { CognitiveModeSelector } from '../common/CognitiveModeSelector'
import { useAuthStore } from '../../stores/authStore'
import { queueSave, registerHydrator } from '../../services/settingsSync'

// ── Card view mode persistence (expanded / compact / icon) ──
type ViewMode = 'expanded' | 'compact' | 'icon'
const VIEW_KEY = 'fd:process-view'
function loadViewModes(): Record<string, ViewMode> {
  try { return JSON.parse(localStorage.getItem(VIEW_KEY) || '{}') } catch { return {} }
}
function saveViewModes(m: Record<string, ViewMode>) {
  const val = JSON.stringify(m)
  localStorage.setItem(VIEW_KEY, val)
  if (useAuthStore.getState().authEnabled) queueSave(VIEW_KEY, val)
}
registerHydrator((settings) => {
  if (settings[VIEW_KEY]) localStorage.setItem(VIEW_KEY, settings[VIEW_KEY])
})
function cycleMode(current: ViewMode): ViewMode {
  if (current === 'expanded') return 'compact'
  if (current === 'compact') return 'icon'
  return 'expanded'
}

// ── ANSI escape code → HTML converter ──
const ANSI_COLORS: Record<number, string> = {
  30: '#71717a', 31: '#f87171', 32: '#4ade80', 33: '#facc15',
  34: '#60a5fa', 35: '#c084fc', 36: '#22d3ee', 37: '#d4d4d8',
  90: '#a1a1aa', 91: '#fca5a5', 92: '#86efac', 93: '#fde68a',
  94: '#93c5fd', 95: '#d8b4fe', 96: '#67e8f9', 97: '#f4f4f5',
}

function ansiToHtml(text: string): string {
  let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  let result = ''
  let openSpans = 0
  const regex = /\x1b\[([0-9;]*)m/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = regex.exec(html)) !== null) {
    result += html.slice(lastIndex, match.index)
    lastIndex = regex.lastIndex
    const codes = match[1].split(';').map(Number)
    for (const code of codes) {
      if (code === 0) { while (openSpans > 0) { result += '</span>'; openSpans-- } }
      else if (code === 1) { result += '<span style="font-weight:700">'; openSpans++ }
      else if (code === 2) { result += '<span style="opacity:0.6">'; openSpans++ }
      else if (ANSI_COLORS[code]) { result += `<span style="color:${ANSI_COLORS[code]}">`; openSpans++ }
    }
  }
  result += html.slice(lastIndex)
  while (openSpans > 0) { result += '</span>'; openSpans-- }
  return result
}

const statusColors: Record<string, string> = {
  running: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  stopped: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
}

export function ProcessCard({ process: proc, onBrowseFiles, onDragStart, isDragging }: {
  process: ProcessInfo
  onBrowseFiles?: () => void
  onDragStart?: (e: React.PointerEvent) => void
  isDragging?: boolean
}) {
  const { stopProcess, startProcess, restartProcess, removeProcess, cloneProcess, setDescription, setNameOverride, setForwardingTask, getForwardingTask, setConsultApproval, getConsultApproval, setCognitiveMode: storeCognitiveMode, getCognitiveMode } = useProcessStore()
  const openChat = useChatStore((s) => s.openChat)
  const session = useChatStore((s) => s.sessions.get(`proc-${proc.slug}`))
  const busy = session?.busy ?? false
  const statusText = session?.statusText ?? ''
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState('')
  const [logsLoading, setLogsLoading] = useState(false)
  const logsRef = useRef<HTMLPreElement>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [editingDesc, setEditingDesc] = useState(false)
  const [descDraft, setDescDraft] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [editingFwdTask, setEditingFwdTask] = useState(false)
  const [fwdTaskDraft, setFwdTaskDraft] = useState('')
  const [viewMode, setViewMode] = useState<ViewMode>(() => loadViewModes()[proc.slug] || 'expanded')
  const [showConfig, setShowConfig] = useState(false)
  const [showDatastore, setShowDatastore] = useState(false)
  const cognitiveMode = getCognitiveMode(proc.slug)
  const [modeSaved, setModeSaved] = useState(false)

  const isRunning = proc.status === 'running'
  const agentName = proc.name || proc.slug
  const chatId = `proc-${proc.slug}`

  const toggleViewMode = () => {
    const next = cycleMode(viewMode)
    setViewMode(next)
    const m = loadViewModes()
    if (next === 'expanded') delete m[proc.slug]; else m[proc.slug] = next
    saveViewModes(m)
  }

  const doAction = async (action: string, fn: () => Promise<void>) => {
    setActionLoading(action)
    try { await fn() } catch (e) { console.error(e) }
    finally { setActionLoading(null) }
  }

  const toggleLogs = async () => {
    if (showLogs) { setShowLogs(false); return }
    setLogsLoading(true)
    try {
      const text = await getProcessLogs(proc.slug, 100)
      setLogs(text)
      setShowLogs(true)
    } catch (e) {
      setLogs(`Failed to fetch logs: ${e}`)
      setShowLogs(true)
    } finally { setLogsLoading(false) }
  }

  // Live log polling while logs are visible
  useEffect(() => {
    if (!showLogs || !isRunning) return
    const interval = setInterval(async () => {
      try {
        const text = await getProcessLogs(proc.slug, 100)
        setLogs(text)
      } catch { /* ignore polling errors */ }
    }, 3000)
    return () => clearInterval(interval)
  }, [showLogs, isRunning, proc.slug])

  // Auto-scroll logs to bottom on update
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const badgeCls = statusColors[proc.status] ?? statusColors.stopped

  const actionProps = {
    isRunning,
    actionLoading,
    onStart: () => doAction('start', () => startProcess(proc.slug)),
    onStop: () => doAction('stop', () => stopProcess(proc.slug)),
    onRestart: () => doAction('restart', () => restartProcess(proc.slug)),
    onClone: () => {
      const newName = prompt(`Clone '${agentName}'\n\nEnter a name for the cloned agent:`, `${agentName}-clone`)
      if (newName?.trim())
        doAction('clone', () => cloneProcess(proc.slug, newName.trim()))
    },
    onRemove: () => {
      if (confirm(`Remove process agent '${agentName}' from the registry?\n\nThe agent data in fd-data/${proc.slug}/ will be preserved.`))
        doAction('remove', () => removeProcess(proc.slug))
    },
    onConfig: () => setShowConfig(true),
  }

  // ── Config modal (rendered in all view modes via portal) ──
  const configModal = showConfig && createPortal(
    <AgentConfigEditor kind="process" identifier={proc.slug} agentName={agentName} onClose={() => setShowConfig(false)} />,
    document.body
  )

  const datastoreModal = showDatastore && isRunning && createPortal(
    <DatastoreBrowser host="localhost" port={proc.web_port} auth={proc.web_auth} agentName={agentName} onClose={() => setShowDatastore(false)} />,
    document.body
  )

  // ── Icon view (ultra-compact single row) ──
  if (viewMode === 'icon') {
    return (
      <><div
        onPointerDown={onDragStart}
        className={`group flex items-center gap-2 rounded-lg border bg-zinc-900/50 px-3 py-1.5 ${busy ? 'border-emerald-500/40' : 'border-zinc-800'} ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-emerald-500/10' : ''}`}
      >
        <Cpu className="h-3.5 w-3.5 text-emerald-500/70 shrink-0" />
        <span className="text-xs font-medium truncate min-w-0 flex-1">{agentName}</span>
        {busy ? (
          <div className="flex items-center gap-1 shrink-0">
            <Loader2 className="h-3 w-3 animate-spin text-emerald-400" />
            <span className="text-[10px] text-emerald-300 truncate max-w-[100px]">{statusText || 'Working...'}</span>
          </div>
        ) : isRunning ? (
          <span className="text-[10px] text-zinc-600 shrink-0">Idle</span>
        ) : null}
        <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium shrink-0 ${badgeCls}`}>
          {isRunning && (
            <span className="relative flex h-1 w-1">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
              <span className="relative inline-flex h-1 w-1 rounded-full bg-current" />
            </span>
          )}
          {proc.status}
        </span>
        <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Expand card">
          <Maximize2 className="h-3 w-3" />
        </button>
      </div>{configModal}{datastoreModal}</>
    )
  }

  // ── Compact view ──
  if (viewMode === 'compact') {
    return (
      <><div className={`rounded-xl border bg-zinc-900/50 overflow-hidden ${busy ? 'border-emerald-500/40' : 'border-zinc-800'}`}>
        <div
          onPointerDown={onDragStart}
          className={`flex items-center justify-end px-2 py-0.5 bg-emerald-900/10 ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-emerald-500/10' : ''}`}
        >
          <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 relative z-10" title="Shrink to icon">
            <Minimize2 className="h-3 w-3" />
          </button>
        </div>
        <div className="px-4 py-2.5 space-y-2">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-emerald-500/70 shrink-0" />
            <div className="group/name flex items-center gap-1 min-w-0 flex-1">
              {editingName ? (
                <div className="flex items-center gap-1">
                  <input value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && nameDraft.trim()) { setNameOverride(proc.slug, nameDraft.trim()); setEditingName(false) }
                      if (e.key === 'Escape') setEditingName(false)
                    }}
                    className="w-32 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-sm font-semibold text-zinc-200 focus:border-emerald-500/50 focus:outline-none" autoFocus
                  />
                  <button onClick={() => { if (nameDraft.trim()) { setNameOverride(proc.slug, nameDraft.trim()); setEditingName(false) } }} className="rounded p-0.5 text-emerald-400 hover:bg-zinc-800"><Check className="h-3 w-3" /></button>
                  <button onClick={() => setEditingName(false)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800"><X className="h-3 w-3" /></button>
                </div>
              ) : (
                <>
                  <span className="text-sm font-semibold truncate">{agentName}</span>
                  <button onClick={() => { setNameDraft(agentName); setEditingName(true) }}
                    className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover/name:opacity-100 hover:text-zinc-300">
                    <Pencil className="h-2.5 w-2.5" />
                  </button>
                </>
              )}
            </div>
            <span className="rounded bg-zinc-800 px-1 py-0.5 text-[10px] font-mono text-emerald-400/80 shrink-0">:{proc.web_port}</span>
            {isRunning && (
              <button onClick={() => openChat(chatId, agentName, 'localhost', proc.web_port, proc.web_auth)}
                className="flex items-center gap-1 rounded bg-emerald-600/20 px-1.5 py-0.5 text-[11px] font-medium text-emerald-300 hover:bg-emerald-600/30 shrink-0">
                <MessageSquare className="h-3 w-3" /> Chat
              </button>
            )}
            <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium shrink-0 ${badgeCls}`}>
              {isRunning && (
                <span className="relative flex h-1 w-1">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                  <span className="relative inline-flex h-1 w-1 rounded-full bg-current" />
                </span>
              )}
              {proc.status}
            </span>
          </div>
          <div className="flex items-center gap-2 min-h-[20px]">
            <div className="flex-1 min-w-0"><AgentGroupBadges agentId={chatId} /></div>
            {busy ? (
              <div className="flex items-center gap-1.5 shrink-0">
                <Loader2 className="h-3 w-3 animate-spin text-emerald-400" />
                <span className="text-[11px] text-emerald-300 truncate max-w-[160px]">{statusText || 'Working...'}</span>
              </div>
            ) : isRunning ? (
              <span className="text-[11px] text-zinc-600 shrink-0">Idle</span>
            ) : null}
          </div>
          <div className="flex items-center gap-1 -mx-1">
            {onBrowseFiles && (
              <button onClick={onBrowseFiles} className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                <FolderOpen className="h-3 w-3" /> Files
              </button>
            )}
            <button onClick={toggleLogs} disabled={logsLoading} className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">
              <ScrollText className={`h-3 w-3 ${logsLoading ? 'animate-spin' : ''}`} /> {showLogs ? 'Hide' : 'Logs'}
            </button>
            {isRunning && (
              <button onClick={() => setShowDatastore(true)} className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                <Database className="h-3 w-3" /> Data
              </button>
            )}
            <div className="flex-1" />
            <ProcessActionsDropdown {...actionProps} />
          </div>
        </div>
        {showLogs && (
          <div className="border-t border-zinc-800">
            <div className="flex items-center justify-between px-3 py-1 bg-zinc-950/50">
              <span className="text-[11px] text-zinc-500">Logs (last 100 lines)</span>
              <button onClick={() => toggleLogs()} className="text-zinc-500 hover:text-zinc-300"><ChevronUp className="h-3 w-3" /></button>
            </div>
            <pre ref={logsRef} className="max-h-40 overflow-auto px-3 py-1.5 text-[11px] text-zinc-400 font-mono leading-relaxed bg-zinc-950/30"
              dangerouslySetInnerHTML={{ __html: logs ? ansiToHtml(logs) : '(empty)' }} />
          </div>
        )}
        {isRunning && (
          <EmbeddedChat containerId={chatId} containerName={agentName} host="localhost" port={proc.web_port} auth={proc.web_auth} />
        )}
      </div>{configModal}{datastoreModal}</>
    )
  }

  // ── Normal (expanded) view ──
  return (
    <div className={`rounded-xl border bg-zinc-900/50 overflow-hidden ${busy ? 'border-emerald-500/40' : 'border-zinc-800'}`}>
      <div
        onPointerDown={onDragStart}
        className={`flex items-center justify-end px-2 py-0.5 bg-emerald-900/10 ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-emerald-500/10' : ''}`}
      >
        <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 relative z-10" title="Compact card">
          <Minimize2 className="h-3 w-3" />
        </button>
      </div>
      <div className="px-5 pb-5 pt-3">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-emerald-900/30 p-2">
              <Cpu className="h-5 w-5 text-emerald-400" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="group/name flex items-center gap-1">
                {editingName ? (
                  <div className="flex items-center gap-1">
                    <input value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && nameDraft.trim()) { setNameOverride(proc.slug, nameDraft.trim()); setEditingName(false) }
                        if (e.key === 'Escape') setEditingName(false)
                      }}
                      className="w-40 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-0.5 text-base font-semibold text-zinc-200 focus:border-emerald-500/50 focus:outline-none" autoFocus
                    />
                    <button onClick={() => { if (nameDraft.trim()) { setNameOverride(proc.slug, nameDraft.trim()); setEditingName(false) } }} className="rounded p-0.5 text-emerald-400 hover:bg-zinc-800"><Check className="h-3.5 w-3.5" /></button>
                    <button onClick={() => setEditingName(false)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800"><X className="h-3.5 w-3.5" /></button>
                  </div>
                ) : (
                  <>
                    <h3 className="text-base font-semibold truncate">{agentName}</h3>
                    <button onClick={() => { setNameDraft(agentName); setEditingName(true) }}
                      className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover/name:opacity-100 hover:bg-zinc-800 hover:text-zinc-300">
                      <Pencil className="h-3 w-3" />
                    </button>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2">
                {proc.pid && <span className="text-xs text-zinc-500 font-mono">PID {proc.pid}</span>}
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-xs font-mono text-emerald-400/80">:{proc.web_port}</span>
                <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${badgeCls}`}>
                  {isRunning && (
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
                    </span>
                  )}
                  {proc.status}
                </span>
                {isRunning && (
                  <>
                    <button onClick={() => openChat(chatId, agentName, 'localhost', proc.web_port, proc.web_auth)}
                      className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                      <MessageSquare className="h-3 w-3" /> Chat
                    </button>
                    <OpenDropdown host="localhost" port={proc.web_port} auth={proc.web_auth} />
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="mb-3 group/desc">
          {editingDesc ? (
            <div className="flex items-center gap-1.5">
              <input value={descDraft} onChange={(e) => setDescDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { setDescription(proc.slug, descDraft.trim()); setEditingDesc(false) }
                  if (e.key === 'Escape') setEditingDesc(false)
                }}
                placeholder="What this agent does..."
                className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-300 placeholder-zinc-600 focus:border-emerald-500/50 focus:outline-none" autoFocus
              />
              <button onClick={() => { setDescription(proc.slug, descDraft.trim()); setEditingDesc(false) }} className="rounded p-1 text-emerald-400 hover:bg-zinc-800"><Check className="h-3.5 w-3.5" /></button>
              <button onClick={() => setEditingDesc(false)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"><X className="h-3.5 w-3.5" /></button>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <p className="text-sm text-zinc-400 flex-1">{proc.description || <span className="text-zinc-600 italic">No description</span>}</p>
              <button onClick={() => { setDescDraft(proc.description || ''); setEditingDesc(true) }}
                className="rounded p-1 text-zinc-600 opacity-0 transition-opacity group-hover/desc:opacity-100 hover:bg-zinc-800 hover:text-zinc-300">
                <Pencil className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>

        {/* Cognitive Mode (runtime switch) */}
        <CognitiveModeSelector
          value={cognitiveMode}
          saved={modeSaved}
          onChange={async (newMode) => {
            storeCognitiveMode(proc.slug, newMode)
            try {
              const { token, authEnabled } = useAuthStore.getState()
              const headers: Record<string, string> = { 'Content-Type': 'application/json' }
              if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
              const res = await fetch(`/fd/agent-mode/process/${proc.slug}`, {
                method: 'PUT', headers, credentials: 'include',
                body: JSON.stringify({ mode: newMode }),
              })
              if (res.ok) { setModeSaved(true); setTimeout(() => setModeSaved(false), 2000) }
            } catch (err) { console.error('Failed to update cognitive mode:', err) }
          }}
        />

        {/* Forwarding Task */}
        <div className="mb-3 group/fwd">
          {editingFwdTask ? (
            <div className="flex items-start gap-1.5">
              <textarea value={fwdTaskDraft} onChange={(e) => setFwdTaskDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); setForwardingTask(proc.slug, fwdTaskDraft.trim()); setEditingFwdTask(false) }
                  if (e.key === 'Escape') setEditingFwdTask(false)
                }}
                placeholder="Task to suggest when forwarding context to this agent..." rows={2}
                className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 placeholder-zinc-600 focus:border-emerald-500/50 focus:outline-none resize-none" autoFocus
              />
              <button onClick={() => { setForwardingTask(proc.slug, fwdTaskDraft.trim()); setEditingFwdTask(false) }} className="rounded p-1 text-emerald-400 hover:bg-zinc-800"><Check className="h-3.5 w-3.5" /></button>
              <button onClick={() => setEditingFwdTask(false)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"><X className="h-3.5 w-3.5" /></button>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-zinc-500 flex-1">
                {getForwardingTask(proc.slug)
                  ? <><span className="text-zinc-600">Forwarding task:</span> <span className="text-zinc-400">{getForwardingTask(proc.slug)}</span></>
                  : <span className="text-zinc-600 italic">No forwarding task</span>}
              </p>
              <button onClick={() => { setFwdTaskDraft(getForwardingTask(proc.slug)); setEditingFwdTask(true) }}
                className="rounded p-1 text-zinc-600 opacity-0 transition-opacity group-hover/fwd:opacity-100 hover:bg-zinc-800 hover:text-zinc-300">
                <Pencil className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>

        {/* Peer consultation */}
        <div className="mb-3 flex items-center gap-2">
          <label className="flex items-center gap-1.5 cursor-pointer text-xs text-zinc-500">
            <input type="checkbox" checked={getConsultApproval(proc.slug)} onChange={(e) => setConsultApproval(proc.slug, e.target.checked)} className="accent-emerald-500 h-3 w-3" />
            Require approval for peer consultations
          </label>
        </div>

        {/* Group badges */}
        <div className="mb-3"><AgentGroupBadges agentId={chatId} /></div>

        <div className="mb-4 text-xs text-zinc-500">
          <span className="font-mono">{proc.provider}:{proc.model}</span>
          <span className="text-zinc-600 ml-2">pip process</span>
        </div>

        {/* Busy indicator */}
        {busy && (
          <div className="mb-3 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-1.5">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-400" />
            <span className="text-xs text-emerald-300">{statusText || 'Working...'}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-1">
          {onBrowseFiles && (
            <button onClick={onBrowseFiles} className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <FolderOpen className="h-3.5 w-3.5" /> Files
            </button>
          )}
          <button onClick={toggleLogs} disabled={logsLoading} className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">
            <ScrollText className={`h-3.5 w-3.5 ${logsLoading ? 'animate-spin' : ''}`} /> {showLogs ? 'Hide Logs' : 'Logs'}
          </button>
          {isRunning && (
            <button onClick={() => setShowDatastore(true)} className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <Database className="h-3.5 w-3.5" /> Data
            </button>
          )}
          <div className="flex-1" />
          <ProcessActionsDropdown {...actionProps} />
        </div>
      </div>

      {/* Logs */}
      {showLogs && (
        <div className="border-t border-zinc-800">
          <div className="flex items-center justify-between px-4 py-1.5 bg-zinc-950/50">
            <span className="text-xs text-zinc-500">Logs (last 100 lines)</span>
            <button onClick={() => toggleLogs()} className="text-xs text-zinc-500 hover:text-zinc-300"><ChevronUp className="h-3.5 w-3.5" /></button>
          </div>
          <pre ref={logsRef} className="max-h-60 overflow-auto px-4 py-2 text-xs text-zinc-400 font-mono leading-relaxed bg-zinc-950/30"
            dangerouslySetInnerHTML={{ __html: logs ? ansiToHtml(logs) : '(empty)' }} />
        </div>
      )}

      {/* Embedded Chat */}
      {isRunning && (
        <EmbeddedChat containerId={chatId} containerName={agentName} host="localhost" port={proc.web_port} auth={proc.web_auth} />
      )}

      {configModal}
      {datastoreModal}
    </div>
  )
}

function ProcessActionsDropdown({ isRunning, actionLoading, onStart, onStop, onRestart, onClone, onRemove, onConfig }: {
  isRunning: boolean
  actionLoading: string | null
  onStart: () => void
  onStop: () => void
  onRestart: () => void
  onClone: () => void
  onRemove: () => void
  onConfig: () => void
}) {
  const [open, setOpen] = useState(false)
  const btnRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ top: 0, left: 0 })

  const updatePos = useCallback(() => {
    if (!btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    setPos({ top: r.bottom + 4, left: r.right })
  }, [])

  useEffect(() => {
    if (!open) return
    updatePos()
    const handler = (e: MouseEvent) => {
      const t = e.target as Node
      if (menuRef.current?.contains(t) || btnRef.current?.contains(t)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    window.addEventListener('scroll', updatePos, true)
    return () => {
      document.removeEventListener('mousedown', handler)
      window.removeEventListener('scroll', updatePos, true)
    }
  }, [open, updatePos])

  const items: { icon: typeof Play; label: string; onClick: () => void; loading?: boolean; danger?: boolean; accent?: boolean; show?: boolean }[] = [
    { icon: Play,      label: 'Start',   onClick: onStart,   loading: actionLoading === 'start',   accent: true, show: !isRunning },
    { icon: Square,    label: 'Stop',    onClick: onStop,    loading: actionLoading === 'stop',    show: isRunning },
    { icon: RotateCcw, label: 'Restart', onClick: onRestart, loading: actionLoading === 'restart', show: isRunning },
    { icon: Settings,  label: 'Config',  onClick: onConfig },
    { icon: Copy,      label: 'Clone',   onClick: onClone,   loading: actionLoading === 'clone' },
    { icon: Trash2,    label: 'Remove',  onClick: onRemove,  loading: actionLoading === 'remove',  danger: true },
  ]

  const visibleItems = items.filter((i) => i.show !== false)

  return (
    <>
      <button ref={btnRef} onClick={() => { updatePos(); setOpen(!open) }}
        className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors ${
          open ? 'bg-zinc-800 text-zinc-200' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
        }`}>
        <MoreVertical className="h-3.5 w-3.5" /> Actions
      </button>
      {open && createPortal(
        <div ref={menuRef} className="fixed z-[100] w-44 rounded-lg border border-zinc-700/50 bg-zinc-900 py-1 shadow-xl shadow-black/40"
          style={{ top: pos.top, left: pos.left - 176 }}>
          {visibleItems.map((item, i) => {
            const Icon = item.icon
            const isDivider = item.danger && i > 0
            return (
              <div key={item.label}>
                {isDivider && <div className="my-1 border-t border-zinc-800" />}
                <button onClick={() => { setOpen(false); item.onClick() }} disabled={item.loading}
                  className={`flex w-full items-center gap-2.5 px-3 py-2 text-[13px] transition-colors disabled:opacity-40 ${
                    item.danger ? 'text-red-400 hover:bg-red-500/10'
                    : item.accent ? 'text-emerald-400 hover:bg-emerald-500/10'
                    : 'text-zinc-300 hover:bg-zinc-800'
                  }`}>
                  <Icon className={`h-4 w-4 ${item.loading ? 'animate-spin' : ''} ${
                    item.danger ? 'text-red-400/70' : item.accent ? 'text-emerald-400/70' : 'text-zinc-500'
                  }`} />
                  {item.label}
                </button>
              </div>
            )
          })}
        </div>,
        document.body,
      )}
    </>
  )
}
