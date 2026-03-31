import { useState } from 'react'
import { MessageSquare, Trash2, RefreshCw, Cpu, ExternalLink, Loader2, FolderOpen, Pencil, Check, X, Minimize2, Maximize2 } from 'lucide-react'
import type { LocalAgent } from '../../stores/localAgentStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'
import { useChatStore } from '../../stores/chatStore'
import { EmbeddedChat } from './EmbeddedChat'
import { AgentGroupBadges } from '../common/AgentGroups'

const statusStyles: Record<string, string> = {
  online: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  offline: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  unknown: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
}

// ── Card view mode persistence (expanded / compact / icon) ──
type ViewMode = 'expanded' | 'compact' | 'icon'
const VIEW_KEY = 'fd:local-agent-view'
function loadViewModes(): Record<string, ViewMode> {
  try { return JSON.parse(localStorage.getItem(VIEW_KEY) || '{}') } catch { return {} }
}
function saveViewModes(m: Record<string, ViewMode>) {
  localStorage.setItem(VIEW_KEY, JSON.stringify(m))
}
function cycleMode(current: ViewMode): ViewMode {
  if (current === 'expanded') return 'compact'
  if (current === 'compact') return 'icon'
  return 'expanded'
}

export function LocalAgentCard({ agent, onBrowseFiles, onDragStart, isDragging }: { agent: LocalAgent; onBrowseFiles?: () => void; onDragStart?: (e: React.PointerEvent) => void; isDragging?: boolean }) {
  const { removeAgent, probeAgent, updateAgent } = useLocalAgentStore()
  const openChat = useChatStore((s) => s.openChat)
  const session = useChatStore((s) => s.sessions.get(agent.id))
  const busy = session?.busy ?? false
  const statusText = session?.statusText ?? ''
  const [editingDesc, setEditingDesc] = useState(false)
  const [descDraft, setDescDraft] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [editingFwdTask, setEditingFwdTask] = useState(false)
  const [fwdTaskDraft, setFwdTaskDraft] = useState('')
  const [viewMode, setViewMode] = useState<ViewMode>(() => loadViewModes()[agent.id] || 'expanded')

  const toggleViewMode = () => {
    const next = cycleMode(viewMode)
    setViewMode(next)
    const m = loadViewModes()
    if (next === 'expanded') delete m[agent.id]; else m[agent.id] = next
    saveViewModes(m)
  }

  const isOnline = agent.status === 'online'

  // ── Icon view (ultra-compact single row) ──
  if (viewMode === 'icon') {
    return (
      <div
        onPointerDown={onDragStart}
        className={`group flex items-center gap-2 rounded-lg border bg-zinc-900/50 px-3 py-1.5 ${busy ? 'border-violet-500/40' : 'border-zinc-800'} ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-violet-500/10' : ''}`}
      >
        <Cpu className="h-3.5 w-3.5 text-zinc-500 shrink-0" />
        <span className="text-xs font-medium truncate min-w-0 flex-1">{agent.name}</span>
        {busy ? (
          <div className="flex items-center gap-1 shrink-0">
            <Loader2 className="h-3 w-3 animate-spin text-violet-400" />
            <span className="text-[10px] text-violet-300 truncate max-w-[100px]">{statusText || 'Working...'}</span>
          </div>
        ) : isOnline ? (
          <span className="text-[10px] text-zinc-600 shrink-0">Idle</span>
        ) : null}
        <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium shrink-0 ${statusStyles[agent.status]}`}>
          {isOnline && (
            <span className="relative flex h-1 w-1">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
              <span className="relative inline-flex h-1 w-1 rounded-full bg-current" />
            </span>
          )}
          {agent.status}
        </span>
        <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Expand card">
          <Maximize2 className="h-3 w-3" />
        </button>
      </div>
    )
  }

  // ── Compact view ──
  if (viewMode === 'compact') {
    return (
      <div className={`rounded-xl border bg-zinc-900/50 overflow-hidden ${busy ? 'border-violet-500/40' : 'border-zinc-800'}`}>
        {/* Drag handle area with compact toggle */}
        <div
          onPointerDown={onDragStart}
          className={`flex items-center justify-end px-2 py-0.5 bg-zinc-800/30 ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-violet-500/10' : ''}`}
        >
          <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 relative z-10" title="Shrink to icon">
            <Minimize2 className="h-3 w-3" />
          </button>
        </div>

        <div className="px-4 py-2.5 space-y-2">
          {/* Line 1: icon, name, chat button, status */}
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-zinc-500 shrink-0" />
            <div className="group/name flex items-center gap-1 min-w-0 flex-1">
              {editingName ? (
                <div className="flex items-center gap-1">
                  <input
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && nameDraft.trim()) { updateAgent(agent.id, { name: nameDraft.trim() }); setEditingName(false) }
                      if (e.key === 'Escape') setEditingName(false)
                    }}
                    className="w-32 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-sm font-semibold text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    autoFocus
                  />
                  <button onClick={() => { if (nameDraft.trim()) { updateAgent(agent.id, { name: nameDraft.trim() }); setEditingName(false) } }} className="rounded p-0.5 text-emerald-400 hover:bg-zinc-800"><Check className="h-3 w-3" /></button>
                  <button onClick={() => setEditingName(false)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800"><X className="h-3 w-3" /></button>
                </div>
              ) : (
                <>
                  <span className="text-sm font-semibold truncate">{agent.name}</span>
                  <button
                    onClick={() => { setNameDraft(agent.name); setEditingName(true) }}
                    className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover/name:opacity-100 hover:text-zinc-300"
                  >
                    <Pencil className="h-2.5 w-2.5" />
                  </button>
                </>
              )}
            </div>
            <span className="rounded bg-zinc-800 px-1 py-0.5 text-[10px] font-mono text-zinc-500 shrink-0">
              {agent.host}:{agent.port}
            </span>
            {agent.status === 'online' && (
              <button
                onClick={() => openChat(agent.id, agent.name, agent.host, agent.port, agent.authToken)}
                className="flex items-center gap-1 rounded bg-violet-600/20 px-1.5 py-0.5 text-[11px] font-medium text-violet-300 hover:bg-violet-600/30 shrink-0"
              >
                <MessageSquare className="h-3 w-3" />
                Chat
              </button>
            )}
            <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium shrink-0 ${statusStyles[agent.status]}`}>
              {agent.status === 'online' && (
                <span className="relative flex h-1 w-1">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                  <span className="relative inline-flex h-1 w-1 rounded-full bg-current" />
                </span>
              )}
              {agent.status}
            </span>
          </div>

          {/* Line 2: groups + status text */}
          <div className="flex items-center gap-2 min-h-[20px]">
            <div className="flex-1 min-w-0">
              <AgentGroupBadges agentId={agent.id} />
            </div>
            {busy ? (
              <div className="flex items-center gap-1.5 shrink-0">
                <Loader2 className="h-3 w-3 animate-spin text-violet-400" />
                <span className="text-[11px] text-violet-300 truncate max-w-[160px]">{statusText || 'Working...'}</span>
              </div>
            ) : agent.status === 'online' ? (
              <span className="text-[11px] text-zinc-600 shrink-0">Idle</span>
            ) : null}
          </div>

          {/* Line 3: files, probe, remove */}
          <div className="flex items-center gap-1 -mx-1">
            {onBrowseFiles && (
              <button onClick={onBrowseFiles} className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
                <FolderOpen className="h-3 w-3" /> Files
              </button>
            )}
            <button onClick={() => probeAgent(agent.id)} className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
              <RefreshCw className="h-3 w-3" /> Probe
            </button>
            <div className="flex-1" />
            <button
              onClick={() => { if (confirm(`Remove '${agent.name}'?`)) removeAgent(agent.id) }}
              className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] font-medium text-red-400/60 hover:bg-red-500/10 hover:text-red-400"
            >
              <Trash2 className="h-3 w-3" /> Remove
            </button>
          </div>
        </div>

        {/* Embedded Chat */}
        {agent.status === 'online' && (
          <EmbeddedChat containerId={agent.id} containerName={agent.name} host={agent.host} port={agent.port} auth={agent.authToken} />
        )}
      </div>
    )
  }

  // ── Normal (expanded) view ──
  return (
    <div className={`rounded-xl border bg-zinc-900/50 overflow-hidden ${busy ? 'border-violet-500/40' : 'border-zinc-800'}`}>
      {/* Drag handle area with compact toggle */}
      <div
        onPointerDown={onDragStart}
        className={`flex items-center justify-end px-2 py-0.5 bg-zinc-800/30 ${onDragStart ? 'cursor-grab active:cursor-grabbing' : ''} ${isDragging ? 'bg-violet-500/10' : ''}`}
      >
        <button onPointerDown={(e) => e.stopPropagation()} onClick={toggleViewMode} className="rounded p-0.5 text-zinc-600 hover:text-zinc-400 relative z-10" title="Compact card">
          <Minimize2 className="h-3 w-3" />
        </button>
      </div>

      <div className="px-5 pb-5 pt-3">
      {/* Header */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-zinc-800 p-2">
            <Cpu className="h-5 w-5 text-zinc-400" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="group/name flex items-center gap-1">
              {editingName ? (
                <div className="flex items-center gap-1">
                  <input
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && nameDraft.trim()) { updateAgent(agent.id, { name: nameDraft.trim() }); setEditingName(false) }
                      if (e.key === 'Escape') setEditingName(false)
                    }}
                    className="w-40 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-0.5 text-base font-semibold text-zinc-200 focus:border-violet-500/50 focus:outline-none"
                    autoFocus
                  />
                  <button onClick={() => { if (nameDraft.trim()) { updateAgent(agent.id, { name: nameDraft.trim() }); setEditingName(false) } }} className="rounded p-0.5 text-emerald-400 hover:bg-zinc-800"><Check className="h-3.5 w-3.5" /></button>
                  <button onClick={() => setEditingName(false)} className="rounded p-0.5 text-zinc-500 hover:bg-zinc-800"><X className="h-3.5 w-3.5" /></button>
                </div>
              ) : (
                <>
                  <h3 className="text-base font-semibold truncate">{agent.name}</h3>
                  <button
                    onClick={() => { setNameDraft(agent.name); setEditingName(true) }}
                    className="rounded p-0.5 text-zinc-600 opacity-0 transition-opacity group-hover/name:opacity-100 hover:bg-zinc-800 hover:text-zinc-300"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                </>
              )}
            </div>
            <span className="text-xs text-zinc-500 font-mono">{agent.host}:{agent.port}</span>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {agent.status === 'online' && (
            <>
              <button
                onClick={() => openChat(agent.id, agent.name, agent.host, agent.port, agent.authToken)}
                className="flex items-center gap-1 rounded-lg bg-violet-600/20 px-2 py-0.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30"
              >
                <MessageSquare className="h-3 w-3" />
                Chat
              </button>
              <button
                onClick={() => window.open(`http://${agent.host}:${agent.port}/chat`, '_blank')}
                className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <ExternalLink className="h-3 w-3" />
                Open
              </button>
            </>
          )}
          <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${statusStyles[agent.status]}`}>
            {agent.status === 'online' && (
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
              </span>
            )}
            {agent.status}
          </span>
        </div>
      </div>

      {/* Description (editable) */}
      <div className="mb-3 group/desc">
        {editingDesc ? (
          <div className="flex items-center gap-1.5">
            <input
              value={descDraft}
              onChange={(e) => setDescDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { updateAgent(agent.id, { description: descDraft.trim() }); setEditingDesc(false) }
                if (e.key === 'Escape') setEditingDesc(false)
              }}
              placeholder="What this agent does..."
              className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-300 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              autoFocus
            />
            <button
              onClick={() => { updateAgent(agent.id, { description: descDraft.trim() }); setEditingDesc(false) }}
              className="rounded p-1 text-emerald-400 hover:bg-zinc-800"
            >
              <Check className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setEditingDesc(false)}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <p className="text-sm text-zinc-400 flex-1">
              {agent.description || <span className="text-zinc-600 italic">No description</span>}
            </p>
            <button
              onClick={() => { setDescDraft(agent.description || ''); setEditingDesc(true) }}
              className="rounded p-1 text-zinc-600 opacity-0 transition-opacity group-hover/desc:opacity-100 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <Pencil className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>

      {/* Forwarding Task (editable) */}
      <div className="mb-3 group/fwd">
        {editingFwdTask ? (
          <div className="flex items-start gap-1.5">
            <textarea
              value={fwdTaskDraft}
              onChange={(e) => setFwdTaskDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); updateAgent(agent.id, { forwardingTask: fwdTaskDraft.trim() }); setEditingFwdTask(false) }
                if (e.key === 'Escape') setEditingFwdTask(false)
              }}
              placeholder="Task to suggest when forwarding context to this agent..."
              rows={2}
              className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none resize-none"
              autoFocus
            />
            <button
              onClick={() => { updateAgent(agent.id, { forwardingTask: fwdTaskDraft.trim() }); setEditingFwdTask(false) }}
              className="rounded p-1 text-emerald-400 hover:bg-zinc-800"
            >
              <Check className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => setEditingFwdTask(false)}
              className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <p className="text-xs text-zinc-500 flex-1">
              {agent.forwardingTask
                ? <><span className="text-zinc-600">Forwarding task:</span> <span className="text-zinc-400">{agent.forwardingTask}</span></>
                : <span className="text-zinc-600 italic">No forwarding task</span>}
            </p>
            <button
              onClick={() => { setFwdTaskDraft(agent.forwardingTask || ''); setEditingFwdTask(true) }}
              className="rounded p-1 text-zinc-600 opacity-0 transition-opacity group-hover/fwd:opacity-100 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <Pencil className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>

      {/* Peer consultation approval toggle */}
      <div className="mb-3 flex items-center gap-2">
        <label className="flex items-center gap-1.5 cursor-pointer text-xs text-zinc-500">
          <input
            type="checkbox"
            checked={agent.consultApproval ?? false}
            onChange={(e) => updateAgent(agent.id, { consultApproval: e.target.checked })}
            className="accent-violet-500 h-3 w-3"
          />
          Require approval for peer consultations
        </label>
      </div>

      {/* Group badges */}
      <div className="mb-3">
        <AgentGroupBadges agentId={agent.id} />
      </div>

      {/* Persona / Model override (visible when chat connected) */}
      {session?.connected && (session.models.length > 0 || session.personalities.length > 0) && (
        <div className="mb-3 flex flex-wrap gap-2">
          {session.personalities.length > 0 && (
            <select
              value={session.activePersonality}
              onChange={(e) => useChatStore.getState().setPersonality(agent.id, e.target.value)}
              className="flex-1 min-w-[120px] rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              <option value="">Default persona</option>
              {session.personalities.map((p) => (
                <option key={p.id || p.name} value={p.id || p.name}>{p.name}</option>
              ))}
            </select>
          )}
          {session.models.length > 0 && (
            <select
              value={session.activeModel}
              onChange={(e) => useChatStore.getState().setModel(agent.id, e.target.value)}
              className="flex-1 min-w-[120px] rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              <option value="">Default model</option>
              {session.models.map((m) => (
                <option key={m.selector || m.id} value={m.selector || m.id}>{m.label || m.id}</option>
              ))}
            </select>
          )}
        </div>
      )}

      {/* Busy indicator */}
      {busy && (
        <div className="mb-3 flex items-center gap-2 rounded-lg bg-violet-500/10 px-3 py-1.5">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet-400" />
          <span className="text-xs text-violet-300">{statusText || 'Working...'}</span>
        </div>
      )}

      {/* Actions */}
      <div className="mt-4 flex items-center gap-1">
        {onBrowseFiles && (
          <button
            onClick={onBrowseFiles}
            className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
          >
            <FolderOpen className="h-3.5 w-3.5" />
            Files
          </button>
        )}
        <button
          onClick={() => probeAgent(agent.id)}
          className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Probe
        </button>
        <div className="flex-1" />
        <button
          onClick={() => { if (confirm(`Remove '${agent.name}'?`)) removeAgent(agent.id) }}
          className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium text-red-400/60 hover:bg-red-500/10 hover:text-red-400"
        >
          <Trash2 className="h-3.5 w-3.5" />
          Remove
        </button>
      </div>
      </div>

      {/* Embedded Chat */}
      {agent.status === 'online' && (
        <EmbeddedChat
          containerId={agent.id}
          containerName={agent.name}
          host={agent.host}
          port={agent.port}
          auth={agent.authToken}
        />
      )}
    </div>
  )
}
