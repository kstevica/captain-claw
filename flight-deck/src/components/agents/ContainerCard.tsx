import { useState } from 'react'
import { Box, Play, Square, RotateCcw, Trash2, ScrollText, ChevronDown, ChevronUp, MessageSquare, ExternalLink, Loader2, FolderOpen, Pencil, Check, X } from 'lucide-react'
import type { ContainerInfo } from '../../services/docker'
import { getContainerLogs } from '../../services/docker'
import { useContainerStore } from '../../stores/containerStore'
import { useChatStore } from '../../stores/chatStore'

const statusColors: Record<string, string> = {
  running: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  exited: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  created: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  restarting: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  paused: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  dead: 'bg-red-500/20 text-red-400 border-red-500/30',
}

export function ContainerCard({ container, onBrowseFiles }: { container: ContainerInfo; onBrowseFiles?: () => void }) {
  const { stopContainer, startContainer, restartContainer, removeContainer, setDescription } = useContainerStore()
  const openChat = useChatStore((s) => s.openChat)
  const session = useChatStore((s) => s.sessions.get(container.id))
  const busy = session?.busy ?? false
  const statusText = session?.statusText ?? ''
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState('')
  const [logsLoading, setLogsLoading] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [showConnect, setShowConnect] = useState(false)
  const [connectPort, setConnectPort] = useState('24080')
  const [editingDesc, setEditingDesc] = useState(false)
  const [descDraft, setDescDraft] = useState('')

  const isRunning = container.status === 'running'

  const doAction = async (action: string, fn: () => Promise<void>) => {
    setActionLoading(action)
    try { await fn() } catch (e) { console.error(e) }
    finally { setActionLoading(null) }
  }

  const toggleLogs = async () => {
    if (showLogs) {
      setShowLogs(false)
      return
    }
    setLogsLoading(true)
    try {
      const text = await getContainerLogs(container.id, 100)
      setLogs(text)
      setShowLogs(true)
    } catch (e) {
      setLogs(`Failed to fetch logs: ${e}`)
      setShowLogs(true)
    } finally {
      setLogsLoading(false)
    }
  }

  const badgeCls = statusColors[container.status] ?? statusColors.created

  return (
    <div className={`rounded-xl border bg-zinc-900/50 overflow-hidden ${busy ? 'border-violet-500/40' : 'border-zinc-800'}`}>
      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-zinc-800 p-2">
              <Box className="h-5 w-5 text-zinc-400" />
            </div>
            <div>
              <h3 className="text-base font-semibold">{container.agent_name || container.name}</h3>
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500 font-mono">{container.id}</span>
                {container.web_port != null && (
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-xs font-mono text-emerald-400/80">
                    :{container.web_port}
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {isRunning && container.web_port && (
              <>
                <button
                  onClick={() => openChat(container.id, container.agent_name || container.name, 'localhost', container.web_port!, container.web_auth)}
                  className="flex items-center gap-1 rounded-lg bg-violet-600/20 px-2 py-0.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30"
                >
                  <MessageSquare className="h-3 w-3" />
                  Chat
                </button>
                <button
                  onClick={() => window.open(`http://localhost:${container.web_port}/chat`, '_blank')}
                  className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                >
                  <ExternalLink className="h-3 w-3" />
                  Open
                </button>
              </>
            )}
            {isRunning && !container.web_port && (
              <button
                onClick={() => setShowConnect(!showConnect)}
                className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-xs font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <MessageSquare className="h-3 w-3" />
                Chat
              </button>
            )}
            <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${badgeCls}`}>
              {isRunning && (
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
                </span>
              )}
              {container.status}
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
                  if (e.key === 'Enter') { setDescription(container.id, descDraft.trim()); setEditingDesc(false) }
                  if (e.key === 'Escape') setEditingDesc(false)
                }}
                placeholder="What this agent does..."
                className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-300 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                autoFocus
              />
              <button
                onClick={() => { setDescription(container.id, descDraft.trim()); setEditingDesc(false) }}
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
                {container.description || <span className="text-zinc-600 italic">No description</span>}
              </p>
              <button
                onClick={() => { setDescDraft(container.description || ''); setEditingDesc(true) }}
                className="rounded p-1 text-zinc-600 opacity-0 transition-opacity group-hover/desc:opacity-100 hover:bg-zinc-800 hover:text-zinc-300"
              >
                <Pencil className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>

        <div className="mb-4 text-xs text-zinc-500">
          <span className="font-mono">{container.image}</span>
        </div>

        {/* Persona / Model override (visible when chat connected) */}
        {session?.connected && (session.models.length > 0 || session.personalities.length > 0) && (
          <div className="mb-3 flex flex-wrap gap-2">
            {session.personalities.length > 0 && (
              <select
                value={session.activePersonality}
                onChange={(e) => useChatStore.getState().setPersonality(container.id, e.target.value)}
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
                onChange={(e) => useChatStore.getState().setModel(container.id, e.target.value)}
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
        <div className="flex items-center gap-1">
          {isRunning ? (
            <>
              <ActionBtn icon={Square} label="Stop" loading={actionLoading === 'stop'}
                onClick={() => doAction('stop', () => stopContainer(container.id))} />
              <ActionBtn icon={RotateCcw} label="Restart" loading={actionLoading === 'restart'}
                onClick={() => doAction('restart', () => restartContainer(container.id))} />
            </>
          ) : (
            <ActionBtn icon={Play} label="Start" loading={actionLoading === 'start'} accent
              onClick={() => doAction('start', () => startContainer(container.id))} />
          )}
          {onBrowseFiles && (
            <ActionBtn icon={FolderOpen} label="Files" onClick={onBrowseFiles} />
          )}
          <ActionBtn icon={ScrollText} label={showLogs ? 'Hide Logs' : 'Logs'} loading={logsLoading}
            onClick={toggleLogs} />
          <div className="flex-1" />
          <ActionBtn icon={Trash2} label="Remove" loading={actionLoading === 'remove'} danger
            onClick={() => {
              if (confirm(`Remove container '${container.name}'?`))
                doAction('remove', () => removeContainer(container.id))
            }} />
        </div>
      </div>

      {/* Connect prompt (for containers without web_port label) */}
      {showConnect && (
        <div className="border-t border-zinc-800 px-4 py-2.5 bg-zinc-950/50">
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Port:</span>
            <input
              value={connectPort}
              onChange={(e) => setConnectPort(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const p = parseInt(connectPort, 10)
                  if (!isNaN(p)) {
                    openChat(container.id, container.agent_name || container.name, 'localhost', p, '')
                    setShowConnect(false)
                  }
                }
              }}
              placeholder="24080"
              className="w-20 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              autoFocus
            />
            <button
              onClick={() => {
                const p = parseInt(connectPort, 10)
                if (!isNaN(p)) {
                  openChat(container.id, container.agent_name || container.name, 'localhost', p, '')
                  setShowConnect(false)
                }
              }}
              className="rounded bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500"
            >
              Connect
            </button>
            <button onClick={() => setShowConnect(false)} className="text-xs text-zinc-500 hover:text-zinc-300">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Logs */}
      {showLogs && (
        <div className="border-t border-zinc-800">
          <div className="flex items-center justify-between px-4 py-1.5 bg-zinc-950/50">
            <span className="text-xs text-zinc-500">Logs (last 100 lines)</span>
            <button onClick={() => toggleLogs()} className="text-xs text-zinc-500 hover:text-zinc-300">
              {showLogs ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
          </div>
          <pre className="max-h-60 overflow-auto px-4 py-2 text-xs text-zinc-400 font-mono leading-relaxed bg-zinc-950/30">
            {logs || '(empty)'}
          </pre>
        </div>
      )}
    </div>
  )
}

function ActionBtn({ icon: Icon, label, onClick, loading, accent, danger }: {
  icon: typeof Play; label: string; onClick: () => void; loading?: boolean; accent?: boolean; danger?: boolean
}) {
  let cls = 'flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 '
  if (accent) cls += 'bg-violet-600/20 text-violet-300 hover:bg-violet-600/30'
  else if (danger) cls += 'text-red-400/60 hover:text-red-400 hover:bg-red-500/10'
  else cls += 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'

  return (
    <button onClick={onClick} disabled={loading ?? false} title={label} className={cls}>
      <Icon className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
      {label}
    </button>
  )
}
