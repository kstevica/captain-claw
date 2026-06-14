import { useState, useEffect } from 'react'
import { RefreshCw, Loader2, ListChecks, ChevronDown, Send, CheckSquare, Sparkles, Check, AlertTriangle } from 'lucide-react'
import type { CouncilArtifact, ActionPoint } from '../../stores/councilStore'

interface ActionPointsPanelProps {
  artifacts: CouncilArtifact[]   // kind === 'action_points', one per agent
  generating: boolean
  busy: boolean                  // an agent is currently speaking/recording
  onGenerate: () => void
  onSend: (agentId: string, item: ActionPoint) => void | Promise<void>
}

type SendState = 'idle' | 'sending' | 'sent' | 'error'

function parse(content: string): ActionPoint[] {
  try {
    const arr = JSON.parse(content)
    return Array.isArray(arr) ? arr : []
  } catch { return [] }
}

export function ActionPointsPanel({ artifacts, generating, busy, onGenerate, onSend }: ActionPointsPanelProps) {
  const [collapsed, setCollapsed] = useState(false)
  // Per-item send state, keyed by `${artifactId}-${index}`.
  const [sendState, setSendState] = useState<Record<string, SendState>>({})

  // Reset confirmations when a new extraction starts.
  useEffect(() => { if (generating) setSendState({}) }, [generating])

  const handleSend = async (agentId: string, item: ActionPoint, key: string) => {
    setSendState(p => ({ ...p, [key]: 'sending' }))
    try {
      await onSend(agentId, item)
      setSendState(p => ({ ...p, [key]: 'sent' }))
    } catch {
      setSendState(p => ({ ...p, [key]: 'error' }))
    }
  }

  return (
    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <button onClick={() => setCollapsed(!collapsed)} className="flex items-center gap-2 min-w-0">
          <ListChecks className="h-4 w-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <h3 className="text-sm font-medium text-emerald-700 dark:text-emerald-300">Action Points</h3>
          <ChevronDown className={`h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400 transition-transform ${collapsed ? '-rotate-90' : ''}`} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onGenerate() }}
          disabled={generating}
          className="ml-auto flex items-center gap-1 rounded-lg border border-emerald-500/30 px-2 py-1 text-[10px] font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10 disabled:opacity-40"
        >
          {generating
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Extracting...</>
            : <><RefreshCw className="h-3 w-3" /> {artifacts.length > 0 ? 'Regenerate' : 'Extract'}</>}
        </button>
      </div>

      {!collapsed && (
        <>
          {artifacts.length === 0 && !generating && (
            <p className="text-xs text-zinc-500">
              No action points yet. Each agent extracts its own outstanding next steps from the discussion (scoped to its part — no full-transcript dump).
            </p>
          )}
          {generating && artifacts.length === 0 && (
            <div className="flex items-center gap-2 py-2 text-xs text-zinc-400">
              <Loader2 className="h-3 w-3 animate-spin" /> Collecting action points from agents...
            </div>
          )}

          <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
            {artifacts.map(art => {
              const items = parse(art.content)
              if (items.length === 0) return null
              return (
                <div key={`${art.agentId}-${art.id}`} className="rounded-lg dark:bg-zinc-800/40 p-2.5">
                  <div className="mb-2 text-xs font-semibold text-zinc-100">{art.agentName}</div>
                  <div className="space-y-2">
                    {items.map((item, i) => {
                      const key = `${art.id}-${i}`
                      // item.sent is persisted in the artifact, so a recorded
                      // item stays "Recorded" across reloads (and can't re-send).
                      const st = sendState[key] || (item.sent ? 'sent' : 'idle')
                      return (
                      <div key={i} className="rounded-lg border border-zinc-700/50 bg-white dark:bg-zinc-900/40 p-2.5">
                        <div className="flex items-start gap-2">
                          <span className={`mt-0.5 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium shrink-0 ${
                            item.kind === 'intent'
                              ? 'bg-violet-600/20 border border-violet-500/30 text-violet-700 dark:text-violet-300'
                              : 'bg-emerald-600/20 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
                          }`}>
                            {item.kind === 'intent' ? <Sparkles className="h-2.5 w-2.5" /> : <CheckSquare className="h-2.5 w-2.5" />}
                            {item.kind}
                          </span>
                          <span className="text-xs font-semibold text-zinc-100 flex-1 min-w-0">{item.title}</span>
                          <button
                            onClick={() => handleSend(art.agentId, item, key)}
                            disabled={busy || st === 'sending' || st === 'sent'}
                            className={`flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] disabled:opacity-60 shrink-0 ${
                              st === 'sent'
                                ? 'border-emerald-500/40 text-emerald-600 dark:text-emerald-400'
                                : st === 'error'
                                  ? 'border-red-500/40 text-red-400 hover:bg-red-500/10'
                                  : 'border-zinc-600 text-zinc-300 hover:bg-zinc-700/40'
                            }`}
                            title={
                              st === 'sent' ? `Recorded in ${art.agentName}'s ${item.kind === 'intent' ? 'intentions' : 'todos'}`
                                : st === 'error' ? 'Failed to record — click to retry'
                                : `Record this in ${art.agentName}'s ${item.kind === 'intent' ? 'intentions' : 'todos'}`
                            }
                          >
                            {st === 'sending' ? <><Loader2 className="h-2.5 w-2.5 animate-spin" /> Sending</>
                              : st === 'sent' ? <><Check className="h-2.5 w-2.5" /> Recorded</>
                              : st === 'error' ? <><AlertTriangle className="h-2.5 w-2.5" /> Retry</>
                              : <><Send className="h-2.5 w-2.5" /> Send</>}
                          </button>
                        </div>
                        <div className="mt-1.5 space-y-1 text-[11px] leading-relaxed text-zinc-400">
                          {item.context && <p><span className="text-zinc-500">Context:</span> {item.context}</p>}
                          {item.task && <p><span className="text-zinc-500">Task:</span> {item.task}</p>}
                          {item.done_when && <p><span className="text-zinc-500">Done when:</span> {item.done_when}</p>}
                          {item.refs?.length > 0 && (
                            <p><span className="text-zinc-500">Refs:</span> {item.refs.join(', ')}</p>
                          )}
                        </div>
                      </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
