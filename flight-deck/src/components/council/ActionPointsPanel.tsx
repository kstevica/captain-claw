import { useState, useEffect } from 'react'
import { RefreshCw, Loader2, ListChecks, Send, CheckSquare, Sparkles, Check, AlertTriangle } from 'lucide-react'
import type { CouncilArtifact, ActionPoint } from '../../stores/councilStore'
import { Button } from '../console/parts'
import { AgentAvatar, PanelHead } from './councilLook'
import { SMALL_BTN } from './councilStyles'

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
    <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40">
      <PanelHead
        icon={ListChecks}
        tone="emerald"
        title="Action Points"
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        action={
          <Button
            onClick={(e) => { e.stopPropagation(); onGenerate() }}
            disabled={generating}
            icon={generating ? Loader2 : RefreshCw}
            className={`${SMALL_BTN} ${generating ? '[&>svg]:animate-spin' : ''}`}
          >
            {generating ? 'Extracting...' : artifacts.length > 0 ? 'Regenerate' : 'Extract'}
          </Button>
        }
      />

      {!collapsed && (
        <div className="space-y-3 border-t border-zinc-800/80 px-4 pb-4 pt-3">
          {artifacts.length === 0 && !generating && (
            <p className="text-xs leading-relaxed text-zinc-500">
              No action points yet. Each agent extracts its own outstanding next steps from the discussion (scoped to its part — no full-transcript dump).
            </p>
          )}
          {generating && artifacts.length === 0 && (
            <div className="flex items-center gap-2 py-1 text-xs text-zinc-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Collecting action points from agents...
            </div>
          )}

          <div className="max-h-[60vh] space-y-4 overflow-y-auto pr-1">
            {artifacts.map(art => {
              const items = parse(art.content)
              if (items.length === 0) return null
              return (
                <div key={`${art.agentId}-${art.id}`}>
                  <div className="mb-2 flex items-center gap-2">
                    <AgentAvatar name={art.agentName} id={art.agentId} size="xs" />
                    <span className="text-xs font-semibold text-zinc-200">{art.agentName}</span>
                    <span className="text-[11px] tabular-nums text-zinc-500">{items.length}</span>
                  </div>
                  <div className="space-y-2">
                    {items.map((item, i) => {
                      const key = `${art.id}-${i}`
                      // item.sent is persisted in the artifact, so a recorded
                      // item stays "Recorded" across reloads (and can't re-send).
                      const st = sendState[key] || (item.sent ? 'sent' : 'idle')
                      return (
                      <div key={i} className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-3">
                        <div className="flex items-start gap-2.5">
                          <span className={`mt-px inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${
                            item.kind === 'intent'
                              ? 'bg-violet-500/10 text-violet-700 dark:text-violet-300'
                              : 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                          }`}>
                            {item.kind === 'intent' ? <Sparkles className="h-3 w-3" /> : <CheckSquare className="h-3 w-3" />}
                            {item.kind}
                          </span>
                          <span className="min-w-0 flex-1 text-sm font-semibold leading-snug text-zinc-100">{item.title}</span>
                          <button
                            onClick={() => handleSend(art.agentId, item, key)}
                            disabled={busy || st === 'sending' || st === 'sent'}
                            className={`flex shrink-0 items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-medium transition-colors disabled:cursor-not-allowed ${
                              st === 'sent'
                                ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                                : st === 'error'
                                  ? 'border-red-500/40 text-red-700 hover:bg-red-500/10 dark:text-red-400'
                                  : 'border-zinc-700 bg-zinc-900/60 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800 disabled:opacity-50'
                            }`}
                            title={
                              st === 'sent' ? `Recorded in ${art.agentName}'s ${item.kind === 'intent' ? 'intentions' : 'todos'}`
                                : st === 'error' ? 'Failed to record — click to retry'
                                : `Record this in ${art.agentName}'s ${item.kind === 'intent' ? 'intentions' : 'todos'}`
                            }
                          >
                            {st === 'sending' ? <><Loader2 className="h-3 w-3 animate-spin" /> Sending</>
                              : st === 'sent' ? <><Check className="h-3 w-3" /> Recorded</>
                              : st === 'error' ? <><AlertTriangle className="h-3 w-3" /> Retry</>
                              : <><Send className="h-3 w-3" /> Send</>}
                          </button>
                        </div>
                        <div className="mt-2 space-y-1 text-xs leading-relaxed text-zinc-400">
                          {item.context && <p><span className="font-medium text-zinc-500">Context:</span> {item.context}</p>}
                          {item.task && <p><span className="font-medium text-zinc-500">Task:</span> {item.task}</p>}
                          {item.done_when && <p><span className="font-medium text-zinc-500">Done when:</span> {item.done_when}</p>}
                          {item.refs?.length > 0 && (
                            <p><span className="font-medium text-zinc-500">Refs:</span> {item.refs.join(', ')}</p>
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
        </div>
      )}
    </section>
  )
}
