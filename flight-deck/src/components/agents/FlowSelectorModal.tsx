import { useEffect, useMemo, useState } from 'react'
import { Workflow, X, Play, Search, Loader2 } from 'lucide-react'
import { useFlowsStore } from '../../stores/flowsStore'
import { useChatStore } from '../../stores/chatStore'

interface FlowSelectorModalProps {
  /** The chat session to start flows from / note into. */
  containerId: string
  onClose: () => void
}

/**
 * Flow selector for the chat: lists every flow (name + description, including
 * disabled ones), lets the user toggle each flow's trigger on/off, and start a
 * flow now. Starting runs it in Flight Deck's background (POST /flows/{id}/run);
 * its output is delivered to the flow's configured channel — web/same flows land
 * back here in the chat. A local note marks the start.
 */
export function FlowSelectorModal({ containerId, onClose }: FlowSelectorModalProps) {
  const flows = useFlowsStore((s) => s.flows)
  const loading = useFlowsStore((s) => s.loading)
  const fetchFlows = useFlowsStore((s) => s.fetchFlows)
  const toggleEnabled = useFlowsStore((s) => s.toggleEnabled)
  const sendMessage = useChatStore((s) => s.sendMessage)

  const [q, setQ] = useState('')

  useEffect(() => { fetchFlows() }, [fetchFlows])

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase()
    const list = t
      ? flows.filter((f) => `${f.name} ${f.description || ''}`.toLowerCase().includes(t))
      : flows
    // Enabled first, then alphabetical — so the active flows are up top.
    return [...list].sort(
      (a, b) => Number(b.enabled) - Number(a.enabled) || (a.name || '').localeCompare(b.name || ''),
    )
  }, [flows, q])

  const start = (name: string) => {
    // Route the start THROUGH the chat ("/flow run <name>") so Flight Deck
    // binds the run to THIS web channel — /flow status|stop and the input
    // step's resume then all target it, and the flow delivers its intro/output
    // right here. (Works for disabled flows too — that only gates the trigger.)
    sendMessage(containerId, `/flow run ${name}`)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-[620px] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <Workflow className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold">Flows</h2>
            <span className="text-xs text-zinc-600">{flows.length}</span>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Search */}
        <div className="border-b border-zinc-800 px-5 py-2.5">
          <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5">
            <Search className="h-3.5 w-3.5 text-zinc-600" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search flows…"
              autoFocus
              className="w-full bg-transparent text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none"
            />
          </div>
        </div>

        {/* List */}
        <div className="flex-1 space-y-1.5 overflow-y-auto px-3 py-3">
          {loading && flows.length === 0 && (
            <div className="flex items-center justify-center py-12 text-zinc-600">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          )}
          {!loading && filtered.length === 0 && (
            <div className="py-12 text-center text-sm text-zinc-600">
              {q ? 'No flows match your search.' : 'No flows yet — build one in the Flows tab.'}
            </div>
          )}
          {filtered.map((f) => (
            <div
              key={f.id}
              className={`rounded-lg border px-3.5 py-3 transition-colors ${
                f.enabled ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-800/60 bg-zinc-900/20'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`truncate text-sm font-medium ${f.enabled ? 'text-zinc-100' : 'text-zinc-400'}`}>
                      {f.name || '(untitled flow)'}
                    </span>
                    {!f.enabled && (
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500">disabled</span>
                    )}
                  </div>
                  {f.description && <div className="mt-0.5 line-clamp-2 text-xs text-zinc-500">{f.description}</div>}
                  <div className="mt-1.5 flex items-center gap-2 text-[10px] text-zinc-600">
                    {f.trigger?.channel && (
                      <span className="rounded bg-zinc-800/80 px-1.5 py-0.5 text-zinc-500">{f.trigger.channel}</span>
                    )}
                    <span>{f.steps?.length ?? 0} step{f.steps?.length === 1 ? '' : 's'}</span>
                    {typeof f.priority === 'number' && <span>· p{f.priority}</span>}
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  {/* Enable / disable the flow's trigger */}
                  <button
                    onClick={() => toggleEnabled(f.id, !f.enabled)}
                    role="switch"
                    aria-checked={f.enabled}
                    title={f.enabled ? 'Enabled — trigger fires. Click to disable.' : 'Disabled — trigger off. Click to enable.'}
                    className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
                      f.enabled ? 'bg-emerald-500/80' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${
                        f.enabled ? 'left-[18px]' : 'left-0.5'
                      }`}
                    />
                  </button>

                  {/* Start it now (runs in this chat) */}
                  <button
                    onClick={() => start(f.name || 'flow')}
                    title="Start this flow now — runs in this chat"
                    className="flex items-center gap-1 rounded-md bg-violet-600/20 px-2.5 py-1.5 text-xs font-medium text-violet-300 hover:bg-violet-600/30"
                  >
                    <Play className="h-3 w-3" />
                    Start
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="border-t border-zinc-800 px-5 py-2.5 text-[11px] leading-relaxed text-zinc-600">
          Toggle to enable/disable a flow's trigger. <span className="text-zinc-500">Start</span> runs it in this chat —
          control it with <code className="text-zinc-500">/flow status</code> · <code className="text-zinc-500">/flow stop</code>.
        </div>
      </div>
    </div>
  )
}
