import { useEffect, useState } from 'react'
import {
  Workflow,
  Plus,
  Power,
  Play,
  Edit3,
  Trash2,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  CircleDot,
  Search,
  Sparkles,
} from 'lucide-react'
import { useFlowsStore } from '../../stores/flowsStore'
import { triggerSummary, type Flow } from '../../services/flowsApi'
import * as api from '../../services/flowsApi'

function LastRunBadge({ flow }: { flow: Flow }) {
  const lr = flow.last_run
  if (!lr) return <span className="text-[11px] text-zinc-600">never run</span>
  const s = (lr.status || '').toLowerCase()
  const map: Record<string, { icon: typeof CheckCircle2; cls: string }> = {
    done: { icon: CheckCircle2, cls: 'text-emerald-400' },
    success: { icon: CheckCircle2, cls: 'text-emerald-400' },
    error: { icon: XCircle, cls: 'text-red-400' },
    failed: { icon: XCircle, cls: 'text-red-400' },
    running: { icon: Loader2, cls: 'text-violet-400 animate-spin' },
    parked: { icon: Clock, cls: 'text-amber-400' },
  }
  const m = map[s] || { icon: CircleDot, cls: 'text-zinc-500' }
  const Icon = m.icon
  return (
    <span className="flex items-center gap-1 text-[11px] text-zinc-500">
      <Icon className={`h-3 w-3 ${m.cls}`} /> {lr.status}
    </span>
  )
}

export function FlowsPanel() {
  const {
    flows,
    loading,
    error,
    fetchFlows,
    openNew,
    openEdit,
    openRunLog,
    toggleEnabled,
    removeFlow,
    runNow,
  } = useFlowsStore()
  const [query, setQuery] = useState('')
  const [running, setRunning] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [tab, setTab] = useState<'persistent' | 'synthesized'>('persistent')
  const [scratch, setScratch] = useState<Flow[]>([])

  const refreshScratch = () => { api.listScratchFlows().then(setScratch).catch(() => setScratch([])) }
  useEffect(() => { fetchFlows() }, [fetchFlows])
  useEffect(() => { refreshScratch() }, [])

  const shown = flows.filter((f) => {
    if (!query.trim()) return true
    const q = query.toLowerCase()
    return (
      f.name.toLowerCase().includes(q) ||
      (f.description || '').toLowerCase().includes(q) ||
      triggerSummary(f.trigger).toLowerCase().includes(q)
    )
  })

  const handleRun = async (id: string) => {
    setRunning(id)
    try {
      const runId = await runNow(id)
      if (runId) openRunLog(runId)
    } catch {
      // surfaced by store
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-6">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-semibold text-zinc-100">
              <Workflow className="h-5 w-5 text-violet-400" />
              Flows
            </h2>
            <p className="mt-1 text-sm text-zinc-500">
              Declarative, multi-step automations — reliable steps with agent judgment where it matters.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchFlows}
              className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            <button
              onClick={openNew}
              className="flex items-center gap-1.5 rounded-xl bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 transition-colors"
            >
              <Plus className="h-4 w-4" /> New Flow
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-4 flex items-center gap-1 rounded-xl border border-zinc-800 p-1 text-sm">
          <button
            onClick={() => setTab('persistent')}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-colors ${tab === 'persistent' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <Workflow className="h-3.5 w-3.5" /> Persistent
            <span className="text-zinc-600">{flows.length}</span>
          </button>
          <button
            onClick={() => { setTab('synthesized'); refreshScratch() }}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-colors ${tab === 'synthesized' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <Sparkles className="h-3.5 w-3.5 text-violet-400" /> Synthesized
            <span className="text-zinc-600">{scratch.length}</span>
          </button>
        </div>

        {tab === 'synthesized' ? (
          <div className="space-y-4">
            <SynthesizeComposer onChanged={() => { refreshScratch(); fetchFlows() }} />
            <ScratchList
              flows={scratch}
              onChanged={() => { refreshScratch(); fetchFlows() }}
            />
          </div>
        ) : (
        <>
        {/* Search */}
        <div className="mb-4 flex items-center gap-2 rounded-xl border border-zinc-700/50 bg-zinc-900/50 px-3 py-2">
          <Search className="h-4 w-4 text-zinc-600" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search flows…"
            className="flex-1 bg-transparent text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none"
          />
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/[0.06] px-3 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* List */}
        {loading && flows.length === 0 ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
            <span className="ml-2 text-sm text-zinc-500">Loading flows…</span>
          </div>
        ) : shown.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-zinc-800 px-8 py-16 text-center">
            <Workflow className="mx-auto h-10 w-10 text-zinc-700" />
            <p className="mt-3 text-sm text-zinc-500">{query ? 'No flows match your search' : 'No flows yet'}</p>
            {!query && (
              <p className="mt-1 text-xs text-zinc-600">Click + New Flow to create your first automation.</p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {shown.map((flow) => (
              <div
                key={flow.id}
                className={`rounded-2xl border transition-all ${
                  flow.enabled
                    ? 'border-zinc-800 bg-zinc-900/30 hover:border-zinc-700'
                    : 'border-zinc-800/50 bg-zinc-950/30 opacity-70 hover:opacity-90'
                }`}
              >
                <div className="flex items-center gap-3 px-4 py-3.5">
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                      flow.enabled ? 'bg-violet-500/10 text-violet-400' : 'bg-zinc-800 text-zinc-600'
                    }`}
                  >
                    <Workflow className="h-4 w-4" />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-zinc-200">{flow.name}</span>
                      <span className="shrink-0 rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">
                        {triggerSummary(flow.trigger)}
                      </span>
                    </div>
                    <div className="mt-0.5 flex items-center gap-3">
                      {flow.description && (
                        <span className="truncate text-[11px] text-zinc-500">{flow.description}</span>
                      )}
                      <span className="shrink-0 text-[11px] text-zinc-600">{flow.steps.length} steps</span>
                      <LastRunBadge flow={flow} />
                      {flow.last_run && (
                        <button
                          onClick={() => openRunLog(flow.last_run!.id)}
                          className="shrink-0 text-[11px] text-violet-400/80 hover:text-violet-300"
                        >
                          view log
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      onClick={() => toggleEnabled(flow.id, !flow.enabled)}
                      className={`rounded-lg p-2 transition-colors ${
                        flow.enabled
                          ? 'text-emerald-400 hover:bg-emerald-500/10'
                          : 'text-zinc-600 hover:bg-zinc-800'
                      }`}
                      title={flow.enabled ? 'Disable' : 'Enable'}
                    >
                      <Power className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleRun(flow.id)}
                      disabled={running === flow.id}
                      className="rounded-lg p-2 text-zinc-400 hover:bg-violet-500/10 hover:text-violet-400 disabled:opacity-40 transition-colors"
                      title="Run now"
                    >
                      {running === flow.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                    </button>
                    <button
                      onClick={() => openEdit(flow.id)}
                      className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
                      title="Edit"
                    >
                      <Edit3 className="h-3.5 w-3.5" />
                    </button>
                    {confirmDelete === flow.id ? (
                      <button
                        onClick={() => { removeFlow(flow.id); setConfirmDelete(null) }}
                        className="rounded-lg bg-red-500/15 px-2 py-1.5 text-[11px] font-medium text-red-300 hover:bg-red-500/25"
                        onMouseLeave={() => setConfirmDelete(null)}
                      >
                        Confirm
                      </button>
                    ) : (
                      <button
                        onClick={() => setConfirmDelete(flow.id)}
                        className="rounded-lg p-2 text-zinc-400 hover:bg-red-500/10 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        </>
        )}
      </div>
    </div>
  )
}

function SynthesizeComposer({ onChanged }: { onChanged: () => void }) {
  const [goal, setGoal] = useState('')
  const [agent, setAgent] = useState('')
  const [run, setRun] = useState(false)
  const [fleet, setFleet] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err' | 'info'; text: string } | null>(null)

  useEffect(() => {
    api.listFleet().then((a) => setFleet(a.map((x) => x.name).filter(Boolean))).catch(() => {})
  }, [])

  const go = async () => {
    if (!goal.trim()) return
    setBusy(true)
    setMsg({ kind: 'info', text: 'Asking the model to build it…' })
    try {
      const r = await api.synthesizeFlow(goal, agent, run)
      if (r.ok) {
        const ran = run ? ` — ran (${r.status})` : ''
        setMsg({ kind: 'ok', text: `${r.reused ? 'Reused' : 'Created'} “${r.name}”${ran}.` })
        setGoal('')
        onChanged()
      } else {
        setMsg({ kind: 'err', text: r.error || 'synthesis failed' })
      }
    } catch (e) {
      setMsg({ kind: 'err', text: String(e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-4">
      <label className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
        <Sparkles className="h-3 w-3 text-violet-400" /> Synthesize a flow
      </label>
      <textarea
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        rows={3}
        placeholder="Describe a repeatable goal, e.g. “When I say standup, ask what I did yesterday, then write a one-line status.”"
        className="w-full resize-y rounded-lg border border-zinc-700/50 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/20"
      />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          onClick={go}
          disabled={busy || !goal.trim()}
          className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40 transition-colors"
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />} Synthesize
        </button>
        <span className="text-[10px] uppercase tracking-wide text-zinc-500">using</span>
        <select
          value={agent}
          onChange={(e) => setAgent(e.target.value)}
          className="rounded-lg border border-zinc-700/50 bg-zinc-950 px-2 py-1 text-xs text-zinc-200"
          title="Which agent's model writes the flow"
        >
          <option value="">Auto</option>
          {fleet.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <label className="flex cursor-pointer items-center gap-1.5 text-xs text-zinc-400">
          <input type="checkbox" checked={run} onChange={(e) => setRun(e.target.checked)} className="accent-violet-500" /> run now
        </label>
        {msg && (
          <span className={msg.kind === 'ok' ? 'text-xs text-emerald-400' : msg.kind === 'err' ? 'text-xs text-rose-400' : 'text-xs text-zinc-400'}>
            {msg.text}
          </span>
        )}
      </div>
      <p className="mt-2 text-[10px] text-zinc-600">
        Creates a call-only flow in the scratch space (validated by the real parser). Run it a few times — it earns a ⭐ before you promote it.
      </p>
    </div>
  )
}

function ScratchList({ flows, onChanged }: { flows: Flow[]; onChanged: () => void }) {
  const editScratch = useFlowsStore((s) => s.editScratch)
  const [busy, setBusy] = useState<string | null>(null)

  const promote = async (f: Flow) => {
    setBusy(f.id)
    try {
      await api.promoteFlow(f.id)
      onChanged()
    } finally {
      setBusy(null)
    }
  }
  const remove = async (f: Flow) => {
    setBusy(f.id)
    try {
      await api.deleteFlow(f.id)
      onChanged()
    } finally {
      setBusy(null)
    }
  }

  if (flows.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-zinc-800 px-8 py-16 text-center">
        <Sparkles className="mx-auto h-10 w-10 text-zinc-700" />
        <p className="mt-3 text-sm text-zinc-500">No synthesized flows yet</p>
        <p className="mt-1 text-xs text-zinc-600">
          Ask an agent to “synthesize a flow that …”. Flows it authors land here — review, edit, then promote the good ones.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="divide-y divide-zinc-800/60 rounded-xl border border-zinc-800">
          {flows.map((f) => (
            <div key={f.id} className="flex items-center gap-3 px-3.5 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm text-zinc-200">{f.name}</span>
                  {f.state === 'candidate' && (
                    <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300" title="Proven — ready to promote">⭐ ready</span>
                  )}
                  {f.state === 'quarantined' && (
                    <span className="rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300" title="Failed repeatedly — will be discarded">⚠️ failing</span>
                  )}
                  <span className="rounded bg-violet-500/15 px-1.5 py-0.5 text-[10px] text-violet-300">agent</span>
                  {(f.success_count || f.fail_count) ? (
                    <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">{f.success_count ?? 0}✓ {f.fail_count ?? 0}✗</span>
                  ) : null}
                </div>
                <div className="text-[11px] text-zinc-600">
                  {f.author ? `by ${f.author}` : 'synthesized'}{f.description ? ` — ${f.description}` : ''}
                </div>
              </div>
              <button
                onClick={() => editScratch(f)}
                className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                title="Open in the builder / code view"
              >
                <Edit3 className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => promote(f)}
                disabled={busy === f.id}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium disabled:opacity-40 ${f.state === 'candidate' ? 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30' : 'bg-emerald-500/10 text-emerald-300/80 hover:bg-emerald-500/20'}`}
                title="Promote into your permanent flows (call-only; enable it there to make it user-facing)"
              >
                Promote
              </button>
              <button
                onClick={() => remove(f)}
                disabled={busy === f.id}
                className="rounded-lg p-2 text-zinc-400 hover:bg-red-500/10 hover:text-red-400 disabled:opacity-40"
                title="Discard"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
      </div>
    </div>
  )
}
