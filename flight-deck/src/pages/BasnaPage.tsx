import { useEffect, useState } from 'react'
import {
  Network, Play, Sparkles, Plus, Trash2, ThumbsUp, ThumbsDown,
  ChevronDown, Loader2, Check, X,
} from 'lucide-react'
import { useBasnaStore, type BasnaSession, type BasnaRun } from '../stores/basnaStore'

const DIFFICULTY_COLOR: Record<string, string> = {
  trivial: 'text-emerald-700 dark:text-emerald-300',
  moderate: 'text-amber-700 dark:text-amber-300',
  hard: 'text-rose-700 dark:text-rose-300',
}

function Badge({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

function WeightBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <div className="h-full rounded-full bg-sky-500" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

function SessionCard({ s, active, onOpen, onDelete }: {
  s: BasnaSession; active: boolean; onOpen: () => void; onDelete: () => void
}) {
  return (
    <button
      onClick={onOpen}
      className={`group w-full rounded-lg border p-2.5 text-left transition-colors ${
        active ? 'border-sky-600/60 bg-sky-950/30' : 'border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/50'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="line-clamp-2 text-xs font-medium text-zinc-200">{s.intent || '(untitled)'}</p>
        <span
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="shrink-0 rounded p-0.5 text-zinc-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {s.domain && <Badge className="text-zinc-300">{s.domain}</Badge>}
        {s.difficulty && <Badge className={DIFFICULTY_COLOR[s.difficulty] || 'text-zinc-300'}>{s.difficulty}</Badge>}
        <Badge className="text-zinc-400">{s.status}</Badge>
      </div>
    </button>
  )
}

function AgentRow({ run, onFeedback }: { run: BasnaRun; onFeedback: (success: boolean) => void }) {
  const scored = run.success !== null
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-zinc-200">{run.role || run.archetype_id}</span>
          <Badge className="text-sky-700 dark:text-sky-300">{run.tier}</Badge>
          {scored && (
            run.success === 1
              ? <Badge className="text-emerald-700 dark:text-emerald-300">success</Badge>
              : <Badge className="text-rose-700 dark:text-rose-300">fail</Badge>
          )}
        </div>
        <span className="text-[11px] text-zinc-500">{(run.latency_ms / 1000).toFixed(1)}s</span>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span className="w-16 shrink-0 text-[11px] text-zinc-500">weight {run.weight_at_run.toFixed(2)}</span>
        <WeightBar value={run.weight_at_run} />
      </div>
      {run.output && (
        <p className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap text-xs text-zinc-400">{run.output}</p>
      )}
      <div className="mt-2 flex items-center gap-2">
        <span className="text-[11px] text-zinc-500">Was this contribution good?</span>
        <button
          onClick={() => onFeedback(true)}
          className={`rounded p-1 transition-colors ${run.success === 1 ? 'text-emerald-400' : 'text-zinc-500 hover:text-emerald-400'}`}
          title="Mark as good"
        >
          <ThumbsUp className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => onFeedback(false)}
          className={`rounded p-1 transition-colors ${run.success === 0 ? 'text-rose-400' : 'text-zinc-500 hover:text-rose-400'}`}
          title="Mark as poor"
        >
          <ThumbsDown className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

export function BasnaPage() {
  const {
    sessions, activeSession, routePlan, runs, lastExecute,
    routing, executing, error,
    apiKey, maxAgents, setApiKey, setMaxAgents,
    loadSessions, selectSession, newSession, route, execute, sendFeedback, deleteSession,
  } = useBasnaStore()

  const [intent, setIntent] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  useEffect(() => { loadSessions() }, [loadSessions])
  useEffect(() => { setIntent(activeSession?.intent || '') }, [activeSession?.id, activeSession?.intent])

  const canRoute = intent.trim().length > 0 && !routing
  const canRun = !!routePlan && !!activeSession && !executing
  const truth = lastExecute?.truth ?? activeSession?.truth ?? ''
  const confidence = lastExecute?.confidence ?? activeSession?.confidence ?? 0

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex shrink-0 items-center gap-2 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <Network className="h-5 w-5 text-sky-600 dark:text-sky-400" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Basna</h1>
          <p className="text-[11px] text-zinc-500">Route → spawn the minimal team → merge by reliability</p>
        </div>
        <button
          onClick={() => { newSession(); setIntent('') }}
          className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
        >
          <Plus className="h-3.5 w-3.5" /> New
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Session list */}
        <div className="w-64 shrink-0 space-y-2 overflow-auto border-r border-zinc-800 p-3">
          {sessions.length === 0 && <p className="px-1 text-xs text-zinc-600">No runs yet.</p>}
          {sessions.map((s) => (
            <SessionCard
              key={s.id}
              s={s}
              active={activeSession?.id === s.id}
              onOpen={() => selectSession(s.id)}
              onDelete={() => deleteSession(s.id)}
            />
          ))}
        </div>

        {/* Detail */}
        <div className="flex-1 overflow-auto p-4 md:p-6">
          <div className="mx-auto max-w-3xl space-y-5">
            {/* Intent + controls */}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
              <label className="mb-1.5 block text-xs font-medium text-zinc-400">Task / intent</label>
              <textarea
                value={intent}
                onChange={(e) => setIntent(e.target.value)}
                rows={3}
                placeholder="Describe the task. The router picks the smallest team that can answer it well."
                className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 p-2.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
              />
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  Max agents
                  <input
                    type="number" min={1} max={10} value={maxAgents}
                    onChange={(e) => setMaxAgents(Number(e.target.value))}
                    className="w-16 rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-zinc-200 focus:border-sky-600 focus:outline-none"
                  />
                </label>
                <button
                  onClick={() => setShowAdvanced((v) => !v)}
                  className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300"
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                  Advanced
                </button>
                <div className="ml-auto flex items-center gap-2">
                  <button
                    onClick={() => route(intent)}
                    disabled={!canRoute}
                    className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  >
                    {routing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    Route
                  </button>
                  <button
                    onClick={() => execute()}
                    disabled={!canRun}
                    className="flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
                  >
                    {executing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                    Run ensemble
                  </button>
                </div>
              </div>
              {showAdvanced && (
                <div className="mt-3 border-t border-zinc-800 pt-3">
                  <label className="mb-1.5 block text-xs font-medium text-zinc-400">
                    Anthropic API key (optional — falls back to the server's env key)
                  </label>
                  <input
                    type="password" value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-ant-…"
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
                  />
                </div>
              )}
              {error && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-rose-900/50 bg-rose-950/30 p-2.5 text-xs text-rose-300">
                  <X className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {error}
                </div>
              )}
            </div>

            {/* Route plan */}
            {routePlan && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Route plan</span>
                  <Badge className="text-zinc-300">{routePlan.domain}</Badge>
                  <Badge className={DIFFICULTY_COLOR[routePlan.difficulty] || 'text-zinc-300'}>{routePlan.difficulty}</Badge>
                  <Badge className="text-sky-700 dark:text-sky-300">{routePlan.merge_kind}</Badge>
                  {routePlan.source && <Badge className="text-zinc-500">{routePlan.source}</Badge>}
                  <span className="ml-auto text-[11px] text-zinc-600">{routePlan.selected.length} agent(s)</span>
                </div>
                {routePlan.rationale && <p className="mb-3 text-xs text-zinc-400">{routePlan.rationale}</p>}
                <div className="space-y-2">
                  {routePlan.selected.map((sel) => (
                    <div key={sel.archetype_id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium text-zinc-200">{sel.role || sel.archetype_id}</span>
                        <Badge className="text-sky-700 dark:text-sky-300">{sel.tier}</Badge>
                      </div>
                      {sel.why && <p className="mt-1 text-xs text-zinc-500">{sel.why}</p>}
                      <div className="mt-2 flex items-center gap-2">
                        <span className="w-20 shrink-0 text-[11px] text-zinc-500">prior {sel.prior_weight.toFixed(2)}</span>
                        <WeightBar value={sel.prior_weight} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Truth */}
            {truth && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Compiled truth</span>
                  {lastExecute?.method && <Badge className="text-sky-700 dark:text-sky-300">{lastExecute.method}</Badge>}
                  <span className="ml-auto flex items-center gap-2 text-[11px] text-zinc-500">
                    confidence {(confidence * 100).toFixed(0)}%
                    <span className="h-1.5 w-20 overflow-hidden rounded-full bg-zinc-800">
                      <span className="block h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(confidence * 100)}%` }} />
                    </span>
                  </span>
                </div>
                <p className="whitespace-pre-wrap text-sm text-zinc-200">{truth}</p>
              </div>
            )}

            {/* Agent runs */}
            {runs.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Agents ({runs.length})</span>
                {runs.map((run) => (
                  <AgentRow key={run.id} run={run} onFeedback={(s) => sendFeedback(run.id, s)} />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
