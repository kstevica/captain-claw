import { useState, useEffect, useCallback } from 'react'
import { X, Wand2, Loader2, AlertTriangle, Trash2, Pin, ListPlus } from 'lucide-react'
import { useAuthStore, refreshAccessToken } from '../../stores/authStore'
import { useChatStore, LANES, LANE_MAIN, laneKey } from '../../stores/chatStore'

/**
 * Turn one description of a repetitive job into a reviewed list of queue tasks.
 *
 * The plan is a PROPOSAL: nothing reaches the queue until "Send". The model
 * returns one template plus a list of ranges — never the messages themselves —
 * and the backend expands them, so every task is identical except its range
 * and no standing rule can be dropped between batch 3 and batch 19
 * (docs/queue-task-planner-plan.md).
 */

interface PlanResult {
  template: string
  batches: Record<string, unknown>[]
  messages: string[]
  rationale: string
  warnings: string[]
  facts: { table?: string; key_min?: number; key_max?: number; tables?: { name: string }[] }
}

async function fdPost<T>(path: string, body: unknown): Promise<T> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  const call = () => fetch(`/fd${path}`, {
    method: 'POST', headers, credentials: 'include', body: JSON.stringify(body),
  })
  let res = await call()
  if (res.status === 401 && authEnabled && await refreshAccessToken()) res = await call()
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* keep status */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export function QueuePlannerModal({ agentId, agentName, host, port, auth, onClose }: {
  agentId: string
  agentName: string
  host: string
  port: number
  auth: string
  onClose: () => void
}) {
  const enqueue = useChatStore((s) => s.enqueueQueueMessage)
  const sessions = useChatStore((s) => s.sessions)
  const activeLane = useChatStore((s) => s.activeLane[agentId] || LANE_MAIN)

  const [intent, setIntent] = useState('')
  const [table, setTable] = useState('')
  const [keyColumn, setKeyColumn] = useState('_id')
  const [batchSize, setBatchSize] = useState(10)
  const [maxTasks, setMaxTasks] = useState(50)
  const [lane, setLane] = useState(activeLane)
  const [newSession, setNewSession] = useState(true)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [plan, setPlan] = useState<PlanResult | null>(null)
  const [template, setTemplate] = useState('')
  const [messages, setMessages] = useState<string[]>([])
  // A message the user edited by hand is pinned: re-expanding the template
  // must not silently overwrite their correction.
  const [pinned, setPinned] = useState<Set<number>>(new Set())
  const [warnings, setWarnings] = useState<string[]>([])
  const [sent, setSent] = useState(0)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const runPlan = async () => {
    if (!intent.trim() || busy) return
    setBusy(true); setError(''); setSent(0)
    try {
      const res = await fdPost<PlanResult>('/queue/plan', {
        intent, host, port, auth, table, key_column: keyColumn,
        batch_size: batchSize, max_tasks: maxTasks,
      })
      setPlan(res)
      setTemplate(res.template)
      setMessages(res.messages)
      setWarnings(res.warnings || [])
      setPinned(new Set())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // Editing the template re-renders every message — that's what "always write
  // in English" means when you notice it on task 4 of 25. Costs no LLM call:
  // the backend expands, so there's only ever one implementation of it.
  const reExpand = useCallback(async (tpl: string) => {
    if (!plan) return
    try {
      const res = await fdPost<{ messages: string[]; warnings: string[] }>(
        '/queue/expand', { template: tpl, batches: plan.batches, max_tasks: maxTasks })
      setMessages((prev) => res.messages.map((m, i) => (pinned.has(i) ? prev[i] : m)))
      setWarnings(res.warnings || [])
    } catch { /* keep the last good expansion */ }
  }, [plan, maxTasks, pinned])

  useEffect(() => {
    if (!plan || template === plan.template) return
    const t = setTimeout(() => void reExpand(template), 400)
    return () => clearTimeout(t)
  }, [template, plan, reExpand])

  const editMessage = (i: number, text: string) => {
    setMessages((prev) => prev.map((m, idx) => (idx === i ? text : m)))
    setPinned((prev) => new Set(prev).add(i))
  }

  const dropMessage = (i: number) => {
    setMessages((prev) => prev.filter((_, idx) => idx !== i))
    setPinned((prev) => {
      const next = new Set<number>()
      prev.forEach((p) => { if (p < i) next.add(p); else if (p > i) next.add(p - 1) })
      return next
    })
  }

  const send = () => {
    const key = laneKey(agentId, lane)
    if (!sessions.get(key)) {
      // The lane has never been opened — open it so the queue has somewhere to land.
      useChatStore.getState().setActiveLane(agentId, lane)
    }
    messages.forEach((m, i) => {
      // A fresh session between tasks is what makes each one self-contained:
      // no leftover working state from the previous batch.
      if (newSession && i > 0) enqueue(key, '/new')
      enqueue(key, m)
    })
    setSent(messages.length)
  }

  const laneLabel = (l: string) => `${l} - ${agentName}`
  const pendingIn = (l: string) =>
    sessions.get(laneKey(agentId, l))?.queue.filter((q) => q.status === 'pending').length || 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="flex h-[85vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-zinc-700 bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2">
            <Wand2 className="h-4 w-4 text-violet-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Plan tasks</h2>
            <span className="text-xs text-zinc-500">— one description becomes a queue</span>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1">
          {/* ── Request ── */}
          <div className="flex w-[380px] shrink-0 flex-col gap-3 overflow-y-auto border-r border-zinc-800 p-4">
            <label className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              What needs doing
            </label>
            <textarea
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              rows={10}
              placeholder={'Describe the whole job, including every standing rule the agent must follow each time.\n\ne.g. enrich fund_portfolio in batches of 10: research company_description, stage, investment_amount… always write in English. _id and id are identical, never do +1 on the id!'}
              className="w-full resize-y rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
            />
            <p className="text-[10px] leading-relaxed text-zinc-500">
              Rules are copied into <em>every</em> task verbatim — each one runs in its own
              session and can't see the others.
            </p>

            <div className="grid grid-cols-2 gap-2">
              <Field label="Table">
                <input value={table} onChange={(e) => setTable(e.target.value)}
                  placeholder="(first table)" className={inputCls} />
              </Field>
              <Field label="Batch key">
                <input value={keyColumn} onChange={(e) => setKeyColumn(e.target.value)} className={inputCls} />
              </Field>
              <Field label="Rows per task">
                <input type="number" min={1} max={50} value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))} className={inputCls} />
              </Field>
              <Field label="Max tasks">
                <input type="number" min={1} max={200} value={maxTasks}
                  onChange={(e) => setMaxTasks(Number(e.target.value))} className={inputCls} />
              </Field>
            </div>

            <Field label="Send to lane">
              <select value={lane} onChange={(e) => setLane(e.target.value)} className={inputCls}>
                {LANES.map((l) => (
                  <option key={l} value={l}>
                    {laneLabel(l)}{pendingIn(l) ? ` · ${pendingIn(l)} pending` : ''}
                  </option>
                ))}
              </select>
            </Field>

            <label className="flex items-center gap-2 text-[11px] text-zinc-400">
              <input type="checkbox" checked={newSession}
                onChange={(e) => setNewSession(e.target.checked)} className="accent-violet-500" />
              Start a new session (<code>/new</code>) between tasks
            </label>

            <button
              onClick={runPlan}
              disabled={!intent.trim() || busy}
              className="mt-1 flex items-center justify-center gap-2 rounded-md bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
              {plan ? 'Re-plan' : 'Plan tasks'}
            </button>
            {error && (
              <p className="rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-[11px] text-red-300">
                {error}
              </p>
            )}
          </div>

          {/* ── Plan ── */}
          <div className="flex min-w-0 flex-1 flex-col">
            {!plan ? (
              <div className="flex flex-1 items-center justify-center px-8 text-center">
                <p className="max-w-md text-xs leading-relaxed text-zinc-500">
                  The plan appears here for review before anything is queued. The model reads
                  the agent's real tables and id ranges, so batches cover rows that exist —
                  it never invents a range.
                </p>
              </div>
            ) : (
              <>
                <div className="border-b border-zinc-800 px-4 py-2">
                  {plan.rationale && <p className="text-[11px] text-zinc-400">{plan.rationale}</p>}
                  {plan.facts?.table && (
                    <p className="mt-0.5 text-[10px] text-zinc-600">
                      {plan.facts.table} · {plan.facts.key_min}–{plan.facts.key_max}
                    </p>
                  )}
                  {warnings.map((w, i) => (
                    <p key={i} className="mt-1 flex items-start gap-1.5 text-[11px] text-amber-300">
                      <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />{w}
                    </p>
                  ))}
                </div>

                {/* The template: edit once, every task follows. */}
                <details className="border-b border-zinc-800 px-4 py-2" open>
                  <summary className="cursor-pointer text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                    Template — edits apply to every task
                  </summary>
                  <textarea
                    value={template}
                    onChange={(e) => setTemplate(e.target.value)}
                    rows={6}
                    className="mt-2 w-full resize-y rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 font-mono text-[11px] leading-relaxed text-zinc-300 focus:border-violet-500/60 focus:outline-none"
                  />
                </details>

                <div className="min-h-0 flex-1 overflow-y-auto px-4 py-2">
                  <ul className="flex flex-col gap-1.5">
                    {messages.map((m, i) => (
                      <li key={i} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2">
                        <div className="mb-1 flex items-center justify-between">
                          <span className="text-[10px] font-medium text-zinc-500">
                            Task {i + 1} of {messages.length}
                            {pinned.has(i) && (
                              <span className="ml-1.5 inline-flex items-center gap-0.5 text-violet-400">
                                <Pin className="h-2.5 w-2.5" /> edited — template edits skip this one
                              </span>
                            )}
                          </span>
                          <button onClick={() => dropMessage(i)}
                            className="rounded p-0.5 text-zinc-600 hover:bg-red-500/20 hover:text-red-300">
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                        <textarea
                          value={m}
                          onChange={(e) => editMessage(i, e.target.value)}
                          rows={3}
                          className="w-full resize-y rounded border border-transparent bg-transparent text-[11px] leading-relaxed text-zinc-300 hover:border-zinc-800 focus:border-violet-500/60 focus:bg-zinc-950 focus:outline-none"
                        />
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex items-center justify-between border-t border-zinc-800 px-4 py-2.5">
                  <span className="text-[11px] text-zinc-500">
                    {sent > 0
                      ? `${sent} task${sent === 1 ? '' : 's'} sent to lane ${lane}.`
                      : `${messages.length} tasks${newSession ? ` + ${Math.max(0, messages.length - 1)} /new` : ''} → lane ${lane}`}
                  </span>
                  <div className="flex items-center gap-2">
                    {sent > 0 && (
                      <button onClick={onClose}
                        className="rounded-md px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
                        Close
                      </button>
                    )}
                    <button
                      onClick={send}
                      disabled={messages.length === 0}
                      className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
                    >
                      <ListPlus className="h-3.5 w-3.5" />
                      {sent > 0 ? 'Send again' : `Send ${messages.length} to ${lane}`}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

const inputCls =
  'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</label>
      {children}
    </div>
  )
}
