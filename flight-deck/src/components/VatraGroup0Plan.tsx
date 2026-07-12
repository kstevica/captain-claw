import { useEffect, useMemo, useState } from 'react'
import { ClipboardList, Pencil, Check } from 'lucide-react'
import type { Group0Plan, Group0Agent, Group0Question, VatraSubtask } from '../stores/basnaStore'

// The Group 0 gate: the Long Horizon Planner's per-agent coordination plan, presented
// for review and editing before any Group-A worker runs. Master–detail — the agent
// list on the left, the selected agent's fields (mandate / produces / consumes-from /
// hand-off) stacked vertically with room to read on the right. Controlled: every edit
// calls onChange with the next plan; the parent owns the draft + the Execute/Cancel CTAs.

// Theme-aware violet accent (matches VatraTeamPlan).
const PANEL = 'rounded-lg border border-violet-300/70 bg-violet-50/60 p-4 dark:border-violet-800/40 dark:bg-violet-950/10'
const GROUPS = ['A', 'B', 'C', 'D'] as const

function Chip({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

function TabBtn({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`-mb-px flex items-center rounded-t-md border-b-2 px-3 py-1.5 text-xs font-medium transition-colors ${
        on ? 'border-violet-500 text-violet-700 dark:text-violet-300'
           : 'border-transparent text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {children}
    </button>
  )
}

function Field({ label, value, onChange, rows = 3, placeholder, editing }: {
  label: string
  value: string
  onChange: (v: string) => void
  rows?: number
  placeholder?: string
  editing: boolean
}) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">{label}</div>
      {editing ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={rows}
          placeholder={placeholder}
          className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2.5 py-2 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
        />
      ) : (
        <div className="whitespace-pre-wrap px-0.5 text-xs leading-relaxed text-zinc-300">
          {value.trim() || <span className="italic text-zinc-600">{placeholder || '—'}</span>}
        </div>
      )}
    </div>
  )
}

export function VatraGroup0Plan({
  plan, subtasks, onChange, dirty = false,
}: {
  plan: Group0Plan
  subtasks?: VatraSubtask[]
  onChange: (plan: Group0Plan) => void
  dirty?: boolean
}) {
  const agents = plan.agents || []
  // Select by subtask_id (not index) so re-sorting the list on a group change keeps
  // the same agent selected.
  const [selId, setSelId] = useState('')
  const a: Group0Agent | undefined = agents.find((x) => x.subtask_id === selId) || agents[0]
  useEffect(() => {
    if (agents.length && !agents.some((x) => x.subtask_id === selId)) setSelId(agents[0].subtask_id)
  }, [agents, selId])

  // Left-list order: grouped by phase (A→B→C→D, ungrouped last), stable within a group.
  // Re-sorts live as the user re-groups an agent.
  const ordered = useMemo(() => {
    const rank = (g: string) => { const k = GROUPS.indexOf(g as (typeof GROUPS)[number]); return k < 0 ? GROUPS.length : k }
    return agents
      .map((x, idx) => ({ x, idx }))
      .sort((p, q) => rank(p.x.group) - rank(q.x.group) || p.idx - q.idx)
      .map((o) => o.x)
  }, [agents])

  const titleFor = (id: string) => (subtasks || []).find((s) => s.id === id)?.title || id
  const patch = (p: Partial<Group0Agent>) => {
    if (!a) return
    onChange({ ...plan, agents: agents.map((x) => (x.subtask_id === a.subtask_id ? { ...x, ...p } : x)) })
  }
  const toggleConsume = (cid: string) => {
    if (!a) return
    const has = (a.consumes_from || []).includes(cid)
    patch({ consumes_from: has ? a.consumes_from.filter((x) => x !== cid) : [...(a.consumes_from || []), cid] })
  }
  const teammates = a ? (subtasks || []).filter((s) => s.id !== a.subtask_id) : []

  // ── Clarifying questions (dynamic form) ──────────────────────────────
  const questions = plan.questions || []
  const [tab, setTab] = useState<'plan' | 'questions'>('plan')
  const [selQId, setSelQId] = useState('')
  const q: Group0Question | undefined = questions.find((x) => x.id === selQId) || questions[0]
  useEffect(() => {
    if (questions.length && !questions.some((x) => x.id === selQId)) setSelQId(questions[0].id)
  }, [questions, selQId])
  useEffect(() => { if (!questions.length && tab === 'questions') setTab('plan') }, [questions.length, tab])

  const isAnswered = (x: Group0Question) => (x.selected?.length || 0) > 0 || !!(x.other || '').trim()
  const unanswered = questions.filter((x) => !isAnswered(x)).length
  const patchQ = (p: Partial<Group0Question>) => {
    if (!q) return
    onChange({ ...plan, questions: questions.map((x) => (x.id === q.id ? { ...x, ...p } : x)) })
  }
  const toggleOption = (opt: string) => {
    if (!q) return
    const has = (q.selected || []).includes(opt)
    if (q.multi) patchQ({ selected: has ? q.selected.filter((o) => o !== opt) : [...(q.selected || []), opt] })
    else patchQ({ selected: has ? [] : [opt] })   // single-select: replace / toggle off
  }

  // Read by default; the Edit toggle turns the plan's text fields into editors.
  const [editing, setEditing] = useState(false)

  return (
    <div className={PANEL}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ClipboardList className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Group 0 — coordination plan</span>
        <Chip className="text-violet-600 dark:text-violet-300">Long Horizon Planner</Chip>
        <Chip className="text-amber-600 dark:text-amber-300">review before run</Chip>
        {tab === 'plan' && (
          <button
            onClick={() => setEditing((v) => !v)}
            title={editing ? 'Done editing — show the plan as text' : 'Edit the plan text'}
            className={`ml-auto flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${
              editing ? 'border-violet-500 bg-violet-500/10 text-violet-700 dark:text-violet-300'
                      : 'border-zinc-700 text-zinc-400 hover:text-zinc-200'
            }`}
          >
            {editing ? <><Check className="h-3.5 w-3.5" /> Done</> : <><Pencil className="h-3.5 w-3.5" /> Edit</>}
          </button>
        )}
      </div>
      <p className="mb-3 text-[11px] leading-snug text-zinc-500">
        The Long Horizon Planner drafted how the team will operate. Review and edit the coordination
        plan{questions.length ? ', and answer its clarifying questions,' : ''} then Execute above to
        run Group A (or Cancel to discard; nothing runs).
      </p>

      {dirty && (
        <div className="mb-3 rounded-md border border-amber-400/60 bg-amber-500/10 p-2.5 text-[11px] leading-snug text-amber-700 dark:border-amber-500/40 dark:text-amber-300">
          You changed the grouping or answered a question. The mandates, dependencies, and hand-offs
          were written for the <span className="font-semibold">previous</span> plan — click
          <span className="font-semibold"> Re-plan</span> above to regenerate the coordination, or
          Execute to run as-is.
        </div>
      )}

      {questions.length > 0 && (
        <div className="mb-3 flex items-center gap-1 border-b border-zinc-800">
          <TabBtn on={tab === 'plan'} onClick={() => setTab('plan')}>Coordination plan</TabBtn>
          <TabBtn on={tab === 'questions'} onClick={() => setTab('questions')}>
            Questions
            <span className={`ml-1.5 rounded-full px-1.5 py-0.5 text-[9px] font-semibold ${
              unanswered ? 'bg-amber-500/20 text-amber-700 dark:text-amber-300'
                         : 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300'}`}>
              {unanswered ? `${unanswered} to answer` : 'answered'}
            </span>
          </TabBtn>
        </div>
      )}

      {tab === 'plan' ? (
      <>
      <div className="mb-3">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-300">Coordination overview</div>
        {editing ? (
          <textarea
            value={plan.overview}
            onChange={(e) => onChange({ ...plan, overview: e.target.value })}
            rows={3}
            placeholder="How the team works together end to end…"
            className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2.5 py-2 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
          />
        ) : (
          <div className="whitespace-pre-wrap px-0.5 text-xs leading-relaxed text-zinc-300">
            {plan.overview.trim() || <span className="italic text-zinc-600">How the team works together end to end…</span>}
          </div>
        )}
      </div>

      {!agents.length ? (
        <p className="text-xs text-zinc-500">The planner produced no plan entries.</p>
      ) : (
        <div className="flex flex-col gap-3 md:flex-row md:items-start">
          {/* Left column — agent list (≈30%). */}
          <div className="shrink-0 space-y-1 md:w-[30%]">
            {ordered.map((g) => {
              const on = g.subtask_id === a?.subtask_id
              return (
                <button
                  key={g.subtask_id}
                  onClick={() => setSelId(g.subtask_id)}
                  className={`flex w-full items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors ${
                    on ? 'border-violet-500 bg-violet-500/10'
                       : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700'
                  }`}
                >
                  {g.group && (
                    <span className="mt-0.5 shrink-0 rounded bg-violet-500/20 px-1 text-[9px] font-semibold text-violet-700 dark:text-violet-300">{g.group}</span>
                  )}
                  <span className="min-w-0">
                    <span className={`block truncate text-[11px] font-medium ${on ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-300'}`}>
                      {g.agent_id}
                    </span>
                    <span className="block truncate text-[10px] text-zinc-500">{titleFor(g.subtask_id)}</span>
                  </span>
                </button>
              )
            })}
          </div>

          {/* Right column — the selected agent's fields, stacked vertically. */}
          {a && (
            <div className="min-w-0 flex-1 rounded-lg border border-zinc-800 bg-zinc-900/30 p-3">
              <div className="mb-3 flex items-center gap-2">
                <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-200">
                  <span className="text-violet-700 dark:text-violet-300/90">{a.agent_id}</span>
                  <span className="mx-1 text-zinc-700">·</span>{titleFor(a.subtask_id)}
                </span>
                {/* Group — the phase this agent runs in (A→B→C→D). Selector in edit mode. */}
                <div className="inline-flex shrink-0 items-center gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Group</span>
                  {editing ? (
                    <div className="inline-flex rounded-md border border-zinc-700 bg-zinc-950/50 p-0.5">
                      {GROUPS.map((g) => (
                        <button
                          key={g}
                          onClick={() => patch({ group: g })}
                          title={`Run this agent in group ${g}`}
                          className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                            a.group === g ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200'
                          }`}
                        >
                          {g}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[11px] font-semibold text-violet-700 dark:text-violet-300">{a.group || '—'}</span>
                  )}
                </div>
              </div>
              <Field label="Mandate" value={a.mandate} rows={8} editing={editing}
                onChange={(v) => patch({ mandate: v })} placeholder="What this agent must accomplish…" />
              <Field label="Produces" value={a.produces} rows={2} editing={editing}
                onChange={(v) => patch({ produces: v })} placeholder="The artifact it hands off…" />
              <div className="mb-3">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Consumes from</div>
                {editing ? (
                  teammates.length ? (
                    <div className="flex flex-wrap gap-1.5">
                      {teammates.map((s) => {
                        const on = (a.consumes_from || []).includes(s.id)
                        return (
                          <button
                            key={s.id}
                            onClick={() => toggleConsume(s.id)}
                            title={s.title}
                            className={`rounded border px-2 py-1 text-[10px] transition-colors ${
                              on ? 'border-violet-500 bg-violet-600 text-white'
                                 : 'border-zinc-700 text-zinc-400 hover:text-zinc-200'
                            }`}
                          >
                            {s.owner_archetype_id}
                          </button>
                        )
                      })}
                    </div>
                  ) : (
                    <span className="text-[10px] text-zinc-600">no teammates</span>
                  )
                ) : (a.consumes_from || []).length ? (
                  <div className="flex flex-wrap gap-1.5">
                    {(a.consumes_from || []).map((cid) => {
                      const s = (subtasks || []).find((x) => x.id === cid)
                      return (
                        <span key={cid} className="rounded border border-zinc-700 bg-zinc-900/40 px-2 py-1 text-[10px] text-zinc-300">
                          {s?.owner_archetype_id || cid}
                        </span>
                      )
                    })}
                  </div>
                ) : (
                  <span className="text-[10px] text-zinc-600">nothing — starts from scratch</span>
                )}
              </div>
              <Field label="Hand-off notes" value={a.hand_off_notes} rows={5} editing={editing}
                onChange={(v) => patch({ hand_off_notes: v })} placeholder="What downstream teammates need from its output…" />
            </div>
          )}
        </div>
      )}

      <p className="mt-3 text-[10px] text-zinc-600">
        Groups run in order A→B→C→D (barrier between). Each agent runs with its mandate, output, and
        the teammates it consumes from injected into its prompt.
      </p>
      </>
      ) : (
        <>
          <div className="flex flex-col gap-3 md:flex-row md:items-start">
            {/* Left — question list. */}
            <div className="shrink-0 space-y-1 md:w-[30%]">
              {questions.map((x) => {
                const on = x.id === q?.id
                const done = isAnswered(x)
                return (
                  <button
                    key={x.id}
                    onClick={() => setSelQId(x.id)}
                    className={`flex w-full items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors ${
                      on ? 'border-violet-500 bg-violet-500/10'
                         : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700'
                    }`}
                  >
                    <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${done ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                    <span className={`min-w-0 text-[11px] leading-snug ${on ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-300'}`}>
                      {x.question}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Right — the selected question's answer. */}
            {q && (
              <div className="min-w-0 flex-1 rounded-lg border border-zinc-800 bg-zinc-900/30 p-3">
                <div className="mb-2 text-sm font-medium leading-snug text-zinc-200">{q.question}</div>
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                  {q.multi ? 'Pick any that apply' : 'Pick one'}
                </div>
                {q.options.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-1.5">
                    {q.options.map((opt) => {
                      const on = (q.selected || []).includes(opt)
                      return (
                        <button
                          key={opt}
                          onClick={() => toggleOption(opt)}
                          className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                            on ? 'border-violet-500 bg-violet-600 text-white'
                               : 'border-zinc-700 text-zinc-300 hover:text-zinc-100'
                          }`}
                        >
                          {opt}
                        </button>
                      )
                    })}
                  </div>
                )}
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Other</div>
                <textarea
                  value={q.other}
                  onChange={(e) => patchQ({ other: e.target.value })}
                  rows={2}
                  placeholder="Your own answer (optional)…"
                  className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2.5 py-2 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
                />
              </div>
            )}
          </div>
          <p className="mt-3 text-[10px] text-zinc-600">
            Answer what matters, then click Re-plan above — your answers are folded back into the
            planner and the coordination is regenerated. Unanswered questions use the planner's
            defaults.
          </p>
        </>
      )}
    </div>
  )
}
