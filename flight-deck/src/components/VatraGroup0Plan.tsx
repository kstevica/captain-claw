import { useEffect, useMemo, useState } from 'react'
import { ClipboardList, CheckCircle2, Trash2, Loader2 } from 'lucide-react'
import type { Group0Plan, Group0Agent, VatraSubtask } from '../stores/basnaStore'

// The Group 0 gate: the Long Horizon Planner's per-agent coordination plan, presented
// for review and editing before any Group-A worker runs. Mirrors the Code-mode plan
// gate (edit → Execute / Cancel). The edited draft is sent back verbatim on Execute.

// Theme-aware violet accent (matches VatraTeamPlan).
const PANEL = 'rounded-lg border border-violet-300/70 bg-violet-50/60 p-4 dark:border-violet-800/40 dark:bg-violet-950/10'

function Chip({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`rounded-full border border-zinc-700/60 bg-zinc-800/60 px-2 py-0.5 text-[11px] font-medium ${className}`}>
      {children}
    </span>
  )
}

function Field({ label, value, onChange, rows = 2, placeholder }: {
  label: string
  value: string
  onChange: (v: string) => void
  rows?: number
  placeholder?: string
}) {
  return (
    <div className="mb-1.5">
      <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">{label}</div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={rows}
        placeholder={placeholder}
        className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
      />
    </div>
  )
}

export function VatraGroup0Plan({
  plan, subtasks, busy = false, onExecute, onCancel,
}: {
  plan?: Group0Plan
  subtasks?: VatraSubtask[]
  busy?: boolean
  onExecute: (plan: Group0Plan) => void
  onCancel: () => void
}) {
  const [draft, setDraft] = useState<Group0Plan>(() => plan || { overview: '', agents: [] })
  // Re-seed when a fresh plan arrives (the gate opens, or a poll re-fetches the route).
  // Skip while busy so an in-flight approve doesn't clobber the user's last edit.
  useEffect(() => { if (plan && !busy) setDraft(plan) }, [plan, busy])

  const titleFor = useMemo(() => {
    const m = new Map<string, string>()
    for (const s of subtasks || []) m.set(s.id, s.title)
    return (id: string) => m.get(id) || id
  }, [subtasks])

  const patch = (i: number, p: Partial<Group0Agent>) =>
    setDraft((d) => ({ ...d, agents: d.agents.map((a, j) => (j === i ? { ...a, ...p } : a)) }))
  const toggleConsume = (i: number, cid: string) =>
    setDraft((d) => ({
      ...d,
      agents: d.agents.map((a, j) => {
        if (j !== i) return a
        const has = (a.consumes_from || []).includes(cid)
        return {
          ...a,
          consumes_from: has ? a.consumes_from.filter((x) => x !== cid) : [...(a.consumes_from || []), cid],
        }
      }),
    }))

  const agents = draft.agents || []

  return (
    <div className={PANEL}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ClipboardList className="h-3.5 w-3.5 text-violet-600 dark:text-violet-400" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Group 0 — coordination plan</span>
        <Chip className="text-violet-600 dark:text-violet-300">Long Horizon Planner</Chip>
        <Chip className="text-amber-600 dark:text-amber-300">review before run</Chip>
      </div>
      <p className="mb-3 text-[11px] leading-snug text-zinc-500">
        The Long Horizon Planner drafted how the team will operate. Edit any mandate, hand-off, or
        dependency, then <span className="text-zinc-300">Execute</span> to run Group A — or{' '}
        <span className="text-zinc-300">Cancel</span> to discard (nothing runs).
      </p>

      <div className="mb-3">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-violet-700 dark:text-violet-300">Coordination overview</div>
        <textarea
          value={draft.overview}
          onChange={(e) => setDraft((d) => ({ ...d, overview: e.target.value }))}
          rows={3}
          placeholder="How the team works together end to end…"
          className="w-full resize-y rounded border border-zinc-700 bg-zinc-950/60 px-2 py-1.5 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none"
        />
      </div>

      <div className="space-y-2.5">
        {agents.map((a, i) => {
          const teammates = (subtasks || []).filter((s) => s.id !== a.subtask_id)
          return (
            <div key={a.subtask_id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
              <div className="mb-2 flex items-center gap-2">
                {a.group && (
                  <span className="rounded bg-violet-500/20 px-1 text-[9px] font-semibold text-violet-700 dark:text-violet-300">{a.group}</span>
                )}
                <span className="min-w-0 flex-1 truncate text-xs font-medium text-zinc-200">
                  <span className="text-violet-700 dark:text-violet-300/90">{a.agent_id}</span>
                  <span className="mx-1 text-zinc-700">·</span>{titleFor(a.subtask_id)}
                </span>
              </div>
              <Field label="Mandate" value={a.mandate} rows={2}
                onChange={(v) => patch(i, { mandate: v })} placeholder="What this agent must accomplish…" />
              <Field label="Produces" value={a.produces} rows={1}
                onChange={(v) => patch(i, { produces: v })} placeholder="The artifact it hands off…" />
              <div className="mt-1.5 mb-1.5">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Consumes from</div>
                {teammates.length ? (
                  <div className="flex flex-wrap gap-1">
                    {teammates.map((s) => {
                      const on = (a.consumes_from || []).includes(s.id)
                      return (
                        <button
                          key={s.id}
                          onClick={() => toggleConsume(i, s.id)}
                          title={s.title}
                          className={`rounded border px-1.5 py-0.5 text-[10px] transition-colors ${
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
                )}
              </div>
              <Field label="Hand-off notes" value={a.hand_off_notes} rows={2}
                onChange={(v) => patch(i, { hand_off_notes: v })} placeholder="What downstream teammates need from its output…" />
            </div>
          )
        })}
        {!agents.length && <p className="text-xs text-zinc-500">The planner produced no plan entries.</p>}
      </div>

      <p className="mt-2 text-[10px] text-zinc-600">
        Groups run in order A→B→C→D (barrier between). Each agent runs with its mandate, output, and
        the teammates it consumes from injected into its prompt.
      </p>

      <div className="mt-3 flex items-center justify-end gap-2">
        <button
          onClick={onCancel}
          disabled={busy}
          className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-red-500/50 hover:text-red-300 disabled:opacity-50"
        >
          <Trash2 className="h-3.5 w-3.5" /> Cancel
        </button>
        <button
          onClick={() => onExecute(draft)}
          disabled={busy}
          className="inline-flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />} Execute
        </button>
      </div>
    </div>
  )
}
