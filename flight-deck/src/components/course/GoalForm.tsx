import { useState } from 'react'
import {
  DEFAULT_REVIEW_DAYS,
  HORIZON_HINTS,
  HORIZON_LABELS,
  HORIZONS,
  METRIC_LABELS,
  eligibleParents,
  type Goal,
  type GoalDraft,
  type Horizon,
  type MetricKind,
  type Scope,
} from '../../services/course'

const SCOPE_TAG: Record<Scope, string> = { instance: 'Deck', user: 'You', agent: 'This agent' }
const SAVE_KEYS = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ Enter' : 'Ctrl + Enter'
const SCOPE_FOR: Record<Scope, string> = { instance: 'for the whole deck', user: 'for you', agent: 'for this agent' }

// The form for a new goal, or an edit of one. `allowed` is the horizons the
// admin lets this scope use; an existing goal keeps its own even when it was
// taken away (it can be moved off it, never onto another retired one).
export function GoalForm({
  scope, agentRef, allowed, initial, all, selfId = '', caps, saving, error, onSave, onCancel, submitLabel,
}: {
  scope: Scope
  agentRef: string
  allowed: Horizon[]
  initial: GoalDraft
  all: Goal[]
  selfId?: string
  caps: { title: number; detail: number; success: number }
  saving: boolean
  error: string | null
  onSave: (d: GoalDraft) => void
  onCancel: () => void
  submitLabel?: string
}) {
  const [d, setD] = useState<GoalDraft>(initial)
  const horizons = HORIZONS.filter((h) => allowed.includes(h) || h === initial.horizon)
  const parents = eligibleParents(all, scope, d.horizon, agentRef, selfId)
  const strategic = d.horizon === 'strategic'
  const len = (s: string) => Array.from(s).length
  const tooLong = len(d.title) > caps.title || len(d.success) > caps.success || len(d.detail) > caps.detail
    || len(d.metric_unit) > 30
  const missing = !d.title.trim() || (!strategic && !d.success.trim())
  const target = d.metric_target.trim()
  const badTarget = !!d.metric_kind && !!target && !(Number(target) > 0)
  // A deck-wide goal's work is everyone's: it's measured by a number the admin enters.
  const kinds = (Object.keys(METRIC_LABELS) as MetricKind[]).filter((k) => scope !== 'instance' || k === 'manual')

  const setHorizon = (h: Horizon) => setD((cur) => ({
    ...cur,
    horizon: h,
    // The default review period follows the horizon until someone sets their own.
    review_every_days: cur.review_every_days === DEFAULT_REVIEW_DAYS[cur.horizon]
      ? DEFAULT_REVIEW_DAYS[h] : cur.review_every_days,
    // A parent of a shorter horizon no longer fits.
    parent_id: eligibleParents(all, scope, h, agentRef, selfId).some((p) => p.id === cur.parent_id)
      ? cur.parent_id : '',
  }))

  const input = 'w-full rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 transition-colors focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/15'
  const label = 'mb-1.5 flex items-baseline justify-between gap-2 text-xs font-medium text-zinc-400'
  // The count shows once it matters: near the limit, or past it.
  const counter = (n: number, cap: number) => (n >= cap * 0.8
    ? <span className={`tabular-nums ${n > cap ? 'text-red-500' : 'text-zinc-500'}`}>{n}/{cap}</span>
    : null)
  const submit = () => { if (!missing && !tooLong && !badTarget && !saving) onSave(d) }

  return (
    <form
      className="space-y-4 rounded-[var(--fd-radius-card)] border border-violet-500/30 bg-[var(--fd-surface)] p-4 shadow-[0_0_0_4px_rgba(139,92,246,0.06)] wide:p-5"
      onSubmit={(e) => { e.preventDefault(); submit() }}
      onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit() } }}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-sm font-semibold text-zinc-100">
          {submitLabel ? 'Edit, then accept' : selfId ? 'Edit goal' : 'New goal'}
        </span>
        <span className="text-xs text-zinc-500">{SCOPE_FOR[scope]}</span>
      </div>

      <label className="block">
        <span className={label}><span>Goal</span>{counter(len(d.title), caps.title)}</span>
        <input
          className={`${input} h-11 text-[15px] font-medium`}
          value={d.title}
          autoFocus
          onChange={(e) => setD({ ...d, title: e.target.value })}
          placeholder={strategic ? 'e.g. Be the bookkeeping partner small firms trust most' : 'e.g. Launch EU pricing'}
        />
      </label>

      <div>
        <div className={label}>Horizon</div>
        <div className="inline-flex max-w-full flex-wrap rounded-full border border-[var(--fd-line)] p-0.5" role="radiogroup" aria-label="Horizon">
          {horizons.map((h) => (
            <button
              key={h}
              type="button"
              role="radio"
              aria-checked={d.horizon === h}
              onClick={() => setHorizon(h)}
              disabled={!allowed.includes(h) && h !== d.horizon}
              className={`h-8 rounded-full px-3 text-xs font-medium transition-colors disabled:opacity-40 ${d.horizon === h
                ? 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]'
                : 'text-zinc-500 hover:text-zinc-200'}`}
            >
              {HORIZON_LABELS[h]}
            </button>
          ))}
        </div>
        <p className="mt-1.5 text-xs text-zinc-500">
          {HORIZON_HINTS[d.horizon]}
          {!allowed.includes(d.horizon) && ' This level no longer uses it: move the goal to another horizon.'}
        </p>
      </div>

      <label className="block">
        <span className={label}>
          <span>How you’ll know it’s done{strategic ? ' (optional)' : ''}</span>{counter(len(d.success), caps.success)}
        </span>
        <textarea
          className={input}
          rows={2}
          value={d.success}
          onChange={(e) => setD({ ...d, success: e.target.value })}
          placeholder={strategic ? 'A direction has no finish line; a sign you’re heading the right way helps.' : 'e.g. Three EU customers on the new plan'}
        />
      </label>

      <label className="block">
        <span className={label}><span>Why it matters (optional)</span>{counter(len(d.detail), caps.detail)}</span>
        <textarea
          className={input}
          rows={2}
          value={d.detail}
          onChange={(e) => setD({ ...d, detail: e.target.value })}
          placeholder="Context your agents should weigh when choosing what to do."
        />
      </label>

      <div>
        <div className={label}>How progress is measured (optional)</div>
        <div className="grid gap-2 sm:grid-cols-3">
          <select
            className={input}
            aria-label="Measure"
            value={d.metric_kind}
            onChange={(e) => setD({ ...d, metric_kind: e.target.value as MetricKind | '' })}
          >
            <option value="">Not measured (estimated from the work)</option>
            {kinds.map((k) => <option key={k} value={k}>{METRIC_LABELS[k]}</option>)}
          </select>
          {d.metric_kind && (
            <>
              <input
                className={input}
                inputMode="decimal"
                aria-label="Target"
                placeholder="Target, e.g. 10"
                value={d.metric_target}
                onChange={(e) => setD({ ...d, metric_target: e.target.value })}
              />
              <input
                className={input}
                aria-label="Unit"
                placeholder={d.metric_kind === 'manual' ? 'Unit, e.g. customers' : 'Unit (optional)'}
                value={d.metric_unit}
                onChange={(e) => setD({ ...d, metric_unit: e.target.value })}
              />
            </>
          )}
        </div>
        <p className="mt-1.5 text-xs text-zinc-500">
          {d.metric_kind === 'manual' ? 'You enter the current value on the goal; it fills the progress bar toward the target.'
            : d.metric_kind === 'taskboard_done' ? 'Counts the tasks in Tasks tied to this goal and done since it went on your course.'
              : d.metric_kind === 'link_count' ? 'Counts the work linked to this goal: tasks, plans, Autonomous Work, scheduled tasks.'
                : 'Without a measure, progress is an estimate from the work linked to it.'}
          {badTarget && <span className="text-red-500"> The target must be a number above 0.</span>}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="block sm:col-span-3">
          <span className={label}>Serves</span>
          <select className={input} value={d.parent_id} onChange={(e) => setD({ ...d, parent_id: e.target.value })}>
            <option value="">Nothing above it</option>
            {parents.map((p) => (
              <option key={p.id} value={p.id}>
                {SCOPE_TAG[p.scope]} · {HORIZON_LABELS[p.horizon]} · {p.title}
              </option>
            ))}
          </select>
        </label>
        {!strategic && (
          <label className="block">
            <span className={label}>Due (optional)</span>
            <input type="date" className={input} value={d.due_at} onChange={(e) => setD({ ...d, due_at: e.target.value })} />
          </label>
        )}
        <label className="block">
          <span className={label}>Review every</span>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              max={730}
              className={`${input} w-24`}
              value={d.review_every_days}
              onChange={(e) => setD({ ...d, review_every_days: Math.max(1, Math.min(730, Number(e.target.value) || 1)) })}
            />
            <span className="text-xs text-zinc-500">days</span>
          </div>
        </label>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/[0.07] px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>
      )}
      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-[var(--fd-line)] pt-4">
        <span className="mr-auto hidden text-[11px] text-zinc-500 [@media(pointer:fine)]:inline">{SAVE_KEYS} to save</span>
        <button type="button" onClick={onCancel} disabled={saving} className="fd-btn-quiet h-9 px-4 text-sm disabled:opacity-50">
          Cancel
        </button>
        <button
          type="submit"
          disabled={saving || missing || tooLong || badTarget}
          className="fd-btn-primary h-9 px-4 text-sm disabled:pointer-events-none disabled:opacity-50"
        >
          {saving ? 'Saving…' : submitLabel ?? (selfId ? 'Save goal' : 'Add goal')}
        </button>
      </div>
    </form>
  )
}
