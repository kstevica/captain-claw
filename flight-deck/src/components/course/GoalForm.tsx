import { useState } from 'react'
import {
  DEFAULT_REVIEW_DAYS,
  HORIZON_HINTS,
  HORIZON_LABELS,
  HORIZONS,
  eligibleParents,
  type Goal,
  type GoalDraft,
  type Horizon,
  type Scope,
} from '../../services/course'

const SCOPE_TAG: Record<Scope, string> = { instance: 'Deck', user: 'You', agent: 'This agent' }

// The form for a new goal, or an edit of one. `allowed` is the horizons the
// admin lets this scope use; an existing goal keeps its own even when it was
// taken away (it can be moved off it, never onto another retired one).
export function GoalForm({
  scope, agentRef, allowed, initial, all, selfId = '', caps, saving, error, onSave, onCancel,
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
}) {
  const [d, setD] = useState<GoalDraft>(initial)
  const horizons = HORIZONS.filter((h) => allowed.includes(h) || h === initial.horizon)
  const parents = eligibleParents(all, scope, d.horizon, agentRef, selfId)
  const strategic = d.horizon === 'strategic'
  const len = (s: string) => Array.from(s).length
  const tooLong = len(d.title) > caps.title || len(d.success) > caps.success || len(d.detail) > caps.detail
  const missing = !d.title.trim() || (!strategic && !d.success.trim())

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

  const input = 'w-full rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-violet-500 focus:outline-none'
  const label = 'mb-1 flex items-baseline justify-between text-[11px] font-medium uppercase tracking-wider text-zinc-500'
  const counter = (n: number, cap: number) => (
    <span className={`normal-case tracking-normal ${n > cap ? 'text-red-500' : 'text-zinc-600'}`}>{n}/{cap}</span>
  )

  return (
    <form
      className="space-y-3 rounded-lg border border-violet-500/30 bg-zinc-900/60 p-3"
      onSubmit={(e) => { e.preventDefault(); if (!missing && !tooLong) onSave(d) }}
    >
      <div>
        <div className={label}>Horizon</div>
        <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Horizon">
          {horizons.map((h) => (
            <button
              key={h}
              type="button"
              role="radio"
              aria-checked={d.horizon === h}
              onClick={() => setHorizon(h)}
              disabled={!allowed.includes(h) && h !== d.horizon}
              className={`rounded-md border px-2.5 py-1 text-xs ${d.horizon === h
                ? 'border-violet-500 bg-violet-600/20 text-violet-300'
                : 'border-zinc-800 text-zinc-400 hover:bg-zinc-800'}`}
            >
              {HORIZON_LABELS[h]}
            </button>
          ))}
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">
          {HORIZON_HINTS[d.horizon]}
          {!allowed.includes(d.horizon) && ' This level no longer uses it: move the goal to another horizon.'}
        </p>
      </div>

      <label className="block">
        <span className={label}>Goal {counter(len(d.title), caps.title)}</span>
        <input
          className={input}
          value={d.title}
          autoFocus
          onChange={(e) => setD({ ...d, title: e.target.value })}
          placeholder={strategic ? 'e.g. Be the bookkeeping partner small firms trust most' : 'e.g. Launch EU pricing'}
        />
      </label>

      <label className="block">
        <span className={label}>
          How you’ll know it’s done{strategic ? ' (optional)' : ''} {counter(len(d.success), caps.success)}
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
        <span className={label}>Why it matters (optional) {counter(len(d.detail), caps.detail)}</span>
        <textarea
          className={input}
          rows={2}
          value={d.detail}
          onChange={(e) => setD({ ...d, detail: e.target.value })}
          placeholder="Context your agents should weigh when choosing what to do."
        />
      </label>

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
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>
      )}
      <div className="flex items-center gap-2">
        <button
          type="submit"
          disabled={saving || missing || tooLong}
          className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
        >
          {saving ? 'Saving…' : selfId ? 'Save goal' : 'Add goal'}
        </button>
        <button type="button" onClick={onCancel} disabled={saving}
          className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
          Cancel
        </button>
      </div>
    </form>
  )
}
