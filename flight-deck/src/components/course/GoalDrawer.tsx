import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { CornerDownRight, Loader2, Route, Trash2, X } from 'lucide-react'
import {
  dismissConflict,
  getGoal,
  measureText,
  planRoute,
  progressOf,
  removeLink,
  setGoalValue,
  shortDate,
  HORIZON_LABELS,
  LINK_LABELS,
  type GoalDetail,
  type GoalEvent,
} from '../../services/course'
import { ProgressRing } from './ProgressRing'

const SCOPE_WORD = { instance: 'The deck', user: 'You', agent: 'An agent' } as const

function eventText(e: GoalEvent): string {
  const d = e.detail as Record<string, unknown>
  switch (e.event) {
    case 'created': return 'Added'
    case 'proposed': return 'Proposed'
    case 'edited': return `Edited (${(d.fields as string[] | undefined)?.join(', ').replace(/_/g, ' ') || 'details'})`
    case 'status': return `${String(d.from ?? '')} → ${String(d.to ?? '')}`
    case 'reviewed': return 'Still on course'
    case 'linked': return `Work linked (${LINK_LABELS[d.kind as keyof typeof LINK_LABELS] ?? String(d.kind ?? '')})`
    case 'progress': return `Progress ${Math.round(Number(d.progress ?? 0) * 100)}%`
    case 'measured': return `Value set to ${String(d.value ?? '')}`
    case 'conflict': return 'Flagged: may pull against another goal'
    default: return e.event
  }
}

function actorText(actor: string): string {
  if (!actor) return ''
  if (actor.startsWith('agent:')) return 'an agent'
  if (actor.startsWith('user:')) return 'a person'
  if (actor === 'deck:estimate') return 'weekly estimate'
  if (actor === 'deck:conflict') return 'Flight Deck'
  return actor   // autonomy · plan · taskboard
}

// One goal, whole: what it means, how far along it is, the work that moved it
// (evidence — on a deck-wide goal only yours) and its history; "Plan a route"
// has Autonomous Work plan the steps to it.
export function GoalDrawer({ goalId, onClose, onChanged }: {
  goalId: string
  onClose: () => void
  onChanged: () => void
}) {
  const [data, setData] = useState<GoalDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setData(await getGoal(goalId))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [goalId])
  useEffect(() => { void load() }, [load])
  // Escape closes the drawer only — taken first (capture) and marked handled,
  // so a dialog under it (the Course dialog) stays open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      e.preventDefault()
      onClose()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [onClose])

  const route = async () => {
    setBusy('route'); setError(null); setNotice(null)
    try {
      const r = await planRoute(goalId)
      setNotice(`Planned: ${r.plan.goal}. Advance it step by step in Autonomous Work.`)
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const [value, setValue] = useState('')
  const saveValue = async () => {
    const n = Number(value)
    if (!value.trim() || !(n >= 0)) { setError('Enter a number (0 or more).'); return }
    setBusy('value'); setError(null); setNotice(null)
    try {
      await setGoalValue(goalId, n)
      setValue('')
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const notConflict = async (id: string) => {
    setBusy(`cf:${id}`); setError(null); setNotice(null)
    try {
      await dismissConflict(id)
      setNotice('Marked “not a conflict”. Flight Deck won’t flag that pair again.')
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const unlink = async (kind: GoalDetail['links'][number]['kind'], ref: string) => {
    setBusy(`${kind}:${ref}`); setError(null)
    try {
      await removeLink(goalId, kind, ref)
      await load()
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const g = data?.goal
  const measure = data?.measure ?? null
  const progress = g ? progressOf({ measure, progress: g.progress }) : null
  const live = !!g && (g.status === 'active' || g.status === 'paused')
  const label = 'mb-2 text-xs font-semibold text-zinc-400'
  const STATUS_WORD: Record<string, string> = { paused: 'Paused', achieved: 'Done', dropped: 'Dropped', proposed: 'Suggested', declined: 'Declined', superseded: 'Replaced' }
  return createPortal(
    <div className="fd-fade fixed inset-0 z-[60] flex justify-end bg-black/30 wide:p-3" onClick={onClose}>
      <aside
        role="dialog"
        aria-label={g ? `Goal: ${g.title}` : 'Goal'}
        className="fd-drawer fd-dialog flex h-full w-full max-w-lg flex-col overflow-hidden border-l border-[var(--fd-line)] bg-zinc-950 wide:rounded-[var(--fd-radius-panel)] wide:border"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 px-5 pb-4 pt-5">
          <div className="min-w-0 flex-1">
            {g ? (
              <>
                <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium">
                  <span className="rounded-full bg-[var(--fd-accent-soft)] px-2 py-0.5 text-[var(--fd-accent-text)]">{SCOPE_WORD[g.scope]}</span>
                  <span className="rounded-full bg-zinc-500/10 px-2 py-0.5 text-zinc-400">{HORIZON_LABELS[g.horizon]}</span>
                  {g.status !== 'active' && (
                    <span className="rounded-full bg-zinc-500/10 px-2 py-0.5 text-zinc-400">{STATUS_WORD[g.status] ?? g.status}</span>
                  )}
                </div>
                <h2 className="mt-2 text-lg font-bold leading-snug tracking-tight text-zinc-100">{g.title}</h2>
              </>
            ) : <Loader2 className="mt-1 h-4 w-4 animate-spin text-zinc-500" />}
          </div>
          <button onClick={onClose} title="Close" aria-label="Close"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-5 pb-8 text-sm">
          {error && <div className="rounded-xl border border-red-500/25 bg-red-500/[0.07] px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>}
          {notice && <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/[0.07] px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">{notice}</div>}
          {g && data && (
            <>
              <section className="space-y-2">
                {g.success && (
                  <p className="leading-relaxed text-zinc-300"><span className="text-zinc-500">Done when: </span>{g.success}</p>
                )}
                {g.detail && <p className="whitespace-pre-wrap leading-relaxed text-zinc-400">{g.detail}</p>}
                <p className="flex flex-wrap gap-x-1.5 text-xs text-zinc-500">
                  {g.due_at && <><span>Due {shortDate(g.due_at)}</span><span aria-hidden className="text-zinc-700">·</span></>}
                  {g.status === 'active' && g.review_at && <><span>Review by {shortDate(g.review_at)}</span><span aria-hidden className="text-zinc-700">·</span></>}
                  <span>Reviewed every {g.review_every_days} days</span>
                </p>
              </section>

              {(progress !== null || measure || live) && (
                <section className="fd-card p-4">
                  <div className="flex items-center gap-4">
                    <ProgressRing value={progress} status={g.status} size={56} stroke={5} />
                    <div className="min-w-0 flex-1">
                      <div className="text-2xl font-bold tabular-nums tracking-tight text-zinc-100">
                        {progress !== null ? `${measure?.progress != null ? '' : '~'}${Math.round(progress * 100)}%` : '—'}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {measure ? (
                          <>
                            {measure.label}: <span className="text-zinc-300">{measureText(measure) || '—'}</span>
                            {measure.kind === 'manual' && measure.value_at && <span> · {shortDate(measure.value_at)}</span>}
                          </>
                        ) : progress !== null ? 'Estimated from the work linked to it' : 'No estimate yet — it comes from the work linked to it'}
                      </div>
                    </div>
                  </div>
                  {g.progress != null && (measure?.progress != null || g.progress_note) && (
                    <p className="mt-3 text-xs leading-relaxed text-zinc-500">
                      {measure?.progress != null ? `Estimate: about ${Math.round(g.progress * 100)}%` : ''}
                      {measure?.progress != null && g.progress_note ? ' — ' : ''}{g.progress_note}
                    </p>
                  )}
                  {measure?.kind === 'manual' && data.can_edit && live && (
                    <form className="mt-3 flex items-center gap-2 border-t border-[var(--fd-line)] pt-3" onSubmit={(e) => { e.preventDefault(); void saveValue() }}>
                      <input
                        inputMode="decimal"
                        aria-label="Current value"
                        placeholder="Current value"
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        className="h-8 w-36 rounded-full border border-zinc-800 bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
                      />
                      {measure.unit && <span className="text-xs text-zinc-500">{measure.unit}</span>}
                      <button type="submit" disabled={!!busy || !value.trim()}
                        className="fd-btn-primary ml-auto h-8 px-3.5 text-xs disabled:pointer-events-none disabled:opacity-50">
                        {busy === 'value' ? 'Saving…' : 'Update'}
                      </button>
                    </form>
                  )}
                </section>
              )}

              {g.scope !== 'agent' && g.status === 'active' && (
                <section className="flex items-center gap-3 rounded-[var(--fd-radius-card)] border border-dashed border-zinc-800 px-4 py-3">
                  <Route className="h-4 w-4 shrink-0 text-[var(--fd-accent-text)]" />
                  <p className="min-w-0 flex-1 text-xs leading-relaxed text-zinc-500">
                    Autonomous Work plans the steps to it; you advance the plan one step at a time.
                  </p>
                  <button onClick={() => void route()} disabled={!!busy}
                    className="fd-btn-quiet h-8 shrink-0 px-3.5 text-xs disabled:pointer-events-none disabled:opacity-50">
                    {busy === 'route' && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    Plan a route
                  </button>
                </section>
              )}

              {data.conflicts.length > 0 && (
                <section>
                  <h3 className={label}>May pull against</h3>
                  <ul className="space-y-2">
                    {data.conflicts.map((cf) => (
                      <li key={cf.id} className="rounded-xl bg-red-500/[0.07] px-3 py-2.5">
                        <div className="text-[13px] font-medium text-zinc-200">{cf.other_title}</div>
                        {cf.reason && <div className="mt-0.5 text-xs leading-relaxed text-zinc-400">{cf.reason}</div>}
                        <div className="mt-2 flex items-center gap-2 text-[11px] text-zinc-500">
                          <span className="flex-1">Flagged {shortDate(cf.at)}. Change or drop one of them, or:</span>
                          <button onClick={() => void notConflict(cf.id)} disabled={!!busy}
                            className="rounded-full px-2.5 py-1 font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                            {busy === `cf:${cf.id}` ? 'Saving…' : 'Not a conflict'}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <section>
                <h3 className={label}>Work that moved it{g.scope === 'instance' ? ' (yours)' : ''}</h3>
                {data.links.length === 0 ? (
                  <p className="text-xs leading-relaxed text-zinc-500">
                    Nothing yet. Tasks, Autonomous Work, plans, and items you mark “Serves a goal” on their own page show up here.
                  </p>
                ) : (
                  <ul className="-mx-2">
                    {data.links.map((l) => (
                      <li key={`${l.kind}:${l.ref_id}`} className="group flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-[var(--fd-hover)]">
                        <CornerDownRight className="h-3.5 w-3.5 shrink-0 text-zinc-500" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-[13px] text-zinc-200">{l.note || l.ref_id}</div>
                          <div className="text-[11px] text-zinc-500">{LINK_LABELS[l.kind] ?? l.kind} · {shortDate(l.at)}</div>
                        </div>
                        <button
                          onClick={() => void unlink(l.kind, l.ref_id)}
                          disabled={!!busy}
                          title="It didn't serve this goal"
                          aria-label="Remove link"
                          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-zinc-500 opacity-0 transition-colors hover:bg-zinc-800 hover:text-red-500 focus:opacity-100 group-hover:opacity-100 pointer-coarse:opacity-100 disabled:opacity-50"
                        >
                          {busy === `${l.kind}:${l.ref_id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section>
                <h3 className={label}>History</h3>
                <ol className="ml-1 space-y-3 border-l border-zinc-800 pl-4 text-xs">
                  {data.events.map((e) => (
                    <li key={e.id} className="relative">
                      <span aria-hidden className="absolute -left-[21px] top-[3px] h-[9px] w-[9px] rounded-full border-2 border-zinc-950 bg-zinc-600" />
                      <div className="text-zinc-300">{eventText(e)}</div>
                      <div className="text-[11px] text-zinc-500">{shortDate(e.at)}{actorText(e.actor) && ` · ${actorText(e.actor)}`}</div>
                    </li>
                  ))}
                </ol>
              </section>
            </>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  )
}
