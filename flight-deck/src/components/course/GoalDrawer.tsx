import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Loader2, Route, Trash2, X } from 'lucide-react'
import {
  getGoal,
  planRoute,
  removeLink,
  shortDate,
  HORIZON_LABELS,
  LINK_LABELS,
  type GoalDetail,
  type GoalEvent,
} from '../../services/course'

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
    default: return e.event
  }
}

function actorText(actor: string): string {
  if (!actor) return ''
  if (actor.startsWith('agent:')) return 'an agent'
  if (actor.startsWith('user:')) return 'a person'
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
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
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
  const progress = g?.progress ?? null
  return createPortal(
    <div className="fixed inset-0 z-[60] flex justify-end bg-black/50" onClick={onClose}>
      <aside
        role="dialog"
        aria-label={g ? `Goal: ${g.title}` : 'Goal'}
        className="flex h-full w-full max-w-lg flex-col overflow-hidden border-l border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-2 border-b border-zinc-800 px-4 py-3">
          <div className="min-w-0 flex-1">
            {g ? (
              <>
                <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                  {SCOPE_WORD[g.scope]} · {HORIZON_LABELS[g.horizon]}{g.status !== 'active' ? ` · ${g.status}` : ''}
                </div>
                <h2 className="mt-0.5 text-base font-semibold text-zinc-100">{g.title}</h2>
              </>
            ) : <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />}
          </div>
          <button onClick={onClose} title="Close" aria-label="Close" className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4 text-sm">
          {error && <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>}
          {notice && <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">{notice}</div>}
          {g && data && (
            <>
              <section className="space-y-1.5">
                {g.success && <p className="text-zinc-300"><span className="text-zinc-500">Done when:</span> {g.success}</p>}
                {g.detail && <p className="whitespace-pre-wrap text-zinc-400">{g.detail}</p>}
                <p className="text-xs text-zinc-500">
                  {g.due_at && <>Due {shortDate(g.due_at)} · </>}
                  {g.status === 'active' && g.review_at && <>Review by {shortDate(g.review_at)} · </>}
                  Every {g.review_every_days} days
                </p>
              </section>

              {progress !== null && (
                <section>
                  <div className="mb-1 flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                    <span>Progress</span><span className="normal-case tracking-normal">{Math.round(progress * 100)}%</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                    <div className="h-full rounded-full bg-violet-500" style={{ width: `${Math.round(progress * 100)}%` }} />
                  </div>
                  {g.progress_note && <p className="mt-1 text-xs text-zinc-500">{g.progress_note}</p>}
                </section>
              )}

              {g.scope !== 'agent' && g.status === 'active' && (
                <section>
                  <button
                    onClick={() => void route()}
                    disabled={!!busy}
                    className="inline-flex items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                    title="Autonomous Work plans the steps to this goal; you advance the plan one step at a time"
                  >
                    {busy === 'route' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Route className="h-3.5 w-3.5" />}
                    Plan a route
                  </button>
                </section>
              )}

              <section>
                <h3 className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                  Work that moved it{g.scope === 'instance' ? ' (yours)' : ''}
                </h3>
                {data.links.length === 0 ? (
                  <p className="text-xs text-zinc-500">
                    Nothing yet. Tasks, Autonomous Work, plans, and items you mark “Serves a goal” on their own page show up here.
                  </p>
                ) : (
                  <ul className="space-y-1.5">
                    {data.links.map((l) => (
                      <li key={`${l.kind}:${l.ref_id}`} className="flex items-start gap-2 rounded-md border border-zinc-800 px-2.5 py-1.5">
                        <div className="min-w-0 flex-1">
                          <div className="text-xs text-zinc-300">{l.note || l.ref_id}</div>
                          <div className="text-[11px] text-zinc-500">{LINK_LABELS[l.kind] ?? l.kind} · {shortDate(l.at)}</div>
                        </div>
                        <button
                          onClick={() => void unlink(l.kind, l.ref_id)}
                          disabled={!!busy}
                          title="It didn't serve this goal"
                          aria-label="Remove link"
                          className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-500 disabled:opacity-50"
                        >
                          {busy === `${l.kind}:${l.ref_id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section>
                <h3 className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-zinc-500">History</h3>
                <ul className="space-y-1 text-xs">
                  {data.events.map((e) => (
                    <li key={e.id} className="flex gap-2 text-zinc-400">
                      <span className="w-24 shrink-0 text-zinc-600">{shortDate(e.at)}</span>
                      <span className="min-w-0 flex-1">{eventText(e)}{actorText(e.actor) && <span className="text-zinc-600"> · {actorText(e.actor)}</span>}</span>
                    </li>
                  ))}
                </ul>
              </section>
            </>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  )
}
