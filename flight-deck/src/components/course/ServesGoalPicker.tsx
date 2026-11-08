import { useEffect, useState } from 'react'
import { Compass, Loader2 } from 'lucide-react'
import {
  courseOnce,
  getServed,
  setServed,
  HORIZON_LABELS,
  type CourseResponse,
  type Goal,
  type LinkKind,
  type ServedGoal,
} from '../../services/course'

// "Serves a goal": on an item's own page (a project, a scheduled task, a
// Captain Swarm run), which of the user's Course goals it moves forward. The
// link is the user's own (on a deck-wide goal only they see it) and shows up
// as evidence on that goal in Course.

interface Option { id: string; label: string }

function options(c: CourseResponse): Option[] {
  const live = (g: Goal) => g.status === 'active' || g.status === 'paused'
  const name = new Map(c.agents.map((a) => [a.ref, a.name]))
  const out: Option[] = []
  for (const g of c.goals.deck.filter(live)) out.push({ id: g.id, label: `Deck · ${HORIZON_LABELS[g.horizon]} · ${g.title}` })
  for (const g of c.goals.mine.filter(live)) out.push({ id: g.id, label: `You · ${HORIZON_LABELS[g.horizon]} · ${g.title}` })
  for (const [ref, goals] of Object.entries(c.goals.agents)) {
    for (const g of goals.filter(live)) out.push({ id: g.id, label: `${name.get(ref) || 'Agent'} · ${HORIZON_LABELS[g.horizon]} · ${g.title}` })
  }
  return out
}

export function ServesGoalPicker({ kind, refId, title, compact = false }: {
  kind: LinkKind
  refId: string
  /** The item's name, kept with the link so Course can say what it was. */
  title: string
  compact?: boolean
}) {
  const [served, setServedGoal] = useState<ServedGoal | null>(null)
  const [opts, setOpts] = useState<Option[] | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let gone = false
    setError(null)
    Promise.all([getServed(kind, refId), courseOnce()])
      .then(([s, c]) => { if (!gone) { setServedGoal(s.goal); setOpts(options(c)) } })
      .catch((e) => { if (!gone) { setOpts([]); setError(e instanceof Error ? e.message : String(e)) } })
    return () => { gone = true }
  }, [kind, refId])

  const choose = async (goalId: string) => {
    setSaving(true); setError(null)
    try {
      const r = await setServed(kind, refId, goalId, title)
      setServedGoal(r.goal)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  if (opts === null) return null
  // Nothing to choose and nothing chosen: say nothing.
  if (opts.length === 0 && !served && !error) return null
  const current = served?.id ?? ''
  const shown = served && !opts.some((o) => o.id === served.id)
    ? [{ id: served.id, label: `${HORIZON_LABELS[served.horizon]} · ${served.title}` }, ...opts] : opts
  return (
    <div className={`flex min-w-0 items-center gap-1.5 ${compact ? 'text-[11px]' : 'text-xs'}`}>
      <Compass className="h-3.5 w-3.5 shrink-0 text-violet-500" aria-hidden />
      <label className="sr-only" htmlFor={`serves-${kind}-${refId}`}>Serves a goal</label>
      <select
        id={`serves-${kind}-${refId}`}
        value={current}
        disabled={saving}
        onChange={(e) => void choose(e.target.value)}
        title="Which goal on your Course this moves forward"
        className="min-w-0 max-w-[22rem] truncate rounded-md border border-zinc-800 bg-zinc-950 px-1.5 py-0.5 text-zinc-300 focus:border-violet-500 focus:outline-none disabled:opacity-60"
      >
        <option value="">Serves no goal</option>
        {shown.map((o) => <option key={o.id} value={o.id}>Serves: {o.label}</option>)}
      </select>
      {saving && <Loader2 className="h-3 w-3 animate-spin text-zinc-500" />}
      {error && <span className="truncate text-red-500" title={error}>{error}</span>}
    </div>
  )
}
