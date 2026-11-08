import { useEffect, useState } from 'react'
import { Loader2, SlidersHorizontal } from 'lucide-react'
import {
  getSettings,
  saveSettings,
  HORIZON_LABELS,
  HORIZONS,
  SCOPES,
  SCOPE_LABELS,
  type CourseSettings,
  type Horizon,
  type Scope,
} from '../../services/course'

const same = (a: CourseSettings, b: CourseSettings) => JSON.stringify(a) === JSON.stringify(b)

// Admin: which horizons each level of Course uses (the deck, each user, each
// agent) and how many active goals a horizon may hold
// (/fd/admin/course/settings). On the Admin page, the desktop console and, for
// an admin, the Course page itself (the only place an auth-off deck's one user
// reaches it).
export function DeckCourseSettings({ onSaved }: { onSaved?: () => void }) {
  const [saved, setSaved] = useState<CourseSettings | null>(null)
  const [draft, setDraft] = useState<CourseSettings | null>(null)
  const [range, setRange] = useState<[number, number]>([1, 20])
  const [weekRange, setWeekRange] = useState<[number, number]>([0, 20])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getSettings()
      .then((s) => {
        if (cancelled) return
        const v = { horizons: s.horizons, caps: s.caps, proposals: s.proposals, automation: s.automation }
        setSaved(v); setDraft(v); setRange(s.cap_range); setWeekRange(s.per_week_range ?? [0, 20])
      })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  const toggle = (scope: Scope, h: Horizon) => {
    if (!draft) return
    const cur = draft.horizons[scope]
    const next = cur.includes(h) ? cur.filter((x) => x !== h) : HORIZONS.filter((x) => x === h || cur.includes(x))
    setDraft({ ...draft, horizons: { ...draft.horizons, [scope]: next } })
    setNotice(null)
  }
  const setCap = (h: Horizon, v: number) => {
    if (!draft) return
    setDraft({ ...draft, caps: { ...draft.caps, [h]: v } })
    setNotice(null)
  }

  const setProposals = (patch: Partial<CourseSettings['proposals']>) => {
    if (!draft) return
    setDraft({ ...draft, proposals: { ...draft.proposals, ...patch } })
    setNotice(null)
  }

  const setAutomation = (patch: Partial<CourseSettings['automation']>) => {
    if (!draft) return
    setDraft({ ...draft, automation: { ...draft.automation, ...patch } })
    setNotice(null)
  }

  const dirty = !!saved && !!draft && !same(saved, draft)
  const badCap = !!draft && HORIZONS.some((h) => !(draft.caps[h] >= range[0] && draft.caps[h] <= range[1]))
  const badWeek = !!draft && [draft.proposals.agent_per_week, draft.proposals.deck_per_week]
    .some((n) => !(Number.isInteger(n) && n >= weekRange[0] && n <= weekRange[1]))

  const save = async () => {
    if (!draft) return
    setSaving(true); setError(null); setNotice(null)
    try {
      const s = await saveSettings(draft)
      const v = { horizons: s.horizons, caps: s.caps, proposals: s.proposals, automation: s.automation }
      setSaved(v); setDraft(v)
      setNotice('Saved. Goals already at a horizon you took away are kept; only new ones are refused.')
      onSaved?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
      <div className="flex items-start gap-2">
        <SlidersHorizontal className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">Course: horizons, limits, suggestions and checks</h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            Which horizons each level uses, and how many active goals a horizon may hold at each level.
            Clearing a row switches goals off at that level.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-4"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
      ) : draft && saved ? (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[28rem] text-xs">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-wider text-zinc-500">
                  <th className="py-1.5 pr-3 font-medium">Level</th>
                  {HORIZONS.map((h) => <th key={h} className="py-1.5 px-2 text-center font-medium">{HORIZON_LABELS[h]}</th>)}
                </tr>
              </thead>
              <tbody>
                {SCOPES.map((s) => (
                  <tr key={s} className="border-t border-zinc-800/70">
                    <td className="py-2 pr-3 text-zinc-300">{SCOPE_LABELS[s]}</td>
                    {HORIZONS.map((h) => (
                      <td key={h} className="py-2 px-2 text-center">
                        <input
                          type="checkbox"
                          aria-label={`${SCOPE_LABELS[s]}: ${HORIZON_LABELS[h]}`}
                          checked={draft.horizons[s].includes(h)}
                          onChange={() => toggle(s, h)}
                          className="h-4 w-4 accent-violet-600"
                        />
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t border-zinc-800/70">
                  <td className="py-2 pr-3 text-zinc-300">Most active goals</td>
                  {HORIZONS.map((h) => (
                    <td key={h} className="py-2 px-2 text-center">
                      <input
                        type="number"
                        min={range[0]}
                        max={range[1]}
                        aria-label={`Most active ${HORIZON_LABELS[h].toLowerCase()} goals`}
                        value={draft.caps[h]}
                        onChange={(e) => setCap(h, Number(e.target.value))}
                        className="w-14 rounded-md border border-zinc-800 bg-zinc-950 px-1.5 py-1 text-center text-zinc-200 focus:border-violet-500 focus:outline-none"
                      />
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <div className="space-y-2 border-t border-zinc-800/70 pt-3">
            <label className="flex items-start gap-2 text-xs text-zinc-300">
              <input type="checkbox" checked={draft.proposals.recommendations}
                onChange={(e) => setProposals({ recommendations: e.target.checked })} className="mt-0.5 h-4 w-4 accent-violet-600" />
              <span>
                Agents and Flight Deck may suggest goals
                <span className="block text-[11px] text-zinc-500">Every suggestion waits for the person it’s for (unless they let their agents add their own short-term goals).</span>
              </span>
            </label>
            {draft.proposals.recommendations && (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pl-6 text-xs text-zinc-400">
                <label className="flex items-center gap-1.5">
                  Each agent, at most
                  <input type="number" min={weekRange[0]} max={weekRange[1]} value={draft.proposals.agent_per_week}
                    onChange={(e) => setProposals({ agent_per_week: Number(e.target.value) })}
                    className="w-14 rounded-md border border-zinc-800 bg-zinc-950 px-1.5 py-0.5 text-center text-zinc-200" />
                  a week
                </label>
                <label className="flex items-center gap-1.5">
                  Flight Deck, at most
                  <input type="number" min={weekRange[0]} max={weekRange[1]} value={draft.proposals.deck_per_week}
                    onChange={(e) => setProposals({ deck_per_week: Number(e.target.value) })}
                    className="w-14 rounded-md border border-zinc-800 bg-zinc-950 px-1.5 py-0.5 text-center text-zinc-200" />
                  a week for each person
                </label>
              </div>
            )}
          </div>
          <div className="space-y-2 border-t border-zinc-800/70 pt-3">
            <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">What Flight Deck does on its own</div>
            {([
              ['weekly_estimate', 'Estimate progress once a week',
                'For goals with new work linked to them, through each user’s cheapest running agent. It never marks a goal done; at most it asks.'],
              ['conflict_check', 'Check new goals for conflicts',
                'When a goal is added, accepted or reworded, it is compared with the goals above and beside it. A likely conflict is flagged to the owner of the narrower goal; nothing is changed.'],
              ['goal_driven_candidates', 'Let stalled goals suggest work',
                'A short- or medium-term goal nothing has moved for a while becomes a candidate for Autonomous Work. Every proposal still waits behind each user’s autonomy settings.'],
            ] as const).map(([key, title, hint]) => (
              <label key={key} className="flex items-start gap-2 text-xs text-zinc-300">
                <input type="checkbox" checked={draft.automation[key]}
                  onChange={(e) => setAutomation({ [key]: e.target.checked })} className="mt-0.5 h-4 w-4 accent-violet-600" />
                <span>
                  {title}
                  <span className="block text-[11px] text-zinc-500">{hint}</span>
                </span>
              </label>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={save}
              disabled={saving || !dirty || badCap || badWeek}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
            {dirty && (
              <button
                onClick={() => { setDraft(saved); setError(null) }}
                disabled={saving}
                className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
              >
                Revert
              </button>
            )}
            {badCap && <span className="text-[11px] text-red-500">Limits go from {range[0]} to {range[1]}.</span>}
            {badWeek && <span className="text-[11px] text-red-500">Weekly suggestions go from {weekRange[0]} to {weekRange[1]}.</span>}
          </div>
        </>
      ) : null}

      {notice && (
        <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">{notice}</div>
      )}
      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>
      )}
    </div>
  )
}
