import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { Bot, Calculator, Check, CircleCheck, CircleDashed, Filter, Gavel, Pencil, Plus, Trash2, X } from 'lucide-react'
import { scheduleText } from '../../services/firstMate'
import { goalKind, newGoal, type Goal } from '../../utils/charterGoals'

// The parts of a factory's charter the owner reads and edits: what a run does, the
// goals a run is checked against (a checklist: each goal on or off, all on to start),
// and when it starts.

function useAutoHeight(value: string, max = 520) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight + 2, max)}px`
  }, [value, max])
  return ref
}

export function Field({ label, hint, meta, htmlFor, children }: { label: string; hint?: ReactNode; meta?: ReactNode; htmlFor?: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex items-end gap-2">
        <div className="min-w-0 flex-1">
          <label htmlFor={htmlFor} className="block text-[13px] font-semibold text-zinc-100">{label}</label>
          {hint && <p className="mt-0.5 text-[11px] leading-snug text-zinc-500">{hint}</p>}
        </div>
        {meta && <span className="shrink-0 text-[11px] tabular-nums text-zinc-500">{meta}</span>}
      </div>
      {children}
    </div>
  )
}

/** What a run does, in the owner's words: grows with what's written. */
export function CharterText({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const ref = useAutoHeight(value)
  const words = value.trim() ? value.trim().split(/\s+/).length : 0
  return (
    <Field label="What it does, each run" htmlFor="fm-charter" meta={`${words} words`}
      hint="In your words: what to look at, what to do, what never to do, and what to finish with.">
      <textarea id="fm-charter" ref={ref} value={value} onChange={(e) => onChange(e.target.value)} rows={4}
        className="w-full resize-none rounded-xl border border-[var(--fd-line)] bg-zinc-950 px-3.5 py-3 text-[14px] leading-relaxed text-zinc-100 placeholder-zinc-600 transition-colors focus:border-violet-500/50 focus:outline-none focus:ring-2 focus:ring-violet-500/15" />
    </Field>
  )
}

function KindChip({ text }: { text: string }) {
  const countable = goalKind(text) === 'countable'
  return (
    <span title={countable ? 'Countable: Captain can check this itself on each run' : 'Judgement: the reviewers decide this one'}
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium ${countable
        ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300' : 'bg-zinc-500/10 text-zinc-500'}`}>
      {countable ? <Calculator className="h-2.5 w-2.5" /> : <Gavel className="h-2.5 w-2.5" />}
      {countable ? 'Countable' : 'Judgement'}
    </span>
  )
}

function GoalRow({ goal, index, onChange, onRemove }: { goal: Goal; index: number; onChange: (g: Goal) => void; onRemove: () => void }) {
  const [editing, setEditing] = useState(!goal.text)
  const [draft, setDraft] = useState(goal.text)
  const ref = useAutoHeight(draft, 240)
  useEffect(() => { if (editing) ref.current?.focus() }, [editing, ref])
  const save = () => {
    const text = draft.replace(/\s+/g, ' ').trim()
    if (!text) onRemove()
    else onChange({ ...goal, text })
    setEditing(false)
  }
  return (
    <li className={`group flex items-start gap-3 rounded-xl border px-3 py-2.5 transition-all ${goal.on
      ? 'border-[var(--fd-line)] bg-[var(--fd-surface-raised)] hover:border-violet-500/30'
      : 'border-dashed border-[var(--fd-line)] bg-transparent'}`}>
      <button type="button" role="checkbox" aria-checked={goal.on} onClick={() => onChange({ ...goal, on: !goal.on })}
        aria-label={goal.on ? `Switch off: ${goal.text}` : `Switch on: ${goal.text}`}
        title={goal.on ? 'On: each run is checked against it' : 'Off: kept here, not checked'}
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border-2 transition-colors ${goal.on
          ? 'border-violet-600 bg-violet-600 text-white' : 'border-zinc-600 text-transparent hover:border-violet-500/60'}`}>
        <Check className="h-3.5 w-3.5" strokeWidth={3} />
      </button>
      <span className={`mt-0.5 w-4 shrink-0 text-right text-[11px] font-semibold tabular-nums ${goal.on ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-600'}`}>{index + 1}</span>
      <div className="min-w-0 flex-1">
        {editing ? (
          <textarea ref={ref} value={draft} rows={1} onChange={(e) => setDraft(e.target.value)} onBlur={save}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); save() }
              if (e.key === 'Escape') { e.stopPropagation(); setDraft(goal.text); setEditing(false); if (!goal.text) onRemove() }
            }}
            aria-label="Goal"
            className="w-full resize-none rounded-lg border border-violet-500/40 bg-zinc-950 px-2 py-1 text-[13.5px] leading-relaxed text-zinc-100 focus:outline-none focus:ring-2 focus:ring-violet-500/15" />
        ) : (
          <button type="button" onClick={() => { setDraft(goal.text); setEditing(true) }} title="Change it"
            className={`block w-full text-left text-[13.5px] leading-relaxed ${goal.on ? 'text-zinc-100' : 'text-zinc-500'}`}>
            {goal.text}
          </button>
        )}
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          {goal.on ? <KindChip text={goal.text} /> : <span className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">Off: not checked</span>}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
        {!editing && (
          <button type="button" onClick={() => { setDraft(goal.text); setEditing(true) }} aria-label={`Change: ${goal.text}`} title="Change it"
            className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
            <Pencil className="h-3.5 w-3.5" />
          </button>
        )}
        <button type="button" onClick={onRemove} aria-label={`Remove: ${goal.text}`} title="Remove it"
          className="flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-300">
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </li>
  )
}

/** The goals a run is checked against, as a checklist: all on to start. */
export function GoalList({ goals, onChange }: { goals: Goal[]; onChange: (goals: Goal[]) => void }) {
  const [adding, setAdding] = useState('')
  const on = goals.filter((g) => g.on).length
  const add = () => {
    const text = adding.replace(/\s+/g, ' ').trim()
    if (!text) return
    onChange([...goals, newGoal(text)])
    setAdding('')
  }
  const allOn = on === goals.length
  return (
    <Field label="Done when"
      hint={goals.length ? 'Each run is checked against the goals that are on. Switch one off to leave it out; it stays here.'
        : 'No goals yet: Bat writes its own checks from the charter. Add the ones you care about.'}
      meta={goals.length ? (
        <span className="inline-flex items-center gap-2">
          <span className={`rounded-full px-2 py-0.5 font-medium ${on ? 'bg-violet-500/12 text-violet-700 dark:text-violet-300' : 'bg-zinc-500/10'}`}>{on} of {goals.length} on</span>
          <button type="button" onClick={() => onChange(goals.map((g) => ({ ...g, on: !allOn })))}
            className="font-medium text-violet-700 hover:underline dark:text-violet-300">{allOn ? 'All off' : 'All on'}</button>
        </span>
      ) : undefined}>
      <ul className="space-y-2" aria-label="Goals">
        {goals.map((g, i) => (
          <GoalRow key={g.id} goal={g} index={i}
            onChange={(next) => onChange(goals.map((x) => (x.id === g.id ? next : x)))}
            onRemove={() => onChange(goals.filter((x) => x.id !== g.id))} />
        ))}
      </ul>
      <div className="mt-2 flex items-center gap-2 rounded-xl border border-dashed border-[var(--fd-line)] px-3 py-1.5 transition-colors focus-within:border-violet-500/50">
        <Plus className="h-4 w-4 shrink-0 text-zinc-500" />
        <input value={adding} onChange={(e) => setAdding(e.target.value)} placeholder="Add a goal… (e.g. at least 100 rows, no repeated Email)"
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add() } }} aria-label="Add a goal"
          className="h-8 min-w-0 flex-1 bg-transparent text-[13px] text-zinc-100 placeholder-zinc-600 focus:outline-none" />
        {adding.trim() && (
          <button type="button" onClick={add} className="inline-flex h-7 items-center rounded-full bg-violet-600 px-3 text-[11px] font-semibold text-white hover:bg-violet-500">Add</button>
        )}
      </div>
    </Field>
  )
}

/** A live version's goals, read only: what's checked, then what was switched off. */
export function GoalChecklist({ checks }: { checks: { contract: string[]; off?: string[] } }) {
  return (
    <ul className="min-w-0 flex-1 space-y-1.5">
      {checks.contract.map((c) => (
        <li key={`on:${c}`} className="flex items-start gap-2 text-zinc-200">
          <CircleCheck className="mt-0.5 h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400" />
          <span className="min-w-0 flex-1 leading-relaxed">{c}</span>
          <KindChip text={c} />
        </li>
      ))}
      {(checks.off ?? []).map((c) => (
        <li key={`off:${c}`} className="flex items-start gap-2 text-zinc-500" title="Switched off: not checked">
          <CircleDashed className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="min-w-0 flex-1 leading-relaxed">{c}</span>
          <span className="shrink-0 text-[10px] font-medium uppercase tracking-wide">Off</span>
        </li>
      ))}
    </ul>
  )
}

const PRESETS: { label: string; value: string }[] = [
  { label: 'Only when I press Run now', value: '' },
  { label: 'Weekdays 08:00', value: 'weekdays 08:00' },
  { label: 'Every day 08:00', value: 'daily 08:00' },
  { label: 'Mondays 09:00', value: 'weekly mon 09:00' },
]

/** When it starts on its own: a few common choices, or anything the scheduler reads. */
export function SchedulePicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const custom = !PRESETS.some((p) => p.value === value.trim())
  return (
    <Field label="When it runs" htmlFor="fm-schedule"
      hint={<>{value.trim() ? scheduleText(value) : 'Only when you press Run now'}. It starts on its own while your First Mate is on.</>}>
      <div className="flex flex-wrap items-center gap-1.5">
        {PRESETS.map((p) => {
          const active = p.value === value.trim()
          return (
            <button key={p.label} type="button" onClick={() => onChange(p.value)} aria-pressed={active}
              className={`inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors ${active
                ? 'border-violet-500/50 bg-violet-500/10 text-violet-700 dark:text-violet-200' : 'border-[var(--fd-line)] text-zinc-400 hover:text-zinc-200'}`}>
              {active && <Check className="h-3 w-3" />}{p.label}
            </button>
          )
        })}
        <div className={`flex h-8 items-center gap-1 rounded-full border px-3 transition-colors focus-within:border-violet-500/50 ${custom && value.trim() ? 'border-violet-500/50 bg-violet-500/10' : 'border-[var(--fd-line)]'}`}>
          <input id="fm-schedule" value={custom ? value : ''} onChange={(e) => onChange(e.target.value)} placeholder="or: every 6 h, weekly mon,thu 07:30"
            className="w-56 bg-transparent text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none" />
          {custom && value.trim() && (
            <button type="button" onClick={() => onChange('')} aria-label="Clear" className="text-zinc-500 hover:text-zinc-200"><X className="h-3 w-3" /></button>
          )}
        </div>
      </div>
    </Field>
  )
}


export type HowItRuns = 'bat' | 'funnel'

const HOW: { value: HowItRuns; label: string; icon: typeof Bot; hint: string }[] = [
  { value: 'bat', label: 'Agents', icon: Bot,
    hint: 'Workers search and reason in long turns. For research, documents, anything that isn’t a list.' },
  { value: 'funnel', label: 'List funnel', icon: Filter,
    hint: 'Code searches, reads and checks; a model judges each item once. Far fewer tokens for lists, files you give it included.' },
]

/** How a factory's runs do the work: Bat's agents, or the list funnel (with or without
 *  an agent to finish what's short, and with or without rows on a looser reading of
 *  the soft rules when a group runs short). */
export function RunnerPicker({ value, leftovers, loosen, onChange, listy, searches, onSearches, expert, agents, onExpert }: {
  value: HowItRuns
  leftovers: boolean
  loosen: boolean
  onChange: (value: HowItRuns, leftovers: boolean, loosen: boolean) => void
  /** The searches a run may make, as typed ("" for the usual amount); with its setter, the field shows. */
  searches?: string
  onSearches?: (value: string) => void
  /** The agent that works beside its runs as their expert ("" for none), the owner's agents to
   *  pick from, and its setter; with the setter, the field shows. */
  expert?: string
  agents?: { slug: string; name: string }[]
  onExpert?: (slug: string) => void
  /** Its goals ask for an amount: it reads as a list. */
  listy: boolean
}) {
  return (
    <Field label="How it runs"
      hint={value === 'funnel' && !listy
        ? 'Works when the result is a list; a run that isn’t one goes to agents anyway.'
        : 'Agents for open work; the list funnel when the result is a list of rows.'}>
      <div role="radiogroup" aria-label="How it runs" className="grid gap-2 @2xl:grid-cols-2">
        {HOW.map((h) => {
          const on = value === h.value
          const Icon = h.icon
          return (
            <button key={h.value} type="button" role="radio" aria-checked={on} onClick={() => onChange(h.value, leftovers, loosen)}
              className={`flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-colors ${on
                ? 'border-violet-500/50 bg-violet-500/10' : 'border-[var(--fd-line)] hover:bg-[var(--fd-hover)]'}`}>
              <span className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${on
                ? 'bg-violet-500 text-white' : 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]'}`}>
                <Icon className="h-3.5 w-3.5" />
              </span>
              <span className="min-w-0">
                <span className={`block text-xs font-semibold ${on ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-200'}`}>{h.label}</span>
                <span className="mt-0.5 block text-[11px] leading-snug text-zinc-500">{h.hint}</span>
              </span>
            </button>
          )
        })}
      </div>
      {value === 'funnel' && (
        <>
          <label className="mt-2 flex cursor-pointer items-start gap-2 text-[11px] text-zinc-400">
            <input type="checkbox" checked={leftovers} onChange={(e) => onChange(value, e.target.checked, loosen)}
              className="mt-0.5 h-3.5 w-3.5 accent-violet-600" />
            <span>Finish what’s short with an agent: it checks only the items that came close, in a small context.</span>
          </label>
          <label className="mt-1.5 flex cursor-pointer items-start gap-2 text-[11px] text-zinc-400">
            <input type="checkbox" checked={loosen} onChange={(e) => onChange(value, leftovers, e.target.checked)}
              className="mt-0.5 h-3.5 w-3.5 accent-violet-600" />
            <span>When a group runs short, loosen the soft rules: rows that miss only evidence many never publish (recent news, a signal, a rating) fill it after every strict row. The run says which rules it reads as soft. Not when the charter itself accepts a short list with the reason given: then every rule is kept as written.</span>
          </label>
          {onSearches && (
            <label className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px] text-zinc-400">
              <span>Searches a run may make</span>
              <input inputMode="numeric" value={searches ?? ''} placeholder="Automatic" aria-label="Searches a run may make"
                onChange={(e) => onSearches(e.target.value.replace(/\D/g, '').slice(0, 6))}
                className="h-7 w-24 rounded-lg border border-[var(--fd-line)] bg-zinc-950 px-2 text-xs text-zinc-100 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
              <span className="basis-full text-zinc-500">Left empty: 1,000, or 12 for each row the list asks for when that is more. A run stops searching when they are used.</span>
            </label>
          )}
          {onExpert && (
            <label className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px] text-zinc-400">
              <span>Expert agent</span>
              <select value={expert ?? ''} aria-label="Expert agent" onChange={(e) => onExpert(e.target.value)}
                className="h-7 min-w-0 max-w-full rounded-lg border border-[var(--fd-line)] bg-zinc-950 px-2 text-xs text-zinc-100 focus:border-violet-500/50 focus:outline-none">
                <option value="">None</option>
                {expert && !(agents ?? []).some((a) => a.slug === expert) && <option value={expert}>{expert} (not found)</option>}
                {(agents ?? []).map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
              </select>
              <span className="basis-full text-zinc-500">One of your agents works beside each run, in a new session of its own: it adds what it knows before the run starts, settles what the charter leaves open as you would, and writes the new searches. A run wakes it when it sleeps. What it decided for you is listed in the run’s summary.</span>
            </label>
          )}
        </>
      )}
    </Field>
  )
}
