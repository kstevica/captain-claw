import { useCallback, useEffect, useMemo, useState } from 'react'
import { Bot, Check, ChevronDown, Compass, Eye, Inbox, Loader2, Pencil, Plus, RefreshCw, UserRound, Users, X } from 'lucide-react'
import {
  agentsUpdatedLabel,
  changesOf,
  createGoal,
  draftOf,
  editGoal,
  emptyDraft,
  getCourse,
  getPreview,
  goalAction,
  savePrefs,
  shortDate,
  DEFAULT_TEXT_CAPS,
  HORIZON_LABELS,
  HORIZONS,
  type CoursePreview,
  type CourseResponse,
  type Goal,
  type GoalAction,
  type GoalDraft,
  type Horizon,
  type Scope,
} from '../services/course'
import { GoalForm } from '../components/course/GoalForm'
import { DeckCourseSettings } from '../components/course/DeckCourseSettings'
import { GoalDrawer } from '../components/course/GoalDrawer'

// Course: what the work is for. Three bands — the deck's goals (set by an
// admin), the user's own, and each of their agents' — every goal with a
// horizon, arranged as a tree ("serves"). Flight Deck writes the live ones
// into every agent's prompt; goals guide priorities and never grant
// permission. See docs/course-plan.md.
//
// `kiosk` — shown in the locked Simple layout's dialog: the deck's goals and
// the admin's settings are read-only there.

type Editing =
  | { mode: 'new'; scope: Scope; agentRef: string }
  | { mode: 'edit'; goal: Goal }
  | { mode: 'accept'; goal: Goal }   // a suggestion, edited before it's accepted

// Suggestions wait in their own strip; the bands show the course itself.
const OPEN = ['active', 'paused']
const FINISHED = ['achieved', 'dropped']
const ACTION_DONE: Record<GoalAction, string> = {
  pause: 'Paused', resume: 'Resumed', achieve: 'Marked done', drop: 'Dropped', reopen: 'Reopened',
  review: 'Still on course', accept: 'Accepted', decline: 'Declined',
}

export function CoursePage({ kiosk = false }: { kiosk?: boolean }) {
  const [data, setData] = useState<CourseResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [editing, setEditing] = useState<Editing | null>(null)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [opened, setOpened] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setData(await getCourse())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { void load() }, [load])

  const all = useMemo(() => (data
    ? [...data.goals.deck, ...data.goals.mine, ...Object.values(data.goals.agents).flat()]
    : []), [data])
  const byId = useMemo(() => new Map(all.map((g) => [g.id, g])), [all])
  const caps = data?.text_caps ?? DEFAULT_TEXT_CAPS
  const deckEditable = !!data?.is_admin && !kiosk

  const act = async (g: Goal, action: GoalAction) => {
    if (action === 'drop' && !window.confirm(`Drop “${g.title}”? You can reopen it later.`)) return
    setBusy(g.id); setNotice(null); setError(null)
    try {
      const r = await goalAction(g.id, action)
      setNotice(`${ACTION_DONE[action]}: ${g.title} — ${agentsUpdatedLabel(r.agents_updated)}.`)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const save = async (d: GoalDraft) => {
    if (!editing) return
    setSaving(true); setFormError(null); setNotice(null)
    try {
      let r
      if (editing.mode === 'accept') {
        await editGoal(editing.goal.id, d)
        r = await goalAction(editing.goal.id, 'accept')
      } else {
        r = editing.mode === 'edit'
          ? await editGoal(editing.goal.id, d)
          : await createGoal(editing.scope, d, editing.agentRef)
      }
      setNotice(`${editing.mode === 'edit' ? 'Saved' : editing.mode === 'accept' ? 'Accepted' : 'Added'}: ${r.goal.title} — ${agentsUpdatedLabel(r.agents_updated)}.`)
      setEditing(null)
      await load()
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const form = (scope: Scope, agentRef: string) => {
    if (!data || !editing || editing.mode === 'accept') return null
    const target = editing.mode === 'edit' ? editing.goal : null
    const at = editing.mode === 'edit'
      ? { scope: editing.goal.scope, agentRef: editing.goal.agent_ref }
      : { scope: editing.scope, agentRef: editing.agentRef }
    if (at.scope !== scope || at.agentRef !== agentRef) return null
    const allowed = data.settings.horizons[scope]
    const first = (allowed[0] ?? 'long') as Horizon
    return (
      <GoalForm
        key={target?.id ?? `new-${scope}-${agentRef}`}
        scope={scope}
        agentRef={agentRef}
        allowed={allowed}
        initial={target ? draftOf(target) : emptyDraft(first)}
        all={all}
        selfId={target?.id}
        caps={caps}
        saving={saving}
        error={formError}
        onSave={(d) => void save(d)}
        onCancel={() => { setEditing(null); setFormError(null) }}
      />
    )
  }

  const proposals = useMemo(() => all.filter((g) => g.status === 'proposed'), [all])
  const agentName = useMemo(() => new Map((data?.agents ?? []).map((a) => [a.ref, a.name])), [data])
  const [prefSaving, setPrefSaving] = useState(false)
  const setAutoAccept = async (on: boolean) => {
    setPrefSaving(true); setError(null)
    try {
      await savePrefs({ auto_accept_agent_short: on })
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPrefSaving(false)
    }
  }

  const band = (scope: Scope, goals: Goal[], agentRef = '') => {
    if (!data) return null
    const allowed = data.settings.horizons[scope]
    const canAdd = (scope !== 'instance' || deckEditable) && allowed.length > 0
    return (
      <Band
        goals={goals}
        byId={byId}
        caps={data.settings.caps}
        busy={busy}
        readOnly={scope === 'instance' && !deckEditable}
        off={allowed.length === 0}
        onAdd={canAdd && !editing ? () => { setFormError(null); setEditing({ mode: 'new', scope, agentRef }) } : undefined}
        onEdit={(g) => { setFormError(null); setEditing({ mode: 'edit', goal: g }) }}
        onOpen={(g) => setOpened(g.id)}
        onAction={(g, a) => void act(g, a)}
        form={form(scope, agentRef)}
      />
    )
  }

  return (
    <div className="@container h-full overflow-y-auto bg-zinc-950 text-zinc-200">
      <div className="max-w-5xl mx-auto px-6 py-8 pb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 shrink-0 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center">
            <Compass className="h-5 w-5 text-violet-500" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-xl font-semibold text-zinc-100">Course</h1>
            <p className="text-sm text-zinc-500">
              What your agents’ work is for. Every agent working for you gets the deck’s goals, yours and its
              own, and uses them to choose what matters. A goal is never permission to do anything.
            </p>
          </div>
          <button
            onClick={() => void load()}
            disabled={loading}
            title="Reload"
            className="shrink-0 rounded p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {notice && (
          <div className="mb-4 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">{notice}</div>
        )}
        {error && (
          <div className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">{error}</div>
        )}
        {!data && loading && (
          <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>
        )}

        {data && (
          <div className="space-y-6">
            {proposals.length > 0 && (
              <Section icon={Inbox} title={`Waiting for you (${proposals.length})`}
                hint="Goals your agents and Flight Deck suggest. Nothing goes on your course until you accept it.">
                <div className="space-y-2">
                  {proposals.map((p) => (
                    <ProposalCard
                      key={p.id}
                      goal={p}
                      old={p.supersedes_id ? byId.get(p.supersedes_id) : undefined}
                      forWhom={p.scope === 'agent' ? `For ${agentName.get(p.agent_ref) || 'an agent'}` : 'For you'}
                      titleOf={(id) => byId.get(id)?.title ?? ''}
                      busy={busy === p.id}
                      onAccept={() => void act(p, 'accept')}
                      onDecline={() => void act(p, 'decline')}
                      onEdit={() => { setFormError(null); setEditing({ mode: 'accept', goal: p }) }}
                      form={editing?.mode === 'accept' && editing.goal.id === p.id ? (
                        <GoalForm
                          scope={p.scope}
                          agentRef={p.agent_ref}
                          allowed={data.settings.horizons[p.scope]}
                          initial={draftOf(p)}
                          all={all}
                          selfId={p.id}
                          caps={caps}
                          saving={saving}
                          error={formError}
                          onSave={(d) => void save(d)}
                          onCancel={() => { setEditing(null); setFormError(null) }}
                          submitLabel="Save and accept"
                        />
                      ) : null}
                    />
                  ))}
                </div>
              </Section>
            )}

            <Section
              icon={Users}
              title="The deck"
              hint={deckEditable
                ? 'Set by you as admin. Every agent on this deck works toward these, and so does a shared agent when one of its members talks to it.'
                : 'Set by the deck’s admin. Every agent on this deck works toward these.'}
            >
              {band('instance', data.goals.deck)}
            </Section>

            <Section icon={UserRound} title="You" hint="Your own goals. Every agent working for you gets them; members of your shared agents never do.">
              {band('user', data.goals.mine)}
            </Section>

            <Section icon={Bot} title="Your agents" hint="Goals for one agent only. Only that agent gets them.">
              {data.recommendations && (
                <label className="mb-3 flex items-start gap-2 text-xs text-zinc-400">
                  <input
                    type="checkbox"
                    checked={data.prefs.auto_accept_agent_short}
                    disabled={prefSaving}
                    onChange={(e) => void setAutoAccept(e.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 accent-violet-600"
                  />
                  <span>
                    Let my agents add their own short-term goals without asking
                    <span className="block text-[11px] text-zinc-500">You’re told on your Line each time, and can change or drop them here. Everything else they suggest still waits for you.</span>
                  </span>
                </label>
              )}
              {data.agents.length === 0 ? (
                <p className="text-xs text-zinc-500">You have no agents yet.</p>
              ) : (
                <div className="space-y-4">
                  {data.agents.map((a) => (
                    <div key={a.ref} className="rounded-lg border border-zinc-800/80 bg-zinc-900/30 p-3">
                      <div className="mb-2 text-sm font-medium text-zinc-200">{a.name}</div>
                      {band('agent', data.goals.agents[a.ref] ?? [], a.ref)}
                    </div>
                  ))}
                </div>
              )}
            </Section>

            <PreviewPanel agents={data.agents} version={all.length + (notice ?? '')} />

            {data.is_admin && !kiosk && <DeckCourseSettings onSaved={() => void load()} />}
          </div>
        )}
        {opened && <GoalDrawer goalId={opened} onClose={() => setOpened(null)} onChanged={() => void load()} />}
      </div>
    </div>
  )
}

function Section({ icon: Icon, title, hint, children }: {
  icon: typeof Users; title: string; hint: string; children: React.ReactNode
}) {
  return (
    <section>
      <div className="mb-2 flex items-start gap-2">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
        <div>
          <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
          <p className="text-xs text-zinc-500">{hint}</p>
        </div>
      </div>
      {children}
    </section>
  )
}

function Band({ goals, byId, caps, busy, readOnly, off, onAdd, onEdit, onOpen, onAction, form }: {
  goals: Goal[]
  byId: Map<string, Goal>
  caps: Record<Horizon, number>
  busy: string | null
  readOnly: boolean
  off: boolean
  onAdd?: () => void
  onEdit: (g: Goal) => void
  onOpen: (g: Goal) => void
  onAction: (g: Goal, a: GoalAction) => void
  form: React.ReactNode
}) {
  const open = goals.filter((g) => OPEN.includes(g.status))
  const finished = goals.filter((g) => FINISHED.includes(g.status))
  return (
    <div className="space-y-2">
      {form}
      {HORIZONS.map((h) => {
        const here = open.filter((g) => g.horizon === h)
        if (here.length === 0) return null
        const active = here.filter((g) => g.status === 'active').length
        return (
          <div key={h}>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
              {HORIZON_LABELS[h]} <span className="normal-case tracking-normal text-zinc-600">· {active} of {caps[h]} active</span>
            </div>
            <div className="space-y-1.5">
              {here.map((g) => (
                <GoalCard key={g.id} goal={g} parent={byId.get(g.parent_id)} busy={busy === g.id}
                  readOnly={readOnly || !g.can_edit} onEdit={() => onEdit(g)} onOpen={() => onOpen(g)}
                  onAction={(a) => onAction(g, a)} />
              ))}
            </div>
          </div>
        )
      })}
      {open.length === 0 && !form && (
        <p className="text-xs text-zinc-500">
          {off ? 'The admin has switched goals off at this level.' : readOnly ? 'No goals here yet.' : 'No goals yet.'}
        </p>
      )}
      {onAdd && (
        <button onClick={onAdd} className="inline-flex items-center gap-1 rounded-md border border-zinc-800 px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">
          <Plus className="h-3.5 w-3.5" /> Add goal
        </button>
      )}
      {finished.length > 0 && (
        <details className="group">
          <summary className="flex cursor-pointer list-none items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300">
            <ChevronDown className="h-3 w-3 transition-transform group-open:rotate-180" /> Finished ({finished.length})
          </summary>
          <div className="mt-1.5 space-y-1.5">
            {finished.map((g) => (
              <GoalCard key={g.id} goal={g} parent={byId.get(g.parent_id)} busy={busy === g.id}
                readOnly={readOnly || !g.can_edit} onEdit={() => onEdit(g)} onOpen={() => onOpen(g)}
                onAction={(a) => onAction(g, a)} />
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

const STATUS_CHIP: Partial<Record<Goal['status'], string>> = {
  proposed: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  paused: 'bg-zinc-700/40 text-zinc-400',
  achieved: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  dropped: 'bg-zinc-700/40 text-zinc-500',
}

function GoalCard({ goal: g, parent, busy, readOnly, onEdit, onOpen, onAction }: {
  goal: Goal
  parent?: Goal
  busy: boolean
  readOnly: boolean
  onEdit: () => void
  onOpen: () => void
  onAction: (a: GoalAction) => void
}) {
  const f = g.flags
  const btn = 'rounded px-2 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50'
  const actions: [GoalAction | 'edit', string][] = readOnly ? [] : (
    g.status === 'active' ? [['review', 'Still on course'], ['edit', 'Edit'], ['pause', 'Pause'], ['achieve', 'Done'], ['drop', 'Drop']]
      : g.status === 'paused' ? [['resume', 'Resume'], ['edit', 'Edit'], ['achieve', 'Done'], ['drop', 'Drop']]
        : g.status === 'proposed' ? [['accept', 'Accept'], ['decline', 'Decline']]
          : [['reopen', 'Reopen']])
  const shown = actions.filter(([a]) => a !== 'review' || f?.due)
  return (
    <div className={`rounded-md border px-3 py-2 ${f?.stale ? 'border-red-500/40' : f?.due ? 'border-amber-500/40' : 'border-zinc-800'} bg-zinc-950`}>
      <div className="flex flex-wrap items-start gap-x-2 gap-y-1">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={onOpen}
              title="Open: the work that moved it, its history, Plan a route"
              className={`text-left text-sm hover:underline ${g.status === 'dropped' ? 'text-zinc-500 line-through' : 'text-zinc-100'}`}
            >
              {g.title}
            </button>
            {STATUS_CHIP[g.status] && (
              <span className={`rounded px-1.5 py-px text-[10px] font-medium capitalize ${STATUS_CHIP[g.status]}`}>{g.status}</span>
            )}
          </div>
          {g.success && <p className="mt-0.5 text-xs text-zinc-400"><span className="text-zinc-500">Done when:</span> {g.success}</p>}
          {g.detail && <p className="mt-0.5 line-clamp-2 text-xs text-zinc-500" title={g.detail}>{g.detail}</p>}
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-zinc-500">
            {parent && <span>Serves: {parent.title}</span>}
            {g.due_at && <span>Due {shortDate(g.due_at)}</span>}
            {g.status === 'active' && g.review_at && <span>Review by {shortDate(g.review_at)}</span>}
            {g.progress != null && <span>{Math.round(g.progress * 100)}% along</span>}
            {g.status === 'active' && <span>{g.last_moved ? `Last moved ${shortDate(g.last_moved)}` : 'No work linked yet'}</span>}
          </div>
          {f && (f.stale || f.due || f.horizon_retired || f.serves_nothing || f.parent_gone) && (
            <div className="mt-1 flex flex-wrap gap-1">
              {f.stale ? (
                <Flag tone="red">Out of your agents’ prompts until someone says it’s still on course</Flag>
              ) : f.due ? (
                <Flag tone="amber">Review due: it leaves your agents’ prompts {g.grace_days ?? 3} days after its review date</Flag>
              ) : null}
              {f.horizon_retired && <Flag tone="amber">This level no longer uses {HORIZON_LABELS[g.horizon].toLowerCase()} goals</Flag>}
              {f.parent_gone && <Flag tone="zinc">The goal it served is finished</Flag>}
              {f.serves_nothing && !f.parent_gone && <Flag tone="zinc">Serves nothing above it</Flag>}
            </div>
          )}
        </div>
        {shown.length > 0 && (
          <div className="flex shrink-0 flex-wrap items-center gap-0.5">
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
            {shown.map(([a, label]) => (
              <button key={a} disabled={busy} onClick={() => (a === 'edit' ? onEdit() : onAction(a))}
                className={a === 'review' ? `${btn} text-amber-600 dark:text-amber-300` : btn}>
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Flag({ tone, children }: { tone: 'red' | 'amber' | 'zinc'; children: React.ReactNode }) {
  const cls = tone === 'red' ? 'bg-red-500/10 text-red-700 dark:text-red-300'
    : tone === 'amber' ? 'bg-amber-500/10 text-amber-800 dark:text-amber-200' : 'bg-zinc-800/60 text-zinc-500'
  return <span className={`rounded px-1.5 py-0.5 text-[10px] ${cls}`}>{children}</span>
}

// What an agent receives: its owner's-turn block and its members'-turn block,
// exactly as Flight Deck writes them. Loaded when opened, and again after a change.
function PreviewPanel({ agents, version }: { agents: CourseResponse['agents']; version: string | number }) {
  const [open, setOpen] = useState(false)
  const [ref, setRef] = useState(agents[0]?.ref ?? '')
  const [compact, setCompact] = useState(false)
  const [preview, setPreview] = useState<CoursePreview | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    let gone = false
    getPreview(ref)
      .then((p) => { if (!gone) { setPreview(p); setError(null) } })
      .catch((e) => { if (!gone) setError(e instanceof Error ? e.message : String(e)) })
    return () => { gone = true }
  }, [open, ref, version])

  const mode = compact ? 'compact' : 'full'
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 text-left">
        <Eye className="h-4 w-4 shrink-0 text-violet-400" />
        <span className="flex-1 text-sm font-semibold text-zinc-200">What your agents receive</span>
        <ChevronDown className={`h-4 w-4 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-3 space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <select value={ref} onChange={(e) => setRef(e.target.value)}
              className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1 text-zinc-200">
              {agents.map((a) => <option key={a.ref} value={a.ref}>{a.name}</option>)}
              <option value="">A new agent of yours</option>
            </select>
            <div className="flex rounded-md border border-zinc-800 p-0.5">
              {(['full', 'compact'] as const).map((m) => (
                <button key={m} onClick={() => setCompact(m === 'compact')}
                  className={`rounded px-2 py-0.5 ${mode === m ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500'}`}>
                  {m === 'full' ? 'Full' : 'Short (small models)'}
                </button>
              ))}
            </div>
          </div>
          {error && <p className="text-xs text-red-500">{error}</p>}
          {preview && (
            <div className="grid gap-3 @3xl:grid-cols-2">
              <PreviewBlock label="When you talk to it" text={preview.owner[mode]} />
              <PreviewBlock label="When a member of a shared agent talks to it" text={preview.member[mode]} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function PreviewBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="max-h-80 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-2 text-xs leading-relaxed text-zinc-300">
        {text || <span className="text-zinc-600">Nothing: no live goals reach it.</span>}
      </div>
    </div>
  )
}

// A suggestion: what it is, who suggested it and why — for a change, what it
// changes — with Accept, Edit and accept, and Decline.
function ProposalCard({ goal: g, old, forWhom, titleOf, busy, onAccept, onDecline, onEdit, form }: {
  goal: Goal
  old?: Goal
  forWhom: string
  titleOf: (id: string) => string
  busy: boolean
  onAccept: () => void
  onDecline: () => void
  onEdit: () => void
  form: React.ReactNode
}) {
  const changes = changesOf(g, old, titleOf)
  const btn = 'inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs disabled:opacity-50'
  return (
    <div className="rounded-lg border border-sky-500/30 bg-sky-500/[0.05] p-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-sky-700 dark:text-sky-300">
        {forWhom} · {HORIZON_LABELS[g.horizon]} · {old ? 'a change' : 'a new goal'} suggested by {g.proposer || 'an agent'}
      </div>
      {form ?? (
        <>
          <div className="mt-1 text-sm text-zinc-100">{g.title}</div>
          {g.success && <p className="mt-0.5 text-xs text-zinc-400"><span className="text-zinc-500">Done when:</span> {g.success}</p>}
          {changes.length > 0 && (
            <ul className="mt-1.5 space-y-0.5 text-xs">
              {changes.map(([label, was, becomes]) => (
                <li key={label} className="text-zinc-400">
                  <span className="text-zinc-500">{label}:</span> <span className="line-through decoration-zinc-600">{was}</span> → <span className="text-zinc-200">{becomes}</span>
                </li>
              ))}
            </ul>
          )}
          {g.rationale && <p className="mt-1 text-xs italic text-zinc-500">“{g.rationale}”</p>}
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <button onClick={onAccept} disabled={busy} className={`${btn} bg-violet-600 text-white hover:bg-violet-500`}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Accept
            </button>
            <button onClick={onEdit} disabled={busy} className={`${btn} border border-zinc-700 text-zinc-300 hover:bg-zinc-800`}>
              <Pencil className="h-3.5 w-3.5" /> Edit and accept
            </button>
            <button onClick={onDecline} disabled={busy} className={`${btn} text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300`}>
              <X className="h-3.5 w-3.5" /> Decline
            </button>
          </div>
        </>
      )}
    </div>
  )
}
