import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle, BellDot, Bot, Check, ChevronDown, ChevronRight, Clock, CornerDownRight, Eye, Loader2, MoreHorizontal,
  PanelRightOpen, Pause, Pencil, Play, Plus, RefreshCw, RotateCcw, Sparkles, Trash2, UserRound, Users, X,
} from 'lucide-react'
import {
  agentsUpdatedLabel,
  changesOf,
  createGoal,
  dismissConflict,
  draftOf,
  editGoal,
  eligibleParents,
  emptyDraft,
  getCourse,
  getPreview,
  goalAction,
  measureText,
  progressOf,
  savePrefs,
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
import { ProgressRing } from '../components/course/ProgressRing'
import { QueueMenu, type MenuEntry } from '../components/agents/queue/QueueMenu'
import { agentInitials } from '../components/agents/AgentPicture'

// Course: what the work is for. Three bands — the deck's goals (set by an
// admin), the user's own, and each of their agents' — every goal with a
// horizon, arranged as a tree ("serves"). Flight Deck writes the live ones
// into every agent's prompt; goals guide priorities and never grant
// permission. See docs/course-plan.md.
//
// On top, "Needs you": suggestions to answer, and goals that went off course
// or are due a review. Each band is one card of goals grouped by horizon; a
// goal opens to its drawer (the work that moved it, its history, Plan a
// route) and keeps its actions in a ⋯ menu.
//
// `kiosk` — shown in the locked Simple layout's dialog: the deck's goals and
// the admin's settings are read-only there. `onClose` — in a dialog: a close
// button in the header, and Escape closes it (not while a goal is open in
// the drawer, or a goal is being written).

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
// A date as rows show it: "Nov 21", with the year only when it isn't this one.
function day(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleDateString(undefined, d.getFullYear() === new Date().getFullYear()
    ? { day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short', year: 'numeric' })
}

// Agents with no goals past this many fold under "N more agents".
const QUIET_AGENTS_SHOWN = 4

export function CoursePage({ kiosk = false, onClose }: { kiosk?: boolean; onClose?: () => void }) {
  const [data, setData] = useState<CourseResponse | null>(null)
  const [rev, setRev] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [editing, setEditing] = useState<Editing | null>(null)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [opened, setOpened] = useState<string | null>(null)
  const [scrolled, setScrolled] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setData(await getCourse())
      setRev((r) => r + 1)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { void load() }, [load])

  // A word on what just happened, for a few seconds.
  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(null), 5000)
    return () => clearTimeout(t)
  }, [notice])

  // Escape closes the dialog — unless the drawer is open (its Escape) or a
  // goal is being written (what's typed stays).
  const holdRef = useRef(false)
  holdRef.current = !!opened || !!editing
  useEffect(() => {
    if (!onClose) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented && !holdRef.current) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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

  const dismiss = async (conflictId: string) => {
    setBusy(conflictId); setNotice(null); setError(null)
    try {
      await dismissConflict(conflictId)
      setNotice('Marked “not a conflict”. Flight Deck won’t flag that pair again.')
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

  const cancelEdit = () => { setEditing(null); setFormError(null) }

  // The form, where it belongs: a new goal over its band, an edited goal in
  // place of its row.
  const formFor = (scope: Scope, agentRef: string, target: Goal | null) => {
    if (!data) return null
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
        onCancel={cancelEdit}
      />
    )
  }
  const newForm = (scope: Scope, agentRef: string) => (
    editing?.mode === 'new' && editing.scope === scope && editing.agentRef === agentRef ? formFor(scope, agentRef, null) : null
  )
  const editForm = (g: Goal) => (
    editing?.mode === 'edit' && editing.goal.id === g.id ? formFor(g.scope, g.agent_ref, g) : null
  )

  const proposals = useMemo(() => all.filter((g) => g.status === 'proposed'), [all])
  // What wants a look: off course, or (active) due a review or already out of the prompts.
  const attention = useMemo(() => all.filter((g) => g.flags?.off_course
    || (g.status === 'active' && (g.flags?.due || g.flags?.stale))), [all])
  const agentName = useMemo(() => new Map((data?.agents ?? []).map((a) => [a.ref, a.name])), [data])
  const ownerOf = (g: Goal) => (g.scope === 'agent' ? agentName.get(g.agent_ref) || 'An agent'
    : g.scope === 'instance' ? 'The deck' : 'You')
  const canChange = (g: Goal) => !!g.can_edit && !(g.scope === 'instance' && !deckEditable)

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

  const startNew = (scope: Scope, agentRef = '') => () => { setFormError(null); setEditing({ mode: 'new', scope, agentRef }) }
  const addable = (scope: Scope) => !!data && !editing
    && (scope !== 'instance' || deckEditable) && data.settings.horizons[scope].length > 0

  const list = (scope: Scope, goals: Goal[]) => (
    <GoalList
      goals={goals}
      byId={byId}
      all={all}
      caps={data!.settings.caps}
      busy={busy}
      readOnly={scope === 'instance' && !deckEditable}
      onEdit={(g) => { setFormError(null); setEditing({ mode: 'edit', goal: g }) }}
      onOpen={(g) => setOpened(g.id)}
      onAction={(g, a) => void act(g, a)}
      onDismiss={(id) => void dismiss(id)}
      editForm={editForm}
    />
  )

  const band = (scope: Scope, goals: Goal[], emptyText: string) => {
    if (!data) return null
    const off = data.settings.horizons[scope].length === 0
    const form = newForm(scope, '')
    const any = goals.some((g) => OPEN.includes(g.status) || FINISHED.includes(g.status))
    return (
      <div className="space-y-3">
        {form}
        {any ? (
          <div className="fd-card overflow-hidden">{list(scope, goals)}</div>
        ) : !form && (
          <EmptyBand
            text={off ? 'The admin has switched goals off at this level.' : scope === 'instance' && !deckEditable ? 'No deck goals yet.' : emptyText}
            onAdd={addable(scope) ? startNew(scope) : undefined}
          />
        )}
      </div>
    )
  }

  const addButton = (scope: Scope, goals: Goal[]) => (addable(scope) && goals.some((g) => OPEN.includes(g.status) || FINISHED.includes(g.status))
    ? <AddButton onClick={startNew(scope)} /> : null)

  return (
    <div className="@container relative flex h-full min-h-0 flex-col bg-zinc-950 text-zinc-200">
      <header className={`shrink-0 border-b transition-colors ${scrolled ? 'border-[var(--fd-line)]' : 'border-transparent'}`}>
        <div className="mx-auto w-full max-w-4xl px-5 pb-4 pt-5 wide:px-8 wide:pb-5 wide:pt-7">
          <div className="flex items-center gap-2">
            <h2 className="min-w-0 flex-1 text-xl font-bold tracking-tight text-zinc-100">Course</h2>
              <button onClick={() => void load()} disabled={loading} title="Check again" aria-label="Check again"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-60">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            {onClose && (
              <button onClick={onClose} title="Close" aria-label="Close"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="mt-0.5 max-w-2xl text-sm leading-relaxed text-zinc-500">
            What your agents’ work is for. Each agent steers by the deck’s goals, yours and its own.
            A goal is never permission to do anything.
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto" onScroll={(e) => setScrolled(e.currentTarget.scrollTop > 4)}>
        <div className="mx-auto w-full max-w-4xl space-y-9 px-5 pb-24 wide:px-8">
          {error && (
            <div role="alert" className="flex items-start gap-3 rounded-2xl border border-red-500/25 bg-red-500/[0.07] px-4 py-3 text-sm text-red-700 dark:text-red-300">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span className="min-w-0 flex-1">{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss" className="rounded-full p-0.5 opacity-70 hover:opacity-100">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
          {!data && loading && (
            <div className="flex h-48 items-center justify-center"><Loader2 className="h-5 w-5 animate-spin text-zinc-500" /></div>
          )}

          {data && (
            <>
              {(proposals.length > 0 || attention.length > 0) && (
                <Section icon={BellDot} tone="amber" title="Needs you" count={proposals.length + attention.length}
                  hint="Nothing an agent suggests goes on your course until you accept it.">
                  <div className="space-y-3">
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
                            onCancel={cancelEdit}
                            submitLabel="Save and accept"
                          />
                        ) : null}
                      />
                    ))}
                    {attention.length > 0 && (
                      <div className="fd-card overflow-hidden">
                        <ul className="divide-y divide-[var(--fd-line)]">
                          {attention.map((g) => (
                            <AttentionRow
                              key={g.id}
                              goal={g}
                              owner={ownerOf(g)}
                              busy={busy === g.id}
                              canReview={canChange(g) && g.status === 'active' && !!(g.flags?.due || g.flags?.stale)}
                              onOpen={() => setOpened(g.id)}
                              onReview={() => void act(g, 'review')}
                            />
                          ))}
                        </ul>
                        {attention.some((g) => g.flags?.off_course) && (
                          <p className="border-t border-[var(--fd-line)] px-4 py-2.5 text-xs text-zinc-500">
                            To bring a goal back on course: give one of your agents a goal that serves it, plan a route to it, or pause it.
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </Section>
              )}

              <Section
                icon={Users}
                title="The deck"
                hint={deckEditable
                  ? 'Set by you as admin. Every agent on this deck works toward these, shared agents included.'
                  : 'Set by the deck’s admin. Every agent on this deck works toward these.'}
                action={addButton('instance', data.goals.deck)}
              >
                {band('instance', data.goals.deck, 'Add a goal every agent on this deck works toward')}
              </Section>

              <Section
                icon={UserRound}
                title="You"
                hint="Every agent working for you gets these. Members of your shared agents never do."
                action={addButton('user', data.goals.mine)}
              >
                {band('user', data.goals.mine, 'Add a goal your agents will steer by')}
              </Section>

              <Section icon={Bot} title="Your agents" hint="Each agent gets only its own goals.">
                {data.recommendations && (
                  <div className="mb-3 flex items-center gap-4 rounded-[var(--fd-radius-card)] border border-[var(--fd-line)] px-4 py-3">
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-zinc-200">Let agents add their own short-term goals without asking</div>
                      <div className="mt-0.5 text-xs leading-relaxed text-zinc-500">
                        You’re told on your Line each time, and can change or drop them here. Everything else they suggest still waits for you.
                      </div>
                    </div>
                    <Switch
                      on={data.prefs.auto_accept_agent_short}
                      disabled={prefSaving}
                      label="Let agents add their own short-term goals without asking"
                      onChange={(on) => void setAutoAccept(on)}
                    />
                  </div>
                )}
                {data.agents.length === 0 ? (
                  <EmptyBand text="You have no agents yet." />
                ) : (
                  <AgentList
                    agents={data.agents}
                    goalsOf={(ref) => data.goals.agents[ref] ?? []}
                    off={data.settings.horizons.agent.length === 0}
                    canAdd={addable('agent')}
                    onAdd={(ref) => startNew('agent', ref)()}
                    newForm={(ref) => newForm('agent', ref)}
                    list={(goals) => list('agent', goals)}
                  />
                )}
              </Section>

              <div className="space-y-3">
                <PreviewPanel agents={data.agents} version={rev} />
                {data.is_admin && !kiosk && <DeckCourseSettings onSaved={() => void load()} />}
              </div>
            </>
          )}
        </div>
      </div>

      {notice && (
        <div role="status" aria-live="polite" className="pointer-events-none absolute inset-x-0 bottom-5 z-20 flex justify-center px-4">
          <div className="fd-rise pointer-events-auto flex max-w-xl items-center gap-2.5 rounded-2xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] py-2 pl-2.5 pr-2 text-sm text-zinc-200 shadow-[0_18px_40px_-16px_rgba(0,0,0,0.55)]">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
              <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
            </span>
            <span className="line-clamp-2 min-w-0 flex-1">{notice}</span>
            <button onClick={() => setNotice(null)} aria-label="Dismiss"
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
      {opened && <GoalDrawer goalId={opened} onClose={() => setOpened(null)} onChanged={() => void load()} />}
    </div>
  )
}

// ── Pieces ──────────────────────────────────────────────────────────

function Section({ icon: Icon, tone = 'accent', title, count, hint, action, children }: {
  icon: typeof Users
  tone?: 'accent' | 'amber'
  title: string
  count?: number
  hint?: string
  action?: React.ReactNode
  children: React.ReactNode
}) {
  const tile = tone === 'amber'
    ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
    : 'bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]'
  return (
    <section>
      <div className="mb-3 flex items-center gap-3">
        <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-[0.7rem] ${tile}`}>
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-zinc-100">
            {title}
            {count != null && count > 0 && <span className="fd-count fd-count-hot">{count}</span>}
          </h3>
          {hint && <p className="text-xs text-zinc-500">{hint}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

function AddButton({ onClick, compact = false }: { onClick: () => void; compact?: boolean }) {
  return compact ? (
    <button onClick={onClick}
      className="inline-flex h-8 shrink-0 items-center gap-1 rounded-full px-3 text-xs font-medium text-zinc-400 transition-colors hover:bg-[var(--fd-hover)] hover:text-zinc-100">
      <Plus className="h-3.5 w-3.5" /> Add goal
    </button>
  ) : (
    <button onClick={onClick} className="fd-btn-quiet h-8 shrink-0 px-3 text-xs">
      <Plus className="h-3.5 w-3.5" /> Add goal
    </button>
  )
}

function EmptyBand({ text, onAdd }: { text: string; onAdd?: () => void }) {
  const cls = 'flex w-full items-center gap-3 rounded-[var(--fd-radius-card)] border border-dashed border-zinc-800 px-4 py-4 text-left text-sm text-zinc-500'
  if (!onAdd) return <div className={cls}>{text}</div>
  return (
    <button onClick={onAdd} className={`${cls} group transition-colors hover:border-violet-500/40 hover:bg-[var(--fd-hover)] hover:text-zinc-200`}>
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)] transition-transform group-hover:scale-105">
        <Plus className="h-4 w-4" />
      </span>
      {text}
    </button>
  )
}

function Switch({ on, disabled, label, onChange }: { on: boolean; disabled?: boolean; label: string; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className={`relative h-6 w-11 shrink-0 rounded-full transition-colors disabled:opacity-60 ${on ? 'bg-violet-600' : 'bg-zinc-700'}`}
    >
      <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${on ? 'left-[1.375rem]' : 'left-0.5'}`} />
    </button>
  )
}

const PILL = {
  red: 'bg-red-500/10 text-red-700 dark:text-red-300',
  amber: 'bg-amber-500/12 text-amber-800 dark:text-amber-300',
  green: 'bg-emerald-500/12 text-emerald-700 dark:text-emerald-300',
  zinc: 'bg-zinc-500/10 text-zinc-500',
} as const

function Pill({ tone, title, children }: { tone: keyof typeof PILL; title?: string; children: React.ReactNode }) {
  return <span title={title} className={`inline-flex items-center rounded-full px-2 py-px text-[11px] font-medium ${PILL[tone]}`}>{children}</span>
}

// A band's goals: grouped by horizon, each group with how many are active of
// the most allowed, and the finished ones folded away under them.
function GoalList({ goals, byId, all, caps, busy, readOnly, onEdit, onOpen, onAction, onDismiss, editForm }: {
  goals: Goal[]
  byId: Map<string, Goal>
  all: Goal[]
  caps: Record<Horizon, number>
  busy: string | null
  readOnly: boolean
  onEdit: (g: Goal) => void
  onOpen: (g: Goal) => void
  onAction: (g: Goal, a: GoalAction) => void
  onDismiss: (conflictId: string) => void
  editForm: (g: Goal) => React.ReactNode
}) {
  const [showDone, setShowDone] = useState(false)
  const open = goals.filter((g) => OPEN.includes(g.status))
  const finished = goals.filter((g) => FINISHED.includes(g.status))
  const groups = HORIZONS.map((h) => [h, open.filter((g) => g.horizon === h)] as const)
    .filter(([, here]) => here.length > 0)
  const row = (g: Goal) => {
    const form = editForm(g)
    if (form) return <li key={g.id} className="p-3">{form}</li>
    return (
      <GoalRow
        key={g.id}
        goal={g}
        parent={byId.get(g.parent_id)}
        // "Not linked to a bigger goal" only means something when there is one to serve.
        linkable={!g.parent_id && g.horizon !== 'strategic'
          && eligibleParents(all, g.scope, g.horizon, g.agent_ref, g.id).length > 0}
        busy={busy === g.id}
        busyId={busy}
        readOnly={readOnly || !g.can_edit}
        onEdit={() => onEdit(g)}
        onOpen={() => onOpen(g)}
        onAction={(a) => onAction(g, a)}
        onDismiss={onDismiss}
      />
    )
  }
  return (
    <div>
      {groups.map(([h, here], i) => {
        const active = here.filter((g) => g.status === 'active').length
        return (
          <div key={h} className={i > 0 ? 'border-t border-[var(--fd-line)]' : ''}>
            <div className="flex items-baseline gap-2 px-4 pb-0.5 pt-3">
              <span className="text-xs font-semibold text-zinc-400">{HORIZON_LABELS[h]}</span>
              <span className="text-[11px] tabular-nums text-zinc-500" title={`${active} active of at most ${caps[h]}`}>
                {active} of {caps[h]}
              </span>
            </div>
            <ul>{here.map(row)}</ul>
          </div>
        )
      })}
      {open.length === 0 && (
        <p className="px-4 py-3.5 text-sm text-zinc-500">No open goals.</p>
      )}
      {finished.length > 0 && (
        <div className="border-t border-[var(--fd-line)]">
          <button onClick={() => setShowDone(!showDone)} aria-expanded={showDone}
            className="flex w-full items-center gap-1.5 px-4 py-2.5 text-xs font-medium text-zinc-500 transition-colors hover:bg-[var(--fd-hover)] hover:text-zinc-300">
            <ChevronRight className={`h-3.5 w-3.5 transition-transform ${showDone ? 'rotate-90' : ''}`} />
            Finished <span className="tabular-nums">({finished.length})</span>
          </button>
          {showDone && <ul className="pb-1">{finished.map(row)}</ul>}
        </div>
      )}
    </div>
  )
}

function GoalRow({ goal: g, parent, linkable, busy, busyId, readOnly, onEdit, onOpen, onAction, onDismiss }: {
  goal: Goal
  parent?: Goal
  linkable: boolean
  busy: boolean
  busyId: string | null
  readOnly: boolean
  onEdit: () => void
  onOpen: () => void
  onAction: (a: GoalAction) => void
  onDismiss: (conflictId: string) => void
}) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const f = g.flags
  const live = OPEN.includes(g.status)
  const progress = live ? progressOf(g) : null
  const measured = live ? measureText(g.measure) : ''
  const clashes = g.conflicts ?? []
  const review = !readOnly && g.status === 'active' && !!(f?.due || f?.stale)

  const menu: MenuEntry[] = [
    { label: 'Open', icon: PanelRightOpen, onSelect: onOpen, title: 'The work that moved it, its history, Plan a route' },
    'divider',
    ...(readOnly ? [] : g.status === 'active' ? [
      { label: 'Edit', icon: Pencil, onSelect: onEdit },
      { label: 'Pause', icon: Pause, onSelect: () => onAction('pause') },
      { label: 'Mark done', icon: Check, onSelect: () => onAction('achieve') },
      'divider' as const,
      { label: 'Drop', icon: Trash2, danger: true, onSelect: () => onAction('drop') },
    ] : g.status === 'paused' ? [
      { label: 'Resume', icon: Play, onSelect: () => onAction('resume') },
      { label: 'Edit', icon: Pencil, onSelect: onEdit },
      { label: 'Mark done', icon: Check, onSelect: () => onAction('achieve') },
      'divider' as const,
      { label: 'Drop', icon: Trash2, danger: true, onSelect: () => onAction('drop') },
    ] : [
      { label: 'Reopen', icon: RotateCcw, onSelect: () => onAction('reopen') },
    ]),
  ]

  const meta: React.ReactNode[] = []
  if (parent) {
    meta.push(
      <span key="parent" className="inline-flex min-w-0 max-w-full items-center gap-1">
        <CornerDownRight className="h-3 w-3 shrink-0" /><span className="truncate">Serves {parent.title}</span>
      </span>,
    )
  } else if (linkable && live) {
    meta.push(<span key="parent">Not linked to a bigger goal</span>)
  }
  if (measured) meta.push(<span key="m" className="text-zinc-400">{measured}</span>)
  else if (progress != null) meta.push(<span key="p" className="text-zinc-400" title={g.progress_note || undefined}>About {Math.round(progress * 100)}%</span>)
  if (g.due_at) meta.push(<span key="due">Due {day(g.due_at)}</span>)
  if (g.status === 'active' && g.review_at) meta.push(<span key="rev">Review {day(g.review_at)}</span>)
  if (g.status === 'active') meta.push(<span key="moved">{g.last_moved ? `Moved ${day(g.last_moved)}` : 'No work linked yet'}</span>)

  return (
    <li className="group relative flex gap-3 px-4 py-3 transition-colors hover:bg-[var(--fd-hover)]">
      <div className="pt-px"><ProgressRing value={progress} status={g.status} /></div>
      <div className="min-w-0 flex-1">
        <button onClick={onOpen} title="Open: the work that moved it, its history, Plan a route"
          className="block w-full rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40">
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className={`text-[15px] font-medium leading-snug ${g.status === 'dropped' ? 'text-zinc-500 line-through' : 'text-zinc-100'}`}>
              {g.title}
            </span>
            {g.status === 'paused' && <Pill tone="zinc">Paused</Pill>}
            {g.status === 'achieved' && <Pill tone="green">Done</Pill>}
            {g.status === 'dropped' && <Pill tone="zinc">Dropped</Pill>}
            {f?.stale ? (
              <Pill tone="red" title="Out of your agents’ prompts until someone says it’s still on course">Out of prompts</Pill>
            ) : f?.due && g.status === 'active' ? (
              <Pill tone="amber" title={`It leaves your agents’ prompts ${g.grace_days ?? 3} days after its review date`}>Review due</Pill>
            ) : null}
            {f?.off_course && <Pill tone="amber" title={g.off_course_reason}>Off course</Pill>}
            {f?.horizon_retired && <Pill tone="amber" title={`This level no longer uses ${HORIZON_LABELS[g.horizon].toLowerCase()} goals`}>Horizon retired</Pill>}
            {f?.parent_gone && <Pill tone="zinc">The goal it served is finished</Pill>}
          </span>
          {g.success && (
            <span className="mt-0.5 line-clamp-2 text-[13px] leading-relaxed text-zinc-400">
              <span className="text-zinc-500">Done when: </span>{g.success}
            </span>
          )}
          {meta.length > 0 && (
            <span className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-zinc-500">{meta}</span>
          )}
        </button>
        {review && <ReviewButton onClick={() => onAction('review')} busy={busy} className="mt-2 wide:hidden" />}
        {clashes.length > 0 && (
          <div className="mt-2 space-y-1">
            {clashes.map((cf) => (
              <div key={cf.id} className="flex flex-wrap items-center gap-2 rounded-xl bg-red-500/[0.07] py-1.5 pl-3 pr-1.5 text-xs text-red-700 dark:text-red-300">
                <span className="min-w-0 flex-1">
                  May pull against “{cf.other_title}”{cf.reason ? ` — ${cf.reason}` : ''}
                </span>
                <button onClick={() => onDismiss(cf.id)} disabled={busyId === cf.id}
                  title="Flight Deck won't flag this pair again"
                  className="shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                  {busyId === cf.id ? 'Saving…' : 'Not a conflict'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="flex shrink-0 items-start gap-1">
        {review && <ReviewButton onClick={() => onAction('review')} busy={busy} className="hidden wide:block" />}
        {!readOnly && <button
          onClick={(e) => setAnchor(anchor ? null : e.currentTarget)}
          disabled={busy}
          aria-label={`${g.title}: actions`}
          aria-haspopup="menu"
          aria-expanded={!!anchor}
          className={`flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200 focus:opacity-100 group-hover:opacity-100 pointer-coarse:opacity-100 ${
            anchor || busy ? 'bg-zinc-800 text-zinc-200 opacity-100' : 'opacity-0'}`}
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <MoreHorizontal className="h-4 w-4" />}
        </button>}
        {anchor && <QueueMenu anchor={anchor} label={`${g.title}: actions`} items={menu} onClose={() => setAnchor(null)} />}
      </div>
    </li>
  )
}

function ReviewButton({ onClick, busy, className = '' }: { onClick: () => void; busy: boolean; className?: string }) {
  return (
    <button onClick={onClick} disabled={busy}
      title="Keep it in your agents’ prompts until its next review"
      className={`h-8 rounded-full bg-amber-500/12 px-3 text-xs font-medium text-amber-800 transition-colors hover:bg-amber-500/20 disabled:opacity-50 dark:text-amber-300 ${className}`}>
      Still on course
    </button>
  )
}

// One line in "Needs you": what the goal is, why it's here, whose it is, and
// the one thing to do about it (or the way into it).
function AttentionRow({ goal: g, owner, busy, canReview, onOpen, onReview }: {
  goal: Goal
  owner: string
  busy: boolean
  canReview: boolean
  onOpen: () => void
  onReview: () => void
}) {
  const f = g.flags
  const stale = !!f?.stale && g.status === 'active'
  const due = !!f?.due && g.status === 'active'
  const reasons: string[] = []
  if (stale) reasons.push('Out of your agents’ prompts until you say it’s still on course')
  if (f?.off_course) reasons.push(g.off_course_reason || 'Nothing is moving it')
  if (due && !stale) reasons.push(g.review_at ? `Review was due ${day(g.review_at)}` : 'Review due')
  const red = stale
  const Icon = f?.off_course && !stale ? AlertTriangle : Clock
  return (
    <li className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-[var(--fd-hover)]">
      <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${red ? PILL.red : PILL.amber}`}>
        <Icon className="h-4 w-4" />
      </span>
      <button onClick={onOpen} className="min-w-0 flex-1 text-left focus-visible:outline-none">
        <span className="line-clamp-2 text-sm font-medium text-zinc-100 wide:line-clamp-1">{g.title}</span>
        <span className="mt-0.5 flex flex-wrap gap-x-2.5 text-xs text-zinc-500">
          <span className={red ? 'text-red-700 dark:text-red-300' : 'text-amber-800 dark:text-amber-300'}>{reasons.join(' · ')}</span>
          <span>{owner} · {HORIZON_LABELS[g.horizon]}</span>
        </span>
      </button>
      {canReview ? (
        <button onClick={onReview} disabled={busy}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full bg-amber-500/12 px-3 text-xs font-medium text-amber-800 transition-colors hover:bg-amber-500/20 disabled:opacity-50 dark:text-amber-300">
          {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          Still on course
        </button>
      ) : (
        <button onClick={onOpen} aria-label={`Open ${g.title}`}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
          <ChevronRight className="h-4 w-4" />
        </button>
      )}
    </li>
  )
}

// The agents, in one card: those with goals first, their goals under them;
// those without on a single line each (past a few, folded away).
function AgentList({ agents, goalsOf, off, canAdd, onAdd, newForm, list }: {
  agents: CourseResponse['agents']
  goalsOf: (ref: string) => Goal[]
  off: boolean
  canAdd: boolean
  onAdd: (ref: string) => void
  newForm: (ref: string) => React.ReactNode
  list: (goals: Goal[]) => React.ReactNode
}) {
  const [showAll, setShowAll] = useState(false)
  const has = (ref: string) => goalsOf(ref).some((g) => OPEN.includes(g.status) || FINISHED.includes(g.status))
  const busyAgents = agents.filter((a) => has(a.ref))
  const quiet = agents.filter((a) => !has(a.ref))
  // An agent being given its first goal stays in view.
  const keep = (ref: string) => !!newForm(ref)
  const quietShown = showAll ? quiet : quiet.filter((a, i) => i < QUIET_AGENTS_SHOWN || keep(a.ref))
  const hidden = quiet.length - quietShown.length

  const agentRow = (a: CourseResponse['agents'][number]) => {
    const goals = goalsOf(a.ref)
    const open = goals.filter((g) => OPEN.includes(g.status))
    const offCourse = open.filter((g) => g.flags?.off_course).length
    const form = newForm(a.ref)
    const summary = open.length === 0
      ? (goals.length > 0 ? 'No open goals' : off ? 'Goals are off for agents' : 'No goals yet')
      : `${open.length} goal${open.length === 1 ? '' : 's'}`
    return (
      <li key={a.ref}>
        <div className="flex items-center gap-3 px-4 py-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-[var(--fd-accent-soft)] text-xs font-semibold text-[var(--fd-accent-text)]">
            {agentInitials(a.name)}
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-zinc-100">{a.name}</div>
            <div className="text-xs text-zinc-500">
              {summary}
              {offCourse > 0 && <span className="text-amber-800 dark:text-amber-300"> · {offCourse} off course</span>}
            </div>
          </div>
          {canAdd && <AddButton compact onClick={() => onAdd(a.ref)} />}
        </div>
        {form && <div className="px-4 pb-4 wide:pl-16">{form}</div>}
        {goals.length > 0 && <div className="pb-1 wide:pl-12">{list(goals)}</div>}
      </li>
    )
  }

  return (
    <div className="fd-card overflow-hidden">
      <ul className="divide-y divide-[var(--fd-line)]">
        {busyAgents.map(agentRow)}
        {quietShown.map(agentRow)}
      </ul>
      {hidden > 0 && (
        <button onClick={() => setShowAll(true)}
          className="flex w-full items-center gap-1.5 border-t border-[var(--fd-line)] px-4 py-2.5 text-xs font-medium text-zinc-500 transition-colors hover:bg-[var(--fd-hover)] hover:text-zinc-300">
          <ChevronDown className="h-3.5 w-3.5" />
          {hidden} more agent{hidden === 1 ? '' : 's'} without goals
        </button>
      )}
    </div>
  )
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
    <div className="fd-card overflow-hidden">
      <button onClick={() => setOpen(!open)} aria-expanded={open}
        className="flex w-full items-center gap-3 px-4 py-3.5 text-left transition-colors hover:bg-[var(--fd-hover)]">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[0.7rem] bg-[var(--fd-accent-soft)] text-[var(--fd-accent-text)]">
          <Eye className="h-4 w-4" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-zinc-100">What your agents receive</span>
          <span className="block text-xs text-zinc-500">The goals as they’re written into an agent’s prompt</span>
        </span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="space-y-4 border-t border-[var(--fd-line)] px-4 py-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <select value={ref} onChange={(e) => setRef(e.target.value)} aria-label="Agent"
              className="h-8 max-w-full rounded-full border border-[var(--fd-line)] bg-zinc-950 px-3 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
              {agents.map((a) => <option key={a.ref} value={a.ref}>{a.name}</option>)}
              <option value="">A new agent of yours</option>
            </select>
            <div role="group" aria-label="Length" className="flex rounded-full border border-[var(--fd-line)] p-0.5">
              {(['full', 'compact'] as const).map((m) => (
                <button key={m} onClick={() => setCompact(m === 'compact')} aria-pressed={mode === m}
                  className={`h-7 rounded-full px-3 font-medium transition-colors ${mode === m ? 'bg-[var(--fd-selected)] text-zinc-100' : 'text-zinc-500 hover:text-zinc-200'}`}>
                  {m === 'full' ? 'Full' : 'Short (small models)'}
                </button>
              ))}
            </div>
          </div>
          {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
          {preview ? (
            <div className="grid gap-4 @3xl:grid-cols-2">
              <PreviewBlock label="When you talk to it" text={preview.owner[mode]} />
              <PreviewBlock label="When a member of a shared agent talks to it" text={preview.member[mode]} />
            </div>
          ) : !error && (
            <div className="flex h-24 items-center justify-center"><Loader2 className="h-4 w-4 animate-spin text-zinc-500" /></div>
          )}
        </div>
      )}
    </div>
  )
}

function PreviewBlock({ label, text }: { label: string; text: string }) {
  return (
    <div className="min-w-0">
      <div className="mb-1.5 text-xs font-medium text-zinc-400">{label}</div>
      <div className="max-h-80 overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-[var(--fd-line)] bg-zinc-900/60 px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-zinc-300">
        {text || <span className="font-sans text-zinc-500">Nothing: no live goals reach it.</span>}
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
  return (
    <div className="fd-card p-4">
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-zinc-500">
        <Sparkles className="h-3.5 w-3.5 text-[var(--fd-accent-text)]" />
        <span><span className="font-medium text-zinc-300">{g.proposer || 'An agent'}</span> suggests {old ? 'a change' : 'a new goal'}</span>
        <span aria-hidden className="text-zinc-700">·</span>
        <span>{forWhom}</span>
        <span aria-hidden className="text-zinc-700">·</span>
        <span>{HORIZON_LABELS[g.horizon]}</span>
      </div>
      {form ? <div className="mt-3">{form}</div> : (
        <>
          <div className="mt-2 text-[15px] font-semibold leading-snug text-zinc-100">{g.title}</div>
          {g.success && <p className="mt-0.5 text-[13px] leading-relaxed text-zinc-400"><span className="text-zinc-500">Done when: </span>{g.success}</p>}
          {changes.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs">
              {changes.map(([label, was, becomes]) => (
                <li key={label} className="text-zinc-400">
                  <span className="text-zinc-500">{label}:</span>{' '}
                  <span className="line-through decoration-zinc-600">{was}</span>{' '}
                  <span className="text-zinc-600">→</span>{' '}
                  <span className="text-zinc-200">{becomes}</span>
                </li>
              ))}
            </ul>
          )}
          {g.rationale && (
            <p className="mt-2.5 border-l-2 border-violet-500/40 pl-3 text-[13px] italic leading-relaxed text-zinc-400">{g.rationale}</p>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <button onClick={onAccept} disabled={busy} className="fd-btn-primary h-8 px-3.5 text-xs disabled:pointer-events-none disabled:opacity-50">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Accept
            </button>
            <button onClick={onEdit} disabled={busy} className="fd-btn-quiet h-8 px-3.5 text-xs disabled:pointer-events-none disabled:opacity-50">
              <Pencil className="h-3.5 w-3.5" /> Edit and accept
            </button>
            <button onClick={onDecline} disabled={busy}
              className="inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-zinc-500 transition-colors hover:bg-[var(--fd-hover)] hover:text-zinc-200 disabled:opacity-50">
              <X className="h-3.5 w-3.5" /> Decline
            </button>
          </div>
        </>
      )}
    </div>
  )
}
